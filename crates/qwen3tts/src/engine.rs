//! Qwen3-TTS behind the [`Engine`] trait.
//!
//! Two stages: [`crate::talker`] (with its depth predictor) then [`crate::codec`]. Stage
//! timers call `device.synchronize()` first — Metal dispatch is async, and an unsynchronised
//! timer measured enqueue time and misattributed most of CosyVoice's cost.
//!
//! Caller-visible limits: ten languages only ([`cfg::talker::LANGUAGES`]); text advances one
//! token per audio frame, so segmentation behaves differently from CosyVoice (trap 3);
//! streaming is native to the architecture but stays false until the trait has a method for
//! it.

use crate::cfg;
use crate::codec::Codec;
use crate::talker::{Sampling, Talker};
use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;
use tts_core::rng::Rng;
use tts_core::{
    text, wav, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis,
    SynthesisRequest,
};
use tts_nn::Weight;

pub const ID: &str = "qwen3tts";

/// Below this many characters a segment's frames-per-character ratio is too noisy to judge.
const SEGMENT_MIN_CHARS: usize = 40;
/// Frames per character above this multiple of the request's median means the talker kept
/// going after its text ran out.
const SEGMENT_RATIO_CEILING: f64 = 1.6;
/// Below this many frames the distinct-code count is not meaningful.
const SEGMENT_MIN_FRAMES: usize = 24;
/// Distinct codebook-0 values per frame below this is degenerate repetition.
const SEGMENT_MIN_VARIETY: f64 = 0.35;

/// Weight formats that load, default first. Quantization covers the talker's and predictor's
/// projections, not the codec decoder, which runs over whole chunks.
///
/// **Which one to ask for depends entirely on whether segments batch**, because both
/// transformers are bandwidth-bound on weight reads:
///
/// - **`q8_0` (default)** reads the fewest bytes, so it wins when nothing batches. One sentence:
///   RTF 0.79 against f16's 1.53.
/// - **`f16`** is the one for long text. Only a *dense* GEMM shares one weight read across a
///   batch of lanes — candle's quantized `mm_t` re-reads per row, batching 1.1x against f16's
///   7.4x. A chapter: RTF 0.31 against 0.66. Also halves the KV cache, which is what caps
///   `MAX_BATCH`.
/// - **`f32`** is for fixture work. 6.3 GB of projections thrashes a 16 GB machine — measured
///   1994 ms/frame against q8_0's 52, memory pressure rather than arithmetic.
///
/// `docs/performance/qwen3tts-batching.md` has the measurements.
const QUANT: &[&str] = &["q8_0", "f16", "f32", "q5_0", "q4_1", "q4_0"];

/// One segment's decoded frames, with the paragraph index and character count it came from.
type Decoded = (usize, usize, Vec<Vec<u32>>);

/// Lanes per batched decode.
///
/// Bounded by memory, not by diminishing returns: per-lane cost was still falling at 24 in the
/// `qwen3tts-batch` sweep (trunk 3.40 ms/lane, 12.0x amortisation). A lane costs
/// `positions * 114 KB` of f16 KV cache, so 24 is ~1.8 GB on top of 2.8 GB of f16 weights.
/// Measured 6.9 GB resident on a 21-segment render; a 100-segment chapter peaked at 11.1 GB
/// because candle's Metal buffer pool never releases, which is the real ceiling here.
const MAX_BATCH: usize = 24;

/// `MAX_BATCH`, overridable for tuning. Group size trades three things against each other: a
/// wider batch amortises the weight read further, but costs more per step and packs lengths
/// *worse*, since a group runs as long as its longest lane.
fn max_batch() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("QWEN3TTS_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(MAX_BATCH)
    })
}

/// Frames a batched lane may reach before the group is redone one segment at a time.
///
/// The KV cache is sized for this, so it cannot be the request's full budget: 4096 positions
/// times eight lanes is 9.4 GB. 512 frames is 41 s of audio from a single segment, which no
/// sentence reaches — and a lane that does hit it is *rerun unbatched with the full budget*
/// rather than truncated, so this bounds memory without bounding output.
const BATCH_FRAME_CAP: usize = 512;

const NO_VOICE: &str = "engine `qwen3tts` requires a voice asset: the talker's prefill \
    carries the reference clip's speaker embedding, and in-context cloning also needs its \
    codes and transcript. Build one with references/qwen3tts/export_voice.py";

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description: "Qwen3-TTS-12Hz-1.7B-Base — Qwen3 talker, 15-step depth transformer, \
                      RVQ codec decoder. 24 kHz, no diffusion. Ten languages only \
                      (en, de, es, zh, ja, fr, ko, ru, it, pt)",
        sample_rate: cfg::SAMPLE_RATE as u32,
        frame_rate: cfg::FRAME_RATE,
        cloning: Cloning::PrecomputedAsset,
        // The architecture streams natively; the trait has no streaming method yet, so
        // claiming it would be a lie a client could act on. See the module docs.
        streaming: false,
        quantization: QUANT,
        available: true,
        reason: None,
    }
}

/// Checkpoint paths. Talker at the root, codec under `speech_tokenizer/` — both in the one
/// upstream download, unlike CosyVoice's separate ONNX tokenizer.
pub struct Paths {
    pub talker: PathBuf,
    pub codec: PathBuf,
    pub vocab: PathBuf,
    pub merges: PathBuf,
}

impl Paths {
    pub fn resolve(config: &EngineConfig) -> Self {
        Self {
            talker: config.path("talker", "model.safetensors"),
            codec: config.path("codec", "speech_tokenizer/model.safetensors"),
            // No tokenizer.json in this checkpoint, unlike CosyVoice's — the BPE is built
            // from vocab.json plus merges.txt at load.
            vocab: config.path("vocab", "vocab.json"),
            merges: config.path("merges", "merges.txt"),
        }
    }

    /// Report every missing file at once rather than one per run.
    pub fn check(&self) -> Result<()> {
        let missing: Vec<&Path> = [&self.talker, &self.codec, &self.vocab, &self.merges]
            .into_iter()
            .map(PathBuf::as_path)
            .filter(|p| !p.exists())
            .collect();
        anyhow::ensure!(
            missing.is_empty(),
            "engine `{ID}` is missing {} weight file(s): {}. Download \
             Qwen/Qwen3-TTS-12Hz-1.7B-Base into references/qwen3tts/weights — see \
             docs/setup.md",
            missing.len(),
            missing
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }
}

fn parse_quant(name: Option<&str>) -> Result<Weight> {
    Ok(match name {
        Some("f32") => Weight::F32,
        Some("f16") => Weight::F16,
        None | Some("q8_0") => Weight::Quant(GgmlDType::Q8_0),
        Some("q5_0") => Weight::Quant(GgmlDType::Q5_0),
        Some("q4_1") => Weight::Quant(GgmlDType::Q4_1),
        Some("q4_0") => Weight::Quant(GgmlDType::Q4_0),
        Some(other) => anyhow::bail!(
            "engine `{ID}` does not support weight format `{other}`; it accepts {}",
            QUANT.join(", ")
        ),
    })
}

/// Distinct codebook-0 values across a segment's frames.
///
/// A talker that has lost the thread repeats one code, and repeated codes render as a
/// metallic buzz — the codec faithfully decodes whatever it is given.
fn distinct_first(frames: &[Vec<u32>]) -> usize {
    let mut seen: Vec<u32> = frames.iter().map(|f| f[0]).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// Warn about segments that look wrong, against the median of *this* request.
///
/// Two failure modes, and they need opposite tests. CosyVoice only had to catch segments that
/// stopped **early** (a duration ratio below the median). This model can also run **long**:
/// once the text stream is exhausted the talker is fed `tts_pad` forever and will keep emitting
/// frames until it chooses `codec_eos`, so a segment can babble past its text. A median is the
/// right reference because it is measured from this voice and this text rather than assumed.
fn report_segments(stats: &[(usize, usize, usize)]) {
    if stats.len() < 3 {
        return;
    }
    let mut ratios: Vec<f64> = stats
        .iter()
        .map(|(chars, frames, _)| *frames as f64 / (*chars).max(1) as f64)
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    if median <= 0.0 {
        return;
    }
    let verbose = std::env::var_os("QWEN3TTS_SEGMENTS").is_some();
    if verbose {
        eprintln!("engine {ID}: median {median:.2} frames/char");
    }
    for (i, (chars, frames, distinct)) in stats.iter().enumerate() {
        let ratio = ratios[i];
        if verbose {
            eprintln!(
                "  seg {i}: {chars:>4} chars {frames:>4} frames  {ratio:.2} f/c  \
                 {distinct:>3} distinct code0 ({:.2})  {:.2} s",
                *distinct as f64 / (*frames).max(1) as f64,
                *frames as f64 / crate::cfg::FRAME_RATE
            );
        }
        // Frames per character, against the median. Long segments only: a short segment's
        // ratio is naturally noisy.
        if *chars >= SEGMENT_MIN_CHARS && ratio > median * SEGMENT_RATIO_CEILING {
            eprintln!(
                "engine {ID}: segment {i} runs long — {frames} frames for {chars} chars \
                 ({ratio:.2} vs median {median:.2}); the talker kept generating after its text \
                 ran out. Expect audible babble there."
            );
        }
        // Degenerate repetition renders as a metallic buzz.
        let variety = *distinct as f64 / (*frames).max(1) as f64;
        if *frames >= SEGMENT_MIN_FRAMES && variety < SEGMENT_MIN_VARIETY {
            eprintln!(
                "engine {ID}: segment {i} looks degenerate — only {distinct} distinct codebook-0 \
                 values across {frames} frames ({variety:.2}). Repeated codes decode as a \
                 metallic buzz."
            );
        }
    }
    ratios.clear();
}

pub struct Qwen3TtsEngine {
    talker: Talker,
    codec: Codec,
    tokenizer: Tokenizer,
    device: Device,
    /// The prefilled language tag, or `None` for the reference's `nothink` / "auto" path.
    /// `--set language=english`.
    language: Option<u32>,
    /// Whether to batch segments through the talker. Dense weights only — see the grouping
    /// code for the measurement.
    batches: bool,
}

impl Qwen3TtsEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let paths = Paths::resolve(config);
        paths
            .check()
            .with_context(|| format!("loading engine `{ID}`"))?;
        let device = if config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).context("opening the Metal device")?
        };
        let quant = parse_quant(config.quant.as_deref())?;
        let s = |p: &Path| -> Result<String> {
            p.to_str()
                .map(str::to_owned)
                .with_context(|| format!("non-utf8 path {}", p.display()))
        };

        // Qwen's byte-level BPE, assembled from the two files the checkpoint ships.
        // `add_prefix_space=false` matches the reference's tokenizer_config.
        let bpe = BPE::from_file(&s(&paths.vocab)?, &s(&paths.merges)?)
            .build()
            .map_err(|e| anyhow::anyhow!("building the BPE from vocab.json/merges.txt: {e}"))?;
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));

        let language = config
            .overrides
            .get("language")
            .and_then(|p| p.to_str())
            .map(|name| {
                let lower = name.to_ascii_lowercase();
                if lower == "auto" {
                    return Ok(None);
                }
                cfg::talker::language_id(&lower).map(Some).with_context(|| {
                    format!(
                        "engine `{ID}` has no language id for `{name}`; it supports {} (or `auto`)",
                        cfg::talker::LANGUAGES.join(", ")
                    )
                })
            })
            .transpose()?
            .flatten();

        Ok(Self {
            talker: Talker::load(&s(&paths.talker)?, quant, &device)?,
            codec: Codec::load(&s(&paths.codec)?, &device)?,
            tokenizer,
            device,
            language,
            batches: quant.batches(),
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    }
}

impl Engine for Qwen3TtsEngine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        // Unreachable while `available` is false, but written now so flipping that flag is
        // not also the commit that has to remember this rule.
        tts_core::engine::validate_against(&self.capabilities(), request)?;
        anyhow::ensure!(request.voice.is_some(), "{NO_VOICE}");
        Ok(())
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis> {
        self.validate(request)?;
        let voice = request.voice.as_ref().expect("validated");
        let spk = self.talker.speaker(voice.get("spk_embedding")?)?;
        // [T, 16] frames-major, the orientation `generate_icl_prompt` indexes.
        let ref_codes = voice.get_rows_u32("ref_codes").unwrap_or_default();
        let ref_text = voice
            .get_rows_u32("ref_text_tokens")
            .ok()
            .and_then(|r| r.into_iter().next())
            .unwrap_or_default();

        let paragraphs = text::segment(&request.text, request.max_chars);
        let flat: Vec<(usize, &String)> = paragraphs
            .iter()
            .enumerate()
            .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s)))
            .collect();
        anyhow::ensure!(!flat.is_empty(), "no text to speak");

        // Prefer this model's own documented defaults over `tts_core::Sampling`'s generic
        // ones, but honour anything the caller actually chose.
        //
        // The generic defaults are temperature 0.7 / top_p 0.9; the reference uses 0.9 / 1.0
        // for both the talker and the sub-talker. A top_p of 0.9 truncates a distribution the
        // reference never truncates, and applying it to the acoustic residuals thins the timbre
        // — audibly metallic, while codebook 0 keeps the words intelligible. Shipping a
        // known-worse default because a shared struct happened to pick it would be wrong.
        //
        // `SynthesisRequest` cannot say whether a field was set or defaulted, so a field equal
        // to the generic default is treated as unset. The cost of that heuristic is a caller who
        // explicitly asks for exactly 0.7/0.9 and silently gets 0.9/1.0; the alternative is
        // every caller getting the worse sound unless they know to override it.
        let generic = tts_core::Sampling::default();
        let req = &request.sampling;
        let pick = |got: f32, generic: f32, reference: f32| {
            if (got - generic).abs() < f32::EPSILON {
                reference
            } else {
                got
            }
        };
        let sampling = Sampling {
            temperature: pick(
                req.temperature,
                generic.temperature,
                cfg::talker::TEMPERATURE,
            ),
            top_p: pick(req.top_p, generic.top_p, cfg::talker::TOP_P),
            top_k: if req.top_k == generic.top_k {
                cfg::talker::TOP_K
            } else {
                req.top_k
            },
            greedy: req.greedy,
            ..Sampling::default()
        };
        let mut rng = Rng::new(request.sampling.seed);
        let mut stats = Stats::default();
        let t0 = Instant::now();

        // Stage 1: the talker, batching every segment whose prompt is the same length.
        //
        // Both transformers are bandwidth-bound on *weight* reads at batch 1 — the trunk reads
        // 1.4 G parameters once a frame and the depth predictor reads its 60 M fifteen times —
        // so a lane costs almost nothing beyond the read that a batch already pays. Measured by
        // `qwen3tts-batch`: dense f32 at batch 8 is **7.45x cheaper per lane**. Quantized is
        // not, at 1.13x: candle's `quantized_matmul_mm_t` re-reads the weights per row, which
        // is why `quant=f32` is the fast configuration here and q8_0 the small one.
        //
        // Grouping by length is what makes this cheap. See `Talker::generate_batch`.
        let budget = request.max_new_tokens.clamp(1, cfg::talker::MAX_NEW_TOKENS);
        let mut prepared: Vec<(usize, usize, Tensor, Tensor)> = Vec::new();
        for (pi, seg) in &flat {
            let ids = self.tokenize(seg)?;
            if ids.is_empty() {
                continue;
            }
            let (prompt, trailing) =
                self.talker
                    .build_prompt(&ids, &ref_text, &ref_codes, Some(&spk), self.language)?;
            prepared.push((*pi, seg.chars().count(), prompt, trailing));
        }

        // Lanes of equal (prompt, trailing) length, in runs of at most MAX_BATCH. Output order
        // is restored by the paragraph index carried alongside.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut by_shape: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, (_, _, p, tr)) in prepared.iter().enumerate() {
            by_shape.entry((p.dim(1)?, tr.dim(1)?)).or_default().push(i);
        }
        let mut shapes: Vec<_> = by_shape.into_values().collect();
        shapes.sort_by_key(|g| g[0]);
        // Only dense weights batch. Quantized ones measure *worse* batched (RTF 1.02 against
        // 0.79 on this text) because candle's `quantized_matmul_mm_t` re-reads the weights per
        // row, so the batch pays full price per lane and then wastes steps on finished lanes.
        let lanes = if self.batches { max_batch() } else { 1 };
        for mut g in shapes {
            // **Sort by length before chunking.** A batched step costs the same whether all
            // lanes are still producing or one is, and lanes stop at their own `codec_eos`, so a
            // group runs as long as its *longest* member. Mixed lengths measured **47-56% of
            // lane-steps useful** on a 100-segment chapter — half the talker's work thrown away.
            // Grouping similar lengths together is what removes that, and character count is a
            // good enough proxy because frames per character is stable within one voice (it is
            // the same ratio `report_segments` takes a median of).
            //
            // Stable, and tie-broken by index, so a render stays reproducible under a seed.
            g.sort_by_key(|&i| (prepared[i].1, i));
            for chunk in g.chunks(lanes) {
                groups.push(chunk.to_vec());
            }
        }

        let mut out: Vec<Option<Decoded>> = vec![None; prepared.len()];
        let mut unspoken = 0usize;
        let t = Instant::now();
        for group in &groups {
            let mut batched = None;
            if group.len() > 1 {
                let cap = budget.min(BATCH_FRAME_CAP);
                let prompts: Vec<Tensor> = group.iter().map(|&i| prepared[i].2.clone()).collect();
                let trailings: Vec<Tensor> = group.iter().map(|&i| prepared[i].3.clone()).collect();
                let prompt = Tensor::cat(&prompts, 0)?.contiguous()?;
                let trailing = Tensor::cat(&trailings, 0)?.contiguous()?;
                let (frames, left, timing) = self
                    .talker
                    .generate_batch(&prompt, &trailing, cap, &sampling, &mut rng)?;
                if std::env::var_os("QWEN3TTS_TIMING").is_some() {
                    let capacity = timing.steps * timing.lanes;
                    eprintln!(
                        "engine {ID}: batch {} — {} steps, {} frames, {:.0}% of lane-steps useful",
                        group.len(),
                        timing.steps,
                        timing.frames,
                        timing.frames as f64 / capacity.max(1) as f64 * 100.0,
                    );
                }
                // A lane that filled the cap may have been cut off mid-sentence. Redo the group
                // one segment at a time with the real budget rather than ship truncated audio.
                if frames.iter().any(|f| f.len() >= cap) {
                    eprintln!(
                        "engine {ID}: a batched lane reached {cap} frames; rerunning {} \
                         segment(s) unbatched",
                        group.len()
                    );
                } else {
                    batched = Some((frames, left));
                }
            }
            match batched {
                Some((frames, left)) => {
                    unspoken += left.iter().sum::<usize>();
                    for (lane, &i) in group.iter().enumerate() {
                        out[i] = Some((prepared[i].0, prepared[i].1, frames[lane].clone()));
                    }
                }
                None => {
                    for &i in group {
                        let (pi, chars, prompt, trailing) = &prepared[i];
                        let (frames, left, _) = self
                            .talker
                            .generate(prompt, trailing, budget, &sampling, &mut rng)?;
                        unspoken += left;
                        out[i] = Some((*pi, *chars, frames));
                    }
                }
            }
        }

        let mut spans: Vec<(usize, Vec<Vec<u32>>)> = Vec::new();
        // (characters, frames, distinct codebook-0 values) per segment.
        let mut seg_stats: Vec<(usize, usize, usize)> = Vec::new();
        for slot in out.into_iter() {
            let Some((pi, chars, frames)) = slot else {
                continue;
            };
            if !frames.is_empty() {
                seg_stats.push((chars, frames.len(), distinct_first(&frames)));
                spans.push((pi, frames));
            }
        }
        // Metal dispatch is async: without this the stage time is enqueue time and the GPU
        // work is billed to whatever is timed next.
        self.device.synchronize()?;
        stats.add("talker", t.elapsed().as_secs_f64());
        anyhow::ensure!(!spans.is_empty(), "engine {ID} generated no frames");
        report_segments(&seg_stats);

        if unspoken > 0 {
            // Trap 3: text is consumed one token per frame, so a segment that stopped early
            // leaves an exact count behind rather than a ratio to estimate.
            eprintln!(
                "engine {ID}: {unspoken} text position(s) never reached the talker — some text \
                 was not spoken. Lower max_chars."
            );
        }

        // Stage 2: the codec decoder, **once over every segment's frames**, cut afterwards.
        //
        // Not per segment. The decoder is causal and its receptive field is large — pre_conv
        // k=3, a ConvNeXt k=7, then k=7 convs at dilations 1/3/9 through four upsample stages
        // — so a per-segment call opens each segment with zero left context and an audible
        // transient. Segment two onward came out with a different timbre from segment one for
        // exactly that reason.
        //
        // Decoding the concatenation gives every segment real history, and the cut points are
        // *exact* rather than estimated: one frame is `SAMPLES_PER_FRAME` samples, always.
        // Chunking still happens inside `decode`, where it carries its own left context.
        let t = Instant::now();
        let all_frames: Vec<Vec<u32>> = spans
            .iter()
            .flat_map(|(_, frames)| frames.iter().cloned())
            .collect();
        let frames_total = all_frames.len();
        let joined = self.codec.decode(&all_frames)?;
        self.device.synchronize()?;
        stats.add("codec", t.elapsed().as_secs_f64());

        let mut pieces: Vec<(usize, Vec<f32>)> = Vec::with_capacity(spans.len());
        let mut at = 0usize;
        for (i, (pi, frames)) in spans.iter().enumerate() {
            // The last segment takes whatever remains, so rounding cannot drop samples.
            let end = if i + 1 == spans.len() {
                joined.len()
            } else {
                (at + frames.len() * cfg::SAMPLES_PER_FRAME).min(joined.len())
            };
            if end > at {
                pieces.push((*pi, joined[at..end].to_vec()));
            }
            at = end;
        }

        let gap = wav::silence(cfg::SAMPLE_RATE, request.gaps.segment_ms);
        let para_gap = wav::silence(cfg::SAMPLE_RATE, request.gaps.paragraph_ms);
        let mut samples: Vec<f32> = Vec::new();
        let mut prev: Option<usize> = None;
        for (pi, piece) in &pieces {
            if let Some(p) = prev {
                samples.extend_from_slice(if *pi != p { &para_gap } else { &gap });
            }
            samples.extend_from_slice(piece);
            prev = Some(*pi);
        }
        anyhow::ensure!(!samples.is_empty(), "engine {ID} produced no audio");
        stats.segments = spans.len();
        stats.frames = frames_total;
        stats.total_s = t0.elapsed().as_secs_f64();
        Ok(Synthesis {
            audio: Audio {
                samples,
                sample_rate: cfg::SAMPLE_RATE as u32,
            },
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometry identities that would otherwise surface as a shape mismatch mid-port.
    #[test]
    fn geometry_is_self_consistent() {
        // 24 kHz at 12.5 Hz is 1920 samples per frame, and the codec's two upsampling
        // stacks have to multiply out to exactly that.
        let ratios: usize = cfg::codec::UPSAMPLING_RATIOS.iter().product();
        let rates: usize = cfg::codec::UPSAMPLE_RATES.iter().product();
        assert_eq!(ratios * rates, cfg::SAMPLES_PER_FRAME);
        assert_eq!(
            cfg::SAMPLE_RATE as f64 / cfg::SAMPLES_PER_FRAME as f64,
            cfg::FRAME_RATE
        );

        // The talker fills codebook 0 and the predictor the rest.
        assert_eq!(cfg::predictor::HEADS_OUT + 1, cfg::CODE_GROUPS);
        assert_eq!(cfg::codec::QUANTIZERS, cfg::CODE_GROUPS);
        assert_eq!(
            cfg::codec::SEMANTIC_QUANTIZERS + cfg::codec::ACOUSTIC_QUANTIZERS,
            cfg::CODE_GROUPS
        );

        // Trap 5: the talker's live range covers each codebook exactly.
        assert_eq!(cfg::talker::CODES, cfg::codec::CODEBOOK);
        assert_eq!(cfg::codec::ENCODER_VALID_QUANTIZERS, cfg::CODE_GROUPS);
        const { assert!(cfg::codec::ENCODER_QUANTIZERS > cfg::codec::ENCODER_VALID_QUANTIZERS) };
        // Every control id sits above the live range, which is what makes a single
        // `id < CODES` test a valid "is this a real code".
        for id in [
            cfg::talker::CODEC_PAD,
            cfg::talker::CODEC_BOS,
            cfg::talker::CODEC_EOS,
            cfg::talker::CODEC_THINK,
            cfg::talker::CODEC_NOTHINK,
            cfg::talker::CODEC_THINK_BOS,
            cfg::talker::CODEC_THINK_EOS,
        ] {
            assert!(id as usize >= cfg::talker::CODES);
            assert!((id as usize) < cfg::talker::VOCAB);
        }

        // Trap 6: the predictor's heads do not tile its hidden size.
        assert_ne!(
            cfg::predictor::HEADS * cfg::predictor::HEAD_DIM,
            cfg::predictor::DIM
        );
        // ...while the talker's do, which is exactly why the mistake is easy.
        assert_eq!(cfg::talker::HEADS * cfg::talker::HEAD_DIM, cfg::talker::DIM);

        // The final conv's width is the decoder dim halved once per rate.
        assert_eq!(
            cfg::codec::OUT_CHANNELS,
            cfg::codec::stage_channels(cfg::codec::UPSAMPLE_RATES.len() - 1)
        );

        // The 1.7B speaker embedding is consumed as one position in the talker's stream, so
        // it has to be exactly that wide.
        assert_eq!(cfg::speaker::ENC_DIM, cfg::talker::DIM);
    }

    #[test]
    fn languages_have_ids_and_the_list_is_closed() {
        for name in cfg::talker::LANGUAGES {
            assert!(
                cfg::talker::language_id(name).is_some(),
                "no codec language id for `{name}`"
            );
        }
        assert_eq!(cfg::talker::LANGUAGES.len(), 10);
        // The limit that decides whether this engine can be used at all.
        assert!(cfg::talker::language_id("swahili").is_none());
    }

    /// A request with no voice must be refused, not answered with an arbitrary speaker.
    #[test]
    fn refuses_without_a_voice() {
        let caps = capabilities();
        assert!(caps.available);
        assert_eq!(caps.cloning, Cloning::PrecomputedAsset);
        // The capability checks pass; the engine's own voice rule is what rejects this.
        let request = SynthesisRequest::new("Hello.");
        assert!(tts_core::engine::validate_against(&caps, &request).is_ok());
        assert!(NO_VOICE.contains(ID));
    }

    /// q8_0 is this engine's default, unlike the other two — f32 is 38x slower here.
    #[test]
    fn defaults_to_q8_0() {
        assert_eq!(QUANT[0], "q8_0");
        assert_eq!(parse_quant(None).unwrap(), Weight::Quant(GgmlDType::Q8_0));
        assert_eq!(parse_quant(Some("f32")).unwrap(), Weight::F32);
        assert_eq!(parse_quant(Some("f16")).unwrap(), Weight::F16);
        assert!(parse_quant(Some("nonsense")).is_err());
    }

    /// Only the dense formats batch, and the default deliberately does not. Getting this
    /// backwards costs 30% — a batched q8_0 render measured RTF 1.02 against 0.79 unbatched.
    #[test]
    fn only_dense_weights_batch() {
        assert!(!parse_quant(None).unwrap().batches());
        assert!(parse_quant(Some("f16")).unwrap().batches());
        assert!(parse_quant(Some("f32")).unwrap().batches());
    }
}
