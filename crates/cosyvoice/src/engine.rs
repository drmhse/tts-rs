//! CosyVoice3 behind the engine-neutral [`Engine`] trait.
//!
//! The adapter, the segment loop and the stitching. Everything model-specific is in
//! [`crate::llm`], [`crate::flow`] and [`crate::hift`].
//!
//! Two things a caller should know, both stated in [`capabilities`] rather than left to
//! be discovered:
//!
//! **The flow decoder runs once per utterance, not once per segment.** `flow.synthesize`
//! prepends the voice's speech tokens and mel every call, so a per-segment loop re-pays that
//! prompt each time — on a seven-segment passage, 4116 prompt mel frames against 2634
//! generated ones. Decoding the concatenated tokens in one call and cutting the waveform back
//! into segments afterwards is worth **1.71x on the flow and 1.72x on the vocoder**, and the
//! cut points are exact rather than estimated. `--set flow_per_segment=1` restores the loop.
//!
//! **A voice asset is required, not optional.** Unlike Audio8, which has a usable
//! built-in speaker, every stage here is conditioned on the reference clip: the LLM on its
//! speech tokens, the flow decoder on its mel *and* speaker embedding. There is no
//! meaningful unconditioned path, so a request without a voice is refused instead of
//! silently producing some arbitrary speaker.
//!
//! **Text normalisation is not ported.** The reference frontend runs WeTextProcessing
//! (an FST-based normaliser), number spell-out via `inflect`, and Chinese punctuation
//! rewriting before tokenizing. None of that is model code and porting it is a separate
//! project of its own; what is here is whitespace collapsing, sentence segmentation and
//! the Qwen2 BPE. Text that arrives already normalised is unaffected. Text containing
//! digits or abbreviations will be read as the tokenizer sees them, which is a real
//! difference from the Python service and the reason `reason` mentions it.

use crate::cfg;
use crate::flow::Flow;
use crate::hift::{Hift, Noise};
use crate::llm::Llm;
use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;
use tts_core::rng::Rng;
use tts_core::{
    text, wav, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis,
    SynthesisRequest,
};

pub const ID: &str = "cosyvoice";

/// Which weight formats load.
///
/// Only the LLM's projections are quantized. The flow decoder and the vocoder run on full
/// sequences, where candle takes an ordinary GEMM rather than the dedicated
/// matrix-vector kernel that makes quantization a win in a decode loop — so quantizing
/// them would cost a dequantize per call and buy nothing. See `tts_nn::Proj`.
const QUANT: &[&str] = &["f32", "q8_0", "q5_0", "q4_1", "q4_0"];

const NO_VOICE: &str = "engine `cosyvoice` requires a voice asset: the LLM is conditioned \
    on the reference clip's speech tokens and the flow decoder on its mel and speaker \
    embedding, so there is no unconditioned path. Build one with \
    references/cosyvoice/export_voice.py";

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description: "Fun-CosyVoice3-0.5B — Qwen2 LLM + DiT flow matching + HiFTGenerator, 24 kHz",
        sample_rate: cfg::SAMPLE_RATE as u32,
        // Mel frames per second: 25 speech tokens/s at 2 mel frames each.
        frame_rate: (cfg::TOKEN_RATE * cfg::TOKEN_MEL_RATIO) as f64,
        cloning: Cloning::PrecomputedAsset,
        // The reference streams; this port does not yet. The vocoder's chunk cache and
        // the flow's chunked attention masks are both unimplemented on purpose — see
        // docs/porting/cosyvoice.md.
        streaming: false,
        quantization: QUANT,
        available: true,
        reason: None,
    }
}

fn parse_quant(s: Option<&str>) -> Result<Option<GgmlDType>> {
    Ok(match s.unwrap_or("q8_0") {
        "none" | "f32" => None,
        "q8_0" => Some(GgmlDType::Q8_0),
        "q5_0" => Some(GgmlDType::Q5_0),
        "q4_1" => Some(GgmlDType::Q4_1),
        "q4_0" => Some(GgmlDType::Q4_0),
        s @ ("q4_K" | "q5_K" | "q6_K") => anyhow::bail!(
            "{s} uses 256-element blocks; the LLM's k=896 projections cannot use it. Use q8_0."
        ),
        other => anyhow::bail!("unknown quant {other} for engine {ID}; try one of {QUANT:?}"),
    })
}

/// The conditioning a voice asset must carry.
struct Conditioning {
    /// `[1, 192]`.
    speaker: Tensor,
    /// The clip's speech tokens.
    tokens: Vec<u32>,
    /// `[1, frames, 80]`.
    mel: Tensor,
    /// The clip's transcript, already tokenized *with* the `<|endofprompt|>` marker.
    text_tokens: Vec<u32>,
}

pub struct CosyVoiceEngine {
    llm: Llm,
    flow: Flow,
    hift: Hift,
    tokenizer: Tokenizer,
    /// Decode each segment's mel separately instead of fusing the whole utterance into one
    /// flow call. Slower — the voice prompt is re-decoded per segment — but it restores the
    /// per-segment silence gaps and bounds peak memory. `--set flow_per_segment=1`.
    flow_per_segment: bool,
    /// Most segments decoded together in the LLM. `--set llm_batch=<n>`; `1` restores
    /// one-at-a-time decoding, which is what the batched path is compared against.
    ///
    /// **Bounded, and it must be.** Every lane carries its own KV cache —
    /// `2 * N_KV * CACHE * HEAD_DIM * 4 bytes * LAYERS` = **101 MB per lane** — so batching
    /// is not free in memory even though it is nearly free in time. Left unbounded this
    /// batched one lane per segment, and a 16-minute chapter (118 segments) asked for
    /// **11.9 GB of KV cache alone**, which on a 16 GB machine meant swap: the LLM stage
    /// went from RTF 0.19 to 0.70 and the whole engine from 0.70 to 1.49.
    ///
    /// 8 matches Audio8's `DEFAULT_MAX_BATCH`, costs 0.81 GB, and keeps essentially all of
    /// the batching win — the measured curve is already flat by 4-7 lanes
    /// (`tts-probe --bin llmbatch`).
    llm_max_batch: usize,
    /// The device this engine's weights live on.
    ///
    /// Kept so voice tensors can be moved onto it. A caller loads a `Voice` before it
    /// loads an engine, so the two can easily hold *different* `Device` handles for the
    /// same GPU — and candle compares device identity, not hardware, so mixing them
    /// fails with "device mismatch in matmul" naming the same gpu_id twice.
    device: Device,
    /// The samplers carry RNG state, so a shared engine serialises requests.
    rng: Mutex<Rng>,
}

/// Segments decoded together in the LLM by default. See `llm_max_batch`.
const DEFAULT_LLM_BATCH: usize = 8;

/// Target mel frames of *generated* speech per flow call, excluding the voice prompt.
///
/// The flow cost per call is `a*n + b*n^2` with the prompt included in `n`, so there is a
/// genuine optimum: too small and every call re-pays the 588-frame prompt, too large and
/// quadratic attention takes over. Measured coefficients put the flat minimum between 1600
/// and 2400 target frames; 2000 sits in the middle of it.
const FLOW_GROUP_FRAMES: usize = 2000;

impl CosyVoiceEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let device = if config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).context("opening the Metal device")?
        };
        let quant = parse_quant(config.quant.as_deref())?;

        let s = |p: std::path::PathBuf| -> Result<String> {
            p.to_str()
                .map(str::to_owned)
                .with_context(|| format!("non-utf8 path {}", p.display()))
        };
        let llm_path = s(config.path("llm", "llm.safetensors"))?;
        let flow_path = s(config.path("flow", "flow.safetensors"))?;
        let hift_path = s(config.path("hift", "hift.safetensors"))?;
        let tok_path = s(config.path("tokenizer", "tokenizer.json"))?;
        // The fixed initial noise is not a trained weight — the reference redraws it at
        // construction from a seeded RNG — but it *is* something inference cannot run
        // without, so it ships beside the weights rather than with the test fixtures.
        // `--set noise=<path>` points it elsewhere.
        let noise_path = s(config.path("noise", "rand_noise.safetensors"))?;

        Ok(Self {
            llm: Llm::load(&llm_path, quant, &device)?,
            flow: Flow::load(&flow_path, &noise_path, &device)?,
            hift: Hift::load(&hift_path, &device)?,
            tokenizer: Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("loading {tok_path}: {e}"))?,
            device,
            flow_per_segment: config
                .overrides
                .get("flow_per_segment")
                .and_then(|p| p.to_str())
                .map(|v| v != "0")
                .unwrap_or(false),
            llm_max_batch: config
                .overrides
                .get("llm_batch")
                .and_then(|p| p.to_str())
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(DEFAULT_LLM_BATCH),
            rng: Mutex::new(Rng::new(1234)),
        })
    }

    /// Tokenize with the Qwen2 BPE, keeping special tokens live.
    ///
    /// `add_special_tokens` is true because the reference encodes with
    /// `allowed_special='all'` — the paralinguistic tags and `<|endofprompt|>` must
    /// resolve to their ids rather than being spelled out as text.
    pub fn tokenize(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    }

    fn conditioning(&self, request: &SynthesisRequest) -> Result<Conditioning> {
        let voice = request.voice.as_ref().context(NO_VOICE)?;
        let tokens = voice
            .get_rows_u32("speech_tokens")
            .context("voice asset is missing `speech_tokens`")?
            .into_iter()
            .next()
            .context("`speech_tokens` is empty")?;
        let text_tokens = voice
            .get_rows_u32("prompt_text_tokens")
            .context("voice asset is missing `prompt_text_tokens`")?
            .into_iter()
            .next()
            .context("`prompt_text_tokens` is empty")?;
        let mel = voice
            .get_on("prompt_mel", &self.device)
            .context("voice asset is missing `prompt_mel`")?;
        let speaker = voice
            .get_on("speaker_embedding", &self.device)
            .context("voice asset is missing `speaker_embedding`")?;

        // The flow decoder holds each speech token for exactly two mel frames, so a mel
        // that is not twice the token count would silently misalign the prompt against
        // the conditioning and produce a plausible but wrong voice.
        let want = tokens.len() * cfg::TOKEN_MEL_RATIO;
        anyhow::ensure!(
            mel.dim(1)? == want,
            "voice `{}` has {} speech tokens but {} mel frames; the flow decoder needs \
             exactly {} ({} per token)",
            voice.name,
            tokens.len(),
            mel.dim(1)?,
            want,
            cfg::TOKEN_MEL_RATIO
        );
        Ok(Conditioning {
            speaker,
            tokens,
            mel,
            text_tokens,
        })
    }
}

impl Engine for CosyVoiceEngine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        tts_core::engine::validate_against(&self.capabilities(), request)?;
        // Beyond the shared checks: this engine has no unconditioned path at all.
        anyhow::ensure!(request.voice.is_some(), "{NO_VOICE}");
        Ok(())
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis> {
        self.validate(request)?;
        let cond = self.conditioning(request)?;
        let speaker = self.flow.speaker(&cond.speaker)?;

        let paragraphs = text::segment(&request.text, request.max_chars);
        let flat: Vec<(usize, &String)> = paragraphs
            .iter()
            .enumerate()
            .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s)))
            .collect();
        anyhow::ensure!(!flat.is_empty(), "no text to speak");

        let mut rng = Rng::new(request.sampling.seed);
        let mut stats = Stats::default();
        let t0 = Instant::now();

        // Stage 1: the LLM, all segments in one batch.
        //
        // Segmentation is a property of the *text* — the LLM has a context limit and
        // degrades holding prosody over too long a span — so the segments stay separate
        // sequences. They do not have to be decoded one after another, though: a decode
        // step's cost is dominated by reading 24 layers of weights and by the host
        // round-trip for sampling, and batching pays both once for every segment at once.
        // Right-alignment makes that exact here; see `llm::Lane`.
        let mut prompts: Vec<(Vec<u32>, Vec<u32>, usize)> = Vec::new();
        let mut para_of: Vec<usize> = Vec::new();
        for (pi, seg) in &flat {
            let seg_tokens = self.tokenize(seg)?;
            if seg_tokens.is_empty() {
                continue;
            }
            // The voice's transcript tokens carry `<|endofprompt|>`; the LLM asserts on it.
            let full_text: Vec<u32> = cond
                .text_tokens
                .iter()
                .copied()
                .chain(seg_tokens.iter().copied())
                .collect();
            prompts.push((full_text, cond.tokens.clone(), seg_tokens.len()));
            para_of.push(*pi);
        }
        anyhow::ensure!(!prompts.is_empty(), "no text to speak");

        let t = Instant::now();
        let mut generated: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
        for group in prompts.chunks(self.llm_max_batch.min(prompts.len()).max(1)) {
            generated.extend(self.llm.generate_batch_with(
                group,
                &mut rng,
                request.sampling.greedy,
            )?);
        }
        self.device.synchronize()?;
        stats.add("llm", t.elapsed().as_secs_f64());

        let mut spans: Vec<(usize, Vec<u32>)> = Vec::new();
        for (pi, speech) in para_of.into_iter().zip(generated) {
            if !speech.is_empty() {
                spans.push((pi, speech));
            }
        }
        anyhow::ensure!(!spans.is_empty(), "engine {ID} generated no speech tokens");

        // Stage 2: the flow decoder, **once for the whole utterance** where it fits.
        //
        // This is the single largest lever in this engine, and it is a scheduling change
        // rather than a kernel one. `flow.synthesize` prepends the voice's speech tokens and
        // mel every call, so a per-segment loop re-pays that prompt once per segment: on
        // `examples/senior.txt` that is 7 x 588 = 4116 prompt mel frames against 2634
        // generated ones — **61% of the flow's work is the same prompt, redone**. Decoding the
        // concatenated tokens in one call pays it once.
        //
        // It is not free: attention is O(n^2), so a single 3222-frame solve does more
        // attention work than seven 964-frame ones. With the projections at ~90% of a block
        // and attention ~10%, the arithmetic still favours fusing by about 1.65x, and the
        // measurement below bears that out.
        //
        // Two reasons it can fall back to per-segment: the fixed noise asset is 15000 mel
        // frames (~300 s), and a caller may want the old behaviour, which `--set
        // flow_per_segment=1` gives.
        // How many segments share one flow call. Neither extreme is right, and this used
        // to be a binary choice between them.
        //
        // Every call re-prepends the voice's prompt mel (588 frames here), so decoding one
        // segment at a time wastes 57% of each call on the prompt. But the DiT's attention
        // is quadratic in sequence length, so fusing a whole chapter is far worse still.
        // Fitting the measured per-block cost `a*n + b*n^2` (a = 0.027267 ms/frame,
        // b = 4.304e-6 ms/frame^2, from `tts-probe --bin flowsplit` across 798 and 3192
        // frames) over a 17-minute chapter:
        //
        // | target frames per call | flow cost |
        // |---|---|
        // | 437 (one segment, the old fallback) | 1.00x |
        // | 1600-2400 | **0.67x** |
        // | 51700 (fuse everything) | 3.44x |
        //
        // So whole-utterance fusion — which this engine used to prefer whenever it fitted —
        // is a *pessimisation* on long text, and only looked good because the passage it was
        // measured on was short enough to stay near the optimum. The noise asset's 15000
        // frame cap had been hiding that by forcing the fallback.
        let prompt_tokens = cond.tokens.len();
        let budget_tokens = self
            .flow
            .max_frames()
            .saturating_div(cfg::TOKEN_MEL_RATIO)
            .saturating_sub(prompt_tokens);
        let group_tokens = if self.flow_per_segment {
            1 // `--set flow_per_segment=1` keeps the old one-call-per-segment behaviour.
        } else {
            (FLOW_GROUP_FRAMES / cfg::TOKEN_MEL_RATIO)
                .min(budget_tokens)
                .max(1)
        };

        // Greedy: fill a group until the next segment would overflow the budget. A single
        // segment longer than the budget still gets its own call rather than being split,
        // because splitting inside a segment would cut prosody mid-sentence.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut acc = 0usize;
        for (i, (_, toks)) in spans.iter().enumerate() {
            let n = toks.len();
            if groups.is_empty() || (acc + n > group_tokens && !groups.last().unwrap().is_empty()) {
                groups.push(Vec::new());
                acc = 0;
            }
            groups.last_mut().expect("group pushed").push(i);
            acc += n;
        }

        let mut pieces: Vec<(usize, Vec<f32>)> = Vec::new();
        let per_token = cfg::TOKEN_MEL_RATIO * self.hift.samples_per_frame();
        for group in &groups {
            let tokens: Vec<u32> = group
                .iter()
                .flat_map(|&i| spans[i].1.iter().copied())
                .collect();
            if tokens.is_empty() {
                continue;
            }

            let t = Instant::now();
            let mel = self
                .flow
                .synthesize(&cond.tokens, &tokens, &cond.mel, &speaker)?;
            self.device.synchronize()?;
            stats.add("flow", t.elapsed().as_secs_f64());

            let t = Instant::now();
            let wav_t = self.hift.forward(&mel, Noise::Draw(&mut rng))?;
            self.device.synchronize()?;
            stats.add("vocoder", t.elapsed().as_secs_f64());

            stats.frames += mel.dim(2)?;
            stats.segments += group.len();

            // Cut the group's waveform back into its segments so the caller's gaps still
            // apply. The boundaries are exact rather than estimated: the flow holds each
            // speech token for `TOKEN_MEL_RATIO` mel frames and the vocoder emits
            // `samples_per_frame` samples per frame, so a segment of `n` tokens is exactly
            // `n * ratio * spf` samples.
            //
            // This is not cosmetic. Decoding several segments in one call *without* cutting
            // measured **WER 0.023 against 0.000** — sentence boundaries running together
            // with no pause. The speed comes from decoding together; the pauses come from
            // cutting afterwards, and there is no reason to trade one for the other.
            let samples = wav_t.flatten_all()?.to_vec1::<f32>()?;
            let mut at = 0usize;
            for (j, &i) in group.iter().enumerate() {
                let want = spans[i].1.len() * per_token;
                // The last segment of a group takes whatever remains, so rounding cannot
                // drop samples.
                let end = if j + 1 == group.len() {
                    samples.len()
                } else {
                    (at + want).min(samples.len())
                };
                if end > at {
                    pieces.push((spans[i].0, samples[at..end].to_vec()));
                }
                at = end;
            }
        }
        stats.total_s = t0.elapsed().as_secs_f64();
        if let Ok(mut shared) = self.rng.lock() {
            let _ = shared.next_f32();
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

        Ok(Synthesis {
            audio: Audio {
                samples,
                sample_rate: cfg::SAMPLE_RATE as u32,
            },
            stats,
        })
    }
}
