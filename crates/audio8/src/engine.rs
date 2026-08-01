//! Audio8 behind the engine-neutral [`Engine`] trait.
//!
//! Everything model-specific stays in the other modules; this is the adapter that
//! makes Audio8 selectable alongside other engines, plus the segment loop and the
//! stitching that used to live in the CLI.

use crate::ar::{GenConfig, Model};
use crate::cfg;
use crate::codec::Codec;
use crate::prompt::PromptBuilder;
use crate::sample::Rng;
use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use std::sync::Mutex;
use std::time::Instant;
use tts_core::{
    text, wav, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis,
    SynthesisRequest,
};

pub const ID: &str = "audio8";

/// Which weight formats load. Only the block-32 ggml types apply: the K-quants use
/// 256-element blocks and every projection here has `k = dim = 896`.
const QUANT: &[&str] = &["f32", "q8_0", "q5_0", "q4_1", "q4_0"];

/// How many segments to decode at once.
///
/// Every lane carries its own KV cache — `MAX_SEQ_LEN * n_kv * head_dim` f32 per layer per
/// lane, about 4 MB across the 24 slow layers — so this is a memory decision as much as a
/// throughput one. 8 keeps the caches under ~32 MB while capturing most of the available
/// win: the measured curve is nearly flat from batch 4 to 32.
///
/// Overridable through `EngineConfig::overrides` with `--set max_batch=<n>`; `1` restores the
/// exactly-sequential behaviour, which is what to reach for when comparing against a
/// PyTorch render token by token.
const DEFAULT_MAX_BATCH: usize = 8;

pub struct Audio8Engine {
    model: Model,
    codec: Codec,
    prompts: PromptBuilder,
    max_batch: usize,
    /// The sampler carries RNG state, so a shared engine must serialise requests.
    rng: Mutex<Rng>,
}

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description: "Audio8-TTS-Preview-0.6b — DualAR + RVQ codec, 44.1 kHz",
        sample_rate: cfg::SAMPLE_RATE as u32,
        frame_rate: cfg::frame_rate(),
        cloning: Cloning::PrecomputedAsset,
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
        s @ ("q4_K" | "q5_K" | "q6_K") => {
            anyhow::bail!("{s} uses 256-element blocks; every k=896 projection fails. Use q8_0.")
        }
        other => anyhow::bail!("unknown quant {other} for engine {ID}; try one of {QUANT:?}"),
    })
}

impl Audio8Engine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let device = if config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).context("opening the Metal device")?
        };
        let quant = parse_quant(config.quant.as_deref())?;
        let weights = config.path("weights", "model.safetensors");
        let codec = config.path("codec", "codec.safetensors");
        let tokenizer = config.path("tokenizer", "tokenizer.json");
        Ok(Self {
            model: Model::load(
                weights.to_str().context("non-utf8 weights path")?,
                &device,
                quant,
            )?,
            codec: Codec::load(codec.to_str().context("non-utf8 codec path")?, &device)?,
            prompts: PromptBuilder::load(tokenizer.to_str().context("non-utf8 tokenizer path")?)?,
            max_batch: match config.overrides.get("max_batch") {
                None => DEFAULT_MAX_BATCH,
                Some(p) => p
                    .to_str()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| n >= 1)
                    .context("--set max_batch=<n> needs a positive integer")?,
            },
            rng: Mutex::new(Rng::new(1234)),
        })
    }
}

impl Engine for Audio8Engine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis> {
        self.validate(request)?;

        // Cloning needs the clip's codes and its transcript together: the prompt
        // interleaves them.
        let reference = match &request.voice {
            None => None,
            Some(v) => {
                anyhow::ensure!(
                    !v.text.trim().is_empty(),
                    "voice `{}` has no transcript; the prompt interleaves the clip's text \
                     with its codes, so `text` in voice.json is required",
                    v.name
                );
                Some((v.get_rows_u32("reference_codes")?, v.text.clone()))
            }
        };

        let paragraphs = text::segment(&request.text, request.max_chars);
        let flat: Vec<(usize, &String)> = paragraphs
            .iter()
            .enumerate()
            .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s)))
            .collect();
        anyhow::ensure!(!flat.is_empty(), "no text to speak");

        let gen = GenConfig {
            max_new_tokens: request.max_new_tokens,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            do_sample: !request.sampling.greedy,
        };
        let mut rng = Rng::new(request.sampling.seed);

        let mut stats = Stats::default();
        let t0 = Instant::now();

        // Build every prompt up front so they can be grouped by width. A decode step is a
        // matrix-vector product — bus-bound, arithmetic units idle — so batching is the
        // single largest lever available here, up to 11.95x per sequence.
        let reference = reference
            .as_ref()
            .map(|(codes, txt)| (codes.as_slice(), txt.as_str()));
        let prompts: Vec<crate::prompt::Prompt> = flat
            .iter()
            .map(|(_, seg)| self.prompts.build(seg, reference))
            .collect::<Result<Vec<_>>>()?;

        // Sort by prompt width before grouping. Right-alignment pads every sequence up to
        // the widest in its group, and a finished lane keeps computing until the group is
        // done, so both wastes shrink when similar lengths travel together.
        let mut order: Vec<usize> = (0..prompts.len()).collect();
        order.sort_by_key(|&i| prompts[i].len);

        let mut codes_by_index: Vec<Option<Vec<Vec<u32>>>> = vec![None; prompts.len()];
        for group in crate::ar::plan_batches(order.len(), self.max_batch) {
            let idx: Vec<usize> = group.iter().map(|&g| order[g]).collect();
            let refs: Vec<&crate::prompt::Prompt> = idx.iter().map(|&i| &prompts[i]).collect();
            let t_ar = Instant::now();
            let out = self.model.generate_batch(&refs, &gen, &mut rng)?;
            stats.add("ar", t_ar.elapsed().as_secs_f64());
            for (slot, codes) in idx.iter().zip(out) {
                codes_by_index[*slot] = Some(codes);
            }
        }

        let mut pieces: Vec<(usize, Vec<f32>)> = Vec::new();
        for (k, (pi, _)) in flat.iter().enumerate() {
            let codes = match codes_by_index[k].take() {
                Some(c) => c,
                None => continue,
            };
            let frames = codes[0].len();
            if frames == 0 {
                continue;
            }
            stats.frames += frames;
            stats.segments += 1;

            let t_voc = Instant::now();
            let audio = self.codec.decode(&codes)?;
            let samples = audio.flatten_all()?.to_vec1::<f32>()?;
            stats.add("codec", t_voc.elapsed().as_secs_f64());
            pieces.push((*pi, samples));
        }
        stats.total_s = t0.elapsed().as_secs_f64();
        // Keep the shared RNG advancing so successive requests on one engine instance
        // do not all replay the same stream when the caller reuses a seed.
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
