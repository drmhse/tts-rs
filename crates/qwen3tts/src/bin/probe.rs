//! Look at what the talker actually emits, a few frames at a time.
//!
//! The engine's decode loop runs to `max_new_tokens` when the model never emits `codec_eos`,
//! which on a wrong prompt means minutes of garbage before anything is visible. This prints
//! the first frames' codes and the per-frame cost instead.
//!
//! Run: `cargo run -p qwen3tts --release --bin qwen3tts-probe -- [frames]`

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use qwen3tts::cfg;
use qwen3tts::talker::{Sampling, Talker};
use std::time::Instant;
use tts_core::rng::Rng;
use tts_core::Voice;
use tts_nn::Weight;

const WEIGHTS: &str = "references/qwen3tts/weights/model.safetensors";
const VOICE: &str = "voices/cosy-default-qwen3tts";

fn main() -> Result<()> {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(16);

    // `q8_0` by default: f32 on a bf16 checkpoint is 6.3 GB for this trunk alone.
    let quant = match std::env::args().nth(2).as_deref() {
        Some("f32") => Weight::F32,
        Some("f16") => Weight::F16,
        Some("q4_0") => Weight::Quant(GgmlDType::Q4_0),
        _ => Weight::Quant(GgmlDType::Q8_0),
    };
    let device = Device::new_metal(0).context("opening the Metal device")?;
    let t = Instant::now();
    let talker = Talker::load(WEIGHTS, quant, &device)?;
    println!("loaded talker in {:.1} s", t.elapsed().as_secs_f64());

    let voice = Voice::load(VOICE)?;
    let spk = talker.speaker(voice.get("spk_embedding")?)?;
    let ref_codes = voice.get_rows_u32("ref_codes").unwrap_or_default();
    let ref_text = voice
        .get_rows_u32("ref_text_tokens")
        .ok()
        .and_then(|r| r.into_iter().next())
        .unwrap_or_default();
    println!(
        "voice: {} ref frames, {} ref text tokens, x-vector {:?}",
        ref_codes.len(),
        ref_text.len(),
        spk.dims()
    );

    // The text ids the engine would produce. Hardcoded here so the probe does not need the
    // BPE: these are Qwen's ids for TEXT, taken from the reference tokenizer.
    let ids: Vec<u32> = vec![9707, 504, 23361, 13];

    for (label, use_icl) in [("icl", true), ("x-vector only", false)] {
        let (prompt, trailing) = talker.build_prompt(
            &ids,
            if use_icl { &ref_text } else { &[] },
            if use_icl { &ref_codes } else { &[] },
            Some(&spk),
            cfg::talker::language_id("english"),
        )?;
        println!(
            "\n=== {label}: prompt {:?}, trailing {:?}",
            prompt.dims(),
            trailing.dims()
        );

        let s = Sampling {
            greedy: true,
            ..Sampling::default()
        };
        let mut rng = Rng::new(1234);
        let t = Instant::now();
        let (out, left, timing) = talker.generate(&prompt, &trailing, frames, &s, &mut rng)?;
        device.synchronize()?;
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{} frames in {secs:.2} s ({:.0} ms/frame), {left} text position(s) unconsumed",
            out.len(),
            secs / out.len().max(1) as f64 * 1000.0
        );
        let per = |s: f64| s / timing.frames.max(1) as f64 * 1000.0;
        println!(
            "  prefill {:.0} ms ({} prompt positions), talker {:.1} ms/frame, \
             predictor {:.1} ms/frame",
            timing.prefill_s * 1000.0,
            prompt.dim(1).unwrap_or(0),
            per(timing.talker_s),
            per(timing.predictor_s),
        );
        // The 15 depth steps, each timer synchronised at both ends.
        println!(
            "    depth: {:.1} ms head gemm + {:.1} ms host reads + {:.1} ms stack, \
             {:.1} ms elsewhere",
            per(timing.depth_gemm_s),
            per(timing.depth_read_s),
            per(timing.depth_stack_s),
            per(timing.predictor_s
                - timing.depth_gemm_s
                - timing.depth_read_s
                - timing.depth_stack_s),
        );
        for (i, f) in out.iter().enumerate().take(8) {
            println!("  frame {i:>2}: {:?}", f);
        }
        // A stuck loop repeats one code; a working one does not.
        let distinct: std::collections::BTreeSet<u32> = out.iter().map(|f| f[0]).collect();
        println!(
            "  codebook 0: {} distinct value(s) over {} frames",
            distinct.len(),
            out.len()
        );
    }
    Ok(())
}
