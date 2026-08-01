//! Does batching the real AR loop pay?
//!
//! `arloop.rs` measured a *layer* at various batch sizes and found per-sequence cost falling
//! almost linearly — up to 11.95x at batch 32. Wiring `generate_batch` into the engine then
//! produced **1.008x** end to end. One of those two numbers is not measuring what it appears
//! to, and this finds out which.
//!
//! The distinction that matters: a layer benchmark holds work constant and varies batch. A
//! batched *generation* also changes how many steps run — sequential decodes
//! `sum(frames_i)` steps, batched decodes `max(frames_i)` — so the end-to-end win is
//! `sum/max` times the per-step penalty, and if the segments are ragged the first factor can
//! be far smaller than the batch size.
//!
//! Run: `cargo run -p a8-probe --release --bin arbatch`

use a8::ar::{GenConfig, Model};
use a8::prompt::PromptBuilder;
use a8::sample::Rng;
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use std::time::Instant;
use tts_bench::Harness;

const WEIGHTS: &str = "oracle/weights/model.safetensors";
const TOKENIZER: &str = "oracle/weights/tokenizer.json";

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;

    let model = Model::load(WEIGHTS, &dev, Some(GgmlDType::Q8_0))?;
    let builder = PromptBuilder::load(TOKENIZER)?;
    println!("loaded model (q8_0)\n");

    // A fixed number of decode steps for every configuration, so this isolates the
    // *per-step* cost of a batch from the question of how many steps a batch has to run.
    const STEPS: usize = 24;
    // Sampling on, because that is the production path and because the sampler is the one
    // part of a step that scales linearly with the batch. Measuring this greedy would hide
    // exactly the cost in question — `gumbel_argmax` is never called when `do_sample` is
    // false. A fixed step count keeps the comparison about per-step cost; the sampled tokens
    // differ per configuration, which does not matter here.
    let gen = GenConfig {
        max_new_tokens: STEPS,
        do_sample: true,
        ..GenConfig::default()
    };

    let prompt = builder.build("Welcome to Audio8 TTS, a test of batched decoding.", None)?;
    println!(
        "prompt width {}, {STEPS} greedy steps per sequence",
        prompt.len
    );

    // Per-step cost against batch size. Batch 2 and 3 are included precisely because they
    // are the documented trap: candle takes its matrix-vector kernel only at `dim(-2) == 1`.
    for batch in [1usize, 2, 3, 4, 8, 16] {
        let refs: Vec<&a8::prompt::Prompt> = (0..batch).map(|_| &prompt).collect();
        let mut rng = Rng::new(1);
        // Warm up, then time.
        let _ = model.generate_batch(&refs, &gen, &mut rng)?;
        dev.synchronize()?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut rng = Rng::new(1);
            let t = Instant::now();
            let out = model.generate_batch(&refs, &gen, &mut rng)?;
            dev.synchronize()?;
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(out.len(), batch);
        }
        let ms = tts_bench::median(&samples);
        if batch == 1 {
            BASE.set(ms).ok();
        }
        let base = *BASE.get().expect("batch 1 runs first");
        let per_seq = ms / batch as f64;
        println!(
            "batch {batch:>2}: {ms:>9.1} ms / {STEPS} steps = {:>7.2} ms/step, \
             {:>7.3} ms/step/seq  ->  {:>5.2}x per sequence   (break-even sum/max: {:>5.2})",
            ms / STEPS as f64,
            per_seq / STEPS as f64,
            base / per_seq,
            ms / base,
        );
    }

    // How much of a decode step is *waiting*, not computing.
    //
    // Every frame ends in eleven `to_vec1` calls — one for the semantic logits, ten for the
    // fast AR's codebooks — because the sampler runs on the host. Each is a pipeline flush.
    // This compares the real loop against the identical kernels with no readback at all, so
    // the gap is synchronisation plus host-side sampling. If that gap is large, the next win
    // for this loop is moving sampling onto the device, not making the matmuls faster.
    {
        const N: usize = 24;
        let sampled = GenConfig {
            max_new_tokens: N,
            do_sample: true,
            ..GenConfig::default()
        };
        let (m1, p1) = (&model, prompt.clone());
        let (m2, p2) = (&model, prompt.clone());
        h.ab(
            "a decode step: how much is synchronisation?",
            &mut [
                ("full loop (samples on the host)", &mut move || {
                    let mut rng = Rng::new(1);
                    m1.generate(&p1, &sampled, &mut rng)
                        .map_err(|e: anyhow::Error| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("same kernels, no readback", &mut move || {
                    m2.bench_frame_gpu_only(&p2, N)
                        .map_err(|e: anyhow::Error| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
            ],
        )?;
    }

    h.report_drift()?;
    Ok(())
}

static BASE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
