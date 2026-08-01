//! Does the batching *schedule* pay, on the real segment set?
//!
//! Separate from `arbatch`, which measures per-step cost at a fixed step count. The question
//! here is the other factor in
//!
//! ```text
//! end-to-end gain = (sum(frames) / max(frames)) x (cost_1 / cost_b)
//! ```
//!
//! — how ragged the real segments are, and therefore whether a given `max_batch` clears its
//! break-even.
//!
//! **Why this is a probe and not a CLI sweep.** Running `tts speak` once per `max_batch` and
//! comparing wall clock is invalid on this machine: consecutive heavy runs throttle, and a
//! sweep like that produced RTF 0.946 for a configuration that is genuinely faster, with the
//! *codec* — untouched by the change — appearing 29% slower in the same run. The canary read
//! 204 ms against 60 ms cool. Interleaving the variants inside one process is the only way to
//! get a trustworthy ratio; see `docs/benchmarking.md`.
//!
//! Run: `cargo run -p a8-probe --release --bin arschedule`

use a8::ar::{plan_batches, GenConfig, Model};
use a8::prompt::PromptBuilder;
use a8::sample::Rng;
use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};
use std::path::Path;
use tts_bench::Harness;

const WEIGHTS: &str = "oracle/weights/model.safetensors";
const TOKENIZER: &str = "oracle/weights/tokenizer.json";
const VOICE: &str = "voices/cosy-default";
const TEXT: &str = "examples/senior.txt";

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;

    let model = Model::load(WEIGHTS, &dev, Some(GgmlDType::Q8_0))?;
    let builder = PromptBuilder::load(TOKENIZER)?;

    // The real voice and the real segmentation, so the raggedness is the raggedness that
    // actually occurs rather than something synthetic.
    let voice = tts_core::Voice::load(Path::new(VOICE))?;
    let codes = voice.get_rows_u32("reference_codes")?;
    let text = std::fs::read_to_string(TEXT).with_context(|| format!("reading {TEXT}"))?;
    let paragraphs = tts_core::text::segment(&text, 220);
    let segments: Vec<&String> = paragraphs.iter().flatten().collect();

    let prompts: Vec<a8::prompt::Prompt> = segments
        .iter()
        .map(|s| builder.build(s, Some((codes.as_slice(), voice.text.as_str()))))
        .collect::<Result<Vec<_>>>()?;
    let widths: Vec<usize> = prompts.iter().map(|p| p.len).collect();
    println!("{} segments, prompt widths {widths:?}", prompts.len());

    let gen = GenConfig {
        max_new_tokens: 512,
        do_sample: true,
        ..GenConfig::default()
    };

    // Frame counts per segment at batch 1, which is what sets the raggedness. Reported
    // because it is the number that decides whether any of this can pay.
    {
        let mut frames: Vec<usize> = Vec::new();
        for p in &prompts {
            let mut rng = Rng::new(1234);
            frames.push(model.generate(p, &gen, &mut rng)?[0].len());
        }
        let sum: usize = frames.iter().sum();
        let max = *frames.iter().max().unwrap();
        println!("frames per segment {frames:?}");
        println!(
            "sum {sum}, max {max}  ->  sum/max = {:.2}  (batch 2 needs 1.48, 4 needs 2.43, \
             8 needs 4.28)",
            sum as f64 / max as f64
        );
    }

    // The schedule the engine runs: sort by width, group, decode each group.
    let run = |max_batch: usize| -> candle_core::Result<()> {
        let mut order: Vec<usize> = (0..prompts.len()).collect();
        order.sort_by_key(|&i| prompts[i].len);
        let mut rng = Rng::new(1234);
        for group in plan_batches(order.len(), max_batch) {
            let refs: Vec<&a8::prompt::Prompt> =
                group.iter().map(|&g| &prompts[order[g]]).collect();
            model
                .generate_batch(&refs, &gen, &mut rng)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
        Ok(())
    };

    h.ab(
        "AR schedule over the real 7 segments",
        &mut [
            ("max_batch 1 (sequential)", &mut || run(1)),
            ("max_batch 2", &mut || run(2)),
            ("max_batch 4", &mut || run(4)),
            ("max_batch 8 (one group)", &mut || run(8)),
        ],
    )?;

    let _ = Tensor::zeros(1, candle_core::DType::F32, &dev)?;
    h.report_drift()?;
    Ok(())
}
