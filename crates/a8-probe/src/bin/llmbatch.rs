//! What does a CosyVoice decode step actually cost as the batch grows?
//!
//! Batching the LLM across segments *should* be most of a 30% stage: seven segments take
//! 1317 sequential steps but only about as many batched steps as the longest one, and a
//! decode step is supposed to be dominated by reading 24 layers of weights — a cost the
//! whole batch shares.
//!
//! It measured **slower** (RTF 0.217 -> 0.263 quantized, 0.239 dense). So either the step
//! count is not what I assume, or a batch-b step costs ~b times a batch-1 step, which would
//! mean the decode is not weight-bound at all.
//!
//! This measures the step cost directly. If it scales linearly with `b`, batching cannot
//! win no matter how the loop is arranged.
//!
//! Run: `cargo run -p a8-probe --release --bin llmbatch`

use anyhow::Result;
use candle_core::Device;
use cosy::llm::Llm;
use tts_bench::Harness;

const WEIGHTS: &str = "oracle-cosy/weights";
/// A plausible prompt width: voice transcript + target text + ~294 prompt speech tokens.
const PROMPT: usize = 420;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    for quant in ["q8_0", "none"] {
        let q = match quant {
            "q8_0" => Some(candle_core::quantized::GgmlDType::Q8_0),
            _ => None,
        };
        let llm = Llm::load(&format!("{WEIGHTS}/llm.safetensors"), q, &dev)?;
        println!("--- weights: {quant} ---");

        let mut rows = Vec::new();
        for b in [1usize, 2, 4, 7] {
            // A batch of identical prompts: the cost depends on width and batch, not on
            // which tokens they are.
            let text: Vec<u32> = (0..PROMPT - 300)
                .map(|i| (i % 1000) as u32 + 100)
                .chain(std::iter::once(cosy::cfg::llm::ENDOFPROMPT))
                .collect();
            let speech: Vec<u32> = (0..294).map(|i| (i % 5000) as u32).collect();
            let prompts: Vec<_> = (0..b).map(|_| (text.clone(), speech.clone(), 40)).collect();
            let built: Vec<_> = prompts
                .iter()
                .map(|(t, s, _)| llm.build_prompt(t, s).unwrap())
                .collect();

            let state = llm.prefill_batch(&built)?;
            let feed: Vec<u32> = vec![7; b];

            // One decode step, measured by handing the state back and forth.
            let mut st = Some(state);
            let mut step = || -> candle_core::Result<()> {
                let s = st.take().expect("state present");
                let s = llm
                    .step_batch(s, &feed)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                // Keep the width from running away over repeated samples.
                st = Some(s);
                Ok(())
            };
            // The same step, plus what the real loop does around it: project to logits,
            // read them back to the host, and sample. That readback is a full pipeline
            // flush, and the difference between these two variants is exactly the window
            // in which the GPU has nothing to do.
            let mut st2 = Some(llm.prefill_batch(&built)?);
            let mut sampler = cosy::sample::Sampler::new(cosy::cfg::llm::VOCAB);
            let mut rng = tts_core::rng::Rng::new(0);
            let hist: Vec<u32> = Vec::new();
            let mut full = || -> candle_core::Result<()> {
                let s = st2.take().expect("state present");
                let logits = llm
                    .logits(&s.hidden)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let flat = logits.flatten_all()?.to_vec1::<f32>()?;
                for lane in 0..b {
                    let mut sc = flat
                        [lane * cosy::cfg::llm::VOCAB..(lane + 1) * cosy::cfg::llm::VOCAB]
                        .to_vec();
                    cosy::sample::ras_sampling(&mut sampler, &mut sc, &hist, &mut rng);
                }
                let s = llm
                    .step_batch(s, &feed)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                st2 = Some(s);
                Ok(())
            };
            let stats = h.ab(
                &format!("{quant} decode b={b}"),
                &mut [("gpu only", &mut step), ("with sampling", &mut full)],
            )?;
            rows.push((b, stats[0].median, stats[1].median));
        }

        let base = rows[0].1;
        println!(
            "\n{:>6} {:>9} {:>11} {:>10} {:>13} {:>12}",
            "batch", "gpu ms", "full ms", "host ms", "host share", "per seq (gpu)"
        );
        for (b, gpu, full) in &rows {
            println!(
                "{b:>6} {gpu:>9.2} {full:>11.2} {:>10.2} {:>12.0}% {:>11.2}x",
                full - gpu,
                100.0 * (full - gpu) / full,
                (gpu / *b as f64) / base
            );
        }
        println!();
    }

    h.report_drift()?;
    Ok(())
}
