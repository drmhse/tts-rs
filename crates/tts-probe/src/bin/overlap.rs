//! Do two GPU stages overlap, or just interleave?
//!
//! CosyVoice is now LLM 190 s + flow 499 s + vocoder 56 s on a 17-minute chapter (RTF 0.72).
//! The LLM is 25% of that and runs strictly before the flow, so if the two could overlap the
//! wall time would approach `max(llm, flow + vocoder)` instead of their sum — about RTF 0.54.
//! The Python service does exactly this and reports that its wall time approaches the larger
//! stage.
//!
//! I refuted the idea once on the wrong evidence: the *host* share of a decode step is 1-3%,
//! from which I concluded both stages were GPU-bound with no headroom. Host share is not GPU
//! utilisation. A decode step is many small dispatches over 896-wide matmuls and cannot
//! saturate the device; `llmbatch` shows it directly — 8 lanes cost 5.13x one lane, not 8x,
//! so ~36% of the width is idle even while the GPU is continuously busy.
//!
//! Whether that idle width can be *filled by another stream* is a different question, and
//! this answers it before any threading goes into the engine. Two threads, one running decode
//! steps and one running DiT blocks, against the same work done sequentially.
//!
//! Run: `cargo run -p tts-probe --release --bin overlap`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice::flow::Flow;
use cosyvoice::llm::Llm;
use std::collections::HashMap;
use std::time::Instant;

const WEIGHTS: &str = "references/cosyvoice/weights";
const FIXTURES: &str = "fixtures/cosyvoice/oracle.safetensors";
const NOISE: &str = "references/cosyvoice/weights/rand_noise.safetensors";
/// Enough of each to dominate any start-up noise.
const DECODE_STEPS: usize = 60;
const FLOW_EVALS: usize = 3;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &dev)
        .with_context(|| format!("loading {FIXTURES}"))?;
    let get = |n: &str| -> Result<Tensor> {
        Ok(fx
            .get(n)
            .with_context(|| format!("missing {n}"))?
            .to_dtype(DType::F32)?)
    };

    let llm = Llm::load(
        &format!("{WEIGHTS}/llm.safetensors"),
        Some(candle_core::quantized::GgmlDType::Q8_0),
        &dev,
    )?;
    let flow = Flow::load(&format!("{WEIGHTS}/flow.safetensors"), NOISE, &dev)?;
    let (mu, cond, spks, z) = (
        get("flow.mu")?,
        get("flow.cond")?,
        get("flow.spks")?,
        get("flow.z")?,
    );

    // A realistic decode batch: 8 lanes, the engine's default.
    let text: Vec<u32> = (0..120)
        .map(|i| (i % 900) as u32 + 100)
        .chain(std::iter::once(cosyvoice::cfg::llm::ENDOFPROMPT))
        .collect();
    let speech: Vec<u32> = (0..294).map(|i| (i % 5000) as u32).collect();
    let built: Vec<Tensor> = (0..8)
        .map(|_| llm.build_prompt(&text, &speech).unwrap())
        .collect();
    let feed = vec![7u32; 8];

    let decode = || -> Result<()> {
        let mut st = llm.prefill_batch(&built)?;
        for _ in 0..DECODE_STEPS {
            st = llm.step_batch(st, &feed)?;
        }
        Ok(())
    };
    let dit = || -> Result<()> {
        for _ in 0..FLOW_EVALS {
            flow.estimate(&z, &mu, &cond, &spks, 0.3)?;
        }
        Ok(())
    };

    // Warm up so neither measurement pays shader compilation.
    decode()?;
    dit()?;
    dev.synchronize()?;

    let t = Instant::now();
    decode()?;
    dev.synchronize()?;
    let llm_only = t.elapsed().as_secs_f64();

    let t = Instant::now();
    dit()?;
    dev.synchronize()?;
    let flow_only = t.elapsed().as_secs_f64();

    let t = Instant::now();
    std::thread::scope(|s| {
        let a = s.spawn(decode);
        let b = s.spawn(dit);
        a.join().expect("decode thread")?;
        b.join().expect("dit thread")?;
        Ok::<_, anyhow::Error>(())
    })?;
    dev.synchronize()?;
    let together = t.elapsed().as_secs_f64();

    let sequential = llm_only + flow_only;
    let ideal = llm_only.max(flow_only);
    println!("\n  decode alone      {llm_only:7.2} s");
    println!("  dit alone         {flow_only:7.2} s");
    println!("  sequential        {sequential:7.2} s");
    println!("  concurrent        {together:7.2} s");
    println!("  perfect overlap   {ideal:7.2} s  (the larger stage)");
    let recovered = (sequential - together) / (sequential - ideal).max(1e-9);
    println!(
        "\n  speedup {:.2}x, recovering {:.0}% of what perfect overlap would give",
        sequential / together,
        100.0 * recovered
    );
    if recovered < 0.15 {
        println!("  -> not worth threading the engine: the GPU has no width to spare here.");
    } else {
        println!("  -> worth pipelining the engine.");
    }
    Ok(())
}
