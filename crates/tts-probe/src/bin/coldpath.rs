//! Why CosyVoice's vocoder costs 7.6 s in the engine and 3.45 s in a warm A/B loop.
//!
//! The kernels in `tts_nn::{im2col, fused}` are worth 1.4x to 6x wherever they are timed,
//! and Audio8's codec took the full win end-to-end (RTF 0.260 -> 0.158). CosyVoice's
//! vocoder took none. The difference is that Audio8's codec runs seven times per utterance
//! and CosyVoice's fused-flow path calls `hift.forward` exactly once.
//!
//! The standing hypothesis is candle's Metal buffer pool: `new_buffer` reuses a pooled
//! allocation of the right size class when one is free, so a warm loop allocates once and
//! recycles, while a single cold call pays first-touch for every buffer. But there is a
//! second candidate that the warm loop equally hides — the flow runs immediately before
//! and leaves gigabytes of its own buffers in the pool, so the vocoder may be allocating
//! under memory pressure rather than merely allocating.
//!
//! Those two have different fixes, so this separates them. Each mode runs in its own
//! process, because the pool is process-wide and that is the whole point.
//!
//! Run: `cargo run -p tts-probe --release --bin coldpath -- <cold|warm|afterflow|prewarm>`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice::flow::Flow;
use cosyvoice::hift::{Hift, Noise};
use std::collections::HashMap;
use std::time::Instant;
use tts_core::rng::Rng;

const FIXTURES: &str = "fixtures/cosyvoice/oracle.safetensors";
const NOISE_ASSET: &str = "fixtures/cosyvoice/rand_noise.safetensors";
const WEIGHTS: &str = "references/cosyvoice/weights";
/// Tiles the 210-frame fixture up to roughly the engine's 2634.
const TILES: usize = 13;

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "cold".into());
    let dev = Device::new_metal(0)?;

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &dev)
        .with_context(|| format!("loading {FIXTURES}"))?;
    let get = |n: &str| -> Result<Tensor> {
        Ok(fx
            .get(n)
            .with_context(|| format!("missing {n}"))?
            .to_dtype(DType::F32)?)
    };

    let hift = Hift::load(&format!("{WEIGHTS}/hift.safetensors"), &dev)?;
    let mel = get("hift.mel_in")?;
    let utt = Tensor::cat(&[&mel; TILES], 2)?.contiguous()?;
    let frames = utt.dim(2)?;
    let audio_s = (frames * hift.samples_per_frame()) as f64 / 24000.0;

    let mut rng = Rng::new(0);
    let run = |label: &str, hift: &Hift, x: &Tensor, rng: &mut Rng| -> Result<f64> {
        let t = Instant::now();
        let w = hift.forward(x, Noise::Draw(rng))?;
        // Touch the result so nothing is deferred past the timer.
        let _ = w.dims();
        w.device().synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {label:<28} {ms:>9.1} ms   RTF {:.4}",
            ms / 1000.0 / audio_s
        );
        Ok(ms)
    };

    println!("mode {mode}, {frames} frames ({audio_s:.1} s of audio)");
    match mode.as_str() {
        // The number the engine actually experiences.
        "cold" => {
            run("first call (cold)", &hift, &utt, &mut rng)?;
            run("second call", &hift, &utt, &mut rng)?;
            run("third call", &hift, &utt, &mut rng)?;
        }
        // What the A/B harness reports, for reference.
        "warm" => {
            for _ in 0..3 {
                hift.forward(&utt, Noise::Draw(&mut rng))?;
            }
            run("warm", &hift, &utt, &mut rng)?;
        }
        // The engine's real sequence: a full flow solve, then the vocoder once.
        "afterflow" => {
            let flow = Flow::load(&format!("{WEIGHTS}/flow.safetensors"), NOISE_ASSET, &dev)?;
            let (mu, cond, spks) = (get("flow.mu")?, get("flow.cond")?, get("flow.spks")?);
            let t = Instant::now();
            let out = flow.solve(&mu, &cond, &spks)?;
            dev.synchronize()?;
            println!(
                "  flow solve ({} frames)      {:>9.1} ms",
                out.dim(2)?,
                t.elapsed().as_secs_f64() * 1000.0
            );
            run("vocoder after flow", &hift, &utt, &mut rng)?;
            run("vocoder again", &hift, &utt, &mut rng)?;
        }
        // Can the cost be paid up front? One throwaway pass at the same shape should
        // populate the pool with exactly the size classes the real call needs.
        "prewarm" => {
            let t = Instant::now();
            hift.forward(&utt, Noise::Silent)?;
            dev.synchronize()?;
            println!(
                "  prewarm pass                {:>9.1} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
            run("real call after prewarm", &hift, &utt, &mut rng)?;
        }
        other => anyhow::bail!("unknown mode {other}"),
    }
    Ok(())
}
