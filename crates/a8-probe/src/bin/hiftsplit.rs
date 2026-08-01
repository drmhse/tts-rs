//! Where CosyVoice's vocoder actually spends its time.
//!
//! The custom im2col gather took Audio8's codec 1.42x (0.260 -> 0.183 RTF) but moved
//! CosyVoice's vocoder not at all (0.132 -> 0.134). Both are conv stacks, so either the
//! vocoder's convs are not on the GEMM route or they are not what it spends its time on.
//! Guessing between those is how the 3.5x kernel projection got retracted; this measures.
//!
//! `Hift::forward` is three parts: the F0 predictor at frame rate, the harmonic source at
//! full sample rate, and the decoder (upsamples, ResBlocks, iSTFT).
//!
//! Run: `cargo run -p a8-probe --release --bin hiftsplit`

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cosy::hift::{Hift, Noise};
use std::collections::HashMap;
use tts_bench::Harness;
use tts_core::rng::Rng;

const FIXTURES: &str = "fixtures-cosy/oracle.safetensors";
const WEIGHTS: &str = "oracle-cosy/weights";

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &dev)?;
    let hift = Hift::load(&format!("{WEIGHTS}/hift.safetensors"), &dev)?;

    // A whole-utterance mel, the shape the fused flow path now produces.
    let mel = fx
        .get("hift.mel_in")
        .ok_or_else(|| anyhow::anyhow!("missing hift.mel"))?
        .to_dtype(DType::F32)?;
    let frames = mel.dim(2)?;
    println!(
        "mel {:?}  ->  {} samples\n",
        mel.dims(),
        frames * hift.samples_per_frame()
    );

    // The engine's fused-flow path calls the vocoder once for the whole utterance, so
    // the fixture's 210 frames are not the shape that matters. Tile it up to the real
    // length: every conv here is length-invariant, so a tiled mel exercises the same
    // kernels at the same sizes the engine actually hits.
    let utt = Tensor::cat(&[&mel; 13], 2)?.contiguous()?;
    println!("tiled mel {:?}  ({} frames)\n", utt.dims(), utt.dim(2)?);

    let f0 = hift.predict_f0(&mel)?;
    let src = hift.source(&f0, Noise::Silent)?;
    let f0u = hift.predict_f0(&utt)?;
    let srcu = hift.source(&f0u, Noise::Silent)?;

    let mut whole = || -> candle_core::Result<()> {
        hift.forward(&mel, Noise::Silent).unwrap();
        Ok(())
    };
    let mut f0_only = || -> candle_core::Result<()> {
        hift.predict_f0(&mel).unwrap();
        Ok(())
    };
    let mut source_only = || -> candle_core::Result<()> {
        hift.source(&f0, Noise::Silent).unwrap();
        Ok(())
    };
    let mut decode_only = || -> candle_core::Result<()> {
        hift.decode(&mel, &src).unwrap();
        Ok(())
    };

    let mut decode_utt = || -> candle_core::Result<()> {
        hift.decode(&utt, &srcu).unwrap();
        Ok(())
    };
    let mut forward_utt = || -> candle_core::Result<()> {
        hift.forward(&utt, Noise::Silent).unwrap();
        Ok(())
    };

    // The engine synthesises with `Noise::Draw`, which the stage split above never
    // exercised — it used `Silent` throughout. `Draw` fills 9 * samples uniforms on the
    // host and uploads them, so it has to be timed separately or it hides inside nothing.
    let (mut rng_a, mut rng_b) = (Rng::new(0), Rng::new(1));
    let mut forward_draw = || -> candle_core::Result<()> {
        hift.forward(&utt, Noise::Draw(&mut rng_a)).unwrap();
        Ok(())
    };
    let mut source_draw = || -> candle_core::Result<()> {
        hift.source(&f0u, Noise::Draw(&mut rng_b)).unwrap();
        Ok(())
    };
    let mut source_silent = || -> candle_core::Result<()> {
        hift.source(&f0u, Noise::Silent).unwrap();
        Ok(())
    };

    let ustats = h.ab(
        "utterance-length",
        &mut [
            ("decode_utt", &mut decode_utt),
            ("forward_utt", &mut forward_utt),
            ("forward_draw", &mut forward_draw),
            ("source_draw", &mut source_draw),
            ("source_silent", &mut source_silent),
        ],
    )?;
    let frames_u = utt.dim(2)?;
    for s in &ustats {
        let secs = (frames_u * hift.samples_per_frame()) as f64 / 24000.0;
        println!(
            "  {:<12} {:>9.1} ms   RTF {:.4}",
            s.name,
            s.median,
            s.median / 1000.0 / secs
        );
    }

    let stats = h.ab(
        "hift stages",
        &mut [
            ("forward", &mut whole),
            ("predict_f0", &mut f0_only),
            ("source", &mut source_only),
            ("decode", &mut decode_only),
        ],
    )?;
    let total = stats[0].median;
    println!("\n{:>12}  {:>9}  {:>7}", "stage", "median ms", "share");
    for s in &stats[1..] {
        println!(
            "{:>12}  {:>9.2}  {:>6.1}%",
            s.name,
            s.median,
            100.0 * s.median / total
        );
    }
    println!("{:>12}  {:>9.2}  {:>6.1}%", "forward", total, 100.0);

    h.report_drift()?;
    Ok(())
}
