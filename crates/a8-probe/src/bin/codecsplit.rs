//! Where the Audio8 codec's time goes, now that its causal convs are GEMMs.
//!
//! The codec is 43% of this engine's runtime and `causal_conv1d_gemm` touched only part of
//! it, so this says what is left worth attacking. Timed per stage with a device sync between
//! each, which inflates the total slightly against a single fused run — the shares are what
//! matter, not the sum.
//!
//! Run: `cargo run -p a8-probe --release --bin codecsplit`

use a8::codec::Codec;
use anyhow::Result;
use candle_core::Device;
use tts_bench::Harness;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    let codec = Codec::load("oracle/weights/codec.safetensors", &dev)?;

    // 150 frames is about one segment's worth at 21.53 Hz.
    let frames = 150;
    let codes: Vec<Vec<u32>> = (0..10)
        .map(|i| vec![(i * 37 + 11) as u32; frames])
        .collect();

    // Warm up, then take the median of several passes per stage.
    let _ = codec.bench_stages(&codes)?;
    let mut runs: Vec<Vec<(&'static str, f64)>> = Vec::new();
    for _ in 0..5 {
        runs.push(codec.bench_stages(&codes)?);
    }
    let names: Vec<&'static str> = runs[0].iter().map(|(n, _)| *n).collect();
    let mut totals = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let mut v: Vec<f64> = runs.iter().map(|r| r[i].1).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        totals.push((*name, v[v.len() / 2]));
    }
    let sum: f64 = totals.iter().map(|(_, t)| t).sum();
    println!("\ncodec decode, {frames} frames — per stage (median of 5)\n");
    println!("{:<26} {:>10} {:>8}", "stage", "ms", "share");
    println!("{}", "-".repeat(46));
    for (name, ms) in &totals {
        println!("{name:<26} {ms:>10.2} {:>7.1}%", 100.0 * ms / sum);
    }
    println!("{:<26} {sum:>10.2}", "total (with syncs)");

    h.report_drift()?;
    Ok(())
}
