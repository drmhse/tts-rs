//! Where the codec's time goes, stage by stage, synchronised.
//!
//! The codec was 21% of RTF and is now 26% of a much smaller total, so it is the larger
//! remaining share on long jobs. This says which stage to attack rather than guessing — the
//! convs, the transposed convs, the pre-transformer and the RVQ are all plausible and only one
//! of them can be the answer.
//!
//! Run: `cargo run -p qwen3tts --release --bin qwen3tts-codecsplit [frames]`

use anyhow::{Context, Result};
use candle_core::Device;
use qwen3tts::cfg::codec as k;
use qwen3tts::codec::Codec;
use std::time::Instant;

const WEIGHTS: &str = "references/qwen3tts/weights/speech_tokenizer/model.safetensors";

fn main() -> Result<()> {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(k::CHUNK_FRAMES);
    let device = Device::new_metal(0).context("opening the Metal device")?;
    let codec = Codec::load(WEIGHTS, &device)?;

    // Codes in range, deterministic, and varied — a constant frame would let the RVQ's
    // index_select hit one cache line and flatter it.
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) % k::CODEBOOK as u64) as u32
    };
    let all: Vec<Vec<u32>> = (0..frames)
        .map(|_| (0..k::QUANTIZERS).map(|_| next()).collect())
        .collect();

    // Warm up: first call per shape compiles pipelines and grows the buffer pool.
    codec.decode(&all)?;
    device.synchronize()?;

    let reps = 3;
    let t = Instant::now();
    for _ in 0..reps {
        codec.decode(&all)?;
    }
    device.synchronize()?;
    let whole = t.elapsed().as_secs_f64() / reps as f64;

    let audio = frames as f64 / qwen3tts::cfg::FRAME_RATE;
    println!(
        "{frames} frames ({audio:.2} s of audio): {:.1} ms/decode, RTF {:.4}",
        whole * 1000.0,
        whole / audio
    );

    println!("\nper stage, each timer synchronised at both ends:");
    let stages = codec.profile(&all, reps)?;
    let total: f64 = stages.iter().map(|(_, s)| *s).sum();
    for (name, secs) in &stages {
        println!(
            "  {name:<22} {:>7.1} ms  {:>5.1}%",
            secs * 1000.0,
            secs / total * 100.0
        );
    }
    println!("  {:<22} {:>7.1} ms", "sum", total * 1000.0);
    Ok(())
}
