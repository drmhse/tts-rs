//! Fused snake against the composed form, at the shapes both vocoders actually run.
//!
//! `hiftsplit` says the fused kernels take CosyVoice's `decode` from 157.4 ms to 131.9 ms
//! on a 210-frame mel, and Audio8's codec agrees end-to-end (RTF 0.183 -> 0.156 with the
//! AR stage steady at 0.339 as a control). But CosyVoice's *end-to-end* vocoder went the
//! other way, 0.132 -> 0.141. One of those measurements is misleading and averaging them
//! would hide which.
//!
//! The obvious difference is length: the probe's mel is 210 frames, where a real segment
//! is ~376 and the fused utterance is 2634. So this sweeps length directly.
//!
//! Run: `cargo run -p tts-probe --release --bin snakefuse`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_bench::Harness;
use tts_nn::fused;

/// `(channels, length, label)` — the CosyVoice decoder's three upsample stages at both
/// per-segment and whole-utterance mel lengths, plus Audio8's codec tail.
const SHAPES: &[(usize, usize, &str)] = &[
    (256, 3008, "cosy up0 / segment"),
    (128, 15040, "cosy up1 / segment"),
    (64, 45120, "cosy up2 / segment"),
    (256, 21072, "cosy up0 / utterance"),
    (128, 105360, "cosy up1 / utterance"),
    (64, 316080, "cosy up2 / utterance"),
    (96, 131072, "a8 codec tail"),
    (1024, 2048, "a8 codec entry"),
];

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    // Correctness first, at one representative shape per kernel.
    let x = Tensor::randn(0f32, 1., (1, 96, 4096), &dev)?;
    let alpha = Tensor::randn(0f32, 1., 96, &dev)?;
    let want = (&x + x.sin()?.sqr()?)?;
    let d = tts_nn::max_abs_diff(&want, &fused::snake_folded(&x)?)?;
    println!("snake_folded vs composed: {d:.2e}");
    anyhow::ensure!(d < 1e-5, "snake_folded disagrees");
    let u = x.broadcast_mul(&alpha.reshape((1, 96, 1))?)?.contiguous()?;
    let want = (&u + u.sin()?.sqr()?)?;
    let d = tts_nn::max_abs_diff(&want, &fused::snake_alpha(&x, &alpha)?)?;
    println!("snake_alpha  vs composed: {d:.2e}\n");
    anyhow::ensure!(d < 1e-5, "snake_alpha disagrees");

    println!(
        "{:>22} {:>6} {:>8}   {:>8} {:>8} {:>7}   {:>8} {:>8} {:>7}",
        "shape", "ch", "len", "fold cmp", "fold fus", "speedup", "alph cmp", "alph fus", "speedup"
    );
    let mut rows = Vec::new();
    for &(ch, len, label) in SHAPES {
        let x = Tensor::randn(0f32, 1., (1, ch, len), &dev)?;
        let alpha = Tensor::randn(0f32, 1., ch, &dev)?;
        let a3 = alpha.reshape((1, ch, 1))?;

        let mut t = [0f64; 4];
        {
            let (x, alpha, a3) = (&x, &alpha, &a3);
            let mut fold_composed = || -> candle_core::Result<()> {
                (x + x.sin()?.sqr()?)?;
                Ok(())
            };
            let mut fold_fused = || -> candle_core::Result<()> {
                fused::snake_folded(x)?;
                Ok(())
            };
            let mut alpha_composed = || -> candle_core::Result<()> {
                let u = x.broadcast_mul(a3)?.contiguous()?;
                (&u + u.sin()?.sqr()?)?;
                Ok(())
            };
            let mut alpha_fused = || -> candle_core::Result<()> {
                fused::snake_alpha(x, alpha)?;
                Ok(())
            };
            let stats = h.ab(
                label,
                &mut [
                    ("fold_composed", &mut fold_composed),
                    ("fold_fused", &mut fold_fused),
                    ("alpha_composed", &mut alpha_composed),
                    ("alpha_fused", &mut alpha_fused),
                ],
            )?;
            for (i, s) in stats.iter().enumerate() {
                t[i] = s.median;
            }
        }
        rows.push((label, ch, len, t));
    }

    for (label, ch, len, t) in &rows {
        println!(
            "{label:>22} {ch:>6} {len:>8}   {:>8.2} {:>8.2} {:>6.2}x   {:>8.2} {:>8.2} {:>6.2}x",
            t[0],
            t[1],
            t[0] / t[1],
            t[2],
            t[3],
            t[2] / t[3]
        );
    }

    h.report_drift()?;
    Ok(())
}
