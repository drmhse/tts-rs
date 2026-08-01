//! The upsample convs: does the GEMM route still win when the im2col matrix is huge?
//!
//! Routing CosyVoice's three `ups` convs through `causal_conv1d_gemm` improved a
//! 210-frame `decode` (181.4 -> 157.4 ms) but made the *end-to-end* vocoder slower
//! (RTF 0.132 -> 0.141). The fused-snake sweep rules that change out as the cause — it is
//! 1.4x to 3.2x faster at every shape — which leaves the convs.
//!
//! The suspicion is size. im2col expands the input by `k`, and `ups.0` has **k = 16**:
//!
//! | conv | cin | k | len (utterance) | im2col matrix |
//! |---|---|---|---|---|
//! | ups.0 | 512 | 16 | 21072 | **690 MB** |
//! | ups.1 | 256 | 11 | 105360 | 1.19 GB |
//! | ups.2 | 128 | 7 | 316080 | 1.13 GB |
//!
//! At 210 frames those are 55 MB / 95 MB / 90 MB — comfortable. At utterance length they
//! are not, and a 16-bit-per-input-element blowup stops being a good trade.
//!
//! So this compares three routes at both lengths: candle's direct conv, the GEMM route,
//! and a GEMM chunked along length so the matrix stays within a fixed budget.
//!
//! Run: `cargo run -p tts-probe --release --bin upsconv`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_bench::Harness;
use tts_nn::{causal_conv1d, causal_conv1d_gemm, max_abs_diff, tap_major_weight};

/// `(cin, cout, k, len_segment, len_utterance, label)` for CosyVoice's three upsamples.
const UPS: &[(usize, usize, usize, usize, usize, &str)] = &[
    (512, 256, 16, 3008, 21072, "ups.0 k16"),
    (256, 128, 11, 15040, 105360, "ups.1 k11"),
    (128, 64, 7, 45120, 316080, "ups.2 k7"),
];

/// Chunk budget in im2col elements — 32 M floats is 128 MB, which comfortably fits.
const BUDGET: usize = 32 << 20;

/// The GEMM route, but sliced along length so the im2col matrix stays bounded.
fn chunked(
    x: &Tensor,
    w_tap: &Tensor,
    b: &Tensor,
    k: usize,
    budget: usize,
) -> candle_core::Result<Tensor> {
    let (_, cin, len) = x.dims3()?;
    // Each output column costs `k * cin` elements of im2col.
    let cols = (budget / (k * cin)).max(1);
    if cols >= len {
        return causal_conv1d_gemm(x, w_tap, Some(b), k, 1).map_err(candle_core::Error::wrap);
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < len {
        let width = cols.min(len - start);
        // Each chunk needs `k - 1` samples of left context to stay causal.
        let ctx = (k - 1).min(start);
        let piece = x.narrow(2, start - ctx, ctx + width)?.contiguous()?;
        let y =
            causal_conv1d_gemm(&piece, w_tap, Some(b), k, 1).map_err(candle_core::Error::wrap)?;
        out.push(y.narrow(2, ctx, width)?);
        start += width;
    }
    Tensor::cat(&out, 2)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    println!(
        "{:>12} {:>9} {:>9}   {:>8} {:>8} {:>8}   {:>18}",
        "conv", "len", "im2col", "direct", "gemm", "chunked", "best"
    );
    for &(cin, cout, k, seg, utt, label) in UPS {
        for (len, tag) in [(seg, "seg"), (utt, "utt")] {
            let x = Tensor::randn(0f32, 1., (1, cin, len), &dev)?;
            let w = Tensor::randn(0f32, 1., (cout, cin, k), &dev)?;
            let b = Tensor::randn(0f32, 1., cout, &dev)?;
            let w_tap = tap_major_weight(&w)?;

            // All three routes must agree before any of them is timed.
            let want = causal_conv1d(&x, &w, Some(&b), 1)?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()? as f64;
            for (name, got) in [
                ("gemm", causal_conv1d_gemm(&x, &w_tap, Some(&b), k, 1)?),
                ("chunked", chunked(&x, &w_tap, &b, k, BUDGET)?),
            ] {
                let d = max_abs_diff(&want, &got)? as f64 / scale;
                anyhow::ensure!(d < 1e-5, "{label} {tag}: {name} differs, rel {d:.2e}");
            }

            let mut t = [0f64; 3];
            {
                let (x, w, b, w_tap) = (&x, &w, &b, &w_tap);
                let mut direct = || -> candle_core::Result<()> {
                    causal_conv1d(x, w, Some(b), 1).unwrap();
                    Ok(())
                };
                let mut gemm = || -> candle_core::Result<()> {
                    causal_conv1d_gemm(x, w_tap, Some(b), k, 1).unwrap();
                    Ok(())
                };
                let mut chunk = || -> candle_core::Result<()> {
                    chunked(x, w_tap, b, k, BUDGET)?;
                    Ok(())
                };
                let stats = h.ab(
                    &format!("{label} {tag}"),
                    &mut [
                        ("direct", &mut direct),
                        ("gemm", &mut gemm),
                        ("chunked", &mut chunk),
                    ],
                )?;
                for (i, s) in stats.iter().enumerate() {
                    t[i] = s.median;
                }
            }
            let mb = (k * cin * len * 4) as f64 / (1 << 20) as f64;
            let best = ["direct", "gemm", "chunked"][t
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0];
            println!(
                "{:>12} {len:>9} {:>7.0} MB   {:>8.2} {:>8.2} {:>8.2}   {:>18}",
                format!("{label} {tag}"),
                mb,
                t[0],
                t[1],
                t[2],
                format!(
                    "{best} ({:.2}x direct)",
                    t[0] / t.iter().cloned().fold(f64::MAX, f64::min)
                )
            );
        }
    }

    h.report_drift()?;
    Ok(())
}
