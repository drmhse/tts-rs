//! Can a custom Metal gather beat the `cat`-built im2col matrix?
//!
//! `codecsplit` put **79% of the conv route in the gather** — 28.6 ms against a 7.1 ms GEMM
//! at `96ch @ 131072, k=7`, about 24 GB/s on a ~120 GB/s bus. `index_select` does the same
//! gather at ~81 GB/s but needs a 352 MB index, so the hardware is not the limit.
//!
//! Two candidates, in the order they were tried:
//!
//! 1. **candle's own `call_im2col1d_strided`**, private to `conv1d`. Measured **0.66x** —
//!    slower than `cat`. Its kernel recovers four indices from a linear thread id with
//!    three `size_t` divisions per element, and at 88 M elements that arithmetic dominates.
//! 2. **A 3-D dispatch grid** with no division at all, writing tap-major so the GEMM needs
//!    no transpose and the causal pad folds into the gather. That is what this measures.
//!
//! Correctness is checked against the `cat` matrix before anything is timed — this port has
//! already been burned once by benchmarking an incorrect sdpa split.
//!
//! Run: `cargo run -p tts-probe --release --bin im2col`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_bench::Harness;
use tts_nn::im2col::im2col_tap_major;
use tts_nn::{causal_conv1d_gemm, max_abs_diff, tap_major_weight};

/// The shapes the Audio8 codec decoder actually runs, plus the canary shape for scale.
const SHAPES: &[(usize, usize, usize, usize, usize)] = &[
    // (cin, cout, len, k, dilation)
    (1024, 1024, 2048, 7, 1),
    (512, 512, 4096, 7, 1),
    (256, 256, 16384, 7, 1),
    (128, 128, 65536, 7, 1),
    (96, 96, 131072, 7, 1),
    (96, 96, 131072, 7, 9),
    (96, 1, 131072, 7, 1),
];

/// What the `cat` route builds, for both the correctness check and the A/B baseline.
fn cat_matrix(x: &Tensor, k: usize, dil: usize) -> candle_core::Result<Tensor> {
    let (_, cin, len) = x.dims3()?;
    let xpad = x.pad_with_zeros(2, (k - 1) * dil, 0)?;
    let taps: Vec<_> = (0..k)
        .map(|t| xpad.narrow(2, t * dil, len))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Tensor::cat(&taps, 0)?.reshape((k * cin, len))
}

/// The whole conv through the custom gather.
fn conv_via_kernel(
    x: &Tensor,
    w_tap: &Tensor,
    k: usize,
    dil: usize,
) -> candle_core::Result<Tensor> {
    let (_, _, len) = x.dims3()?;
    let out = w_tap.dim(0)?;
    let cols = im2col_tap_major(x, k, dil)?;
    w_tap.matmul(&cols)?.reshape((1, out, len))
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    println!("correctness on device, vs the cat-built matrix");
    for &(cin, cout, len, k, dil) in SHAPES {
        let len = len.min(4096);
        let x = Tensor::randn(0f32, 1., (1, cin, len), &dev)?;
        let w = Tensor::randn(0f32, 1., (cout, cin, k), &dev)?;
        let w_tap = tap_major_weight(&w)?;

        // The gather must be bit-identical: it is the same values in the same order.
        let gather_d = max_abs_diff(&cat_matrix(&x, k, dil)?, &im2col_tap_major(&x, k, dil)?)?;
        // The conv may differ in the last bits only through GEMM input ordering, which is
        // unchanged here — so this should also be exact.
        let want = causal_conv1d_gemm(&x, &w_tap, None, k, dil)?;
        let got = conv_via_kernel(&x, &w_tap, k, dil)?;
        let conv_d = max_abs_diff(&want, &got)? as f64;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()? as f64;
        println!(
            "  {cin:>4}->{cout:<4} k{k} d{dil}  gather {gather_d:.1e}  conv rel {:.2e}",
            conv_d / scale
        );
        anyhow::ensure!(gather_d == 0.0, "gather differs at {cin}x{len} d{dil}");
        anyhow::ensure!(conv_d / scale < 1e-6, "conv differs at {cin}x{len} d{dil}");
    }

    println!("\nper-shape timings (ms)");
    let mut rows = Vec::new();
    for &(cin, cout, len, k, dil) in SHAPES {
        let x = Tensor::randn(0f32, 1., (1, cin, len), &dev)?;
        let w = Tensor::randn(0f32, 1., (cout, cin, k), &dev)?;
        let w_tap = tap_major_weight(&w)?;

        let mut t = [0f64; 4];
        {
            let (x, w_tap) = (&x, &w_tap);
            let mut cat_gather = || -> candle_core::Result<()> {
                cat_matrix(x, k, dil)?;
                Ok(())
            };
            let mut ker_gather = || -> candle_core::Result<()> {
                im2col_tap_major(x, k, dil)?;
                Ok(())
            };
            let mut cat_full = || -> candle_core::Result<()> {
                causal_conv1d_gemm(x, w_tap, None, k, dil).unwrap();
                Ok(())
            };
            let mut ker_full = || -> candle_core::Result<()> {
                conv_via_kernel(x, w_tap, k, dil)?;
                Ok(())
            };
            let stats = h.ab(
                &format!("{cin}x{len} d{dil}"),
                &mut [
                    ("cat_gather", &mut cat_gather),
                    ("ker_gather", &mut ker_gather),
                    ("cat_full", &mut cat_full),
                    ("ker_full", &mut ker_full),
                ],
            )?;
            for (i, s) in stats.iter().enumerate() {
                t[i] = s.median;
            }
        }
        // The gather moves k*cin*len writes plus about as many reads.
        let bytes = 2.0 * (k * cin * len) as f64 * 4.0;
        rows.push((
            format!("{cin}->{cout} @{len} d{dil}"),
            t,
            bytes / (t[1] * 1e6),
        ));
    }

    println!(
        "\n{:>22}  {:>8} {:>8} {:>7} {:>8}   {:>8} {:>8} {:>7}",
        "shape", "cat gath", "ker gath", "speedup", "GB/s", "cat conv", "ker conv", "speedup"
    );
    for (name, t, gbs) in &rows {
        println!(
            "{name:>22}  {:>8.2} {:>8.2} {:>6.2}x {:>8.1}   {:>8.2} {:>8.2} {:>6.2}x",
            t[0],
            t[1],
            t[0] / t[1],
            gbs,
            t[2],
            t[3],
            t[2] / t[3]
        );
    }

    h.report_drift()?;
    Ok(())
}
