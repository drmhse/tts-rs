//! Strided views into `sdpa`, or a fast transpose first?
//!
//! `flow.rs` deliberately passes non-contiguous `[b, n, h, d] -> [b, h, n, d]` views to
//! `sdpa`, with a comment that calling `contiguous()` first measured **2.7x slower**. That
//! was measured against candle's own transpose, which moves this shape at ~4.6 GB/s.
//!
//! `tts_nn::fused::head_transpose` does the same movement in one coalesced pass. If it is
//! anywhere near bandwidth, the trade-off that justified strided views may invert — a
//! strided `sdpa` cannot vectorise its loads, and at the engine's real length (3192 frames,
//! not the 798 the original decision was measured at) attention is 34% of the flow.
//!
//! Both variants reproduce the real code exactly: the head permutation puts the rotated
//! head last, so it is `narrow(1, 0, 15)` plus a separate single-head call.
//!
//! Run: `cargo run -p a8-probe --release --bin attnlayout`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_bench::Harness;
use tts_nn::fused::head_transpose;
use tts_nn::{matmul_2d, max_abs_diff};

const B: usize = 2;
const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;

/// The current path: reshape and transpose as a lazy view, then two `sdpa` calls.
fn strided(q: &Tensor, k: &Tensor, v: &Tensor, n: usize, sc: f32) -> candle_core::Result<Tensor> {
    let h = |t: &Tensor| -> candle_core::Result<Tensor> {
        t.reshape((B, n, HEADS, HD))?.transpose(1, 2)
    };
    let (qh, kh, vh) = (h(q)?, h(k)?, h(v)?);
    let plain = HEADS - 1;
    let head = |t: &Tensor| t.narrow(1, 0, plain);
    let o_plain =
        candle_nn::ops::sdpa(&head(&qh)?, &head(&kh)?, &head(&vh)?, None, false, sc, 1.0)?;
    let last = |t: &Tensor| -> candle_core::Result<Tensor> { t.narrow(1, plain, 1)?.contiguous() };
    let o_rot = candle_nn::ops::sdpa(&last(&qh)?, &last(&kh)?, &last(&vh)?, None, false, sc, 1.0)?;
    Tensor::cat(&[o_plain, o_rot], 1)
}

/// The same, but each of q/k/v made contiguous by the custom transpose kernel first.
fn contig(q: &Tensor, k: &Tensor, v: &Tensor, _n: usize, sc: f32) -> candle_core::Result<Tensor> {
    let (qh, kh, vh) = (
        head_transpose(q, HEADS, HD)?,
        head_transpose(k, HEADS, HD)?,
        head_transpose(v, HEADS, HD)?,
    );
    let plain = HEADS - 1;
    let head = |t: &Tensor| t.narrow(1, 0, plain);
    let o_plain =
        candle_nn::ops::sdpa(&head(&qh)?, &head(&kh)?, &head(&vh)?, None, false, sc, 1.0)?;
    let last = |t: &Tensor| -> candle_core::Result<Tensor> { t.narrow(1, plain, 1)?.contiguous() };
    let o_rot = candle_nn::ops::sdpa(&last(&qh)?, &last(&kh)?, &last(&vh)?, None, false, sc, 1.0)?;
    Tensor::cat(&[o_plain, o_rot], 1)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    for n in [798usize, 3192] {
        let x = Tensor::randn(0f32, 1., (B, n, D), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (D, D), &dev)?;
        let q = matmul_2d(&x, &w)?;
        let k = matmul_2d(&x, &w)?;
        let v = matmul_2d(&x, &w)?;
        let sc = 1.0 / (HD as f32).sqrt();

        // The transpose must be exact before its speed means anything.
        let want = q
            .reshape((B, n, HEADS, HD))?
            .transpose(1, 2)?
            .contiguous()?;
        let got = head_transpose(&q, HEADS, HD)?;
        anyhow::ensure!(want.dims() == got.dims(), "shape mismatch");
        anyhow::ensure!(max_abs_diff(&want, &got)? == 0.0, "head_transpose differs");

        let a = strided(&q, &k, &v, n, sc)?;
        let b = contig(&q, &k, &v, n, sc)?;
        let rel = max_abs_diff(&a, &b)? as f64 / a.abs()?.max_all()?.to_scalar::<f32>()? as f64;
        println!("n={n}: strided vs contiguous attention rel {rel:.2e}");
        anyhow::ensure!(rel < 1e-5, "attention paths disagree");

        let (q1, k1, v1) = (&q, &k, &v);
        let mut f_strided = || -> candle_core::Result<()> {
            strided(q1, k1, v1, n, sc)?;
            Ok(())
        };
        let mut f_contig = || -> candle_core::Result<()> {
            contig(q1, k1, v1, n, sc)?;
            Ok(())
        };
        let mut f_candle_tr = || -> candle_core::Result<()> {
            q1.reshape((B, n, HEADS, HD))?
                .transpose(1, 2)?
                .contiguous()?;
            Ok(())
        };
        let mut f_kernel_tr = || -> candle_core::Result<()> {
            head_transpose(q1, HEADS, HD)?;
            Ok(())
        };
        let stats = h.ab(
            &format!("attention @ n={n}"),
            &mut [
                ("attention, strided views", &mut f_strided),
                ("attention, kernel transpose", &mut f_contig),
                ("transpose: candle", &mut f_candle_tr),
                ("transpose: kernel", &mut f_kernel_tr),
            ],
        )?;
        let g = |i: usize| stats[i].median;
        let bytes = 2.0 * (B * n * D) as f64 * 4.0;
        println!(
            "  attention {:.2} -> {:.2} ms  ({:.2}x)   transpose {:.2} -> {:.2} ms  ({:.2}x, {:.0} GB/s)\n",
            g(0),
            g(1),
            g(0) / g(1),
            g(2),
            g(3),
            g(2) / g(3),
            bytes / (g(3) * 1e6)
        );
    }

    h.report_drift()?;
    Ok(())
}
