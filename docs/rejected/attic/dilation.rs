//! Is the dilated access pattern itself the problem, and can it be removed exactly?
//!
//! Every conv measurement so far used dilation 9. A dilation-d conv reads with
//! stride d, which is hostile to coalescing. But a dilated conv is *exactly* a
//! dense conv on d interleaved subsequences:
//!
//!   out[co, t] = sum_{c,kk} w[co,c,kk] * xpad[c, t + kk*d]
//!   write t = q*d + r  =>  t + kk*d = (q+kk)*d + r
//!   so with xpad_r[j] = xpad[r + j*d]:
//!       out[co, q*d + r] = sum_{c,kk} w[co,c,kk] * xpad_r[c, q+kk]
//!
//! which is a dense (dilation-1) k-tap conv on each subsequence r, and the d
//! subsequences can ride the batch dimension. Reshape + permute + one batched
//! dense conv, then interleave back. No approximation.
//!
//! Run:  cargo run -p a8-probe --release --bin dilation

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

const ITERS: usize = 3;

fn conv_direct(x: &Tensor, w: &Tensor, k: usize, d: usize) -> candle_core::Result<Tensor> {
    x.pad_with_zeros(2, (k - 1) * d, 0)?.conv1d(w, 0, 1, d, 1)
}

/// Dilated conv as a batched dense conv over d de-interleaved subsequences.
fn conv_deinterleave(x: &Tensor, w: &Tensor, k: usize, d: usize) -> candle_core::Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    let cout = w.dim(0)?;
    if d == 1 {
        return conv_direct(x, w, k, 1);
    }
    // Pad left by (k-1)*d for causality, right up to a multiple of d so the
    // de-interleave divides evenly.
    let lp = len.div_ceil(d) * d;
    let xpad = x.pad_with_zeros(2, (k - 1) * d, 0)?.pad_with_zeros(2, 0, lp - len)?;
    let n = lp / d + (k - 1); // subsequence length, total = d*n
    debug_assert_eq!(xpad.dim(2)?, d * n);

    // [1,C,d*n] -> [C,n,d] (index = j*d + r) -> [d,C,n]
    let sub = xpad
        .squeeze(0)?
        .reshape((c, n, d))?
        .permute((2, 0, 1))?
        .contiguous()?;
    // dense conv, dilation 1, no padding: [d, Cout, n-(k-1)] = [d, Cout, lp/d]
    let out = sub.conv1d(w, 0, 1, 1, 1)?;
    // interleave back: [d,Cout,lp/d] -> [Cout,lp/d,d] -> [Cout,lp] -> trim
    out.permute((1, 2, 0))?
        .contiguous()?
        .reshape((cout, lp))?
        .narrow(1, 0, len)?
        .unsqueeze(0)
}

fn time<F>(dev: &Device, f: F) -> Result<f64>
where
    F: Fn() -> candle_core::Result<Tensor>,
{
    let warm = f()?;
    dev.synchronize()?;
    drop(warm);
    let start = Instant::now();
    for _ in 0..ITERS {
        let out = f()?;
        dev.synchronize()?;
        drop(out);
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64)
}

fn gflops(c: usize, k: usize, len: usize, ms: f64) -> f64 {
    2.0 * (c * c * k * len) as f64 / (ms / 1000.0) / 1e9
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;

    // ------------------------------------------------------------ correctness
    for (c, len, d) in [(8usize, 64usize, 3usize), (8, 64, 9), (16, 129, 9)] {
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let w = Tensor::randn(0f32, 0.3, (c, c, 7), &dev)?;
        let reference = conv_direct(&x, &w, 7, d)?;
        let got = conv_deinterleave(&x, &w, 7, d)?;
        let diff = (&reference - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = reference.abs()?.max_all()?.to_scalar::<f32>()?;
        println!(
            "correctness C={c} L={len} d={d}: shape {:?} vs {:?}  max|diff| {diff:.3e} (scale {scale:.3e}) {}",
            got.dims(), reference.dims(),
            if diff < 1e-3 { "OK" } else { "MISMATCH" }
        );
    }

    // ------------------------------------- is dilation itself costing anything?
    println!("\ndilation sensitivity, candle conv1d direct (k=7):");
    println!("{:<22} {:>9} {:>9} {:>9}   {:>12}", "shape", "d=1", "d=3", "d=9", "d9/d1");
    println!("{}", "-".repeat(68));
    for (c, len) in [(768usize, 2048usize), (384, 16384), (192, 65536), (96, 131072)] {
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (c, c, 7), &dev)?;
        let t1 = time(&dev, || conv_direct(&x, &w, 7, 1))?;
        let t3 = time(&dev, || conv_direct(&x, &w, 7, 3))?;
        let t9 = time(&dev, || conv_direct(&x, &w, 7, 9))?;
        println!(
            "{:<22} {t1:>9.2} {t3:>9.2} {t9:>9.2}   {:>11.2}x",
            format!("{c}ch @ {len}"), t9 / t1
        );
    }

    // -------------------------------- de-interleave vs direct, at the real shapes
    println!("\nde-interleaved dense conv vs direct dilated conv:");
    println!("{:<22} {:>4} {:>10} {:>12} {:>9}   {:>9}",
             "shape", "d", "direct ms", "deinterl ms", "speedup", "GFLOPS");
    println!("{}", "-".repeat(76));
    let mut sum_direct = 0.0;
    let mut sum_best = 0.0;
    for (c, len) in [(768usize, 2048usize), (384, 16384), (192, 65536), (96, 131072)] {
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (c, c, 7), &dev)?;
        // The three residual units in a block use dilations 1, 3, 9.
        for d in [3usize, 9] {
            let td = time(&dev, || conv_direct(&x, &w, 7, d))?;
            let tn = time(&dev, || conv_deinterleave(&x, &w, 7, d))?;
            println!(
                "{:<22} {d:>4} {td:>10.2} {tn:>12.2} {:>8.2}x   {:>9.0}",
                format!("{c}ch @ {len}"), td / tn, gflops(c, 7, len, tn)
            );
            sum_direct += td;
            sum_best += td.min(tn);
        }
    }
    println!("{}", "-".repeat(76));
    println!("dilated convs (d=3,9) across all stages: {sum_direct:.1} ms -> {sum_best:.1} ms ({:.2}x)",
             sum_direct / sum_best);
    println!("\n(each block has 3 residual units at dilations 1, 3, 9; only d=3 and d=9");
    println!(" can benefit, so the whole-decoder win is roughly 2/3 of the above)");

    Ok(())
}
