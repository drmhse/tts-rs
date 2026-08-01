//! Why is candle's conv1d 5.5x less efficient at 96 channels than at 768, for
//! identical FLOPs? And can we recover it by routing convolution through GEMM?
//!
//! Evidence from the first probe (same 8.46e9 MACs in both rows):
//!     residual_unit 768ch @ len 2048    17.83 ms   ~949 GFLOPS
//!     residual_unit  96ch @ len 131072  97.89 ms   ~173 GFLOPS
//! and the pathological case:
//!     final conv 96->1 k7 @ 131072      47.48 ms   ~3.7 GFLOPS
//!
//! Hypothesis: the Metal conv kernel parallelises over output channels and starves
//! when Cout is small. A GEMM has no notion of channels -- it sees M=L, K=C*k,
//! N=Cout -- so im2col + matmul should be insensitive to the channel count.
//!
//! Run:  cargo run -p tts-probe --release --bin convopt

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const ITERS: usize = 3;

/// Causal conv1d via candle's built-in kernel.
fn conv_direct(x: &Tensor, w: &Tensor, k: usize, d: usize) -> candle_core::Result<Tensor> {
    let pad = (k - 1) * d;
    x.pad_with_zeros(2, pad, 0)?.conv1d(w, 0, 1, d, 1)
}

/// Causal conv1d as im2col + one GEMM.
///
/// y[co,t] = sum_{c,kk} w[co,c,kk] * xpad[c, t + kk*d]
/// Build M[L, k*C] with M[t, kk*C + c] = xpad[c, t + kk*d], and reorder the weight
/// to Wr[k*C, Cout] = w.permute(2,1,0).reshape(k*C, Cout) so the contractions line up.
fn conv_im2col(x: &Tensor, w: &Tensor, k: usize, d: usize) -> candle_core::Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    let cout = w.dim(0)?;
    let pad = (k - 1) * d;
    let xpad = x.pad_with_zeros(2, pad, 0)?;

    let mut blocks = Vec::with_capacity(k);
    for kk in 0..k {
        // [1,C,L] -> [C,L] -> [L,C]
        blocks.push(xpad.narrow(2, kk * d, len)?.squeeze(0)?.t()?);
    }
    let m = Tensor::cat(&blocks, 1)?.contiguous()?; // [L, k*C]
    let wr = w.permute((2, 1, 0))?.contiguous()?.reshape((k * c, cout))?;
    m.matmul(&wr)?.t()?.contiguous()?.unsqueeze(0) // [1, Cout, L]
}

/// Causal conv1d as k accumulated GEMMs -- avoids materialising the [L, k*C] matrix.
fn conv_taps(x: &Tensor, w: &Tensor, k: usize, d: usize) -> candle_core::Result<Tensor> {
    let (_, _, len) = x.dims3()?;
    let pad = (k - 1) * d;
    let xpad = x.pad_with_zeros(2, pad, 0)?;
    let mut acc: Option<Tensor> = None;
    for kk in 0..k {
        let slice = xpad.narrow(2, kk * d, len)?.squeeze(0)?.t()?.contiguous()?; // [L,C]
        let wk = w.narrow(2, kk, 1)?.squeeze(2)?.t()?.contiguous()?; // [C,Cout]
        let term = slice.matmul(&wk)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    acc.unwrap().t()?.contiguous()?.unsqueeze(0)
}

/// im2col + GEMM, blocked along length. The decoder is strictly causal with a
/// bounded receptive field, so chunking is exact -- each chunk just needs
/// (k-1)*d samples of left context. torch processes the whole clip monolithically;
/// blocking is something a from-scratch implementation can do and a port cannot.
fn conv_chunked(
    x: &Tensor,
    w: &Tensor,
    k: usize,
    d: usize,
    chunk: usize,
) -> candle_core::Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    let cout = w.dim(0)?;
    let pad = (k - 1) * d;
    let xpad = x.pad_with_zeros(2, pad, 0)?;
    let wr = w.permute((2, 1, 0))?.contiguous()?.reshape((k * c, cout))?;

    let mut outputs = Vec::new();
    let mut start = 0usize;
    while start < len {
        let this = chunk.min(len - start);
        let mut blocks = Vec::with_capacity(k);
        for kk in 0..k {
            blocks.push(xpad.narrow(2, start + kk * d, this)?.squeeze(0)?.t()?);
        }
        let m = Tensor::cat(&blocks, 1)?.contiguous()?;
        outputs.push(m.matmul(&wr)?); // [this, Cout]
        start += this;
    }
    Tensor::cat(&outputs, 0)?.t()?.contiguous()?.unsqueeze(0)
}

fn time<F>(dev: &Device, f: F) -> Result<(f64, Tensor)>
where
    F: Fn() -> candle_core::Result<Tensor>,
{
    let warm = f()?;
    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERS {
        let out = f()?;
        dev.synchronize()?;
        drop(out);
    }
    Ok((start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64, warm))
}

fn gflops(cin: usize, cout: usize, k: usize, len: usize, ms: f64) -> f64 {
    2.0 * (cin * cout * k * len) as f64 / (ms / 1000.0) / 1e9
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let dt = DType::F32;

    // ---------------------------------------------------------- correctness
    // Small case first: does im2col/taps/chunked actually equal conv1d?
    {
        let (c, len, k, d) = (8usize, 64usize, 7usize, 9usize);
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let w = Tensor::randn(0f32, 0.3, (c, c, k), &dev)?;
        let reference = conv_direct(&x, &w, k, d)?;
        for (name, got) in [
            ("im2col", conv_im2col(&x, &w, k, d)?),
            ("taps", conv_taps(&x, &w, k, d)?),
            ("chunked(16)", conv_chunked(&x, &w, k, d, 16)?),
        ] {
            let diff = (&reference - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = reference.abs()?.max_all()?.to_scalar::<f32>()?;
            println!(
                "correctness {name:<12} shape {:?}  max|diff| {diff:.3e}  (scale {scale:.3e})  {}",
                got.dims(),
                if diff < 1e-3 { "OK" } else { "MISMATCH" }
            );
        }
        println!();
    }

    // ------------------------------------------- the four residual-unit shapes
    // (channels, length). MACs = C*C*7*L, so blocks 2 and 3 do 2x the work of 1 and 4.
    println!("{:<22} {:>9} {:>9} {:>9} {:>9} {:>10}",
             "shape (k7, dil9)", "direct", "im2col", "taps", "chunk8k", "best/direct");
    println!("{}", "-".repeat(76));

    let mut total_direct = 0.0;
    let mut total_best = 0.0;

    for (c, len) in [(768usize, 2048usize), (384, 16384), (192, 65536), (96, 131072)] {
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (c, c, 7), &dev)?;

        let (t_direct, _) = time(&dev, || conv_direct(&x, &w, 7, 9))?;
        let (t_im2col, _) = time(&dev, || conv_im2col(&x, &w, 7, 9))?;
        let (t_taps, _) = time(&dev, || conv_taps(&x, &w, 7, 9))?;
        let (t_chunk, _) = time(&dev, || conv_chunked(&x, &w, 7, 9, 8192))?;

        let best = t_im2col.min(t_taps).min(t_chunk);
        total_direct += t_direct;
        total_best += best;
        println!(
            "{:<22} {t_direct:>8.2} {t_im2col:>9.2} {t_taps:>9.2} {t_chunk:>9.2} {:>9.2}x",
            format!("{c}ch @ {len}"),
            t_direct / best
        );
        println!(
            "{:<22} {:>8.0} {:>9.0} {:>9.0} {:>9.0}   GFLOPS",
            "",
            gflops(c, c, 7, len, t_direct),
            gflops(c, c, 7, len, t_im2col),
            gflops(c, c, 7, len, t_taps),
            gflops(c, c, 7, len, t_chunk),
        );
    }
    println!("{}", "-".repeat(76));
    println!("one k7 conv per stage: direct {total_direct:.1} ms -> best {total_best:.1} ms ({:.2}x)",
             total_direct / total_best);

    // ------------------------------------------------ the pathological tail conv
    // Cout = 1. Measured at 47.48 ms in the first probe -- 3.7 GFLOPS.
    println!("\ntail conv 96->1 k7 @ 131072  (Cout=1, the worst case):");
    {
        let x = Tensor::randn(0f32, 1.0, (1, 96, 131072), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1, 96, 7), &dev)?;
        let (t_direct, _) = time(&dev, || conv_direct(&x, &w, 7, 1))?;
        let (t_im2col, _) = time(&dev, || conv_im2col(&x, &w, 7, 1))?;
        let (t_chunk, _) = time(&dev, || conv_chunked(&x, &w, 7, 1, 8192))?;
        println!("  direct {t_direct:8.2} ms   im2col {t_im2col:8.2} ms   chunk8k {t_chunk:8.2} ms   -> {:.1}x",
                 t_direct / t_im2col.min(t_chunk));
    }

    // ----------------------------------------------- k1 conv: literally a GEMM
    println!("\nk1 conv 96->96 @ 131072  (a pure matmul in disguise):");
    {
        let x = Tensor::randn(0f32, 1.0, (1, 96, 131072), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (96, 96, 1), &dev)?;
        let w2 = w.squeeze(2)?.t()?.contiguous()?;
        let (t_direct, _) = time(&dev, || conv_direct(&x, &w, 1, 1))?;
        let (t_mm, _) = time(&dev, || {
            x.squeeze(0)?.t()?.contiguous()?.matmul(&w2)?.t()?.contiguous()?.unsqueeze(0)
        })?;
        println!("  direct {t_direct:8.2} ms   matmul {t_mm:8.2} ms   -> {:.1}x",
                 t_direct / t_mm);
    }

    // ------------------------------------------------------- snake alone, cost
    // 2 per residual unit, 24 total. Each reads+writes the full activation.
    println!("\nsnake alone, 96ch @ 131072 (elementwise traffic floor):");
    {
        let x = Tensor::randn(0f32, 1.0, (1, 96, 131072), &dev)?;
        let alpha = Tensor::ones((1, 96, 1), dt, &dev)?;
        let (ms, _) = time(&dev, || {
            let recip = (&alpha + 1e-9)?.recip()?;
            &x + x.broadcast_mul(&alpha)?.sin()?.sqr()?.broadcast_mul(&recip)?
        })?;
        let bytes = (96 * 131072 * 4 * 2) as f64;
        println!("  {ms:.2} ms  ({:.0} GB/s effective on 2x50 MB)", bytes / (ms / 1000.0) / 1e9);
    }

    Ok(())
}
