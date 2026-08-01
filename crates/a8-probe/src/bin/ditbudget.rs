//! The DiT block's budget at the engine's real sequence length, op by op.
//!
//! `flowsplit` puts the flow at 68.5% of CosyVoice and one block at 130.9 ms for 3192
//! frames. Fitting `a*n + b*n^2` across 798 and 3192 frames splits that into **66% linear
//! (projections and feed-forward) and 34% attention**.
//!
//! The linear part runs at ~1.23 TFLOP/s where this machine's GEMM reaches 2.4 TFLOP/s on
//! a well-shaped problem, so the arithmetic is not the limit — the shape and the count of
//! launches are. Attention is already at 1.90 TFLOP/s through the fused kernel and has
//! much less headroom.
//!
//! So this measures every op in a block at `[2, 3192, 1024]`, and tests the two structural
//! changes that need no numerical compromise:
//!
//! - **q/k/v as one projection** instead of three. The LLM already fuses its `wqkv`; the
//!   DiT never did. One `[6384,1024]@[1024,3072]` should beat three `@[1024,1024]`.
//! - **a fused gelu**, on the same footing as the snake kernels.
//!
//! Run: `cargo run -p a8-probe --release --bin ditbudget`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_bench::Harness;
use tts_nn::{gelu_tanh, matmul_2d, LayerNormPlain};

/// `[batch, frames, dim]` as the engine runs it: CFG doubles the batch.
const B: usize = 2;
const N: usize = 3192;
const D: usize = 1024;
const FF: usize = 2048;
const HEADS: usize = 16;
const HD: usize = 64;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    let x = Tensor::randn(0f32, 1., (B, N, D), &dev)?;
    let wq = Tensor::randn(0f32, 0.02, (D, D), &dev)?;
    let wk = Tensor::randn(0f32, 0.02, (D, D), &dev)?;
    let wv = Tensor::randn(0f32, 0.02, (D, D), &dev)?;
    let wqkv = Tensor::cat(&[&wq, &wk, &wv], 1)?.contiguous()?;
    let wo = Tensor::randn(0f32, 0.02, (D, D), &dev)?;
    let w_ffin = Tensor::randn(0f32, 0.02, (D, FF), &dev)?;
    let w_ffout = Tensor::randn(0f32, 0.02, (FF, D), &dev)?;
    let ffh = Tensor::randn(0f32, 1., (B, N, FF), &dev)?;
    let norm = LayerNormPlain::new(D, 1e-6, &dev)?;
    let scale = Tensor::randn(0f32, 1., (B, 1, D), &dev)?;
    let shift = Tensor::randn(0f32, 1., (B, 1, D), &dev)?;

    let heads = |t: &Tensor| -> candle_core::Result<Tensor> {
        t.reshape((B, N, HEADS, HD))?.transpose(1, 2)?.contiguous()
    };
    let q = heads(&matmul_2d(&x, &wq).unwrap())?;
    let k = heads(&matmul_2d(&x, &wk).unwrap())?;
    let v = heads(&matmul_2d(&x, &wv).unwrap())?;
    let sc = 1.0 / (HD as f32).sqrt();

    // The fused projection must produce exactly what three separate ones do, or the
    // timing is meaningless.
    let split = matmul_2d(&x, &wqkv)?;
    let want = matmul_2d(&x, &wk)?;
    let got = split.narrow(2, D, D)?.contiguous()?;
    let d = tts_nn::max_abs_diff(&want, &got)?;
    println!("fused qkv vs separate k: {d:.2e}\n");
    anyhow::ensure!(d == 0.0, "fused qkv disagrees");

    let mut qkv_split = || -> candle_core::Result<()> {
        matmul_2d(&x, &wq).unwrap();
        matmul_2d(&x, &wk).unwrap();
        matmul_2d(&x, &wv).unwrap();
        Ok(())
    };
    let mut qkv_fused = || -> candle_core::Result<()> {
        matmul_2d(&x, &wqkv).unwrap();
        Ok(())
    };
    let mut attn = || -> candle_core::Result<()> {
        candle_nn::ops::sdpa(&q, &k, &v, None, false, sc, 1.0)?;
        Ok(())
    };
    let mut out_proj = || -> candle_core::Result<()> {
        matmul_2d(&x, &wo).unwrap();
        Ok(())
    };
    let mut ff_up = || -> candle_core::Result<()> {
        matmul_2d(&x, &w_ffin).unwrap();
        Ok(())
    };
    let mut ff_down = || -> candle_core::Result<()> {
        matmul_2d(&ffh, &w_ffout).unwrap();
        Ok(())
    };
    let mut gelu = || -> candle_core::Result<()> {
        gelu_tanh(&ffh).unwrap();
        Ok(())
    };
    let mut modulate = || -> candle_core::Result<()> {
        let nx = norm.forward(&x).unwrap();
        nx.broadcast_mul(&(scale.clone() + 1.0).unwrap())
            .unwrap()
            .broadcast_add(&shift)
            .unwrap();
        Ok(())
    };
    let mut reshape_heads = || -> candle_core::Result<()> {
        heads(&x)?;
        Ok(())
    };

    let stats = h.ab(
        &format!("dit block ops @ [{B}, {N}, {D}]"),
        &mut [
            ("qkv as 3 matmuls", &mut qkv_split),
            ("qkv as 1 matmul", &mut qkv_fused),
            ("sdpa", &mut attn),
            ("out proj", &mut out_proj),
            ("ff up (d->4864/2)", &mut ff_up),
            ("ff down", &mut ff_down),
            ("gelu_tanh", &mut gelu),
            ("layernorm + modulate", &mut modulate),
            ("reshape to heads", &mut reshape_heads),
        ],
    )?;

    println!("\n{:>24} {:>10} {:>10}", "op", "ms", "share of block");
    // A block runs: modulate, qkv, 3x reshape, sdpa, out, modulate, ff_up, gelu, ff_down.
    let by = |name: &str| stats.iter().find(|s| s.name == name).unwrap().median;
    let block = by("qkv as 3 matmuls")
        + 3.0 * by("reshape to heads")
        + by("sdpa")
        + by("out proj")
        + by("ff up (d->4864/2)")
        + by("gelu_tanh")
        + by("ff down")
        + 2.0 * by("layernorm + modulate");
    for s in &stats {
        println!(
            "{:>24} {:>10.2} {:>13.1}%",
            s.name,
            s.median,
            100.0 * s.median / block
        );
    }
    println!(
        "{:>24} {:>10.2}   (measured in flowsplit: 130.9)",
        "modelled block", block
    );
    println!(
        "\nqkv fusion saves {:.2} ms/block -> {:.1}% of the block, {:.1}% of the flow",
        by("qkv as 3 matmuls") - by("qkv as 1 matmul"),
        100.0 * (by("qkv as 3 matmuls") - by("qkv as 1 matmul")) / block,
        100.0 * (by("qkv as 3 matmuls") - by("qkv as 1 matmul")) / block,
    );

    h.report_drift()?;
    Ok(())
}
