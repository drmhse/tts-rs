//! Where the CosyVoice flow decoder's time goes, and whether the matmul shape is the
//! problem.
//!
//! Measured motivation: an end-to-end render put the flow at RTF 2.007, 68% of total.
//! Its arithmetic is ~2.95 TMAC (10 steps x 2 guidance x 22 blocks x 798 positions x
//! 8.4 MMAC), so 9.5 s implies about 0.62 TFLOP/s — far under what this GPU does on a
//! large f32 GEMM. That gap says implementation, not algorithm, which is a claim worth
//! testing rather than acting on.
//!
//! The hypothesis under test: `Linear::forward` uses `broadcast_matmul`, so a
//! `[2, 798, 1024] @ [1024, 1024]` becomes a *batched* matmul with the weight expanded
//! along the batch axis. Collapsing the leading dimensions into one `[1596, 1024]` GEMM
//! instead should hit a single large kernel with no expansion.
//!
//! Run: `cargo run -p a8-probe --release --bin dit`

use a8_probe::bench::Harness;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const DIM: usize = 1024;
const FF: usize = 2048;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
/// The fixture's mel length: 588 prompt frames + 210 generated.
const N: usize = 798;
const BATCH: usize = 2;

/// `x @ w` with `x` of any leading shape, as a single 2-D GEMM.
fn flat_matmul(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let k = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let out = w.dim(1)?;
    let y = x.reshape((rows, k))?.matmul(w)?;
    let y = match b {
        Some(b) => y.broadcast_add(b)?,
        None => y,
    };
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out);
    y.reshape(shape)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 7)?;
    println!("DiT shapes: dim {DIM}, ff {FF}, {HEADS} heads x {HEAD_DIM}, n {N}, batch {BATCH}");

    let x = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
    let w = Tensor::randn(0f32, 1.0, (DIM, DIM), &dev)?;
    let b = Tensor::randn(0f32, 1.0, DIM, &dev)?;

    // ---------------------------------------------------------------- projection shape
    {
        let (x, w, b) = (x.clone(), w.clone(), b.clone());
        let (x2, w2, b2) = (x.clone(), w.clone(), b.clone());
        h.ab(
            "one 1024x1024 projection over [2, 798, 1024]",
            &mut [
                ("broadcast_matmul (what Linear does)", &mut move || {
                    let y = x.broadcast_matmul(&w)?;
                    let _ = y.broadcast_add(&b)?;
                    Ok(())
                }),
                ("flattened 2-D matmul", &mut move || {
                    let _ = flat_matmul(&x2, &w2, Some(&b2))?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ---------------------------------------------------------------- a whole block
    // Everything a DiTBlock does, both ways, so the projection result is seen in context
    // rather than in isolation.
    let wq = Tensor::randn(0f32, 1.0, (DIM, DIM), &dev)?;
    let wk = Tensor::randn(0f32, 1.0, (DIM, DIM), &dev)?;
    let wv = Tensor::randn(0f32, 1.0, (DIM, DIM), &dev)?;
    let wo = Tensor::randn(0f32, 1.0, (DIM, DIM), &dev)?;
    let w1 = Tensor::randn(0f32, 1.0, (DIM, FF), &dev)?;
    let w2m = Tensor::randn(0f32, 1.0, (FF, DIM), &dev)?;

    let attention = |q: &Tensor, k: &Tensor, v: &Tensor| -> candle_core::Result<Tensor> {
        let heads = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((BATCH, N, HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (heads(q)?, heads(k)?, heads(v)?);
        let s = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (HEAD_DIM as f64).sqrt()))?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        p.matmul(&v)?
            .transpose(1, 2)?
            .reshape((BATCH, N, DIM))?
            .contiguous()
    };

    {
        let (x, wq, wk, wv, wo, w1, w2m) = (
            x.clone(),
            wq.clone(),
            wk.clone(),
            wv.clone(),
            wo.clone(),
            w1.clone(),
            w2m.clone(),
        );
        let (xf, wqf, wkf, wvf, wof, w1f, w2f) = (
            x.clone(),
            wq.clone(),
            wk.clone(),
            wv.clone(),
            wo.clone(),
            w1.clone(),
            w2m.clone(),
        );
        h.ab(
            "one DiT block (4 x 1024^2 + 2 x 1024x2048 + attention)",
            &mut [
                ("broadcast_matmul", &mut move || {
                    let q = x.broadcast_matmul(&wq)?;
                    let k = x.broadcast_matmul(&wk)?;
                    let v = x.broadcast_matmul(&wv)?;
                    let a = attention(&q, &k, &v)?;
                    let a = a.broadcast_matmul(&wo)?;
                    let y = (&x + a)?;
                    let f = y.broadcast_matmul(&w1)?.gelu()?;
                    let _ = (&y + f.broadcast_matmul(&w2m)?)?;
                    Ok(())
                }),
                ("flattened", &mut move || {
                    let q = flat_matmul(&xf, &wqf, None)?;
                    let k = flat_matmul(&xf, &wkf, None)?;
                    let v = flat_matmul(&xf, &wvf, None)?;
                    let a = attention(&q, &k, &v)?;
                    let a = flat_matmul(&a, &wof, None)?;
                    let y = (&xf + a)?;
                    let f = flat_matmul(&y, &w1f, None)?.gelu()?;
                    let _ = (&y + flat_matmul(&f, &w2f, None)?)?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ---------------------------------------------------------------- dtype
    // The reference ships an `fp16=True` path, so half precision is not out of bounds
    // upstream. What it costs in accuracy is a separate question from what it buys.
    {
        let xh = x.to_dtype(DType::F16)?;
        let wh = w.to_dtype(DType::F16)?;
        let (x32, w32) = (x.clone(), w.clone());
        h.ab(
            "projection dtype, flattened",
            &mut [
                ("f32", &mut move || {
                    let _ = flat_matmul(&x32, &w32, None)?;
                    Ok(())
                }),
                ("f16", &mut move || {
                    let _ = flat_matmul(&xh, &wh, None)?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ---------------------------------------------------------------- attention share
    {
        let q = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
        let k = q.clone();
        let v = q.clone();
        let proj_x = x.clone();
        let proj_w = w.clone();
        h.ab(
            "attention vs one projection, for the ratio",
            &mut [
                ("attention (qk + softmax + av)", &mut move || {
                    let _ = attention(&q, &k, &v)?;
                    Ok(())
                }),
                ("one flattened projection", &mut move || {
                    let _ = flat_matmul(&proj_x, &proj_w, None)?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ---------------------------------------------------------------- attention forms
    // Attention costs 8x a projection while doing 5x *less* arithmetic (2.6 GMAC against
    // 13.4 GMAC per block), so it is not compute-bound. The suspect is the scores tensor:
    // [2, 16, 798, 798] f32 is 81.5 MB, and the naive form writes it, reads and rewrites
    // it to apply the scale, reads and rewrites it in softmax, then reads it again — about
    // 490 MB of traffic on a ~120 GB/s bus.
    //
    // Two ways out, both exact: fold the scale into `q` before the matmul so the big
    // tensor is touched once less, or hand the whole thing to candle's fused Metal SDPA
    // kernel and never materialise it at all.
    {
        let heads4 = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((BATCH, N, HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        let base = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
        let (q0, k0, v0) = (heads4(&base)?, heads4(&base)?, heads4(&base)?);
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        let (qa, ka, va) = (q0.clone(), k0.clone(), v0.clone());
        let (qb, kb, vb) = (q0.clone(), k0.clone(), v0.clone());
        let (qc, kc, vc) = (q0.clone(), k0.clone(), v0.clone());
        h.ab(
            "attention forms, [2, 16, 798, 64]",
            &mut [
                ("scale the scores (naive)", &mut move || {
                    let s = (qa.matmul(&ka.transpose(2, 3)?)? * scale)?;
                    let _ = candle_nn::ops::softmax_last_dim(&s)?.matmul(&va)?;
                    Ok(())
                }),
                ("scale q instead", &mut move || {
                    let s = (&qb * scale)?.matmul(&kb.transpose(2, 3)?)?;
                    let _ = candle_nn::ops::softmax_last_dim(&s)?.matmul(&vb)?;
                    Ok(())
                }),
                ("candle_nn::ops::sdpa (fused)", &mut move || {
                    let _ = candle_nn::ops::sdpa(&qc, &kc, &vc, None, false, scale as f32, 1.0)?;
                    Ok(())
                }),
            ],
        )?;

        // The 11.5 ms "attention" row above includes the head reshapes; the 6.3 ms naive
        // row starts from already-shaped tensors. The ~5 ms difference is three
        // `contiguous()` copies of a 6.5 MB tensor. `sdpa` takes strides, so it may not
        // need them — worth asking, because it is the difference between one kernel and
        // one kernel plus 20 MB of copying.
        let src = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
        let lazy = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((BATCH, N, HEADS, HEAD_DIM))?.transpose(1, 2)
        };
        let (s1, s2) = (src.clone(), src.clone());
        h.ab(
            "does fused sdpa need contiguous heads?",
            &mut [
                ("contiguous first", &mut move || {
                    let q = lazy(&s1)?.contiguous()?;
                    let k = lazy(&s1)?.contiguous()?;
                    let v = lazy(&s1)?.contiguous()?;
                    let _ = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
                    Ok(())
                }),
                ("transposed views only", &mut move || {
                    let q = lazy(&s2)?;
                    let k = lazy(&s2)?;
                    let v = lazy(&s2)?;
                    let _ = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
                    Ok(())
                }),
            ],
        )?;

        // Fused or not, it has to be the same function.
        let want = {
            let s = (q0.matmul(&k0.transpose(2, 3)?)? * scale)?;
            candle_nn::ops::softmax_last_dim(&s)?.matmul(&v0)?
        };
        let got = candle_nn::ops::sdpa(&q0, &k0, &v0, None, false, scale as f32, 1.0)?;
        let d = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let s = want.abs()?.max_all()?.to_scalar::<f32>()?;
        println!(
            "\nsdpa vs naive: max|diff| {d:.3e} on a scale of {s:.3e}  (rel {:.2e})",
            d / s
        );
    }

    // ---------------------------------------------------------------- norms
    // `AdaLayerNormZero` normalises with no affine parameters, then applies a predicted
    // scale and shift. Written out of primitives that is six passes over a 6.5 MB tensor
    // for the norm plus two more for the modulation, and it happens three times per block
    // — 660 times per utterance. candle has a fused `layer_norm`; using it with a unit
    // weight and zero bias is the same function and one kernel.
    {
        let xs = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
        let one = Tensor::ones(DIM, DType::F32, &dev)?;
        let zero = Tensor::zeros(DIM, DType::F32, &dev)?;
        let eps = 1e-6;

        let manual = |x: &Tensor| -> candle_core::Result<Tensor> {
            let mean = x.mean_keepdim(candle_core::D::Minus1)?;
            let c = x.broadcast_sub(&mean)?;
            let var = c.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            c.broadcast_div(&(var + eps)?.sqrt()?)
        };

        let x1 = xs.clone();
        let (x2, o2, z2) = (xs.clone(), one.clone(), zero.clone());
        h.ab(
            "affine-free layer norm over [2, 798, 1024]",
            &mut [
                ("hand-written (6 passes)", &mut move || {
                    let _ = manual(&x1)?;
                    Ok(())
                }),
                ("candle_nn::ops::layer_norm (fused)", &mut move || {
                    let _ = candle_nn::ops::layer_norm(&x2, &o2, &z2, eps as f32)?;
                    Ok(())
                }),
            ],
        )?;

        let got = candle_nn::ops::layer_norm(&xs, &one, &zero, eps as f32)?;
        let want = manual(&xs)?;
        let d = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("fused layer_norm vs hand-written: max|diff| {d:.3e}");
    }

    // ---------------------------------------------------------------- position embedding
    // Two grouped convolutions, k=31, groups=16, over [2, 1024, 798] — 3.2 GMAC each.
    // This runs once per solver step rather than per block, so 20 times per utterance, but
    // candle's grouped conv1d is the op this project already measured at 19x off the
    // hardware for the depthwise case (see `tts_nn::depthwise_k7`).
    {
        let xs = Tensor::randn(0f32, 1.0, (BATCH, DIM, N), &dev)?;
        let wg = Tensor::randn(0f32, 1.0, (DIM, DIM / 16, 31), &dev)?;

        let looped = |x: &Tensor, w: &Tensor| -> candle_core::Result<Tensor> {
            let xpad = x.pad_with_zeros(2, 30, 0)?;
            let mut parts = Vec::with_capacity(16);
            for g in 0..16 {
                let xg = xpad.narrow(1, g * 64, 64)?.contiguous()?;
                let wgp = w.narrow(0, g * 64, 64)?.contiguous()?;
                parts.push(xg.conv1d(&wgp, 0, 1, 1, 1)?);
            }
            Tensor::cat(&parts, 1)
        };

        let (x1, w1g) = (xs.clone(), wg.clone());
        let (x2, w2g) = (xs.clone(), wg.clone());
        h.ab(
            "grouped conv1d, k=31 g=16, [2, 1024, 798]",
            &mut [
                ("candle groups=16 directly", &mut move || {
                    let _ = x1.pad_with_zeros(2, 30, 0)?.conv1d(&w1g, 0, 1, 1, 16)?;
                    Ok(())
                }),
                ("16 separate convs (what tts-nn does)", &mut move || {
                    let _ = looped(&x2, &w2g)?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ---------------------------------------------------------------- realistic block
    // The mocked block above came out at 20.3 ms; the loaded model spends 28.4 ms. The
    // difference has to be in the parts the mock left out — the AdaLayerNorm modulation
    // and the partial rotary embedding. `partial_rope` is the suspect: it narrows 64 of
    // 1024 channels, rotates them, and concatenates the result back, which materialises a
    // fresh 6.5 MB tensor per projection.
    {
        let xs = Tensor::randn(0f32, 1.0, (BATCH, N, DIM), &dev)?;
        let ada = Tensor::randn(0f32, 1.0, (DIM, 6 * DIM), &dev)?;
        let t = Tensor::randn(0f32, 1.0, (BATCH, DIM), &dev)?;
        let one = Tensor::ones(DIM, DType::F32, &dev)?;
        let zero = Tensor::zeros(DIM, DType::F32, &dev)?;
        let half = HEAD_DIM / 2;
        let cos = Tensor::randn(0f32, 1.0, (N, half), &dev)?;
        let sin = Tensor::randn(0f32, 1.0, (N, half), &dev)?;
        let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;

        // Rotate the first 64 channels and concatenate — what the port does.
        let rope_cat = |x: &Tensor| -> candle_core::Result<Tensor> {
            let rot = x.narrow(2, 0, HEAD_DIM)?.reshape((BATCH, 1, N, HEAD_DIM))?;
            let rot = candle_nn::rotary_emb::rope_i(&rot.contiguous()?, &cos, &sin)?;
            let rot = rot.reshape((BATCH, N, HEAD_DIM))?;
            let rest = x.narrow(2, HEAD_DIM, DIM - HEAD_DIM)?;
            Tensor::cat(&[rot, rest], 2)?.contiguous()
        };
        let heads = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((BATCH, N, HEADS, HEAD_DIM))?.transpose(1, 2)
        };

        let block = |x: &Tensor, rope: bool| -> candle_core::Result<Tensor> {
            let e = flat_matmul(&candle_nn::ops::silu(t.as_ref())?, &ada, None)?;
            let take = |i: usize| e.narrow(1, i * DIM, DIM)?.reshape((BATCH, 1, DIM));
            let (sh_a, sc_a, g_a) = (take(0)?, take(1)?, take(2)?);
            let (sh_f, sc_f, g_f) = (take(3)?, take(4)?, take(5)?);

            let nx = candle_nn::ops::layer_norm(x, &one, &zero, 1e-6)?;
            let nx = nx.broadcast_mul(&(sc_a + 1.0)?)?.broadcast_add(&sh_a)?;

            let (q, k) = if rope {
                (
                    rope_cat(&flat_matmul(&nx, &wq, None)?)?,
                    rope_cat(&flat_matmul(&nx, &wk, None)?)?,
                )
            } else {
                (flat_matmul(&nx, &wq, None)?, flat_matmul(&nx, &wk, None)?)
            };
            let v = flat_matmul(&nx, &wv, None)?;
            let a = candle_nn::ops::sdpa(
                &heads(&q)?,
                &heads(&k)?,
                &heads(&v)?,
                None,
                false,
                scale,
                1.0,
            )?;
            let a = a.transpose(1, 2)?.reshape((BATCH, N, DIM))?;
            let a = flat_matmul(&a, &wo, None)?;
            let x = (x + a.broadcast_mul(&g_a)?)?;

            let nf = candle_nn::ops::layer_norm(&x, &one, &zero, 1e-6)?;
            let nf = nf.broadcast_mul(&(sc_f + 1.0)?)?.broadcast_add(&sh_f)?;
            let f = flat_matmul(&nf, &w1, None)?.gelu()?;
            let f = flat_matmul(&f, &w2m, None)?;
            &x + f.broadcast_mul(&g_f)?
        };

        // Split attention by head group instead. Only head 0 is rotated and attention is
        // independent per head, so the rotated head and the other fifteen can be two
        // `sdpa` calls over *views*, with nothing rebuilt. That replaces two 6.5 MB
        // concatenations with one — and the one that remains subsumes the contiguous copy
        // the `transpose -> reshape` after attention needed anyway.
        let split = |x: &Tensor| -> candle_core::Result<Tensor> {
            let e = flat_matmul(&candle_nn::ops::silu(t.as_ref())?, &ada, None)?;
            let take = |i: usize| e.narrow(1, i * DIM, DIM)?.reshape((BATCH, 1, DIM));
            let (sh_a, sc_a, g_a) = (take(0)?, take(1)?, take(2)?);
            let (sh_f, sc_f, g_f) = (take(3)?, take(4)?, take(5)?);

            let nx = candle_nn::ops::layer_norm(x, &one, &zero, 1e-6)?;
            let nx = nx.broadcast_mul(&(sc_a + 1.0)?)?.broadcast_add(&sh_a)?;

            let qh = heads(&flat_matmul(&nx, &wq, None)?)?;
            let kh = heads(&flat_matmul(&nx, &wk, None)?)?;
            let vh = heads(&flat_matmul(&nx, &wv, None)?)?;
            let rot = |t: &Tensor| -> candle_core::Result<Tensor> {
                candle_nn::rotary_emb::rope_i(&t.narrow(1, 0, 1)?.contiguous()?, &cos, &sin)
            };
            let o0 = candle_nn::ops::sdpa(
                &rot(&qh)?,
                &rot(&kh)?,
                &vh.narrow(1, 0, 1)?,
                None,
                false,
                scale,
                1.0,
            )?;
            let rest = |t: &Tensor| t.narrow(1, 1, HEADS - 1);
            let orest = candle_nn::ops::sdpa(
                &rest(&qh)?,
                &rest(&kh)?,
                &rest(&vh)?,
                None,
                false,
                scale,
                1.0,
            )?;
            let a = Tensor::cat(&[o0, orest], 1)?
                .transpose(1, 2)?
                .reshape((BATCH, N, DIM))?;
            let a = flat_matmul(&a, &wo, None)?;
            let x = (x + a.broadcast_mul(&g_a)?)?;

            let nf = candle_nn::ops::layer_norm(&x, &one, &zero, 1e-6)?;
            let nf = nf.broadcast_mul(&(sc_f + 1.0)?)?.broadcast_add(&sh_f)?;
            let f = flat_matmul(&nf, &w1, None)?.gelu()?;
            let f = flat_matmul(&f, &w2m, None)?;
            &x + f.broadcast_mul(&g_f)?
        };

        let x1 = xs.clone();
        let x2 = xs.clone();
        let x3 = xs.clone();
        h.ab(
            "the real block: how to pay for partial rotary",
            &mut [
                ("rotate then concatenate (as implemented)", &mut move || {
                    let _ = block(&x1, true)?;
                    Ok(())
                }),
                ("split attention by head group", &mut move || {
                    let _ = split(&x3)?;
                    Ok(())
                }),
                (
                    "no rope at all (lower bound, not correct)",
                    &mut move || {
                        let _ = block(&x2, false)?;
                        Ok(())
                    },
                ),
            ],
        )?;
    }

    println!("\nsdpa on narrowed views:");
    sdpa_offset_check(&dev)?;
    println!("\nsdpa with an additive mask:");
    sdpa_mask_check(&dev)?;

    h.report_drift()?;
    Ok(())
}

// Appended diagnostic: does `sdpa` compute the right thing on a *narrowed* transposed
// view? The head-split variant benchmarked above was never checked for correctness, and
// wiring it into the engine put block 0 off by rel 1.5e-1 — so speed was measured on
// something that did not compute the same function. This isolates the reason.
#[allow(dead_code)]
fn sdpa_mask_check(dev: &Device) -> Result<()> {
    // Second candle `sdpa` defect, found by swapping it into the Audio8 codec's windowed
    // attention: with an additive mask it is wrong at short sequence lengths. The codec's
    // 24-frame fixture passed at 7.6e-6 while its 8-frame fixture came back at 6.4e-1 — a
    // relative error of 7.0, i.e. not an answer. Same call, same shapes, only `t` differs.
    let (b, hh, hd) = (1usize, 8usize, 64usize);
    let scale = 1.0 / (hd as f64).sqrt();
    println!("  masked sdpa vs naive, by sequence length:");
    for t in [4usize, 8, 12, 16, 24, 32, 64] {
        let q = Tensor::randn(0f32, 1.0, (b, hh, t, hd), dev)?;
        let mut m = vec![0f32; t * t];
        for r in 0..t {
            for c in 0..t {
                if c > r {
                    m[r * t + c] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(m, (1, 1, t, t), dev)?
            .broadcast_as((b, hh, t, t))?
            .contiguous()?;
        let want = {
            let s = (q.matmul(&q.transpose(2, 3)?.contiguous()?)? * scale)?;
            candle_nn::ops::softmax_last_dim(&s.broadcast_add(&mask)?)?.matmul(&q)?
        };
        let got = candle_nn::ops::sdpa(&q, &q, &q, Some(&mask), false, scale as f32, 1.0)?;
        let d = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let sc = want.abs()?.max_all()?.to_scalar::<f32>()?;
        println!(
            "    t = {t:>3}: max|diff| {d:.3e}  rel {:.2e}{}",
            d / sc,
            if d / sc > 1e-4 { "   <-- WRONG" } else { "" }
        );
    }
    Ok(())
}

#[allow(dead_code)]
fn sdpa_offset_check(dev: &Device) -> Result<()> {
    let (b, hh, n, hd) = (2usize, 16usize, 64usize, 64usize);
    let src = Tensor::randn(0f32, 1.0, (b, n, hh, hd), dev)?;
    let view = src.transpose(1, 2)?; // [b, 16, n, 64], strided
    let scale = 1.0 / (hd as f64).sqrt();

    let naive = |q: &Tensor, k: &Tensor, v: &Tensor| -> candle_core::Result<Tensor> {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let s = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)
    };

    for (label, q) in [
        ("full view [b,16,n,64]", view.clone()),
        ("narrow(1,0,1) -> [b,1,n,64]", view.narrow(1, 0, 1)?),
        ("narrow(1,1,15) -> [b,15,n,64]", view.narrow(1, 1, 15)?),
    ] {
        let want = naive(&q, &q, &q)?;
        let got = candle_nn::ops::sdpa(&q, &q, &q, None, false, scale as f32, 1.0)?;
        let d = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let s = want.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("  {label:<32} max|diff| {d:.3e}  rel {:.2e}", d / s);
    }
    Ok(())
}
