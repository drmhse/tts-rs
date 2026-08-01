//! Re-examining conv-as-GEMM with the *other* im2col layout.
//!
//! `docs/performance/candle-on-metal.md` Finding 1 refuted conv-as-GEMM and attributed the loss to
//! materialisation traffic: "the `[131072, 672]` matrix is 352 MB to write and 352 MB to read
//! back ... That traffic exceeds what the better GEMM shape saves."
//!
//! The arithmetic does not support that attribution. 704 MB at ~120 GB/s is about 6 ms, but
//! im2col measured **95.40 ms** against a direct conv's 59.78 ms. So ~89 ms went somewhere
//! else, and 17 GFLOP in 89 ms is 0.19 TFLOP/s — which is what a GEMM with `N = 96` looks
//! like, not what memory traffic looks like.
//!
//! That points at the *layout*, not the traffic. `[131072, 672] @ [672, 96]` is
//! `M = 131072, K = 672, N = 96`: a very skinny output. The same convolution can be written
//! `[96, 672] @ [672, 131072]` — `M = 96, K = 672, N = 131072` — which is the shape Metal's
//! GEMM actually wants, and which produces channels-first output directly, as the codec needs.
//!
//! Same traffic either way. If the transposed form wins, the original conclusion was right
//! that im2col loses but wrong about why, and the low-channel deficit has a fix in candle
//! after all.
//!
//! Run: `cargo run -p a8-probe --release --bin convgemm`

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tts_bench::Harness;

/// Causal conv as `[cout, cin*k] @ [cin*k, L]`.
///
/// Row order of the im2col matrix must match the weight's memory order. A candle conv weight
/// is `[out, in, k]` contiguous, so flattening its trailing dims gives row index
/// `in * k + tap` — hence the im2col is built as `[cin, k, L]` and reshaped, not `[k, cin, L]`.
/// Getting that backwards produces a plausible-looking waveform from a transposed kernel.
fn conv_via_gemm(x: &Tensor, w: &Tensor, dilation: usize) -> candle_core::Result<Tensor> {
    let (_, cin, len) = x.dims3()?;
    let cout = w.dim(0)?;
    let k = w.dim(2)?;
    let xpad = x.pad_with_zeros(2, (k - 1) * dilation, 0)?;

    let mut taps = Vec::with_capacity(k);
    for t in 0..k {
        taps.push(xpad.narrow(2, t * dilation, len)?.squeeze(0)?);
    }
    // [cin, k, len] -> [cin * k, len]
    let cols = Tensor::stack(&taps, 1)?.reshape((cin * k, len))?;
    let wm = w.reshape((cout, cin * k))?;
    wm.matmul(&cols)?.reshape((1, cout, len))
}

/// The same, blocked along length so the im2col matrix never exceeds `chunk` columns.
fn conv_via_gemm_chunked(
    x: &Tensor,
    w: &Tensor,
    dilation: usize,
    chunk: usize,
) -> candle_core::Result<Tensor> {
    let (_, cin, len) = x.dims3()?;
    let cout = w.dim(0)?;
    let k = w.dim(2)?;
    let pad = (k - 1) * dilation;
    let xpad = x.pad_with_zeros(2, pad, 0)?;
    let wm = w.reshape((cout, cin * k))?;

    let mut parts = Vec::new();
    let mut start = 0;
    while start < len {
        let n = chunk.min(len - start);
        let mut taps = Vec::with_capacity(k);
        for t in 0..k {
            taps.push(xpad.narrow(2, start + t * dilation, n)?.squeeze(0)?);
        }
        let cols = Tensor::stack(&taps, 1)?.reshape((cin * k, n))?;
        parts.push(wm.matmul(&cols)?);
        start += n;
    }
    Tensor::cat(&parts, 1)?.reshape((1, cout, len))
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 7)?;

    // Every stage of the codec's decoder, so the answer is not specific to the canary.
    let stages = [
        (768usize, 2048usize),
        (384, 16384),
        (192, 65536),
        (96, 131072),
    ];
    let (k, dil) = (7usize, 9usize);

    for (ch, len) in stages {
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (ch, ch, k), &dev)?;

        // Correctness before speed — the mistake this project already made once.
        let want = x
            .pad_with_zeros(2, (k - 1) * dil, 0)?
            .conv1d(&w, 0, 1, dil, 1)?;
        let got = conv_via_gemm(&x, &w, dil)?;
        let d = tts_nn::max_abs_diff(&got, &want)?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
        let got_c = conv_via_gemm_chunked(&x, &w, dil, 16384)?;
        let dc = tts_nn::max_abs_diff(&got_c, &want)?;
        println!(
            "\n{ch}ch @ {len}: gemm rel {:.2e}, chunked rel {:.2e}  ({})",
            d / scale,
            dc / scale,
            if d / scale < 1e-5 && dc / scale < 1e-5 {
                "both exact"
            } else {
                "MISMATCH — do not trust the timings below"
            }
        );

        let (x1, w1) = (x.clone(), w.clone());
        let (x2, w2) = (x.clone(), w.clone());
        let (x3, w3) = (x.clone(), w.clone());
        h.ab(
            &format!("k7 d9 conv, {ch}ch @ {len}"),
            &mut [
                ("candle conv1d (direct)", &mut move || {
                    let _ = x1
                        .pad_with_zeros(2, (k - 1) * dil, 0)?
                        .conv1d(&w1, 0, 1, dil, 1)?;
                    Ok(())
                }),
                ("gemm, [cout, cin*k] @ [cin*k, L]", &mut move || {
                    let _ = conv_via_gemm(&x2, &w2, dil)?;
                    Ok(())
                }),
                ("gemm, chunked at 16384", &mut move || {
                    let _ = conv_via_gemm_chunked(&x3, &w3, dil, 16384)?;
                    Ok(())
                }),
            ],
        )?;
    }

    // Neither layout wins, so the interesting question is where the extra time goes. Split
    // the GEMM route into its two halves at the canary shape: building the im2col matrix, and
    // multiplying by it.
    {
        let (ch, len) = (96usize, 131072usize);
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (ch, ch, k), &dev)?;
        let xpad = x.pad_with_zeros(2, (k - 1) * dil, 0)?;
        let wm = w.reshape((ch, ch * k))?;
        let cols = {
            let mut taps = Vec::with_capacity(k);
            for t in 0..k {
                taps.push(xpad.narrow(2, t * dil, len)?.squeeze(0)?);
            }
            Tensor::stack(&taps, 1)?.reshape((ch * k, len))?
        };
        println!(
            "\nim2col matrix is [{}, {len}] = {:.0} MB",
            ch * k,
            (ch * k * len * 4) as f64 / 1e6
        );

        let (xp1, wm1) = (xpad.clone(), wm.clone());
        let (wm2, cols2) = (wm.clone(), cols.clone());
        h.ab(
            "the GEMM route, split in two (96ch @ 131072)",
            &mut [
                ("build im2col only (7 narrows + stack)", &mut move || {
                    let mut taps = Vec::with_capacity(k);
                    for t in 0..k {
                        taps.push(xp1.narrow(2, t * dil, len)?.squeeze(0)?);
                    }
                    let _ = Tensor::stack(&taps, 1)?.reshape((ch * k, len))?;
                    let _ = &wm1;
                    Ok(())
                }),
                ("GEMM only, matrix prebuilt", &mut move || {
                    let _ = wm2.matmul(&cols2)?;
                    Ok(())
                }),
            ],
        )?;

        // The GEMM is fine — 2.4 TFLOP/s, and on its own already faster than candle's
        // direct conv. Everything is lost in the gather, which runs at ~4.9 GB/s. So the
        // question becomes: is there a *cheaper way to materialise the same matrix*?
        //
        // Two candidates. `stack(dim=1)` interleaves taps within each channel, so every
        // source row is scattered across `k` destination rows — poor locality. Concatenating
        // along dim 0 instead gives each tap one contiguous destination block, and the row
        // order becomes tap-major, which only means the weight must be permuted to
        // `[cout, k, cin]` — free, done once at load. The other candidate is an explicit
        // `index_select`, i.e. candle's gather kernel with a precomputed index vector.
        let idx = {
            let plen = xpad.dim(2)?;
            let mut v = Vec::with_capacity(ch * k * len);
            for c in 0..ch {
                for t in 0..k {
                    for i in 0..len {
                        v.push((c * plen + t * dil + i) as u32);
                    }
                }
            }
            Tensor::from_vec(v, ch * k * len, &dev)?
        };
        let flat = xpad.flatten_all()?;
        let (xp2, xp3, flat2, idx2) = (xpad.clone(), xpad.clone(), flat.clone(), idx.clone());
        let (xp4, d4) = (xpad.clone(), dev.clone());
        h.ab(
            "materialising the im2col matrix, 352 MB",
            &mut [
                ("stack(dim=1), tap-interleaved", &mut move || {
                    let mut taps = Vec::with_capacity(k);
                    for t in 0..k {
                        taps.push(xp2.narrow(2, t * dil, len)?.squeeze(0)?);
                    }
                    let _ = Tensor::stack(&taps, 1)?.reshape((ch * k, len))?;
                    Ok(())
                }),
                ("cat(dim=0), tap-major", &mut move || {
                    let mut taps = Vec::with_capacity(k);
                    for t in 0..k {
                        taps.push(xp3.narrow(2, t * dil, len)?);
                    }
                    let _ = Tensor::cat(&taps, 0)?.reshape((k * ch, len))?;
                    Ok(())
                }),
                ("index_select with a precomputed index", &mut move || {
                    let _ = flat2.index_select(&idx2, 0)?.reshape((ch * k, len))?;
                    Ok(())
                }),
                // Each tap is a strided 2-D block copy — `[ch, len]` out of a `[ch, plen]`
                // parent into a contiguous destination. candle has a `copy2d` Metal kernel
                // for exactly that; the question is whether `slice_set` reaches it, because
                // if so this is the fast gather without leaving candle.
                ("zeros + slice_set per tap", &mut move || {
                    let out = Tensor::zeros((k * ch, len), DType::F32, &d4)?;
                    for t in 0..k {
                        let tap = xp4.narrow(2, t * dil, len)?.squeeze(0)?;
                        out.slice_set(&tap, 0, t * ch)?;
                    }
                    Ok(())
                }),
            ],
        )?;
    }

    // `index_select` is 9.5x the naive gather, but a persistent index is as large as the
    // matrix it addresses — 352 MB for this shape alone, and the codec has a dozen distinct
    // (channels, length, dilation) combinations. So the question is whether the index can be
    // *generated* on device for less than it saves, since it is pure arithmetic:
    // `idx[c, t, i] = c * plen + t * dilation + i`, a broadcast sum of three ranges.
    //
    // And the index-free alternative: `cat(dim=0)` gives tap-major rows, which costs only a
    // weight permutation at load.
    for (ch, len) in stages {
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (ch, ch, k), &dev)?;
        let want = x
            .pad_with_zeros(2, (k - 1) * dil, 0)?
            .conv1d(&w, 0, 1, dil, 1)?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;

        // Tap-major weight: [cout, cin, k] -> [cout, k, cin] -> [cout, k * cin].
        let w_tapmajor = w.permute((0, 2, 1))?.contiguous()?.reshape((ch, k * ch))?;
        let plen = len + (k - 1) * dil;

        let build_index = |dev: &Device| -> candle_core::Result<Tensor> {
            let c = Tensor::arange(0u32, ch as u32, dev)?.reshape((ch, 1, 1))?;
            let t = Tensor::arange(0u32, k as u32, dev)?.reshape((1, k, 1))?;
            let i = Tensor::arange(0u32, len as u32, dev)?.reshape((1, 1, len))?;
            let idx = (c * plen as f64)?
                .broadcast_add(&(t * dil as f64)?)?
                .broadcast_add(&i)?;
            idx.flatten_all()
        };

        let cat_route = |x: &Tensor, wt: &Tensor| -> candle_core::Result<Tensor> {
            let xpad = x.pad_with_zeros(2, (k - 1) * dil, 0)?;
            let mut taps = Vec::with_capacity(k);
            for t in 0..k {
                taps.push(xpad.narrow(2, t * dil, len)?);
            }
            let cols = Tensor::cat(&taps, 0)?.reshape((k * ch, len))?;
            wt.matmul(&cols)?.reshape((1, ch, len))
        };
        let gather_route = |x: &Tensor, wm: &Tensor, dev: &Device| -> candle_core::Result<Tensor> {
            let xpad = x.pad_with_zeros(2, (k - 1) * dil, 0)?;
            let idx = build_index(dev)?;
            let cols = xpad
                .flatten_all()?
                .index_select(&idx, 0)?
                .reshape((ch * k, len))?;
            wm.matmul(&cols)?.reshape((1, ch, len))
        };

        let wm = w.reshape((ch, ch * k))?;
        let dc = tts_nn::max_abs_diff(&cat_route(&x, &w_tapmajor)?, &want)? / scale;
        let dg = tts_nn::max_abs_diff(&gather_route(&x, &wm, &dev)?, &want)? / scale;
        println!(
            "\n{ch}ch @ {len}: cat-route rel {dc:.2e}, gather-route rel {dg:.2e}  ({})",
            if dc < 1e-5 && dg < 1e-5 {
                "both exact"
            } else {
                "MISMATCH"
            }
        );

        let (x1, w1) = (x.clone(), w.clone());
        let (x2, wt2) = (x.clone(), w_tapmajor.clone());
        let (x3, wm3, d3) = (x.clone(), wm.clone(), dev.clone());
        h.ab(
            &format!("full conv route, {ch}ch @ {len}"),
            &mut [
                ("candle conv1d (direct)", &mut move || {
                    let _ = x1
                        .pad_with_zeros(2, (k - 1) * dil, 0)?
                        .conv1d(&w1, 0, 1, dil, 1)?;
                    Ok(())
                }),
                ("cat(dim=0) + GEMM, tap-major weight", &mut move || {
                    let _ = cat_route(&x2, &wt2)?;
                    Ok(())
                }),
                ("device-built index + gather + GEMM", &mut move || {
                    let _ = gather_route(&x3, &wm3, &d3)?;
                    Ok(())
                }),
            ],
        )?;
    }

    h.report_drift()?;
    Ok(())
}
