//! Phase C go/no-go probe: does Candle cover the Audio8 codec decode path on
//! Metal, at the real shapes, at a usable speed?
//!
//! No weights needed -- random tensors of the correct shape are enough to answer
//! the only question that can kill the port: op coverage and per-op cost. Shapes
//! below are the actual decoder cascade for a 64-frame clip (~3.0 s of audio):
//!
//!   codes[10,64] -> post_module 1024ch/64 -> upsample x4 -> 1024ch/256
//!   -> conv 1024->1536 -> convT s8 -> 768ch/2048  -> convT s8 -> 384ch/16384
//!   -> convT s4 -> 192ch/65536 -> convT s2 -> 96ch/131072 -> conv 96->1 -> tanh
//!
//! Run:  cargo run -p a8-probe --release

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const FRAMES: usize = 64;
const ITERS: usize = 3;

struct Report {
    name: String,
    outcome: Result<(Vec<usize>, f64)>,
}

fn bench<F>(name: &str, dev: &Device, f: F) -> Report
where
    F: Fn() -> candle_core::Result<Tensor>,
{
    let outcome = (|| -> Result<(Vec<usize>, f64)> {
        // Warm-up: first call pays kernel pipeline construction.
        let warm = f()?;
        warm.device().synchronize()?;
        let shape = warm.dims().to_vec();
        drop(warm);

        let start = Instant::now();
        for _ in 0..ITERS {
            let out = f()?;
            out.device().synchronize()?;
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
        Ok((shape, ms))
    })();
    let _ = dev;
    Report { name: name.to_string(), outcome }
}

/// Snake: x + (alpha + 1e-9)^-1 * sin(alpha * x)^2, alpha shaped [1, C, 1].
fn snake(x: &Tensor, alpha: &Tensor) -> candle_core::Result<Tensor> {
    let scaled = x.broadcast_mul(alpha)?;
    let recip = (alpha + 1e-9)?.recip()?;
    x + scaled.sin()?.sqr()?.broadcast_mul(&recip)?
}

fn main() -> Result<()> {
    let dev = match Device::new_metal(0) {
        Ok(d) => {
            println!("device: Metal");
            d
        }
        Err(e) => {
            println!("device: Metal UNAVAILABLE ({e}); falling back to CPU");
            Device::Cpu
        }
    };
    let dt = DType::F32;
    println!("dtype: {dt:?}   frames: {FRAMES}   iters: {ITERS}\n");

    let mut reports = Vec::new();

    // ---------------------------------------------------------- RVQ from_codes
    // Embedding lookup (codebook dim 8) then a k=1 conv projecting 8 -> 1024.
    {
        let codes = Tensor::ones((1, FRAMES), DType::U32, &dev)?;
        let codebook = Tensor::randn(0f32, 1.0, (4096, 8), &dev)?;
        let out_proj = Tensor::randn(0f32, 0.02, (1024, 8, 1), &dev)?;
        reports.push(bench("rvq.embed+out_proj (8->1024, k1)", &dev, || {
            let e = codebook.index_select(&codes.flatten_all()?, 0)?;
            let e = e.reshape((1, FRAMES, 8))?.transpose(1, 2)?.contiguous()?;
            e.conv1d(&out_proj, 0, 1, 1, 1)
        }));
    }

    // ------------------------------------------------- post_module attention
    // 8 layers, dim 1024, 16 heads / 8 kv heads, head_dim 64, window 128.
    // One layer's attention, materialised scores (the reference uses SDPA with an
    // explicit bool mask, which lowers to this anyway).
    {
        let len = FRAMES;
        let q = Tensor::randn(0f32, 1.0, (1, 16, len, 64), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (1, 16, len, 64), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (1, 16, len, 64), &dev)?;
        let mut mask = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..len {
                // causal AND within a 128-wide window
                let lo = (i as i64 - 128 + 1).max(0) as usize;
                if j > i || j < lo {
                    mask[i * len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, len, len), &dev)?;
        reports.push(bench("post_module.attn (1024d, 16h, window 128)", &dev, || {
            let scores = (q.matmul(&k.transpose(2, 3)?)? / 8.0)?;
            let scores = scores.broadcast_add(&mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            probs.matmul(&v)
        }));
    }

    // --------------------------------------------------- upsample convT s2 x2
    {
        let x = Tensor::randn(0f32, 1.0, (1, 1024, FRAMES), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1024, 1024, 2), &dev)?;
        reports.push(bench("upsample.convT 1024->1024 k2 s2", &dev, || {
            x.conv_transpose1d(&w, 0, 0, 2, 1, 1)
        }));
    }

    // ------------------------------------------------ ConvNeXt depthwise conv
    // groups == channels. This is the second-riskiest op after convT.
    {
        let x = Tensor::randn(0f32, 1.0, (1, 1024, FRAMES * 2), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1024, 1, 7), &dev)?;
        reports.push(bench("convnext.dwconv 1024ch k7 groups=1024", &dev, || {
            x.conv1d(&w, 6, 1, 1, 1024)
        }));
    }

    // ---------------------------------------------------- decoder entry conv
    {
        let x = Tensor::randn(0f32, 1.0, (1, 1024, FRAMES * 4), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1536, 1024, 7), &dev)?;
        reports.push(bench("decoder.conv 1024->1536 k7", &dev, || {
            x.conv1d(&w, 6, 1, 1, 1)
        }));
    }

    // ------------------------------------------ the four decoder upsamplings
    // (in_ch, out_ch, stride, input_len) -- kernel is always 2*stride.
    let stages = [
        (1536usize, 768usize, 8usize, FRAMES * 4),
        (768, 384, 8, FRAMES * 32),
        (384, 192, 4, FRAMES * 256),
        (192, 96, 2, FRAMES * 1024),
    ];
    for (cin, cout, stride, len) in stages {
        let x = Tensor::randn(0f32, 1.0, (1, cin, len), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (cin, cout, 2 * stride), &dev)?;
        let name = format!("decoder.convT {cin}->{cout} k{} s{stride} @len {len}", 2 * stride);
        reports.push(bench(&name, &dev, || {
            x.conv_transpose1d(&w, 0, 0, stride, 1, 1)
        }));
    }

    // ------------------------------- residual units at each decoder resolution
    // Snake -> dilated k=7 conv -> Snake -> k=1 conv. Dilation 9 is the widest.
    for (ch, len) in [(768usize, FRAMES * 32usize), (192, FRAMES * 1024), (96, FRAMES * 2048)] {
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &dev)?;
        let alpha = Tensor::ones((1, ch, 1), dt, &dev)?;
        let w7 = Tensor::randn(0f32, 0.02, (ch, ch, 7), &dev)?;
        let w1 = Tensor::randn(0f32, 0.02, (ch, ch, 1), &dev)?;
        let name = format!("residual_unit {ch}ch dil9 @len {len}");
        reports.push(bench(&name, &dev, || {
            let h = snake(&x, &alpha)?;
            let h = h.conv1d(&w7, 54, 1, 9, 1)?;
            let h = snake(&h, &alpha)?;
            h.conv1d(&w1, 0, 1, 1, 1)
        }));
    }

    // ------------------------------------------------------ final conv + tanh
    {
        let x = Tensor::randn(0f32, 1.0, (1, 96, FRAMES * 2048), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1, 96, 7), &dev)?;
        reports.push(bench("final conv 96->1 k7 + tanh", &dev, || {
            x.conv1d(&w, 6, 1, 1, 1)?.tanh()
        }));
    }

    // ------------------------------------------------------------- slow AR bits
    // Fused wqkv matmul and a tied-embedding logit projection over 155776 rows --
    // the two heaviest per-token matmuls in the AR loop.
    {
        let x = Tensor::randn(0f32, 1.0, (1, 896), &dev)?;
        let wqkv = Tensor::randn(0f32, 0.02, (896, 1152), &dev)?;
        reports.push(bench("slow.wqkv 896->1152 (1 token)", &dev, || x.matmul(&wqkv)));

        let emb = Tensor::randn(0f32, 0.02, (896, 155776), &dev)?;
        reports.push(bench("slow.logits 896->155776 (1 token)", &dev, || x.matmul(&emb)));
    }

    // ------------------------------------------------------------ dtype sweep
    // The decoder is compute-bound, so dtype is the biggest single lever. Probe
    // the heaviest residual unit and one convT in f16/bf16 against the f32 above.
    for probe_dt in [DType::F16, DType::BF16] {
        let ch = 96usize;
        let len = FRAMES * 2048;
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &dev)?.to_dtype(probe_dt)?;
        let alpha = Tensor::ones((1, ch, 1), probe_dt, &dev)?;
        let w7 = Tensor::randn(0f32, 0.02, (ch, ch, 7), &dev)?.to_dtype(probe_dt)?;
        let w1 = Tensor::randn(0f32, 0.02, (ch, ch, 1), &dev)?.to_dtype(probe_dt)?;
        reports.push(bench(&format!("residual_unit 96ch dil9 @131072 [{probe_dt:?}]"), &dev, || {
            let h = snake(&x, &alpha)?;
            let h = h.conv1d(&w7, 54, 1, 9, 1)?;
            let h = snake(&h, &alpha)?;
            h.conv1d(&w1, 0, 1, 1, 1)
        }));

        let xt = Tensor::randn(0f32, 1.0, (1, 384, FRAMES * 256), &dev)?.to_dtype(probe_dt)?;
        let wt = Tensor::randn(0f32, 0.02, (384, 192, 8), &dev)?.to_dtype(probe_dt)?;
        reports.push(bench(&format!("decoder.convT 384->192 k8 s4 [{probe_dt:?}]"), &dev, || {
            xt.conv_transpose1d(&wt, 0, 0, 4, 1, 1)
        }));
    }

    // ------------------------------------------- depthwise conv, alternate form
    // The grouped-conv result above is anomalously slow for the work involved
    // (1024ch x k7 x len256 is only ~1.8 MMAC). Probe whether expressing the same
    // depthwise conv as 7 shifted broadcast-muls beats candle's grouped path.
    {
        let len = FRAMES * 4;
        let x = Tensor::randn(0f32, 1.0, (1, 1024, len), &dev)?;
        let w = Tensor::randn(0f32, 0.02, (1024, 1, 7), &dev)?;
        let taps: Vec<Tensor> = (0..7)
            .map(|k| w.narrow(2, k, 1).and_then(|t| t.reshape((1, 1024, 1))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        reports.push(bench("convnext.dwconv as 7 shifted muls", &dev, || {
            let padded = x.pad_with_zeros(2, 6, 0)?;
            let mut acc: Option<Tensor> = None;
            for (k, tap) in taps.iter().enumerate() {
                let slice = padded.narrow(2, k, len)?;
                let term = slice.broadcast_mul(tap)?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => (a + term)?,
                });
            }
            Ok(acc.unwrap())
        }));
    }

    // ------------------------------------------------------------------ report
    println!("{:<48} {:>22} {:>10}", "op", "output shape", "ms");
    println!("{}", "-".repeat(82));
    let mut failures = 0;
    let mut total_ms = 0.0;
    for r in &reports {
        match &r.outcome {
            Ok((shape, ms)) => {
                total_ms += ms;
                println!("{:<48} {:>22} {:>10.2}", r.name, format!("{shape:?}"), ms);
            }
            Err(e) => {
                failures += 1;
                let msg = e.to_string();
                let msg = msg.lines().next().unwrap_or("?");
                println!("{:<48} {:>33}", r.name, format!("FAILED: {}", &msg[..msg.len().min(60)]));
            }
        }
    }
    println!("{}", "-".repeat(82));
    let audio_s = (FRAMES * 2048) as f64 / 44100.0;
    println!(
        "\n{} ops, {failures} failed.  summed one-shot cost {:.1} ms for {:.2} s of audio",
        reports.len(), total_ms, audio_s
    );
    if failures == 0 {
        println!(
            "very rough codec-only RTF floor: {:.3}  (sum of probed ops / audio duration)",
            total_ms / 1000.0 / audio_s
        );
        println!("\nNOTE: this is a LOWER BOUND on one pass of the decoder, not a real RTF.");
        println!("It omits the 8 post_module layers (only 1 attn probed), 11 of 12 residual");
        println!("units, all norms, and the entire AR loop. Treat it as an op-coverage");
        println!("result with an order-of-magnitude cost hint, nothing more.");
    }
    Ok(())
}
