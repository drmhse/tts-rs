//! Why is quantized matvec stuck at ~33 GB/s on a ~120 GB/s bus?
//!
//! `quant` showed every format — f32, f16, q8_0, q4_1 — achieving 22–37 GB/s at
//! batch 1. A flat ceiling across formats that differ 7x in bytes is not a
//! bandwidth limit, it is under-occupancy: a matvec over an `[N, K]` weight has
//! only N independent output rows to spread across the GPU, and at N = 896 that is
//! not enough work to fill an M4.
//!
//! If that is the cause then GB/s should climb with N, and there is a free exact
//! fix available before any custom kernel: **`w1` and `w3` consume the same
//! input**, so concatenating them into one `[2*ffn, dim]` weight halves the
//! dispatches and doubles N. (`wqkv` is already fused this way in the checkpoint —
//! the reference model does this for q/k/v and not for the FFN gate.)
//!
//! Run:  cargo run -p a8-probe --release --bin matvec

use a8_probe::bench::Harness;
use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

const DIM: usize = 896;
const FFN: usize = 4864;
const SAMPLES: usize = 7;
const REPS: usize = 256;

fn qmm(dev: &Device, n: usize, k: usize, q: GgmlDType) -> Result<QMatMul> {
    let t = Tensor::randn(0f32, 0.02, (n, k), &Device::Cpu)?;
    Ok(QMatMul::from_qtensor(QTensor::quantize_onto(&t, q, dev)?)?)
}

fn bpw(q: GgmlDType) -> f64 {
    q.type_size() as f64 / q.block_size() as f64
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let q = GgmlDType::Q8_0;
    println!(
        "matvec probe — q8_0 ({:.2} B/param), {REPS} calls per sample",
        bpw(q)
    );

    let mut h = Harness::new(&dev, SAMPLES)?;
    let x = Tensor::randn(0f32, 1.0, (1, DIM), &dev)?;

    // --------------------------------------------- does GB/s climb with N?
    let ns = [DIM, 1152, FFN, 2 * FFN, 4 * FFN];
    let mats: Vec<QMatMul> = ns.iter().map(|&n| qmm(&dev, n, DIM, q).unwrap()).collect();
    let mut fns: Vec<Box<dyn FnMut() -> candle_core::Result<()>>> = Vec::new();
    for m in mats.iter() {
        let m = m as *const QMatMul;
        let x = x.clone();
        fns.push(Box::new(move || {
            for _ in 0..REPS {
                // SAFETY: `mats` outlives every harness call.
                let _ = unsafe { &*m }.forward(&x)?;
            }
            Ok(())
        }));
    }
    let labels: Vec<String> = ns.iter().map(|n| format!("[{n}, {DIM}]")).collect();
    let mut refs: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
    for (l, f) in labels.iter().zip(fns.iter_mut()) {
        refs.push((l.as_str(), f.as_mut()));
    }
    let stats = h.ab("q8_0 matvec vs output width N", &mut refs)?;

    println!(
        "\n{:<16} {:>10} {:>10} {:>10}",
        "shape", "us / call", "MB", "GB/s"
    );
    println!("{}", "-".repeat(50));
    for (s, &n) in stats.iter().zip(ns.iter()) {
        let us = s.median * 1000.0 / REPS as f64;
        let mb = (n * DIM) as f64 * bpw(q) / 1e6;
        println!(
            "{:<16} {us:>10.1} {mb:>10.2} {:>10.0}",
            s.name,
            mb / 1e3 / (us / 1e6)
        );
    }

    // ------------------------- the free win: fuse w1 and w3 into one dispatch
    let w1 = qmm(&dev, FFN, DIM, q)?;
    let w3 = qmm(&dev, FFN, DIM, q)?;
    let w13 = qmm(&dev, 2 * FFN, DIM, q)?;
    let mut split = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            let g = w1.forward(&x)?;
            let u = w3.forward(&x)?;
            let _ = (g * u)?;
        }
        Ok(())
    };
    let mut fused = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            let both = w13.forward(&x)?;
            let g = both.narrow(1, 0, FFN)?;
            let u = both.narrow(1, FFN, FFN)?;
            let _ = (g * u)?;
        }
        Ok(())
    };
    h.ab(
        "FFN gate: two matvecs vs one fused [2*ffn, dim]",
        &mut [
            (
                "w1, w3 separate",
                &mut split as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("w13 fused", &mut fused),
        ],
    )?;

    // ---------------------------------- and the k direction (w2 is [dim, ffn])
    let w2 = qmm(&dev, DIM, FFN, q)?;
    let xf = Tensor::randn(0f32, 1.0, (1, FFN), &dev)?;
    let mut narrow_n = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            let _ = w2.forward(&xf)?;
        }
        Ok(())
    };
    let wide = qmm(&dev, FFN, DIM, q)?;
    let mut wide_n = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            let _ = wide.forward(&x)?;
        }
        Ok(())
    };
    let s = h.ab(
        "same bytes, transposed: is it N or K that matters?",
        &mut [
            (
                "[896, 4864]  (narrow N)",
                &mut narrow_n as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("[4864, 896]  (wide N)", &mut wide_n),
        ],
    )?;
    println!(
        "\nboth read {:.2} MB; wide-N is {:.2}x faster -> the limit is output-row\n\
         parallelism, not bandwidth",
        (DIM * FFN) as f64 * bpw(q) / 1e6,
        s[0].median / s[1].median
    );

    h.report_drift()?;
    Ok(())
}
