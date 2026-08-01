//! The AR loop is weight-bandwidth bound. So shrink the weights.
//!
//! `dispatch` showed candle's op-issue cost is only ~9 us, but a
//! `[1,896] x [896,4864]` matmul takes 313 us for 8.7 MFLOP — 28 GFLOPS, i.e.
//! nowhere near compute-bound. What it *is* near is the weight read: 17.4 MB of
//! f32 at ~120 GB/s is 145 us. Every matmul in a batch-1 decode step reads the
//! whole weight and uses each element once. The loop is a memory test.
//!
//! That makes the dominant lever obvious and it is not kernel fusion: fewer bytes
//! per weight. candle ships ggml-style quantized matmul with a dedicated
//! matrix-vector Metal kernel (`quantized/metal.rs::fwd_mv`, taken whenever
//! `dim(-2) == 1` — exactly a decode step), so this is an existing, tested path
//! rather than hand-written MSL.
//!
//! Measures one slow-AR layer's worth of matmuls (wqkv, wo, w1, w3, w2 =
//! 14.9 M params) per dtype, then projects the whole DualAR loop.
//!
//! Run:  cargo run -p a8-probe --release --bin quant

use a8_probe::bench::Harness;
use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

const DIM: usize = 896;
const QKV: usize = 1152;
const FFN: usize = 4864;
const N_LAYER: usize = 24;
const N_FAST_LAYER: usize = 4;
const NUM_CODEBOOKS: usize = 10;
const VOCAB: usize = 155776;
const REACHABLE: usize = 4097;
const SAMPLES: usize = 7;
/// Repeats per timed sample: one layer is ~1 ms, too short to time cleanly.
const REPS: usize = 64;

/// Params in one layer's five projections. Same for slow and fast (fast_dim == dim).
const LAYER_PARAMS: usize = DIM * QKV + DIM * DIM + 2 * (DIM * FFN) + FFN * DIM;

/// A layer's projections as either dense or quantized matmuls.
enum Proj {
    Dense(Vec<Tensor>),
    Quant(Vec<QMatMul>),
}

impl Proj {
    /// One layer's forward, in dependency order so nothing can be elided.
    fn run(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            // Dense weights are stored [k, n] (pre-transposed, as a port would).
            Proj::Dense(w) => {
                let qkv = x.matmul(&w[0])?;
                let a = qkv.narrow(1, 0, DIM)?.contiguous()?.matmul(&w[1])?;
                let g = a.matmul(&w[2])?;
                let u = a.matmul(&w[3])?;
                (g * u)?.matmul(&w[4])
            }
            // QMatMul weights are stored [n, k]; forward takes [.., k] -> [.., n].
            Proj::Quant(w) => {
                let qkv = w[0].forward(x)?;
                let a = qkv.narrow(1, 0, DIM)?.contiguous()?;
                let a = w[1].forward(&a)?;
                let g = w[2].forward(&a)?;
                let u = w[3].forward(&a)?;
                w[4].forward(&(g * u)?)
            }
        }
    }
}

fn dense(dev: &Device, dt: DType) -> Result<Proj> {
    let r = |a, b| -> Result<Tensor> { Ok(Tensor::randn(0f32, 0.02, (a, b), dev)?.to_dtype(dt)?) };
    Ok(Proj::Dense(vec![
        r(DIM, QKV)?,
        r(DIM, DIM)?,
        r(DIM, FFN)?,
        r(DIM, FFN)?,
        r(FFN, DIM)?,
    ]))
}

fn quant(dev: &Device, q: GgmlDType) -> Result<Proj> {
    // quantize_onto requires a CPU source, then writes the quantized blocks
    // straight to the device.
    let cpu = Device::Cpu;
    let mk = |n: usize, k: usize| -> Result<QMatMul> {
        let t = Tensor::randn(0f32, 0.02, (n, k), &cpu)?;
        Ok(QMatMul::from_qtensor(QTensor::quantize_onto(&t, q, dev)?)?)
    };
    Ok(Proj::Quant(vec![
        mk(QKV, DIM)?,
        mk(DIM, DIM)?,
        mk(FFN, DIM)?,
        mk(FFN, DIM)?,
        mk(DIM, FFN)?,
    ]))
}

/// Bytes per weight element for a ggml block type.
fn bpw(q: GgmlDType) -> f64 {
    q.type_size() as f64 / q.block_size() as f64
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    println!(
        "quant probe — one slow-AR layer's projections ({LAYER_PARAMS} params), x{REPS} per sample"
    );

    let mut h = Harness::new(&dev, SAMPLES)?;

    let x32 = Tensor::randn(0f32, 1.0, (1, DIM), &dev)?;
    let x16 = x32.to_dtype(DType::F16)?;

    let p_f32 = dense(&dev, DType::F32)?;
    let p_f16 = dense(&dev, DType::F16)?;
    // NOTE: the K-quants (Q4K/Q5K/Q6K) are *unavailable for this model*. They use
    // 256-element blocks and every projection with k = dim = 896 fails
    // `last dim divisible by block size` (896 = 3.5 x 256). Only the block-32
    // legacy types apply — which costs some quality per bit, and is worth knowing
    // before anyone plans around q4_K.
    let p_q8 = quant(&dev, GgmlDType::Q8_0)?;
    let p_q5 = quant(&dev, GgmlDType::Q5_0)?;
    let p_q41 = quant(&dev, GgmlDType::Q4_1)?;
    let p_q40 = quant(&dev, GgmlDType::Q4_0)?;

    let mk = |p: &Proj, x: &Tensor| {
        let (p, x) = (p as *const Proj, x.clone());
        move || -> candle_core::Result<()> {
            for _ in 0..REPS {
                // SAFETY: `p` outlives the harness call; the Proj values live in
                // main's frame for the whole run.
                let _ = unsafe { &*p }.run(&x)?;
            }
            Ok(())
        }
    };
    let mut f1 = mk(&p_f32, &x32);
    let mut f2 = mk(&p_f16, &x16);
    let mut f3 = mk(&p_q8, &x32);
    let mut f4 = mk(&p_q5, &x32);
    let mut f5 = mk(&p_q41, &x32);
    let mut f6 = mk(&p_q40, &x32);

    let stats = h.ab(
        "one layer's projections, batch 1",
        &mut [
            (
                "f32 dense",
                &mut f1 as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("f16 dense", &mut f2),
            ("q8_0", &mut f3),
            ("q5_0", &mut f4),
            ("q4_1", &mut f5),
            ("q4_0", &mut f6),
        ],
    )?;

    // Per layer, and the bandwidth each implies. If achieved GB/s is flat across
    // dtypes, the loop is bandwidth-bound and bytes-per-weight is the whole game.
    let kinds: [(f64, f64); 6] = [
        (4.0, 0.0),
        (2.0, 0.0),
        (bpw(GgmlDType::Q8_0), 0.0),
        (bpw(GgmlDType::Q5_0), 0.0),
        (bpw(GgmlDType::Q4_1), 0.0),
        (bpw(GgmlDType::Q4_0), 0.0),
    ];
    println!("\nper layer, and the bandwidth it implies:");
    println!(
        "{:<14} {:>8} {:>12} {:>10} {:>10}",
        "variant", "B/param", "us / layer", "MB read", "GB/s"
    );
    println!("{}", "-".repeat(60));
    let mut per_layer = Vec::new();
    for (s, (b, _)) in stats.iter().zip(kinds.iter()) {
        let us = s.median * 1000.0 / REPS as f64;
        let mb = LAYER_PARAMS as f64 * b / 1e6;
        println!(
            "{:<14} {b:>8.2} {us:>12.1} {mb:>10.2} {:>10.0}",
            s.name,
            mb / 1e3 / (us / 1e6)
        );
        per_layer.push((s.name.clone(), us, *b));
    }

    // The logit head separately: it is 139.6 M params, ~9x a whole layer, and the
    // semantic mask makes 151679 of its 155776 rows unreachable.
    let head_f32 = Tensor::randn(0f32, 0.02, (DIM, VOCAB), &dev)?;
    let cut_f32 = head_f32
        .narrow(1, VOCAB - REACHABLE, REACHABLE)?
        .contiguous()?;
    let cpu = Device::Cpu;
    let head_q4 = QMatMul::from_qtensor(QTensor::quantize_onto(
        &Tensor::randn(0f32, 0.02, (VOCAB, DIM), &cpu)?,
        GgmlDType::Q4_0,
        &dev,
    )?)?;
    let cut_q4 = QMatMul::from_qtensor(QTensor::quantize_onto(
        &Tensor::randn(0f32, 0.02, (REACHABLE, DIM), &cpu)?,
        GgmlDType::Q4_0,
        &dev,
    )?)?;
    const HREPS: usize = 16;
    let mut g1 = || -> candle_core::Result<()> {
        for _ in 0..HREPS {
            let _ = x32.matmul(&head_f32)?;
        }
        Ok(())
    };
    let mut g2 = || -> candle_core::Result<()> {
        for _ in 0..HREPS {
            let _ = x32.matmul(&cut_f32)?;
        }
        Ok(())
    };
    let mut g3 = || -> candle_core::Result<()> {
        for _ in 0..HREPS {
            let _ = head_q4.forward(&x32)?;
        }
        Ok(())
    };
    let mut g4 = || -> candle_core::Result<()> {
        for _ in 0..HREPS {
            let _ = cut_q4.forward(&x32)?;
        }
        Ok(())
    };
    let hs = h.ab(
        "logit head, batch 1",
        &mut [
            (
                "f32 full 155776",
                &mut g1 as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("f32 sliced 4097", &mut g2),
            ("q4_0 full", &mut g3),
            ("q4_0 sliced", &mut g4),
        ],
    )?;
    let head_us: Vec<f64> = hs
        .iter()
        .map(|s| s.median * 1000.0 / HREPS as f64)
        .collect();

    // ------------------------------------------------------------- projection
    // Per frame the loop runs: 1 slow token (24 layers + 1 head) and, for the
    // fast AR, 10 positions x 4 layers (+10 small heads, ignored — 3.7 M params
    // each, under 3% of a layer).
    println!("\nprojected AR loop for 2.97 s of audio (64 frames at 21.53 Hz):");
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>9}",
        "weights", "slow ms", "fast ms", "total ms", "AR RTF"
    );
    println!("{}", "-".repeat(58));
    for (i, (name, us, _)) in per_layer.iter().enumerate() {
        // Pair each weight format with the matching head cost: dense formats get
        // the sliced f32 head, quantized get the sliced q4_K head.
        let head = if i < 2 { head_us[1] } else { head_us[3] };
        let slow = 64.0 * (N_LAYER as f64 * us + head) / 1000.0;
        let fast = 64.0 * (NUM_CODEBOOKS * N_FAST_LAYER) as f64 * us / 1000.0;
        println!(
            "{name:<14} {slow:>10.1} {fast:>10.1} {:>10.1} {:>9.3}",
            slow + fast,
            (slow + fast) / 2970.0
        );
    }
    println!(
        "\n(the fast AR runs {} layer-passes per frame vs the slow AR's {} — it is the\n \
         larger half of the AR loop, and it was never in any earlier estimate)",
        NUM_CODEBOOKS * N_FAST_LAYER,
        N_LAYER
    );

    h.report_drift()?;
    Ok(())
}
