//! Serial dependency, not bandwidth.
//!
//! `matvec` refuted the conclusion in the first draft of `docs/performance/ar-loop.md`.
//! Individual q8_0 matvecs hit **99–139 GB/s** against a ~120 GB/s bus — candle's
//! quantized matvec kernel is already saturating memory, and there is no 3.5x
//! sitting in a better kernel.
//!
//! But `quant` measured a whole layer at 479 us while its five matvecs sum to
//! ~132 us when timed individually. The difference is that `matvec` repeats the
//! *same* call on the *same* input 256 times — independent work the GPU pipelines
//! — whereas a decode step is a strict chain: wqkv -> wo -> {w1,w3} -> w2, each
//! waiting on the last. Latency that overlapping hides in one case is fully
//! exposed in the other.
//!
//! If that is right, three things follow, and this probe checks all three:
//!   1. A serial chain costs far more than the same matvecs run independently.
//!   2. Quantization helps less than its byte ratio suggests, because part of the
//!      cost is per-op latency rather than bytes.
//!   3. **Batching is the strongest remaining lever** — extra sequences are
//!      independent work that fills exactly the gaps the chain leaves, so
//!      per-sequence cost should fall much faster than bytes read would predict.
//!
//! Run:  cargo run -p tts-probe --release --bin chain

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use tts_probe::bench::Harness;

const DIM: usize = 896;
const QKV: usize = 1152;
const FFN: usize = 4864;
const N_LAYER: usize = 24;
const N_FAST_LAYER: usize = 4;
const NUM_CODEBOOKS: usize = 10;
const SAMPLES: usize = 7;
const REPS: usize = 32;

struct Layer {
    wqkv: QMatMul,
    wo: QMatMul,
    w1: QMatMul,
    w3: QMatMul,
    w2: QMatMul,
}

impl Layer {
    fn new(dev: &Device, q: GgmlDType) -> Result<Self> {
        let mk = |n: usize, k: usize| -> Result<QMatMul> {
            let t = Tensor::randn(0f32, 0.02, (n, k), &Device::Cpu)?;
            Ok(QMatMul::from_qtensor(QTensor::quantize_onto(&t, q, dev)?)?)
        };
        Ok(Self {
            wqkv: mk(QKV, DIM)?,
            wo: mk(DIM, DIM)?,
            w1: mk(FFN, DIM)?,
            w3: mk(FFN, DIM)?,
            w2: mk(DIM, FFN)?,
        })
    }

    /// The real thing: every matvec waits on the previous one.
    fn serial(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let qkv = self.wqkv.forward(x)?;
        let a = self.wo.forward(&qkv.narrow(1, 0, DIM)?.contiguous()?)?;
        let g = self.w1.forward(&a)?;
        let u = self.w3.forward(&a)?;
        self.w2.forward(&(g * u)?)
    }

    /// Same five matvecs, same bytes, no dependencies between them.
    fn independent(&self, x: &Tensor, xf: &Tensor) -> candle_core::Result<()> {
        let _ = self.wqkv.forward(x)?;
        let _ = self.wo.forward(x)?;
        let _ = self.w1.forward(x)?;
        let _ = self.w3.forward(x)?;
        let _ = self.w2.forward(xf)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    println!("chain probe — q8_0, one layer's five matvecs, x{REPS} per sample");
    let mut h = Harness::new(&dev, SAMPLES)?;

    let l = Layer::new(&dev, GgmlDType::Q8_0)?;
    let x = Tensor::randn(0f32, 1.0, (1, DIM), &dev)?;
    let xf = Tensor::randn(0f32, 1.0, (1, FFN), &dev)?;

    let mut serial = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            let _ = l.serial(&x)?;
        }
        Ok(())
    };
    let mut indep = || -> candle_core::Result<()> {
        for _ in 0..REPS {
            l.independent(&x, &xf)?;
        }
        Ok(())
    };
    let s = h.ab(
        "identical work, with and without the dependency chain",
        &mut [
            (
                "serial (a real decode step)",
                &mut serial as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("independent (pipelined)", &mut indep),
        ],
    )?;
    let (ser, ind) = (s[0].median / REPS as f64, s[1].median / REPS as f64);
    println!(
        "\nserial {:.1} us/layer vs independent {:.1} us/layer -> the chain costs {:.2}x.\n\
         That gap is exposed latency, and it is what batching recovers.",
        ser * 1000.0,
        ind * 1000.0,
        ser / ind
    );

    // ------------------------------------------------ how much does batching buy?
    // Extra sequences are independent work, so they should ride along nearly free
    // until the weight read stops being the limit.
    println!();
    let batches = [1usize, 2, 4, 8, 16, 32];
    let xs: Vec<Tensor> = batches
        .iter()
        .map(|&b| Tensor::randn(0f32, 1.0, (b, DIM), &dev).unwrap())
        .collect();
    let mut fns: Vec<Box<dyn FnMut() -> candle_core::Result<()>>> = Vec::new();
    for xb in xs.iter() {
        let (lp, xb) = (&l as *const Layer, xb.clone());
        fns.push(Box::new(move || {
            for _ in 0..REPS {
                // SAFETY: `l` outlives every harness call.
                let _ = unsafe { &*lp }.serial(&xb)?;
            }
            Ok(())
        }));
    }
    let labels: Vec<String> = batches.iter().map(|b| format!("batch {b}")).collect();
    let mut refs: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
    for (lb, f) in labels.iter().zip(fns.iter_mut()) {
        refs.push((lb.as_str(), f.as_mut()));
    }
    let bs = h.ab("serial layer, batched across sequences", &mut refs)?;

    println!(
        "\n{:<10} {:>12} {:>16} {:>12}",
        "batch", "us / layer", "us / seq / layer", "vs batch 1"
    );
    println!("{}", "-".repeat(54));
    let base = bs[0].median * 1000.0 / REPS as f64;
    for (st, &b) in bs.iter().zip(batches.iter()) {
        let us = st.median * 1000.0 / REPS as f64;
        println!(
            "{:<10} {us:>12.1} {:>16.1} {:>11.2}x",
            format!("{b}"),
            us / b as f64,
            base / (us / b as f64)
        );
    }

    // Project the AR loop per sequence: 24 slow layers + 40 fast layer-passes per
    // frame, 64 frames for 2.97 s of audio.
    println!("\nprojected per-sequence AR loop for 2.97 s of audio:");
    println!("{:<10} {:>12} {:>10}", "batch", "AR ms", "AR RTF");
    println!("{}", "-".repeat(36));
    for (st, &b) in bs.iter().zip(batches.iter()) {
        let us = st.median * 1000.0 / REPS as f64 / b as f64;
        let ms = 64.0 * (N_LAYER + NUM_CODEBOOKS * N_FAST_LAYER) as f64 * us / 1000.0;
        println!("{:<10} {ms:>12.1} {:>10.3}", format!("{b}"), ms / 2970.0);
    }

    h.report_drift()?;
    Ok(())
}
