//! How much does a candle/Metal op *cost to issue*, independent of its work?
//!
//! `arloop` decodes 2.97 s of audio in 11.5 s — RTF 3.87 — while doing only a
//! few GFLOP. That cannot be compute. Each candle op on Metal takes a mutex,
//! creates a fresh `MTLComputeCommandEncoder`, encodes one dispatch, and every
//! `CANDLE_METAL_COMPUTE_PER_BUFFER` (default 50) ops commits the command
//! buffer. This measures the floor that imposes.
//!
//! Also A/B-able across processes via the env var, using the harness canary to
//! normalise for thermal state:
//!
//!   for n in 10 50 200 1000; do CANDLE_METAL_COMPUTE_PER_BUFFER=$n \
//!     cargo run -q --release --bin dispatch; done
//!
//! Run:  cargo run -p a8-probe --release --bin dispatch

use a8_probe::bench::Harness;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const N: usize = 4000;
const SAMPLES: usize = 5;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let per_buffer =
        std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").unwrap_or("50 (default)".into());
    println!("dispatch probe — {N} ops per variant, CANDLE_METAL_COMPUTE_PER_BUFFER={per_buffer}");

    let mut h = Harness::new(&dev, SAMPLES)?;

    let small = Tensor::randn(0f32, 1.0, (1, 896), &dev)?;
    let w896 = Tensor::randn(0f32, 0.02, (896, 896), &dev)?;
    let w4864 = Tensor::randn(0f32, 0.02, (896, 4864), &dev)?;
    let big4864 = Tensor::randn(0f32, 1.0, (1, 4864), &dev)?;
    let w_back = Tensor::randn(0f32, 0.02, (4864, 896), &dev)?;
    let vocab = Tensor::randn(0f32, 0.02, (896, 155776), &dev)?;
    let cut = vocab.narrow(1, 155776 - 4097, 4097)?.contiguous()?;

    let mut f_sqr = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = small.sqr()?;
        }
        Ok(())
    };
    let mut f_add = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = (&small + &small)?;
        }
        Ok(())
    };
    let mut f_mm896 = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = small.matmul(&w896)?;
        }
        Ok(())
    };
    let mut f_mm4864 = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = small.matmul(&w4864)?;
        }
        Ok(())
    };
    let mut f_mmback = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = big4864.matmul(&w_back)?;
        }
        Ok(())
    };

    let stats = h.ab(
        "4000 issues of one tiny op — this is the dispatch floor",
        &mut [
            (
                "sqr [1,896] (0.9 kFLOP)",
                &mut f_sqr as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("add [1,896]", &mut f_add),
            ("matmul 896x896 (1.6 MFLOP)", &mut f_mm896),
            ("matmul 896x4864 (8.7 MFLOP)", &mut f_mm4864),
            ("matmul 4864x896 (8.7 MFLOP)", &mut f_mmback),
        ],
    )?;
    println!("\nper-op issue cost:");
    for s in &stats {
        println!("  {:<32} {:>7.1} us", s.name, s.median * 1000.0 / N as f64);
    }

    // The one op in the AR loop that is genuinely large. If the tiny ops above
    // cost anywhere near this, the loop is issue-bound and no amount of matmul
    // tuning matters.
    const NV: usize = 200;
    let mut f_full = || -> candle_core::Result<()> {
        for _ in 0..NV {
            let _ = small.matmul(&vocab)?;
        }
        Ok(())
    };
    let mut f_cut = || -> candle_core::Result<()> {
        for _ in 0..NV {
            let _ = small.matmul(&cut)?;
        }
        Ok(())
    };
    let stats = h.ab(
        "200 logit projections — the 38x slice claim, measured properly",
        &mut [
            (
                "896 -> 155776 (279 MFLOP)",
                &mut f_full as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("896 -> 4097 (7.3 MFLOP)", &mut f_cut),
        ],
    )?;
    println!(
        "\nlogit projection: {:.3} ms -> {:.3} ms per token ({:.2}x, not 38x — the small one is issue-bound)",
        stats[0].median / NV as f64,
        stats[1].median / NV as f64,
        stats[0].median / stats[1].median
    );

    // Does dtype change the issue cost? (It should not — same number of
    // dispatches.) If f16 helps here it is bandwidth, not math units.
    let sf = small.to_dtype(DType::F16)?;
    let wf = w4864.to_dtype(DType::F16)?;
    let mut f32_mm = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = small.matmul(&w4864)?;
        }
        Ok(())
    };
    let mut f16_mm = || -> candle_core::Result<()> {
        for _ in 0..N {
            let _ = sf.matmul(&wf)?;
        }
        Ok(())
    };
    h.ab(
        "dtype vs issue cost, matmul 896x4864",
        &mut [
            (
                "f32",
                &mut f32_mm as &mut dyn FnMut() -> candle_core::Result<()>,
            ),
            ("f16", &mut f16_mm),
        ],
    )?;

    h.report_drift()?;
    Ok(())
}
