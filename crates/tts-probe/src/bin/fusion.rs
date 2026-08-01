//! How much is candle losing to unfused elementwise kernels, and what is the
//! ceiling if they were fused?
//!
//! `snake alone, 96ch @ 131072` measured 11.69 ms at an effective 9 GB/s on a
//! machine with ~120 GB/s. Snake is elementwise: one read, one write, ~0.8 ms if
//! fused. The suspicion is that each candle op is a separate Metal dispatch with
//! its own full round-trip to device memory, so a 5-op expression pays 5x the
//! traffic. This measures each pass in isolation to confirm, and establishes the
//! real achievable bandwidth so the fused target is grounded rather than assumed.
//!
//! Run:  cargo run -p tts-probe --release --bin fusion

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const ITERS: usize = 5;
const CH: usize = 96;
const LEN: usize = 131072;

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

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let dt = DType::F32;
    let elems = CH * LEN;
    let mb = (elems * 4) as f64 / 1e6;

    let x = Tensor::randn(0f32, 1.0, (1, CH, LEN), &dev)?;
    let y = Tensor::randn(0f32, 1.0, (1, CH, LEN), &dev)?;
    let alpha = Tensor::ones((1, CH, 1), dt, &dev)?;
    let recip = (&alpha + 1e-9)?.recip()?;

    println!("tensor: [1, {CH}, {LEN}] = {elems} elems = {mb:.1} MB\n");

    // Each of these is one Metal dispatch: read ~50 MB (or 100 for binary), write 50.
    let single = [
        (
            "affine (1 read 1 write)",
            time(&dev, || x.affine(2.0, 1.0))?,
        ),
        ("sin", time(&dev, || x.sin())?),
        ("sqr", time(&dev, || x.sqr())?),
        (
            "broadcast_mul by [1,C,1]",
            time(&dev, || x.broadcast_mul(&alpha))?,
        ),
        ("add (2 reads 1 write)", time(&dev, || &x + &y)?),
        (
            "copy / contiguous",
            time(&dev, || x.t()?.t()?.contiguous())?,
        ),
    ];

    println!("{:<30} {:>9} {:>14}", "single op", "ms", "GB/s");
    println!("{}", "-".repeat(56));
    let mut best_bw: f64 = 0.0;
    for (name, ms) in &single {
        // 2 streams for unary (read+write), 3 for binary add.
        let streams = if name.starts_with("add") { 3.0 } else { 2.0 };
        let bw = mb * streams / (ms / 1000.0) / 1e3;
        best_bw = best_bw.max(bw);
        println!("{name:<30} {ms:>9.3} {bw:>13.0}");
    }
    println!("\nbest observed elementwise bandwidth: {best_bw:.0} GB/s");

    // ------------------------------------------------------------- snake, as-is
    let snake_ms = time(&dev, || {
        &x + x
            .broadcast_mul(&alpha)?
            .sin()?
            .sqr()?
            .broadcast_mul(&recip)?
    })?;
    let sum_of_parts: f64 = single[1].1 + single[2].1 + single[3].1 * 2.0 + single[4].1;
    println!("\nsnake as 5 candle ops : {snake_ms:.3} ms");
    println!("sum of those 5 alone  : {sum_of_parts:.3} ms   <- if unfused, these match");

    // A single fused kernel would move 1 read + 1 write, i.e. the cost of `affine`.
    let fused_target = single[0].1;
    println!("one fused pass (=affine): {fused_target:.3} ms");
    println!(
        "=> fusion headroom on snake: {:.1}x",
        snake_ms / fused_target
    );

    // -------------------------------------- whole-decoder elementwise accounting
    // Snake instances in the decode path, as (channels, length):
    //   4 decoder blocks x 3 residual units x 2 snakes, at the block's output width
    //   + 1 pre-convT snake per block, at the block's input width
    //   + 1 tail snake
    let residual_stages = [
        (768usize, 2048usize),
        (384, 16384),
        (192, 65536),
        (96, 131072),
    ];
    let pre_convt = [
        (1536usize, 256usize),
        (768, 2048),
        (384, 16384),
        (192, 65536),
    ];

    // Cost model: elementwise time scales with element count at the measured rate.
    let per_elem_unfused = snake_ms / elems as f64;
    let per_elem_fused = fused_target / elems as f64;

    let mut unfused_total = 0.0;
    let mut fused_total = 0.0;
    for (c, l) in residual_stages {
        unfused_total += 6.0 * (c * l) as f64 * per_elem_unfused;
        fused_total += 6.0 * (c * l) as f64 * per_elem_fused;
    }
    for (c, l) in pre_convt {
        unfused_total += (c * l) as f64 * per_elem_unfused;
        fused_total += (c * l) as f64 * per_elem_fused;
    }
    unfused_total += (CH * LEN) as f64 * per_elem_unfused;
    fused_total += (CH * LEN) as f64 * per_elem_fused;

    println!("\n--- all 29 snake instances in the decode path ---");
    println!("unfused (candle ops today) : {unfused_total:7.1} ms");
    println!("fused (1 pass each)        : {fused_total:7.1} ms");
    println!(
        "saving                     : {:7.1} ms",
        unfused_total - fused_total
    );

    println!("\nmeasured full cascade today: 990.0 ms");
    println!(
        "with snake fused           : {:.1} ms",
        990.0 - (unfused_total - fused_total)
    );
    println!("torch/mps fp16 to beat     : 403.6 ms");

    // ------------------------------------------- alpha folding: no kernel needed
    // snake(x) = x + a^-1 sin^2(ax).  Let u = ax:
    //     snake(x) = u/a + a^-1 sin^2(u) = a^-1 (u + sin^2 u)
    // So `a` can be folded into the PRECEDING conv's output weights (and bias),
    // and `a^-1` into the FOLLOWING conv's input-channel weights -- both offline,
    // both exact. Every snake in this decoder sits between two convs, so this
    // removes BOTH broadcast_muls and leaves 3 plain unary passes.
    println!("\n--- alpha folding (offline weight transform, zero custom kernels) ---");
    let u = x.broadcast_mul(&alpha)?.contiguous()?; // stands in for the folded conv output
    let folded_ms = time(&dev, || &u + u.sin()?.sqr()?)?;
    println!("snake, 5 ops with broadcasts : {snake_ms:.3} ms");
    println!(
        "snake, 3 unary ops (folded)  : {folded_ms:.3} ms   -> {:.2}x",
        snake_ms / folded_ms
    );

    // Verify the algebra numerically: a^-1 (u + sin^2 u) must equal the original.
    let original = (&x
        + x.broadcast_mul(&alpha)?
            .sin()?
            .sqr()?
            .broadcast_mul(&recip)?)?;
    let refolded = (&u + u.sin()?.sqr()?)?.broadcast_mul(&recip)?;
    let diff = (&original - &refolded)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let scale = original.abs()?.max_all()?.to_scalar::<f32>()?;
    println!(
        "algebra check: max|diff| {diff:.3e} (scale {scale:.3e}) {}",
        if diff < 1e-4 { "EXACT" } else { "MISMATCH" }
    );

    let folded_total = unfused_total * (folded_ms / snake_ms);
    println!("\nall 29 snakes, folded      : {folded_total:.1} ms (from {unfused_total:.1})");
    println!(
        "cascade with folding only  : {:.1} ms",
        990.0 - (unfused_total - folded_total)
    );
    println!(
        "cascade with folding+fusion: {:.1} ms",
        990.0 - (unfused_total - fused_total)
    );

    Ok(())
}
