//! Where the flow decoder's time goes — at the length the engine actually runs.
//!
//! Correcting the engine's stage timers to synchronize before recording moved the picture
//! completely: the flow is **68.5% of CosyVoice**, not 52%, and the vocoder is 4.4%, not
//! 17%. So the flow is the only thing worth optimising, and the existing `cosy-bench`
//! profiles it on the 798-frame fixture while the engine runs **3222** (2634 target plus
//! 588 prompt frames).
//!
//! That difference is not a detail. Attention is quadratic, so going 798 -> 3222 multiplies
//! its share by ~16 relative to the projections. Whatever `cosy-bench` says the split is,
//! it is not the split that matters.
//!
//! Run: `cargo run -p a8-probe --release --bin flowsplit`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosy::flow::Flow;
use std::collections::HashMap;
use tts_bench::Harness;

const FIXTURES: &str = "fixtures-cosy/oracle.safetensors";
const NOISE_ASSET: &str = "fixtures-cosy/rand_noise.safetensors";
const WEIGHTS: &str = "oracle-cosy/weights";

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;
    println!("canary {:.2} ms\n", h.canary()?);

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &dev)
        .with_context(|| format!("loading {FIXTURES}"))?;
    let get = |n: &str| -> Result<Tensor> {
        Ok(fx
            .get(n)
            .with_context(|| format!("missing {n}"))?
            .to_dtype(DType::F32)?)
    };

    let flow = Flow::load(&format!("{WEIGHTS}/flow.safetensors"), NOISE_ASSET, &dev)?;
    let (mu0, cond0, spks) = (get("flow.mu")?, get("flow.cond")?, get("flow.spks")?);
    let z0 = get("flow.z")?;

    // Tile along the frame axis to reach the engine's length. The DiT is length-agnostic,
    // so tiled content exercises exactly the same kernels at the right sizes.
    for tiles in [1usize, 4] {
        let mu = Tensor::cat(&vec![&mu0; tiles], 2)?.contiguous()?;
        let cond = Tensor::cat(&vec![&cond0; tiles], 2)?.contiguous()?;
        let z = Tensor::cat(&vec![&z0; tiles], 2)?.contiguous()?;
        let n = mu.dim(2)?;

        println!("--- {n} frames ---");
        let (f1, f2, f3) = (&flow, &flow, &flow);
        let (za, mua, ca, sa) = (&z, &mu, &cond, &spks);
        let mut estimate = || -> candle_core::Result<()> {
            f1.estimate(za, mua, ca, sa, 0.3)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            Ok(())
        };
        let mut embed = || -> candle_core::Result<()> {
            f2.embed_only(za, mua, ca, sa)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            Ok(())
        };
        let mut solve = || -> candle_core::Result<()> {
            f3.solve(mua, ca, sa)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            Ok(())
        };
        let stats = h.ab(
            &format!("flow @ {n}"),
            &mut [
                ("one estimate", &mut estimate),
                ("embed (input + pos conv)", &mut embed),
                ("full 10-step solve", &mut solve),
            ],
        )?;

        let (est, emb, sol) = (stats[0].median, stats[1].median, stats[2].median);
        let blocks = est - emb;
        println!(
            "  estimate {est:.1} ms = embed {emb:.1} ({:.0}%) + 22 blocks {blocks:.1} ({:.0}%)",
            100.0 * emb / est,
            100.0 * blocks / est
        );
        println!("  per block {:.2} ms", blocks / 22.0);
        println!(
            "  solve {sol:.1} ms  vs  10 x estimate = {:.1} ms  ({:.0}% solver overhead)\n",
            10.0 * est,
            100.0 * (sol - 10.0 * est) / sol
        );
    }

    h.report_drift()?;
    Ok(())
}
