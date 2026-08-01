//! Quantize the real AR weights with candle, dequantize, write them back out.
//!
//! The point is to answer "does quantization change the voice?" using candle's
//! *own* quantizer rather than a reimplementation, then measure the answer in the
//! reference implementation where we already have fixtures and a working codec.
//!
//! What this is and is not: candle's Metal matvec kernel dequantizes blocks on the
//! fly and accumulates in f32, so a round-trip followed by an f32 matmul is not
//! bit-identical to it — the difference is accumulation order, which is second
//! order next to the quantization error itself. As a proxy for "what do these
//! weights do to the model" it is tight.
//!
//! Scope matters more than the bit width. Only the 28 transformer layers'
//! projections are quantized: 417 M of 601 M params, and ~94% of the bytes a decode
//! step reads. Deliberately left alone:
//!
//!   - `embeddings.weight` [155776, 896] — it is both the input embedding and,
//!     tied, the logit head. Quantizing it hits token *choice* directly. And it is
//!     nearly free to keep: the logit slice reduces the head to 4097 of 155776
//!     rows, so the full-precision table costs 3.7 M params of reads per token
//!     instead of 139.6 M. The slice and this decision reinforce each other.
//!   - `codebook_embeddings`, `fast_embeddings`, `fast_output` — gathers and one
//!     small head; negligible bandwidth, direct effect on sampling.
//!   - all `*_norm.weight` and `wqkv.bias` — tiny, and RMSNorm scale errors
//!     propagate through everything.
//!
//! Run:  cargo run -p a8-probe --release --bin qroundtrip -- q8_0 out.safetensors

use anyhow::{bail, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

const IN: &str = "oracle/weights/model.safetensors";

/// Is this a transformer projection weight — the only thing we quantize?
fn is_projection(name: &str) -> bool {
    let layer = name.starts_with("layers.") || name.starts_with("fast_layers.");
    let proj = name.ends_with("attention.wqkv.weight")
        || name.ends_with("attention.wo.weight")
        || name.ends_with("feed_forward.w1.weight")
        || name.ends_with("feed_forward.w2.weight")
        || name.ends_with("feed_forward.w3.weight");
    layer && proj
}

fn parse_dtype(s: &str) -> Result<GgmlDType> {
    Ok(match s {
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        // The K-quants need k divisible by 256 and dim is 896; they will fail
        // loudly rather than silently, but say so up front.
        "q4_K" | "q5_K" | "q6_K" => {
            bail!("{s} uses 256-element blocks; every k=896 projection fails. Use a block-32 type.")
        }
        other => bail!("unknown quant type {other}"),
    })
}

/// Relative Frobenius error, ||W - W'|| / ||W||.
fn rel_err(a: &Tensor, b: &Tensor) -> Result<f64> {
    let num = (a - b)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let den = a.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((num / den).sqrt())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let qname = args.next().unwrap_or_else(|| "q8_0".into());
    let out = args
        .next()
        .unwrap_or_else(|| format!("fixtures/ar_{qname}.safetensors"));
    let q = parse_dtype(&qname)?;
    let cpu = Device::Cpu;

    println!("loading {IN}");
    let src = candle_core::safetensors::load(IN, &cpu)?;
    println!(
        "{} tensors; quantizing projections to {qname} ({:.2} bytes/param)",
        src.len(),
        q.type_size() as f64 / q.block_size() as f64
    );

    let mut outmap: HashMap<String, Tensor> = HashMap::new();
    let mut errs: Vec<(String, f64)> = Vec::new();
    let mut q_params = 0usize;
    let mut kept_params = 0usize;

    let mut names: Vec<&String> = src.keys().collect();
    names.sort();
    for name in names {
        // Everything goes out as f32 so the comparison isolates quantization from
        // the bf16 the checkpoint ships in.
        let t = src[name].to_dtype(DType::F32)?;
        if is_projection(name) {
            let qt = QTensor::quantize(&t, q)?;
            let deq = qt.dequantize(&cpu)?;
            errs.push((name.clone(), rel_err(&t, &deq)?));
            q_params += t.elem_count();
            outmap.insert(name.clone(), deq);
        } else {
            kept_params += t.elem_count();
            outmap.insert(name.clone(), t);
        }
    }

    let total = q_params + kept_params;
    println!(
        "quantized {} params ({:.1}%), kept {} at f32 ({:.1}%)",
        q_params,
        100.0 * q_params as f64 / total as f64,
        kept_params,
        100.0 * kept_params as f64 / total as f64
    );

    errs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let vals: Vec<f64> = errs.iter().map(|(_, e)| *e).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    println!("\nrelative weight error over {} projections:", errs.len());
    println!(
        "  mean {:.4}  min {:.4}  max {:.4}",
        mean,
        vals[vals.len() - 1],
        vals[0]
    );
    println!("  worst 5:");
    for (n, e) in errs.iter().take(5) {
        println!("    {e:.4}  {n}");
    }

    candle_core::safetensors::save(&outmap, &out)?;
    println!("\nwrote {out}");
    Ok(())
}
