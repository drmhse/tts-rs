//! Does batching pay? One decode step at batch 1, 2, 4, 8, on both transformers.
//!
//! Both stages are bandwidth-bound on *weight* reads at batch 1 — the talker reads its 1.4 G
//! parameters once per frame, and the depth predictor reads its 60 M fifteen times. Batching
//! lanes amortises exactly that, but only if the quantized matmul keeps the weights quantized:
//! a path that dequantizes to f32 first turns a 1.06 byte/param read into a 4 byte/param read
//! plus an allocation, and CosyVoice measured a batch-7 q8_0 step at 5.13x a batch-1 step for
//! that reason.
//!
//! Printed as cost *per lane*: flat means the weight read amortised, rising means the batch is
//! paying for something. The answer, measured: quantized does not amortise (1.1x at batch 8),
//! dense does (f16 7.4x, still falling at 24).
//!
//! Run: `cargo run -p qwen3tts --release --bin qwen3tts-batch [q8_0|f16|f32|q4_0]`

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use qwen3tts::cfg::{predictor as pk, talker as tk};
use qwen3tts::qwen3::{Geometry, Stack};
use std::time::Instant;
use tts_nn::Weight;
use tts_nn::Weights;

const WEIGHTS: &str = "references/qwen3tts/weights/model.safetensors";
const BATCHES: [usize; 5] = [1, 4, 8, 16, 24];
/// Enough to be past warm-up without making the sweep slow.
const STEPS: usize = 20;
/// Prompt positions, matching the engine's ICL prompt.
const PROMPT: usize = 156;

fn bench(label: &str, stack: &Stack, dim: usize, device: &Device) -> Result<()> {
    println!("\n{label}");
    let mut baseline = 0f64;
    for b in BATCHES {
        let mut state = stack.new_state_with(b, PROMPT + 256)?;
        // A realistic span, not a token or two: the ICL prompt is 156 positions and a segment
        // decodes ~100 more, and every decode step copies the whole K and V span per layer. A
        // bench at span 4 said batch-8 f32 cost 65 ms/step where the real run saw 167.
        let prompt = Tensor::zeros((b, PROMPT, dim), DType::F32, device)?;
        stack.forward(&prompt, &mut state)?;
        let x = Tensor::zeros((b, 1, dim), DType::F32, device)?;

        // Warm up: the first call per shape compiles pipelines and grows the buffer pool.
        for _ in 0..3 {
            stack.forward(&x, &mut state)?;
        }
        device.synchronize()?;

        let t = Instant::now();
        for _ in 0..STEPS {
            stack.forward(&x, &mut state)?;
        }
        device.synchronize()?;
        let per_step = t.elapsed().as_secs_f64() / STEPS as f64 * 1000.0;
        let per_lane = per_step / b as f64;
        if b == 1 {
            baseline = per_lane;
        }
        println!(
            "  batch {b}: {per_step:6.2} ms/step   {per_lane:6.3} ms/lane   \
             {:.2}x cheaper per lane",
            baseline / per_lane
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let quant = match std::env::args().nth(1).as_deref() {
        Some("f32") => Weight::F32,
        Some("f16") => Weight::F16,
        Some("q4_0") => Weight::Quant(GgmlDType::Q4_0),
        _ => Weight::Quant(GgmlDType::Q8_0),
    };
    let device = Device::new_metal(0).context("opening the Metal device")?;
    let w = Weights::load(WEIGHTS, &device)?;

    let talker_geo = Geometry {
        dim: tk::DIM,
        layers: tk::LAYERS,
        heads: tk::HEADS,
        n_kv: tk::N_KV,
        head_dim: tk::HEAD_DIM,
        ffn: tk::FFN,
        eps: tk::NORM_EPS,
        rope_base: tk::ROPE_BASE,
        qk_norm: true,
        layer_scale: false,
        window: None,
    };
    let predictor_geo = Geometry {
        dim: pk::DIM,
        layers: pk::LAYERS,
        heads: pk::HEADS,
        n_kv: pk::N_KV,
        head_dim: pk::HEAD_DIM,
        ffn: pk::FFN,
        eps: pk::NORM_EPS,
        rope_base: pk::ROPE_BASE,
        qk_norm: pk::QK_NORM,
        layer_scale: false,
        window: pk::SLIDING_WINDOW,
    };

    println!("quant {:?}", quant);
    let talker = Stack::load(
        &w,
        "talker.model.",
        talker_geo,
        quant,
        PROMPT + 256,
        &device,
    )?;
    bench(
        &format!("talker trunk ({} layers @ {})", tk::LAYERS, tk::DIM),
        &talker,
        tk::DIM,
        &device,
    )?;
    drop(talker);

    let predictor = Stack::load(
        &w,
        "talker.code_predictor.model.",
        predictor_geo,
        quant,
        PROMPT + 256,
        &device,
    )?;
    bench(
        &format!("depth predictor ({} layers @ {})", pk::LAYERS, pk::DIM),
        &predictor,
        pk::DIM,
        &device,
    )?;
    Ok(())
}
