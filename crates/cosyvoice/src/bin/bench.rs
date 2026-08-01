//! Where CosyVoice's time actually goes, on the real loaded weights.
//!
//! The `dit` probe in `tts-probe` measures op shapes in isolation, which is how the
//! attention and layer-norm wins were found. It is not a substitute for timing the real
//! thing: a mocked block came out at ~11 ms while the loaded model spends ~22 ms per
//! block, and only measuring the actual code path says where the rest is.
//!
//! Run: `cargo run -p cosyvoice --release --bin cosyvoice-bench`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice::flow::Flow;
use cosyvoice::hift::{Hift, Noise};
use cosyvoice::llm::Llm;
use std::collections::HashMap;
use tts_bench::Harness;
use tts_core::rng::Rng;

const FIXTURES: &str = "fixtures/cosyvoice/oracle.safetensors";
const NOISE_ASSET: &str = "fixtures/cosyvoice/rand_noise.safetensors";
const WEIGHTS: &str = "references/cosyvoice/weights";

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 5)?;

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &dev)
        .with_context(|| format!("loading {FIXTURES}"))?;
    let get = |n: &str| -> Result<Tensor> {
        Ok(fx
            .get(n)
            .with_context(|| format!("missing {n}"))?
            .to_dtype(DType::F32)?)
    };

    let flow = Flow::load(&format!("{WEIGHTS}/flow.safetensors"), NOISE_ASSET, &dev)?;
    let hift = Hift::load(&format!("{WEIGHTS}/hift.safetensors"), &dev)?;

    let mu = get("flow.mu")?;
    let cond = get("flow.cond")?;
    let spks = get("flow.spks")?;
    let z = get("flow.z")?;
    let n = mu.dim(2)?;
    println!("flow: {n} mel frames ({} prompt + generated)", n);

    // ------------------------------------------------------------------ flow
    // One DiT evaluation against the whole solve. The solve should be almost exactly
    // `N_TIMESTEPS` times one evaluation; if it is not, the overhead is in the solver.
    {
        let (f1, z1, mu1, c1, s1) = (&flow, z.clone(), mu.clone(), cond.clone(), spks.clone());
        let (f2, mu2, c2, s2) = (&flow, mu.clone(), cond.clone(), spks.clone());
        h.ab(
            "flow decoder",
            &mut [
                ("one DiT evaluation (batch 2)", &mut move || {
                    let _ = f1
                        .estimate(&z1, &mu1, &c1, &s1, 0.0)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("full 10-step Euler solve", &mut move || {
                    let _ = f2
                        .solve(&mu2, &c2, &s2)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
            ],
        )?;
    }

    // Inside one evaluation: the 22 blocks against everything before them.
    {
        let (f1, z1, mu1, c1, s1) = (&flow, z.clone(), mu.clone(), cond.clone(), spks.clone());
        let (f2, z2, mu2, c2, s2) = (&flow, z.clone(), mu.clone(), cond.clone(), spks.clone());
        h.ab(
            "inside one DiT evaluation",
            &mut [
                ("everything (trace)", &mut move || {
                    let _ = f1
                        .trace(&z1, &mu1, &c1, &s1, 0.0)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                (
                    "input embedding + position embedding only",
                    &mut move || {
                        let _ = f2
                            .embed_only(&z2, &mu2, &c2, &s2)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                        Ok(())
                    },
                ),
            ],
        )?;
    }

    // ------------------------------------------------------------------ vocoder
    {
        let mel = get("hift.mel_in")?;
        let noise = get("hift.nsf_noise")?;
        let f0_ref = get("hift.f0")?;
        let src = get("hift.source")?;
        let (h1, m1) = (&hift, mel.clone());
        let (h2, f2, n2) = (&hift, f0_ref.clone(), noise.clone());
        let (h3, m3, s3) = (&hift, mel.clone(), src.clone());
        let (h4, m4, n4) = (&hift, mel.clone(), noise.clone());
        h.ab(
            "vocoder stages",
            &mut [
                ("f0 predictor", &mut move || {
                    let _ = h1
                        .predict_f0(&m1)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("harmonic source", &mut move || {
                    let _ = h2
                        .source(&f2, Noise::Reference(&n2))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("upsampling decoder + iSTFT", &mut move || {
                    let _ = h3
                        .decode(&m3, &s3)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("all of it", &mut move || {
                    let _ = h4
                        .forward(&m4, Noise::Reference(&n4))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
            ],
        )?;

        // The engine reported 1.6 s of vocoder against this bench's 0.46 s for the same
        // work. The only difference is where the NSF noise comes from — synthesis draws
        // its own, ~1.0 M uniforms per utterance. That should be a few milliseconds of
        // xoshiro plus a 4 MB upload, so if it is not, the cost is in how it is built.
        let (h5, m5, n5) = (&hift, mel.clone(), noise.clone());
        let (h6, m6) = (&hift, mel.clone());
        let mut rng = Rng::new(0);
        h.ab(
            "NSF noise: reference asset vs drawn",
            &mut [
                ("reference slice", &mut move || {
                    let _ = h5
                        .forward(&m5, Noise::Reference(&n5))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("drawn (what synthesis does)", &mut move || {
                    let _ = h6
                        .forward(&m6, Noise::Draw(&mut rng))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
            ],
        )?;
    }

    // ------------------------------------------------------------------ LLM
    // Prefill against a decode step: the split that says whether the prompt or the loop
    // dominates, which for a 356-position prompt and ~105 generated tokens is not obvious.
    {
        let llm = Llm::load(&format!("{WEIGHTS}/llm.safetensors"), None, &dev)?;
        let lm_input = get("llm.lm_input")?;
        let state = llm.prefill(&lm_input).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut rng = Rng::new(0);
        let _ = rng.next_f32();

        let (l1, i1) = (&llm, lm_input.clone());
        let l2 = &llm;
        let hidden = state.hidden.clone();
        h.ab(
            "LLM",
            &mut [
                ("prefill (356 positions)", &mut move || {
                    let _ = l1
                        .prefill(&i1)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
                ("logit head, one position", &mut move || {
                    let _ = l2
                        .logits(&hidden)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    Ok(())
                }),
            ],
        )?;
    }

    h.report_drift()?;
    Ok(())
}
