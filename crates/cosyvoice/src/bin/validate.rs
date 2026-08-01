//! The CosyVoice fixture gate: every stage against `fixtures/cosyvoice/oracle.safetensors`.
//!
//! Fixtures are fp32 CPU reference tensors, so a mismatch is a port bug rather than a
//! precision difference. Stages are checked in dependency order and each is fed the
//! *reference's* inputs, not the previous stage's output — otherwise one early error
//! propagates and every row fails, which tells you nothing about where the bug is.
//!
//! Run: `cargo run -p cosyvoice --release --bin cosyvoice-validate`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice::{
    cfg,
    flow::Flow,
    hift::{Hift, Noise},
    llm::Llm,
};
use std::collections::HashMap;
use tts_core::rng::Rng;
use tts_nn::abs_and_rel;

const FIXTURES: &str = "fixtures/cosyvoice/oracle.safetensors";
const NOISE_ASSET: &str = "fixtures/cosyvoice/rand_noise.safetensors";
const WEIGHTS: &str = "references/cosyvoice/weights";

struct Report {
    failures: usize,
    rows: usize,
}

impl Report {
    fn new() -> Self {
        Self {
            failures: 0,
            rows: 0,
        }
    }

    /// A tensor comparison, with the tolerance stated per row rather than globally —
    /// an iSTFT over 100 k samples and a single 80-wide affine do not deserve the same
    /// threshold, and one number for both would either be vacuous or wrong.
    fn tensor(&mut self, name: &str, got: &Tensor, want: &Tensor, tol: f32) -> Result<()> {
        self.rows += 1;
        if got.dims() != want.dims() {
            println!(
                "{name:<26} SHAPE {:?} vs reference {:?}   FAIL",
                got.dims(),
                want.dims()
            );
            self.failures += 1;
            return Ok(());
        }
        let (abs, rel) = abs_and_rel(got, want)?;
        let ok = abs <= tol;
        if !ok {
            self.failures += 1;
        }
        println!(
            "{name:<26} max|diff| {abs:>10.3e}  rel {rel:>9.2e}  tol {tol:>8.1e}  {}",
            if ok { "OK" } else { "FAIL" }
        );
        Ok(())
    }

    fn ints(&mut self, name: &str, got: &[u32], want: &[u32]) {
        self.rows += 1;
        let n = got.len().min(want.len());
        let same = (0..n).filter(|&i| got[i] == want[i]).count();
        let ok = got.len() == want.len() && same == want.len();
        if !ok {
            self.failures += 1;
        }
        println!(
            "{name:<26} {same}/{} identical{}   {}",
            want.len(),
            if got.len() == want.len() {
                String::new()
            } else {
                format!(" (got {} ids)", got.len())
            },
            if ok { "OK" } else { "FAIL" }
        );
    }

    fn note(&self, text: &str) {
        println!("{:<26} {text}", "");
    }
}

fn main() -> Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("validating on {device:?}\n");

    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(FIXTURES, &device)
        .with_context(|| {
            format!("loading {FIXTURES} — run references/cosyvoice/dump_fixtures.py")
        })?;
    println!("loaded {} fixture tensors", fx.len());
    let get = |n: &str| -> Result<Tensor> {
        Ok(fx
            .get(n)
            .with_context(|| format!("fixture {n} missing"))?
            .to_dtype(DType::F32)?)
    };
    let get_ids = |n: &str| -> Result<Vec<u32>> {
        Ok(fx
            .get(n)
            .with_context(|| format!("fixture {n} missing"))?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?)
    };

    let mut r = Report::new();

    // ------------------------------------------------------------------ vocoder
    println!("\n--- vocoder (hift) ---");
    let hift = Hift::load(&format!("{WEIGHTS}/hift.safetensors"), &device)?;
    let mel_in = get("hift.mel_in")?;
    let f0_ref = get("hift.f0")?;
    let f0_ref_f32 = get("hift.f0_f32")?;

    let f0 = hift.predict_f0(&mel_in)?;
    // Against the f32 reference this should be a rounding; against the f64 one it
    // carries the whole cost of not having f64 on Metal.
    r.tensor("hift.f0 vs f32 ref", &f0, &f0_ref_f32, 5e-3)?;
    r.tensor("hift.f0 vs f64 ref", &f0, &f0_ref, 5e-2)?;

    // The source, from the *reference's* f0 so this row isolates the harmonic stack.
    let noise = get("hift.nsf_noise")?;
    let source = hift.source(&f0_ref, Noise::Reference(&noise))?;
    // 1.93e-5 is exactly the reference's own f32-vs-f64 error on this tensor, so this
    // row is at the floor: the harmonic stack contributes nothing measurable.
    r.tensor("hift.source", &source, &get("hift.source")?, 4e-5)?;

    // The decoder, from the reference's mel and source.
    let wav = hift.decode(&mel_in, &get("hift.source")?)?;
    r.tensor("hift.wav (ref source)", &wav, &get("hift.wav")?, 2e-4)?;

    // Decompose the end-to-end divergence rather than quoting one number for it. Three
    // things differ from the reference and they need separating, because two of them are
    // the reference's own f32 error and only one would be a port bug:
    //
    //   a) our harmonic phase is accurate where the reference's is f32 noise,
    //   b) our F0 is f32 where the reference's is f64,
    //   c) the NSF noise cannot be reproduced without the asset.
    //
    // Rows 4 and 5 isolate (a) and (a + b); the note below quantifies (c).
    let wav_phase = hift.decode(&mel_in, &hift.source(&f0_ref, Noise::Reference(&noise))?)?;
    r.tensor("hift.wav (ref f0)", &wav_phase, &get("hift.wav")?, 3e-2)?;
    let wav_e2e = hift.forward(&mel_in, Noise::Reference(&noise))?;
    r.tensor("hift.wav (own f0)", &wav_e2e, &get("hift.wav")?, 3e-2)?;
    r.note(
        "the last two rows are not port error: rows 2-3 show the harmonic stack and the \
         decoder are exact, so what is left is the reference's own f32 phase noise \
         (1 ulp = 1.0 rad at its peak accumulated phase of 1.7e7) plus f32 vs f64 F0",
    );

    // How much the f32 F0 costs, and how much substituting our own noise costs. Both
    // are reported rather than asserted, because both are deliberate divergences.
    let (abs_noise, _) = abs_and_rel(
        &hift.forward(&mel_in, Noise::Draw(&mut Rng::new(0)))?,
        &get("hift.wav")?,
    )?;
    let (abs_silent, _) = abs_and_rel(&hift.forward(&mel_in, Noise::Silent)?, &get("hift.wav")?)?;
    let rms = get("hift.wav")?
        .sqr()?
        .mean_all()?
        .sqrt()?
        .to_scalar::<f32>()?;
    r.note(&format!(
        "signal rms {rms:.4}; own NSF noise costs max|diff| {abs_noise:.4}, \
         no noise at all {abs_silent:.4}"
    ));

    // ------------------------------------------------------------------ flow
    println!("\n--- flow decoder ---");
    let flow = Flow::load(&format!("{WEIGHTS}/flow.safetensors"), NOISE_ASSET, &device)?;
    let spk = get("prompt.speaker_embedding")?;
    r.tensor("flow.spks", &flow.speaker(&spk)?, &get("flow.spks")?, 2e-6)?;

    let prompt_tokens = get_ids("prompt.speech_tokens")?;
    let gen_tokens = get_ids("llm.speech_tokens")?;
    let all: Vec<u32> = prompt_tokens
        .iter()
        .chain(gen_tokens.iter())
        .copied()
        .collect();
    let tok_emb = flow.embed_tokens(&all)?;
    r.tensor("flow.token_emb", &tok_emb, &get("flow.token_emb")?, 2e-6)?;

    let look = flow.pre_lookahead(&tok_emb)?;
    r.tensor("flow.lookahead", &look, &get("flow.lookahead")?, 2e-5)?;

    let mu = flow.mu(&look)?;
    r.tensor("flow.mu", &mu, &get("flow.mu")?, 2e-5)?;

    let prompt_mel = get("prompt.prompt_mel")?;
    let cond = flow.conditioning(&prompt_mel, mu.dim(2)?)?;
    r.tensor("flow.cond", &cond, &get("flow.cond")?, 1e-6)?;

    // One DiT evaluation on the doubled CFG batch at the solver's first timestep. This
    // is the row that separates "the DiT block is wrong" from "the solver is wrong".
    let t_span = get("flow.t_span")?;
    let t0 = t_span.narrow(0, 0, 1)?.to_vec1::<f32>()?[0];
    let dit = flow.estimate(
        &get("flow.z")?,
        &get("flow.mu")?,
        &get("flow.cond")?,
        &get("flow.spks")?,
        t0,
    )?;
    // Per-block trace first: it tells an accumulated f32 drift apart from one wrong
    // layer, which the aggregate row below cannot.
    let (time, dit_in, blocks) = flow.trace(
        &get("flow.z")?,
        &get("flow.mu")?,
        &get("flow.cond")?,
        &get("flow.spks")?,
        t0,
    )?;
    r.tensor("flow.dit_time", &time, &get("flow.dit_time")?, 1e-5)?;
    r.tensor("flow.dit_input", &dit_in, &get("flow.dit_input")?, 1e-4)?;
    // Per-block budgets come from a *measured* floor, not from taste. Running the
    // reference DiT in f64 and comparing against its own f32 gives max|diff| 1.14e-4,
    // 1.45e-4, 7.90e-3 and 1.56e-1 at blocks 0, 1, 10 and 21: the network amplifies
    // rounding error about 500x over its depth, so f32 cannot resolve it better than that.
    //
    // The multiplier grows with depth on purpose. That floor is one *sample* of where f32
    // lands, not a bound on where a correct implementation must land — at depth 21 the
    // computation is chaotic enough that changing the attention kernel for an equally
    // exact one (verified at rel 2.1e-6 in isolation) moved this row from 1.9e-1 to
    // 3.4e-1. So the early blocks carry the diagnostic weight, at 2x the floor, where a
    // wrong layer cannot hide; the last block is checked at 4x, which still catches a
    // structural error but does not fail for a legitimate change of op ordering. The
    // outcome that actually matters is `flow.mel` below.
    const DIT_FLOOR: [(usize, f32); 4] = [(0, 2.3e-4), (1, 2.9e-4), (10, 1.6e-2), (21, 6.3e-1)];
    for (b, tol) in DIT_FLOOR {
        r.tensor(
            &format!("flow.dit_block{b}"),
            &blocks[b],
            &get(&format!("flow.dit_block{b}"))?,
            tol,
        )?;
    }

    r.tensor("flow.dit_step0", &dit, &get("flow.dit_step0")?, 2e-2)?;

    // The full solver: 10 Euler steps, each on a doubled batch.
    let mel_full = flow.solve(&get("flow.mu")?, &get("flow.cond")?, &get("flow.spks")?)?;
    // The solver does not amplify: 10 Euler steps land at rel 3.7e-4, the same order as
    // a single DiT evaluation's 4.0e-4 floor. Budgeted like `dit_step0`.
    r.tensor("flow.mel_full", &mel_full, &get("flow.mel_full")?, 2e-2)?;

    let mel_len1 = prompt_mel.dim(1)?;
    let trimmed = mel_full
        .narrow(2, mel_len1, mel_full.dim(2)? - mel_len1)?
        .contiguous()?;
    r.tensor("flow.mel", &trimmed, &get("flow.mel")?, 2e-2)?;

    // ------------------------------------------------------------------ LLM
    println!("\n--- LLM ---");
    let llm = Llm::load(&format!("{WEIGHTS}/llm.safetensors"), None, &device)?;
    let text_tokens = get_ids("prompt.text_tokens")?;
    let prompt_text = get_ids("prompt.prompt_text_tokens")?;
    let full_text: Vec<u32> = prompt_text
        .iter()
        .chain(text_tokens.iter())
        .copied()
        .collect();

    let lm_input = llm.build_prompt(&full_text, &prompt_tokens)?;
    r.tensor("llm.lm_input", &lm_input, &get("llm.lm_input")?, 2e-6)?;

    let mut state = llm.prefill(&get("llm.lm_input")?)?;
    r.tensor(
        "llm.prefill_hidden",
        &state.hidden,
        &get("llm.prefill_hidden")?,
        2e-4,
    )?;
    r.tensor(
        "llm.prefill_logits",
        &llm.logits(&state.hidden)?,
        &get("llm.prefill_logits")?,
        5e-4,
    )?;

    // Teacher forcing over the reference's sampled sequence: the strong gate. Both
    // sides are deterministic, the inputs are non-degenerate, and every weight and the
    // whole KV cache contribute to the comparison.
    let mut rows = Vec::with_capacity(gen_tokens.len());
    let mut argmax = Vec::with_capacity(gen_tokens.len());
    for (i, &tok) in gen_tokens.iter().enumerate() {
        let logits = llm.logits(&state.hidden)?;
        argmax.push(argmax_of(&logits)?);
        rows.push(logits);
        let _ = i;
        state = llm.step(state, tok)?;
    }
    let tf = Tensor::cat(&rows, 0)?;
    r.tensor("llm.tf_logits", &tf, &get("llm.tf_logits")?, 5e-3)?;
    r.ints("llm.tf_argmax", &argmax, &get_ids("llm.tf_argmax")?);

    // Batching the LLM across segments right-aligns prompts of different lengths into one
    // padded batch, which is only sound if a lane's scores are unchanged by the constant
    // position shift the padding introduces. That needs `R(p)^T R(j) = R(p - j)` to hold in
    // the RoPE table's arithmetic.
    //
    // It holds to **f32, not exactly**: the padded lane comes back at rel 2.0e-6, while the
    // unpadded lane is bit-identical to decoding it alone. Two things contribute — the
    // rotary identity in f32, and the batched matmuls reducing in a different order. Audio8
    // batches identically and is far worse (its tables are bf16-rounded, where the identity
    // only survives to ~4e-3), so the f32 tables buy about three orders of magnitude here,
    // but "exact" would be the wrong word. The bound below is the same order as the
    // `prefill_hidden` fixture tolerance; a regression would show up as segments that drift
    // only when several are decoded together.
    let lm_input = get("llm.lm_input")?;
    let short = lm_input
        .narrow(1, lm_input.dim(1)? - 137, 137)?
        .contiguous()?;
    let solo_long = llm.prefill(&lm_input)?;
    let solo_short = llm.prefill(&short)?;
    let batched = llm.prefill_batch(&[lm_input.clone(), short.clone()])?;
    r.tensor(
        "llm.batch_lane0 (unpadded)",
        &batched.hidden.narrow(0, 0, 1)?,
        &solo_long.hidden,
        1e-6,
    )?;
    r.tensor(
        "llm.batch_lane1 (left-padded)",
        &batched.hidden.narrow(0, 1, 1)?,
        &solo_short.hidden,
        2e-4,
    )?;

    // ------------------------------------------------------------------
    println!();
    if r.failures == 0 {
        println!("all {} checks passed", r.rows);
        Ok(())
    } else {
        anyhow::bail!("{} of {} checks failed", r.failures, r.rows)
    }
}

fn argmax_of(logits: &Tensor) -> Result<u32> {
    Ok(logits
        .flatten_all()?
        .argmax(0)?
        .to_dtype(DType::U32)?
        .to_scalar::<u32>()?)
}

#[allow(dead_code)]
fn unused() {
    let _ = cfg::SAMPLE_RATE;
}
