//! Validate the port against the Phase A fixtures.
//!
//! The fixtures are fp32 CPU reference tensors, so a mismatch here is a port bug
//! rather than a precision difference — that separation is the whole reason
//! `dump_fixtures.py` runs on CPU in fp32 when the model ships in bfloat16.
//!
//! Run:  cargo run -p audio8 --release --bin audio8-validate -- [--cpu]

use anyhow::Result;
use audio8::ar::{GenConfig, Model};
use audio8::cfg;
use audio8::codec::Codec;
use audio8::nn::{max_abs_diff, Weights};
use audio8::prompt::PromptBuilder;
use audio8::sample::Rng;
use candle_core::{DType, Device, Tensor};

/// Fixture tolerance. Conv reordering and a different reduction order across ~50
/// sequential layers accumulate; anything under this is arithmetic noise, anything
/// far above it is a wiring bug. Reported alongside signal RMS so a small absolute
/// number on a small signal cannot masquerade as a pass.
const TOL: f32 = 2e-3;

fn rms(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?
        .sqrt())
}

/// Pull `[1, num_codebooks, T]` int64 codes out of a fixture into row-major u32.
fn codes_from(t: &Tensor) -> Result<Vec<Vec<u32>>> {
    let t = t.squeeze(0)?.to_dtype(DType::U32)?;
    let (n, len) = t.dims2()?;
    let flat = t.flatten_all()?.to_vec1::<u32>()?;
    Ok((0..n)
        .map(|i| flat[i * len..(i + 1) * len].to_vec())
        .collect())
}

fn main() -> Result<()> {
    let cpu = std::env::args().any(|a| a == "--cpu");
    let device = if cpu {
        Device::Cpu
    } else {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    };
    println!("validating on {device:?}\n");

    let fx = Weights::load("fixtures/audio8/oracle.safetensors", &Device::Cpu)?;
    println!("loaded {} fixture tensors", fx.len());

    let codec = Codec::load("references/audio8/weights/codec.safetensors", &device)?;
    println!("loaded codec\n");

    let mut failures = 0usize;
    println!(
        "{:<26} {:>8} {:>12} {:>12} {:>10}  verdict",
        "case", "frames", "max|diff|", "signal rms", "rel"
    );
    println!("{}", "-".repeat(88));

    // Two independent code sequences, both with a genuine reference waveform.
    for (name, codes_key, wav_key) in [
        ("codec synthetic", "codec_syn.codes", "codec_syn.wav"),
        ("codec e2e", "e2e.codes", "e2e.wav"),
    ] {
        let codes = codes_from(&fx.raw(codes_key)?)?;
        let want = fx.get(wav_key)?;
        let frames = codes[0].len();
        let got = codec.decode(&codes)?.to_device(&Device::Cpu)?;
        if got.dims() != want.dims() {
            println!(
                "{name:<26} {frames:>8} {:>12} {:>12} {:>10}  SHAPE {:?} vs {:?}",
                "-",
                "-",
                "-",
                got.dims(),
                want.dims()
            );
            failures += 1;
            continue;
        }
        let diff = max_abs_diff(&got, &want)?;
        let r = rms(&want)?;
        let ok = diff < TOL;
        if !ok {
            failures += 1;
        }
        println!(
            "{name:<26} {frames:>8} {diff:>12.3e} {r:>12.3e} {:>10.2e}  {}",
            diff / r.max(1e-12),
            if ok { "OK" } else { "FAIL" }
        );
    }

    // Expected sample count is a structural check independent of numerics: any
    // padding or cropping error in the causal convs shows up here first.
    let codes = codes_from(&fx.raw("e2e.codes")?)?;
    let frames = codes[0].len();
    let got = codec.decode(&codes)?;
    let expect = frames * cfg::CODEC_FRAME_SIZE;
    let actual = got.dim(2)?;
    println!(
        "\nsample count: {actual} for {frames} frames, expected {expect}  {}",
        if actual == expect { "OK" } else { "MISMATCH" }
    );
    if actual != expect {
        failures += 1;
    }

    // ----------------------------------------------------------------- AR half
    // The fixture prompt is the no-reference branch for "Welcome to Audio8 TTS.",
    // 27 tokens. Tokenizer *and* prompt template are both checked by comparing ids
    // before any arithmetic runs: a prompt that is off by one token produces
    // plausible audio and would be very hard to find later.
    println!("\n--- AR ---");
    let builder = PromptBuilder::load("references/audio8/weights/tokenizer.json")?;
    let prompt = builder.build("Welcome to Audio8 TTS.", None)?;
    let want_ids: Vec<u32> = fx
        .raw("prompt.prefix_input_ids")?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;
    let ids_ok = prompt.rows[0] == want_ids;
    println!(
        "prompt ids: {} tokens vs fixture {}  {}",
        prompt.len,
        want_ids.len(),
        if ids_ok { "OK" } else { "MISMATCH" }
    );
    if !ids_ok {
        failures += 1;
        let n = prompt.rows[0].len().min(want_ids.len());
        let first = (0..n).find(|&i| prompt.rows[0][i] != want_ids[i]);
        println!("  first difference at {first:?}");
        println!("  got   {:?}", &prompt.rows[0][..n.min(12)]);
        println!("  want  {:?}", &want_ids[..n.min(12)]);
    }

    // Dense f32 weights here, not q8_0: this compares against an fp32 reference, so
    // quantization error would be indistinguishable from a wiring bug. The quantized
    // path is validated on audio instead (docs/performance/quantization-quality.md).
    let model = Model::load("references/audio8/weights/model.safetensors", &device, None)?;
    println!("loaded AR model (dense f32)");
    let (logits, normed) = model.debug_prefill(&prompt)?;
    let logits = logits.to_device(&Device::Cpu)?;
    let normed = normed.to_device(&Device::Cpu)?;

    // `slow.norm` is the normed hidden state over all 27 positions.
    let want_norm = fx.get("slow.norm")?;
    let d = max_abs_diff(&normed, &want_norm)?;
    let r = rms(&want_norm)?;
    let ok = d < 5e-3;
    if !ok {
        failures += 1;
    }
    println!(
        "slow.norm       max|diff| {d:.3e}  rms {r:.3e}  rel {:.2e}  {}",
        d / r.max(1e-12),
        if ok { "OK" } else { "FAIL" }
    );

    // Compare logits only over the reachable rows, because that is all the port
    // computes — and prove the slice is legitimate by checking that the reference's
    // argmax over the *full* vocabulary lands inside it.
    let want_logits = fx.get("slow.logits")?.squeeze(0)?;
    let want_slice = Tensor::cat(
        &[
            &want_logits.narrow(1, cfg::SEMANTIC_BEGIN_ID as usize, cfg::CODEBOOK_SIZE)?,
            &want_logits.narrow(1, cfg::EOS_TOKEN_ID as usize, 1)?,
        ],
        1,
    )?;
    let d = max_abs_diff(&logits, &want_slice)?;
    let r = rms(&want_slice)?;
    let ok = d < 5e-2;
    if !ok {
        failures += 1;
    }
    println!(
        "slow.logits     max|diff| {d:.3e}  rms {r:.3e}  rel {:.2e}  {}",
        d / r.max(1e-12),
        if ok { "OK" } else { "FAIL" }
    );

    // Greedy agreement on the last position is the check that actually matters: it is
    // what decides the first emitted token.
    let last_got = logits
        .narrow(0, prompt.len - 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let last_want = want_slice
        .narrow(0, prompt.len - 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            })
            .0
    };
    let (g, wv) = (am(&last_got), am(&last_want));
    println!(
        "greedy argmax at last position: got {g}, reference {wv}  {}",
        if g == wv { "OK" } else { "MISMATCH" }
    );
    if g != wv {
        failures += 1;
    }

    // ------------------------------------------------- end-to-end greedy codes
    // `e2e.codes` is the reference's greedy generation for this exact prompt with
    // max_new_tokens=24. Greedy is deterministic on both sides, so this compares the
    // whole AR loop token for token: embedding, RoPE, GQA, KV cache, the discarded
    // fast-AR priming step, the legacy filter and the argmax. Nothing else in this
    // file exercises the fast AR at all.
    let want_codes = codes_from(&fx.raw("e2e.codes")?)?;
    let greedy = GenConfig {
        max_new_tokens: want_codes[0].len(),
        do_sample: false,
        ..Default::default()
    };
    let mut rng = Rng::new(0);
    let got_codes = model.generate(&prompt, &greedy, &mut rng)?;
    let n_want = want_codes[0].len();
    let n_got = got_codes[0].len();
    if n_got != n_want {
        println!("\ngreedy codes: {n_got} frames vs reference {n_want}  MISMATCH");
        failures += 1;
    } else {
        let mut first_bad: Option<(usize, usize)> = None;
        let mut same = 0usize;
        for t in 0..n_want {
            let mut frame_ok = true;
            for cb in 0..cfg::NUM_CODEBOOKS {
                if got_codes[cb][t] != want_codes[cb][t] {
                    frame_ok = false;
                    if first_bad.is_none() {
                        first_bad = Some((t, cb));
                    }
                }
            }
            if frame_ok {
                same += 1;
            }
        }
        let ok = same == n_want;
        if !ok {
            failures += 1;
        }
        println!(
            "\ngreedy codes: {same}/{n_want} frames identical across all 10 codebooks  {}",
            if ok { "OK" } else { "FAIL" }
        );
        if let Some((t, cb)) = first_bad {
            println!(
                "  first difference at frame {t}, codebook {cb}: got {} want {}",
                got_codes[cb][t], want_codes[cb][t]
            );
            println!("  got  semantic row: {:?}", &got_codes[0][..n_want.min(12)]);
            println!(
                "  want semantic row: {:?}",
                &want_codes[0][..n_want.min(12)]
            );
        }
    }

    // ---------------------------------------------------------------- batching
    // The gate for `generate_batch`. Batching has no fixture of its own — the reference is
    // batch-1 — so the property checked instead is self-consistency: a batch of four
    // *distinct* prompts, decoded greedily, must reproduce exactly what those four prompts
    // produce one at a time. Greedy is what makes this a hard equality; sampled output
    // legitimately differs, because per-sequence sampling inside a batched step consumes the
    // RNG in a different order.
    //
    // Distinct prompts matter twice over. Their widths differ, so this is the only check that
    // exercises right-alignment, the per-sequence prefill mask and the padded-column decode
    // mask at all — four copies of one prompt would have `pad == 0` everywhere and prove
    // nothing about the part most likely to be wrong.
    {
        println!("\n--- batching ---");
        let texts = [
            "Welcome to Audio8 TTS.",
            "A shorter one.",
            "This third prompt is deliberately a good deal longer than the others, so that \
             right-alignment has something to align.",
            "Four.",
        ];
        let prompts: Vec<_> = texts
            .iter()
            .map(|t| builder.build(t, None))
            .collect::<Result<Vec<_>>>()?;
        let widths: Vec<usize> = prompts.iter().map(|p| p.len).collect();
        println!(
            "prompt widths {widths:?} -> right-aligned to {}",
            widths.iter().max().unwrap()
        );

        let greedy = GenConfig {
            max_new_tokens: 24,
            do_sample: false,
            ..GenConfig::default()
        };

        let mut solo: Vec<Vec<Vec<u32>>> = Vec::new();
        for p in &prompts {
            let mut rng = Rng::new(1);
            solo.push(model.generate(p, &greedy, &mut rng)?);
        }

        // First isolate the two mechanisms. Four copies of one prompt all have `pad == 0`,
        // so this exercises the batch axis with no masking at all; if it passes and the
        // mixed-width case does not, the fault is in right-alignment rather than in
        // batching.
        {
            let same: Vec<&audio8::prompt::Prompt> = (0..4).map(|_| &prompts[0]).collect();
            let mut rng = Rng::new(1);
            let out = model.generate_batch(&same, &greedy, &mut rng)?;
            let ok = out.iter().all(|o| *o == solo[0]);
            if !ok {
                failures += 1;
            }
            println!(
                "  batch of 4 identical prompts (no padding): {}",
                if ok { "all match solo" } else { "DIFFER" }
            );
        }

        // Then the alignment logic itself, with the bf16 rounding taken out of the way. See
        // `Model::with_f32_rope`: right-alignment is exact in real arithmetic, but a
        // bf16-rounded table only satisfies `R(p)^T R(j) = R(p - j)` to ~4e-3, so this is
        // where an exact equality can legitimately be demanded.
        {
            let f32rope =
                Model::load("references/audio8/weights/model.safetensors", &device, None)?
                    .with_f32_rope()?;
            let mut solo32: Vec<Vec<Vec<u32>>> = Vec::new();
            for p in &prompts {
                let mut rng = Rng::new(1);
                solo32.push(f32rope.generate(p, &greedy, &mut rng)?);
            }
            let refs: Vec<&audio8::prompt::Prompt> = prompts.iter().collect();
            let mut rng = Rng::new(1);
            let batched32 = f32rope.generate_batch(&refs, &greedy, &mut rng)?;
            let ok = (0..prompts.len()).all(|i| solo32[i] == batched32[i]);
            if !ok {
                failures += 1;
            }
            println!(
                "  mixed widths with f32 RoPE tables (alignment logic alone): {}",
                if ok { "all match solo" } else { "DIFFER" }
            );
        }

        let refs: Vec<&audio8::prompt::Prompt> = prompts.iter().collect();
        let mut rng = Rng::new(1);
        let batched = model.generate_batch(&refs, &greedy, &mut rng)?;

        // Finally the same comparison under the *real* bf16 tables. This is reported, not
        // asserted, and the reason is on the two rows above: batching is exact and
        // right-alignment is exact, so whatever appears here is the table rounding — a
        // ~4e-3 perturbation of the attention scores, which under greedy decoding can flip a
        // near-tie and then diverge. Demanding equality here would be demanding that bf16
        // rounding commute with a position shift, which it does not.
        //
        // What makes shipping this defensible is not this row but the audio: see
        // docs/performance/ar-loop.md for WER and voice metrics on a batched long-form render.
        let mut agree = 0usize;
        for i in 0..prompts.len() {
            let (a, b) = (&solo[i], &batched[i]);
            let common = a[0].len().min(b[0].len());
            let prefix = (0..common)
                .take_while(|&t| (0..cfg::NUM_CODEBOOKS).all(|c| a[c][t] == b[c][t]))
                .count();
            if a == b {
                agree += 1;
            }
            println!(
                "  seq {i}: width {:>3}, {} frames solo vs {} batched, agree for first {prefix}",
                widths[i],
                a[0].len(),
                b[0].len()
            );
        }
        println!(
            "  under bf16 tables {agree}/{} sequences identical — expected, see above",
            prompts.len()
        );

        // The planner must partition: every segment decoded exactly once, no group over the
        // cap. Losing or duplicating a segment here would drop or repeat audio.
        let mut planner_ok = true;
        for n in 0..40usize {
            for max in [1usize, 2, 4, 8] {
                let groups = audio8::ar::plan_batches(n, max);
                let mut seen: Vec<usize> = groups.iter().flatten().copied().collect();
                seen.sort_unstable();
                if seen != (0..n).collect::<Vec<_>>()
                    || !groups.iter().all(|g| !g.is_empty() && g.len() <= max)
                {
                    planner_ok = false;
                }
            }
        }
        if !planner_ok {
            failures += 1;
        }
        println!(
            "plan_batches partitions 0..40 segments over caps 1/2/4/8  {}",
            if planner_ok { "OK" } else { "FAIL" }
        );
    }

    println!(
        "\n{}",
        if failures == 0 {
            "all checks passed".to_string()
        } else {
            format!("{failures} check(s) FAILED")
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
