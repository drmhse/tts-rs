//! The Qwen3-TTS gate. Two tiers:
//!
//! 1. **Shape audit** — every [`qwen3tts::cfg`] constant against the checkpoint's tensor
//!    shapes. Reads only the safetensors header, so it costs milliseconds on 3.9 GB.
//! 2. **Numerics** vs `fixtures/qwen3tts/oracle.safetensors`. Not written; reports as
//!    skipped rather than passing vacuously.
//!
//! Run: `cargo run -p qwen3tts --release --bin qwen3tts-validate`

use anyhow::{Context, Result};
use qwen3tts::cfg;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

const TALKER: &str = "references/qwen3tts/weights/model.safetensors";
const CODEC: &str = "references/qwen3tts/weights/speech_tokenizer/model.safetensors";
const FIXTURES: &str = "fixtures/qwen3tts/oracle.safetensors";
const VOICE: &str = "voices/cosy-default-qwen3tts";

/// A header entry. `dtype`/`data_offsets` are declared so an unexpected schema fails to
/// parse instead of deserialising empty.
#[derive(Deserialize)]
struct Entry {
    #[allow(dead_code)]
    dtype: String,
    shape: Vec<usize>,
    #[allow(dead_code)]
    data_offsets: (usize, usize),
}

/// Parse a safetensors header: 8-byte LE length, then that many bytes of JSON.
///
/// Not `SafeTensors::read_metadata`, which requires the buffer to be the whole file — that
/// would mean mapping 3.9 GB, and would refuse a file still downloading.
fn shapes(path: &str) -> Result<HashMap<String, Vec<usize>>> {
    let mut file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("reading the header length of {path}"))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .with_context(|| format!("reading the {header_len}-byte header of {path}"))?;
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
        .with_context(|| format!("parsing the safetensors header of {path}"))?;
    let mut out = HashMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: Entry = serde_json::from_value(value)
            .with_context(|| format!("parsing the header entry for `{name}` in {path}"))?;
        out.insert(name, entry.shape);
    }
    Ok(out)
}

struct Report {
    rows: usize,
    failures: usize,
}

impl Report {
    fn new() -> Self {
        Self {
            rows: 0,
            failures: 0,
        }
    }

    /// `want` is `Some(n)` per fixed dimension, `None` for a free one.
    fn shape(&mut self, shapes: &HashMap<String, Vec<usize>>, name: &str, want: &[Option<usize>]) {
        self.rows += 1;
        let Some(got) = shapes.get(name) else {
            self.failures += 1;
            println!("{name:<62} MISSING");
            return;
        };
        let rank_ok = got.len() == want.len();
        let dims_ok = rank_ok
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| w.map(|w| w == *g).unwrap_or(true));
        if dims_ok {
            println!("{name:<62} {got:?}  OK");
        } else {
            self.failures += 1;
            let expected: Vec<String> = want
                .iter()
                .map(|w| w.map(|w| w.to_string()).unwrap_or_else(|| "*".into()))
                .collect();
            println!(
                "{name:<62} {got:?} vs expected [{}]  FAIL",
                expected.join(", ")
            );
        }
    }

    /// A tensor comparison, tolerance stated per row: a 76800-sample waveform and a
    /// 2048-wide hidden state do not deserve the same threshold.
    fn tensor(
        &mut self,
        name: &str,
        got: &candle_core::Tensor,
        want: &candle_core::Tensor,
        tol: f32,
    ) -> Result<()> {
        self.rows += 1;
        if got.dims() != want.dims() {
            self.failures += 1;
            println!(
                "{name:<62} SHAPE {:?} vs {:?}  FAIL",
                got.dims(),
                want.dims()
            );
            return Ok(());
        }
        let (abs, rel) = tts_nn::abs_and_rel(got, want)?;
        let ok = abs <= tol;
        if !ok {
            self.failures += 1;
        }
        println!(
            "{name:<44} max|d| {abs:>9.2e}  rel {rel:>8.1e}  tol {tol:>7.1e}  {}",
            if ok { "OK" } else { "FAIL" }
        );
        Ok(())
    }

    /// A fact about the checkpoint that is not one tensor's shape.
    fn claim(&mut self, name: &str, ok: bool, detail: &str) {
        self.rows += 1;
        if !ok {
            self.failures += 1;
        }
        println!("{name:<62} {detail}  {}", if ok { "OK" } else { "FAIL" });
    }

    /// A tensor that must *not* be present — mainly attention biases, which CosyVoice's
    /// Qwen2 has and this checkpoint does not.
    fn absent(&mut self, shapes: &HashMap<String, Vec<usize>>, name: &str) {
        self.rows += 1;
        let present = shapes.contains_key(name);
        if present {
            self.failures += 1;
            println!("{name:<62} PRESENT, expected absent  FAIL");
        } else {
            println!("{name:<62} absent  OK");
        }
    }
}

/// How many `{prefix}layers.{i}.{suffix}` tensors exist, counting up from 0.
fn count_layers(shapes: &HashMap<String, Vec<usize>>, prefix: &str, suffix: &str) -> usize {
    (0..)
        .take_while(|i| shapes.contains_key(&format!("{prefix}layers.{i}.{suffix}")))
        .count()
}

/// Each stage against the reference's own inputs, so one early error does not fail every row.
///
/// f32 deliberately: the engine defaults to q8_0 for memory reasons, and quantization can flip
/// an argmax on a near-tie. The first run of this gate failed `prompt.embeds` at rel 5.3e-2
/// while every model stage passed — the fixture, not the port, was wrong (raw text where the
/// reference wraps it in a chat template). Separating prompt assembly from the model is what
/// made that one row instead of ten.
fn numerics(report: &mut Report) -> Result<()> {
    use candle_core::{Device, Tensor};

    // CPU, not Metal. The fixtures are fp32 CPU by convention, and an f32 talker is 6.3 GB
    // of projections — enough to fail Metal buffer allocation outright on a 16 GB machine.
    let device = Device::Cpu;
    let fx = candle_core::safetensors::load(FIXTURES, &device)?;
    let get = |name: &str| -> Result<Tensor> {
        fx.get(name)
            .cloned()
            .with_context(|| format!("fixture `{name}` missing"))
    };

    let voice = tts_core::Voice::load(VOICE)?;
    let talker = qwen3tts::talker::Talker::load(TALKER, tts_nn::Weight::F32, &device)?;
    let spk = talker.speaker(voice.get("spk_embedding")?)?;
    let ref_codes = voice.get_rows_u32("ref_codes")?;
    let ref_text = voice.get_rows_u32("ref_text_tokens")?.remove(0);

    // The reference's own text ids, so a tokenizer difference does not show up as a model
    // difference.
    let text: Vec<u32> = get("tokens.text")?
        .flatten_all()?
        .to_dtype(candle_core::DType::U32)?
        .to_vec1::<u32>()?;

    let (prompt, trailing) = talker.build_prompt(
        &text,
        &ref_text,
        &ref_codes,
        Some(&spk),
        cfg::talker::language_id("english"),
    )?;
    report.tensor("prompt.embeds", &prompt, &get("prompt.embeds")?, 2e-3)?;
    report.tensor("prompt.trailing", &trailing, &get("prompt.trailing")?, 2e-3)?;

    let (hidden, logits) = talker.trace(&get("prompt.embeds")?)?;
    report.tensor("talker.hidden", &hidden, &get("talker.hidden")?, 5e-3)?;
    report.tensor("talker.logits", &logits, &get("talker.logits")?, 5e-2)?;

    // The argmax is what actually drives generation, so it gets its own row.
    let want_code0 = get("talker.logits")?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    let got_code0 = logits
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    report.claim(
        "talker argmax codebook 0",
        got_code0 == want_code0,
        &format!("{got_code0} vs reference {want_code0}"),
    );

    let want_codes: Vec<u32> = get("predictor.codes")?
        .to_dtype(candle_core::DType::U32)?
        .to_vec1::<u32>()?;
    let got_codes = talker.predict_frame(&get("talker.hidden")?, want_code0, true)?;
    let same = got_codes
        .iter()
        .zip(&want_codes)
        .filter(|(a, b)| a == b)
        .count();
    report.claim(
        "predictor argmax codes 1..15",
        same == want_codes.len(),
        &format!("{same}/{} identical", want_codes.len()),
    );

    // The loop update, and one decode step through it.
    let mut frame0 = vec![want_code0];
    frame0.extend_from_slice(&want_codes);
    let text_next = get("prompt.trailing")?.narrow(1, 0, 1)?;
    let step_input = talker.step_input(&frame0, &text_next)?;
    report.tensor("step1.input", &step_input, &get("step1.input")?, 2e-3)?;
    let (s1_hidden, s1_logits) = talker.trace_step(&get("prompt.embeds")?, &get("step1.input")?)?;
    report.tensor("step1.hidden", &s1_hidden, &get("step1.hidden")?, 5e-3)?;
    report.tensor("step1.logits", &s1_logits, &get("step1.logits")?, 5e-2)?;

    let codec = qwen3tts::codec::Codec::load(CODEC, &device)?;
    let codes = get("codec.codes")?.to_dtype(candle_core::DType::U32)?;
    let (frames_n, groups) = codes.dims2()?;
    let flat = codes.flatten_all()?.to_vec1::<u32>()?;
    let frames: Vec<Vec<u32>> = (0..frames_n)
        .map(|i| flat[i * groups..(i + 1) * groups].to_vec())
        .collect();

    let (quantized, pre_tf) = codec.trace(&frames)?;
    report.tensor(
        "codec.quantized",
        &quantized,
        &get("codec.quantized")?,
        1e-4,
    )?;
    report.tensor("codec.pre_tf", &pre_tf, &get("codec.pre_tf")?, 5e-3)?;

    let wav = codec.forward(&frames)?;
    report.tensor("codec.wav", &wav, &get("codec.wav")?, 5e-3)?;

    // The chunked path is what the engine calls.
    let chunked = codec.decode(&frames)?;
    let want = get("codec.wav_chunked")?.flatten_all()?.to_vec1::<f32>()?;
    let got = Tensor::from_vec(chunked, want.len().min(usize::MAX), &device);
    match got {
        Ok(got) if got.elem_count() == want.len() => {
            let want_t = Tensor::from_vec(want, got.elem_count(), &device)?;
            report.tensor("codec.wav_chunked", &got, &want_t, 5e-3)?;
        }
        _ => report.claim("codec.wav_chunked", false, "sample count differs"),
    }

    // The long block: past the pre-transformer's 72-frame sliding window *and* past
    // `chunked_decode`'s 300-frame split. The original fixture was 40 frames and exercised
    // neither, which left the window mask and the chunk seam unvalidated while real segments
    // average ~88 frames.
    let long_codes = get("codec.long.codes")?.to_dtype(candle_core::DType::U32)?;
    let (ln, lg) = long_codes.dims2()?;
    let lflat = long_codes.flatten_all()?.to_vec1::<u32>()?;
    let long_frames: Vec<Vec<u32>> = (0..ln)
        .map(|i| lflat[i * lg..(i + 1) * lg].to_vec())
        .collect();
    let long_wav = codec.forward(&long_frames)?;
    report.tensor("codec.long.wav", &long_wav, &get("codec.long.wav")?, 5e-3)?;

    let long_chunked = codec.decode(&long_frames)?;
    let long_want = get("codec.long.wav_chunked")?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if long_chunked.len() == long_want.len() {
        let a = Tensor::from_vec(long_chunked, long_want.len(), &device)?;
        let b = Tensor::from_vec(long_want, a.elem_count(), &device)?;
        report.tensor("codec.long.wav_chunked", &a, &b, 5e-3)?;
    } else {
        report.claim(
            "codec.long.wav_chunked",
            false,
            &format!("{} samples vs {}", long_chunked.len(), long_want.len()),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut report = Report::new();

    // ------------------------------------------------------------------ talker
    println!("\n=== talker: {TALKER}");
    match shapes(TALKER) {
        Err(e) => {
            println!("skipped: {e:#}");
            println!("Download Qwen/Qwen3-TTS-12Hz-1.7B-Base into references/qwen3tts/weights.");
        }
        Ok(t) => {
            println!("{} tensors in the header\n", t.len());

            // --- trunk ----------------------------------------------------------------
            {
                use cfg::talker as k;
                let p = "talker.model.";
                let l0 = format!("{p}layers.0");
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.q_proj.weight"),
                    &[Some(k::HEADS * k::HEAD_DIM), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.k_proj.weight"),
                    &[Some(k::N_KV * k::HEAD_DIM), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.v_proj.weight"),
                    &[Some(k::N_KV * k::HEAD_DIM), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.o_proj.weight"),
                    &[Some(k::DIM), Some(k::HEADS * k::HEAD_DIM)],
                );
                // Trap 9: no attention biases here.
                report.absent(&t, &format!("{l0}.self_attn.q_proj.bias"));
                report.absent(&t, &format!("{l0}.self_attn.k_proj.bias"));
                // QK-norm is over the head dim only: [128], not [DIM].
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.q_norm.weight"),
                    &[Some(k::HEAD_DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.k_norm.weight"),
                    &[Some(k::HEAD_DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.mlp.gate_proj.weight"),
                    &[Some(k::FFN), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.mlp.up_proj.weight"),
                    &[Some(k::FFN), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.mlp.down_proj.weight"),
                    &[Some(k::DIM), Some(k::FFN)],
                );
                report.shape(&t, &format!("{l0}.input_layernorm.weight"), &[Some(k::DIM)]);
                report.shape(
                    &t,
                    &format!("{l0}.post_attention_layernorm.weight"),
                    &[Some(k::DIM)],
                );
                report.shape(&t, &format!("{p}norm.weight"), &[Some(k::DIM)]);

                let layers = count_layers(&t, p, "self_attn.q_proj.weight");
                report.claim(
                    "talker layer count",
                    layers == k::LAYERS,
                    &format!("{layers} layers, cfg says {}", k::LAYERS),
                );

                // Embeddings and heads. `codec_embedding` and `codec_head` are both
                // [VOCAB, DIM] and are *not* tied (`tie_word_embeddings: false`), so both
                // must be loaded — reusing one for the other is a plausible optimisation
                // and a wrong model.
                report.shape(
                    &t,
                    &format!("{p}codec_embedding.weight"),
                    &[Some(k::VOCAB), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    "talker.codec_head.weight",
                    &[Some(k::VOCAB), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{p}text_embedding.weight"),
                    &[Some(k::TEXT_VOCAB), Some(k::TEXT_DIM)],
                );

                // Trap 9: text_projection is biased where attention is not.
                report.shape(
                    &t,
                    "talker.text_projection.linear_fc1.weight",
                    &[Some(k::TEXT_DIM), Some(k::TEXT_DIM)],
                );
                report.shape(
                    &t,
                    "talker.text_projection.linear_fc2.weight",
                    &[Some(k::DIM), Some(k::TEXT_DIM)],
                );
                report.claim(
                    "text_projection is biased",
                    t.contains_key("talker.text_projection.linear_fc1.bias")
                        && t.contains_key("talker.text_projection.linear_fc2.bias")
                            == k::TEXT_PROJECTION_BIAS,
                    "linear_fc1.bias and linear_fc2.bias present",
                );
            }

            // --- depth transformer ----------------------------------------------------
            {
                use cfg::predictor as k;
                let p = "talker.code_predictor.model.";
                let l0 = format!("{p}layers.0");
                // Trap 7: HEADS * HEAD_DIM is 2048 while DIM is 1024.
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.q_proj.weight"),
                    &[Some(k::HEADS * k::HEAD_DIM), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.k_proj.weight"),
                    &[Some(k::N_KV * k::HEAD_DIM), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.o_proj.weight"),
                    &[Some(k::DIM), Some(k::HEADS * k::HEAD_DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.self_attn.q_norm.weight"),
                    &[Some(k::HEAD_DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.mlp.gate_proj.weight"),
                    &[Some(k::FFN), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{l0}.mlp.down_proj.weight"),
                    &[Some(k::DIM), Some(k::FFN)],
                );
                report.shape(&t, &format!("{p}norm.weight"), &[Some(k::DIM)]);

                let layers = count_layers(&t, p, "self_attn.q_proj.weight");
                report.claim(
                    "predictor layer count",
                    layers == k::LAYERS,
                    &format!("{layers} layers, cfg says {}", k::LAYERS),
                );

                // Resize from talker width into the predictor's, biased.
                report.shape(
                    &t,
                    "talker.code_predictor.small_to_mtp_projection.weight",
                    &[Some(k::DIM), Some(k::EMBED_DIM)],
                );
                report.shape(
                    &t,
                    "talker.code_predictor.small_to_mtp_projection.bias",
                    &[Some(k::DIM)],
                );

                // Trap 8: 15 heads and 15 tables, tables at talker width.
                let heads = (0..)
                    .take_while(|i| {
                        t.contains_key(&format!("talker.code_predictor.lm_head.{i}.weight"))
                    })
                    .count();
                report.claim(
                    "predictor head count",
                    heads == k::HEADS_OUT,
                    &format!("{heads} heads, cfg says {}", k::HEADS_OUT),
                );
                report.shape(
                    &t,
                    "talker.code_predictor.lm_head.0.weight",
                    &[Some(k::VOCAB), Some(k::DIM)],
                );
                report.shape(
                    &t,
                    &format!("{p}codec_embedding.0.weight"),
                    &[Some(k::VOCAB), Some(k::EMBED_DIM)],
                );
                let tables = (0..)
                    .take_while(|i| t.contains_key(&format!("{p}codec_embedding.{i}.weight")))
                    .count();
                report.claim(
                    "predictor codebook tables",
                    tables == k::HEADS_OUT,
                    &format!("{tables} tables, cfg says {}", k::HEADS_OUT),
                );
            }

            // --- speaker encoder ------------------------------------------------------
            {
                use cfg::speaker as k;
                // Not in this crate — `export_voice.py` runs it. Checked because the voice
                // asset's width is decided here.
                report.shape(
                    &t,
                    "speaker_encoder.fc.weight",
                    &[Some(k::ENC_DIM), None, Some(1)],
                );
                report.claim(
                    "x-vector width equals the talker's hidden size",
                    k::ENC_DIM == cfg::talker::DIM,
                    &format!("{} == {}", k::ENC_DIM, cfg::talker::DIM),
                );
            }
        }
    }

    // ------------------------------------------------------------------ codec
    println!("\n=== codec: {CODEC}");
    match shapes(CODEC) {
        Err(e) => println!("skipped: {e:#}"),
        Ok(t) => {
            println!("{} tensors in the header\n", t.len());
            use cfg::codec as k;

            // Trap 5: the config claims 4096 semantic entries. Asserted against both stacks
            // so the claim cannot rot back to the config's version.
            let semantic = "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
            let acoustic = "decoder.quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum";
            report.shape(&t, semantic, &[Some(k::CODEBOOK), Some(k::QUANTIZER_INNER)]);
            report.shape(&t, acoustic, &[Some(k::CODEBOOK), Some(k::QUANTIZER_INNER)]);
            report.claim(
                "semantic and acoustic codebooks are the same size",
                t.get(semantic) == t.get(acoustic),
                "config's semantic_codebook_size: 4096 is dead",
            );

            // Trap 4: stored as a sum plus a usage count, divided at load.
            report.shape(
                &t,
                "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
                &[Some(k::CODEBOOK)],
            );
            report.claim(
                "codebooks need embedding_sum / cluster_usage",
                t.contains_key(semantic)
                    && t.contains_key(
                        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
                    ),
                "fold the division at load",
            );

            // `force_projection=True`: 1x1 convs in and out. `project_out` is an Identity
            // (codebook_dim == dim), so it has no tensors.
            report.shape(
                &t,
                "decoder.quantizer.rvq_first.input_proj.weight",
                &[Some(k::QUANTIZER_INNER), Some(k::CODEBOOK_DIM), Some(1)],
            );
            report.shape(
                &t,
                "decoder.quantizer.rvq_first.output_proj.weight",
                &[Some(k::CODEBOOK_DIM), Some(k::QUANTIZER_INNER), Some(1)],
            );
            report.absent(
                &t,
                "decoder.quantizer.rvq_first.vq.layers.0.project_out.weight",
            );

            // Trap 6: projected in and out rather than running at LATENT_DIM.
            report.shape(
                &t,
                "decoder.pre_transformer.input_proj.weight",
                &[Some(k::TF_DIM), Some(k::LATENT_DIM)],
            );
            report.shape(
                &t,
                "decoder.pre_transformer.output_proj.weight",
                &[Some(k::LATENT_DIM), Some(k::TF_DIM)],
            );

            // Trap 5: does the pre-transformer run at TF_DIM or at LATENT_DIM? The config
            // says hidden_size 512 while pre_conv emits latent_dim 1024, and exactly one of
            // those is the data path.
            let tf_q = t
                .iter()
                .filter(|(name, _)| {
                    name.contains("pre_transformer") && name.ends_with("q_proj.weight")
                })
                .min_by_key(|(name, _)| name.len())
                .map(|(name, shape)| (name.clone(), shape.clone()));
            match tf_q {
                None => report.claim(
                    "pre-transformer width",
                    false,
                    "no pre_transformer q_proj found",
                ),
                Some((name, shape)) => {
                    let input = shape.get(1).copied().unwrap_or(0);
                    let at_tf_dim = input == k::TF_DIM;
                    report.claim(
                        "pre-transformer input width",
                        at_tf_dim || input == k::LATENT_DIM,
                        &format!(
                            "{name} {shape:?} -> runs at {}",
                            if at_tf_dim {
                                "TF_DIM 512"
                            } else {
                                "LATENT_DIM 1024"
                            }
                        ),
                    );
                }
            }

            let tf_layers = (0..)
                .take_while(|i| {
                    t.keys().any(|key| {
                        key.contains("pre_transformer") && key.contains(&format!("layers.{i}."))
                    })
                })
                .count();
            report.claim(
                "pre-transformer layer count",
                tf_layers == k::TF_LAYERS,
                &format!("{tf_layers} layers, cfg says {}", k::TF_LAYERS),
            );

            let quantizers = t
                .keys()
                .filter(|key| key.contains("rvq_rest") && key.contains("embed"))
                .count();
            report.claim(
                "acoustic quantizer count",
                quantizers >= k::ACOUSTIC_QUANTIZERS,
                &format!(
                    "{quantizers} rvq_rest embeddings, cfg says {}",
                    k::ACOUSTIC_QUANTIZERS
                ),
            );
        }
    }

    // ------------------------------------------------------------------ numerics
    println!("\n=== numerics: {FIXTURES}");
    if !std::path::Path::new(FIXTURES).exists() {
        println!("skipped: run references/qwen3tts/dump_fixtures.py");
    } else {
        numerics(&mut report)?;
    }

    println!("\n{} row(s), {} failure(s)", report.rows, report.failures);
    anyhow::ensure!(report.failures == 0, "{} check(s) failed", report.failures);
    Ok(())
}
