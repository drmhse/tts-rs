//! The talker and its depth predictor.
//!
//! One file because they are not separable: the 15 codebook embedding tables belong to the
//! predictor but are read at talker width to build the talker's own next input (trap 8).
//!
//! Per frame:
//! 1. talker step -> `codec_head` -> codebook 0
//! 2. predictor prefills `[talker_hidden, code0_embed]` then runs 15 AR steps (trap 2)
//! 3. all 16 embeddings sum, plus the next text token, become the talker's next input (trap 3)

use crate::cfg::{self, predictor as pk, talker as tk};
use crate::qwen3::{Geometry, Stack};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;
use tts_core::rng::Rng;
use tts_nn::{Linear, Proj, Weight, Weights};

/// Per-stage frame cost, filled when `QWEN3TTS_TIMING` is set. Guessing which of the two
/// transformers dominates is how this repo previously optimised the wrong stage twice.
#[derive(Default, Clone, Copy, Debug)]
pub struct Timing {
    /// The prompt's prefill. Counted separately because it is per *segment*, not per frame,
    /// and it scales with the reference clip: ICL puts every one of its frames in the prompt.
    pub prefill_s: f64,
    pub talker_s: f64,
    pub predictor_s: f64,
    pub sample_s: f64,
    /// The predictor's 15 depth steps, split three ways.
    ///
    /// Every one of these timers has a `synchronize` at *both* ends. Anything less and the
    /// queue drains into whichever call happens to wait on it: with a sync only after the
    /// GEMM, the previous step's stack pass is billed to the GEMM and reads 26 ms instead of
    /// 2. Third time this repo has been caught by that, after CosyVoice's vocoder and the
    /// first reading of these same steps.
    pub depth_gemm_s: f64,
    pub depth_read_s: f64,
    pub depth_stack_s: f64,
    pub frames: usize,
    /// Decode steps actually run. A batched step costs the same whether every lane is still
    /// producing or only one is, so `frames / (steps * lanes)` is how much of the work was
    /// useful — lanes finish at different frames and a finished lane keeps being computed.
    pub steps: usize,
    pub lanes: usize,
}

fn timing_on() -> bool {
    std::env::var_os("QWEN3TTS_TIMING").is_some()
}

/// Cache capacity in positions, prompt included.
///
/// Sized rather than generous: 28 layers of `[1, 8, cap, 128]` k *and* v is 218 KB per
/// position, so 5120 positions is 1.09 GB — a real cost on a 16 GB machine where the weights
/// already want several. 1536 positions is 123 s of audio at 12.5 Hz, well past one segment.
const CAPACITY: usize = 1536;

pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub greedy: bool,
}

impl Default for Sampling {
    /// The reference's talker defaults.
    fn default() -> Self {
        Self {
            temperature: tk::TEMPERATURE,
            top_p: tk::TOP_P,
            top_k: tk::TOP_K,
            repetition_penalty: tk::REPETITION_PENALTY,
            greedy: false,
        }
    }
}

impl Sampling {
    /// The **sub-talker's** settings, which the reference keeps separate from the talker's
    /// (`subtalker_top_p` and friends) and which must not inherit a caller's talker knobs.
    ///
    /// The residual codebooks carry acoustic detail, and `top_p` is 1.0 here: truncating their
    /// distribution pushes every frame's timbre toward the mode while codebook 0 keeps the words
    /// intelligible, which sounds *metallic* rather than obviously wrong. Inheriting a talker
    /// `top_p` of 0.9 did exactly that.
    ///
    /// `repetition_penalty` is 1.0 — the reference applies none to the residuals, and penalising
    /// them would fight the codec's own use of repeated acoustic tokens.
    pub fn subtalker(greedy: bool) -> Self {
        Self {
            temperature: pk::TEMPERATURE,
            top_p: pk::TOP_P,
            top_k: pk::TOP_K,
            repetition_penalty: 1.0,
            greedy,
        }
    }
}

/// Top-k then top-p over one row of logits, with an optional repetition penalty.
///
/// `banned` ids are removed before anything else — the talker's `suppress_tokens` covers the
/// top 1024 of its vocabulary except `codec_eos`, so without this the sampler can emit a
/// control id as if it were a code.
fn sample(
    logits: &[f32],
    seen: &[u32],
    banned: &dyn Fn(usize) -> bool,
    s: &Sampling,
    rng: &mut Rng,
) -> usize {
    sample_with_u(logits, seen, banned, s, rng.next_f32())
}

/// The body of [`sample`] with the draw supplied, so the device path can be compared against it
/// exactly rather than distributionally.
fn sample_with_u(
    logits: &[f32],
    seen: &[u32],
    banned: &dyn Fn(usize) -> bool,
    s: &Sampling,
    u_draw: f32,
) -> usize {
    let mut l: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if banned(i) { f32::NEG_INFINITY } else { x })
        .collect();

    if s.repetition_penalty != 1.0 {
        for &id in seen {
            let i = id as usize;
            if i < l.len() && l[i].is_finite() {
                // HF's convention: divide when positive, multiply when negative, so the
                // penalty always moves a logit down.
                l[i] = if l[i] > 0.0 {
                    l[i] / s.repetition_penalty
                } else {
                    l[i] * s.repetition_penalty
                };
            }
        }
    }

    if s.greedy {
        return l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let mut order: Vec<usize> = (0..l.len()).filter(|&i| l[i].is_finite()).collect();
    order.sort_unstable_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
    if s.top_k > 0 && order.len() > s.top_k {
        order.truncate(s.top_k);
    }

    let t = if s.temperature > 0.0 {
        s.temperature
    } else {
        1.0
    };
    let max = l[order[0]];
    let mut probs: Vec<f32> = order.iter().map(|&i| ((l[i] - max) / t).exp()).collect();
    let total: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }

    // top_p == 1.0 is the reference's default, so the nucleus cut is usually inert; keep it
    // anyway because a caller can lower it.
    let mut keep = probs.len();
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= s.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    let renorm: f32 = probs[..keep].iter().sum();
    let mut u = u_draw * renorm;
    for i in 0..keep {
        u -= probs[i];
        if u <= 0.0 {
            return order[i];
        }
    }
    order[keep - 1]
}

/// Per-lane frames, per-lane unconsumed trailing text, and the timing.
pub type BatchOutput = (Vec<Vec<Vec<u32>>>, Vec<usize>, Timing);

/// Codebooks 1..15, conditioned on the talker's hidden state and codebook 0.
struct Predictor {
    stack: Stack,
    /// `[DIM, EMBED_DIM]` with bias: talker width -> predictor width.
    resize: Linear,
    /// 15 output heads. Quantized with the trunk: f32 is 8 MB each and all fifteen are read
    /// every frame, which at q8_0 is 4.8 ms/frame of pure weight traffic.
    heads: Vec<Proj>,
    /// The 15 codebook tables at *talker* width, concatenated to `[15 * CODES, DIM]`.
    ///
    /// One tensor rather than fifteen so a frame's embeddings are a single `index_select` of
    /// 15 rows plus one sum — two dispatches where the incremental form spent 15 gathers,
    /// 15 casts and 15 adds. Safe because the running sum is never read inside the depth loop.
    tables: Tensor,
    /// `tables[i] @ resize`, bias folded in: `[CODES, predictor::DIM]`.
    ///
    /// Every depth step fed `resize` a single row of `tables[step]`, so the projection is a
    /// row lookup once precomputed — 15 matmuls over an 8 MB weight become 15 row fetches,
    /// ~126 MB/frame of weight traffic gone. Exact, not an approximation.
    resized: Vec<Tensor>,
}

impl Predictor {
    fn load(w: &Weights, how: Weight, device: &Device) -> Result<Self> {
        let geo = Geometry {
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
        // 17 positions at most: 2 prefilled plus 15 steps.
        let stack = Stack::load(
            w,
            "talker.code_predictor.model.",
            geo,
            how,
            cfg::CODE_GROUPS + 2,
            device,
        )?;
        let mut heads = Vec::with_capacity(pk::HEADS_OUT);
        let mut tables = Vec::with_capacity(pk::HEADS_OUT);
        for i in 0..pk::HEADS_OUT {
            heads.push(Proj::load_as(
                w,
                &format!("talker.code_predictor.lm_head.{i}.weight"),
                how,
                device,
            )?);
            // Raw dtype: 15 x [2048, 2048] is 503 MB as f32, and each use selects one row.
            tables.push(w.raw(&format!(
                "talker.code_predictor.model.codec_embedding.{i}.weight"
            ))?);
        }
        let resize = Linear::load(w, "talker.code_predictor.small_to_mtp_projection", true)?;
        let resized = tables
            .iter()
            .map(|t| resize.forward(&t.to_dtype(DType::F32)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            stack,
            resize,
            heads,
            tables: Tensor::cat(&tables, 0)?.contiguous()?,
            resized,
        })
    }

    /// The 15 residual codes for one frame, plus their embeddings summed at talker width.
    ///
    /// `hidden` and `code0_embed` are both `[1, 1, talker::DIM]`.
    fn frame(
        &self,
        hidden: &Tensor,
        code0_embed: &Tensor,
        sub: &Sampling,
        rng: &mut Rng,
        timing: &mut Timing,
    ) -> Result<(Vec<u32>, Tensor)> {
        let mut state = self.stack.new_state(1)?;
        // Prefill both positions at once; `generation_steps` is 0 for the first head.
        let prefill = Tensor::cat(&[hidden, code0_embed], 1)?.contiguous()?;
        let mut h = self
            .stack
            .forward(&self.resize.forward(&prefill)?, &mut state)?;

        // Sampled on the host. Doing it on device to avoid fifteen round trips per frame was
        // tried and reverted — candle 0.10.2's Metal `sort_last_dim` silently returns zeros
        // above n=1024 and the vocabulary here is 2048. See `docs/reference.md#what-did-not-work`.
        let mut codes = Vec::with_capacity(pk::HEADS_OUT);
        // Off unless `QWEN3TTS_TIMING`: the syncs it needs are themselves a cost.
        let split = timing_on();
        let sync = || -> Result<()> {
            if split {
                self.stack.device().synchronize()?;
            }
            Ok(())
        };
        sync()?;
        for step in 0..pk::HEADS_OUT {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let t = Instant::now();
            let row = self.heads[step].forward(&last.reshape((1, pk::DIM))?)?;
            sync()?;
            if split {
                timing.depth_gemm_s += t.elapsed().as_secs_f64();
            }
            let t_read = Instant::now();
            let logits = row.reshape(pk::VOCAB)?.to_vec1::<f32>()?;
            if split {
                timing.depth_read_s += t_read.elapsed().as_secs_f64();
            }
            timing.sample_s += t.elapsed().as_secs_f64();
            // Residual codebooks have no control ids: the whole 2048 is live.
            let code = sample(&logits, &[], &|_| false, sub, rng) as u32;
            codes.push(code);

            if step + 1 < pk::HEADS_OUT {
                let idx = Tensor::from_vec(vec![code], 1, self.stack.device())?;
                let t = Instant::now();
                let projected =
                    self.resized[step]
                        .index_select(&idx, 0)?
                        .reshape((1, 1, pk::DIM))?;
                h = self.stack.forward(&projected, &mut state)?;
                sync()?;
                if split {
                    timing.depth_stack_s += t.elapsed().as_secs_f64();
                }
            }
        }
        let sum = (code0_embed + self.gather_sum(&[&codes], 1)?)?;
        Ok((codes, sum))
    }

    /// [`Self::frame`] for `b` lanes at once: `[b, 1, talker::DIM]` in, `codes[lane][14]` and a
    /// `[b, 1, talker::DIM]` sum out.
    ///
    /// Every lane advances in lockstep — the 15 depth steps are the same count for all of them,
    /// so unlike the talker loop there is nothing to mask or repack. The win is that one read of
    /// the 60 M parameters now serves `b` lanes instead of one; see `qwen3tts-batch`.
    fn frame_batch(
        &self,
        hidden: &Tensor,
        code0_embed: &Tensor,
        sub: &Sampling,
        rng: &mut Rng,
        timing: &mut Timing,
    ) -> Result<(Vec<Vec<u32>>, Tensor)> {
        let b = hidden.dim(0)?;
        let mut state = self.stack.new_state(b)?;
        let prefill = Tensor::cat(&[hidden, code0_embed], 1)?.contiguous()?;
        let mut h = self
            .stack
            .forward(&self.resize.forward(&prefill)?, &mut state)?;

        let mut codes = vec![Vec::with_capacity(pk::HEADS_OUT); b];
        // Same both-ends-synchronised split as `frame`; off unless `QWEN3TTS_TIMING`.
        let split = timing_on();
        let sync = || -> Result<()> {
            if split {
                self.stack.device().synchronize()?;
            }
            Ok(())
        };
        sync()?;
        for step in 0..pk::HEADS_OUT {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let t = Instant::now();
            // One [b, VOCAB] read instead of b of them: the sync, not the bytes, is the cost.
            let row = self.heads[step].forward(&last.reshape((b, pk::DIM))?)?;
            sync()?;
            if split {
                timing.depth_gemm_s += t.elapsed().as_secs_f64();
            }
            let t_read = Instant::now();
            let rows = row.to_vec2::<f32>()?;
            if split {
                timing.depth_read_s += t_read.elapsed().as_secs_f64();
            }
            timing.sample_s += t.elapsed().as_secs_f64();

            let mut picked = Vec::with_capacity(b);
            for (lane, logits) in rows.iter().enumerate() {
                let code = sample(logits, &[], &|_| false, sub, rng) as u32;
                codes[lane].push(code);
                picked.push(code);
            }

            if step + 1 < pk::HEADS_OUT {
                let idx = Tensor::from_vec(picked, b, self.stack.device())?;
                let t = Instant::now();
                let projected =
                    self.resized[step]
                        .index_select(&idx, 0)?
                        .reshape((b, 1, pk::DIM))?;
                h = self.stack.forward(&projected, &mut state)?;
                sync()?;
                if split {
                    timing.depth_stack_s += t.elapsed().as_secs_f64();
                }
            }
        }
        let lanes: Vec<&Vec<u32>> = codes.iter().collect();
        let sum = (code0_embed + self.gather_sum(&lanes, b)?)?;
        Ok((codes, sum))
    }

    /// Codebooks 1..15 embedded at talker width and summed, `[b, 1, EMBED_DIM]`.
    ///
    /// `codes[lane][step]` indexes the concatenated table at `step * CODES + code`, so the
    /// whole frame is one gather and one sum.
    fn gather_sum(&self, codes: &[&Vec<u32>], b: usize) -> Result<Tensor> {
        let mut idx = Vec::with_capacity(b * pk::HEADS_OUT);
        for lane in codes {
            if lane.len() != pk::HEADS_OUT {
                bail!(
                    "expected {} residual codes, got {}",
                    pk::HEADS_OUT,
                    lane.len()
                );
            }
            for (step, &c) in lane.iter().enumerate() {
                idx.push((step * tk::CODES) as u32 + c);
            }
        }
        let idx = Tensor::from_vec(idx, b * pk::HEADS_OUT, self.stack.device())?;
        Ok(self
            .tables
            .index_select(&idx, 0)?
            .to_dtype(DType::F32)?
            .reshape((b, pk::HEADS_OUT, pk::EMBED_DIM))?
            .sum_keepdim(1)?
            .reshape((b, 1, pk::EMBED_DIM))?)
    }
}

pub struct Talker {
    stack: Stack,
    /// `[VOCAB, DIM]` — codec-side embeddings, and *not* tied to `codec_head`.
    codec_embed: Tensor,
    /// `[TEXT_VOCAB, TEXT_DIM]`, kept in the checkpoint's dtype.
    ///
    /// 151936 x 2048 is 1.24 GB as f32 and 622 MB as bf16, and every use selects a handful of
    /// rows — so it stays raw and only the selected rows are cast.
    text_embed: Tensor,
    /// `codec_head`. Quantized with the trunk — it is 25 MB as f32 and read once a frame.
    head: Proj,
    /// `linear_fc1` -> SiLU -> `linear_fc2`, both biased (trap 9).
    text_fc1: Linear,
    text_fc2: Linear,
    predictor: Predictor,
    device: Device,
}

impl Talker {
    pub fn load(path: &str, how: Weight, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let geo = Geometry {
            dim: tk::DIM,
            layers: tk::LAYERS,
            heads: tk::HEADS,
            n_kv: tk::N_KV,
            head_dim: tk::HEAD_DIM,
            ffn: tk::FFN,
            eps: tk::NORM_EPS,
            rope_base: tk::ROPE_BASE,
            qk_norm: tk::QK_NORM,
            layer_scale: false,
            window: tk::SLIDING_WINDOW,
        };
        Ok(Self {
            stack: Stack::load(&w, "talker.model.", geo, how, CAPACITY, device)?,
            codec_embed: w.get("talker.model.codec_embedding.weight")?,
            text_embed: w.raw("talker.model.text_embedding.weight")?,
            head: Proj::load_as(&w, "talker.codec_head.weight", how, device)?,
            text_fc1: Linear::load(&w, "talker.text_projection.linear_fc1", true)?,
            text_fc2: Linear::load(&w, "talker.text_projection.linear_fc2", true)?,
            predictor: Predictor::load(&w, how, device)?,
            device: device.clone(),
        })
    }

    /// `text_projection`: embed text ids and project into the talker's width.
    fn text_hidden(&self, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        let e = self
            .text_embed
            .index_select(&idx, 0)?
            .to_dtype(DType::F32)?
            .reshape((1, ids.len(), tk::TEXT_DIM))?;
        let h = candle_nn::ops::silu(&self.text_fc1.forward(&e)?)?;
        self.text_fc2.forward(&h)
    }

    fn codec_row(&self, id: u32) -> Result<Tensor> {
        let idx = Tensor::from_vec(vec![id], 1, &self.device)?;
        Ok(self
            .codec_embed
            .index_select(&idx, 0)?
            .reshape((1, 1, tk::DIM))?)
    }

    /// Embed one frame of 16 codes and sum at talker width — codebook 0 from the talker's
    /// table, 1..15 from the predictor's.
    fn frame_embed(&self, codes: &[u32]) -> Result<Tensor> {
        if codes.len() != cfg::CODE_GROUPS {
            bail!("expected {} codes, got {}", cfg::CODE_GROUPS, codes.len());
        }
        let rest = codes[1..].to_vec();
        Ok((self.codec_row(codes[0])? + self.predictor.gather_sum(&[&rest], 1)?)?)
    }

    /// The prompt, and the text hidden states the decode loop consumes one per frame.
    ///
    /// `text` is the target text's BPE ids; `ref_text`/`ref_codes` are the voice asset's, and
    /// enable in-context cloning when present. `spk` is the x-vector, `[1, DIM]`.
    ///
    /// Mirrors `Qwen3TTSForConditionalGeneration::generate`'s streaming path.
    pub fn build_prompt(
        &self,
        text: &[u32],
        ref_text: &[u32],
        ref_codes: &[Vec<u32>],
        spk: Option<&Tensor>,
        language: Option<u32>,
    ) -> Result<(Tensor, Tensor)> {
        let pad = self.text_hidden(&[tk::TTS_PAD])?;
        let bos = self.text_hidden(&[tk::TTS_BOS])?;
        let eos = self.text_hidden(&[tk::TTS_EOS])?;

        // Codec-stream prefill: the think/nothink tag, then optionally the x-vector, then
        // pad and bos.
        let tags: Vec<u32> = match language {
            Some(lang) => vec![
                tk::CODEC_THINK,
                tk::CODEC_THINK_BOS,
                lang,
                tk::CODEC_THINK_EOS,
            ],
            None => vec![tk::CODEC_NOTHINK, tk::CODEC_THINK_BOS, tk::CODEC_THINK_EOS],
        };
        let mut codec: Vec<Tensor> = Vec::new();
        for t in &tags {
            codec.push(self.codec_row(*t)?);
        }
        if let Some(spk) = spk {
            codec.push(spk.reshape((1, 1, tk::DIM))?);
        }
        codec.push(self.codec_row(tk::CODEC_PAD)?);
        codec.push(self.codec_row(tk::CODEC_BOS)?);
        let codec = Tensor::cat(&codec, 1)?.contiguous()?;
        let l = codec.dim(1)?;

        // Text stream over the prefill: pads then bos, added to every codec position but the
        // last. The final `codec_bos` is consumed by the ICL block or the first text token.
        let mut text_side = vec![pad.clone(); l - 2];
        text_side.push(bos.clone());
        let text_side = Tensor::cat(&text_side, 1)?;
        let prefix = (text_side + codec.narrow(1, 0, l - 1)?)?;

        let role = self.text_hidden(&[tk::IM_START, tk::ASSISTANT, 198])?;
        let mut parts = vec![role, prefix];

        let trailing;
        if !ref_codes.is_empty() && !ref_text.is_empty() {
            // ICL: text is ref transcript + target text + eos; the codec stream is codec_bos
            // followed by the reference frames. The two are summed position-wise and
            // whichever is longer decides what spills into `trailing`.
            let mut ids = ref_text.to_vec();
            ids.extend_from_slice(text);
            let text_embed = Tensor::cat(&[self.text_hidden(&ids)?, eos.clone()], 1)?;

            let mut codec_parts = vec![self.codec_row(tk::CODEC_BOS)?];
            for frame in ref_codes {
                codec_parts.push(self.frame_embed(frame)?);
            }
            let codec_embed = Tensor::cat(&codec_parts, 1)?.contiguous()?;

            let text_len = text_embed.dim(1)?;
            let codec_len = codec_embed.dim(1)?;
            if text_len > codec_len {
                parts.push((text_embed.narrow(1, 0, codec_len)? + &codec_embed)?);
                trailing = text_embed.narrow(1, codec_len, text_len - codec_len)?;
            } else {
                let mut padded = vec![text_embed];
                for _ in 0..codec_len - text_len {
                    padded.push(pad.clone());
                }
                parts.push((Tensor::cat(&padded, 1)? + &codec_embed)?);
                trailing = pad.clone();
            }
        } else {
            // x-vector only: the first text token rides on `codec_bos`, the rest trails.
            if text.is_empty() {
                bail!("no text to speak");
            }
            let first = self.text_hidden(&text[..1])?;
            parts.push((first + codec.narrow(1, l - 1, 1)?)?);
            trailing = if text.len() > 1 {
                Tensor::cat(&[self.text_hidden(&text[1..])?, eos.clone()], 1)?
            } else {
                eos.clone()
            };
        }
        Ok((Tensor::cat(&parts, 1)?.contiguous()?, trailing))
    }

    /// Decode one segment. Returns `[frames][16]` codes and how many trailing text
    /// positions were never consumed — trap 3's exact truncation measure.
    pub fn generate(
        &self,
        prompt: &Tensor,
        trailing: &Tensor,
        max_new: usize,
        s: &Sampling,
        rng: &mut Rng,
    ) -> Result<(Vec<Vec<u32>>, usize, Timing)> {
        let mut state = self.stack.new_state(1)?;
        let mut timing = Timing::default();
        let clock = timing_on();
        let t_pre = Instant::now();
        let mut h = self.stack.forward(prompt, &mut state)?;
        if clock {
            self.stack.device().synchronize()?;
        }
        timing.prefill_s = t_pre.elapsed().as_secs_f64();
        let trailing_len = trailing.dim(1)?;
        let pad = trailing.narrow(1, trailing_len - 1, 1)?;

        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut seen: Vec<u32> = Vec::new();
        let mut step = 0usize;
        // Never the caller's talker settings — see `Sampling::subtalker`.
        let sub = Sampling::subtalker(s.greedy);

        // `suppress_tokens`: the top 1024 of VOCAB except eos. Codes live below CODES.
        let banned = |i: usize| i >= tk::VOCAB - 1024 && i != tk::CODEC_EOS as usize;

        loop {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let t = Instant::now();
            let logits = self
                .head
                .forward(&last.reshape((1, tk::DIM))?)?
                .reshape(tk::VOCAB)?
                .to_vec1::<f32>()?;
            timing.sample_s += t.elapsed().as_secs_f64();
            // `min_new_tokens=2`: refuse eos before the model has produced anything usable.
            let code0 = if frames.len() < tk::MIN_NEW_TOKENS {
                sample(
                    &logits,
                    &seen,
                    &|i| banned(i) || i == tk::CODEC_EOS as usize,
                    s,
                    rng,
                )
            } else {
                sample(&logits, &seen, &banned, s, rng)
            } as u32;

            if code0 == tk::CODEC_EOS {
                break;
            }
            seen.push(code0);

            let code0_embed = self.codec_row(code0)?;
            let t = Instant::now();
            let (rest, mut sum) =
                self.predictor
                    .frame(&last, &code0_embed, &sub, rng, &mut timing)?;
            if clock {
                self.stack.device().synchronize()?;
            }
            timing.predictor_s += t.elapsed().as_secs_f64();
            let mut frame = vec![code0];
            frame.extend_from_slice(&rest);
            frames.push(frame);

            step += 1;
            if step >= max_new {
                break;
            }

            // Trap 3: the next text token rides on the codec sum, and once the text runs out
            // it is `tts_pad` forever.
            let text_next = if step - 1 < trailing_len {
                trailing.narrow(1, step - 1, 1)?
            } else {
                pad.clone()
            };
            sum = (sum + text_next)?;
            let t = Instant::now();
            h = self.stack.forward(&sum, &mut state)?;
            if clock {
                self.stack.device().synchronize()?;
            }
            timing.talker_s += t.elapsed().as_secs_f64();
        }

        timing.frames = frames.len();
        let consumed = step.saturating_sub(1);
        Ok((frames, trailing_len.saturating_sub(consumed), timing))
    }

    /// Decode `b` segments at once. `prompt` is `[b, L, DIM]`, `trailing` is `[b, Lt, DIM]`.
    ///
    /// Requires every lane to share `L` and `Lt`. That is not the restriction it looks like: in
    /// the ICL path the prompt is `role + prefix + max(text_len, codec_len)` positions, and
    /// `codec_len` is fixed by the *voice*, so every segment whose text fits inside the
    /// reference block — a hundred-odd tokens, far more than one sentence — comes out exactly
    /// the same length, with `trailing` a single `tts_pad`. The caller groups by length and
    /// falls back to [`Self::generate`] for anything that does not fit, rather than left-padding
    /// and carrying a per-lane mask and position offset through the whole stack.
    ///
    /// Lanes finish at different frames. A finished lane keeps being computed — its codes are
    /// simply no longer recorded — because repacking the KV cache mid-decode would cost more
    /// than the wasted steps for the segment lengths this sees.
    ///
    /// Returns per-lane frames and per-lane unconsumed trailing positions.
    pub fn generate_batch(
        &self,
        prompt: &Tensor,
        trailing: &Tensor,
        max_new: usize,
        s: &Sampling,
        rng: &mut Rng,
    ) -> Result<BatchOutput> {
        let b = prompt.dim(0)?;
        if trailing.dim(0)? != b {
            bail!("prompt batch {b} != trailing batch {}", trailing.dim(0)?);
        }
        // Only what the lanes can actually reach. At 229 KB per position per lane the full
        // 1536 would be 2.8 GB for eight lanes, which on a 16 GB machine is the difference
        // between batching and swapping.
        let need = prompt.dim(1)? + max_new + 1;
        let mut state = self.stack.new_state_with(b, need)?;
        let mut timing = Timing::default();
        let clock = timing_on();
        let t_pre = Instant::now();
        let mut h = self.stack.forward(prompt, &mut state)?;
        if clock {
            self.stack.device().synchronize()?;
        }
        timing.prefill_s = t_pre.elapsed().as_secs_f64();

        let trailing_len = trailing.dim(1)?;
        let pad = trailing.narrow(1, trailing_len - 1, 1)?;
        let sub = Sampling::subtalker(s.greedy);
        let banned = |i: usize| i >= tk::VOCAB - 1024 && i != tk::CODEC_EOS as usize;

        let mut frames: Vec<Vec<Vec<u32>>> = vec![Vec::new(); b];
        let mut seen: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        let mut consumed = vec![0usize; b];
        let mut step = 0usize;

        loop {
            let last = h.narrow(1, h.dim(1)? - 1, 1)?;
            let t = Instant::now();
            let rows = self
                .head
                .forward(&last.reshape((b, tk::DIM))?)?
                .to_vec2::<f32>()?;
            timing.sample_s += t.elapsed().as_secs_f64();

            let mut code0 = Vec::with_capacity(b);
            for (lane, logits) in rows.iter().enumerate() {
                // A finished lane still needs *a* code to keep its row of the batch well
                // formed; it is never recorded.
                if done[lane] {
                    code0.push(0);
                    continue;
                }
                let c = if frames[lane].len() < tk::MIN_NEW_TOKENS {
                    sample(
                        logits,
                        &seen[lane],
                        &|i| banned(i) || i == tk::CODEC_EOS as usize,
                        s,
                        rng,
                    )
                } else {
                    sample(logits, &seen[lane], &banned, s, rng)
                } as u32;
                if c == tk::CODEC_EOS {
                    done[lane] = true;
                    consumed[lane] = step;
                    code0.push(0);
                } else {
                    seen[lane].push(c);
                    code0.push(c);
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }

            let idx = Tensor::from_vec(code0.clone(), b, &self.device)?;
            let code0_embed = self
                .codec_embed
                .index_select(&idx, 0)?
                .reshape((b, 1, tk::DIM))?;

            let t = Instant::now();
            let (rest, mut sum) =
                self.predictor
                    .frame_batch(&last, &code0_embed, &sub, rng, &mut timing)?;
            if clock {
                self.stack.device().synchronize()?;
            }
            timing.predictor_s += t.elapsed().as_secs_f64();

            for lane in 0..b {
                if done[lane] {
                    continue;
                }
                let mut frame = vec![code0[lane]];
                frame.extend_from_slice(&rest[lane]);
                frames[lane].push(frame);
            }

            step += 1;
            if step >= max_new {
                break;
            }

            let text_next = if step - 1 < trailing_len {
                trailing.narrow(1, step - 1, 1)?
            } else {
                pad.clone()
            };
            sum = (sum + text_next)?;
            let t = Instant::now();
            h = self.stack.forward(&sum, &mut state)?;
            if clock {
                self.stack.device().synchronize()?;
            }
            timing.talker_s += t.elapsed().as_secs_f64();
        }

        // Frames counted once per lane-frame, so `ms/frame` stays comparable with batch 1.
        timing.frames = frames.iter().map(|f| f.len()).sum();
        timing.steps = step;
        timing.lanes = b;
        let left = (0..b)
            .map(|lane| {
                let used = if done[lane] { consumed[lane] } else { step };
                trailing_len.saturating_sub(used.saturating_sub(1))
            })
            .collect();
        Ok((frames, left, timing))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Prefill only: the last normed hidden state and `codec_head` over it. For the gate.
    pub fn trace(&self, prompt: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut state = self.stack.new_state(1)?;
        let h = self.stack.forward(prompt, &mut state)?;
        let last = h.narrow(1, h.dim(1)? - 1, 1)?;
        let logits = self
            .head
            .forward(&last.reshape((1, tk::DIM))?)?
            .reshape((1, 1, tk::VOCAB))?;
        Ok((last, logits))
    }

    /// The loop's next-frame input: the frame's 16 codebook embeddings summed, plus the next
    /// text position. For the gate — this is trap 3's update in isolation.
    pub fn step_input(&self, frame: &[u32], text_next: &Tensor) -> Result<Tensor> {
        Ok((self.frame_embed(frame)? + text_next)?)
    }

    /// Prefill then one decode step, returning that step's hidden state and logits.
    pub fn trace_step(&self, prompt: &Tensor, step_input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut state = self.stack.new_state(1)?;
        self.stack.forward(prompt, &mut state)?;
        let h = self.stack.forward(step_input, &mut state)?;
        let last = h.narrow(1, h.dim(1)? - 1, 1)?;
        let logits = self
            .head
            .forward(&last.reshape((1, tk::DIM))?)?
            .reshape((1, 1, tk::VOCAB))?;
        Ok((last, logits))
    }

    /// One frame of the depth predictor, teacher-forced on `code0`. For the gate.
    pub fn predict_frame(&self, hidden: &Tensor, code0: u32, greedy: bool) -> Result<Vec<u32>> {
        let s = Sampling::subtalker(greedy);
        let mut rng = Rng::new(0);
        let mut timing = Timing::default();
        let embed = self.codec_row(code0)?;
        let (codes, _) = self
            .predictor
            .frame(hidden, &embed, &s, &mut rng, &mut timing)?;
        Ok(codes)
    }

    /// The x-vector, reshaped and moved onto this engine's device.
    pub fn speaker(&self, spk: &Tensor) -> Result<Tensor> {
        let t = spk.to_device(&self.device)?.to_dtype(DType::F32)?;
        let n = t.elem_count();
        if n != tk::DIM {
            bail!("x-vector is {n}-wide, expected {} — 0.6B asset?", tk::DIM);
        }
        Ok(t.reshape((1, tk::DIM))?)
    }
}
