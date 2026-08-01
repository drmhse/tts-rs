//! `CosyVoice3LM`: a Qwen2-0.5B trunk with a speech-token head.
//!
//! The prompt is `[sos] + text_embeddings + [task_id] + prompt_speech_embeddings`, decoded
//! one speech token at a time until a control id appears. Two embedding tables are in
//! play and mixing them up is a silent error: **text ids index Qwen2's
//! `embed_tokens` (151936 rows) while `sos`, `task_id` and every generated token index
//! `speech_embedding` (6761 rows)**. They are different matrices of the same width.
//!
//! # What transfers from Audio8, and what does not
//!
//! The trunk geometry is *identical* to Audio8's slow AR — dim 896, 24 layers, 14 query
//! heads over 2 KV heads, head_dim 64, ffn 4864, RMS eps 1e-6, RoPE base 1e6 — so the two
//! structural levers that mattered there apply unchanged:
//!
//! - **Narrow KV.** Attention runs over `0..=pos` rather than a `max_seq_len`-wide buffer
//!   with a mask. Worth 5.10x on Audio8 and the arithmetic is the same here.
//! - **GQA by query reshape.** The 7 query heads sharing a KV head fold into the matmul's
//!   row dimension instead of materialising a 7x copy of K and V per layer per token.
//!
//! Two things do *not* transfer:
//!
//! - **No sliced logit head.** Audio8's head could be cut from 155776 rows to 4097
//!   because a semantic mask ran first and made the rest unreachable. Here the head is
//!   `[6761, 896]` and all of it is live — `ras_sampling` can select any speech token,
//!   and the control ids are the stop condition. 6761 rows is small enough that this
//!   costs nothing; the point is that the trick was specific to Audio8's masking, not a
//!   general one.
//! - **A different sampler.** `ras_sampling` is nucleus sampling with a repetition
//!   guard, not Gumbel-max. See [`crate::sample`].
//!
//! # Qwen2 specifics
//!
//! Separate `q_proj`, `k_proj`, `v_proj` — with **biases on all three**, which Qwen2 has
//! and most Llama-shaped models do not — rather than Audio8's fused `wqkv`. The port
//! fuses them at load into one `[1152, 896]` matrix so a decode step issues one matmul
//! instead of three, which is the same shape Audio8's checkpoint ships in.
//!
//! The hidden state taken is `hidden_states[-1]`, which in HF's `Qwen2Model` is *after*
//! the final `norm`. Taking the last layer's raw output instead would be off by one
//! RMSNorm and produce logits that look reasonable.

use crate::cfg::llm as k;
use crate::sample::{ras_sampling, Sampler};
use anyhow::{bail, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor, D};
use tts_core::rng::Rng;
use tts_nn::{rms_norm, rope_table_f32, Proj, Weights};

/// Preallocated K/V for one layer, `[batch, n_kv, capacity, head_dim]`.
struct Cache {
    k: Tensor,
    v: Tensor,
}

impl Cache {
    fn new(capacity: usize, batch: usize, device: &Device) -> Result<Self> {
        let shape = (batch, k::N_KV, capacity, k::HEAD_DIM);
        Ok(Self {
            k: Tensor::zeros(shape, DType::F32, device)?,
            v: Tensor::zeros(shape, DType::F32, device)?,
        })
    }
}

/// One lane of a batch: how many left-pad columns it carries.
///
/// See [`Llm::prefill_batch`] for why right-alignment is sound.
#[derive(Clone, Copy, Debug)]
struct Lane {
    pad: usize,
}

/// Decode state: the caches plus where in them we are.
pub struct State {
    caches: Vec<Cache>,
    lanes: Vec<Lane>,
    /// Positions already written, including each lane's left padding.
    pub width: usize,
    /// The last position's normed hidden state, `[batch, 896]`.
    pub hidden: Tensor,
}

impl State {
    fn batch(&self) -> usize {
        self.lanes.len()
    }
}

struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// The fused `[1152, 896]` q/k/v projection and its `[1152]` bias.
    wqkv: Proj,
    bqkv: Tensor,
    wo: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Layer {
    fn load(w: &Weights, i: usize, quant: Option<GgmlDType>, device: &Device) -> Result<Self> {
        let p = format!("llm.model.model.layers.{i}");
        // Fuse q/k/v at load. Qwen2 stores them separately and biases all three.
        let wq = w.get(&format!("{p}.self_attn.q_proj.weight"))?;
        let wk = w.get(&format!("{p}.self_attn.k_proj.weight"))?;
        let wv = w.get(&format!("{p}.self_attn.v_proj.weight"))?;
        let wqkv = Tensor::cat(&[wq, wk, wv], 0)?.contiguous()?;
        let bq = w.get(&format!("{p}.self_attn.q_proj.bias"))?;
        let bk = w.get(&format!("{p}.self_attn.k_proj.bias"))?;
        let bv = w.get(&format!("{p}.self_attn.v_proj.bias"))?;

        Ok(Self {
            attn_norm: w.get(&format!("{p}.input_layernorm.weight"))?,
            ffn_norm: w.get(&format!("{p}.post_attention_layernorm.weight"))?,
            wqkv: Proj::from_tensor(&wqkv, quant, device)?,
            bqkv: Tensor::cat(&[bq, bk, bv], 0)?.contiguous()?,
            wo: Proj::load(w, &format!("{p}.self_attn.o_proj.weight"), quant, device)?,
            gate: Proj::load(w, &format!("{p}.mlp.gate_proj.weight"), quant, device)?,
            up: Proj::load(w, &format!("{p}.mlp.up_proj.weight"), quant, device)?,
            down: Proj::load(w, &format!("{p}.mlp.down_proj.weight"), quant, device)?,
        })
    }
}

pub struct Llm {
    /// `[151936, 896]` — Qwen2's text embeddings.
    text_embed: Tensor,
    /// `[6761, 896]` — speech tokens and the control ids.
    speech_embed: Tensor,
    /// `[896, 6761]`, pre-transposed: the head runs every decode step.
    head_t: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    cos: Tensor,
    sin: Tensor,
    capacity: usize,
    device: Device,
}

impl Llm {
    /// `quant` applies to the 24 layers' projections only; embeddings, the head and the
    /// norms stay f32, as in the Audio8 port.
    pub fn load(path: &str, quant: Option<GgmlDType>, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let layers = (0..k::LAYERS)
            .map(|i| Layer::load(&w, i, quant, device))
            .collect::<Result<Vec<_>>>()?;

        // Generous but bounded: 4096 positions is ~164 s of speech tokens, well past
        // what one segment can be, and the cache is the only thing that grows with
        // length. Allocating it up front is what keeps memory flat by construction.
        let capacity = 4096;
        let (cos, sin) = rope_table_f32(capacity, k::HEAD_DIM, k::ROPE_BASE, device)?;

        let head = w.get("llm_decoder.weight")?;
        let llm = Self {
            text_embed: w.get("llm.model.model.embed_tokens.weight")?,
            speech_embed: w.get("speech_embedding.weight")?,
            head_t: head.t()?.contiguous()?,
            norm: w.get("llm.model.model.norm.weight")?,
            layers,
            cos,
            sin,
            capacity,
            device: device.clone(),
        };
        llm.check_geometry()?;
        Ok(llm)
    }

    fn check_geometry(&self) -> Result<()> {
        if self.speech_embed.dim(0)? != k::VOCAB {
            bail!(
                "speech_embedding has {} rows, expected {}",
                self.speech_embed.dim(0)?,
                k::VOCAB
            );
        }
        if self.head_t.dim(1)? != k::VOCAB {
            bail!(
                "llm_decoder emits {} logits, expected {}",
                self.head_t.dim(1)?,
                k::VOCAB
            );
        }
        if self.text_embed.dim(1)? != k::DIM {
            bail!(
                "embed_tokens width {} != {}",
                self.text_embed.dim(1)?,
                k::DIM
            );
        }
        Ok(())
    }

    /// `[sos] + text + [task_id] + prompt_speech` as embeddings, `[1, width, 896]`.
    ///
    /// `text` must already be the concatenation of the voice's transcript tokens and the
    /// target text's, and must contain `<|endofprompt|>` — the reference asserts it and
    /// nothing in its frontend adds it.
    pub fn build_prompt(&self, text: &[u32], prompt_speech: &[u32]) -> Result<Tensor> {
        if !text.contains(&k::ENDOFPROMPT) {
            bail!(
                "the prompt text has no <|endofprompt|> (id {}) — the voice asset's \
                 `prompt_text` must carry it; see trap 2 in docs/porting/cosyvoice.md",
                k::ENDOFPROMPT
            );
        }
        let row = |table: &Tensor, id: usize| -> Result<Tensor> {
            let i = Tensor::from_vec(vec![id as u32], 1, &self.device)?;
            Ok(table.index_select(&i, 0)?.reshape((1, 1, k::DIM))?)
        };
        let text_idx = Tensor::from_vec(text.to_vec(), text.len(), &self.device)?;
        let text_emb =
            self.text_embed
                .index_select(&text_idx, 0)?
                .reshape((1, text.len(), k::DIM))?;

        let mut parts = vec![
            row(&self.speech_embed, k::SOS)?,
            text_emb,
            row(&self.speech_embed, k::TASK_ID)?,
        ];
        if !prompt_speech.is_empty() {
            let idx = Tensor::from_vec(prompt_speech.to_vec(), prompt_speech.len(), &self.device)?;
            parts.push(self.speech_embed.index_select(&idx, 0)?.reshape((
                1,
                prompt_speech.len(),
                k::DIM,
            ))?);
        }
        Ok(Tensor::cat(&parts, 1)?.contiguous()?)
    }

    /// Run the prompt through, filling the caches. Returns the state at its last
    /// position.
    pub fn prefill(&self, lm_input: &Tensor) -> Result<State> {
        self.prefill_batch(std::slice::from_ref(lm_input))
    }

    /// Run several prompts through at once, right-aligned into one batch.
    ///
    /// Each prompt is padded on the *left* to the widest, so all lanes finish at the same
    /// column and one position index serves every lane. That is sound because a RoPE score
    /// depends only on `p - j`: shifting a whole lane by a constant leaves every score it
    /// computes unchanged, and the padding itself is removed by the mask.
    ///
    /// The argument rests on `R(p)^T R(j) = R(p - j)` holding in the table's arithmetic, and
    /// it holds **to f32, not exactly**. `cosyvoice-validate` measures it: an unpadded lane is
    /// bit-identical to decoding it alone, and a left-padded lane comes back at rel 2.0e-6 —
    /// the rotary identity in f32 plus a different reduction order in the batched matmuls.
    /// Audio8 batches the same way on bf16-rounded tables, where the identity only survives
    /// to ~4e-3, so f32 buys about three orders of magnitude.
    ///
    /// It is nevertheless exact where it counts: **greedy decoding of a whole utterance is
    /// byte-identical between `llm_batch=1` and `llm_batch=7`**, because a 2e-6 perturbation
    /// does not move an argmax. Sampled output *does* differ between the two, but only
    /// because lanes draw from the shared RNG interleaved rather than one segment at a time
    /// — a different draw, not a different model.
    pub fn prefill_batch(&self, prompts: &[Tensor]) -> Result<State> {
        let b = prompts.len();
        if b == 0 {
            bail!("prefill_batch needs at least one prompt");
        }
        let width = prompts
            .iter()
            .map(|p| p.dim(1).unwrap_or(0))
            .max()
            .unwrap_or(0);
        if width > self.capacity {
            bail!(
                "prompt is {width} positions, cache capacity is {}",
                self.capacity
            );
        }

        let mut lanes = Vec::with_capacity(b);
        let mut rows = Vec::with_capacity(b);
        for p in prompts {
            let n = p.dim(1)?;
            let pad = width - n;
            lanes.push(Lane { pad });
            rows.push(if pad == 0 {
                p.clone()
            } else {
                // Zero embeddings in the pad columns. Their contribution is removed by the
                // mask, not by the value, so any filler would do.
                let z = Tensor::zeros((1, pad, k::DIM), DType::F32, &self.device)?;
                Tensor::cat(&[&z, p], 1)?
            });
        }
        let batched = Tensor::cat(&rows, 0)?.contiguous()?;

        let mut caches = Vec::with_capacity(k::LAYERS);
        for _ in 0..k::LAYERS {
            caches.push(Cache::new(self.capacity, b, &self.device)?);
        }
        let mut state = State {
            caches,
            lanes,
            width: 0,
            hidden: Tensor::zeros((b, k::DIM), DType::F32, &self.device)?,
        };
        let mask = self.prefill_mask(&state.lanes, width)?;
        let hidden = self.run(&batched, &mut state, 0, mask.as_ref())?;
        state.width = width;
        state.hidden = hidden;
        Ok(state)
    }

    /// `[b, 1, width, width]`: causal, and blind to every lane's left padding.
    ///
    /// `None` when nothing is padded, which keeps the single-prompt path on exactly the
    /// code it used before — plain causal mask, no per-lane tensor.
    fn prefill_mask(&self, lanes: &[Lane], width: usize) -> Result<Option<Tensor>> {
        if lanes.iter().all(|l| l.pad == 0) {
            return Ok(None);
        }
        let b = lanes.len();
        let mut v = vec![0f32; b * width * width];
        for (i, l) in lanes.iter().enumerate() {
            for r in 0..width {
                // A padded *row* is never read, but it must still see something or its
                // softmax is all -inf and produces NaN that propagates through the
                // residual stream into the real rows. Give it ordinary causal visibility.
                let lo = if r < l.pad { 0 } else { l.pad };
                for c in 0..width {
                    if c > r || c < lo {
                        v[i * width * width + r * width + c] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Ok(Some(Tensor::from_vec(
            v,
            (b, 1, width, width),
            &self.device,
        )?))
    }

    /// `[b, 1, 1, span]` for a decode step, or `None` when no lane is padded.
    fn decode_mask(&self, lanes: &[Lane], span: usize) -> Result<Option<Tensor>> {
        if lanes.iter().all(|l| l.pad == 0) {
            return Ok(None);
        }
        let b = lanes.len();
        let mut v = vec![0f32; b * span];
        for (i, l) in lanes.iter().enumerate() {
            for c in 0..l.pad.min(span) {
                v[i * span + c] = f32::NEG_INFINITY;
            }
        }
        Ok(Some(Tensor::from_vec(v, (b, 1, 1, span), &self.device)?))
    }

    /// Feed one generated speech token and advance by a position.
    pub fn step(&self, state: State, token: u32) -> Result<State> {
        self.step_batch(state, &[token])
    }

    /// Feed one token per lane and advance every lane by a position.
    pub fn step_batch(&self, mut state: State, tokens: &[u32]) -> Result<State> {
        if state.width >= self.capacity {
            bail!("cache is full at {} positions", self.capacity);
        }
        if tokens.len() != state.batch() {
            bail!("{} tokens for {} lanes", tokens.len(), state.batch());
        }
        let b = tokens.len();
        let idx = Tensor::from_vec(tokens.to_vec(), b, &self.device)?;
        let emb = self
            .speech_embed
            .index_select(&idx, 0)?
            .reshape((b, 1, k::DIM))?;
        let pos = state.width;
        let mask = self.decode_mask(&state.lanes, pos + 1)?;
        let hidden = self.run(&emb, &mut state, pos, mask.as_ref())?;
        state.width = pos + 1;
        state.hidden = hidden;
        Ok(state)
    }

    /// `[b, 896] -> [b, 6761]`.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(hidden.broadcast_matmul(&self.head_t)?)
    }

    /// The trunk. Writes `t` positions starting at `start` into the caches and returns
    /// the final position's normed hidden state, `[1, 896]`.
    fn run(
        &self,
        x: &Tensor,
        state: &mut State,
        start: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let mut h = x.clone();
        for (li, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&h, &layer.attn_norm, k::NORM_EPS)?;
            let qkv = layer.wqkv.forward(&normed)?.broadcast_add(&layer.bqkv)?;

            let q_width = k::N_HEADS * k::HEAD_DIM;
            let kv_width = k::N_KV * k::HEAD_DIM;
            let q = qkv.narrow(D::Minus1, 0, q_width)?;
            let kk = qkv.narrow(D::Minus1, q_width, kv_width)?;
            let v = qkv.narrow(D::Minus1, q_width + kv_width, kv_width)?;

            let q = q
                .reshape((b, t, k::N_HEADS, k::HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
            let kk = kk
                .reshape((b, t, k::N_KV, k::HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((b, t, k::N_KV, k::HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;

            let cos = self.cos.narrow(0, start, t)?;
            let sin = self.sin.narrow(0, start, t)?;
            // Half-split rotary, *not* interleaved. HF's Qwen2 uses `rotate_half`,
            // which splits the head dimension down the middle; Audio8's Fish-Speech
            // weights use `torch.polar`, which pairs adjacent channels. Same geometry,
            // opposite convention, and both run without error — so this is the one line
            // in the file most likely to be wrong in a port that reuses `audio8::ar`.
            // Measured cost of getting it backwards: hidden state off by rel 0.78.
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let kk = candle_nn::rotary_emb::rope(&kk, &cos, &sin)?;

            let cache = &mut state.caches[li];
            // `slice_set` writes in place; `slice_assign` would allocate and copy the whole
            // cache. That distinction is not cosmetic here: the cache is `[1, 2, 4096, 64]`
            // f32 = 2 MB, and with k and v across 24 layers that is **~100 MB copied per
            // generated token** to write 64 floats per head. Measured at ~9% of this stage.
            cache.k.slice_set(&kk, 2, start)?;
            cache.v.slice_set(&v, 2, start)?;

            // Narrow KV: attend over exactly the written span, never the full capacity.
            let span = start + t;
            let k_all = cache.k.narrow(2, 0, span)?;
            let v_all = cache.v.narrow(2, 0, span)?;

            let attn = if t == 1 {
                // Decode: GQA by query reshape. The 7 query heads sharing a KV head
                // become rows of one matmul instead of a 7x copy of K and V.
                let qg = q.reshape((b, k::N_KV, k::GQA, k::HEAD_DIM))?;
                let scale = 1.0 / (k::HEAD_DIM as f64).sqrt();
                let scores = (qg.matmul(&k_all.transpose(2, 3)?.contiguous()?)? * scale)?;
                // `[b, 1, 1, span]` over `[b, n_kv, gqa, span]`: masks each lane's left pad.
                let scores = match mask {
                    Some(m) => scores.broadcast_add(m)?,
                    None => scores,
                };
                let probs = candle_nn::ops::softmax_last_dim(&scores)?;
                let out = probs.matmul(&v_all.contiguous()?)?;
                out.reshape((b, 1, k::N_HEADS * k::HEAD_DIM))?
            } else {
                // Prefill: repeat the KV heads and mask causally. Done once, so the
                // copy is not worth avoiding here.
                let k_rep = repeat_kv(&k_all)?;
                let v_rep = repeat_kv(&v_all)?;
                let scale = 1.0 / (k::HEAD_DIM as f64).sqrt();
                let scores = (q.matmul(&k_rep.transpose(2, 3)?.contiguous()?)? * scale)?;
                let causal = tts_nn::causal_window_mask(t, None, &self.device)?;
                let scores = match mask {
                    // The batched mask is already causal *and* pad-aware, so it replaces
                    // the plain causal one rather than stacking with it.
                    Some(m) => scores.broadcast_add(m)?,
                    None => scores.broadcast_add(&causal)?,
                };
                let probs = candle_nn::ops::softmax_last_dim(&scores)?;
                let out = probs.matmul(&v_rep.contiguous()?)?;
                out.transpose(1, 2)?
                    .reshape((b, t, k::N_HEADS * k::HEAD_DIM))?
                    .contiguous()?
            };

            h = (h + layer.wo.forward(&attn)?)?;

            let normed = rms_norm(&h, &layer.ffn_norm, k::NORM_EPS)?;
            let gate = candle_nn::ops::silu(&layer.gate.forward(&normed)?)?;
            let up = layer.up.forward(&normed)?;
            h = (h + layer.down.forward(&(gate * up)?)?)?;
        }
        let last = h.narrow(1, t - 1, 1)?.reshape((b, k::DIM))?;
        rms_norm(&last, &self.norm, k::NORM_EPS)
    }

    /// Generate speech tokens for a prompt, stopping on a control id.
    ///
    /// `text_len` is the *target* text's token count (not including the voice's
    /// transcript), which is what sets the reference's length bounds.
    pub fn generate(
        &self,
        text: &[u32],
        prompt_speech: &[u32],
        text_len: usize,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let mut out =
            self.generate_batch(&[(text.to_vec(), prompt_speech.to_vec(), text_len)], rng)?;
        Ok(out.pop().expect("one prompt in, one sequence out"))
    }

    /// Decode several segments at once.
    ///
    /// The saving is not in the arithmetic — it is that a decode step's cost is dominated
    /// by reading 24 layers of weights and by the host round-trip for sampling, and both
    /// are paid once for the whole batch instead of once per segment. Seven segments that
    /// took 1317 sequential steps take about as many steps as the longest one alone.
    ///
    /// Lanes stop at different times. A finished lane keeps being fed a filler token so
    /// the batch stays rectangular, and when a *contiguous tail* of lanes has finished the
    /// batch is narrowed to drop them — a prefix narrow shares storage, so shedding a tail
    /// is free where shedding an interior lane would mean rebuilding every cache.
    pub fn generate_batch(
        &self,
        prompts: &[(Vec<u32>, Vec<u32>, usize)],
        rng: &mut Rng,
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_batch_with(prompts, rng, false)
    }

    /// As [`Llm::generate_batch`], but `greedy` replaces the reference's repetition-aware
    /// sampler with an argmax.
    ///
    /// This exists to make batching testable. Sampling draws from a shared RNG, so a
    /// batched run and a sequential one consume the stream in different orders and produce
    /// different (equally valid) audio — which means sampled output cannot distinguish
    /// "batching changed the draw" from "batching changed the model". Greedy is
    /// deterministic, so `--greedy --set llm_batch=1` against `--greedy` compares the two
    /// paths directly.
    pub fn generate_batch_with(
        &self,
        prompts: &[(Vec<u32>, Vec<u32>, usize)],
        rng: &mut Rng,
        greedy: bool,
    ) -> Result<Vec<Vec<u32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let b = prompts.len();
        // Longest-first, because only a contiguous *tail* of finished lanes can be shed —
        // a prefix narrow shares the caches' storage where an interior gap would mean
        // rebuilding them. Unsorted, a long segment in the middle pins the whole batch:
        // measured 2072 lane-steps for 1317 real tokens, so 36% of the work was idle
        // lanes. Sorted, the tail sheds monotonically and the total is the sequential
        // count. The key is the length *bound* rather than the true length, which is not
        // known yet, but longer text does generate longer speech.
        let mut order: Vec<usize> = (0..b).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(prompts[i].2));
        let built: Vec<Tensor> = order
            .iter()
            .map(|&i| self.build_prompt(&prompts[i].0, &prompts[i].1))
            .collect::<Result<_>>()?;
        let mut state = self.prefill_batch(&built)?;

        let mut out: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        let mut samplers: Vec<Sampler> = (0..b).map(|_| Sampler::new(k::VOCAB)).collect();
        let min_len: Vec<usize> = order
            .iter()
            .map(|&i| prompts[i].2 * k::MIN_TOKEN_RATIO)
            .collect();
        let max_len: Vec<usize> = order
            .iter()
            .map(|&i| (prompts[i].2 * k::MAX_TOKEN_RATIO).min(self.capacity - state.width - 1))
            .collect();
        let steps = max_len.iter().copied().max().unwrap_or(0);
        // How many lanes are still being computed; always a prefix of the batch.
        let mut live = b;
        let mut taken = 0usize;

        for i in 0..steps {
            let logits = self.logits(&state.hidden)?;
            let flat = logits.flatten_all()?.to_vec1::<f32>()?;
            let mut feed = Vec::with_capacity(live);
            for lane in 0..live {
                if done[lane] {
                    // Inert filler: this lane's cache is no longer read from.
                    feed.push(0);
                    continue;
                }
                let mut scores = flat[lane * k::VOCAB..(lane + 1) * k::VOCAB].to_vec();
                if i < min_len[lane] {
                    // `sampling_ids` masks index `speech_token_size`, which is `sos`, not
                    // `eos`. Reproduced as written: masking the id the reference masks.
                    scores[k::SPEECH_TOKENS] = f32::NEG_INFINITY;
                }
                let token = if greedy {
                    scores
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .expect("non-empty scores")
                } else {
                    ras_sampling(&mut samplers[lane], &mut scores, &out[lane], rng)
                };
                if token as usize >= k::SPEECH_TOKENS {
                    done[lane] = true;
                    feed.push(0);
                } else {
                    out[lane].push(token);
                    feed.push(token);
                    // The sequential loop runs exactly `max_len` iterations, so the token
                    // drawn on the last one is kept; only the step after it is skipped.
                    if i + 1 >= max_len[lane] {
                        done[lane] = true;
                    }
                }
            }
            // Shed a finished tail. Anything short of the whole tail is left alone.
            while live > 0 && done[live - 1] {
                live -= 1;
            }
            if live == 0 {
                break;
            }
            if live < feed.len() {
                feed.truncate(live);
                state = state.narrow_to(live)?;
            }
            state = self.step_batch(state, &feed)?;
            taken = i + 1;
        }
        if std::env::var("COSY_LLM_TRACE").is_ok() {
            let lens: Vec<usize> = out.iter().map(|o| o.len()).collect();
            eprintln!(
                "llm batch: {taken} steps, lanes {lens:?}, sequential would be {}",
                lens.iter().sum::<usize>()
            );
        }
        // Back to the caller's order.
        let mut restored: Vec<Vec<u32>> = vec![Vec::new(); b];
        for (slot, &orig) in order.iter().enumerate() {
            restored[orig] = std::mem::take(&mut out[slot]);
        }
        Ok(restored)
    }
}

impl State {
    /// Keep only the first `live` lanes. A prefix narrow shares the caches' storage, so
    /// this costs a view rather than a copy of ~100 MB.
    fn narrow_to(mut self, live: usize) -> Result<Self> {
        if live == self.batch() {
            return Ok(self);
        }
        for c in self.caches.iter_mut() {
            c.k = c.k.narrow(0, 0, live)?;
            c.v = c.v.narrow(0, 0, live)?;
        }
        self.lanes.truncate(live);
        self.hidden = self.hidden.narrow(0, 0, live)?;
        Ok(self)
    }
}

/// Expand `[1, n_kv, t, d]` to `[1, n_heads, t, d]` by repeating each KV head.
fn repeat_kv(x: &Tensor) -> Result<Tensor> {
    let (b, n_kv, t, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, n_kv, k::GQA, t, d))?
        .reshape((b, n_kv * k::GQA, t, d))?)
}
