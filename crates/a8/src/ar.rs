//! The DualAR model: prompt grid -> codec codes.
//!
//! A slow AR of 24 layers emits one semantic token per audio frame; a fast AR of 4
//! layers then emits the 9 residual codebooks for that frame, one position at a time.
//! Note the arithmetic that surprised this project: the fast AR runs 4 layers x 10
//! positions = **40 layer-passes per frame against the slow AR's 24**, so it is the
//! larger half of the cost. See `docs/performance/ar-loop.md`.
//!
//! The three measured levers are structural here, not options:
//!
//! - **Narrow KV** (5.10x). Attention runs over `0..=pos`, not over a
//!   `max_seq_len`-wide buffer with a mask. The reference allocates 2048 wide and
//!   masks, doing 16x the necessary work by frame 64.
//! - **GQA by query reshape** (1.18x). The 7 query heads sharing a KV head are folded
//!   into the matmul's row dimension instead of materialising a 7x copy of K and V
//!   every layer every token.
//! - **Sliced logit head** (29x on that op). Only 4097 of 155776 rows are reachable
//!   after the semantic mask, so the projection is against a 4097-row slice. This is
//!   exact *because* the semantic mask runs first — and it is why `embeddings` can stay
//!   in full precision while everything else is quantized.
//!
//! Weights: the 28 layers' projections are q8_0 (3.35x, measured to cost nothing
//! audible — see `docs/performance/quantization-quality.md`); embeddings, heads and norms stay f32.

use crate::cfg;
use crate::nn::{rms_norm, rope_table, Proj, Weights};
use crate::sample::{argmax, gumbel_argmax_with, processed_scores_with, Rng, Scratch};
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor, D};

struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    wqkv: Proj,
    bqkv: Option<Tensor>,
    wo: Proj,
    w1: Proj,
    w2: Proj,
    w3: Proj,
}

impl Layer {
    fn load(w: &Weights, prefix: &str, quant: Option<GgmlDType>, device: &Device) -> Result<Self> {
        let bias_key = format!("{prefix}.attention.wqkv.bias");
        Ok(Self {
            attn_norm: w.get(&format!("{prefix}.attention_norm.weight"))?,
            ffn_norm: w.get(&format!("{prefix}.ffn_norm.weight"))?,
            wqkv: Proj::load(w, &format!("{prefix}.attention.wqkv.weight"), quant, device)?,
            // The slow layers carry a qkv bias; the fast layers do not
            // (`fast_attention_qkv_bias` is false).
            bqkv: if w.has(&bias_key) {
                Some(w.get(&bias_key)?)
            } else {
                None
            },
            wo: Proj::load(w, &format!("{prefix}.attention.wo.weight"), quant, device)?,
            w1: Proj::load(
                w,
                &format!("{prefix}.feed_forward.w1.weight"),
                quant,
                device,
            )?,
            w2: Proj::load(
                w,
                &format!("{prefix}.feed_forward.w2.weight"),
                quant,
                device,
            )?,
            w3: Proj::load(
                w,
                &format!("{prefix}.feed_forward.w3.weight"),
                quant,
                device,
            )?,
        })
    }
}

/// Preallocated K/V for one layer, `[batch, n_kv, capacity, head_dim]`.
struct Cache {
    k: Tensor,
    v: Tensor,
}

impl Cache {
    fn new(capacity: usize, batch: usize, device: &Device) -> Result<Self> {
        let shape = (batch, cfg::N_KV, capacity, cfg::HEAD_DIM);
        Ok(Self {
            k: Tensor::zeros(shape, DType::F32, device)?,
            v: Tensor::zeros(shape, DType::F32, device)?,
        })
    }
}

/// Everything a decode step needs that is not a weight.
struct Caches {
    slow: Vec<Cache>,
    fast: Vec<Cache>,
}

impl Caches {
    fn new(batch: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            slow: (0..cfg::N_LAYER)
                .map(|_| Cache::new(cfg::MAX_SEQ_LEN, batch, device))
                .collect::<Result<_>>()?,
            fast: (0..cfg::N_FAST_LAYER)
                .map(|_| Cache::new(cfg::NUM_CODEBOOKS, batch, device))
                .collect::<Result<_>>()?,
        })
    }
}

/// One sequence inside a batch: its right-aligned rows and its decode state.
///
/// `pad` is what makes a batch possible at all. Segments have different prompt widths, so
/// the prompts are **right-aligned** — padded on the left until every sequence's last
/// prompt token sits at index `width - 1` — and the leading `pad` columns are masked out.
///
/// That is exact, and the argument is worth stating because it is the whole reason batching
/// works here: RoPE rotates `q` and `k` by absolute position, but an attention *score*
/// `q_p . k_j` depends only on `p - j`. Right-alignment shifts every position within a
/// sequence by the same constant, so every difference — and therefore every score — is
/// unchanged. `v` is never rotated. Left-padding instead would shift each sequence by a
/// *different* amount relative to its own content, which is not the same model.
struct Seq {
    /// 11 rows: the semantic row then ten codebooks, `pad` entries of filler in front.
    rows: Vec<Vec<u32>>,
    /// Leading masked-out positions.
    pad: usize,
    /// Emitted frames, `[frame][10]`.
    frames: Vec<Vec<u32>>,
    /// The RAS window, or `None` before the first emission.
    previous: Option<Vec<u32>>,
    /// Set once this sequence has emitted EOS. A finished lane keeps being computed to
    /// keep the batch rectangular, and its output is discarded.
    done: bool,
}

/// A non-semantic filler id for padded positions.
///
/// It has to be non-semantic so `embed`'s `keep` mask zeroes the codebook contribution
/// there, exactly as it does for the prompt's text tokens. The value is otherwise
/// irrelevant: every padded position is masked out of every real position's attention.
const PAD_ID: u32 = 0;

pub struct Model {
    /// `[vocab, dim]` — needed whole for prompt token lookups. The logit *projection*
    /// uses `head` instead, which is what makes keeping this in f32 affordable.
    embeddings: Tensor,
    /// `[dim, 4097]`: the 4096 semantic rows then the eos row, already transposed.
    /// Transposing per step re-materialised 14.7 MB each time — and the fast head, 10
    /// times a frame, another 147 MB of pure copying per frame.
    head_t: Tensor,
    codebook_embeddings: Tensor,
    fast_embeddings: Tensor,
    fast_output_t: Tensor,
    norm: Tensor,
    fast_norm: Tensor,
    slow: Vec<Layer>,
    fast: Vec<Layer>,
    cos: Tensor,
    sin: Tensor,
    /// The same tables pre-scaled by `1/sqrt(head_dim)`. RoPE is linear in `q`, so
    /// scaling the table scales the rotated query — which folds the attention scale
    /// into a lookup and removes one elementwise pass per layer per token. With ~1660
    /// dispatches per frame and ~9.4 us of issue cost each, ops-per-layer is the
    /// currency that matters here, not FLOPs.
    cos_q: Tensor,
    sin_q: Tensor,
    fast_cos: Tensor,
    fast_sin: Tensor,
    fast_cos_q: Tensor,
    fast_sin_q: Tensor,
    device: Device,
}

/// Generation knobs, defaults from `generation_config.json`.
pub struct GenConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub do_sample: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            do_sample: true,
        }
    }
}

impl Model {
    pub fn load(path: &str, device: &Device, quant: Option<GgmlDType>) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let mut slow = Vec::new();
        for i in 0..cfg::N_LAYER {
            slow.push(Layer::load(&w, &format!("layers.{i}"), quant, device)?);
        }
        let mut fast = Vec::new();
        for i in 0..cfg::N_FAST_LAYER {
            fast.push(Layer::load(&w, &format!("fast_layers.{i}"), quant, device)?);
        }
        let embeddings = w.get("embeddings.weight")?;
        // The reachable rows: [semantic_begin, semantic_end] then eos. eos sits *below*
        // semantic_begin in id space, so the set is not contiguous and index 4096 of the
        // slice maps back to eos rather than to semantic_begin + 4096.
        let semantic = embeddings.narrow(0, cfg::SEMANTIC_BEGIN_ID as usize, cfg::CODEBOOK_SIZE)?;
        let eos = embeddings.narrow(0, cfg::EOS_TOKEN_ID as usize, 1)?;
        let head = Tensor::cat(&[&semantic, &eos], 0)?.contiguous()?;
        debug_assert_eq!(head.dim(0)?, cfg::REACHABLE);
        let head_t = head.t()?.contiguous()?;

        let (cos, sin) = rope_table(cfg::MAX_SEQ_LEN, cfg::HEAD_DIM, cfg::ROPE_BASE, device)?;
        let (fast_cos, fast_sin) =
            rope_table(cfg::NUM_CODEBOOKS, cfg::HEAD_DIM, cfg::ROPE_BASE, device)?;
        let scale = 1.0 / (cfg::HEAD_DIM as f64).sqrt();
        let (cos_q, sin_q) = ((&cos * scale)?, (&sin * scale)?);
        let (fast_cos_q, fast_sin_q) = ((&fast_cos * scale)?, (&fast_sin * scale)?);
        Ok(Self {
            embeddings,
            head_t,
            codebook_embeddings: w.get("codebook_embeddings.weight")?,
            fast_embeddings: w.get("fast_embeddings.weight")?,
            fast_output_t: w.get("fast_output.weight")?.t()?.contiguous()?,
            norm: w.get("norm.weight")?,
            fast_norm: w.get("fast_norm.weight")?,
            slow,
            fast,
            cos,
            sin,
            cos_q,
            sin_q,
            fast_cos,
            fast_sin,
            fast_cos_q,
            fast_sin_q,
            device: device.clone(),
        })
    }

    /// `_embed`: row 0 through the token embedding, plus the summed codebook embeddings
    /// wherever row 0 holds a semantic id.
    ///
    /// The reference gathers each codebook separately; the ten tables are slices of one
    /// `[10 * 4096, dim]` buffer at offset `i * codebook_size`, so adding the offsets to
    /// the indices turns ten gathers into one. Identical result, one dispatch.
    /// Rebuild the RoPE tables in plain f32, without the bf16 round-trip.
    ///
    /// **A validation aid, not a production option** — the bf16 rounding is part of the
    /// model's arithmetic (the reference builds the table with
    /// `torch.polar(...).to(bfloat16)`), so using this for synthesis puts the port off its
    /// fixtures.
    ///
    /// It exists to make one specific claim testable. Right-alignment in
    /// [`Model::generate_batch`] is exact *in real arithmetic*, because an attention score
    /// `q_p . k_j` depends only on `p - j`. Under a bf16-rounded table that stops being quite
    /// true: `R(p)` and `R(j)` are rounded independently, so `R(p)^T R(j)` equals `R(p - j)`
    /// only to about 4e-3 — an 8-bit mantissa. With f32 tables the identity is restored to
    /// f32 precision, so batched and unbatched greedy decoding must then agree exactly, which
    /// is what `a8-validate` asserts. Any difference that survives under bf16 is thereby
    /// attributable to the table rounding rather than to the alignment logic.
    pub fn with_f32_rope(mut self) -> Result<Self> {
        let (cos, sin) = crate::nn::rope_table_f32(
            cfg::MAX_SEQ_LEN,
            cfg::HEAD_DIM,
            cfg::ROPE_BASE,
            &self.device,
        )?;
        let scale = 1.0 / (cfg::HEAD_DIM as f64).sqrt();
        self.cos_q = (&cos * scale)?;
        self.sin_q = (&sin * scale)?;
        self.cos = cos;
        self.sin = sin;
        Ok(self)
    }

    /// One sequence's rows; `[1, len, dim]`.
    fn embed(&self, rows: &[Vec<u32>], start: usize, len: usize) -> Result<Tensor> {
        self.embed_batch(std::slice::from_ref(&rows), start, len)
    }

    /// `b` sequences' rows; `[b, len, dim]`.
    ///
    /// Still one gather for the text row and one for all ten codebooks across the whole
    /// batch — the batch axis simply extends the index vectors, so a batch of 8 costs the
    /// same number of dispatches as a batch of 1.
    fn embed_batch<R: AsRef<[Vec<u32>]>>(
        &self,
        seqs: &[R],
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        let b = seqs.len();
        let n = b * len;

        let mut ids0 = Vec::with_capacity(n);
        for s in seqs {
            ids0.extend_from_slice(&s.as_ref()[0][start..start + len]);
        }
        let ids = Tensor::from_slice(&ids0, n, &self.device)?;
        let text = self.embeddings.index_select(&ids, 0)?;

        let mut offset_idx = Vec::with_capacity(cfg::NUM_CODEBOOKS * n);
        for i in 0..cfg::NUM_CODEBOOKS {
            let base = (i * cfg::CODEBOOK_SIZE) as u32;
            for s in seqs {
                offset_idx.extend(
                    s.as_ref()[i + 1][start..start + len]
                        .iter()
                        .map(|&c| c + base),
                );
            }
        }
        let idx = Tensor::from_vec(offset_idx, cfg::NUM_CODEBOOKS * n, &self.device)?;
        let gathered = self.codebook_embeddings.index_select(&idx, 0)?.reshape((
            cfg::NUM_CODEBOOKS,
            n,
            cfg::DIM,
        ))?;
        let codebook_sum = gathered.sum(0)?;

        // Zero the codebook contribution wherever row 0 is not a semantic id. This is also
        // what makes `PAD_ID` inert.
        let keep: Vec<f32> = ids0
            .iter()
            .map(|&t| {
                if (cfg::SEMANTIC_BEGIN_ID..=cfg::SEMANTIC_END_ID).contains(&t) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let keep = Tensor::from_vec(keep, (n, 1), &self.device)?;
        let summed = (text + codebook_sum.broadcast_mul(&keep)?)?;
        Ok(summed.reshape((b, len, cfg::DIM))?)
    }

    /// One transformer layer over `q_len` positions starting at `pos`.
    ///
    /// `mask` is `Some` only for prefill: a single decode step attends to everything in
    /// the cache, so there is nothing to mask.
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &Layer,
        x: &Tensor,
        cache_k: &Tensor,
        cache_v: &Tensor,
        pos: usize,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        cos_q: &Tensor,
        sin_q: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qs = cfg::N_HEAD * cfg::HEAD_DIM;
        let kvs = cfg::N_KV * cfg::HEAD_DIM;
        let rep = cfg::N_HEAD / cfg::N_KV;

        let h = rms_norm(x, &layer.attn_norm, cfg::NORM_EPS)?;
        let mut qkv = layer.wqkv.forward(&h)?;
        if let Some(bias) = &layer.bqkv {
            qkv = qkv.broadcast_add(bias)?;
        }

        // Splitting q/k/v by narrowing the last dimension yields non-contiguous views,
        // so each one costs a copy. At t == 1 the whole thing is one contiguous run, and
        // narrowing dimension 0 of a flattened view stays contiguous — same three
        // tensors, three fewer copies per layer per token.
        //
        // Only valid at `b == 1`: with a batch the three blocks interleave per sequence, so
        // absolute offsets into the flattened buffer no longer name q, k and v.
        let (q, k, v) = if t == 1 && b == 1 {
            let flat = qkv.reshape(((),))?;
            (
                flat.narrow(0, 0, qs)?
                    .reshape((b, cfg::N_HEAD, 1, cfg::HEAD_DIM))?,
                flat.narrow(0, qs, kvs)?
                    .reshape((b, cfg::N_KV, 1, cfg::HEAD_DIM))?,
                flat.narrow(0, qs + kvs, kvs)?
                    .reshape((b, cfg::N_KV, 1, cfg::HEAD_DIM))?,
            )
        } else {
            (
                qkv.narrow(2, 0, qs)?
                    .reshape((b, t, cfg::N_HEAD, cfg::HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
                qkv.narrow(2, qs, kvs)?
                    .reshape((b, t, cfg::N_KV, cfg::HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
                qkv.narrow(2, qs + kvs, kvs)?
                    .reshape((b, t, cfg::N_KV, cfg::HEAD_DIM))?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        };
        // q takes the pre-scaled table, which is where the attention scale now lives.
        let q = candle_nn::rotary_emb::rope_i(
            &q,
            &cos_q.narrow(0, pos, t)?,
            &sin_q.narrow(0, pos, t)?,
        )?;
        let k =
            candle_nn::rotary_emb::rope_i(&k, &cos.narrow(0, pos, t)?, &sin.narrow(0, pos, t)?)?;

        // `cache_k`/`cache_v` may be a `narrow(0, 0, active)` view of the full cache. Dim-0
        // narrowing from offset 0 keeps the layout a prefix of the parent's, so `slice_set`
        // writes through to the real storage — which is what makes shrinking the batch free.
        cache_k.slice_set(&k, 2, pos)?;
        cache_v.slice_set(&v, 2, pos)?;
        let attend = pos + t;
        let kk = cache_k.narrow(2, 0, attend)?;
        let vv = cache_v.narrow(2, 0, attend)?;

        // GQA without copying K/V: fold the query heads sharing a KV head into rows.
        let qg = q.reshape((b, cfg::N_KV, rep * t, cfg::HEAD_DIM))?;
        let scores = qg.matmul(&kk.transpose(2, 3)?.contiguous()?)?;
        let scores = match mask {
            None => scores,
            Some(m) => scores
                .reshape((b, cfg::N_HEAD, t, attend))?
                .broadcast_add(m)?
                .reshape((b, cfg::N_KV, rep * t, attend))?,
        };
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&vv.contiguous()?)?;
        // At t == 1 the [b, n_kv, rep, head_dim] result is already in head-major order,
        // so the transpose the general case needs is a no-op on the layout.
        let merged = if t == 1 {
            ctx.reshape((b, 1, qs))?
        } else {
            ctx.reshape((b, cfg::N_HEAD, t, cfg::HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, qs))?
        };
        let attn = layer.wo.forward(&merged)?;
        let x = (x + attn)?;

        let h = rms_norm(&x, &layer.ffn_norm, cfg::NORM_EPS)?;
        let gate = candle_nn::ops::silu(&layer.w1.forward(&h)?)?;
        let up = layer.w3.forward(&h)?;
        let ffn = layer.w2.forward(&(gate * up)?)?;
        Ok((&x + ffn)?)
    }

    /// The slow AR over `b` sequences. Returns `b * REACHABLE` logits, row-major by
    /// sequence, and the normed hidden states `[b, 1, dim]`.
    fn slow_step_batch<R: AsRef<[Vec<u32>]>>(
        &self,
        seqs: &[R],
        start: usize,
        len: usize,
        caches: &mut Caches,
        mask: Option<&Tensor>,
    ) -> Result<(Vec<f32>, Tensor)> {
        // `seqs` is the *live* prefix of the batch, so its length is how wide the caches
        // should be read and written this step.
        let b = seqs.len();
        let active = b;
        let mut h = self.embed_batch(seqs, start, len)?;
        for (layer, cache) in self.slow.iter().zip(caches.slow.iter()) {
            let (ck, cv) = if active == cache.k.dim(0)? {
                (cache.k.clone(), cache.v.clone())
            } else {
                (cache.k.narrow(0, 0, active)?, cache.v.narrow(0, 0, active)?)
            };
            h = self.layer_forward(
                layer,
                &h,
                &ck,
                &cv,
                start,
                mask,
                &self.cos,
                &self.sin,
                &self.cos_q,
                &self.sin_q,
            )?;
        }
        // `hidden = hidden[:, -1:]` then norm, so only the last position matters.
        let last = h.narrow(1, len - 1, 1)?;
        let normed = rms_norm(&last, &self.norm, cfg::NORM_EPS)?;
        let logits = normed.reshape((b, cfg::DIM))?.matmul(&self.head_t)?;
        Ok((logits.flatten_all()?.to_vec1::<f32>()?, normed))
    }

    /// One fast-AR position. `hidden` is `[b, 1, dim]`; returns `[b, codebook_size]`.
    fn fast_step(&self, hidden: &Tensor, position: usize, caches: &mut Caches) -> Result<Tensor> {
        let b = hidden.dim(0)?;
        let mut h = hidden.clone();
        for (layer, cache) in self.fast.iter().zip(caches.fast.iter()) {
            let (ck, cv) = if b == cache.k.dim(0)? {
                (cache.k.clone(), cache.v.clone())
            } else {
                (cache.k.narrow(0, 0, b)?, cache.v.narrow(0, 0, b)?)
            };
            h = self.layer_forward(
                layer,
                &h,
                &ck,
                &cv,
                position,
                None,
                &self.fast_cos,
                &self.fast_sin,
                &self.fast_cos_q,
                &self.fast_sin_q,
            )?;
        }
        let normed = rms_norm(&h, &self.fast_norm, cfg::NORM_EPS)?;
        Ok(normed.reshape((b, cfg::DIM))?.matmul(&self.fast_output_t)?)
    }

    /// The 9 residual codebooks for one frame, for `b` sequences at once.
    ///
    /// `active[i]` false means the lane is finished: it is still computed to keep the batch
    /// rectangular, but nothing is sampled for it and it consumes no RNG.
    // Eight tensors and indices, all genuinely independent per call. Bundling them into
    // a struct would move the argument list rather than shorten it.
    #[allow(clippy::too_many_arguments)]
    fn generate_codebooks_batch(
        &self,
        slow_hidden: &Tensor,
        semantics: &[u32],
        active: &[bool],
        cfgen: &GenConfig,
        caches: &mut Caches,
        rng: &mut Rng,
        scratch: &mut Scratch,
    ) -> Result<Vec<Vec<u32>>> {
        let b = semantics.len();
        // Trap: this first call's *result is discarded*. It exists to prime the fast KV
        // cache at position 0, and dropping it silently shifts every position by one.
        let _ = self.fast_step(slow_hidden, 0, caches)?;

        let mut current: Vec<u32> = semantics
            .iter()
            .map(|&s| {
                s.saturating_sub(cfg::SEMANTIC_BEGIN_ID)
                    .min(cfg::CODEBOOK_SIZE as u32 - 1)
            })
            .collect();
        let mut codebooks: Vec<Vec<u32>> = current.iter().map(|&c| vec![c]).collect();

        for position in 1..cfg::NUM_CODEBOOKS {
            let idx = Tensor::from_slice(&current, b, &self.device)?;
            let hidden = self
                .fast_embeddings
                .index_select(&idx, 0)?
                .reshape((b, 1, cfg::DIM))?;
            let scores = self.fast_step(&hidden, position, caches)?;
            let host = scores.flatten_all()?.to_vec1::<f32>()?;
            for i in 0..b {
                if !active[i] {
                    codebooks[i].push(0);
                    continue;
                }
                let mut row = host[i * cfg::CODEBOOK_SIZE..(i + 1) * cfg::CODEBOOK_SIZE].to_vec();
                processed_scores_with(
                    scratch,
                    &mut row,
                    cfgen.top_k,
                    cfgen.top_p,
                    cfgen.temperature,
                );
                current[i] = if cfgen.do_sample {
                    gumbel_argmax_with(scratch, &row, rng) as u32
                } else {
                    argmax(&row) as u32
                };
                codebooks[i].push(current[i]);
            }
        }
        Ok(codebooks)
    }

    /// Map an index in the 4097-row reachable slice back to a vocabulary id.
    fn reachable_to_token(index: usize) -> u32 {
        if index < cfg::CODEBOOK_SIZE {
            cfg::SEMANTIC_BEGIN_ID + index as u32
        } else {
            cfg::EOS_TOKEN_ID
        }
    }

    /// Sample one semantic token, with RAS.
    ///
    /// Repetition-aware sampling draws **twice** per step — once at the caller's
    /// `top_p`/`temperature` and once at the config's `ras_top_p`/`ras_temperature` —
    /// and prefers the second draw when the first repeats something in the recent
    /// window. Both draws always happen, so both always consume RNG.
    fn sample_semantic(
        &self,
        logits: &[f32],
        cfgen: &GenConfig,
        previous: Option<&[u32]>,
        rng: &mut Rng,
        scratch: &mut Scratch,
    ) -> u32 {
        let mut regular = logits.to_vec();
        processed_scores_with(
            scratch,
            &mut regular,
            cfgen.top_k,
            cfgen.top_p,
            cfgen.temperature,
        );
        if !cfgen.do_sample {
            return Self::reachable_to_token(argmax(&regular));
        }
        let normal = Self::reachable_to_token(gumbel_argmax_with(scratch, &regular, rng));

        let mut high_scores = logits.to_vec();
        processed_scores_with(
            scratch,
            &mut high_scores,
            cfgen.top_k,
            cfg::RAS_TOP_P,
            cfg::RAS_TEMPERATURE,
        );
        let high = Self::reachable_to_token(gumbel_argmax_with(scratch, &high_scores, rng));

        match previous {
            None => normal,
            Some(window) => {
                let repeated = window.contains(&normal);
                let is_semantic = (cfg::SEMANTIC_BEGIN_ID..=cfg::SEMANTIC_END_ID).contains(&normal);
                if repeated && is_semantic {
                    high
                } else {
                    normal
                }
            }
        }
    }

    /// Generate codec codes for one prompt. Returns `[num_codebooks, frames]`.
    pub fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        cfgen: &GenConfig,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<u32>>> {
        let mut out = self.generate_batch(std::slice::from_ref(&prompt), cfgen, rng)?;
        Ok(out.pop().expect("one prompt in, one out"))
    }

    /// Generate for several prompts at once. Returns one `[num_codebooks, frames]` per
    /// prompt, in the order given.
    ///
    /// # Why this is worth the complexity
    ///
    /// At `dim(-2) == 1` a decode step is a matrix-*vector* product: every weight element is
    /// read once and multiplied once, so the loop runs at bus speed and the arithmetic units
    /// idle. Measured, a batched layer costs about what a batch-2 layer costs all the way out
    /// to 32, so per-sequence cost falls almost linearly — up to **11.95x at batch 32**.
    ///
    /// # Why batch 2 and 3 are forbidden
    ///
    /// candle takes its dedicated matrix-vector kernel only when `dim(-2) == 1`. At batch 2
    /// it falls back to the general quantized matmul, which costs 2.5x more and then stays
    /// flat — so batch 2 and 3 are per-sequence *regressions*. Callers should use
    /// [`plan_batches`], which never emits a group of 2 or 3.
    ///
    /// # What differs from running the prompts one at a time
    ///
    /// Two things, both deliberate:
    ///
    /// - **Sampling order.** Per-sequence sampling inside a batched step consumes the RNG in
    ///   a different order than the sequential path, so batched and unbatched renders of the
    ///   same text with the same seed are different audio. Both are valid draws. Greedy
    ///   output is unaffected, which is what `a8-validate` checks: a batch of four distinct
    ///   prompts decoded greedily must reproduce all four unbatched results exactly.
    /// - **Finished lanes keep computing.** Sequences reach EOS at different steps; a
    ///   finished lane is carried to keep the batch rectangular and its output discarded.
    ///   Bucketing by prompt width (see [`plan_batches`]) is what keeps that waste small.
    pub fn generate_batch(
        &self,
        prompts: &[&crate::prompt::Prompt],
        cfgen: &GenConfig,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<Vec<u32>>>> {
        let b = prompts.len();
        anyhow::ensure!(b > 0, "no prompts to generate for");
        let width = prompts.iter().map(|p| p.len).max().expect("non-empty");

        // Longest prompt first. Two reasons, both about the tail: a longer prompt usually
        // means a longer segment and so more generated frames, and only a *contiguous tail*
        // of finished lanes can be dropped for free (a prefix narrow shares storage; an
        // interior gap would need a copy). Ordering this way makes the lanes that finish
        // early the ones that can actually be shed. `order` maps back on the way out.
        let mut order: Vec<usize> = (0..b).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(prompts[i].len));
        let prompts: Vec<&crate::prompt::Prompt> = order.iter().map(|&i| prompts[i]).collect();
        let prompts = prompts.as_slice();
        let budget = cfgen
            .max_new_tokens
            .min(cfg::MAX_SEQ_LEN.saturating_sub(width));
        anyhow::ensure!(budget > 0, "prompt of {width} leaves no room to generate");

        // Right-align every prompt to `width`; see `Seq` for why that is exact.
        let mut seqs: Vec<Seq> = prompts
            .iter()
            .map(|p| {
                let pad = width - p.len;
                let rows: Vec<Vec<u32>> = p
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(r, row)| {
                        let filler = if r == 0 { PAD_ID } else { 0 };
                        let mut v = vec![filler; pad];
                        v.extend_from_slice(&row[..p.len]);
                        v
                    })
                    .collect();
                Seq {
                    rows,
                    pad,
                    frames: Vec::new(),
                    previous: None,
                    done: false,
                }
            })
            .collect();

        let mut caches = Caches::new(b, &self.device)?;
        let prefill_mask = self.prefill_mask(&seqs, width)?;
        // Padded columns stay masked for the whole decode, so the mask is built once at
        // full capacity and narrowed each step rather than rebuilt.
        let decode_mask = self.decode_mask(&seqs)?;

        // Two scratches because the two heads differ in width: the sliced semantic head is
        // REACHABLE rows, the fast head CODEBOOK_SIZE. Sharing one would resize on every
        // alternation, which is the allocation this is here to avoid.
        let mut slow_scratch = Scratch::new(cfg::REACHABLE);
        let mut fast_scratch = Scratch::new(cfg::CODEBOOK_SIZE);

        let seq_rows: Vec<&Vec<Vec<u32>>> = seqs.iter().map(|s| &s.rows).collect();
        let (mut logits, mut hidden) =
            self.slow_step_batch(&seq_rows, 0, width, &mut caches, prefill_mask.as_ref())?;

        // Lanes are ordered longest-prompt-first (see the sort in `generate_batch`'s caller),
        // so the sequences likely to finish first sit at the tail and `live` can shrink by a
        // contiguous prefix narrow. Without this the group runs `max(frames)` steps at full
        // width, which measured as a wash on ragged text: the longest segment sets the step
        // count for everyone. Shrinking makes the cost track the *active* lane count instead.
        let mut live = b;

        for step in 0..budget {
            let mut semantics = vec![0u32; live];
            let mut active = vec![false; live];
            for i in 0..live {
                if seqs[i].done {
                    continue;
                }
                let row = &logits[i * cfg::REACHABLE..(i + 1) * cfg::REACHABLE];
                let s = self.sample_semantic(
                    row,
                    cfgen,
                    seqs[i].previous.as_deref(),
                    rng,
                    &mut slow_scratch,
                );
                if s == cfg::EOS_TOKEN_ID {
                    seqs[i].done = true;
                    continue;
                }
                semantics[i] = s;
                active[i] = true;
            }
            if active.iter().all(|a| !a) {
                break;
            }

            let codebooks = self.generate_codebooks_batch(
                &hidden,
                &semantics,
                &active,
                cfgen,
                &mut caches,
                rng,
                &mut fast_scratch,
            )?;

            // The prompt occupies positions 0..width-1 and its last position produced the
            // logits just sampled from, so the emitted column is fed at `width + step` —
            // not one past it. Getting this wrong shifts every position by one and RoPE
            // quietly desynchronises from the cache.
            let pos = width + step;
            if pos >= cfg::MAX_SEQ_LEN {
                break;
            }
            for i in 0..live {
                if active[i] {
                    seqs[i].frames.push(codebooks[i].clone());
                    // The RAS window starts as zeros and — faithfully to the reference —
                    // never records the *first* emitted token.
                    match seqs[i].previous.as_mut() {
                        None => seqs[i].previous = Some(vec![0u32; cfg::RAS_WINDOW_SIZE]),
                        Some(window) => {
                            window.remove(0);
                            window.push(semantics[i]);
                        }
                    }
                }
                // Finished lanes are fed inert filler: `PAD_ID` is non-semantic, so
                // `embed`'s keep mask zeroes its codebook contribution just as it does for
                // the prompt's text tokens.
                let (tok, cb) = if active[i] {
                    (semantics[i], codebooks[i].as_slice())
                } else {
                    (PAD_ID, [0u32; cfg::NUM_CODEBOOKS].as_slice())
                };
                seqs[i].rows[0].push(tok);
                for (j, &c) in cb.iter().enumerate() {
                    seqs[i].rows[j + 1].push(c);
                }
                debug_assert_eq!(seqs[i].rows[0].len(), pos + 1);
            }

            // Drop finished lanes off the tail. Only a contiguous tail can go, because the
            // caches are indexed by lane and a prefix narrow is the only free one.
            while live > 1 && seqs[live - 1].done {
                live -= 1;
            }

            let step_mask = match &decode_mask {
                None => None,
                Some(m) => Some(m.narrow(0, 0, live)?.narrow(3, 0, pos + 1)?),
            };
            let seq_rows: Vec<&Vec<Vec<u32>>> = seqs[..live].iter().map(|s| &s.rows).collect();
            let (l, h) =
                self.slow_step_batch(&seq_rows, pos, 1, &mut caches, step_mask.as_ref())?;
            logits = l;
            hidden = h;
        }

        // [frames][10] -> [10][frames], per sequence, then undo the longest-first sort.
        let mut out: Vec<Option<Vec<Vec<u32>>>> = vec![None; b];
        for (lane, s) in seqs.iter().enumerate() {
            // `vec![Vec::with_capacity(n); k]` clones an empty Vec, which does not carry
            // the capacity with it; build each row so the reserve actually happens.
            let mut rows: Vec<Vec<u32>> = (0..cfg::NUM_CODEBOOKS)
                .map(|_| Vec::with_capacity(s.frames.len()))
                .collect();
            for frame in &s.frames {
                for (i, &c) in frame.iter().enumerate() {
                    rows[i].push(c);
                }
            }
            out[order[lane]] = Some(rows);
        }
        Ok(out
            .into_iter()
            .map(|o| o.expect("every lane assigned"))
            .collect())
    }

    /// The prefill mask, `[b, 1, width, width]`, or `None` when there is nothing to mask.
    ///
    /// Real row `r` of sequence `i` may attend to `pad_i <= c <= r`. **Padded rows are given
    /// ordinary causal visibility instead of being fully masked**, which matters more than it
    /// looks: a row with every column at `-inf` softmaxes to NaN, and that NaN would land in
    /// the padded position's hidden state, become a padded K/V entry in the next layer, and
    /// then poison every real row through `NaN + -inf = NaN`. Masking a position out of
    /// attention does not stop it being *computed*; it only stops it being read.
    fn prefill_mask(&self, seqs: &[Seq], width: usize) -> Result<Option<Tensor>> {
        let b = seqs.len();
        if width <= 1 {
            return Ok(None);
        }
        let mut v = vec![0f32; b * width * width];
        for (i, s) in seqs.iter().enumerate() {
            for r in 0..width {
                let lo = if r < s.pad { 0 } else { s.pad };
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

    /// The decode mask, `[b, 1, 1, MAX_SEQ_LEN]`, or `None` when no sequence is padded.
    ///
    /// `None` for a single prompt, which keeps the batch-1 path exactly as it was — no mask
    /// tensor, no broadcast add, 24 layers a step.
    fn decode_mask(&self, seqs: &[Seq]) -> Result<Option<Tensor>> {
        if seqs.iter().all(|s| s.pad == 0) {
            return Ok(None);
        }
        let b = seqs.len();
        let mut v = vec![0f32; b * cfg::MAX_SEQ_LEN];
        for (i, s) in seqs.iter().enumerate() {
            for c in 0..s.pad {
                v[i * cfg::MAX_SEQ_LEN + c] = f32::NEG_INFINITY;
            }
        }
        Ok(Some(Tensor::from_vec(
            v,
            (b, 1, 1, cfg::MAX_SEQ_LEN),
            &self.device,
        )?))
    }

    /// One frame's worth of GPU work with **no host readback**, for benchmarking only.
    ///
    /// A real frame is one slow step plus ten fast steps, and each of those eleven ends in a
    /// `to_vec1` so the sampler can run on the host — eleven pipeline flushes per frame. This
    /// runs the identical kernels and never reads back, so the difference against
    /// `generate_batch` is the cost of synchronising plus sampling. Emits nothing usable;
    /// the tokens it feeds the fast AR are fixed rather than sampled.
    #[doc(hidden)]
    pub fn bench_frame_gpu_only(
        &self,
        prompt: &crate::prompt::Prompt,
        frames: usize,
    ) -> Result<()> {
        let width = prompt.len;
        let mut caches = Caches::new(1, &self.device)?;
        let rows = vec![prompt.rows.clone()];
        let mask = crate::nn::causal_window_mask(width, None, &self.device)?
            .reshape((1, 1, width, width))?;
        let (_, mut hidden) = self.slow_step_batch(&rows, 0, width, &mut caches, Some(&mask))?;

        let mut seq = prompt.rows.clone();
        for step in 0..frames {
            let _ = self.fast_step(&hidden, 0, &mut caches)?;
            let mut current = 0u32;
            for position in 1..cfg::NUM_CODEBOOKS {
                let idx = Tensor::from_slice(&[current], 1, &self.device)?;
                let h = self
                    .fast_embeddings
                    .index_select(&idx, 0)?
                    .reshape((1, 1, cfg::DIM))?;
                // The kernels a real step runs; the argmax that would follow is skipped.
                let _scores = self.fast_step(&h, position, &mut caches)?;
                current = (current + 1) % cfg::CODEBOOK_SIZE as u32;
            }
            let pos = width + step;
            if pos >= cfg::MAX_SEQ_LEN {
                break;
            }
            seq[0].push(cfg::SEMANTIC_BEGIN_ID);
            for row in seq.iter_mut().skip(1) {
                row.push(0);
            }
            let rows = vec![seq.clone()];
            let (_, h) = self.slow_step_batch(&rows, pos, 1, &mut caches, None)?;
            hidden = h;
        }
        Ok(())
    }

    /// Slow-AR logits over every prompt position, for fixture validation only.
    /// Returns `[T, reachable]` plus the normed hidden states `[1, T, dim]`.
    pub fn debug_prefill(&self, prompt: &crate::prompt::Prompt) -> Result<(Tensor, Tensor)> {
        let width = prompt.len;
        let caches = Caches {
            slow: (0..cfg::N_LAYER)
                .map(|_| Cache::new(cfg::MAX_SEQ_LEN, 1, &self.device))
                .collect::<Result<_>>()?,
            fast: Vec::new(),
        };
        let mut h = self.embed(&prompt.rows, 0, width)?;
        let mask = crate::nn::causal_window_mask(width, None, &self.device)?
            .reshape((1, 1, width, width))?;
        for (layer, cache) in self.slow.iter().zip(caches.slow.iter()) {
            h = self.layer_forward(
                layer,
                &h,
                &cache.k,
                &cache.v,
                0,
                Some(&mask),
                &self.cos,
                &self.sin,
                &self.cos_q,
                &self.sin_q,
            )?;
        }
        let normed = rms_norm(&h, &self.norm, cfg::NORM_EPS)?;
        let logits = normed.reshape((width, cfg::DIM))?.matmul(&self.head_t)?;
        Ok((logits, normed))
    }
}

/// Group `n` items into batches of at most `max_batch`, in order.
///
/// Plain chunking, and the reason it is plain is a correction worth recording. An earlier
/// version deliberately refused to emit a group of 2 or 3, because a *layer* benchmark had
/// found batch 2 to be a 0.80x per-sequence regression: candle takes its dedicated
/// matrix-vector kernel only at `dim(-2) == 1`, and at batch 2 it falls back to the general
/// quantized matmul.
///
/// Measured on the real loop (`a8-probe --bin arbatch`), that rule is wrong. Per-sequence
/// gain against batch 1:
///
/// | batch | 2 | 3 | 4 | 8 | 16 |
/// |---|---|---|---|---|---|
/// | gain | 1.36x | 1.52x | 1.65x | 1.87x | 2.00x |
///
/// Both measurements are true; they measure different things. The projections do get worse
/// per sequence at batch 2, but they are not the whole step — the embedding gather, the
/// per-step host synchronisations and the fast AR's ten positions all amortise across lanes,
/// and the net is a gain from batch 2 onward. Scheduling has to follow the loop-level number.
///
/// Note also what the same measurement says about the *ceiling*: per-sequence gain saturates
/// near **2x**, not the 11.95x the layer benchmark projected, because the sampler runs on the
/// host once per sequence per codebook and does not amortise at all. See
/// `docs/performance/ar-loop.md`.
///
/// Returns index groups into the original slice; the caller reassembles in order.
pub fn plan_batches(n: usize, max_batch: usize) -> Vec<Vec<usize>> {
    let max_batch = max_batch.max(1);
    (0..n)
        .step_by(max_batch)
        .map(|i| (i..(i + max_batch).min(n)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_planning_partitions_every_index_exactly_once() {
        for n in 0..40usize {
            for max in [1usize, 2, 4, 8, 16] {
                let groups = plan_batches(n, max);
                let mut seen: Vec<usize> = groups.iter().flatten().copied().collect();
                seen.sort_unstable();
                assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n} max={max}");
                assert!(
                    groups.iter().all(|g| !g.is_empty() && g.len() <= max),
                    "n={n} max={max} produced {groups:?}"
                );
            }
        }
    }

    #[test]
    fn batch_planning_fills_groups_before_starting_another() {
        assert_eq!(plan_batches(8, 8), vec![(0..8).collect::<Vec<_>>()]);
        assert_eq!(plan_batches(1, 8), vec![vec![0]]);
        assert_eq!(plan_batches(0, 8), Vec::<Vec<usize>>::new());
        let g = plan_batches(10, 8);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].len(), 8);
        assert_eq!(g[1], vec![8, 9]);
    }
}
