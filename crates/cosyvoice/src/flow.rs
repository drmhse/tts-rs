//! `CausalMaskedDiffWithDiT`: speech tokens -> mel, by conditional flow matching.
//!
//! Token embedding, a speaker affine, a small look-ahead convolution pair, then a
//! 22-block DiT driven by a 10-step Euler solve with classifier-free guidance. This is
//! the expensive stage: **10 steps x 2 (guidance) x 22 blocks = 440 block passes over
//! the full mel length**, against Audio8's codec which is one feed-forward pass.
//!
//! # The trap that dominates this file
//!
//! **RoPE is applied to head 0 only.** The reference calls
//! `apply_rotary_pos_emb(query, freqs)` on the *pre-reshape* `[b, n, 1024]` projection,
//! and `x_transformers` implements partial rotary embeddings: it rotates the first
//! `freqs.shape[-1]` channels and passes the rest through. `freqs` is built from
//! `RotaryEmbedding(dim_head=64)`, so `rot_dim` is 64 — and channels 0..63 are exactly
//! head 0 of 16. Verified numerically: applying it to a tensor of ones changes channels
//! 0..63 and nothing else.
//!
//! So heads 1..15 have no positional information at all. This is upstream behaviour, not
//! a simplification, and it is the single most likely thing for a port to "fix" into
//! being wrong — applying RoPE per-head to all 16 heads is what any reasonable
//! implementation would do, produces plausible audio, and is a different model. It also
//! means 15/16 of the rotary work does not exist, which the port takes advantage of.
//!
//! # Other things worth knowing
//!
//! **The initial noise is a fixed asset, not a draw.** `CausalConditionalCFM.__init__`
//! seeds torch and builds `randn([1, 80, 15000])` once, then slices it every call. Output
//! is therefore deterministic given the tokens. Reproducing torch's Philox stream from
//! Rust is impractical, so the tensor ships as `rand_noise.safetensors`.
//!
//! **Ten Euler steps, not six.** `flow.inference` passes `n_timesteps=10`. An earlier
//! draft of this port's plan said six, which would have been a 40% underestimate of the
//! stage that dominates.
//!
//! **Non-streaming has no attention mask.** `add_optional_chunk_mask` with
//! `static_chunk_size=0` returns the padding mask unchanged, which for a single
//! unpadded sequence is all-ones; the subsequent `repeat` and `masked_fill` are then
//! both no-ops. The port omits the mask entirely rather than building an all-true one
//! and paying for it 440 times.
//!
//! **74% of the solver's work is discarded.** The prompt occupies 588 of 798 frames here
//! and `flow.inference` throws its mel away, keeping only the generated tail. That is
//! inherent, not wasteful: attention is bidirectional, so the generated frames need the
//! prompt present to be conditioned on it. It does mean the stage's RTF improves with
//! longer text, because the fixed prompt cost amortises — worth remembering when reading
//! a benchmark taken on one short utterance.

use crate::cfg::flow as k;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::f32::consts::PI;
use tts_nn::{
    causal_conv1d, fused, gelu_tanh, grouped_causal_conv1d, l2_normalize, leaky_relu,
    lookahead_conv1d, mish, LayerNormPlain, Linear, Weights,
};

/// Six modulation vectors per DiT block, in the reference's chunk order.
struct Modulation {
    shift_msa: Tensor,
    scale_msa: Tensor,
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

struct DiTBlock {
    /// `[1024, 6144]` after transposition — the AdaLayerNormZero projection.
    ada: Linear,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    ff_in: Linear,
    ff_out: Linear,
}

/// Reorder the head blocks of a `[out, in]` projection so head 0 moves to the end.
///
/// Attention is independent across heads, so relabelling them changes nothing — provided
/// every projection agrees. `to_q`, `to_k` and `to_v` have heads along their *output* rows;
/// `to_out` consumes them along its *input* columns, so it gets the same permutation applied
/// to `dim 1`. Done once at load.
///
/// The point is to make the *rotated* head the last one. See [`DiTBlock::attention`].
fn move_head0_last(t: &Tensor, dim: usize) -> Result<Tensor> {
    let width = t.dim(dim)?;
    debug_assert_eq!(width, k::HEADS * k::HEAD_DIM);
    let rest = t.narrow(dim, k::HEAD_DIM, width - k::HEAD_DIM)?;
    let head0 = t.narrow(dim, 0, k::HEAD_DIM)?;
    Ok(Tensor::cat(&[rest, head0], dim)?.contiguous()?)
}

fn load_permuted(w: &Weights, prefix: &str, dim: usize) -> Result<Linear> {
    let weight = move_head0_last(&w.get(&format!("{prefix}.weight"))?, dim)?;
    let bias = if dim == 0 {
        Some(move_head0_last(&w.get(&format!("{prefix}.bias"))?, 0)?)
    } else {
        Some(w.get(&format!("{prefix}.bias"))?)
    };
    Linear::new(&weight, bias)
}

impl DiTBlock {
    fn load(w: &Weights, i: usize) -> Result<Self> {
        let p = format!("decoder.estimator.transformer_blocks.{i}");
        Ok(Self {
            ada: Linear::load(w, &format!("{p}.attn_norm.linear"), true)?,
            // Heads relabelled so the rotated one is last; `to_out` matches on its inputs.
            to_q: load_permuted(w, &format!("{p}.attn.to_q"), 0)?,
            to_k: load_permuted(w, &format!("{p}.attn.to_k"), 0)?,
            to_v: load_permuted(w, &format!("{p}.attn.to_v"), 0)?,
            to_out: load_permuted(w, &format!("{p}.attn.to_out.0"), 1)?,
            // `FeedForward` nests a Sequential inside a Sequential, hence `ff.ff.0.0`.
            ff_in: Linear::load(w, &format!("{p}.ff.ff.0.0"), true)?,
            ff_out: Linear::load(w, &format!("{p}.ff.ff.2"), true)?,
        })
    }

    /// `t` is `[b, 1024]`; the modulation vectors come out `[b, 1, 1024]` so they
    /// broadcast over positions.
    fn modulation(&self, t: &Tensor) -> Result<Modulation> {
        let emb = self.ada.forward(&candle_nn::ops::silu(t)?)?;
        let b = emb.dim(0)?;
        let take = |i: usize| -> Result<Tensor> {
            Ok(emb.narrow(1, i * k::DIM, k::DIM)?.reshape((b, 1, k::DIM))?)
        };
        Ok(Modulation {
            shift_msa: take(0)?,
            scale_msa: take(1)?,
            gate_msa: take(2)?,
            shift_mlp: take(3)?,
            scale_mlp: take(4)?,
            gate_mlp: take(5)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        norm: &LayerNormPlain,
    ) -> Result<Tensor> {
        let m = self.modulation(t)?;
        let normed = modulate(norm, x, &m.scale_msa, &m.shift_msa)?;
        let attn = self.attention(&normed, cos, sin)?;
        let x = fused::gate_residual(x, &attn, &m.gate_msa)?;

        let ff_in = modulate(norm, &x, &m.scale_mlp, &m.shift_mlp)?;
        let ff = self
            .ff_out
            .forward(&gelu_tanh(&self.ff_in.forward(&ff_in)?)?)?;
        Ok(fused::gate_residual(&x, &ff, &m.gate_mlp)?)
    }

    /// Multi-head attention through candle's fused Metal SDPA kernel.
    ///
    /// Two measured choices here, both exact (the fused kernel agrees with the naive form
    /// to rel 2.1e-6, inside the f32 floor this network already sits at):
    ///
    /// - **Fused rather than assembled.** Writing `softmax(qk^T * scale) v` out of
    ///   separate ops materialises a `[2, 16, 798, 798]` scores tensor — 81.5 MB — and
    ///   then touches it four times: write, scale, softmax, read. That is ~490 MB of
    ///   traffic for 2.6 GMAC of arithmetic, which is why attention measured *8.2x* the
    ///   cost of a projection while doing 5x less work. The fused kernel is **2.6x**
    ///   faster and never materialises it.
    /// - **Transposed views, not contiguous copies.** `sdpa` takes strides, so the
    ///   `[b, n, 16, 64] -> [b, 16, n, 64]` transposes can stay lazy. Calling
    ///   `contiguous()` first — the obvious thing to write — costs three 6.5 MB copies
    ///   and measured **2.7x slower**. Together these take attention from 11.5 ms to
    ///   2.25 ms per block.
    ///
    /// No mask is passed: non-streaming attention here is fully bidirectional, so there is
    /// nothing to add and nothing to fill afterwards.
    ///
    /// # Partial rotary, without rebuilding the projection
    ///
    /// Only head 0 is rotated, and attention is independent across heads, so the rotated head
    /// and the other fifteen can be two `sdpa` calls over views — instead of
    /// [`partial_rope`], which rebuilds a 6.5 MB tensor for `q` and again for `k` to rotate a
    /// sixteenth of it. Measured **1.25x on a whole block**, landing within 0.25 ms of a
    /// variant with the rotary deleted entirely.
    ///
    /// That was blocked, and the way round it is the head permutation at load. candle
    /// 0.10.2's Metal `sdpa` returns wrong results when the head axis is narrowed to a
    /// **non-zero offset**: `narrow(1, 0, 1)` agrees with the naive form to rel 6.7e-7, but
    /// `narrow(1, 1, 15)` comes back at rel **1.24**. Wired in naively that put DiT block 0
    /// off by rel 1.5e-1.
    ///
    /// So the heads are relabelled at load to put the rotated one *last*. The fifteen
    /// unrotated heads are then `narrow(1, 0, 15)` — offset zero, which the kernel handles —
    /// and the rotated head is `narrow(1, 15, 1)`, which does have an offset but is one
    /// sixteenth of the tensor and so is cheap to make contiguous first. The bug is avoided
    /// rather than worked around, and the 1.25x is recovered.
    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        // Deliberately *not* contiguous: `sdpa` takes strides, and calling `contiguous()`
        // first measured 2.7x slower.
        let heads = |t: &Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, n, k::HEADS, k::HEAD_DIM))?.transpose(1, 2)?)
        };
        let qh = heads(&self.to_q.forward(x)?)?;
        let kh = heads(&self.to_k.forward(x)?)?;
        let vh = heads(&self.to_v.forward(x)?)?;
        let scale = (1.0 / (k::HEAD_DIM as f64).sqrt()) as f32;

        // Heads 0..14 after the permutation: unrotated, and an offset-zero view.
        let plain = k::HEADS - 1;
        let head = |t: &Tensor| -> Result<Tensor> { Ok(t.narrow(1, 0, plain)?) };
        let o_plain = candle_nn::ops::sdpa(
            &head(&qh)?,
            &head(&kh)?,
            &head(&vh)?,
            None,
            false,
            scale,
            1.0,
        )?;

        // The last head is the rotated one. `contiguous` on `[b, 1, n, 64]` costs a
        // sixteenth of what rebuilding the whole projection did.
        let last = |t: &Tensor| -> Result<Tensor> { Ok(t.narrow(1, plain, 1)?.contiguous()?) };
        let rot = |t: &Tensor| -> Result<Tensor> {
            Ok(candle_nn::rotary_emb::rope_i(&last(t)?, cos, sin)?)
        };
        let o_rot =
            candle_nn::ops::sdpa(&rot(&qh)?, &rot(&kh)?, &last(&vh)?, None, false, scale, 1.0)?;

        let out = Tensor::cat(&[o_plain, o_rot], 1)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, n, k::HEADS * k::HEAD_DIM))?;
        self.to_out.forward(&out)
    }
}

/// `layer_norm(x) * (1 + scale) + shift`, with no affine parameters in the norm.
///
/// The norm goes through [`LayerNormPlain`]'s fused kernel rather than being written out
/// of primitives — the hand-written form is six passes over a 6.5 MB tensor and this runs
/// three times per block, 660 times per utterance. Measured 5.51x.
fn modulate(norm: &LayerNormPlain, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let n = norm.forward(x)?;
    Ok(fused::modulate_affine(&n, scale, shift)?)
}

/// Rotate the first `HEAD_DIM` channels of `[b, n, dim]` and pass the rest through.
///
/// This is the partial-rotary trap from the module docstring, written out literally: the
/// pairing is interleaved — `x_transformers`' `rotate_half` splits `(d r)` with `r = 2`,
/// so adjacent channels form each pair, which is `rope_i`'s convention and not `rope`'s.
///
/// This costs a full rebuild of the projection to rotate a sixteenth of it — 5.0 ms of a
/// block's 23 ms — so [`DiTBlock::attention`] does not use it. Kept because it states the
/// rotation plainly, and because the unit test below asserts the property the fast form
/// depends on: that exactly the first head moves.
#[allow(dead_code)] // the readable reference; `DiTBlock::attention` is the fast equivalent
fn partial_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, n, dim) = x.dims3()?;
    let rot = x
        .narrow(2, 0, k::HEAD_DIM)?
        .reshape((b, 1, n, k::HEAD_DIM))?;
    let rot = candle_nn::rotary_emb::rope_i(&rot.contiguous()?, cos, sin)?;
    let rot = rot.reshape((b, n, k::HEAD_DIM))?;
    if dim == k::HEAD_DIM {
        return Ok(rot);
    }
    let rest = x.narrow(2, k::HEAD_DIM, dim - k::HEAD_DIM)?;
    Ok(Tensor::cat(&[rot, rest], 2)?.contiguous()?)
}

/// The cosine timestep schedule, `[n_timesteps + 1]`.
///
/// Computed in f32 in the same op order as the reference, because the solver accumulates
/// `t` and the accumulated value feeds a sinusoidal embedding — so a schedule that is
/// merely close gives a slightly different conditioning at every step.
pub fn t_span() -> Vec<f32> {
    (0..=k::N_TIMESTEPS)
        .map(|i| {
            let v = i as f32 / k::N_TIMESTEPS as f32;
            1.0 - (v * 0.5 * PI).cos()
        })
        .collect()
}

/// The DiT estimator: the flow field `dphi/dt`.
struct Dit {
    time_mlp_1: Linear,
    time_mlp_2: Linear,
    /// `[1024, 320]`: `[x | cond | mu | spks]` -> the model width.
    input_proj: Linear,
    /// The convolutional position embedding: two grouped causal convs with Mish.
    pos_w1: Tensor,
    pos_b1: Tensor,
    pos_w2: Tensor,
    pos_b2: Tensor,
    blocks: Vec<DiTBlock>,
    norm_out: Linear,
    proj_out: Linear,
    /// Sinusoidal timestep-embedding frequencies, `[1, 128]`.
    time_freqs: Tensor,
    /// One affine-free LayerNorm, shared by all 22 blocks and the final modulation.
    norm: LayerNormPlain,
}

impl Dit {
    fn load(w: &Weights, device: &Device) -> Result<Self> {
        let e = "decoder.estimator";
        let blocks = (0..k::DEPTH)
            .map(|i| DiTBlock::load(w, i))
            .collect::<Result<Vec<_>>>()?;

        // SinusPositionEmbedding: exp(arange(half) * -(ln(10000) / (half - 1))).
        let half = k::TIME_EMBED_DIM / 2;
        let step = (10_000f64).ln() / (half - 1) as f64;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(i as f64) * step).exp() as f32)
            .collect();

        Ok(Self {
            time_mlp_1: Linear::load(w, &format!("{e}.time_embed.time_mlp.0"), true)?,
            time_mlp_2: Linear::load(w, &format!("{e}.time_embed.time_mlp.2"), true)?,
            input_proj: Linear::load(w, &format!("{e}.input_embed.proj"), true)?,
            pos_w1: w.get(&format!("{e}.input_embed.conv_pos_embed.conv1.0.weight"))?,
            pos_b1: w.get(&format!("{e}.input_embed.conv_pos_embed.conv1.0.bias"))?,
            pos_w2: w.get(&format!("{e}.input_embed.conv_pos_embed.conv2.0.weight"))?,
            pos_b2: w.get(&format!("{e}.input_embed.conv_pos_embed.conv2.0.bias"))?,
            blocks,
            norm_out: Linear::load(w, &format!("{e}.norm_out.linear"), true)?,
            proj_out: Linear::load(w, &format!("{e}.proj_out"), true)?,
            time_freqs: Tensor::from_vec(freqs, (1, half), device)?,
            norm: LayerNormPlain::new(k::DIM, k::LAYER_NORM_EPS, device)?,
        })
    }

    /// `[b]` timesteps -> `[b, 1024]`.
    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let b = t.dim(0)?;
        let scaled = t.reshape((b, 1))?.affine(k::TIME_SCALE, 0.0)?;
        let arg = scaled.broadcast_mul(&self.time_freqs)?;
        let emb = Tensor::cat(&[arg.sin()?, arg.cos()?], 1)?.contiguous()?;
        let h = self.time_mlp_1.forward(&emb)?;
        self.time_mlp_2.forward(&candle_nn::ops::silu(&h)?)
    }

    /// The convolutional position embedding, applied residually.
    fn pos_embed(&self, x: &Tensor) -> Result<Tensor> {
        // [b, n, d] -> [b, d, n]
        let h = x.transpose(1, 2)?.contiguous()?;
        let h = mish(&grouped_causal_conv1d(
            &h,
            &self.pos_w1,
            Some(&self.pos_b1),
            k::CONV_POS_GROUPS,
        )?)?;
        let h = mish(&grouped_causal_conv1d(
            &h,
            &self.pos_w2,
            Some(&self.pos_b2),
            k::CONV_POS_GROUPS,
        )?)?;
        Ok((h.transpose(1, 2)?.contiguous()? + x)?)
    }

    /// The `[x | cond | mu | spks]` concatenation and its projection to the model width.
    ///
    /// `InputEmbedding.forward` takes `(x, cond, text_embed, spks)` and `DiT.forward`
    /// passes `mu` as `text_embed`, so the concatenation order is x, cond, mu, spks —
    /// not the order the parameter names suggest.
    fn input_embed(&self, x: &Tensor, mu: &Tensor, cond: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let (b, _, n) = x.dims3()?;
        let spks = spks
            .reshape((b, 1, k::SPK_DIM))?
            .broadcast_as((b, n, k::SPK_DIM))?;
        let cat = Tensor::cat(
            &[
                x.transpose(1, 2)?.contiguous()?,
                cond.transpose(1, 2)?.contiguous()?,
                mu.transpose(1, 2)?.contiguous()?,
                spks.contiguous()?,
            ],
            2,
        )?;
        self.input_proj.forward(&cat)
    }

    /// `x, mu, cond` are `[b, 80, n]`, `spks` is `[b, 80]`, `t` is `[b]`.
    /// Returns `[b, 80, n]`.
    // The DiT's inputs are what the reference's signature is; grouping them would only
    // hide which of them the CFG batch doubles.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        spks: &Tensor,
        t: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let b = x.dim(0)?;
        let t = self.time_embed(t)?;
        let mut h = self.pos_embed(&self.input_embed(x, mu, cond, spks)?)?;

        for block in &self.blocks {
            h = block.forward(&h, &t, cos, sin, &self.norm)?;
        }

        // AdaLayerNormZero_Final: note the chunk order is scale *then* shift, the
        // opposite of the per-block modulation's shift-then-scale.
        let emb = self.norm_out.forward(&candle_nn::ops::silu(&t)?)?;
        let scale = emb.narrow(1, 0, k::DIM)?.reshape((b, 1, k::DIM))?;
        let shift = emb.narrow(1, k::DIM, k::DIM)?.reshape((b, 1, k::DIM))?;
        let h = modulate(&self.norm, &h, &scale, &shift)?;

        Ok(self.proj_out.forward(&h)?.transpose(1, 2)?.contiguous()?)
    }
}

pub struct Flow {
    /// `[6561, 80]`.
    token_embed: Tensor,
    spk_affine: Linear,
    look_w1: Tensor,
    look_b1: Tensor,
    look_w2: Tensor,
    look_b2: Tensor,
    dit: Dit,
    /// `[1, 80, 15000]`, the fixed initial noise.
    rand_noise: Tensor,
    device: Device,
}

impl Flow {
    pub fn load(path: &str, noise_path: &str, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let noise = Weights::load(noise_path, device)?;
        let rand_noise = noise.get("rand_noise")?;
        if rand_noise.dim(2)? < k::RAND_NOISE_FRAMES {
            bail!(
                "rand_noise has {} frames, expected {}",
                rand_noise.dim(2)?,
                k::RAND_NOISE_FRAMES
            );
        }

        let flow = Self {
            token_embed: w.get("input_embedding.weight")?,
            spk_affine: Linear::load(&w, "spk_embed_affine_layer", true)?,
            look_w1: w.get("pre_lookahead_layer.conv1.weight")?,
            look_b1: w.get("pre_lookahead_layer.conv1.bias")?,
            look_w2: w.get("pre_lookahead_layer.conv2.weight")?,
            look_b2: w.get("pre_lookahead_layer.conv2.bias")?,
            dit: Dit::load(&w, device)?,
            rand_noise,
            device: device.clone(),
        };
        flow.check_geometry()?;
        Ok(flow)
    }

    fn check_geometry(&self) -> Result<()> {
        if self.token_embed.dim(0)? != k::TOKEN_VOCAB {
            bail!(
                "input_embedding has {} rows, expected {}",
                self.token_embed.dim(0)?,
                k::TOKEN_VOCAB
            );
        }
        if self.dit.blocks.len() != k::DEPTH {
            bail!(
                "loaded {} DiT blocks, expected {}",
                self.dit.blocks.len(),
                k::DEPTH
            );
        }
        if self.look_w1.dim(2)? != k::PRE_LOOKAHEAD_LEN + 1 {
            bail!(
                "pre_lookahead conv1 kernel {} != pre_lookahead_len + 1 = {}",
                self.look_w1.dim(2)?,
                k::PRE_LOOKAHEAD_LEN + 1
            );
        }
        Ok(())
    }

    /// `[1, 192]` speaker embedding -> `[1, 80]`.
    pub fn speaker(&self, embedding: &Tensor) -> Result<Tensor> {
        self.spk_affine.forward(&l2_normalize(embedding)?)
    }

    /// Speech tokens -> `[1, n, 80]`.
    ///
    /// The reference multiplies by a padding mask here; for a single unpadded sequence
    /// that mask is all-ones, so it is omitted.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?;
        Ok(self.token_embed.index_select(&idx, 0)?.unsqueeze(0)?)
    }

    /// `PreLookaheadLayer`: a forward-looking conv, a causal one, and a residual.
    ///
    /// Both non-linearities here are bare `F.leaky_relu` calls, so torch's default slope
    /// of 0.01 — not the 0.1 the vocoder configures.
    pub fn pre_lookahead(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.transpose(1, 2)?.contiguous()?;
        let h = lookahead_conv1d(&h, &self.look_w1, Some(&self.look_b1), 1)?;
        let h = leaky_relu(&h, 0.01)?;
        let h = causal_conv1d(&h, &self.look_w2, Some(&self.look_b2), 1)?;
        Ok((h.transpose(1, 2)?.contiguous()? + x)?)
    }

    /// `[1, n, 80]` -> `[1, 80, 2n]`: each token's embedding held for two mel frames.
    pub fn mu(&self, h: &Tensor) -> Result<Tensor> {
        let (b, n, d) = h.dims3()?;
        let r = crate::cfg::TOKEN_MEL_RATIO;
        let repeated = h
            .reshape((b, n, 1, d))?
            .broadcast_as((b, n, r, d))?
            .reshape((b, n * r, d))?;
        Ok(repeated.transpose(1, 2)?.contiguous()?)
    }

    /// The prompt mel in the leading frames, zeros after: `[1, 80, total]`.
    pub fn conditioning(&self, prompt_mel: &Tensor, total: usize) -> Result<Tensor> {
        let (_, len, d) = prompt_mel.dims3()?;
        if len > total {
            bail!("prompt mel is {len} frames but the target is only {total}");
        }
        let pad = Tensor::zeros((1, total - len, d), DType::F32, &self.device)?;
        Ok(Tensor::cat(&[prompt_mel.clone(), pad], 1)?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// One DiT evaluation on the doubled guidance batch, returned as `[2, 80, n]`.
    ///
    /// Exposed so the fixture gate can separate a wrong block from a wrong solver.
    pub fn estimate(
        &self,
        x: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        spks: &Tensor,
        t: f32,
    ) -> Result<Tensor> {
        let (cos, sin) = self.rope(x.dim(2)?)?;
        let (x2, mu2, cond2, spks2, t2) = self.guidance_batch(x, mu, cond, spks, t)?;
        self.dit.forward(&x2, &mu2, &cond2, &spks2, &t2, &cos, &sin)
    }

    /// The doubled classifier-free-guidance batch.
    ///
    /// Row 0 is conditioned; row 1 has `mu`, `cond` and `spks` zeroed while `x` and `t`
    /// are shared. That is the reference's `x_in[:] = x; mu_in[0] = mu; ...` pattern,
    /// where the zero rows come from the buffers being freshly zeroed each step.
    fn guidance_batch(
        &self,
        x: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        spks: &Tensor,
        t: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let n = x.dim(2)?;
        let zeros = Tensor::zeros((1, k::SPK_DIM, n), DType::F32, &self.device)?;
        Ok((
            Tensor::cat(&[x.clone(), x.clone()], 0)?,
            Tensor::cat(&[mu.clone(), zeros.clone()], 0)?,
            Tensor::cat(&[cond.clone(), zeros], 0)?,
            Tensor::cat(
                &[
                    spks.clone(),
                    Tensor::zeros((1, k::SPK_DIM), DType::F32, &self.device)?,
                ],
                0,
            )?,
            Tensor::from_vec(vec![t, t], 2, &self.device)?,
        ))
    }

    /// The DiT's internal stages for one doubled-batch evaluation, for the fixture gate.
    ///
    /// Returns `(time_embedding, post-input-embedding, per-block outputs)`. Exposed
    /// because "the DiT is off by rel 4e-4" is not actionable on its own — it could be
    /// f32 accumulation over 22 blocks or one wrong layer, and only a per-block trace
    /// distinguishes them.
    pub fn trace(
        &self,
        x: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        spks: &Tensor,
        t: f32,
    ) -> Result<(Tensor, Tensor, Vec<Tensor>)> {
        let n = x.dim(2)?;
        let (cos, sin) = self.rope(n)?;
        let (x2, mu2, cond2, spks2, t2) = self.guidance_batch(x, mu, cond, spks, t)?;
        let d = &self.dit;
        let time = d.time_embed(&t2)?;
        let h_in = d.pos_embed(&d.input_embed(&x2, &mu2, &cond2, &spks2)?)?;
        let mut h = h_in.clone();
        let mut blocks = Vec::with_capacity(k::DEPTH);
        for block in &d.blocks {
            h = block.forward(&h, &time, &cos, &sin, &d.norm)?;
            blocks.push(h.clone());
        }
        Ok((time, h_in, blocks))
    }

    /// Just the input and position embeddings, for the benchmark to subtract.
    pub fn embed_only(
        &self,
        x: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        spks: &Tensor,
    ) -> Result<Tensor> {
        let (x2, mu2, cond2, spks2, _) = self.guidance_batch(x, mu, cond, spks, 0.0)?;
        self.dit
            .pos_embed(&self.dit.input_embed(&x2, &mu2, &cond2, &spks2)?)
    }

    fn rope(&self, n: usize) -> Result<(Tensor, Tensor)> {
        // f32 tables, not bf16-rounded: `x_transformers.RotaryEmbedding` disables
        // autocast, so the reference never rounds them.
        tts_nn::rope_table_f32(n, k::HEAD_DIM, k::ROPE_BASE, &self.device)
    }

    /// The full Euler solve: `[1, 80, n]`, including the prompt's frames.
    pub fn solve(&self, mu: &Tensor, cond: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let n = mu.dim(2)?;
        if n > self.rand_noise.dim(2)? {
            bail!(
                "{n} mel frames exceeds the {} frames of fixed noise available",
                self.rand_noise.dim(2)?
            );
        }
        let mut x = self.rand_noise.narrow(2, 0, n)?.contiguous()?;
        let span = t_span();
        let mut t = span[0];
        let mut dt = span[1] - span[0];

        for step in 1..=k::N_TIMESTEPS {
            let d = self.estimate(&x, mu, cond, spks, t)?;
            let d_cond = d.narrow(0, 0, 1)?;
            let d_uncond = d.narrow(0, 1, 1)?;
            // (1 + w) * conditional - w * unconditional
            let dphi = ((d_cond * (1.0 + k::CFG_RATE))? - (d_uncond * k::CFG_RATE)?)?;
            x = (x + (dphi * dt as f64)?)?;
            t += dt;
            if step < k::N_TIMESTEPS {
                dt = span[step + 1] - t;
            }
        }
        Ok(x)
    }

    /// Speech tokens plus the voice's conditioning -> mel `[1, 80, frames]`, with the
    /// prompt's frames already trimmed.
    pub fn synthesize(
        &self,
        prompt_tokens: &[u32],
        tokens: &[u32],
        prompt_mel: &Tensor,
        speaker: &Tensor,
    ) -> Result<Tensor> {
        let all: Vec<u32> = prompt_tokens.iter().chain(tokens.iter()).copied().collect();
        let emb = self.embed_tokens(&all)?;
        let mu = self.mu(&self.pre_lookahead(&emb)?)?;
        let total = mu.dim(2)?;
        let cond = self.conditioning(prompt_mel, total)?;
        let mel = self.solve(&mu, &cond, speaker)?;
        let skip = prompt_mel.dim(1)?;
        Ok(mel.narrow(2, skip, total - skip)?.contiguous()?)
    }

    /// Longest mel, in frames, the fixed noise asset can support.
    pub fn max_frames(&self) -> usize {
        self.rand_noise.dim(2).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestep_schedule_is_monotone_and_front_loaded() {
        // 1 - cos(t * pi / 2) over a uniform t: 0 to 1, monotone, and denser at the
        // start than a linear schedule. If this were linear the solver would spend its
        // steps in the wrong place and the mel would be smooth but wrong.
        let span = t_span();
        assert_eq!(span.len(), k::N_TIMESTEPS + 1);
        assert!(span[0].abs() < 1e-7, "starts at {}", span[0]);
        assert!((span[k::N_TIMESTEPS] - 1.0).abs() < 1e-6);
        for i in 1..span.len() {
            assert!(span[i] > span[i - 1], "not monotone at {i}");
        }
        let first = span[1] - span[0];
        let last = span[k::N_TIMESTEPS] - span[k::N_TIMESTEPS - 1];
        assert!(
            first < last,
            "cosine schedule should front-load: {first} vs {last}"
        );
    }

    #[test]
    fn partial_rope_touches_only_the_first_head() -> Result<()> {
        // The trap in the module docstring, asserted. Channels 0..63 rotate; 64..1023
        // pass through untouched. A port that "fixed" this would fail here, which is
        // the point of having it as a test rather than a comment.
        let d = Device::Cpu;
        let n = 12;
        let x = Tensor::ones((1, n, k::DIM), DType::F32, &d)?;
        let (cos, sin) = tts_nn::rope_table_f32(n, k::HEAD_DIM, k::ROPE_BASE, &d)?;
        let y = partial_rope(&x, &cos, &sin)?;
        let delta = (y - &x)?.abs()?.max(1)?.flatten_all()?.to_vec1::<f32>()?;
        let moved: Vec<usize> = delta
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 1e-6)
            .map(|(i, _)| i)
            .collect();
        assert!(!moved.is_empty(), "rope did nothing at all");
        assert_eq!(*moved.first().unwrap(), 0);
        assert!(
            *moved.last().unwrap() < k::HEAD_DIM,
            "rope reached channel {} — heads beyond the first must be untouched",
            moved.last().unwrap()
        );
        Ok(())
    }
}
