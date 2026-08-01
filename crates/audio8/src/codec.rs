//! The codec decoder: 10 codebooks of RVQ codes -> 44.1 kHz waveform.
//!
//! Mirrors `ArkttsCodec.decode` = `decoder(quantizer.decode(codes))`. Weights come
//! from `references/audio8/weights/codec.safetensors`, where `references/audio8/convert_codec.py` has already
//! folded weight_norm (`weight = g * v / ||v||`) and dropped the encoder.
//!
//! Total upsampling is 4 (quantizer) x 512 (decoder) = 2048 samples per code frame,
//! which is `codec_frame_size` and gives the 21.53 Hz frame rate.

use crate::cfg;
use crate::nn::{
    causal_conv1d, causal_conv1d_gemm, causal_conv_transpose1d, causal_window_mask, depthwise_k7,
    fused, layer_norm, rms_norm, rope_table, swiglu, tap_major_weight, Weights,
};
use anyhow::Result;
use candle_core::{Device, Tensor, D};

/// A snake activation, optionally with its scaling folded into neighbouring convs.
///
/// `snake(x) = x + a^-1 sin^2(a x)`. Substituting `u = a x` gives
///
/// ```text
/// snake(x) = u/a + a^-1 sin^2 u = a^-1 * (u + sin^2 u)
/// ```
///
/// so the whole activation is `a^-1` times a function of `u` alone. Both scalings are
/// per-channel constants, which makes them foldable into the surrounding convolutions
/// at load time — exactly, not approximately:
///
/// - `a^-1` folds into the **following** conv's input-channel weights. Every snake on
///   the decode path is followed by a conv, so this is always available.
/// - `a` folds into the **preceding** conv's output weights and bias, but only where
///   that conv's output feeds nothing but this snake. Inside a residual unit the
///   block input also feeds the skip, so the leading snake cannot take this half.
///
/// The win is dispatch count, not arithmetic: each fold removes a `broadcast_mul`, and
/// `broadcast_mul` measured 4.66 ms against 1.29 ms for a plain unary pass at
/// `[1, 96, 131072]` — 3.6x, because broadcasting a `[1, C, 1]` costs far more than the
/// index arithmetic suggests.
struct Snake {
    alpha: Tensor,
    /// Set when the producer already multiplied by `alpha`, so `u` arrives directly.
    pre_scaled: bool,
}

impl Snake {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self {
            alpha: w.get(name)?,
            pre_scaled: false,
        })
    }

    /// `(alpha + 1e-9)^-1` — note the epsilon is inside the reciprocal in the reference,
    /// not added to the result.
    fn recip(&self) -> Result<Tensor> {
        Ok((&self.alpha + 1e-9)?.recip()?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.pre_scaled {
            // u arrives pre-multiplied and a^-1 lives in the next conv. Three unary
            // passes become one: see `tts_nn::fused`.
            Ok(fused::snake_folded(x)?)
        } else {
            Ok(fused::snake_alpha(x, &self.alpha)?)
        }
    }
}

struct Conv {
    weight: Tensor,
    bias: Tensor,
    dilation: usize,
    /// `[out, k * in]`, built by [`Conv::finalize`] once folding is done.
    ///
    /// `None` for `k == 1`, where the GEMM route has nothing to gather and the direct call is
    /// already a matmul.
    w_tap: Option<Tensor>,
}

impl Conv {
    fn load(w: &Weights, prefix: &str, dilation: usize) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.conv.weight"))?,
            bias: w.get(&format!("{prefix}.conv.bias"))?,
            dilation,
            w_tap: None,
        })
    }

    /// Precompute the tap-major weight. **Must be called after all folding**, and calling it
    /// before would silently bake the unfolded weight into the fast path while `weight` kept
    /// the folded one — so `forward` asserts the two agree in shape rather than trusting call
    /// order.
    fn finalize(&mut self) -> Result<()> {
        if self.weight.dim(2)? > 1 {
            self.w_tap = Some(tap_major_weight(&self.weight)?);
        }
        Ok(())
    }

    /// Scale input channels — absorbs a preceding snake's `alpha^-1`.
    /// Weight is `[out, in, k]`, so the scale broadcasts on dimension 1.
    fn fold_input(&mut self, recip: &Tensor) -> Result<()> {
        let c = self.weight.dim(1)?;
        self.weight = self.weight.broadcast_mul(&recip.reshape((1, c, 1))?)?;
        Ok(())
    }

    /// Scale output channels and bias — makes this conv emit `alpha * y`, so the
    /// following snake receives `u` directly.
    fn fold_output(&mut self, alpha: &Tensor) -> Result<()> {
        let c = self.weight.dim(0)?;
        self.weight = self.weight.broadcast_mul(&alpha.reshape((c, 1, 1))?)?;
        self.bias = self.bias.mul(&alpha.reshape(c)?)?;
        Ok(())
    }

    /// The GEMM route when a tap-major weight exists, the direct kernel otherwise.
    ///
    /// Measured 1.34x to 1.73x over `conv1d` across the codec's four stages, gaining most at
    /// the low-channel long-length end where the deficit was worst. See
    /// [`tts_nn::causal_conv1d_gemm`].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.w_tap {
            Some(w_tap) => causal_conv1d_gemm(
                x,
                w_tap,
                Some(&self.bias),
                self.weight.dim(2)?,
                self.dilation,
            ),
            None => causal_conv1d(x, &self.weight, Some(&self.bias), self.dilation),
        }
    }
}

/// `Snake -> conv k7 (dilated) -> Snake -> conv k1`, plus the skip.
struct ResidualUnit {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
}

impl ResidualUnit {
    fn load(w: &Weights, prefix: &str, dilation: usize) -> Result<Self> {
        let s1 = Snake::load(w, &format!("{prefix}.block.0.alpha"))?;
        let mut c1 = Conv::load(w, &format!("{prefix}.block.1"), dilation)?;
        let mut s2 = Snake::load(w, &format!("{prefix}.block.2.alpha"))?;
        let mut c2 = Conv::load(w, &format!("{prefix}.block.3"), 1)?;

        // s1 keeps its alpha multiply: the unit's input also feeds the skip, so the
        // producer cannot be asked to pre-scale it. Its alpha^-1 does fold forward.
        c1.fold_input(&s1.recip()?)?;
        // c1's output feeds nothing but s2, so both halves of s2 fold away.
        c1.fold_output(&s2.alpha)?;
        s2.pre_scaled = true;
        c2.fold_input(&s2.recip()?)?;

        // Folding is complete; only now is the tap-major weight the right one to bake.
        c1.finalize()?;
        c2.finalize()?;

        Ok(Self { s1, c1, s2, c2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.s1.forward(x)?;
        let h = self.c1.forward(&h)?;
        let h = self.s2.forward(&h)?;
        let h = self.c2.forward(&h)?;
        // Causal, stride-1 convs preserve length, so the reference's trim is a no-op
        // here. Assert rather than silently mis-add if that ever stops being true.
        debug_assert_eq!(h.dim(2)?, x.dim(2)?);
        Ok((x + h)?)
    }
}

/// `Snake -> causal convT (k = 2*stride) -> 3 residual units at dilation 1, 3, 9`.
struct DecoderBlock {
    snake: Snake,
    up_w: Tensor,
    up_b: Tensor,
    stride: usize,
    units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(w: &Weights, index: usize, stride: usize) -> Result<Self> {
        let p = format!("decoder.model.{index}");
        let mut units = Vec::new();
        for (i, d) in [1usize, 3, 9].into_iter().enumerate() {
            units.push(ResidualUnit::load(w, &format!("{p}.block.{}", i + 2), d)?);
        }
        let snake = Snake::load(w, &format!("{p}.block.0.alpha"))?;
        // The transposed conv stores weights as [in, out, k], so the input scale
        // broadcasts on dimension 0 rather than 1.
        let up_w = w.get(&format!("{p}.block.1.conv.weight"))?;
        let cin = up_w.dim(0)?;
        let up_w = up_w.broadcast_mul(&snake.recip()?.reshape((cin, 1, 1))?)?;
        Ok(Self {
            snake,
            up_w,
            up_b: w.get(&format!("{p}.block.1.conv.bias"))?,
            stride,
            units,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let mut h = causal_conv_transpose1d(&h, &self.up_w, Some(&self.up_b), self.stride)?;
        for u in &self.units {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// ConvNeXt block: depthwise k7 -> LayerNorm -> pw 4x -> GELU (erf) -> pw -> gamma.
struct ConvNeXt {
    dw_w: Tensor,
    dw_b: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
    pw1_w: Tensor,
    pw1_b: Tensor,
    pw2_w: Tensor,
    pw2_b: Tensor,
    gamma: Tensor,
}

impl ConvNeXt {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            dw_w: w.get(&format!("{prefix}.dwconv.conv.weight"))?,
            dw_b: w.get(&format!("{prefix}.dwconv.conv.bias"))?,
            norm_w: w.get(&format!("{prefix}.norm.weight"))?,
            norm_b: w.get(&format!("{prefix}.norm.bias"))?,
            pw1_w: w.get(&format!("{prefix}.pwconv1.weight"))?,
            pw1_b: w.get(&format!("{prefix}.pwconv1.bias"))?,
            pw2_w: w.get(&format!("{prefix}.pwconv2.weight"))?,
            pw2_b: w.get(&format!("{prefix}.pwconv2.bias"))?,
            gamma: w.get(&format!("{prefix}.gamma"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = depthwise_k7(x, &self.dw_w, Some(&self.dw_b))?;
        // [B, C, L] -> [B, L, C] for the pointwise stack.
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = layer_norm(&h, &self.norm_w, &self.norm_b, 1e-6)?;
        let h = h
            .broadcast_matmul(&self.pw1_w.t()?)?
            .broadcast_add(&self.pw1_b)?;
        // nn.GELU() is the exact erf form, not the tanh approximation.
        let h = h.gelu_erf()?;
        let h = h
            .broadcast_matmul(&self.pw2_w.t()?)?
            .broadcast_add(&self.pw2_b)?;
        let h = h.broadcast_mul(&self.gamma)?;
        Ok((x + h.transpose(1, 2)?.contiguous()?)?)
    }
}

/// One block of the codec's windowed transformer.
struct CodecBlock {
    attn_norm: Tensor,
    wqkv: Tensor,
    wo: Tensor,
    attn_scale: Tensor,
    ffn_norm: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    ffn_scale: Tensor,
}

impl CodecBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            attn_norm: w.get(&format!("{prefix}.attention_norm.weight"))?,
            wqkv: w.get(&format!("{prefix}.attention.wqkv.weight"))?,
            wo: w.get(&format!("{prefix}.attention.wo.weight"))?,
            attn_scale: w.get(&format!("{prefix}.attention_layer_scale.gamma"))?,
            ffn_norm: w.get(&format!("{prefix}.ffn_norm.weight"))?,
            w1: w.get(&format!("{prefix}.feed_forward.w1.weight"))?,
            w2: w.get(&format!("{prefix}.feed_forward.w2.weight"))?,
            w3: w.get(&format!("{prefix}.feed_forward.w3.weight"))?,
            ffn_scale: w.get(&format!("{prefix}.ffn_layer_scale.gamma"))?,
        })
    }

    /// `x` is `[B, T, dim]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (nh, nkv, hd) = (cfg::CODEC_POST_N_HEAD, cfg::CODEC_POST_N_KV, cfg::HEAD_DIM);
        let h = rms_norm(x, &self.attn_norm, cfg::CODEC_NORM_EPS)?;
        let qkv = h.broadcast_matmul(&self.wqkv.t()?)?;
        let qs = nh * hd;
        let kvs = nkv * hd;
        let q = qkv
            .narrow(2, 0, qs)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, qs, kvs)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, qs + kvs, kvs)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = candle_nn::rotary_emb::rope_i(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, cos, sin)?;

        // GQA by folding the query heads that share a KV head into the matmul's row
        // dimension: no 8x copy of K and V. [b, nkv, rep*t, hd] @ [b, nkv, hd, t].
        //
        // Deliberately *not* `candle_nn::ops::sdpa`, which won 2.6x on the CosyVoice DiT's
        // attention. Its Metal kernel is **wrong for short sequences when given an additive
        // mask**: measured against the naive form at this head configuration, `t <= 8` comes
        // back at relative error ~1.5 while `t >= 12` is exact to 2e-7. Swapping it in here
        // passed the 24-frame codec fixture at 7.6e-6 and failed the 8-frame one at 6.4e-1,
        // which is what caught it. Segments this short do occur, so the windowed attention
        // stays hand-rolled. See `tts-probe --bin dit`.
        let rep = nh / nkv;
        let q = q.reshape((b, nkv, rep * t, hd))?;
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores
            .reshape((b, nh, t, t))?
            .broadcast_add(&mask.reshape((1, 1, t, t))?)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs
            .reshape((b, nkv, rep * t, t))?
            .matmul(&v.contiguous()?)?
            .reshape((b, nh, t, hd))?;
        let attn = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, qs))?
            .broadcast_matmul(&self.wo.t()?)?;
        // LayerScale sits between the sublayer and the residual add.
        let x = (x + attn.broadcast_mul(&self.attn_scale)?)?;

        let h = rms_norm(&x, &self.ffn_norm, cfg::CODEC_NORM_EPS)?;
        let ffn = swiglu(&h, &self.w1, &self.w3, &self.w2)?;
        Ok((&x + ffn.broadcast_mul(&self.ffn_scale)?)?)
    }
}

/// The `post_module`: 8 windowed transformer blocks over `[B, 1024, T]`.
struct WindowTransformer {
    layers: Vec<CodecBlock>,
    norm: Tensor,
    window: usize,
}

impl WindowTransformer {
    fn load(w: &Weights, prefix: &str, n_layer: usize, window: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..n_layer {
            layers.push(CodecBlock::load(w, &format!("{prefix}.layers.{i}"))?);
        }
        Ok(Self {
            layers,
            norm: w.get(&format!("{prefix}.norm.weight"))?,
            window,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // channels_first: [B, C, T] -> [B, T, C]. input_proj/output_proj are Identity
        // because input_dim == dim == 1024.
        let mut h = x.transpose(1, 2)?.contiguous()?;
        let t = h.dim(1)?;
        let (cos, sin) = rope_table(t, cfg::HEAD_DIM, cfg::CODEC_ROPE_BASE, h.device())?;
        let mask = causal_window_mask(t, Some(self.window), h.device())?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        let h = rms_norm(&h, &self.norm, cfg::CODEC_NORM_EPS)?;
        Ok(h.transpose(1, 2)?.contiguous()?)
    }
}

/// The RVQ decode side, collapsed into one gather per codebook.
///
/// The reference does `out_proj(embedding(code))` per codebook: a `[size, 8]` lookup
/// followed by a `k=1` conv from 8 to 1024 channels. Both are linear, so their
/// composition is a single `[size, 1024]` table computable at load time. That removes
/// 10 embedding lookups and 10 conv dispatches from every decode, and it is exact
/// rather than an approximation. The biases are constant per codebook, so their sum
/// folds into one vector added once instead of ten times.
struct Rvq {
    tables: Vec<Tensor>,
    bias_sum: Tensor,
}

impl Rvq {
    fn load(w: &Weights, device: &Device) -> Result<Self> {
        let mut tables = Vec::new();
        let mut bias: Option<Tensor> = None;
        let specs: Vec<(String, usize)> = std::iter::once((
            "quantizer.semantic_quantizer.quantizers.0".to_string(),
            cfg::CODEBOOK_SIZE,
        ))
        .chain((0..9).map(|i| {
            (
                format!("quantizer.quantizer.quantizers.{i}"),
                cfg::RESIDUAL_CODEBOOK_SIZE,
            )
        }))
        .collect();
        for (prefix, size) in specs {
            let codebook = w.get(&format!("{prefix}.codebook.weight"))?; // [size, 8]
            debug_assert_eq!(codebook.dim(0)?, size);
            // out_proj is Conv1d(8 -> 1024, k=1): weight [1024, 8, 1].
            let proj = w.get(&format!("{prefix}.out_proj.weight"))?.squeeze(2)?; // [1024, 8]
            let b = w.get(&format!("{prefix}.out_proj.bias"))?;
            // table[code] = proj @ codebook[code]
            tables.push(codebook.matmul(&proj.t()?.contiguous()?)?.contiguous()?);
            bias = Some(match bias {
                None => b,
                Some(acc) => (acc + b)?,
            });
        }
        let _ = device;
        Ok(Self {
            tables,
            bias_sum: bias.expect("10 codebooks"),
        })
    }

    /// `codes` is `[num_codebooks, T]` of u32. Returns `[1, 1024, T]`.
    ///
    /// Named for what it reads, not as a constructor — it needs the loaded codebooks.
    #[allow(clippy::wrong_self_convention)]
    fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.dim(1)?;
        let mut acc: Option<Tensor> = None;
        for (i, table) in self.tables.iter().enumerate() {
            let idx = codes.narrow(0, i, 1)?.reshape(t)?;
            let rows = table.index_select(&idx, 0)?; // [T, 1024]
            acc = Some(match acc {
                None => rows,
                Some(a) => (a + rows)?,
            });
        }
        let out = acc.expect("10 codebooks").broadcast_add(&self.bias_sum)?;
        Ok(out.t()?.contiguous()?.unsqueeze(0)?)
    }
}

pub struct Codec {
    rvq: Rvq,
    post: WindowTransformer,
    upsample: Vec<(Tensor, Tensor, ConvNeXt)>,
    entry: Conv,
    blocks: Vec<DecoderBlock>,
    tail_snake: Snake,
    tail_conv: Conv,
    device: Device,
}

impl Codec {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let mut upsample = Vec::new();
        for i in 0..2 {
            upsample.push((
                w.get(&format!("quantizer.upsample.{i}.0.conv.weight"))?,
                w.get(&format!("quantizer.upsample.{i}.0.conv.bias"))?,
                ConvNeXt::load(&w, &format!("quantizer.upsample.{i}.1"))?,
            ));
        }
        let mut blocks = Vec::new();
        for (i, stride) in [8usize, 8, 4, 2].into_iter().enumerate() {
            blocks.push(DecoderBlock::load(&w, i + 1, stride)?);
        }

        // The entry conv's output feeds only block 1's leading snake, so that one gets
        // the pre-scaled path too. Blocks 2-4 do not: their input is a residual *add*,
        // not a conv output, and there is nothing to fold into.
        let mut entry = Conv::load(&w, "decoder.model.0", 1)?;
        entry.fold_output(&blocks[0].snake.alpha)?;
        entry.finalize()?;
        blocks[0].snake.pre_scaled = true;

        // Same story at the tail: the last block ends in a residual add, so the final
        // snake keeps its multiply and only alpha^-1 folds into the output conv.
        let tail_snake = Snake::load(&w, "decoder.model.5.alpha")?;
        let mut tail_conv = Conv::load(&w, "decoder.model.6", 1)?;
        tail_conv.fold_input(&tail_snake.recip()?)?;
        tail_conv.finalize()?;

        Ok(Self {
            rvq: Rvq::load(&w, device)?,
            post: WindowTransformer::load(
                &w,
                "quantizer.post_module",
                cfg::CODEC_POST_N_LAYER,
                cfg::CODEC_WINDOW,
            )?,
            upsample,
            entry,
            blocks,
            tail_snake,
            tail_conv,
            device: device.clone(),
        })
    }

    /// `codes` is `[num_codebooks, T]`, values already clamped by the caller or here.
    /// Returns `[1, 1, T * 2048]` in `[-1, 1]`.
    /// Per-stage wall time for one decode, for benchmarking only.
    ///
    /// The codec is 43% of Audio8's runtime and the conv-as-GEMM change touched only its
    /// *causal* convs, so where the rest sits decides what is worth doing next.
    #[doc(hidden)]
    pub fn bench_stages(&self, codes: &[Vec<u32>]) -> Result<Vec<(&'static str, f64)>> {
        use std::time::Instant;
        let sync = || -> Result<()> {
            self.device.synchronize()?;
            Ok(())
        };
        let t = codes[0].len();
        let mut flat = Vec::with_capacity(cfg::NUM_CODEBOOKS * t);
        for (i, row) in codes.iter().enumerate() {
            let hi = if i == 0 {
                cfg::CODEBOOK_SIZE - 1
            } else {
                cfg::RESIDUAL_CODEBOOK_SIZE - 1
            } as u32;
            flat.extend(row.iter().map(|&c| c.min(hi)));
        }
        let codes = Tensor::from_vec(flat, (cfg::NUM_CODEBOOKS, t), &self.device)?;
        let mut out = Vec::new();

        let s = Instant::now();
        let z = self.rvq.from_codes(&codes)?;
        sync()?;
        out.push(("rvq gather", s.elapsed().as_secs_f64() * 1e3));

        let s = Instant::now();
        let mut z = self.post.forward(&z)?;
        sync()?;
        out.push(("window transformer", s.elapsed().as_secs_f64() * 1e3));

        let (mut up_ms, mut cn_ms) = (0.0, 0.0);
        for (w, b, cn) in &self.upsample {
            let s = Instant::now();
            z = causal_conv_transpose1d(&z, w, Some(b), 2)?;
            sync()?;
            up_ms += s.elapsed().as_secs_f64() * 1e3;
            let s = Instant::now();
            z = cn.forward(&z)?;
            sync()?;
            cn_ms += s.elapsed().as_secs_f64() * 1e3;
        }
        out.push(("upsample conv_transpose", up_ms));
        out.push(("convnext (depthwise)", cn_ms));

        let s = Instant::now();
        let mut h = self.entry.forward(&z)?;
        sync()?;
        out.push(("entry conv", s.elapsed().as_secs_f64() * 1e3));

        let s = Instant::now();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        sync()?;
        out.push(("decoder blocks", s.elapsed().as_secs_f64() * 1e3));

        let s = Instant::now();
        let h = self.tail_snake.forward(&h)?;
        let h = self.tail_conv.forward(&h)?;
        let _ = h.tanh()?;
        sync()?;
        out.push(("tail", s.elapsed().as_secs_f64() * 1e3));
        Ok(out)
    }

    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        let n = codes.len();
        anyhow::ensure!(
            n == cfg::NUM_CODEBOOKS,
            "expected {} codebooks, got {n}",
            cfg::NUM_CODEBOOKS
        );
        let t = codes[0].len();
        anyhow::ensure!(t > 0, "no code frames to decode");
        // The clamp the reference applies in `ArkttsDownsampleQuantizer.decode`:
        // codebook 0 spans 4096 entries, the residual codebooks only 1024.
        let mut flat = Vec::with_capacity(n * t);
        for (i, row) in codes.iter().enumerate() {
            anyhow::ensure!(row.len() == t, "ragged codes: row {i} has {}", row.len());
            let hi = if i == 0 {
                cfg::CODEBOOK_SIZE - 1
            } else {
                cfg::RESIDUAL_CODEBOOK_SIZE - 1
            } as u32;
            flat.extend(row.iter().map(|&c| c.min(hi)));
        }
        let codes = Tensor::from_vec(flat, (n, t), &self.device)?;

        let z = self.rvq.from_codes(&codes)?;
        let mut z = self.post.forward(&z)?;
        for (w, b, cn) in &self.upsample {
            z = causal_conv_transpose1d(&z, w, Some(b), 2)?;
            z = cn.forward(&z)?;
        }
        let mut h = self.entry.forward(&z)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.tail_snake.forward(&h)?;
        let h = self.tail_conv.forward(&h)?;
        Ok(h.tanh()?)
    }
}
