//! Model machinery shared by the engines: convolutions, activations, norms, RoPE
//! tables, weight loading, and the quantized projection wrapper.
//!
//! This crate exists so `cosyvoice` can reuse what the Audio8 port established without
//! depending on `audio8` (see `docs/architecture.md`: engines do not know about each other).
//! Everything here is engine-agnostic; anything that needs a model's geometry belongs
//! in that model's crate.
//!
//! Four things here are not the obvious implementation, and each is either measured or
//! required for fidelity:
//!
//! - [`depthwise_k7`] expresses a `groups == channels` convolution as shifted
//!   multiply-accumulates. Candle's grouped `conv1d` costs 37.78 ms for ~1.8 MMAC of
//!   real work at `[1, 1024, L]`; this form is 1.96 ms for an identical result — 19x.
//!   Never call `conv1d` with `groups > 1` on this backend.
//! - [`rope_table`] rounds the RoPE table through bf16. That is not a shortcut, it is
//!   fidelity: Audio8 builds the table with `torch.polar(...).to(bfloat16)` and then
//!   multiplies it against fp32 activations, so the rounding is part of the model's
//!   arithmetic and skipping it puts the port off the fixtures. CosyVoice's DiT keeps
//!   f32 throughout, so it wants [`rope_table_f32`] — the tables are otherwise
//!   identical and picking the wrong one is a silent accuracy loss.
//! - [`grouped_causal_conv1d`] loops over groups rather than calling candle's
//!   `groups > 1` path, for the same reason as `depthwise_k7`.
//! - [`Weights::get_weight_norm`] resolves torch's `parametrizations.weight.{original0,
//!   original1}` at load time. Both engines' vocoders ship weight-normalised convs and
//!   neither needs the parametrisation at inference.

pub mod fused;
pub mod im2col;
pub(crate) mod mtl;

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor, D};
use std::collections::HashMap;

// ---------------------------------------------------------------- weight loading

/// A loaded safetensors file, with errors that name the tensor you asked for.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl Weights {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("loading {path}"))?;
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Fetch as f32. Checkpoints here are f32 (the folded codec, CosyVoice) or bf16
    /// (Audio8's AR); the port computes in f32 unless a tensor is quantized.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        Ok(t.to_dtype(DType::F32)?)
    }

    /// Same, but keeps the on-disk dtype — used where a copy would be wasteful.
    pub fn raw(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing tensor {name}"))
    }

    /// `get`, or `None` when absent. For genuinely optional tensors only — a typo in a
    /// required name should surface as the error `get` produces, not as a silent `None`.
    pub fn get_opt(&self, name: &str) -> Result<Option<Tensor>> {
        match self.tensors.get(name) {
            None => Ok(None),
            Some(t) => Ok(Some(t.to_dtype(DType::F32)?)),
        }
    }

    /// A weight-normalised conv weight, with the parametrisation resolved.
    ///
    /// torch stores `w = g * v / ||v||` as `original0` (the gain `g`) and `original1`
    /// (the direction `v`), normalising over every dimension but the output channel.
    /// Accepts a plain `.weight` too, so a checkpoint that has already had
    /// `remove_weight_norm` applied loads through the same path.
    pub fn get_weight_norm(&self, prefix: &str) -> Result<Tensor> {
        let plain = format!("{prefix}.weight");
        if self.has(&plain) {
            return self.get(&plain);
        }
        let g = self.get(&format!("{prefix}.parametrizations.weight.original0"))?;
        let v = self.get(&format!("{prefix}.parametrizations.weight.original1"))?;
        fold_weight_norm(&g, &v)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Every tensor name, sorted — for inventory checks and error messages.
    pub fn names(&self) -> Vec<&str> {
        let mut n: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        n.sort_unstable();
        n
    }
}

/// `w = g * v / ||v||`, the norm taken over all dimensions except the first.
///
/// torch's `weight_norm` keeps `dim=0`, so the norm is per output channel. Folding at
/// load leaves inference with an ordinary convolution.
pub fn fold_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let out = v.dim(0)?;
    let flat = v.reshape((out, ()))?;
    let norm = flat.sqr()?.sum_keepdim(1)?.sqrt()?;
    let scale = g.reshape((out, 1))?.broadcast_div(&norm)?;
    Ok(flat.broadcast_mul(&scale)?.reshape(v.shape())?)
}

// ---------------------------------------------------------------- projections

/// A projection, either dense f32 or ggml-quantized.
///
/// The quantized path exists because candle's `quantized/metal.rs` takes a dedicated
/// matrix-vector kernel whenever `dim(-2) == 1` — exactly a decode step — and that
/// kernel reaches 99-139 GB/s on a ~120 GB/s bus. Only the block-32 types are usable
/// for a 896-wide model: the K-quants need `k` divisible by 256.
///
/// The corollary, which cost this project a measurement to learn: quantizing a
/// projection only ever evaluated on a *long* sequence (the DiT, a vocoder) buys much
/// less, because those calls do not reach the matrix-vector kernel. Quantize the decode
/// loop first and measure the rest.
pub enum Proj {
    /// The weight **already transposed** to `[in, out]`, so `forward` is one GEMM with no
    /// per-call transpose. Transposing inside `forward` re-materialises the matrix on
    /// every call, which for the AR loop's head was 14.7 MB per token.
    Dense(Tensor),
    Quant(QMatMul),
}

/// `x @ w_t` with `x` of any leading shape, as a single two-dimensional GEMM.
///
/// `broadcast_matmul` on `[b, t, k] @ [k, n]` expands the weight along the batch axis and
/// takes a *batched* matmul; collapsing the leading dimensions takes one large kernel
/// instead. Measured on `[2, 798, 1024] @ [1024, 1024]`: **1.53x**, and 1.40x across a
/// whole transformer block.
pub fn matmul_2d(x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let k = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let flat = if x.is_contiguous() {
        x.reshape((rows, k))?
    } else {
        x.contiguous()?.reshape((rows, k))?
    };
    let y = flat.matmul(w_t)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(w_t.dim(1)?);
    Ok(y.reshape(shape)?)
}

impl Proj {
    /// `name` is the `[out, in]` weight. `quant` of `None` keeps it dense.
    pub fn load(
        w: &Weights,
        name: &str,
        quant: Option<GgmlDType>,
        device: &Device,
    ) -> Result<Self> {
        match quant {
            None => Ok(Proj::Dense(w.get(name)?.t()?.contiguous()?)),
            Some(q) => {
                // `quantize_onto` needs a CPU f32 source and writes the blocks straight
                // to the device.
                let cpu = w.raw(name)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                Ok(Proj::Quant(QMatMul::from_qtensor(QTensor::quantize_onto(
                    &cpu, q, device,
                )?)?))
            }
        }
    }

    /// Quantize an already-loaded `[out, in]` tensor, or keep it dense.
    pub fn from_tensor(t: &Tensor, quant: Option<GgmlDType>, device: &Device) -> Result<Self> {
        match quant {
            None => Ok(Proj::Dense(t.t()?.contiguous()?)),
            Some(q) => {
                let cpu = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                Ok(Proj::Quant(QMatMul::from_qtensor(QTensor::quantize_onto(
                    &cpu, q, device,
                )?)?))
            }
        }
    }

    /// `[.., in] -> [.., out]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Proj::Dense(w_t) => matmul_2d(x, w_t)?,
            Proj::Quant(q) => q.forward(x)?,
        })
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Proj::Quant(_))
    }
}

/// An affine layer holding its weight already transposed to `[in, out]`.
///
/// CosyVoice's DiT and vocoder are full-sequence, so every projection is a real GEMM
/// and the `[out, in]` -> `[in, out]` transpose is pure overhead if repeated. Doing it
/// once at load costs nothing per call and matters across 440 DiT block passes.
pub struct Linear {
    /// `[in, out]`.
    w_t: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    /// From a torch `[out, in]` weight and optional `[out]` bias.
    pub fn new(weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            w_t: weight.t()?.contiguous()?,
            b: bias,
        })
    }

    pub fn load(w: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        let weight = w.get(&format!("{prefix}.weight"))?;
        let b = if bias {
            Some(w.get(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Self::new(&weight, b)
    }

    /// `[.., in] -> [.., out]`, as a single two-dimensional GEMM.
    ///
    /// The leading dimensions are collapsed rather than passed through
    /// `broadcast_matmul`. That is not cosmetic: `broadcast_matmul` on
    /// `[2, 798, 1024] @ [1024, 1024]` expands the weight along the batch axis and takes
    /// a *batched* matmul, where flattening to `[1596, 1024]` takes one large kernel.
    /// Measured on the CosyVoice DiT's shapes: **1.53x on a single projection and 1.40x
    /// across a whole transformer block.**
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let k = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let flat = if x.is_contiguous() {
            x.reshape((rows, k))?
        } else {
            x.contiguous()?.reshape((rows, k))?
        };
        let y = flat.matmul(&self.w_t)?;
        let y = match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut shape = dims[..dims.len() - 1].to_vec();
        shape.push(self.w_t.dim(1)?);
        Ok(y.reshape(shape)?)
    }

    pub fn out_dim(&self) -> Result<usize> {
        Ok(self.w_t.dim(1)?)
    }
}

// ---------------------------------------------------------------- convolutions

/// Causal 1-D convolution, stride 1: left-pad by `(k - 1) * dilation`, no right pad.
///
/// Audio8's reference computes right padding through `_extra_padding`, which for
/// stride 1 is provably zero — and every causal conv on the *decode* path has stride 1
/// (the strided k2/s2 convs are all encoder-side). This is also the trap that silently
/// length-locked the ONNX export: under tracing `math.ceil` on a shape collapses to a
/// constant. See `docs/rejected/onnx.md`.
pub fn causal_conv1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    dilation: usize,
) -> Result<Tensor> {
    let k = w.dim(2)?;
    let pad = (k - 1) * dilation;
    let y = x.pad_with_zeros(2, pad, 0)?.conv1d(w, 0, 1, dilation, 1)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?)?,
        None => y,
    })
}

/// The tap-major weight [`causal_conv1d_gemm`] needs: `[out, in, k] -> [out, k * in]`.
///
/// Precompute once, after any weight folding. The permutation is what makes the fast path
/// possible — see [`causal_conv1d_gemm`].
pub fn tap_major_weight(w: &Tensor) -> Result<Tensor> {
    let (out, cin, k) = w.dims3()?;
    Ok(w.permute((0, 2, 1))?
        .contiguous()?
        .reshape((out, k * cin))?)
}

/// Causal 1-D convolution as a single GEMM. **Faster than candle's `conv1d` at every shape
/// the Audio8 codec uses**, by 1.34x to 1.73x.
///
/// `w_tap` is `[out, k * in]` from [`tap_major_weight`]; `k` and `dilation` describe the
/// kernel it came from.
///
/// # How this differs from the conv-as-GEMM that was refuted
///
/// `docs/performance/candle-on-metal.md` Finding 1 refuted im2col and attributed the loss to
/// materialisation traffic. Splitting the route in two showed that was the wrong
/// attribution: at `96ch @ 131072` the **GEMM takes 7.1 ms** — 2.4 TFLOP/s, already 8.4x
/// faster than the 59.7 ms direct conv — and **the gather took 82.5 ms**, about 4.9 GB/s on a
/// 120 GB/s bus. Nothing was wrong with the arithmetic; everything was wrong with building
/// the matrix.
///
/// The fix is the *order* of the im2col rows. `stack(dim=1)` interleaves taps within each
/// channel, so every source row scatters across `k` destination rows. `cat(dim=0)` gives each
/// tap one contiguous destination block instead — 3.01x faster to build — and makes the row
/// order tap-major, which costs only a weight permutation at load. Hence `w_tap`.
///
/// Two other routes, both refuted:
///
/// - **`index_select` with a precomputed index** builds the matrix in 8.67 ms (9.5x, ~81 GB/s,
///   essentially bus-limited). But the index is `u32` and as large as the matrix it addresses
///   — 352 MB for this one shape, and the codec has a dozen distinct
///   (channels, length, dilation) combinations. Unaffordable to cache.
/// - **Generating that index on device** instead, from a broadcast sum of three `arange`s,
///   measures **0.46-0.64x**: the arithmetic to produce 352 MB of indices costs more than the
///   faster gather saves.
///
/// The remaining cost is real: this materialises a `[k * in, len]` matrix, 352 MB at the
/// codec's widest stage. That is the price of the 1.73x, and it is why the win grows with
/// length rather than shrinking — the direct kernel degrades faster than the traffic does.
///
/// # The gather is now a kernel
///
/// `cat(dim=0)` was still only ~24 GB/s, and `codecsplit` put 79% of this function in it.
/// [`im2col::im2col_tap_major`] does the same build in one Metal dispatch at 93-131 GB/s —
/// **3.98x to 6.45x**, bit-identical, taking the whole conv 1.51x to 3.15x. It also folds
/// the causal pad in, so `pad_with_zeros` disappears from the fast path. The `cat` form
/// stays as the fallback for batched inputs and non-Metal devices.
///
/// # Chunking along length was tried and reverted
///
/// im2col expands the input by `k`, so CosyVoice's `ups.0` (k=16) builds 658 MB of scratch
/// at utterance length and the two stages behind it exceed 1 GB. Slicing the length to
/// bound that is exact — each slice carries `(k-1)*dilation` samples of left context — and
/// it looked justified while a CosyVoice vocoder regression was unexplained. It was not:
/// the regression was thermal drift, and chunking costs Audio8's codec real time
/// (RTF 0.156 -> 0.204, with the AR stage steady at 0.341 as a control). The unchunked
/// GEMM also beats a direct conv by 2.5x to 5.0x at every `ups` shape *including* the
/// 1.1 GB ones, so the large matrix is not the problem it looks like.
pub fn causal_conv1d_gemm(
    x: &Tensor,
    w_tap: &Tensor,
    b: Option<&Tensor>,
    k: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (batch, cin, len) = x.dims3()?;
    let out = w_tap.dim(0)?;
    // The kernel indexes with a 3-D grid that is fully spent on (len, cin, k), so a batch
    // axis would need a division to unpack — the codec never batches these, so it falls
    // back rather than paying for the general case.
    let cols = if batch == 1 && x.device().is_metal() {
        im2col::im2col_tap_major(&x.contiguous()?, k, dilation)?
    } else {
        let xpad = x.pad_with_zeros(2, (k - 1) * dilation, 0)?;
        let mut taps = Vec::with_capacity(k);
        for t in 0..k {
            taps.push(xpad.narrow(2, t * dilation, len)?);
        }
        // [k, cin, len] -> [k * cin, len], tap-major to match `w_tap`.
        Tensor::cat(&taps, 0)?.reshape((k * cin, len))?
    };
    let y = w_tap.matmul(&cols)?.reshape((batch, out, len))?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, out, 1))?)?,
        None => y,
    })
}

/// Right-causal (lookahead) 1-D convolution: pad `(k - 1) * dilation` on the *right*.
///
/// CosyVoice's `CausalConv1d(causal_type='right')` — its `conv_pre` and the F0
/// predictor's first layer both look forward rather than back, which is what lets the
/// vocoder run with a 4-frame lookahead instead of a full-utterance delay.
pub fn lookahead_conv1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    dilation: usize,
) -> Result<Tensor> {
    let k = w.dim(2)?;
    let pad = (k - 1) * dilation;
    let y = x.pad_with_zeros(2, 0, pad)?.conv1d(w, 0, 1, dilation, 1)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?)?,
        None => y,
    })
}

/// Depthwise (`groups == channels`) convolution as shifted multiply-accumulates.
///
/// Left-padded by `k - 1` (causal). Named for the k=7 case that motivated it; the
/// implementation handles any kernel width, and its advantage grows with how badly
/// candle's grouped path handles the channel count.
pub fn depthwise_k7(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    let k = w.dim(2)?;
    let xpad = x.pad_with_zeros(2, k - 1, 0)?;
    let w = w.reshape((c, k))?;
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        let tap = w.narrow(1, i, 1)?.reshape((1, c, 1))?;
        let term = xpad.narrow(2, i, len)?.broadcast_mul(&tap)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    let y = acc.expect("kernel size > 0");
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, c, 1))?)?,
        None => y,
    })
}

/// Symmetric-padded depthwise convolution as shifted multiply-accumulates.
///
/// The non-causal sibling of [`depthwise_k7`], for torch's `padding=(k-1)/2`.
pub fn depthwise_same(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    let k = w.dim(2)?;
    let pad = (k - 1) / 2;
    let xpad = x.pad_with_zeros(2, pad, k - 1 - pad)?;
    let w = w.reshape((c, k))?;
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        let tap = w.narrow(1, i, 1)?.reshape((1, c, 1))?;
        let term = xpad.narrow(2, i, len)?.broadcast_mul(&tap)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    let y = acc.expect("kernel size > 0");
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, c, 1))?)?,
        None => y,
    })
}

/// Grouped 1-D convolution, causal, by looping the groups.
///
/// For the same reason as [`depthwise_k7`]: candle's `groups > 1` conv1d is far off the
/// hardware on this backend. With `groups` small (the DiT's convolutional position
/// embedding uses 16) the loop is a handful of good dispatches against one bad one.
pub fn grouped_causal_conv1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    groups: usize,
) -> Result<Tensor> {
    let (_, c_in, _) = x.dims3()?;
    let c_out = w.dim(0)?;
    let k = w.dim(2)?;
    let xpad = x.pad_with_zeros(2, k - 1, 0)?;
    let (in_per, out_per) = (c_in / groups, c_out / groups);
    let mut parts = Vec::with_capacity(groups);
    for g in 0..groups {
        let xg = xpad.narrow(1, g * in_per, in_per)?.contiguous()?;
        let wg = w.narrow(0, g * out_per, out_per)?.contiguous()?;
        parts.push(xg.conv1d(&wg, 0, 1, 1, 1)?);
    }
    let y = Tensor::cat(&parts, 1)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, c_out, 1))?)?,
        None => y,
    })
}

/// Causal transposed convolution: convolve, then drop the trailing `k - stride`.
pub fn causal_conv_transpose1d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    let k = w.dim(2)?;
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, 1)?;
    let y = match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?)?,
        None => y,
    };
    let crop = k - stride;
    Ok(if crop > 0 {
        let len = y.dim(2)?;
        y.narrow(2, 0, len - crop)?.contiguous()?
    } else {
        y.contiguous()?
    })
}

/// Nearest-neighbour upsampling along the time axis by an integer factor.
///
/// CosyVoice's `CausalConv1dUpsample` and its `f0_upsamp` both need this, and both want
/// it exact rather than interpolated — `mode='nearest'` in the reference.
pub fn upsample_nearest1d(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 1 {
        return Ok(x.clone());
    }
    let (b, c, len) = x.dims3()?;
    Ok(x.reshape((b, c, len, 1))?
        .broadcast_as((b, c, len, factor))?
        .reshape((b, c, len * factor))?)
}

// ---------------------------------------------------------------- normalisation

/// RMSNorm over the last dimension, computed in f32 as the reference does.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    Ok(candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?)
}

/// LayerNorm over the last dimension, with weight and bias.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let normed = layer_norm_plain(x, eps)?;
    Ok(normed.broadcast_mul(weight)?.broadcast_add(bias)?)
}

/// LayerNorm with no affine parameters (`elementwise_affine=False`), written out of
/// primitives.
///
/// Correct but slow: six passes over the tensor. Prefer [`LayerNormPlain`], which is the
/// same function through candle's fused kernel and measured **5.51x** faster on
/// `[2, 798, 1024]`. This one is kept as the readable reference the fused path is checked
/// against.
pub fn layer_norm_plain(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centred = x.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(centred.broadcast_div(&(var + eps)?.sqrt()?)?)
}

/// Affine-free LayerNorm through candle's fused kernel.
///
/// `elementwise_affine=False` is the same as a unit weight and a zero bias, so the fused
/// path applies — it just needs those two tensors to exist. Holding them here rather than
/// allocating per call matters because the DiT normalises three times per block, 660 times
/// per utterance.
///
/// Measured against [`layer_norm_plain`] on `[2, 798, 1024]`: **5.51x faster**, max
/// absolute difference 9.5e-7.
pub struct LayerNormPlain {
    one: Tensor,
    zero: Tensor,
    eps: f32,
}

impl LayerNormPlain {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            one: Tensor::ones(dim, DType::F32, device)?,
            zero: Tensor::zeros(dim, DType::F32, device)?,
            eps: eps as f32,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(
            x, &self.one, &self.zero, self.eps,
        )?)
    }
}

/// L2-normalise along the last dimension: `F.normalize(x, dim=-1)`.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(x.broadcast_div(&norm)?)
}

// ---------------------------------------------------------------- activations

/// The half-folded snake: `u + sin^2(u)` with `u = alpha * x`.
///
/// The `alpha^-1` factor is *not* applied here — it lives in the following conv's input
/// weights. Callers that have not folded must not use this; see [`snake_full`].
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let u = x.broadcast_mul(alpha)?.contiguous()?;
    Ok((&u + u.sin()?.sqr()?)?)
}

/// `x + alpha_recip * sin^2(alpha x)`, both broadcasts intact.
///
/// The unfolded form, for snakes whose output feeds a skip connection as well as a
/// convolution and so cannot have the reciprocal folded away. `alpha_recip` is passed
/// separately because the reference divides by `alpha + 1e-9`, not by `alpha`.
pub fn snake_full(x: &Tensor, alpha: &Tensor, alpha_recip: &Tensor) -> Result<Tensor> {
    let u = x.broadcast_mul(alpha)?.contiguous()?;
    Ok(x.broadcast_add(&u.sin()?.sqr()?.broadcast_mul(alpha_recip)?)?)
}

/// `x * tanh(softplus(x))` — torch's `nn.Mish`.
pub fn mish(x: &Tensor) -> Result<Tensor> {
    // softplus in the numerically stable form: log1p(exp(-|x|)) + max(x, 0).
    let sp = ((x.abs()?.neg()?.exp()? + 1.0)?.log()? + x.relu()?)?;
    Ok((x * sp.tanh()?)?)
}

/// ELU with `alpha = 1`: `x` if `x > 0` else `exp(x) - 1`.
pub fn elu(x: &Tensor) -> Result<Tensor> {
    Ok(x.elu(1.0)?)
}

/// `max(x, slope * x)`.
pub fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = (x - &pos)?; // min(x, 0)
    Ok((pos + (neg * slope)?)?)
}

/// GELU, tanh approximation — torch's `nn.GELU(approximate="tanh")`.
///
/// Candle's `gelu` is already the tanh form and `gelu_erf` is the exact one. Named
/// explicitly because the DiT's feed-forward asks for `approximate="tanh"` while its
/// ConvNeXt blocks ask for the default (exact) GELU, and mixing them up is a small
/// silent error repeated 22 times.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// Exact GELU (erf) — torch's default `nn.GELU()`.
pub fn gelu_exact(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

/// SwiGLU feed-forward: `w2(silu(w1 x) * w3 x)`. Weights are `[out, in]`.
///
/// Each projection goes through [`matmul_2d`], and each `.t()` is a lazy view that the
/// GEMM consumes — so the three transposes cost nothing, unlike a `broadcast_matmul`
/// against an expanded weight. Callers with a hot path should pre-transpose once and use
/// [`swiglu_t`] instead.
pub fn swiglu(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor) -> Result<Tensor> {
    swiglu_t(x, &w1.t()?, &w3.t()?, &w2.t()?)
}

/// SwiGLU with weights already transposed to `[in, out]`.
pub fn swiglu_t(x: &Tensor, w1_t: &Tensor, w3_t: &Tensor, w2_t: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&matmul_2d(x, w1_t)?)?;
    let up = matmul_2d(x, w3_t)?;
    matmul_2d(&(gate * up)?, w2_t)
}

// ---------------------------------------------------------------- rope & masks

/// Interleaved-pair RoPE tables, `[len, head_dim / 2]`, rounded through bf16.
///
/// Returns `(cos, sin)` in f32, ready for [`candle_nn::rotary_emb::rope_i`], whose
/// adjacent-pair convention matches Audio8's `torch.polar` layout. Using the half-split
/// `rope` instead is the single easiest way to get a port that runs, sounds plausible,
/// and is wrong.
pub fn rope_table(
    len: usize,
    head_dim: usize,
    base: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = rope_pairs(len, head_dim, base);
    let half = head_dim / 2;
    let to_bf16_f32 = |v: Vec<f32>| -> Result<Tensor> {
        let t = Tensor::from_vec(v, (len, half), device)?;
        Ok(t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)
    };
    Ok((to_bf16_f32(cos)?, to_bf16_f32(sin)?))
}

/// Interleaved-pair RoPE tables in plain f32, no bf16 rounding.
///
/// For models that build the table in f32 and keep it there — CosyVoice's DiT via
/// `x_transformers.RotaryEmbedding`, which explicitly disables autocast so the table is
/// never rounded.
pub fn rope_table_f32(
    len: usize,
    head_dim: usize,
    base: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = rope_pairs(len, head_dim, base);
    let half = head_dim / 2;
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

fn rope_pairs(len: usize, head_dim: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; len * half];
    let mut sin = vec![0f32; len * half];
    for t in 0..len {
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f64 / head_dim as f64);
            let phase = t as f64 * freq;
            cos[t * half + i] = phase.cos() as f32;
            sin[t * half + i] = phase.sin() as f32;
        }
    }
    (cos, sin)
}

/// Additive attention bias, `[n, n]`: 0 where a query may attend, -inf elsewhere.
///
/// `window` bounds how far back a query can look (`None` = unbounded causal).
pub fn causal_window_mask(n: usize, window: Option<usize>, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * n];
    for row in 0..n {
        let lo = match window {
            Some(w) => (row + 1).saturating_sub(w),
            None => 0,
        };
        for col in 0..n {
            if col > row || col < lo {
                v[row * n + col] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(v, (n, n), device)?)
}

// ---------------------------------------------------------------- comparison

/// Max absolute difference, for fixture comparisons.
pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    Ok((a - b)?.abs()?.max(0)?.to_scalar::<f32>()?)
}

/// Max absolute difference, and that difference relative to `b`'s peak magnitude.
///
/// A fixture report needs both: an absolute error is only interpretable next to the
/// scale it sits on, and a relative one hides how big the tensor was.
pub fn abs_and_rel(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let abs = max_abs_diff(a, b)?;
    let scale = b
        .to_dtype(DType::F32)?
        .flatten_all()?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    Ok((abs, if scale > 0.0 { abs / scale } else { 0.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_folds_to_the_stored_direction_when_the_gain_is_its_norm() -> Result<()> {
        let d = Device::Cpu;
        // Row 0 of `v` is [3, 4] (norm 5), row 1 is [0, 2] (norm 2).
        let v = Tensor::from_vec(vec![3f32, 4.0, 0.0, 2.0], (2, 1, 2), &d)?;
        let g = Tensor::from_vec(vec![5f32, 2.0], (2, 1, 1), &d)?;
        // g == ||v|| per row, so folding is the identity.
        assert!(max_abs_diff(&fold_weight_norm(&g, &v)?, &v)? < 1e-6);
        Ok(())
    }

    #[test]
    fn upsample_nearest_repeats_each_frame() -> Result<()> {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3), &d)?;
        let y = upsample_nearest1d(&x, 2)?;
        assert_eq!(y.dims(), &[1, 1, 6]);
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn mish_matches_its_definition() -> Result<()> {
        let d = Device::Cpu;
        let xs = [-3f32, -0.5, 0.0, 0.5, 3.0];
        let got = mish(&Tensor::from_vec(xs.to_vec(), xs.len(), &d)?)?.to_vec1::<f32>()?;
        for (i, v) in xs.iter().enumerate() {
            let want = v * (1.0f32 + v.exp()).ln().tanh();
            assert!((got[i] - want).abs() < 1e-5, "at {v}: {} vs {want}", got[i]);
        }
        Ok(())
    }

    #[test]
    fn leaky_relu_bends_only_the_negative_half() -> Result<()> {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![-2f32, -1.0, 0.0, 1.0, 2.0], 5, &d)?;
        assert_eq!(
            leaky_relu(&x, 0.1)?.to_vec1::<f32>()?,
            vec![-0.2, -0.1, 0.0, 1.0, 2.0]
        );
        Ok(())
    }

    #[test]
    fn grouped_conv_agrees_with_the_dense_equivalent() -> Result<()> {
        let d = Device::Cpu;
        // A grouped conv equals a dense one whose cross-group blocks are zero.
        let x = Tensor::rand(0f32, 1.0, (1, 4, 6), &d)?;
        let wg = Tensor::rand(0f32, 1.0, (4, 2, 3), &d)?;
        let got = grouped_causal_conv1d(&x, &wg, None, 2)?;
        let zero = Tensor::zeros((2, 2, 3), DType::F32, &d)?;
        let top = Tensor::cat(&[wg.narrow(0, 0, 2)?, zero.clone()], 1)?;
        let bot = Tensor::cat(&[zero, wg.narrow(0, 2, 2)?], 1)?;
        let want = causal_conv1d(&x, &Tensor::cat(&[top, bot], 0)?, None, 1)?;
        assert!(max_abs_diff(&got, &want)? < 1e-5);
        Ok(())
    }

    #[test]
    fn lookahead_conv_sees_forward_and_causal_conv_does_not() -> Result<()> {
        let d = Device::Cpu;
        // A k=2 conv keeping only the second tap: causal shifts right by one,
        // lookahead reads the next sample.
        let w = Tensor::from_vec(vec![0f32, 1.0], (1, 1, 2), &d)?;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3), &d)?;
        assert_eq!(
            causal_conv1d(&x, &w, None, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            lookahead_conv1d(&x, &w, None, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![2.0, 3.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn rope_tables_differ_only_by_the_bf16_rounding() -> Result<()> {
        let d = Device::Cpu;
        let (c16, _) = rope_table(8, 64, 10_000.0, &d)?;
        let (c32, _) = rope_table_f32(8, 64, 10_000.0, &d)?;
        let diff = max_abs_diff(&c16, &c32)?;
        // Close, but not equal — which is exactly why the two exist separately.
        assert!(diff > 0.0, "bf16 rounding should be visible");
        assert!(diff < 1e-2, "but it should still be a rounding: {diff}");
        Ok(())
    }

    #[test]
    fn l2_normalize_gives_unit_rows() -> Result<()> {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4.0, 0.0, 5.0], (2, 2), &d)?;
        let n = l2_normalize(&x)?.sqr()?.sum(D::Minus1)?.to_vec1::<f32>()?;
        for v in n {
            assert!((v - 1.0).abs() < 1e-6);
        }
        Ok(())
    }
}
