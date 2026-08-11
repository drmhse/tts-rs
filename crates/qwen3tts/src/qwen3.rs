//! A Qwen3 decoder stack, shared by all three transformers in this engine.
//!
//! The talker, the depth predictor and the codec's pre-transformer have the same layer
//! shape and differ only in flags: QK-norm (talker, predictor), per-residual `LayerScale`
//! (codec), and a sliding window (codec only). One implementation with those as [`Geometry`]
//! fields beats three near-copies.
//!
//! RoPE is plain half-split at the configured theta — see trap 1, the config's M-RoPE is
//! degenerate. Attention has no biases anywhere.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use tts_nn::{rms_norm, rope_table_f32, Proj, Weight, Weights};

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub rope_base: f64,
    pub qk_norm: bool,
    pub layer_scale: bool,
    pub window: Option<usize>,
}

impl Geometry {
    pub fn gqa(&self) -> usize {
        self.heads / self.n_kv
    }
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// Fused at load: one `[q_width + 2 * kv_width, dim]` matmul per step instead of three.
    wqkv: Proj,
    wo: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
    /// `[head_dim]` each, applied per-head before RoPE.
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    attn_scale: Option<Tensor>,
    mlp_scale: Option<Tensor>,
}

impl Layer {
    fn load(
        w: &Weights,
        prefix: &str,
        geo: &Geometry,
        how: Weight,
        device: &Device,
    ) -> Result<Self> {
        let a = format!("{prefix}.self_attn");
        let wqkv = Tensor::cat(
            &[
                w.get(&format!("{a}.q_proj.weight"))?,
                w.get(&format!("{a}.k_proj.weight"))?,
                w.get(&format!("{a}.v_proj.weight"))?,
            ],
            0,
        )?
        .contiguous()?;
        Ok(Self {
            attn_norm: w.get(&format!("{prefix}.input_layernorm.weight"))?,
            ffn_norm: w.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
            wqkv: Proj::from_tensor_as(&wqkv, how, device)?,
            wo: Proj::load_as(w, &format!("{a}.o_proj.weight"), how, device)?,
            gate: Proj::load_as(w, &format!("{prefix}.mlp.gate_proj.weight"), how, device)?,
            up: Proj::load_as(w, &format!("{prefix}.mlp.up_proj.weight"), how, device)?,
            down: Proj::load_as(w, &format!("{prefix}.mlp.down_proj.weight"), how, device)?,
            // `1/sqrt(head_dim)` folded in. QK-norm rescales q to unit RMS and then multiplies
            // by this weight, and RoPE is a rotation, so scaling the weight is exactly scaling
            // the scores — one fewer dispatch per layer per step, which is what a batch-1
            // decode is actually short of. Layers without QK-norm still scale explicitly.
            q_norm: if geo.qk_norm {
                let s = (geo.head_dim as f64).sqrt().recip();
                Some((w.get(&format!("{a}.q_norm.weight"))? * s)?)
            } else {
                None
            },
            k_norm: if geo.qk_norm {
                Some(w.get(&format!("{a}.k_norm.weight"))?)
            } else {
                None
            },
            attn_scale: if geo.layer_scale {
                Some(w.get(&format!("{prefix}.self_attn_layer_scale.scale"))?)
            } else {
                None
            },
            mlp_scale: if geo.layer_scale {
                Some(w.get(&format!("{prefix}.mlp_layer_scale.scale"))?)
            } else {
                None
            },
        })
    }
}

struct Cache {
    k: Tensor,
    v: Tensor,
}

/// Decode state: preallocated K/V plus how many positions are written.
///
/// Capacity lives here rather than on the [`Stack`] because it is the one dimension that has
/// to shrink as the batch grows: a position costs 229 KB across 28 layers of k and v, so the
/// talker's 1536 is 352 MB for one lane and 2.8 GB for eight.
pub struct State {
    caches: Vec<Cache>,
    pub width: usize,
    pub batch: usize,
    capacity: usize,
}

pub struct Stack {
    layers: Vec<Layer>,
    /// KV cache dtype. f16 with f16 weights: 114 KB per position per lane instead of 229, which
    /// is what caps how many lanes a batch can hold. Decode reads it through `tts_nn::attn`,
    /// which takes either; prefill casts, being once per segment.
    kv: DType,
    norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    pub geo: Geometry,
    capacity: usize,
    device: Device,
}

impl Stack {
    pub fn load(
        w: &Weights,
        prefix: &str,
        geo: Geometry,
        how: Weight,
        capacity: usize,
        device: &Device,
    ) -> Result<Self> {
        let layers = (0..geo.layers)
            .map(|i| Layer::load(w, &format!("{prefix}layers.{i}"), &geo, how, device))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = rope_table_f32(capacity, geo.head_dim, geo.rope_base, device)?;
        Ok(Self {
            layers,
            kv: if how == Weight::F16 {
                DType::F16
            } else {
                DType::F32
            },
            norm: w.get(&format!("{prefix}norm.weight"))?,
            cos,
            sin,
            geo,
            capacity,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_state(&self, batch: usize) -> Result<State> {
        self.new_state_with(batch, self.capacity)
    }

    /// A state holding only `capacity` positions. Clamped to the RoPE tables, which are built
    /// for `self.capacity` and are the real ceiling.
    pub fn new_state_with(&self, batch: usize, capacity: usize) -> Result<State> {
        let capacity = capacity.clamp(1, self.capacity);
        let shape = (batch, self.geo.n_kv, capacity, self.geo.head_dim);
        let caches = (0..self.geo.layers)
            .map(|_| {
                Ok(Cache {
                    k: Tensor::zeros(shape, self.kv, &self.device)?,
                    v: Tensor::zeros(shape, self.kv, &self.device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(State {
            caches,
            width: 0,
            batch,
            capacity,
        })
    }

    /// The largest state [`Self::new_state_with`] will build — the RoPE table length.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Run `t` positions, appending to the cache. Returns the normed hidden states,
    /// `[b, t, dim]` — all of them, since the codec needs the whole sequence and the
    /// decode loops take the last.
    pub fn forward(&self, x: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b, t, dim) = x.dims3()?;
        if b != state.batch {
            bail!("batch {b} != state batch {}", state.batch);
        }
        if dim != self.geo.dim {
            bail!("input width {dim} != {}", self.geo.dim);
        }
        let start = state.width;
        if start + t > state.capacity {
            bail!(
                "{} positions exceeds cache capacity {}",
                start + t,
                state.capacity
            );
        }
        let g = &self.geo;
        // Folded into `q_norm` wherever QK-norm exists; see `Layer::load`.
        let scale = if g.qk_norm {
            None
        } else {
            Some(1.0 / (g.head_dim as f64).sqrt())
        };
        // `[b, t, n, head_dim] -> [b, n, t, head_dim]`. At `t == 1` those are the same bytes
        // in the same order, so the transpose is a free reshape rather than a copy — three
        // fewer whole-tensor copies per layer on every decode step.
        let heads_first = |x: &Tensor, n: usize| -> Result<Tensor> {
            if t == 1 {
                Ok(x.reshape((b, n, 1, g.head_dim))?)
            } else {
                Ok(x.transpose(1, 2)?.contiguous()?)
            }
        };
        let mut h = x.clone();

        for (li, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&h, &layer.attn_norm, g.eps)?;
            let qkv = layer.wqkv.forward(&normed)?;
            let q = qkv.narrow(candle_core::D::Minus1, 0, g.q_width())?;
            let kk = qkv.narrow(candle_core::D::Minus1, g.q_width(), g.kv_width())?;
            let v = qkv.narrow(
                candle_core::D::Minus1,
                g.q_width() + g.kv_width(),
                g.kv_width(),
            )?;

            // QK-norm is over the head dim, so it must happen on the [.., heads, head_dim]
            // view and before the transpose — normalising the flat projection is a
            // different, still-running model.
            let q = q.reshape((b, t, g.heads, g.head_dim))?;
            let kk = kk.reshape((b, t, g.n_kv, g.head_dim))?;
            let q = match &layer.q_norm {
                Some(n) => rms_norm(&q, n, g.eps)?,
                None => q,
            };
            let kk = match &layer.k_norm {
                Some(n) => rms_norm(&kk, n, g.eps)?,
                None => kk,
            };
            let q = heads_first(&q, g.heads)?;
            let kk = heads_first(&kk, g.n_kv)?;
            let v = heads_first(&v.reshape((b, t, g.n_kv, g.head_dim))?, g.n_kv)?;

            let cos = self.cos.narrow(0, start, t)?;
            let sin = self.sin.narrow(0, start, t)?;
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let kk = candle_nn::rotary_emb::rope(&kk, &cos, &sin)?;

            let cache = &mut state.caches[li];
            // In place. `slice_assign` reallocates the whole cache per token and cost 2.0x
            // on CosyVoice's LLM stage.
            cache.k.slice_set(&kk.to_dtype(self.kv)?, 2, start)?;
            cache.v.slice_set(&v.to_dtype(self.kv)?, 2, start)?;

            let span = start + t;
            // Only the prefill branch uses these, and it goes through candle's matmul, so it
            // wants f32. One cast per segment, not per step.
            let k_all = cache.k.narrow(2, 0, span)?;
            let v_all = cache.v.narrow(2, 0, span)?;

            let attn =
                if t == 1 {
                    // Reads the cache in place; candle's route copies the span twice per layer.
                    // Folded into `q_norm` for the talker and predictor; applied here for any
                    // layer without QK-norm, since the kernel takes no scale.
                    let qs = match scale {
                        Some(s) => (q.clone() * s)?,
                        None => q.clone(),
                    };
                    let qg = qs.reshape((b, g.n_kv, g.gqa(), g.head_dim))?;
                    let wstart = match g.window {
                        Some(w) if span > w => span - w,
                        _ => 0,
                    };
                    tts_nn::attn::decode_attention(&qg, &cache.k, &cache.v, span, wstart)?
                        .reshape((b, 1, g.q_width()))?
                } else {
                    // Prefill goes through candle's matmul, which needs matching dtypes. One
                    // cast per segment, not per step.
                    let k_rep = self.repeat_kv(&k_all.to_dtype(DType::F32)?)?;
                    let v_rep = self.repeat_kv(&v_all.to_dtype(DType::F32)?)?;
                    let scores = q.matmul(&k_rep.transpose(2, 3)?.contiguous()?)?;
                    let scores = match scale {
                        Some(s) => (scores * s)?,
                        None => scores,
                    };
                    // Rows are positions `start..start+t`, columns `0..span`, so the mask is the
                    // bottom-right block of a `[span, span]` causal-window mask.
                    let full = tts_nn::causal_window_mask(span, g.window, &self.device)?;
                    let block = full.narrow(0, start, t)?;
                    let scores = scores.broadcast_add(&block)?;
                    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
                    probs
                        .matmul(&v_rep.contiguous()?)?
                        .transpose(1, 2)?
                        .reshape((b, t, g.q_width()))?
                        .contiguous()?
                };

            let attn = layer.wo.forward(&attn)?;
            let attn = match &layer.attn_scale {
                Some(s) => attn.broadcast_mul(s)?,
                None => attn,
            };
            h = (h + attn)?;

            let normed = rms_norm(&h, &layer.ffn_norm, g.eps)?;
            let tail = tts_nn::fused::swiglu_mul(
                &layer.gate.forward(&normed)?,
                &layer.up.forward(&normed)?,
            )?;
            let mlp = layer.down.forward(&tail)?;
            let mlp = match &layer.mlp_scale {
                Some(s) => mlp.broadcast_mul(s)?,
                None => mlp,
            };
            h = (h + mlp)?;
        }
        state.width = start + t;
        rms_norm(&h, &self.norm, g.eps)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let g = &self.geo;
        if g.gqa() == 1 {
            return Ok(x.contiguous()?);
        }
        let (b, n_kv, span, hd) = x.dims4()?;
        Ok(x.unsqueeze(2)?
            .broadcast_as((b, n_kv, g.gqa(), span, hd))?
            .reshape((b, g.heads, span, hd))?
            .contiguous()?)
    }
}
