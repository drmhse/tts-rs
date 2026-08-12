//! Fused elementwise activations.
//!
//! `docs/reference.md#performance` Finding 2 measured snake at 11.5 ms for `[1, 96, 131072]` and
//! showed its five constituent ops sum to 13.7 ms — they match, which is the proof that
//! candle fuses nothing. A single pass costs what `affine` costs, 1.33 ms.
//!
//! Alpha folding (Finding 3) already removed both broadcasts from most call sites, taking
//! snake from five ops to three. This removes the remaining two round-trips.
//!
//! Both entry points fall back to the composed candle form off Metal, and the unit tests
//! check the kernels against exactly that form.

#[cfg(feature = "metal")]
use crate::mtl;
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

/// `x + sin^2(x)` in one pass — the folded snake, where `alpha` already lives in the
/// preceding conv's output weights.
struct SnakeFolded;

impl CustomOp1 for SnakeFolded {
    fn name(&self) -> &'static str {
        "snake_folded"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("snake_folded: only f32"),
        };
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        if !layout.is_contiguous() {
            candle_core::bail!("snake_folded: input must be contiguous");
        }
        let dst = src[start..start + n]
            .iter()
            .map(|&x| x + x.sin().powi(2))
            .collect();
        Ok((CpuStorage::F32(dst), layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !layout.is_contiguous() {
            candle_core::bail!("snake_folded: input must be contiguous");
        }
        if storage.dtype() != DType::F32 {
            candle_core::bail!("snake_folded: only f32, got {:?}", storage.dtype());
        }
        let n = layout.shape().elem_count();
        let device = storage.device();
        let p = mtl::pipeline(device, "snake_folded_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_folded")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_folded");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(storage.buffer()), layout.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(n as u32));
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, n);
        encoder.dispatch_threads(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            layout.shape().clone(),
        ))
    }
}

/// `u + sin^2(u)` with `u = alpha[c] * x`, for `[1, C, L]` inputs.
struct SnakeAlpha {
    channels: usize,
    len: usize,
}

impl candle_core::CustomOp2 for SnakeAlpha {
    fn name(&self) -> &'static str {
        "snake_alpha"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, a) = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(a)) => (x, a),
            _ => candle_core::bail!("snake_alpha: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("snake_alpha: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let mut dst = vec![0f32; self.channels * self.len];
        for c in 0..self.channels {
            let alpha = a[o2 + c];
            for l in 0..self.len {
                let u = alpha * x[o1 + c * self.len + l];
                dst[c * self.len + l] = u + u.sin().powi(2);
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.channels, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("snake_alpha: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("snake_alpha: only f32");
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "snake_alpha_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_alpha")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_alpha");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.len as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// `x + beta_recip[c] * sin^2(alpha[c] * x)` for `[1, C, L]` inputs.
struct SnakeBeta {
    channels: usize,
    len: usize,
}

impl candle_core::CustomOp3 for SnakeBeta {
    fn name(&self) -> &'static str {
        "snake_beta"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, a, br) = match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(a), CpuStorage::F32(br)) => (x, a, br),
            _ => candle_core::bail!("snake_beta: only f32"),
        };
        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("snake_beta: inputs must be contiguous");
            }
        }
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let mut dst = vec![0f32; self.channels * self.len];
        for c in 0..self.channels {
            let (alpha, brecip) = (a[o2 + c], br[o3 + c]);
            for l in 0..self.len {
                let x = x[o1 + c * self.len + l];
                dst[c * self.len + l] = x + brecip * (alpha * x).sin().powi(2);
            }
        }
        Ok((CpuStorage::F32(dst), (1, self.channels, self.len).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("snake_beta: inputs must be contiguous");
            }
        }
        for s in [s1, s2, s3] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("snake_beta: only f32");
            }
        }
        let n = self.channels * self.len;
        let device = s1.device();
        let p = mtl::pipeline(device, "snake_beta_f32")?;
        let dst = device.new_buffer(n, DType::F32, "snake_beta")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_beta");
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2), (s3, l3)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.len as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.len,
                height: self.channels,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (1, self.channels, self.len).into(),
        ))
    }
}

/// `silu(gate) * up`, elementwise over identically shaped inputs.
struct SwigluMul {
    n: usize,
}

impl candle_core::CustomOp2 for SwigluMul {
    fn name(&self) -> &'static str {
        "swiglu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (g, u) = match (s1, s2) {
            (CpuStorage::F32(g), CpuStorage::F32(u)) => (g, u),
            _ => candle_core::bail!("swiglu_mul: only f32"),
        };
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("swiglu_mul: inputs must be contiguous");
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let dst = (0..self.n)
            .map(|i| {
                let x = g[o1 + i];
                (x / (1.0 + (-x).exp())) * u[o2 + i]
            })
            .collect();
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("swiglu_mul: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("swiglu_mul: only f32");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "swiglu_mul_f32")?;
        let dst = device.new_buffer(self.n, DType::F32, "swiglu_mul")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::swiglu_mul");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.n as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.n);
        encoder.dispatch_threads(
            MTLSize {
                width: self.n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), self.n, DType::F32),
            l1.shape().clone(),
        ))
    }
}

/// `silu(gate) * up` in one pass — the tail of a SwiGLU feed-forward.
pub fn swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_metal() && gate.shape() == up.shape() {
        let op = SwigluMul {
            n: gate.elem_count(),
        };
        return gate.contiguous()?.apply_op2_no_bwd(&up.contiguous()?, &op);
    }
    candle_nn::ops::silu(gate)? * up
}

/// `[b, n, h, d] -> [b, h, n, d]` as one coalesced pass.
struct HeadTranspose {
    n: usize,
    heads: usize,
    dim: usize,
}

impl CustomOp1 for HeadTranspose {
    fn name(&self) -> &'static str {
        "head_transpose"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("head_transpose: only f32"),
        };
        if !layout.is_contiguous() {
            candle_core::bail!("head_transpose: input must be contiguous");
        }
        let (b, n, hd, dim) = (layout.shape().dims()[0], self.n, self.heads, self.dim);
        let o = layout.start_offset();
        let mut dst = vec![0f32; b * hd * n * dim];
        for bi in 0..b {
            for h in 0..hd {
                for p in 0..n {
                    let s = o + ((bi * n + p) * hd + h) * dim;
                    let t = ((bi * hd + h) * n + p) * dim;
                    dst[t..t + dim].copy_from_slice(&src[s..s + dim]);
                }
            }
        }
        Ok((CpuStorage::F32(dst), (b, hd, n, dim).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !layout.is_contiguous() {
            candle_core::bail!("head_transpose: input must be contiguous");
        }
        if storage.dtype() != DType::F32 {
            candle_core::bail!("head_transpose: only f32");
        }
        let b = layout.shape().dims()[0];
        let n = self.n * b * self.heads * self.dim;
        let device = storage.device();
        let p = mtl::pipeline(device, "head_transpose_f32")?;
        let dst = device.new_buffer(n, DType::F32, "head_transpose")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::head_transpose");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(storage.buffer()), layout.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(self.n as u32));
        encoder.set_bytes(3, &(self.heads as u32));
        encoder.set_bytes(4, &(self.dim as u32));
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        // `dim` is 64 here, so pair it with several positions to fill a threadgroup.
        let w = mtl::group_width(&p, self.dim);
        encoder.dispatch_threads(
            MTLSize {
                width: self.dim,
                height: self.n,
                depth: b * self.heads,
            },
            MTLSize {
                width: w,
                height: (256 / w).max(1),
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, DType::F32),
            (b, self.heads, self.n, self.dim).into(),
        ))
    }
}

/// `[b, n, h*d] -> [b, h, n, d]`, the reshape-and-transpose multi-head attention needs.
///
/// **Measured 4.0x-7.4x faster than `transpose(1,2).contiguous()`** (63 GB/s against 8.5 at
/// `[2, 3192, 1024]`) and bit-identical — but it does *not* speed up attention, and
/// `flow.rs` does not use it. `sdpa` accepts strides, so the DiT feeds it lazy transposed
/// views and never materialises them; making them contiguous first, even this cheaply,
/// measured **0.98x** at the engine's real sequence length. The strided-view decision in
/// `DiTBlock::attention` was checked against a fast transpose and survives.
///
/// Kept because the negative result is worth being able to re-run (`tts-probe --bin
/// attnlayout`), and because any future path that needs a genuinely contiguous head layout
/// should not pay candle's 8.5 GB/s for it.
pub fn head_transpose(x: &Tensor, heads: usize, dim: usize) -> Result<Tensor> {
    let (b, n, hd) = x.dims3()?;
    if hd != heads * dim {
        candle_core::bail!("head_transpose: {hd} != {heads} * {dim}");
    }
    if x.device().is_metal() {
        let op = HeadTranspose { n, heads, dim };
        return x.contiguous()?.apply_op1_no_bwd(&op);
    }
    x.reshape((b, n, heads, dim))?.transpose(1, 2)?.contiguous()
}

/// `x + sin^2(x)`, one pass on Metal.
pub fn snake_folded(x: &Tensor) -> Result<Tensor> {
    if x.device().is_metal() {
        x.contiguous()?.apply_op1_no_bwd(&SnakeFolded)
    } else {
        x + x.sin()?.sqr()?
    }
}

/// `u + sin^2(u)` with `u = alpha * x` broadcast over `[1, C, L]`'s channel axis.
///
/// `alpha` may be `[C]`, `[1, C, 1]`, or any shape with `C` elements.
pub fn snake_alpha(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b == 1 && x.device().is_metal() && alpha.elem_count() == c {
        let op = SnakeAlpha { channels: c, len };
        return x
            .contiguous()?
            .apply_op2_no_bwd(&alpha.flatten_all()?.contiguous()?, &op);
    }
    let u = x.broadcast_mul(&alpha.reshape((1, c, 1))?)?.contiguous()?;
    &u + u.sin()?.sqr()?
}

/// [`snake_beta`] for channels-last `[b, L, C]`.
pub fn snake_beta_nlc(x: &Tensor, alpha: &Tensor, beta_recip: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let c = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    // Not gated on Metal, unlike the channel-major sibling: `cpu_fwd` handles f16 activations
    // against f32 parameters and the composed fallback cannot, since candle has no mixed-dtype
    // multiply. The gate runs its numerics on CPU, so the fallback is a real path, not a
    // formality.
    if alpha.elem_count() == c && beta_recip.elem_count() == c {
        let op = SnakeBetaNlc {
            rows,
            chan: c,
            half: x.dtype() == candle_core::DType::F16,
        };
        return x
            .contiguous()?
            .apply_op3_no_bwd(
                &alpha.flatten_all()?.contiguous()?,
                &beta_recip.flatten_all()?.contiguous()?,
                &op,
            )?
            .reshape(dims);
    }
    let u = x.broadcast_mul(alpha)?.contiguous()?;
    x.broadcast_add(&u.sin()?.sqr()?.broadcast_mul(beta_recip)?)
}

struct SnakeBetaNlc {
    rows: usize,
    chan: usize,
    half: bool,
}

impl candle_core::CustomOp3 for SnakeBetaNlc {
    fn name(&self) -> &'static str {
        "snake_beta_nlc"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let xf: Vec<f32>;
        let (x, a, br) = match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(a), CpuStorage::F32(br)) => (&x[..], a, br),
            (CpuStorage::F16(x), CpuStorage::F32(a), CpuStorage::F32(br)) => {
                xf = x.iter().map(|v| v.to_f32()).collect();
                (&xf[..], a, br)
            }
            _ => candle_core::bail!("snake_beta_nlc: activations f32 or f16, params f32"),
        };
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let n = self.rows * self.chan;
        let mut dst = vec![0f32; n];
        for r in 0..self.rows {
            for c in 0..self.chan {
                let i = r * self.chan + c;
                let v = x[o1 + i];
                dst[i] = v + br[o3 + c] * (a[o2 + c] * v).sin().powi(2);
            }
        }
        let shape: candle_core::Shape = (self.rows, self.chan).into();
        if self.half {
            let h = dst.into_iter().map(half::f16::from_f32).collect();
            Ok((CpuStorage::F16(h), shape))
        } else {
            Ok((CpuStorage::F32(dst), shape))
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if s2.dtype() != DType::F32 || s3.dtype() != DType::F32 {
            candle_core::bail!("snake_beta_nlc: parameters must be f32");
        }
        let (kernel, dt, esz) = match s1.dtype() {
            DType::F32 => ("snake_beta_nlc_f32", DType::F32, 4),
            DType::F16 => ("snake_beta_nlc_f16", DType::F16, 2),
            d => candle_core::bail!("snake_beta_nlc: activations f32 or f16, got {d:?}"),
        };
        let n = self.rows * self.chan;
        let device = s1.device();
        let p = mtl::pipeline(device, kernel)?;
        let dst = device.new_buffer(n, dt, "snake_beta_nlc")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::snake_beta_nlc");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * esz);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * 4);
        for s in [s1, s2, s3] {
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.chan as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: self.chan,
                height: self.rows,
                depth: 1,
            },
            MTLSize {
                width: mtl::group_width(&p, self.chan),
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), n, dt),
            (self.rows, self.chan).into(),
        ))
    }
}

/// `x + beta_recip * sin^2(alpha * x)`, both parameters per-channel over `[1, C, L]`.
///
/// The unfoldable snake — see [`crate::snake_full`], which is the composed fallback and
/// what the tests compare against.
pub fn snake_beta(x: &Tensor, alpha: &Tensor, beta_recip: &Tensor) -> Result<Tensor> {
    let (b, c, len) = x.dims3()?;
    if b == 1 && x.device().is_metal() && alpha.elem_count() == c && beta_recip.elem_count() == c {
        let op = SnakeBeta { channels: c, len };
        return x.contiguous()?.apply_op3_no_bwd(
            &alpha.flatten_all()?.contiguous()?,
            &beta_recip.flatten_all()?.contiguous()?,
            &op,
        );
    }
    let u = x.broadcast_mul(&alpha.reshape((1, c, 1))?)?.contiguous()?;
    x.broadcast_add(
        &u.sin()?
            .sqr()?
            .broadcast_mul(&beta_recip.reshape((1, c, 1))?)?,
    )
}

// ------------------------------------------------ DiT block elementwise

/// Shared shape for the two `[b, n, d]`-against-`[b, 1, d]` kernels.
struct Bcast3 {
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    kernel: &'static str,
    label: &'static str,
    n: usize,
    dim: usize,
    /// `true` for `x * (1 + v1) + v2`, `false` for `x + y * v`.
    affine: bool,
}

impl candle_core::CustomOp3 for Bcast3 {
    fn name(&self) -> &'static str {
        "bcast3"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (a, b, c) = match (s1, s2, s3) {
            (CpuStorage::F32(a), CpuStorage::F32(b), CpuStorage::F32(c)) => (a, b, c),
            _ => candle_core::bail!("{}: only f32", self.label),
        };
        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("{}: inputs must be contiguous", self.label);
            }
        }
        let batch = l1.shape().dims()[0];
        let (o1, o2, o3) = (l1.start_offset(), l2.start_offset(), l3.start_offset());
        let mut dst = vec![0f32; batch * self.n * self.dim];
        for bi in 0..batch {
            for p in 0..self.n {
                for d in 0..self.dim {
                    let i = (bi * self.n + p) * self.dim + d;
                    let j = bi * self.dim + d;
                    dst[i] = if self.affine {
                        a[o1 + i] * (1.0 + b[o2 + j]) + c[o3 + j]
                    } else {
                        a[o1 + i] + b[o2 + i] * c[o3 + j]
                    };
                }
            }
        }
        Ok((CpuStorage::F32(dst), (batch, self.n, self.dim).into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2, l3] {
            if !l.is_contiguous() {
                candle_core::bail!("{}: inputs must be contiguous", self.label);
            }
        }
        for s in [s1, s2, s3] {
            if s.dtype() != DType::F32 {
                candle_core::bail!("{}: only f32", self.label);
            }
        }
        let batch = l1.shape().dims()[0];
        let count = batch * self.n * self.dim;
        let device = s1.device();
        let p = mtl::pipeline(device, self.kernel)?;
        let dst = device.new_buffer(count, DType::F32, self.label)?;

        let encoder = device.command_encoder()?;
        encoder.set_label(self.label);
        encoder.set_compute_pipeline_state(&p);
        for (i, (s, l)) in [(s1, l1), (s2, l2), (s3, l3)].iter().enumerate() {
            encoder.set_buffer(i, Some(s.buffer()), l.start_offset() * 4);
            encoder.use_resource(s.buffer(), MTLResourceUsage::Read);
        }
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.n as u32));
        encoder.set_bytes(5, &(self.dim as u32));
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.dim);
        encoder.dispatch_threads(
            MTLSize {
                width: self.dim,
                height: self.n,
                depth: batch,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (batch, self.n, self.dim).into(),
        ))
    }
}

/// `x * (1 + scale) + shift` with `scale`/`shift` broadcast over the sequence axis.
pub fn modulate_affine(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let (_, n, dim) = x.dims3()?;
    if x.device().is_metal() {
        let op = Bcast3 {
            kernel: "modulate_affine_f32",
            label: "tts_nn::modulate_affine",
            n,
            dim,
            affine: true,
        };
        return x
            .contiguous()?
            .apply_op3_no_bwd(&scale.contiguous()?, &shift.contiguous()?, &op);
    }
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `residual + y * gate`, with `gate` broadcast over the sequence axis.
pub fn gate_residual(residual: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let (_, n, dim) = residual.dims3()?;
    if residual.device().is_metal() {
        let op = Bcast3 {
            kernel: "gate_residual_f32",
            label: "tts_nn::gate_residual",
            n,
            dim,
            affine: false,
        };
        return residual
            .contiguous()?
            .apply_op3_no_bwd(&y.contiguous()?, &gate.contiguous()?, &op);
    }
    residual + y.broadcast_mul(gate)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Against the composed form the kernels replace. Fused arithmetic is not required to
    /// be bit-identical — it skips intermediate rounding through memory — so this checks a
    /// tight relative bound rather than equality.
    #[test]
    fn snake_forms_agree() -> anyhow::Result<()> {
        let d = Device::Cpu;
        let x = Tensor::randn(0f32, 2., (1, 7, 33), &d)?;
        let alpha = Tensor::randn(0f32, 1., 7, &d)?;

        let want = (&x + x.sin()?.sqr()?)?;
        let got = snake_folded(&x)?;
        assert!(crate::max_abs_diff(&want, &got)? < 1e-5);

        let want = crate::snake(&x, &alpha.reshape((1, 7, 1))?)?;
        let got = snake_alpha(&x, &alpha)?;
        assert_eq!(want.dims(), got.dims());
        assert!(crate::max_abs_diff(&want, &got)? < 1e-5);
        Ok(())
    }

    /// `snake_beta` against `snake_full`, **on every device that is actually available**.
    ///
    /// The CPU arm alone would not be worth much: it exercises `cpu_fwd`, which is the
    /// fallback, not the kernel that runs in production. Reverting device sampling cost a
    /// day to a unit test that passed on a shape the real model never uses — the shapes
    /// here are the codec's own (1536 channels, a chunk's worth of samples).
    #[test]
    fn snake_beta_matches_composed() -> anyhow::Result<()> {
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if let Some(m) = crate::usable_metal() {
            devices.push(m);
        }
        for d in devices {
            for (c, len) in [(7, 33), (1536, 601), (24, 4801)] {
                let x = Tensor::randn(0f32, 2., (1, c, len), &d)?;
                let alpha = Tensor::randn(0f32, 1., c, &d)?.exp()?;
                let beta_recip = Tensor::randn(0f32, 1., c, &d)?.exp()?;

                let want = crate::snake_full(
                    &x,
                    &alpha.reshape((1, c, 1))?,
                    &beta_recip.reshape((1, c, 1))?,
                )?;
                let got = snake_beta(&x, &alpha, &beta_recip)?;
                assert_eq!(want.dims(), got.dims(), "{d:?} at {c}x{len}");
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "{d:?} at {c}x{len}: abs {abs:.3e} rel {rel:.3e}"
                );
            }
        }
        Ok(())
    }
}
