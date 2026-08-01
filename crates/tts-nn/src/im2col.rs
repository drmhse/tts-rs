//! A Metal im2col gather that writes the tap-major matrix `causal_conv1d_gemm` wants.
//!
//! # Why a custom kernel, when candle ships one
//!
//! `causal_conv1d_gemm` builds its im2col matrix with `cat(dim=0)` over `k` narrowed
//! views. `codecsplit` put **79% of the conv route in that gather** — 28.6 ms against a
//! 7.1 ms GEMM at `96ch @ 131072, k=7`, which is ~24 GB/s on a ~120 GB/s bus.
//!
//! candle already has `call_im2col1d_strided`, private to `conv1d`. Wiring it up through
//! a `CustomOp1` and timing it measured **0.66x — slower than `cat`** (43.5 ms vs 28.6 ms,
//! canary-stable at 0.99x drift). That refutes the easy version of this idea, and the
//! reason is instructive. Its kernel indexes by a linear `thread_position_in_grid` and
//! recovers `(b, l, c, k)` with a chain of three `size_t` divisions per element. At
//! 88 M output elements that arithmetic, not the memory traffic, is the cost. Its output
//! layout `(b, l_out, c_in, l_k)` is also the wrong one here: it suits `cols @ wᵀ`, which
//! yields `[l, cout]` and then needs a **full strided transpose** to get back to
//! channels-first (`metal_backend/mod.rs:950`) — that copy is why candle's `conv1d` as a
//! whole lost to the `cat` route in the first place.
//!
//! So the bound to beat is not candle's kernel but the hardware: `index_select` does this
//! same gather at ~81 GB/s (8.67 ms), it just needs a 352 MB index to do it. This kernel
//! reaches the same access pattern with no index at all:
//!
//! - **A 3-D dispatch grid** `(len, cin, k)`, so each index arrives as a grid coordinate
//!   and there is not a single division in the kernel body.
//! - **`l` on the x axis**, so adjacent threads read adjacent source elements *and* write
//!   adjacent destination elements. Both sides coalesce.
//! - **Tap-major output** `[k * cin, len]`, which is exactly
//!   [`tap_major_weight`](crate::tap_major_weight)'s existing contract — so the GEMM stays
//!   `w_tap @ cols -> [cout, len]` and no transpose is ever needed.
//! - **The causal pad folded in.** Out-of-range taps write zero instead of reading, which
//!   removes the `pad_with_zeros` copy the `cat` route paid on every call.
//!
//! The op falls back to a plain CPU implementation off Metal, which is also what the unit
//! tests check the GPU path against.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

/// `[1, cin, l_in] -> [k * cin, l_in]`, tap-major, causally padded.
struct Im2ColTapMajor {
    k: usize,
    dilation: usize,
}

impl CustomOp1 for Im2ColTapMajor {
    fn name(&self) -> &'static str {
        "im2col_tap_major"
    }

    /// Reference implementation. The Metal path is checked against this in `tests`.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, cin, l_in) = layout.shape().dims3()?;
        if b != 1 {
            candle_core::bail!("im2col_tap_major: batch must be 1, got {b}");
        }
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("im2col_tap_major: only f32"),
        };
        let (o, st) = (layout.start_offset(), layout.stride());
        let pad = (self.k - 1) * self.dilation;
        let mut dst = vec![0f32; self.k * cin * l_in];
        for t in 0..self.k {
            for c in 0..cin {
                let row = (t * cin + c) * l_in;
                for l in 0..l_in {
                    // The causal pad: taps reaching before the start contribute zero.
                    let s = (l + t * self.dilation) as isize - pad as isize;
                    if s >= 0 {
                        dst[row + l] = src[o + c * st[1] + (s as usize) * st[2]];
                    }
                }
            }
        }
        Ok((CpuStorage::F32(dst), (self.k * cin, l_in).into()))
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

        let (b, cin, l_in) = layout.shape().dims3()?;
        if b != 1 {
            candle_core::bail!("im2col_tap_major: batch must be 1, got {b}");
        }
        if !layout.is_contiguous() {
            candle_core::bail!("im2col_tap_major: input must be contiguous");
        }
        if storage.dtype() != DType::F32 {
            candle_core::bail!("im2col_tap_major: only f32, got {:?}", storage.dtype());
        }

        let device = storage.device();
        let pipeline = crate::mtl::pipeline(device, "im2col_tap_major_f32")?;
        let dst_el = self.k * cin * l_in;
        let dst = device.new_buffer(dst_el, DType::F32, "im2col_tap_major")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::im2col_tap_major");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(storage.buffer()), layout.start_offset() * 4);
        encoder.set_buffer(1, Some(dst.as_ref()), 0);
        encoder.set_bytes(2, &(l_in as u32));
        encoder.set_bytes(3, &(cin as u32));
        encoder.set_bytes(4, &(self.dilation as u32));
        encoder.set_bytes(5, &(((self.k - 1) * self.dilation) as u32));
        encoder.use_resource(storage.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);

        // One thread per output element, laid out so x (length) is the fast axis: adjacent
        // threads then touch adjacent addresses on both the read and the write side.
        let w = crate::mtl::group_width(&pipeline, l_in);
        encoder.dispatch_threads(
            MTLSize {
                width: l_in,
                height: cin,
                depth: self.k,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        let out = MetalStorage::new(dst, device.clone(), dst_el, DType::F32);
        Ok((out, (self.k * cin, l_in).into()))
    }
}

/// The tap-major im2col matrix `[k * cin, len]` for a causal conv, as one dispatch.
///
/// The causal left-pad is applied inside the gather, so unlike the `cat` route the caller
/// does not pass a pre-padded input.
pub fn im2col_tap_major(x: &Tensor, k: usize, dilation: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&Im2ColTapMajor { k, dilation })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The gather must reproduce exactly what `cat(dim=0)` over narrowed views builds —
    /// that matrix is what `tap_major_weight` was permuted to match, so any disagreement
    /// is a silently transposed kernel.
    #[test]
    fn matches_cat_matrix() -> anyhow::Result<()> {
        let d = Device::Cpu;
        for (cin, len, k, dil) in [
            (3usize, 11usize, 3usize, 1usize),
            (8, 17, 7, 3),
            (6, 9, 1, 1),
        ] {
            let x = Tensor::randn(0f32, 1., (1, cin, len), &d)?;
            let xpad = x.pad_with_zeros(2, (k - 1) * dil, 0)?;
            let taps: Vec<_> = (0..k)
                .map(|t| xpad.narrow(2, t * dil, len))
                .collect::<Result<Vec<_>>>()?;
            let want = Tensor::cat(&taps, 0)?.reshape((k * cin, len))?;
            let got = im2col_tap_major(&x, k, dil)?;
            assert_eq!(
                want.dims(),
                got.dims(),
                "cin={cin} len={len} k={k} dil={dil}"
            );
            assert_eq!(crate::max_abs_diff(&want, &got)?, 0.0);
        }
        Ok(())
    }
}
