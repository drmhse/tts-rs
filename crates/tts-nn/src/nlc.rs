//! Convolutions in channels-last `[b, L, C]` layout.
//!
//! The channel-major path materialises an im2col matrix `k` times the input to turn a conv into
//! one GEMM. Channels-last needs no matrix at all: with `x` as `[L, C]`, tap `t` of a causal conv
//! is the *contiguous* slice `x_pad[t*d .. t*d + L]`, so the conv is `k` accumulating GEMMs of
//! `[L, C_in] x [C_in, C_out]` over a tensor that is only read, never expanded.
//!
//! Two things follow, and both matter for the qwen3tts codec's decoder, which is ~92% of its
//! cost at 96-768 channels over up to 576 k samples:
//!
//! - **Traffic.** im2col writes and re-reads `k * L * C_in`; this reads `L * C_in` `k` times and
//!   writes nothing extra. Measured ~20 GB of im2col traffic per chunk in that decoder.
//! - **GEMM shape.** `M` becomes `L` (hundreds of thousands) instead of `C_out` (as low as 96).
//!   A short `M` is the wrong direction for MPS's tiling.
//!
//! Elementwise work is layout-agnostic, and LayerNorm over channels becomes a last-axis
//! reduction — so the decoder needs no transposes once it is in this layout.
//!
//! The GEMMs stay f32. Casting them to f16 measured **13% slower** (1720 ms against 1518 on a
//! 300-frame chunk): the cast traffic exceeds what the narrower multiply saves, which is the
//! evidence that this stage is bound by moving activations rather than by arithmetic. Running
//! the whole decoder in f16 would avoid the casts, and is untried.

use crate::matmul_2d;
use anyhow::Result;
use candle_core::Tensor;

/// A causal conv weight for [`causal_conv1d`], `[k, in, out]`.
///
/// From torch's `[out, in, k]`. Tap `t` is then a contiguous `[in, out]` slice, which is exactly
/// what each accumulating GEMM wants.
pub fn tap_weight(w: &Tensor) -> Result<Tensor> {
    let (out, cin, k) = w.dims3()?;
    Ok(w.permute((2, 1, 0))?.contiguous()?.reshape((k, cin, out))?)
}

/// Causal conv over `[b, L, C_in]` -> `[b, L, C_out]`.
///
/// `w` is `[k, in, out]` from [`tap_weight`]; `bias` is `[out]`.
pub fn causal_conv1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    dilation: usize,
) -> Result<Tensor> {
    let (b, len, cin) = x.dims3()?;
    let (k, wk_in, out) = w.dims3()?;
    anyhow::ensure!(wk_in == cin, "conv input width {cin} != weight {wk_in}");

    // One pad, reused by every tap, rather than one shifted copy per tap.
    let pad = (k - 1) * dilation;
    let xp = if pad > 0 {
        x.pad_with_zeros(1, pad, 0)?
    } else {
        x.clone()
    };

    // One GEMM over the taps concatenated on the channel axis, not `k` accumulating GEMMs.
    //
    // Accumulating was the obvious form and is the wrong one: each tap materialises a full
    // `[L, C_out]` output and the k-1 adds re-read them, ~7 GB per conv at the decoder's widest
    // stage against ~5 GB for concatenate-then-multiply. The concat is affordable *here* only
    // because a `narrow` on the length axis of a channels-last tensor is already contiguous —
    // the same trick in channel-major layout is the im2col gather that needed its own kernel.
    let y = if k == 1 {
        matmul_2d(&xp, &w.reshape((cin, out))?)?
    } else {
        let mut taps = Vec::with_capacity(k);
        for t in 0..k {
            taps.push(xp.narrow(1, t * dilation, len)?);
        }
        let wide = Tensor::cat(&taps, 2)?.contiguous()?;
        matmul_2d(&wide, &w.reshape((k * cin, out))?)?
    };
    let y = y.reshape((b, len, out))?;
    Ok(match bias {
        Some(bi) => y.broadcast_add(&bi.reshape((1, 1, out))?)?,
        None => y,
    })
}

/// Weights for [`causal_conv_transpose1d`]: one `[in, out * stride]` matrix per tap group.
///
/// A transposed conv with `kernel == m * stride` gives every output exactly **m** contributing
/// taps: output `l * s + p` takes tap `p + i * s` against `x[l - i]`, for `i` in `0..m`. So the
/// whole thing is `m` GEMMs whose columns are the `s` output phases interleaved, and candle's
/// `conv_transpose1d` is not needed. Both `m == 1` (the ratio-2 upsamplers) and `m == 2` (the
/// decoder blocks) occur in the qwen3tts codec.
pub fn transpose_weights(w: &Tensor, stride: usize) -> Result<Vec<Tensor>> {
    let (cin, out, k) = w.dims3()?;
    anyhow::ensure!(
        k % stride == 0,
        "polyphase form needs kernel divisible by stride, got k={k} stride={stride}"
    );
    // [in, out, k] -> [in, k, out], so phases are contiguous per tap.
    let t = w.permute((0, 2, 1))?.contiguous()?;
    (0..k / stride)
        .map(|i| {
            Ok(t.narrow(1, i * stride, stride)?
                .contiguous()?
                .reshape((cin, stride * out))?)
        })
        .collect()
}

/// Causal transposed conv over `[b, L, C_in]` -> `[b, L * stride, C_out]`.
///
/// Matches `crate::causal_conv_transpose1d` (convolve then drop the trailing `k - stride`) for
/// any `kernel` divisible by `stride`.
pub fn causal_conv_transpose1d(
    x: &Tensor,
    taps: &[Tensor],
    bias: Option<&Tensor>,
    stride: usize,
) -> Result<Tensor> {
    let (b, len, _) = x.dims3()?;
    let out = taps[0].dim(1)? / stride;

    let mut acc: Option<Tensor> = None;
    for (i, w) in taps.iter().enumerate() {
        // Tap group i pairs with `x[l - i]`: shift down by i, zero-filled at the start.
        let src = if i == 0 {
            x.contiguous()?
        } else if i < len {
            x.narrow(1, 0, len - i)?
                .pad_with_zeros(1, i, 0)?
                .contiguous()?
        } else {
            continue;
        };
        let term = matmul_2d(&src, w)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }

    // [b, L, stride * out] -> [b, L * stride, out]: phases are already the minor axis.
    let y = acc
        .expect("at least one tap group")
        .reshape((b, len * stride, out))?;
    Ok(match bias {
        Some(bi) => y.broadcast_add(&bi.reshape((1, 1, out))?)?,
        None => y,
    })
}

/// Depthwise causal conv over `[b, L, C]`, as shifted multiply-accumulates.
///
/// `w` is torch's `[C, 1, k]`. Same reason as [`crate::depthwise_k7`]: candle's `groups > 1`
/// conv is far off the hardware on this backend.
pub fn depthwise(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (_, len, c) = x.dims3()?;
    let k = w.dim(2)?;
    let taps = w.reshape((c, k))?;
    let xp = x.pad_with_zeros(1, k - 1, 0)?;

    let mut acc: Option<Tensor> = None;
    for t in 0..k {
        let tap = taps.narrow(1, t, 1)?.reshape((1, 1, c))?;
        let term = xp.narrow(1, t, len)?.broadcast_mul(&tap)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    let y = acc.expect("kernel size > 0");
    Ok(match bias {
        Some(bi) => y.broadcast_add(&bi.reshape((1, 1, c))?)?,
        None => y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Against the channel-major implementations, on every device available. These are the same
    /// arithmetic in a different layout, so the bound is tight.
    #[test]
    fn matches_channel_major() -> Result<()> {
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if let Some(d) = crate::usable_metal() {
            devices.push(d);
        }
        for dev in devices {
            for (cin, cout, len, k, dil) in
                [(8, 12, 40, 7, 1), (16, 16, 33, 7, 3), (6, 9, 20, 1, 1)]
            {
                let x = Tensor::randn(0f32, 1., (1, cin, len), &dev)?;
                let w = Tensor::randn(0f32, 0.1, (cout, cin, k), &dev)?;
                let bias = Tensor::randn(0f32, 0.1, cout, &dev)?;

                let want = crate::causal_conv1d(&x, &w, Some(&bias), dil)?;
                let got = causal_conv1d(
                    &x.transpose(1, 2)?.contiguous()?,
                    &tap_weight(&w)?,
                    Some(&bias),
                    dil,
                )?
                .transpose(1, 2)?
                .contiguous()?;
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "conv {dev:?} {cin}->{cout} k{k} d{dil}: {abs:.2e} {rel:.2e}"
                );
            }

            // Transposed, at the strides the qwen3tts codec upsamples by.
            // `mult` 1 covers the ratio-2 upsamplers (kernel == stride), 2 the decoder blocks.
            for (cin, cout, len, stride, mult) in [
                (8, 6, 12, 2, 1),
                (8, 6, 12, 2, 2),
                (12, 8, 9, 5, 2),
                (10, 7, 11, 3, 2),
                (9, 5, 7, 4, 1),
            ] {
                let x = Tensor::randn(0f32, 1., (1, cin, len), &dev)?;
                let w = Tensor::randn(0f32, 0.1, (cin, cout, mult * stride), &dev)?;
                let bias = Tensor::randn(0f32, 0.1, cout, &dev)?;

                let want = crate::causal_conv_transpose1d(&x, &w, Some(&bias), stride)?;
                let taps = transpose_weights(&w, stride)?;
                let got = causal_conv_transpose1d(
                    &x.transpose(1, 2)?.contiguous()?,
                    &taps,
                    Some(&bias),
                    stride,
                )?
                .transpose(1, 2)?
                .contiguous()?;
                assert_eq!(
                    want.dims(),
                    got.dims(),
                    "transpose {dev:?} stride {stride} m{mult}"
                );
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "transpose {dev:?} stride {stride} m{mult}: {abs:.2e} {rel:.2e}"
                );
            }

            // Depthwise.
            let x = Tensor::randn(0f32, 1., (1, 16, 30), &dev)?;
            let w = Tensor::randn(0f32, 0.1, (16, 1, 7), &dev)?;
            let bias = Tensor::randn(0f32, 0.1, 16, &dev)?;
            let want = crate::depthwise_k7(&x, &w, Some(&bias))?;
            let got = depthwise(&x.transpose(1, 2)?.contiguous()?, &w, Some(&bias))?
                .transpose(1, 2)?
                .contiguous()?;
            let (abs, rel) = crate::abs_and_rel(&want, &got)?;
            assert!(rel < 1e-5, "depthwise {dev:?}: {abs:.2e} {rel:.2e}");
        }
        Ok(())
    }
}
