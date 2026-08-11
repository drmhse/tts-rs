//! The custom Metal kernels this crate ships, and the pipeline cache behind them.
//!
//! Everything here exists because of two measurements in `docs/performance/candle-on-metal.md`:
//!
//! - **Finding 2**: candle performs no fusion whatsoever. `snake` as five composed ops
//!   costs 11.5 ms at `[1, 96, 131072]`, and the five ops measured individually sum to
//!   13.7 ms — so every elementwise expression pays one full round-trip to device memory
//!   *per operator*. A single fused pass costs what `affine` costs, 1.33 ms: **8.7x**.
//! - **The im2col gather**: `cat(dim=0)` builds the conv matrix at ~24 GB/s where the
//!   hardware manages ~81-130 GB/s.
//!
//! Both are the same shape of problem — candle composes correct ops that each re-read
//! memory — and both are fixed the same way, by doing the whole expression in one pass.
//!
//! The kernels live in one source string compiled once per device, because compiling a
//! library costs milliseconds and these run hundreds of times per utterance.

use candle_core::Result;

#[cfg(feature = "metal")]
pub(crate) const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- im2col ------------------------------------------------------------------
//
// dst[(t * cin + c) * l_in + l] = src[c * l_in + l + t * dilation - pad], or 0 when the
// tap reaches before the start of the signal.
//
// The indices arrive as grid coordinates, so there is not a single division in the body.
// candle's own im2col recovers four indices from a linear thread id with three size_t
// divisions each, and at 88 M elements that arithmetic — not the traffic — is what makes
// it 0.66x slower than the `cat` route it was meant to replace.
kernel void im2col_tap_major_f32(
    device const float *src   [[buffer(0)]],
    device float       *dst   [[buffer(1)]],
    constant uint      &l_in  [[buffer(2)]],
    constant uint      &cin   [[buffer(3)]],
    constant uint      &dil   [[buffer(4)]],
    constant uint      &pad   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= l_in) { return; }
    const uint c = gid.y;
    const uint t = gid.z;

    const uint s = l + t * dil;
    dst[(t * cin + c) * l_in + l] = (s < pad) ? 0.0f : src[c * l_in + (s - pad)];
}

// ---- head transpose ------------------------------------------------------------
//
// [b, n, h, d] -> [b, h, n, d], the layout `sdpa` wants for multi-head attention.
//
// `d` is the fast axis and is unit-stride on *both* sides, so every thread's read and
// write coalesce. candle's `transpose(1,2).contiguous()` manages ~4.6 GB/s on this shape;
// there is nothing about the movement that requires that.
kernel void head_transpose_f32(
    device const float *src [[buffer(0)]],
    device float       *dst [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    constant uint      &hd  [[buffer(3)]],
    constant uint      &dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d  = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;          // n
    const uint bh  = gid.z;          // b * hd + h
    const uint b   = bh / hd;
    const uint hh  = bh - b * hd;

    dst[(bh * n + pos) * dim + d] = src[((b * n + pos) * hd + hh) * dim + d];
}

// ---- modulate tail and gated residual -------------------------------------------
//
// Both are `[b, n, d]` against a `[b, 1, d]` vector, which candle does with
// `broadcast_mul` at 22 GB/s — 3.6x slower than a plain unary op, for what is only index
// arithmetic (Finding 2). A DiT block runs two of each, 660 times per utterance.
//
// `d` is the fast axis and unit-stride on both sides; the broadcast vector is indexed
// directly rather than expanded.

// out = x * (1 + scale[b,d]) + shift[b,d]  — the affine half of `modulate`, after the norm.
kernel void modulate_affine_f32(
    device const float *x     [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *shift [[buffer(2)]],
    device float       *dst   [[buffer(3)]],
    constant uint      &n     [[buffer(4)]],
    constant uint      &dim   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;
    const uint b   = gid.z;

    const uint i = (b * n + pos) * dim + d;
    const uint j = b * dim + d;
    dst[i] = x[i] * (1.0f + scale[j]) + shift[j];
}

// out = r + y * gate[b,d]  — the residual add and its gate, in one pass instead of two.
kernel void gate_residual_f32(
    device const float *r    [[buffer(0)]],
    device const float *y    [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float       *dst  [[buffer(3)]],
    constant uint      &n    [[buffer(4)]],
    constant uint      &dim  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= dim) { return; }
    const uint pos = gid.y;
    const uint b   = gid.z;

    const uint i = (b * n + pos) * dim + d;
    dst[i] = r[i] + y[i] * gate[b * dim + d];
}

// ---- snake -------------------------------------------------------------------
//
// y = x + sin^2(x), for inputs whose alpha has already been folded into the preceding
// conv's output weights. Three composed candle ops (sin, sqr, add) become one pass.
kernel void snake_folded_f32(
    device const float *src [[buffer(0)]],
    device float       *dst [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float x = src[gid];
    const float s = sin(x);
    dst[gid] = x + s * s;
}

// ---- snake with a per-channel alpha -------------------------------------------
//
// y = u + sin^2(u) where u = alpha[c] * x, for the leading snake of a residual group —
// there the block input also feeds the skip, so alpha cannot be folded away.
//
// alpha is indexed by the grid's y axis rather than broadcast, which matters: a candle
// `broadcast_mul` of [1,C,1] against [1,C,L] runs at 22 GB/s, 3.6x slower than a plain
// unary op, and snake as written contained two of them.
kernel void snake_alpha_f32(
    device const float *src   [[buffer(0)]],
    device const float *alpha [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &len   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const float u = alpha[c] * src[c * len + l];
    const float s = sin(u);
    dst[c * len + l] = u + s * s;
}

// ---- snake beta ---------------------------------------------------------------
//
// y = x + beta_recip[c] * sin^2(alpha[c] * x) — SnakeBeta, with a per-channel amplitude
// as well as a per-channel frequency. Neither can be folded away: the input also feeds a
// skip, and beta is independent of alpha. `snake_full` is six composed ops, so it pays six
// round trips where this pays one.
kernel void snake_beta_f32(
    device const float *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device float       *dst    [[buffer(3)]],
    constant uint      &len    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint l = gid.x;
    if (l >= len) { return; }
    const uint c = gid.y;

    const uint i = c * len + l;
    const float x = src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = x + brecip[c] * s * s;
}

// ---- decode attention ----------------------------------------------------------
//
// Both read the KV cache in place, indexing with `capacity` as the row stride, which is what
// candle cannot do: `narrow(2, 0, span)` of the cache is non-contiguous, so it copies the span
// twice per layer per step.

// scores[bh, g, p] = dot(q[bh, g, :], k[bh, p, :]), masked outside [window_start, span).
kernel void decode_scores_f32(
    device const float *q     [[buffer(0)]],
    device const float *k     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    constant uint      &wstart [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint p = gid.x;
    if (p >= span) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const uint o = (bh * gqa + g) * span + p;
    if (p < wstart) { dst[o] = -INFINITY; return; }

    const device float *qr = q + (bh * gqa + g) * hd;
    const device float *kr = k + (bh * cap + p) * hd;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) { acc += qr[d] * kr[d]; }
    dst[o] = acc;
}

// out[bh, g, d] = sum_p probs[bh, g, p] * v[bh, p, d].
//
// `d` is the grid's fast axis, so consecutive threads read consecutive `v` — coalesced.
kernel void decode_weighted_f32(
    device const float *probs [[buffer(0)]],
    device const float *v     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= hd) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const device float *pr = probs + (bh * gqa + g) * span;
    const device float *vc = v + bh * cap * hd + d;
    float acc = 0.0f;
    for (uint p = 0; p < span; ++p) { acc += pr[p] * vc[p * hd]; }
    dst[(bh * gqa + g) * hd + d] = acc;
}

// f16 cache variants: half the bytes on the hot read, accumulated in float either way.
kernel void decode_scores_f16(
    device const float *q     [[buffer(0)]],
    device const half  *k     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    constant uint      &wstart [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint p = gid.x;
    if (p >= span) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const uint o = (bh * gqa + g) * span + p;
    if (p < wstart) { dst[o] = -INFINITY; return; }

    const device float *qr = q + (bh * gqa + g) * hd;
    const device half  *kr = k + (bh * cap + p) * hd;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) { acc += qr[d] * (float)kr[d]; }
    dst[o] = acc;
}

kernel void decode_weighted_f16(
    device const float *probs [[buffer(0)]],
    device const half  *v     [[buffer(1)]],
    device float       *dst   [[buffer(2)]],
    constant uint      &span  [[buffer(3)]],
    constant uint      &cap   [[buffer(4)]],
    constant uint      &hd    [[buffer(5)]],
    constant uint      &gqa   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= hd) { return; }
    const uint g  = gid.y;
    const uint bh = gid.z;

    const device float *pr = probs + (bh * gqa + g) * span;
    const device half  *vc = v + bh * cap * hd + d;
    float acc = 0.0f;
    for (uint p = 0; p < span; ++p) { acc += pr[p] * (float)vc[p * hd]; }
    dst[(bh * gqa + g) * hd + d] = acc;
}

// Channels-last SnakeBeta: C is the grid's fast axis, so the parameter lookup is a broadcast
// within a threadgroup and both the read and the write coalesce.
kernel void snake_beta_nlc_f32(
    device const float *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device float       *dst    [[buffer(3)]],
    constant uint      &chan   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    if (c >= chan) { return; }
    const uint i = gid.y * chan + c;
    const float x = src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = x + brecip[c] * s * s;
}

// f16 activations, f32 parameters and math: the sin and the square want the wider type, the
// tensor crossing memory does not.
kernel void snake_beta_nlc_f16(
    device const half  *src    [[buffer(0)]],
    device const float *alpha  [[buffer(1)]],
    device const float *brecip [[buffer(2)]],
    device half        *dst    [[buffer(3)]],
    constant uint      &chan   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    if (c >= chan) { return; }
    const uint i = gid.y * chan + c;
    const float x = (float)src[i];
    const float s = sin(alpha[c] * x);
    dst[i] = (half)(x + brecip[c] * s * s);
}

// ---- swiglu tail ---------------------------------------------------------------
//
// out = silu(g) * u, elementwise. candle spends two dispatches and two full round trips on
// this. That is nothing on a long sequence, but a batch-1 decode step is dispatch-bound: the
// qwen3 predictor runs it 70 times per audio frame, where the cost is the launch and not
// the 3072 lanes of arithmetic.
kernel void swiglu_mul_f32(
    device const float *g   [[buffer(0)]],
    device const float *u   [[buffer(1)]],
    device float       *dst [[buffer(2)]],
    constant uint      &n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float x = g[gid];
    dst[gid] = (x / (1.0f + exp(-x))) * u[gid];
}
"#;

/// Compile the library once per device and hand out cached pipelines by function name.
#[cfg(feature = "metal")]
pub(crate) fn pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<
        Mutex<(
            HashMap<DeviceId, Library>,
            HashMap<(DeviceId, &'static str), ComputePipeline>,
        )>,
    > = OnceLock::new();

    let cache = CACHE.get_or_init(|| Mutex::new((HashMap::new(), HashMap::new())));
    let mut guard = cache
        .lock()
        .map_err(|e| candle_core::Error::Metal(format!("kernel cache poisoned: {e}").into()))?;
    let id = device.id();

    if let Some(p) = guard.1.get(&(id, name)) {
        return Ok(p.clone());
    }
    let (libs, pipelines) = &mut *guard;
    let lib = match libs.get(&id) {
        Some(l) => l.clone(),
        None => {
            let l = device
                .metal_device()
                .new_library_with_source(SHADER, None)
                .map_err(candle_core::Error::wrap)?;
            libs.insert(id, l.clone());
            l
        }
    };
    let func = lib
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(candle_core::Error::wrap)?;
    pipelines.insert((id, name), p.clone());
    Ok(p)
}

/// A threadgroup width for a 1-D fast axis, capped by what the pipeline allows.
#[cfg(feature = "metal")]
pub(crate) fn group_width(
    p: &candle_metal_kernels::metal::ComputePipeline,
    fast_axis: usize,
) -> usize {
    // Two independent caps: what the pipeline permits, and how wide the axis actually
    // is. 256 is the practical ceiling — wider threadgroups do not help these kernels.
    let permitted = p.max_total_threads_per_threadgroup().clamp(1, 256);
    permitted.min(fast_axis.max(1))
}
