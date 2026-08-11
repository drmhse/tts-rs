# Rejected: an f16 codec decoder

The waveform stack — `head_conv`, four decoder blocks, the output conv — is ~95% of the codec and
is bound by moving activations rather than by arithmetic. f16 halves that traffic. It was 1.17x
faster and **misses the fixtures by 6x**, so it is not available.

| | RTF (300-frame chunk) | `codec.wav` rel | tolerance |
|---|---|---|---|
| f32 | 0.0627 | 9.8e-5 | 5.0e-3 |
| f16 | **0.0547** | **3.1e-2** | 5.0e-3 |

## Why, and why the estimate was wrong

Not range: the waveform sits well inside f16. The estimate that f16's ~1e-3 rounding would
random-walk to ~5e-3 over 24 sequential convs was wrong about the mechanism.

`SnakeBeta` is `x + beta_recip * sin(exp(alpha) * x)^2`. The input is multiplied by `exp(alpha)`
*before* the sine, so any error in `x` is amplified by `exp(alpha)` rather than carried. Twenty-four
of those in series down the upsampling stack turns storage rounding into 3.1e-2. A nonlinearity
with a learned input scale is the wrong place to be casual about precision.

## Two related things that also did not work

- **f16 GEMMs only**, casting at each conv and keeping f32 activations: 13% *slower* (1720 ms
  against 1518 on the same chunk). The cast traffic exceeds what the narrower multiply saves —
  which is the measurement that says this stage is traffic-bound, and the reason carrying f16 all
  the way through looked worth trying.
- **Fusing the k taps into one GEMM** over channel-concatenated slices instead of k accumulating
  GEMMs: 4% (1273 -> 1224 ms on the residual units). The concat cost nearly cancels the saved
  accumulator adds. Kept, since it is no worse and slightly simpler to reason about.

## What is left

`crates/qwen3tts/src/codec.rs` keeps `WAV_DTYPE` as the single switch, so re-testing this against
a future candle costs one line. Anything that makes it viable has to address the snake
amplification — per-stage dtype (only the widest, shortest-lived stage in f16), or a fused
conv+snake kernel that keeps the intermediate in registers at f32 and only stores f16.
