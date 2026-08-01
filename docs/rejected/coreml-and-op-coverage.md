# Phase C probe — Candle/Metal op coverage for the Audio8 codec

> **Absolute timings here are unreliable — see [../benchmarking.md](../benchmarking.md).**
> They were measured on a cool machine; the same code ran ~2x slower later in the
> session. Ratios measured within a single run still hold; absolute ms figures do not.
>
> **And this document is aimed at the wrong half of the pipeline.** The AR loop was
> later measured at 10-20x the codec's cost — see
> [../performance/ar-loop.md](../performance/ar-loop.md). Everything here is real, but it is
> optimising 4-8% of the runtime. Read that first.

`cargo run -p a8-probe --release`, M4 / 16 GB, candle 0.10.2, f32, 64 code frames
= 2.97 s of audio at 44.1 kHz. Random tensors at the real decoder shapes — no
weights required, which is why this could run before the download finished.

## Verdict: GO

**20/20 ops run on Metal, 0 failures.** `conv_transpose1d` — the single risk that
could have killed the port — works at every stage (k16/s8, k16/s8, k8/s4, k4/s2)
and is cheap: 34 ms total across all four. Depthwise `conv1d` with `groups=1024`
works. No op in the decode path is missing a Metal kernel.

Note candle **0.11.0 does not build on stable rustc** (1.92.0): its NEON path uses
the unstable `stdarch_neon_f16` feature. 0.10.2 builds clean. Pinned there.

## Measured

| op | ms | note |
|---|---|---|
| rvq embed + out_proj (8→1024, k1) | 0.65 | ×10 codebooks |
| post_module attn (1024d, 16h, window 128) | 0.81 | ×8 layers |
| upsample convT 1024→1024 k2 s2 | 1.35 | ×2 |
| convnext dwconv 1024ch k7 **groups=1024** | **37.78** | pathological — see below |
| convnext dwconv **as 7 shifted muls** | **1.96** | **19× faster** |
| decoder entry conv 1024→1536 k7 | 4.01 | |
| convT 1536→768 k16 s8 @256 | 4.01 | |
| convT 768→384 k16 s8 @2048 | 9.05 | |
| convT 384→192 k8 s4 @16384 | 11.59 | |
| convT 192→96 k4 s2 @65536 | 8.99 | |
| residual_unit 768ch @2048 | 17.83 | ×3 |
| residual_unit 192ch @65536 | 102.71 | ×3 |
| residual_unit 96ch @131072 | 97.89 | ×3 |
| final conv 96→1 k7 + tanh | 47.48 | |
| slow AR wqkv 896→1152, 1 token | 0.26 | |
| slow AR logits 896→**155776**, 1 token | **6.72** | see optimisation 2 |

### Projected full codec decode

Scaling the measured units by their real multiplicity (12 residual units, 8
post_module layers, 10 codebooks):

| stage | ms |
|---|---|
| residual units (12) | ~961 |
| final conv + tanh | 47 |
| convT upsampling (4) | 34 |
| post_module (8 layers) | ~10 |
| quantizer upsample + convnext | ~9 |
| rvq from_codes (10) | 7 |
| decoder entry conv | 4 |
| **total** | **~1072 ms for 2.97 s → codec-only RTF ≈ 0.36** |

**The 12 residual units are 90% of the cost.** Everything else is noise.

For scale: the existing CosyVoice service does the *entire* pipeline at RTF 0.76
at 24 kHz. This codec alone is ~0.36 at 44.1 kHz, before the AR loop runs.

## Three things the probe changes about the plan

### 1. f16/bf16 is not a lever here — do not plan on it

| | f32 | f16 | bf16 |
|---|---|---|---|
| residual_unit 96ch @131072 | 97.89 | 93.62 | 93.35 |
| convT 384→192 k8 s4 | 11.59 | 11.46 | 11.40 |

**~4%.** The fp16 switch on the CosyVoice LLM bought 22% because autoregressive
decode is bandwidth-bound — every token reads every weight. The codec decoder is
the opposite: the weights are small and the activations are enormous, so it is
compute-bound, and candle's Metal conv kernels evidently do not use half-precision
math units. The dtype trick that worked on the AR loop does not transfer.

### 2. The AR logit projection can be 38× cheaper — a real win, not a micro-opt

The slow AR projects hidden → **155776** logits every token (6.72 ms measured).
But `ArkttsSemanticLogitsProcessor` immediately sets every logit to `-inf` except
ids `151678..155773` and `eos`. **Only 4097 of 155776 rows can ever be selected.**

Slicing the tied embedding matrix to those rows turns a 896×155776 matmul into
896×4097. At 21.5 frames/s, a 10 s clip is ~215 tokens — 1.45 s of pure waste on
the full projection. This is free, exact (the discarded logits are provably
unreachable), and the Python implementation leaves it on the table.

Caveat to document: it holds because the semantic mask runs *first*. A
caller-supplied `logits_processor` that un-masks other ids would break it, so the
Rust port should not expose that hook, or should fall back to the full projection
when it is used.

### 3. Skip candle's grouped conv entirely

`groups=1024` depthwise conv costs 37.78 ms for ~1.8 MMAC of actual work — off by
roughly three orders of magnitude. Expressing the same depthwise convolution as 7
shifted `broadcast_mul` accumulations gives an identical result in 1.96 ms. Use
the manual form in `a8-codec`; never call `conv1d` with `groups > 1`.

## ANSWERED: Candle is ~2× slower than PyTorch/MPS here

Both numbers are now measured end to end on the same 64 frames, same machine.
`cargo run -p a8-probe --release --bin cascade` builds the *entire* decode graph
with random weights (shapes determine cost, so correctness is irrelevant) and
`oracle/bench_codec.py` times the real `codec.decode()`.

| | ms | RTF |
|---|---|---|
| **candle / metal f32** | 990.0 | 0.332 |
| **candle / metal f16** | 888.3 | 0.298 |
| torch / mps f32 | 568.9 | 0.191 |
| **torch / mps fp16** | **403.6** | **0.136** |
| torch / cpu f32 | 1199.0 | 0.403 |

**Candle is 1.74× slower than torch in f32, and 2.2× slower than torch's best
configuration.** The earlier arithmetic extrapolation (~1072 ms) was sound — the
measured full cascade is 990 ms.

Note the asymmetry in the dtype row: torch gains 29% from fp16, candle gains 10%.
Torch's MPS kernels use half-precision math units; candle's conv kernels do not
appear to. So the gap *widens* in the configuration you would actually ship.

### What this does to the port's rationale

The memory argument is untouched and still strong: no MPSGraph cache means no
supervisor, no growth budget, no recycling, no 15–17 s reload, flat footprint by
construction, trivial streaming, single binary. Those were always the real prize.

But **the performance premise is refuted for the codec.** A Rust port would decode
audio at roughly half the speed of the Python path. The residual units run at ~5%
of M4 peak fp32, so the headroom exists — but capturing it means hand-written
fused Metal kernels (snake+conv), which is precisely the scope creep that kills
ports. Note also that this says nothing yet about the AR loop, which is where MLX
won 2.7× for CosyVoice and where Candle's per-token cost is unmeasured.
