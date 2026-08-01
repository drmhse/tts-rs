# Where Candle's 990 ms actually goes, and what it takes to beat torch's 403.6 ms

> **Absolute timings here are unreliable — see [../benchmarking.md](../benchmarking.md).**
> They were measured on a cool machine; the same code ran ~2x slower later in the
> session. Ratios measured within a single run still hold; absolute ms figures do not.
>
> **And this document is aimed at the wrong half of the pipeline.** The AR loop was
> later measured at 10-20x the codec's cost — see
> [ar-loop.md](ar-loop.md). Everything here is real, but it is
> optimising 4-8% of the runtime. Read that first.

Follow-up to `../rejected/coreml-and-op-coverage.md`. The first probe reported a number; this one takes
it apart. Conclusion up front: **there is large headroom — 8.7× on the elementwise
chain, ~5× on low-channel convs — and beating torch is achievable, but not by
recombining candle ops.** Two obvious rearrangements were tested and both are
*slower*. The lever is custom Metal kernels.

## The budget

Measured, M4 / 16 GB, f32, 64 frames = 2.97 s of audio:

| component | ms | share |
|---|---|---|
| k7 dilated convs (12 units × 4 stages' worth) | ~512 | 52% |
| snake (29 instances) | ~212 | 21% |
| k1 convs (12) | ~73 | 7% |
| conv_transpose (4) | 34 | 3% |
| post_module, upsample, rvq, entry conv | ~30 | 3% |
| residual adds | ~22 | 2% |
| **total** | **990** | |

> **Superseded, and instructively so.** Finding 1 below refuted conv-as-GEMM and blamed
> materialisation traffic. The refutation of *that specific im2col* stands, but the reason was
> wrong, and with the reason fixed conv-as-GEMM **does** work — 1.34x to 1.73x over candle's
> `conv1d` across all four codec stages. See "Finding 1 revisited" at the end of this document.
> The original text is kept unedited because the mis-attribution is the interesting part.

## Finding 1: conv-as-GEMM does NOT work (hypothesis refuted)

The first probe showed identical MAC counts taking 5.5× different time
(`768ch@2048` = 17.83 ms vs `96ch@131072` = 97.89 ms), which suggested candle's
conv kernel starves at low channel counts and that routing through GEMM — which
has no notion of channels — would fix it. **It does not.**

One k7/dilation-9 conv per stage, `best/direct` < 1 means the rewrite lost:

| shape | direct | im2col | taps | chunk 8k | best/direct |
|---|---|---|---|---|---|
| 768ch @ 2048 | **11.89** | 17.76 | 13.90 | 17.86 | 0.86× |
| 384ch @ 16384 | **35.73** | 54.70 | 41.91 | 51.02 | 0.85× |
| 192ch @ 65536 | **63.23** | 96.22 | 80.15 | 90.10 | 0.79× |
| 96ch @ 131072 | **59.78** | 95.40 | 81.38 | 86.99 | 0.73× |

GFLOPS by stage, direct: **1422 / 947 / 535 / 283**.

All three rewrites are numerically correct (verified against `conv1d`, max abs
diff ~1.9e-6) — they are just slower. Candle's conv1d is a decent kernel; it hits
1422 GFLOPS at 768 channels, ~33% of M4 peak. The im2col route dies on
materialisation: the `[131072, 672]` matrix is 352 MB to write and 352 MB to read
back, plus 7 non-contiguous transpose-and-`cat` copies. That traffic exceeds what
the better GEMM shape saves.

Chunking along length also loses. Worth understanding why, because it kills an
appealing idea: every candle op is a separate Metal dispatch that reads from and
writes to device memory regardless of tile size, so blocking does not keep
intermediates resident between ops. It only makes each dispatch smaller and adds
`cat` overhead. **Blocking cannot substitute for fusion.**

That conclusion was re-tested and held. Chunking was reintroduced later to bound the
658 MB-1.1 GB im2col matrices CosyVoice's `ups` convs build at utterance length, on the
theory that their size was hurting. It was not — the unchunked GEMM beats a direct conv
by **2.5× to 5.0× at every `ups` shape, including the 1.1 GB ones** — and chunking cost
Audio8's codec RTF 0.156 → 0.204 with the AR stage steady at 0.341 as a control. Reverted
a second time, for the same reason as the first.

The `Cout=1` tail conv (47.84 ms, 3.7 GFLOPS) is likewise not rescued by GEMM —
im2col makes it 82.62 ms.

## Finding 2: the elementwise chain is 8.7× off, and it is pure dispatch overhead

> **Resolved.** `tts_nn::fused` now ships Metal kernels for both snake forms. Measured
> against the composed form at eight real shapes: **1.41×-3.18×** for the folded
> `x + sin²x` and **2.45×-5.99×** for the per-channel-alpha form, canary-stable
> throughout. The projected 8.7× was optimistic because alpha folding (Finding 3) had
> already removed two of the five round-trips before the kernel existed — the kernel
> collects what folding could not reach.

Single-op costs at `[1, 96, 131072]` (50.3 MB):

| op | ms | GB/s |
|---|---|---|
| affine (1 read, 1 write) | 1.416 | 71 |
| sin | 1.290 | 78 |
| sqr | 1.287 | 78 |
| **broadcast_mul by `[1,C,1]`** | **4.655** | **22** |
| add (2 reads, 1 write) | 1.802 | 84 |

Snake as written costs **11.5 ms**; the five constituent ops measured individually
sum to **13.7 ms**. They match, which confirms there is no fusion whatsoever —
snake pays five full round-trips to device memory for one elementwise expression.
A single fused pass would cost what `affine` costs: **1.33 ms**, i.e. **8.7×**.

(Ignore the `copy / contiguous` row from the raw output — `t().t()` is a no-op
candle elides, so its "132731 GB/s" is an artifact, not a measurement. Real
achievable elementwise bandwidth here is ~84 GB/s against a ~120 GB/s bus.)

The standout is **`broadcast_mul` at 22 GB/s — 3.6× slower than a plain unary op**,
despite broadcasting a `[1,C,1]` against `[1,C,L]` being nothing but index
arithmetic. Snake contains two of them, accounting for ~9.3 of its 11.5 ms.
General lesson for the port: **avoid broadcasts in hot paths.**

## Finding 3: alpha folding — an exact win with no custom kernels

Because `snake(x) = x + α⁻¹sin²(αx)`, substituting `u = αx`:

```
snake(x) = u/α + α⁻¹sin²(u) = α⁻¹·(u + sin²u)
```

So `α` folds into the **preceding** conv's output weights and bias, and `α⁻¹` into
the **following** conv's input-channel weights — both offline, at conversion time.
Every snake in this decoder is sandwiched between two convs
(`Snake → conv7 → Snake → conv1 → +skip`, and each block's leading snake follows
the previous block's conv1), so this is always available. Both broadcasts vanish,
leaving three plain unary passes.

| | ms |
|---|---|
| snake, 5 ops with broadcasts | 11.504 |
| snake, 3 unary ops (α folded) | **3.852** |
| **speedup** | **2.99×** |

Algebra verified numerically: `max|diff| 0.000e0`. Not an approximation.

Across all 29 snake instances: **211.8 ms → 70.9 ms**, taking the cascade to
**849 ms**. Free, exact, and it should go into the weight converter regardless of
what else is decided.

## What it actually takes to beat 403.6 ms

Stacking the levers, each grounded in a measurement above:

| step | ms | note |
|---|---|---|
| today | 990 | |
| α folding (free, exact) | 849 | measured 2.99× on snake |
| fused snake kernel instead | 803 | measured 8.7× ceiling |
| all conv stages at the 1422 GFLOPS already observed at 768ch | 505 | 512 → 214 ms |
| fuse k1 conv + residual add into the same kernel | 432 | removes 73 + 22 ms of traffic |
| fp16 math that actually engages half-precision units | **~324** | torch gains 29%; candle currently gains 10% |

**~324 ms vs torch's 403.6 ms.** So yes — Candle can win, by ~1.25×, and with
bounded memory and no supervisor. But every step after α folding requires
`candle_core::CustomOp` with hand-written MSL. Specifically one kernel family:

**a fused residual unit** — `snake → dilated k7 conv → snake → k1 conv → add`
in one or two passes, tiled over channels and length, in fp16 with fp32
accumulation. That single kernel is 79% of the decoder's cost (512 + 212 + 73 + 22
of 990 ms).

Why this is a real opportunity rather than wishful thinking: **torch is not doing
this either.** MPS runs the same unfused op-at-a-time graph; its 403.6 ms comes
from better individual kernels, not fusion. Fusing across the residual unit is
something a from-scratch implementation can do and a framework port structurally
cannot — which is the strongest form of the original Rust argument, just relocated
from "Rust is faster" to "we control the fusion boundary."

Why it is also the classic way ports die: it is hand-written GPU code that must be
numerically validated against the Phase A fixtures, and the fp16 accumulation
strategy has to be checked against the WhisperX gate, not just eyeballed.

## Recommended next step

A **single-kernel spike**: implement fused snake as a `CustomOp1` with a Metal
implementation and check it against the measured 1.33 ms target and the fixture
numerics. It is the smallest piece that proves the mechanism — the MSL, the
`CustomOp` wiring, the dispatch, the validation loop — and it de-risks the much
harder fused-conv kernel. If the spike lands near 1.33 ms, the 324 ms projection
is credible and the port is worth building. If custom-kernel dispatch in candle
turns out to be awkward or the win evaporates, we have learned that for the price
of one kernel instead of a whole codec.


## Finding 1 revisited: conv-as-GEMM does work, and the original reason was wrong

Finding 1 concluded that "the im2col route dies on materialisation ... That traffic exceeds
what the better GEMM shape saves." The arithmetic never supported that. 704 MB at ~120 GB/s is
about 6 ms, but im2col measured 95.40 ms against a direct conv's 59.78 ms — so ~89 ms was
unaccounted for, and 17 GFLOP in 89 ms is 0.19 TFLOP/s, which is not what memory traffic looks
like.

Splitting the route in two settles it (`a8-probe --bin convgemm`, canary 60.0 ms both ends):

| half of the GEMM route, 96ch @ 131072 | ms |
|---|---|
| build the im2col matrix (7 narrows + `stack(dim=1)`) | **82.47** |
| the GEMM itself, matrix prebuilt | **7.10** |

The GEMM is *2.4 TFLOP/s* — on its own **8.4x faster** than the 59.7 ms direct conv. The gather
is 352 MB written in 82 ms, about **4.9 GB/s on a 120 GB/s bus**. Nothing was wrong with the
arithmetic; everything was wrong with building the matrix.

### The row order is the whole problem

`stack(dim=1)` interleaves taps within each channel, so every source row scatters across `k`
destination rows. Concatenating along dimension 0 instead gives each tap one contiguous
destination block:

| materialising `[672, 131072]` (352 MB) | ms | vs stack |
|---|---|---|
| `stack(dim=1)`, tap-interleaved | 82.48 | 1.00x |
| `cat(dim=0)`, tap-major | **27.36** | **3.01x** |
| `index_select`, precomputed index | **8.67** | **9.51x** |

`cat(dim=0)` produces rows in tap-major order, which costs only permuting the weight to
`[cout, k * cin]` once at load. That is `tts_nn::causal_conv1d_gemm`, and against candle's
`conv1d` at every stage the codec uses:

| stage | `conv1d` | cat + GEMM | gain |
|---|---|---|---|
| 768ch @ 2048 | 18.31 | 13.67 | **1.34x** |
| 384ch @ 16384 | 42.22 | 30.88 | **1.37x** |
| 192ch @ 65536 | 65.18 | 41.63 | **1.57x** |
| 96ch @ 131072 | 61.12 | 35.36 | **1.73x** |

Exact to 1.3e-6 (f32 reassociation), and the codec fixtures actually tightened — 3.4e-6 to
2.7e-6 — because the GEMM's reduction order is better than the conv kernel's. The win *grows*
with length, which is the opposite of what a traffic-bound explanation predicts and the
signature of the direct kernel degrading faster than the gather does.

End to end this took Audio8 from RTF 0.664 to **0.598**, with the codec stage at 0.327 → 0.260
(both later superseded: the custom im2col and snake kernels took the codec to 0.158 and the
engine to 0.499)
(1.26x — less than the 1.73x on a single conv, because the codec also spends time in its
transformer, the RVQ gathers, the transposed convs and the snakes).

The same change applies to CosyVoice's HiFTGenerator, whose ResBlocks run at 64-256 channels
over lengths up to ~316 k, and its upsampling decoder went **454.0 → 349.7 ms**, a 1.30x, at
matched canaries of 59.9 and 59.3 ms. Worth noting how nearly that win was missed: the
end-to-end vocoder RTF barely moved (0.261 → 0.267) because the vocoder is only 16% of that
engine's runtime and the whole run happened to be 3% slower from mild throttling. The stage
benchmark found it; the end-to-end number hid it.

### Two routes that are faster still, and both refuted

- **`index_select` with a precomputed index** is 9.51x on the gather — ~81 GB/s, essentially
  bus-limited. But the index is `u32` and as large as the matrix it addresses: 352 MB for this
  one shape, and the codec has around a dozen distinct (channels, length, dilation)
  combinations. Caching them all is gigabytes.
- **Generating that index on device** from a broadcast sum of three `arange`s, so nothing is
  cached, measures **0.46-0.64x** across the four stages. Producing 352 MB of indices costs
  more than the faster gather saves.

### What this says about the earlier refutations

Finding 1's *measurement* was right — that im2col was slower — and its **explanation was
wrong**, which is what made it look final. A traffic-bound conclusion implies no layout can
help, so the search stopped. A gather-bound conclusion says the opposite: pick a better gather.
The `taps` and `chunk 8k` rewrites in the original table are also gather-order variants, which
is why they clustered at 0.73-0.86x rather than pointing anywhere.

The general lesson is narrower than "measure": it is that a refutation is only as strong as its
*attribution*, and an explanation that does not survive an order-of-magnitude sanity check
should not be allowed to close a line of investigation.
