# The AR loop — measured, and it changes the whole picture

All numbers here follow the `../benchmarking.md` protocol: variants interleaved
in one process, median of 5–7 samples, a fixed canary conv before and after to
record thermal state. Every table below reports its drift; all were stable
(≤1.08×). These are the first trustworthy numbers in the project.

Probes: `arloop`, `dispatch`, `quant`, `matvec`, `chain` (Rust),
`references/audio8/bench_ar.py` (torch). Quality: `qroundtrip` + `references/audio8/quality_ar.py`, written
up in [quantization-quality.md](quantization-quality.md).

One caution carried through the whole document: the `quant` run was hot (canary
83–108 ms) and `matvec`/`chain` were cool (59 ms). Interleaved ratios from the hot
run are still valid; its absolute microseconds are ~2.5× inflated and are flagged
where they appear.

## Headline: the AR loop is 10–20× the codec, and torch is the slow one

Same work both sides — 64 prompt tokens of prefill, then 64 frames = 2.97 s of
audio at 21.53 Hz, batch 1, driving `_slow_step` and `_generate_codebooks`
directly so no sampling or EOS logic perturbs the comparison.

| | slow AR | fast AR | total | AR-only RTF |
|---|---|---|---|---|
| **torch / mps f32** | 6753 ms | 14414 ms | **21167 ms** | **7.12** |
| torch / mps f16 | 3744 ms | 14307 ms | 18051 ms | 6.07 |
| **candle / metal f32, as the reference is written** | — | — | **31730 ms** | 10.68 |
| **candle / metal f32, three exact levers applied** | — | — | **11490 ms** | **3.87** |
| candle, + q8_0 weights, batch 1 (projected, cool) | — | — | **~782 ms** | **0.263** |
| candle, + q8_0 weights, batch 8 (projected, cool) | — | — | **~251 ms** | **0.084** |

Two things fall out immediately.

**The codec was never the problem.** It is ~990 ms in candle and ~570 ms in torch.
The AR loop is 21 seconds in torch. Every optimization discussed in
`candle-on-metal.md` was aimed at 4% of the runtime.

Torch's fp16 run is the one number here that failed the protocol: 232–262% spread
and 1.37× canary drift, so treat RTF 6.07 as "roughly 6, ±a lot". fp16 halves the
slow AR (6753 → 3744 ms) and does nothing at all for the fast AR
(14414 → 14307 ms), which is where the time is. It does not change the conclusion.

**Candle already beats torch on the AR loop, by 1.84×** (f32 vs f32; 1.57× against
torch's fp16), before any weight
quantization — and torch has no equivalent lever available on MPS. The
performance premise that `../rejected/coreml-and-op-coverage.md` recorded as "refuted" was refuted only
for the codec. For the half that dominates, Rust wins outright.

## Why: it is a memory test, not a compute test

`dispatch` isolates issue cost from work by timing 4000 repetitions of a single op.

| op | per-op | FLOP | achieved |
|---|---|---|---|
| `sqr [1,896]` | 9.4 µs | 0.9 k | — (pure issue cost) |
| `add [1,896]` | 9.2 µs | 0.9 k | — |
| `matmul [1,896]·[896,896]` | 144 µs | 1.6 M | 11 GFLOPS |
| `matmul [1,896]·[896,4864]` | 313 µs | 8.7 M | 28 GFLOPS |
| `matmul [1,4864]·[4864,896]` | 601 µs | 8.7 M | 14 GFLOPS |

Candle's op-issue overhead is **9.4 µs** — small, and not the bottleneck. (Its
Metal backend takes a mutex, builds a fresh `MTLComputeCommandEncoder` per op and
commits the command buffer every `CANDLE_METAL_COMPUTE_PER_BUFFER`, default 50.
That machinery costs ~9 µs and can be ignored.)

The matmuls are the story, and they are nowhere near compute-bound: 11–28 GFLOPS
against ~4 TFLOPS of fp32 peak. At batch 1 every weight element is read once and
used once, so the floor is the weight read, and **the AR loop is a bandwidth test.**
(These f32 figures were taken hot; do not compute GB/s from them. The cool
quantized measurements further down show candle reaching the bus, so the gap here is
f32's 4 bytes per weight, not a bad kernel.)

Per token the slow AR reads 24 × 14.9 M params, the head reads another 139.6 M,
and the fast AR reads 40 × 14.9 M. In f32 that is ~4.5 GB **per frame**. At
21.53 frames/s, real time demands 46 ms/frame; the f32 weight read alone needs
37 ms at full bus speed. f32 batch-1 decode of this model cannot be fast on any
framework. That is why torch is at RTF 7 too.

## The fast AR is the larger half, and nothing had counted it

`n_fast_layer` is 4, which reads small next to 24 — but the fast AR runs all four
layers once per codebook, ten times per frame. **40 layer-passes per frame versus
the slow AR's 24.** Measured, it is worse than that ratio in both frameworks
because each pass is smaller and less efficient:

- torch: fast 14414 ms vs slow 6753 ms — **2.13×**
- candle: fast 8555 ms vs slow 2935 ms — **2.91×**

The fast AR is 62–68% of the AR loop. It had appeared in no estimate in this
project before now.

## Three exact levers, measured (slow AR, 128 steps, interleaved)

| variant | median | vs reference |
|---|---|---|
| reference: 2048-wide KV, `repeat_interleave`, 155776 logits | 22207 ms | 1.00× |
| + narrow KV to `0..pos` | 4350 ms | **5.10×** |
| + GQA by reshaping the query instead of copying K/V | 3691 ms | 6.02× |
| + logits sliced to the 4097 reachable rows | 2935 ms | **7.57×** |

**Narrow KV is worth 5.1× on its own.** The reference allocates the cache at
`max_seq_len` and attends over all 2048 positions with a mask every step, so at
frame 64 it does 16× the attention work it needs. This is by far the biggest of
the three and it is pure waste, not a tradeoff.

The other two are smaller but exact and free: skipping `repeat_interleave` (which
materialises a 7× copy of K and V every layer every token) is 1.18×, and slicing
the logit head is 1.26× of the whole loop.

### The logit slice, measured properly: 15–62×, not 38×

`../rejected/coreml-and-op-coverage.md` projected 38× from the MAC ratio. Measured:

| head | per token |
|---|---|
| f32, full 155776 | 9.45 ms |
| f32, sliced to 4097 | 0.33 ms → **29×** |
| q4_0, full | 3.98 ms |
| q4_0, sliced | 0.115 ms → **62× vs f32 full** |

The MAC-ratio projection was in the right neighbourhood. The claim stands and
gets *better* under quantization, because the sliced head is small enough to
become issue-bound while the full head stays bandwidth-bound.

## The big lever: quantized weights

Because the loop is bandwidth-bound, bytes-per-weight is the whole game — and this
needs no hand-written MSL. Candle ships ggml-style quantized matmul with a
dedicated matrix-vector Metal kernel (`quantized/metal.rs::fwd_mv`, taken whenever
`dim(-2) == 1`, which is exactly a decode step).

One layer's five projections (14.9 M params), batch 1:

Interleaved, so the **speedup column holds**; the absolute µs were taken hot and
are ~2.5× inflated (`chain` later measured q8_0 at 191 µs cool). The GB/s column
that used to be here was computed from those inflated times and was wrong — removed.

| weights | B/param | µs / layer (hot) | MB read | speedup |
|---|---|---|---|---|
| f32 dense | 4.00 | 1607 | 59.6 | 1.00× |
| f16 dense | 2.00 | 1336 | 29.8 | 1.20× |
| **q8_0** | 1.06 | 479 | 15.8 | **3.35×** |
| q5_0 | 0.69 | 386 | 10.3 | 4.16× |
| q4_1 | 0.62 | 338 | 9.3 | 4.76× |
| q4_0 | 0.56 | 369 | 8.4 | 4.35× |

Projected over the full loop for 2.97 s of audio:

| weights | slow | fast | total | AR RTF |
|---|---|---|---|---|
| f32 | 2499 | 4115 | 6614 | 2.23 |
| f16 | 2082 | 3419 | 5501 | 1.85 |
| q8_0 | 743 | 1227 | 1970 | 0.66 |
| **q4_1** | 526 | 864 | **1390** | **0.47** |

**AR RTF 7.12 (torch) → 0.47 (candle, q4_1). 15×.** That is the number that makes
a Rust port compelling on speed rather than only on memory.

### Two constraints found while measuring this

**The K-quants are unavailable for this model.** Q4K/Q5K/Q6K use 256-element
blocks, and every projection with `k = dim = 896` fails — 896 is 3.5 × 256. Only
the legacy block-32 types (q4_0, q4_1, q5_0, q8_0) apply. They give up some
quality per bit, so nobody should plan around q4_K here.

**f16 dense is the worst performer per byte read** — 1.20× for half the bytes,
where q8_0 gets 3.35× for a quarter. Candle's f16 GEMV is weak. Consistent with the
codec finding: f16 is not a lever in candle, quantization is.

## Remaining headroom — a custom matvec kernel is NOT the answer

The first version of this document concluded from the per-layer table above that
candle was achieving 22–37 GB/s against a ~120 GB/s bus, and that a custom matvec
kernel was worth ~3.5x. **That was wrong, and the error was mine: those per-layer
numbers were taken on a hot machine** (canary 83–108 ms), so the GB/s computed from
them is meaningless. `matvec` and `chain`, both run cool (canary 59 ms, 1.00x drift):

| q8_0 matvec | us / call | MB | GB/s |
|---|---|---|---|
| `[896, 896]` | 7.1 | 0.85 | **121** |
| `[1152, 896]` | 7.9 | 1.10 | **139** |
| `[4864, 896]` | 38.7 | 4.63 | **120** |
| `[9728, 896]` | 82.6 | 9.26 | 112 |
| `[19456, 896]` | 186.8 | 18.52 | 99 |

**Candle's quantized matvec already saturates the bus.** There is no 3.5x hiding in
a better kernel and no reason to write MSL for this. Two follow-on hypotheses died
with it:

- **Fusing `w1` and `w3`** into one `[2*ffn, dim]` weight — they share an input, so
  it is exact and halves the dispatches — is **0.94x**, slightly *slower*.
- **The serial dependency chain costs nothing.** A real decode step
  (`wqkv -> wo -> {w1,w3} -> w2`, each waiting on the last) runs at 190.6 us/layer
  against 188.5 us for the same five matvecs with no dependencies between them:
  **1.01x**. Per-op latency is not exposed, so there was nothing for fusion to
  recover.

Measured cool, one q8_0 layer at batch 1 is **191 us**, not the 479 us the hot run
reported. That correction alone moves the batch-1 projection to **AR RTF 0.263**.

## The lever that does work: batching

| batch | us / layer | us / seq / layer | vs batch 1 | per-seq AR RTF |
|---|---|---|---|---|
| 1 | 191 | 191 | 1.00x | 0.263 |
| 2 | 479 | 239 | 0.80x | 0.330 |
| 4 | 487 | 122 | 1.57x | 0.168 |
| 8 | 490 | 61 | 3.12x | **0.084** |
| 16 | 497 | 31 | 6.15x | **0.043** |
| 32 | 511 | 16 | **11.95x** | **0.022** |

A batched layer costs essentially what a batch-2 layer costs, all the way out to 32,
so per-sequence cost falls almost linearly. At batch 32 the layer does 954 MFLOP in
511 us — **1867 GFLOPS**. It has stopped being a memory test and become
compute-bound, which is the right place to end up.

**One trap: batch 2 is a per-sequence regression** (0.80x). At `dim(-2) == 1` candle
takes the dedicated `fwd_mv` matvec path; at batch >=2 it falls back to the general
quantized matmul, which costs 2.5x more and then stays flat. The port should run at
batch 1 or batch >=4, never 2 or 3.

It is also the lever with the least risk attached: batching changes no weights and no
arithmetic beyond accumulation order, and long-form narration already batches
segments through `/v1/tts-jobs`.

### Implemented — and the 11.95× did not survive contact

`generate_batch` is in `crates/audio8/src/ar.rs`, wired into the engine with width bucketing and
`--set max_batch=<n>`. **The per-sequence gain is ~1.9×, not 11.95×**, and the difference is
the most useful thing this section has to say.

Measured on the real loop (`tts-probe --bin arbatch`, canary-stable, sampling on):

| batch | ms/step | ms/step/seq | per-sequence gain | break-even `sum/max` |
|---|---|---|---|---|
| 1 | 21.06 | 21.06 | 1.00× | 1.00 |
| 2 | 31.12 | 15.56 | **1.35×** | 1.48 |
| 3 | 41.64 | 13.88 | 1.52× | 1.98 |
| 4 | 51.22 | 12.81 | 1.64× | 2.43 |
| 8 | 90.11 | 11.26 | 1.87× | 4.28 |
| 16 | 167.55 | 10.47 | **2.01×** | 7.96 |

Two corrections to what the layer benchmark above concluded.

**The 11.95× was measuring one layer, not one step.** A layer's five projections do scale
almost freely with batch, because at `dim(-2) == 1` every weight element is read once and the
arithmetic units idle. But a decode *step* is 64 layer-passes plus ten host synchronisations
plus eleven host-side sampler calls per sequence — and the sampler amortises nothing, because
it runs once per sequence per codebook. That is what flattens the curve at ~2×. A layer
benchmark cannot see it.

**Batch 2 and 3 are not per-sequence regressions.** The layer measurement said batch 2 was
0.80×, because candle drops out of its matrix-vector kernel; on the real loop batch 2 is a
**1.35× gain**, because everything else in the step is shared. `plan_batches` originally
refused to emit a group of 2 or 3 on the strength of the layer number; that rule was wrong and
has been removed. Both measurements are correct about what they measured — only the
loop-level one should drive scheduling.

### The break-even column is the thing to design against

Batching decodes `max(frames)` steps where sequential decodes `sum(frames)`, so

```
end-to-end gain  =  (sum(frames) / max(frames))  x  (cost_1 / cost_b)
```

and the right-hand factor is fixed by the table above. Hence the last column: batch `b` only
pays if the group's frame counts are uniform enough that `sum/max` beats it.

This is not academic. The first wiring put all seven segments of `examples/senior.txt` in one
group of 8, where `sum/max` sits right at the 4.28 break-even — and measured **1.008×**,
almost exactly the wash the arithmetic predicts. Ragged lanes are the whole story: the longest
segment sets the step count for everyone.

Two things follow, and both are in the engine:

- **Sort segments by prompt width before grouping**, so lanes in a group finish at similar
  times.
- **Shed finished lanes off the tail.** Lanes are ordered longest-prompt-first, so the
  sequences that finish early collect at the end of the batch and the live width can shrink
  with a `narrow(0, 0, live)` — which shares storage, so it costs nothing. That matters
  because the alternative, compacting an interior gap, means rebuilding ~96 MB of KV cache
  per compaction. `slice_set` through a prefix view writes to the parent's storage, which is
  what makes the free version possible.

### A measurement I got wrong twice, and how

Worth recording as method rather than result. The first end-to-end sweep ran `tts speak` once
per `max_batch` and compared wall clock. It reported **RTF 0.946 for `max_batch 2`** — worse
than sequential — and in the same run showed the *codec*, which no part of the change touches,
29% slower than in the previous run. That second number is what gave it away. The canary read
**204 ms against 60 ms cool**: the machine was 3.4x throttled by the preceding jobs, and the
comparison was measuring temperature.

A follow-up probe run then drifted 0.29x *within itself* — batch 1 sampled hot, batch 4 and 8
sampled cool — and produced a flattering 3.52x that was equally meaningless.

Consecutive heavy runs cannot be compared on this machine, which is the whole point of
`../benchmarking.md`, and a sweep of separate CLI invocations cannot honour it. The
schedule is therefore measured by `tts-probe --bin arschedule`, which interleaves the
`max_batch` variants round by round inside one process and reports the canary at both ends.

Both of the wrong readings pointed the same way — *batch 2 is best, batch 8 is a wash* — and
both were artifacts. Interleaved, the truth is monotone:

| `max_batch` | median | vs sequential |
|---|---|---|
| 1 (sequential) | 31.94 s | 1.00× |
| 2 | 24.10 s | 1.33× |
| 4 | 19.51 s | 1.64× |
| **8** (one group) | **18.04 s** | **1.77×** |

Canary 59.99 ms at the start, 65.77 ms at the end — 1.10× drift, so the ratios stand. Sample
spread is 18-42%, which is high, but these are 18-32 s runs and the ordering is consistent
across every round.

The frame counts for those seven segments are `[53, 83, 264, 167, 110, 199, 187]`: sum 1063,
max 264, `sum/max = 4.03`. That is *below* batch 8's 4.28 break-even — which is exactly why the
first attempt was a wash, and exactly what shedding finished lanes fixes. `max_batch` therefore
defaults to 8.

### What is exact, and what is not

Two properties are asserted by `audio8-validate`:

- **A batch of identical prompts reproduces the unbatched result exactly.** This is the batch
  axis alone, with no padding involved.
- **A batch of four different-width prompts reproduces it exactly too — with f32 RoPE
  tables.** This is the right-alignment logic alone.

Under the *real* bf16-rounded tables, mixed-width batching diverges: 2 of those 4 sequences
stay identical and the others agree for a few frames and then part. That is expected and the
gate reports rather than asserts it. The reason is precise: right-alignment is exact because
`q_p . k_j` depends only on `p - j`, but `R(p)` and `R(j)` are rounded to **bf16
independently**, so `R(p)^T R(j)` equals `R(p - j)` only to about 4e-3 — an 8-bit mantissa.
Shifting a sequence's positions therefore perturbs its attention scores at that level, and
under greedy decoding a perturbation that size can flip a near-tie and diverge from there.

The bf16 rounding is not optional — the reference builds its table with
`torch.polar(...).to(bfloat16)` and it is part of the model's arithmetic, so `with_f32_rope`
exists only to make the claim above testable. A perturbation of 4e-3 is the same order as the
q8_0 weight error of 5.5e-3, which `quantization-quality.md` measured as costing nothing audible.

So the batched render is checked the same way rather than assumed:

| render | median F0 | ΔF0 vs reference clip | LTAS cos | WER |
|---|---|---|---|---|
| sequential (`max_batch 1`) | 177.8 Hz | −2.0 | 0.9969 | 0.008 |
| `max_batch 2` | 172.0 Hz | −7.7 | 0.9968 | 0.015 |
| `max_batch 4` | 179.8 Hz | +0.0 | 0.9971 | 0.008 |
| `max_batch 8` | 173.9 Hz | −5.9 | 0.9971 | 0.008 |

Every batched render sits inside the spread that changing the sampling seed produces, and two
of the three match the sequential WER exactly (the third is one word in 133). The batched path
is a different draw, not a worse one.

### The other non-obvious part: why right-alignment and not left

What blocks a batched version is not the transformer, which mostly already carries a `b`
dimension, but the fact that **segments have different prompt widths**, and the obvious fixes
are wrong:

- *Left-pad to a common width and mask.* Wrong: it shifts every real token's RoPE position
  by a different amount per sequence, so each sequence sees a different model.
- *Bucket by identical width.* Widths are essentially never equal.
- *Prefill separately, then batch the decode loop.* Wrong as written: each sequence's cache
  is then filled to a different depth, and `cache.k.narrow(2, 0, attend)` has one `attend`.

The form that **is** exact is to **right-align the prompts** — pad on the left so every
sequence's last prompt token sits at the same index — and mask the leading pad columns. The
argument for exactness is worth stating because it is the whole reason this works: RoPE
rotates `q` and `k` by absolute position, but the attention *score* `q_p . k_j` depends only
on `p - j`. Right-alignment shifts every position within a sequence by the same constant, so
every difference is unchanged, so every score is unchanged. `v` is never rotated. The result
is identical up to floating-point reassociation.

The rest was bookkeeping, and all of it turned out to be needed:

1. `Cache::new(capacity, batch)`, and `slice_set` per batch.
2. `embed` over a `[batch, len]` id grid rather than one row set — still one gather for the
   text row and one for all ten codebooks, so a batch of 8 costs the same dispatches as 1.
3. `layer_forward`'s `t == 1` fast path assumes `b == 1` — it flattens `qkv` and narrows
   dimension 0 by absolute offsets, which no longer names q, k and v once lanes interleave.
   Kept for `b == 1`, general path otherwise.
4. A `[b, 1, t, t]` prefill mask with the pad columns at `-inf`, not the shared `[t, t]` one,
   plus a `[b, 1, 1, capacity]` decode mask narrowed each step. The batch-1 path passes `None`
   for both, so it is byte-for-byte the loop it always was.
5. Per-sequence EOS with a `done` flag; finished lanes carry inert filler.
6. A batched fast AR — 10 positions x batch, with per-sequence sampling.

**One trap inside the mask that cost a debugging pass.** Padded *rows* must not be fully
masked. A row whose every column is `-inf` softmaxes to NaN, and that NaN lands in the padded
position's hidden state, becomes a padded K/V entry in the next layer, and then poisons every
real row through `NaN + -inf = NaN`. Masking a position out of attention does not stop it
being *computed* — only from being read. Padded rows are therefore given ordinary causal
visibility and their garbage output ignored.

And one behavioural consequence to document rather than hide: per-sequence sampling from a
batched step consumes the RNG in a different order than the sequential path, so batched and
unbatched renders of the same text with the same seed differ. Both are valid draws; they are
not the same audio.

### A sampler cost worth knowing about

Chasing the flat part of the batch curve turned up something independent of batching. The
sampler ran **eleven times per frame per sequence** — two RAS draws for the semantic token,
nine for the residual codebooks — and each call did a full `n log n` sort of ~4096 indices and
then drew ~4096 uniforms and took ~4096 logarithms.

Both are avoidable exactly:

- **Only the top `top_k` ranks can survive**, since the reference removes everything at
  `rank >= top_k` unconditionally. So a linear-time selection plus a 50-element sort replaces
  a 4096-element sort — the same function.
- **Gumbel-max need only draw for entries that can win.** After the top-k/top-p filter all but
  at most `top_k` entries are `-inf`, hence probability exactly zero, hence `p / -ln(u) == 0`
  whatever is drawn. Skipping them selects from an identical distribution.

Measured, this is worth only ~3-4% of a step (21.81 → 21.06 ms at batch 1), so the estimate
that led here — that the sampler was a quarter of a batch-8 step — was too high. It is kept
because it is exact and strictly less work, and recorded because the reasoning was sound and
the conclusion was still wrong: 64 layer-passes and ten host synchronisations per step
dominate, and no amount of sampler tuning changes that.

## Which puts the codec back on the critical path

With the AR loop at RTF 0.084 (q8_0, batch 8) and the codec at ~0.33, **the codec is
now ~80% of the runtime.** The work in `candle-on-metal.md` — alpha folding, the
fused residual unit, and above all candle's 6–8x deficit on low-channel convs —
matters again. It was not wrong, only premature.

## What this reorders

1. **The three exact levers** — narrow KV (5.10×), GQA by query reshape, the logit
   slice. Zero numerics risk; this is simply how the port must be written.
2. **q8_0 weights.** 3.35×, and measured to cost nothing audible — see
   [quantization-quality.md](quantization-quality.md). Not q4: q4_1 is only 1.42× beyond
   q8_0 and it does degrade.
3. **Batching at ≥4.** Up to 11.95× per sequence, no numerics change, and it
   matches the actual long-form workload. Never batch 2 or 3.
4. **Codec work** — α folding, the low-channel conv deficit, the fused residual
   unit. Back on the critical path now that the AR loop is ~20% of the total.
5. ~~A matvec Metal kernel~~ — **refuted.** Candle's matvec already saturates the
   bus at 99–139 GB/s.

The codec conclusions in `candle-on-metal.md` are not wrong. They were aimed at
the wrong half at the time, and are now aimed at the right one again.

## Footnote: the canary revised the codec story too

The canary conv (96ch @ 131072, k7, dilation 9) runs in **~10 ms under torch** and
**60–108 ms under candle**. Candle's conv1d is not ~2× off torch's as
`../rejected/coreml-and-op-coverage.md` concluded from whole-codec timings — on the low-channel,
long-length stages it is **6–8× off**, which is exactly where the codec spends its
time (`convopt.rs` measured candle at 1422 GFLOPS at 768ch but 283 at 96ch; torch
holds ~1690 GFLOPS at 96ch). The codec gap is narrower than it looks at high
channel counts and much wider at low ones.
