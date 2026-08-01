# Read this before trusting any number in the other documents

All absolute timings in `rejected/coreml-and-op-coverage.md`, `rejected/onnx.md` and
`performance/candle-on-metal.md` were taken on a cool machine early in a long session.
By the end of that session the same code on the same shapes was **~2× slower**.
This was caught by accident: `dilation.rs` reported 119.57 ms for a conv that
`convopt.rs` had measured at 59.78 ms twenty minutes earlier.

## The drift, measured

Interleaved A/B, same 64 frames, back to back:

| | cool | throttled r1 | throttled r2 |
|---|---|---|---|
| torch mps f32 | 568.9 | 675.8 | 853.8 |
| torch mps fp16 | **403.6** | 731.3 | 765.6 |
| torch mps bf16 | 406.3 | 733.5 | 768.1 |
| torch cpu f32 | 1199.0 | 2499.4 | 2596.3 |
| candle metal f32 | 990.0 | 1505.1 | 1451.3 |
| candle metal f16 | 888.3 | 1308.4 | 1331.8 |

Cause is almost certainly thermal: an M4 in a 16 GB machine after a sustained run
of GPU benchmarks plus several 19–26 s CoreML compilations.

## What survives and what does not

**Survives — ratios measured within a single process run:**

- α folding gives 2.99× on snake (`fusion.rs`, one run) — and the algebra check is
  exact regardless of clock speed.
- Snake is unfused, with an 8.7× single-pass ceiling (`fusion.rs`, one run).
- `broadcast_mul` is 3.6× slower than a unary op (`fusion.rs`, one run).
- Candle conv1d spans 1422 → 283 GFLOPS from 768ch to 96ch (`convopt.rs`, one run).
- conv-as-GEMM is 0.73–0.86× of direct, i.e. slower (`convopt.rs`, one run).
- De-interleaving dilation is exact but 0.67–0.85× of direct (`dilation.rs`).
- Dilation itself costs nothing: d=1 vs d=9 within 0.95–1.06× (`dilation.rs`).
- **candle is 1.70–2.23× slower than torch on this codec**, stable across all
  thermal states.

**Does not survive:**

- Every absolute millisecond figure. Treat them as upper-bound-quality-only, cool
  machine, single sample.
- **"torch gains 29% from fp16."** Under thermal load fp16 became *slower* than
  fp32 (731 vs 676 ms). The fp16 win was a cool-machine artifact, so the ×0.75
  fp16 step in the 324 ms projection in `performance/candle-on-metal.md` is unsupported.
- The 403.6 ms "target to beat" — that is torch's best case, not its steady state.
  A steady-state target is somewhere in the 675–855 ms range, which ironically
  makes it *easier* to beat, but nothing here established a trustworthy figure.

## Protocol for any future measurement

1. **Interleave A and B in the same run**, alternating, never A-then-B-much-later.
2. Report **median and spread of ≥5 samples**, not a 3-iteration mean.
3. Include a **fixed canary workload** in every run (e.g. the 96ch@131072 conv) so
   the thermal state of that run is recorded alongside the result.
4. Prefer **ratios within a run** over absolute numbers across runs.
5. Idle-cool between runs, or accept and report the drift.

A 2× thermal drift will swamp any optimization worth less than 2×. Until the
harness does the above, **we cannot evaluate an optimization on this machine** —
we would be tuning against noise. Building that harness comes before the next
optimization, not after.

## The harness now exists

`crates/tts-probe/src/bench.rs` (`Harness`) implements all five points:
interleaved variants in one process, median and spread of ≥5 samples, the fixed
canary conv timed before and after with the drift reported, and a ratio column
against the first variant. `references/audio8/bench_ar.py` mirrors it in Python with the same
canary, so Rust and torch runs are placed on the same thermal scale.

Every number in [performance/ar-loop.md](performance/ar-loop.md) came through it. Those
runs held at ≤1.08× drift, which is why that document's absolute figures are
quotable and this one's are not. The one exception is noted there: torch fp16
drifted 1.37× with a 262% spread and is reported as approximate.

Two habits the harness made visible and worth keeping:

- **Quote the canary next to any millisecond figure.** The same conv measured
  7.8–14.6 ms under torch and 60–108 ms under candle in the same session. Without
  the canary those runs look comparable; with it, the 6–8× conv gap is obvious.
- **Watch the spread, not just the median.** Short variants sitting next to long
  ones in the same interleave pick up 300%+ spreads (see the tiny-op rows in
  `dispatch`). The median is still usable; the resolution is not.


## Two more ways to get it wrong, both learned the hard way

The harness solves *within-run* comparison. Two failure modes sit outside it, and the batching
work walked into both.

**A sweep of separate CLI invocations is not a comparison.** Timing `tts speak` once per
configuration and comparing wall clock reported `max_batch 2` as the best setting and
`max_batch 8` as a wash. Interleaved inside one process, the truth is monotone and `max_batch 8`
is the fastest by 1.77x. The sweep was measuring how hot the machine had become: the canary read
**204 ms against 60 ms cool**, a 3.4x throttle accumulated over the preceding runs.

The tell was not the number under test. It was that the *codec* — code the change does not touch
— appeared 29% slower in the same run. **A stage you did not modify moving is the cheapest
throttling detector available**, and it is worth printing a stage breakdown for exactly that
reason.

**Never run two GPU jobs at once.** Launching an Audio8 render in the foreground while a
CosyVoice render ran in the background produced RTF 1.287 for a configuration that measures
0.599 alone — a 2.1x error, in the same direction and of the same size as a genuine regression.
Both numbers were discarded. Background execution is convenient for waiting; it is not
convenient for measuring.

The general rule both cases point at: **an absolute timing is only meaningful with a statement
about what else the machine was doing.** The harness supplies that for interleaved runs. For
anything else it has to be supplied by hand — check the canary, and check that an unmodified
stage did not move.


## An unsynchronised stage timer measures the wrong stage

This one invalidated a whole profile. Metal dispatch is asynchronous: `hift.forward` enqueues
work and returns, so a timer that stops on return records enqueue time, not execution time —
and the work drains inside whatever is timed *next*. CosyVoice's engine timed its three stages
that way, and the breakdown it produced was wrong in both directions:

| stage | as reported | with `device.synchronize()` before recording |
|---|---|---|
| llm | 0.218 | 0.226 |
| flow | 0.419 (52%) | **0.570 (68.5%)** |
| vocoder | 0.141 (17%) | **0.037 (4.4%)** |

The vocoder was never 17% of CosyVoice. It was 4.4%, and it was being billed for the flow's
drain. Two sessions of vocoder optimisation — the im2col kernel, the fused snakes, the `ups`
convs — were aimed at a stage with almost nothing in it, which is exactly why they kept
"failing to show up end-to-end".

The tell was available the whole time and was misread: a standalone `hift.forward` at the
engine's length measured 2.1 s against the engine's reported 7.6 s. That gap was attributed to
cold-path allocation (see below) rather than to the timer. **When a stage's standalone cost and
its in-pipeline cost disagree, check that the timer synchronises before believing either.**

## A warm A/B loop cannot see first-touch allocation cost

The harness times each variant in a loop, which is what makes its ratios trustworthy — and
also what makes it blind to anything that only happens on the first call. Candle's Metal
device pools buffers by size, so a loop that runs the same shape five times allocates once
and recycles four times. A pipeline stage called **once per utterance** never gets that.

CosyVoice's vocoder looked like exactly that case, and **it was not** — the gap was the
unsynchronised timer above. Running the vocoder in its own process and timing the very first
call after load gives **2162 ms cold against 2054 ms on the third call**: a 5% first-touch
cost, not 2.2×. Running a full flow solve first, to fill the pool with unrelated buffers,
changes nothing (2115 ms). Pre-warming with a throwaway pass at the same shape changes nothing
(2050 ms).

So candle's buffer pool is not a hazard here, and the concern is withdrawn. The general
warning stands for genuinely once-per-process work, but it was not what was happening.
Audio8's codec keeps its warm-loop win in full either way (RTF 0.260 → 0.158, verified against
a steady AR stage).

Four hypotheses were raised and refuted before the real cause was found, all by measurement
rather than argument: that the im2col matrices were too large (they are not — the GEMM wins
at 1.1 GB), that `Noise::Draw`'s host-side RNG was the cost (17 ms of 3.45 s), that the
buffer pool was the cost (5%, above), and that the vocoder mattered at all (4.4%). Each was
plausible and each was wrong, and the cost of not testing them one at a time would have been
a fix aimed at the wrong thing.

The general rule: **when an isolated win does not appear end-to-end, the measurement is the
first suspect, not the change.**


## Sampled output cannot tell "different draw" from "different model"

Batching CosyVoice's LLM across segments changed the audio, and the change looked like a
regression: across four seeds, median F0 sat about 4 Hz lower than the unbatched path, with
the two ranges barely overlapping ([-7.7, -4.0] against [-14.8, -7.7]). LTAS was identical
either way (0.9977-0.9985) and WER was equal or better, but a consistent shift in a
speaker-similarity metric is not something to wave through.

It was not a regression. Batched lanes draw from the shared RNG interleaved one step at a
time, where the sequential path draws each segment's whole sequence consecutively — a
different sample, not a different distribution. The way to see that is to take sampling out:
with greedy decoding, **`llm_batch=1` and `llm_batch=7` produce byte-identical waveforms**.
A 2e-6 perturbation from right-alignment does not move an argmax.

Two lessons. **Any change that perturbs RNG consumption needs a deterministic mode to be
testable at all** — CosyVoice's `--greedy` had been silently ignored by the engine, so this
comparison was impossible until it was wired up. And **a metric shift across four samples is
not evidence of a regression** when the sampling stream changed; it is evidence that four
samples is not enough.
