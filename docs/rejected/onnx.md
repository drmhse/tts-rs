# ONNX / ORT results — the codec decoder

> **Absolute timings here are unreliable — see [../benchmarking.md](../benchmarking.md).**
> They were measured on a cool machine; the same code ran ~2x slower later in the
> session. Ratios measured within a single run still hold; absolute ms figures do not.
>
> **And this document is aimed at the wrong half of the pipeline.** The AR loop was
> later measured at 10-20x the codec's cost — see
> [../performance/ar-loop.md](../performance/ar-loop.md). Everything here is real, but it is
> optimising 4-8% of the runtime. Read that first.

Same 64 code frames = 2.97 s of audio at 44.1 kHz, same M4 / 16 GB, everything
measured rather than projected. ort 1.28.0, opset 17, fp32.

## Verdict: ONNX is the slowest option tested. Do not pursue it.

| runtime | ms | RTF | notes |
|---|---|---|---|
| **torch / mps fp16** | **403.6** | **0.136** | current best |
| torch / mps f32 | 568.9 | 0.191 | |
| candle / metal f16 | 888.3 | 0.298 | |
| candle / metal f32 | 990.0 | 0.332 | |
| torch / cpu f32 | 1199.0 | 0.403 | |
| ort / coreml ANE (static) | 1253.0 | 0.422 | 35 partitions, 19 s compile |
| ort / coreml CPUAndGPU (static) | 1305.7 | 0.439 | |
| ort / coreml ALL (static) | 1345.7 | 0.453 | 26 s compile |
| ort / cpu | 1496.3 | 0.503 | slowest |

Every ORT configuration loses to both Candle and PyTorch. ORT CPU is even slower
than torch's own CPU path (1496 vs 1199 ms).

## The export itself worked, and worked well

This is worth separating from the performance result: **the conversion is not the
problem.** Four patches were needed, all verified bit-exact (`max|diff| 0.000e+00`)
against the Phase A fixture before export:

1. `_arktts_snake` is `@torch.jit.script`ed — replaced with the identical plain
   function so the tracer sees through it.
2. `_rope` uses `torch.polar(...).real/.imag`. ONNX has no complex type, so this
   cannot export. `polar(1, θ) == (cos θ, sin θ)`, so the swap is exact. The
   reference then rounds the table to bf16 before multiplying against fp32
   activations; we reproduce that with a bf16→fp32 round-trip, keeping identical
   rounding without bf16 tensors in the graph.
3. `ArkttsCodecWindowTransformer` builds its window mask with in-place `mask &=`.
   `aten::__iand_` has no ONNX lowering — rewritten non-in-place.
4. `ArkttsCausalConv1d` computes right-padding via `_extra_padding`, which calls
   `math.ceil` on `x.shape[-1]`. Under tracing that **collapses to a constant for
   the export length** — the first export silently worked at 64 frames and threw
   shape errors at every other length. Every causal conv in the *decode* path has
   stride 1 (the strided k2/s2 convs are all encoder-side), and for stride 1 the
   padding is provably 0, so the whole computation can be dropped.

Trap 4 is the interesting one: it produces a model that passes a naive smoke test
and is silently useless. `export_codec.py` now ends with a **dynamic-length gate**
that runs the exported graph at 4/8/24/64/100/137 frames and refuses to ship if
any length breaks.

Numerics on the exported graph, validated at two different lengths under ORT CPU:

| fixture | frames | max abs diff | signal rms |
|---|---|---|---|
| `codec_syn` | 8 | 1.671e-05 | 9.197e-02 |
| `e2e` | 24 | 2.071e-06 | 2.246e-01 |

Relative error ~1.8e-4, which is ordinary fp32 conv-reordering noise.

## Why ORT is slow here: partitioning

CoreML cannot take the graph whole. From `GetCapability`:

| export | nodes | supported | **partitions** |
|---|---|---|---|
| dynamic frame axis | 1611 | 1358 | **108** |
| static (64 frames) | 951 | 892 | **35** |

Even in the best case that is 35 CoreML subgraphs per inference, each with a
boundary crossing on tensors up to 96×131072. The `Snake` activation sits between
every convolution in this decoder, and its `Sin`/`Reciprocal`/`Pow` chain is what
keeps fragmenting the graph — the failure mode predicted before running anything.

Corroborating evidence that boundaries dominate rather than compute: switching
`MLComputeUnits` across ALL / CPUAndGPU / CPUAndNeuralEngine moves the total only
from 1346 → 1306 → 1253 ms. Which compute unit runs the math barely matters when
the time is going into 35 handoffs.

## The dynamic-shape wall

With a dynamic frame axis, CoreML **cannot execute at all**:

- `MLProgram` + `ALL` or `CPUAndGPU`: hard process abort —
  `MPSGraphExecutable.mm:1232: failed assertion 'original module failed verification'`.
  Not a Python exception; it kills the interpreter, which is why each config had
  to be benchmarked in its own process.
- `MLProgram` + `CPUAndNeuralEngine`: compiles in 6.3 s, then fails at inference
  in CoreML subgraph #3 — `Unable to compute the prediction using a neural network
  model ... (error code: -1)`.
- `NeuralNetwork` format: 185 partitions, only 887/1611 nodes taken, also fails.

Only the **static** export runs. That means length bucketing at serve time, and
each bucket costs 19–26 s of CoreML compilation and its own compiled artifact.

## The irony worth recording

The abort is inside **MetalPerformanceShadersGraph**. CoreML is built on MPSGraph
— the same subsystem whose unbounded, shape-keyed, non-evictable compilation cache
produced the original 38.87 GB footprint and 40.9 GB of swap that this whole
architecture exists to escape.

So the ONNX/CoreML path does not merely fail to be faster: with static-shape
bucketing it would **reintroduce per-shape MPSGraph compilation**, i.e. precisely
the failure mode the disposable-worker design was built to contain, except now
behind an opaque provider boundary with no `phys_footprint` hook to police it.

## Untested variant, and why it is not worth testing

fp16 ONNX weights (via `onnxconverter_common.float16`) are CoreML's native
precision and would likely help more than fp16 helped Candle (+10%). But 35
partition boundaries are a *structural* cost that precision does not touch, and
the observed insensitivity to `MLComputeUnits` says boundaries are where the time
goes. Best case would be somewhere near 800–900 ms — still roughly 2× worse than
torch/mps fp16 at 403.6 ms. Not a path to a win.

## Where this leaves the port

Ranked by measured codec speed, the runtime options are:

1. **torch / mps fp16 — 403.6 ms.** Fastest, but it is the thing that leaks
   MPSGraph and requires the whole supervisor apparatus.
2. **candle / metal — 888–990 ms.** ~2.2× slower than torch's best, but no
   MPSGraph, no supervisor, single binary, flat memory, trivial streaming.
3. **ort — 1253–1496 ms.** Slowest, needs static-shape bucketing, and drags
   MPSGraph back in through CoreML.

ONNX is eliminated. The real choice is unchanged from `coreml-and-op-coverage.md`: pay
~2.2× on the codec to get a single-process Rust service with bounded memory by
construction, or keep torch's speed and keep the supervisor that speed requires.
