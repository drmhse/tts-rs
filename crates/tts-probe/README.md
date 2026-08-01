# tts-probe — one binary per question

Every optimisation in this repo was decided by one of these, and several were *reverted*
by one of these. They are kept runnable rather than summarised because a claim about this
hardware has a short shelf life: candle changes, macOS changes, and the machine's thermal
state changes within a single session.

All of them go through [`tts_bench::Harness`](../tts-bench/src/lib.rs), which interleaves
variants inside one process and prints a canary workload before and after. Read
[docs/benchmarking.md](../../docs/benchmarking.md) before trusting any absolute number
you get out of them.

```sh
cargo run -p tts-probe --release --bin <name>
```

Some need converted weights or dumped fixtures; see [docs/setup.md](../../docs/setup.md).

## The shape of the problem

| probe | question |
|---|---|
| `dispatch` | how much does a candle/Metal op cost to *issue*, independent of its work? |
| `fusion` | how much is candle losing to unfused elementwise kernels? |
| `chain` | is a serial dependency or bandwidth the limit? |
| `matvec` | why is quantized matvec stuck at ~33 GB/s on a ~120 GB/s bus? |

## Audio8

| probe | question |
|---|---|
| `arloop` | where does the DualAR loop's time go? |
| `arbatch` | does batching the real AR loop pay? |
| `arschedule` | does the batching *schedule* pay, on the real segment set? |
| `quant` | the AR loop is weight-bandwidth bound — does shrinking the weights help? |
| `qroundtrip` | dump quantized-then-dequantized weights, for the voice-quality check |
| `cascade` | the full codec decode cascade, with random weights |
| `codecsplit` | where the codec's time goes once its convs are GEMMs |
| `convgemm` | conv-as-GEMM, re-examined with the *other* im2col layout |
| `im2col` | can a custom Metal gather beat the `cat`-built matrix? (**yes, 4–6.5×**) |
| `snakefuse` | fused snake against the composed form (**1.4–6×**) |

## CosyVoice

| probe | question |
|---|---|
| `dit` | where does the flow decoder's time go, and is the matmul shape wrong? |
| `flowsplit` | the same, at the length the engine actually runs — the fixture is 4× too short |
| `ditbudget` | one DiT block, op by op, at `[2, 3192, 1024]` |
| `attnlayout` | strided views into `sdpa`, or a fast transpose first? (**strided wins**) |
| `llmbatch` | what does a decode step cost as the batch grows? |
| `hiftsplit` | where the vocoder spends its time |
| `upsconv` | does the GEMM route survive a 1.1 GB im2col matrix? (**yes**) |
| `coldpath` | is the vocoder's first call expensive? (**no — it was a timer bug**) |

## The ones that said no

Four of these produced a *negative* result that is more valuable than a speedup would
have been, because each closed off a plausible direction:

- **`attnlayout`** — a transpose kernel 4–7× faster than candle's makes attention 0.98×.
  `sdpa` takes strides; the existing lazy-view design was already right.
- **`ditbudget`** — fusing q/k/v into one projection saves 0.06 ms. It mattered in the
  LLM and does nothing here.
- **`upsconv`** — chunking a conv to bound a 1.1 GB intermediate costs more than the
  intermediate does.
- **`coldpath`** — the "cold allocation" that appeared to cost 2.2× was an
  unsynchronised stage timer billing one stage's GPU drain to the next.
