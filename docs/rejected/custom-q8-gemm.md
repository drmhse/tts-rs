# Rejected: a hand-written batched q8_0 GEMM

Built for `qwen3tts` and removed. It was correct — validated against candle's q8_0 at the
talker's and predictor's real widths for m = 1..8, on CPU and Metal — and it amortised the weight
read where candle's does not. It still lost to MPS f16 at every configuration tried.

## Why it looked necessary

Both transformers are bandwidth-bound on weight reads, so batching lanes is the only lever, and
no available option does both halves: candle's quantized `mm_t` reads the fewest bytes but
re-reads per row at m = 2..8, dense f32 amortises but 5.6 GB does not fit in 16 GB. A kernel that
dequantizes each k-block once and accumulates it against all batch rows does both.

## What was measured

Talker trunk, batch 8, ms/step (f16 for reference: **110 ms, 13.70 ms/lane**):

| variant | ms/step | ms/lane | amortisation |
|---|---|---|---|
| scalar, 1 column per lane | 213 | 26.6 | 1.56x |
| scalar, 4 columns per lane | **123** | 15.3 | 2.64x |
| scalar, 8 columns per lane | 415 | 51.8 | 5.67x |
| simdgroup 8x8, 32 cols/group | 179 | 22.4 | 4.58x |
| simdgroup 8x8, 8 cols/group | 131 | 16.3 | 2.95x |
| simdgroup 8x8, 16 cols/group, k-chunk 128 | 282 | 35.2 | 5.20x |

End to end the best of these rendered **RTF 0.866 against f16's 0.668**.

## Why it lost

The scalar version was compute-bound: one x load per FMA, ~184 GFLOP/s. Register tiling fixed the
ratio but 8 columns spills `acc[8][8]`.

Moving the arithmetic onto the 8x8 matrix units raised amortisation to 4.6-5.2x but not absolute
speed, because the two costs trade directly against each other:

- **occupancy** — one threadgroup per column tile, so 32 columns leaves only 64 threadgroups for
  n = 2048 and starves the GPU
- **barriers** — a threadgroup-memory staging tile needs two barriers per k-chunk, 192 of them at
  k = 6144
- widening the k-chunk to cut barriers grows the staging tile and costs occupancy again

Best case reached ~11.5 GB/s of weight traffic against a ~100 GB/s bus, so none of these were
near the bandwidth limit the design was aiming at. MPS wins by being a properly tuned GEMM —
double-buffered staging, tuned tile shapes per problem size — not by any single trick that could
be added here.

## What would have to be true

A competitive version needs double-buffered k-staging so the dequantize of chunk *i+1* overlaps
the matrix multiplies of chunk *i*, plus per-shape tile tuning. That is a GEMM project, not a
kernel. Until then f16 is the batching format: dense enough to amortise, small enough to fit.
