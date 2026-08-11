# Qwen3-TTS: where the time goes, and what batching is worth

`examples/senior.txt`, 50 s of audio, 7 segments, this Mac (M-series, 16 GB), synchronised timers.

## Result

| step | RTF | talker | codec |
|---|---|---|---|
| start | 0.876 | 0.696 | 0.181 |
| codec kernels | 0.788 | 0.707 | **0.079** |
| + quantized output heads | 0.751 | 0.669 | 0.079 |
| + folded `resize`, fused frame gather | 0.740 | 0.659 | 0.079 |
| + fused decode attention (q8_0, default) | 0.663 | 0.582 | 0.079 |
| + `--quant f16`, batch 7 | **0.407** | 0.316 | 0.079 |

**2.15x overall.** The codec went from 21% of the time to 19% of a much smaller total.

Codec **2.29x**, same fixtures (`codec.wav` rel 5.4e-5): `causal_conv1d_gemm` with tap-major
weights, `depthwise_k7` instead of candle's `groups == channels` conv1d, and a fused
`snake_beta` kernel replacing six composed ops.

## Cost is linear in weight bytes

| stage | q4_0 | q8_0 | f32 | fitted fixed cost |
|---|---|---|---|---|
| talker trunk (1.4 G params, read 1x/frame) | 11.1 | 18.6 | 68.9 ms | 1.7 ms |
| depth stack (60 M params, read **14x**/frame) | 17.1 | 22.9 | 65.5 ms | 9.2 ms |

The predictor costs more than the 28-layer trunk because of the re-reads, not its size. A frame
is 80 ms of audio, so the q8_0 bandwidth floor is ~30 ms/frame — RTF ~0.37 before the codec.
Reading the weights fewer times is the only lever, which means serving several lanes per read.

## Batching: cost per lane at batch 8, realistic cache span

Before fused decode attention:

| weights | talker b1 | talker b8 | per-lane | predictor b8 per-lane |
|---|---|---|---|---|
| candle q8_0 | 23.24 | 24.07 | **0.97x** | 2.23 (1.02x) |
| candle f16 | 47.55 | 13.70 | 3.47x | 1.87 (2.04x) |
| candle f32 | 61.13 | 8.21 | 7.45x | 0.64 (6.05x) |
| `tts_nn::qgemm` q8_0 | 40.50 | 15.32 | 2.64x | 1.94 (1.85x) |

After, with f16: **41.44 -> 44.73 ms/step from batch 1 to 8, i.e. 5.59 ms/lane and 7.41x** on the
trunk, and 0.446 ms/lane (6.70x) on the predictor. The KV copies were what capped amortisation:
they scale with the batch and never amortise, so f16's 3.47x was mostly them, not attention
arithmetic. Removing them turned batching from a 13% win into a 39% one.

1. **candle's quantized `mm_t` does not amortise** — it pads to a large row tile, so batch 8
   costs 8x batch 1. A batched q8_0 render measured RTF 1.02 against 0.79 unbatched.
   It *is* efficient at prefill widths (156 rows in 236 ms); the pathology is m = 2..8.
2. **f32 amortises and does not fit** — 7.45x per lane in isolation, RTF **6.25** in a real
   render at 6.97 GB resident. This is the wall that made q8_0 the default.
3. **f16 is the compromise that ships**: dense so it amortises, half of f32's bytes so it fits.
4. **A custom q8_0 kernel did not win, and was removed.** See
   `docs/rejected/custom-q8-gemm.md` — it amortised up to 5.2x where candle's does 1.0x, and was
   validated against candle's q8_0 at real widths for m = 1..8, but no configuration beat MPS
   f16 in absolute terms.

Span matters when benchmarking: f32's 7.45x per lane at a 4-position cache became f16's 3.47x
at 170 positions, because the attention work does not amortise across lanes at all. The bench
now defaults to the engine's real prompt length.

### Why grouping by length is enough

`Talker::generate_batch` needs one prompt length per batch. In the ICL path the prompt is
`role + prefix + max(text_len, codec_len)` and `codec_len` is fixed by the *voice*, so every
segment whose text fits inside the reference block (~106 tokens) is identical: all seven
segments of `senior.txt` are `[1, 156, 2048]` with a single `tts_pad` trailing. Anything longer
falls back to unbatched. A lane hitting `BATCH_FRAME_CAP` is rerun unbatched at the full budget,
so the cap bounds memory without truncating audio.

## Which format

- **one segment**: `q8_0` (default) — nothing amortises, fewest bytes wins, f16 is 2x worse.
- **long text**: `--quant f16`.
- **fixtures**: `f32`. The engine cannot choose: weights load before it sees the text.

## Next, by expected payoff

1. **`simdgroup_matrix` in `qgemm`** — the amortisation is proven, the arithmetic is not
   competitive. This is the only path to the ~0.37 floor.
2. **f16 KV cache** — halves the 229 KB/position that caps `MAX_BATCH`, now that the copies
   that dominated cache traffic are gone.
3. **q4_0** — measured cheap, but a quality trade on acoustic residuals. Offer, do not default.
4. **Batch above 8 lanes** — untested; this text only has 7 segments, so it needs longer input.

## Not applicable

**FlashAttention** proper avoids materialising a `[seq, seq]` score matrix; a decode step's
scores are `[heads, 1, span]`, and prefill is ~8% of this render. What *did* pay was the much
narrower fix in `tts_nn::attn` — reading the KV cache in place — which is a layout win, not the
tiled-softmax algorithm. **Speculative decoding** and
**Medusa-style heads** fit a bandwidth-bound AR loop but need a draft model over the codec
vocabulary or trained extra heads — neither ships with this checkpoint. (The depth predictor is
already multi-token prediction, but across codebooks within a frame, not across frames.)
**Continuous batching** helps `tts-serve` throughput, not single-utterance latency.
