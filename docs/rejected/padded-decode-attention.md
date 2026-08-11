# Rejected: attending over the whole KV cache instead of narrowing it

`narrow(2, 0, span)` of a `[b, n_kv, capacity, head_dim]` cache is not contiguous, so both uses
in a decode step copy it. Attending over all of `capacity` with the unwritten tail masked is
exact (`exp(-inf)` is 0, the tail of `v` is zeros) and makes the cache tensors usable directly.

It only removes one of the two copies: `k_all.transpose(2, 3)` is still strided, so the `k` copy
survives and now copies `capacity` rather than `span`. Measured on the predictor, whose cache is
17 positions and which runs 5 layers 15 times a frame:

```
narrowed   predictor 33.3 ms/frame   depth stack 22.9 ms
padded     predictor 32.9 ms/frame   depth stack 22.9 ms
```

The stack, where the unexplained ~9 ms fixed cost lives, did not move. Reverted: 0.4 ms of 33 is
run-to-run noise, and it cost a branch, a threshold constant, and a mask encoding two unrelated
things.

**What worked**: `tts_nn::attn` — two kernels that index the cache in place (scores with the mask
folded in, then the weighted `v` sum), keeping candle's softmax between them. No threadgroup
memory, no cross-lane reductions.

It was worth far more than the ~2 ms/step estimated from single-lane traffic, because the copies
are *per lane* and never amortise: batched f16 went from 13.70 to 5.59 ms/lane (3.47x -> 7.41x)
and the render from RTF 0.740 to **0.407**.
