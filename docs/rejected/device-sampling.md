# Rejected: sampling the residual codebooks on device

Tried for `qwen3tts`, reverted. Two reasons: it was wrong, and it was not faster.

## The idea

The depth predictor runs 15 autoregressive steps per audio frame, and each step read its
`[2048]` logits to the host to sample there. Sixteen device→host round trips per frame
(1 talker + 15 predictor) looked like the dominant cost, so the plan was to keep the whole
sampler — temperature, top-k, top-p, multinomial — on device with `sort_last_dim`, `cumsum`
and `index_select`, so the chosen index never leaves the GPU and the next step's embedding
lookup needs no sync.

## Why it was wrong

**candle 0.10.2's Metal `sort_last_dim` silently returns all zeros for n > 1024.**

```
n=8     metal max    3.98352   cpu max    3.98352   ok
n=1024  metal max    3.99999   cpu max    3.99999   ok
n=1025  metal max    0.00000   cpu max    3.99999   BROKEN
n=2048  metal max    0.00000   cpu max    3.99999   BROKEN
n=4096  metal max    0.00000   cpu max    4.00000   BROKEN
```

A single-threadgroup sort with a 1024-thread ceiling and no fallback. No error, no warning —
sorted values and indices both come back zeroed. The residual vocabulary is **2048**, so this
path was never going to work.

What it looked like from the outside: every segment ran to `max_new_tokens` and the render came
out 288 s instead of 51 s. Garbage residuals corrupt the frame's summed embedding, which is the
talker's next input, so the talker loses the thread and never reaches `codec_eos`.

Three things about how this was found are worth keeping:

1. **An isolated equivalence test passed.** Host and device samplers agreed exactly on the
   same logits and the same draw, on CPU *and* on Metal — because the test used a 512-wide
   vocabulary, under the threshold. A unit test at the wrong size is worse than none: it
   licensed the change.
2. **The gate passed too**, because its numerics tier runs on CPU where the sort is fine.
3. What actually localised it was comparing both samplers **in situ**, inside the real model on
   Metal, and printing mismatches. The device side returned values like `1071352034` —
   recognisably f32 bit patterns, i.e. a tensor carrying the wrong data.

## Why it would not have paid anyway

The frame already contains one unavoidable sync: codebook 0 has to reach the host for the
`codec_eos` test. And with synchronised timers the per-frame split is **talker 17 ms,
predictor 24 ms** — that is GPU compute, not transfer. The earlier reading of "host reads
33-44 ms/frame" was the queue draining and being billed to the read that waited on it: the
same unsynchronised-timer trap this repo documented for CosyVoice's vocoder, hit again.

The predictor's 24 ms is 75 small layer passes (15 steps x 5 layers), so it is dispatch-bound.
Removing syncs does not reduce dispatches.

## What to do instead

- **Batch segments through the talker.** Same dispatch count, several frames of output. This is
  the real lever; the obstacle is per-lane text cursors (trap 3), not the sampler.
- If device sampling is ever wanted again, it needs a **custom Metal kernel** in `tts-nn`
  alongside the im2col and snake kernels — not `sort_last_dim`. Top-k of 50 out of 2048 does
  not need a full sort anyway.
