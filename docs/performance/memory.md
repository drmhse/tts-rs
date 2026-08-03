# Memory: what is known, and two metrics that lied

This document exists because a claim was made here without evidence, defended twice with
measurements that were artifacts, and only then checked properly. The claim was that memory
is "flat by construction."

## What is actually known

- **One process renders a 16-minute chapter at RTF 0.54** on an M4 / 16 GB, 118 segments,
  without the machine struggling. Reproduced three times.
- **Two engines do not fit.** Two concurrent `audio8` renders put a 16 GB machine into swap
  and neither finished. That is the whole of the original failure, and its cause was a
  process-management bug, not the engine: `pkill -f "tts-cli.*ch1.txt"` matches the `cargo`
  wrapper, while the process doing the work is `target/release/tts speak …` and survives.
  **Match the binary, not the cargo invocation.**
- **Long renders use more than short ones**, mostly because more segments meant more KV
  lanes. Now capped; see below.
- `scripts/narrate.sh` therefore holds one engine resident at a time and refuses to start
  when another `tts` process is running.

## Found: an unbounded LLM batch, 101 MB per lane

The bulk of it was not candle and not the codec. CosyVoice's `llm_max_batch` defaulted to
`usize::MAX`, so the engine decoded **one lane per segment** — and every lane carries its own
KV cache:

```
2 (k and v) * N_KV 2 * CACHE 4096 * HEAD_DIM 64 * 4 bytes * LAYERS 24  =  101 MB per lane
```

| lanes | KV cache |
|---|---|
| 1 | 0.10 GB |
| 8 (the new default, matching Audio8) | 0.81 GB |
| 118 (a 16-minute chapter, unbounded) | **11.88 GB** |

That is the growth. It scales with the number of *segments*, which is why it looked like a
leak that grew with document length — and why every theory about size classes and im2col
missed it. Audio8 never showed it because its `max_batch` has always defaulted to 8.

Capping it at 8 took the LLM stage from **727.7 s to 191.7 s (3.8x)** on a 16-minute chapter
and the whole engine from **RTF 1.49 to 0.96**. Batching is nearly free in time and very much
not free in memory; the measured speed curve is already flat by 4-7 lanes
(`tts-probe --bin llmbatch`), so there was never a reason to go wider.

**The lesson worth keeping:** the growth was proportional to a count the engine controls, and
it was found by computing the size of one obvious data structure — not by profiling. Do that
arithmetic first.

## What is still not known

**How much memory a render actually wants.** Two attempts, both saturating:

| metric | reading | why it is useless here |
|---|---|---|
| `/usr/bin/time -l` max RSS | **3.59 GB** at 8, 16, 24 and 49 segments | RSS cannot exceed what is physically resident. Under pressure the kernel compresses and swaps, RSS *falls*, and the number stops tracking demand |
| `phys_footprint_peak` | **exactly 13.00 GB** for a quantised decode, for `max_batch=8`, and for `max_batch=1` | three configurations that should differ enormously, identical to two decimals — a ceiling on what the system grants, not a measure of what was asked for |

The first was believed long enough to produce a "flat at 3.59 GB" plateau that does not
exist. The second was believed long enough to build a fix on it. **A metric that returns the
same value when the workload changes by 6x is reporting its own limit.**

Anyone picking this up again should instrument the allocator rather than the process: count
and total the `MetalDevice::new_buffer` calls, which is where the large allocations are.

## Why the pool is the suspect

candle's Metal buffer pool has **no public way to release anything**:

- `MetalDevice::new_buffer` reuses a pooled buffer only when its `Arc::strong_count` has
  fallen to 1 (`find_available_buffer`, `metal_backend/device.rs:326`).
- `buf_size` is `next_power_of_two`, so a 592 MB request takes a **1 GB** buffer.
- `drop_unused_buffers` is private, and nothing on `MetalDevice`'s public surface trims the
  pool.

So every large buffer that is ever live at the same time as another stays resident for the
life of the process. The engines cannot return memory to the OS even in principle.

## What was done about it

**`tts_nn::causal_conv1d_gemm` chunks along length** (`GEMM_COL_BUDGET`, 64 M elements =
256 MB), so every im2col allocation is the same size and the pool converges instead of
holding one buffer per segment length. It costs the codec time — the original measurement
was RTF 0.156 → 0.204 at a smaller budget — and it took the failing case from a swap storm
to a run that completes.

Note what cannot be claimed: with both metrics saturating, there is no clean before/after
memory figure for it. It is kept because the failure stopped, not because a number improved.

**This chunking was reverted once, wrongly.** The first time it was judged on speed alone
and reverted for costing 31% of the codec stage. Nobody measured what it did for memory. A
codec 31% slower that completes beats a faster one that swaps.

## One thing that was tried and is wrong

**Quantising the decode frame count** — rounding every segment up to a multiple of 64 frames
so all decodes allocate identical shapes — saved nothing *and changed the audio*. The
padding was assumed harmless because "every convolution in this decoder is causal." It is
not: the transpose-convolution upsampling stages look ahead, so frames appended at the end
propagate backwards into earlier output. A byte-comparison against an unpadded render caught
it; the fixture gate did not, because the gate validates short single-segment decodes and
nothing exercises a long multi-segment render.

That gap is worth closing before anyone trusts a change like this again.
