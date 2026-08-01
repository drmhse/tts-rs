# Status: two engines, both ported, both validated

Text in, speech out, one process, bounded memory. 10296 lines of Rust across eight crates, no
ONNX and no Python at runtime.

| engine | model | params | state | RTF | gate |
|---|---|---|---|---|---|
| `audio8` | Audio8-TTS-Preview-0.6b, 44.1 kHz | 601 M | ready | **0.499** | 8 checks, greedy generation bit-identical |
| `cosyvoice` | Fun-CosyVoice3-0.5B, 24 kHz | 995 M | ready | **0.697** | 27 checks, teacher-forced ids 105/105 |

```
cargo test --release                               # 27 suites
cargo run -p audio8   --release --bin audio8-validate      # Audio8 fixture gate
cargo run -p cosyvoice --release --bin cosyvoice-validate    # CosyVoice fixture gate
cargo run -p cosyvoice --release --bin cosyvoice-bench       # where CosyVoice's time goes
cargo run -p tts-cli --release -- engines
```

## Layout

| crate | lines | what |
|---|---|---|
| `tts-core` | 674 | `Engine` trait, voice assets, segmentation, WAV, the PRNG |
| `tts-nn` | 773 | shared model machinery: convs, activations, norms, RoPE tables, `Proj`, `Linear` |
| `tts-bench` | 179 | the thermally-honest measurement harness |
| `tts-engines` | 69 | the registry — the one file that knows which engines exist |
| `tts-cli` | 237 | `tts`: engines / voice / speak |
| `audio8` | 2654 | Audio8: `ar`, `codec`, `sample`, `prompt`, `engine` + gate |
| `cosyvoice` | 3176 | CosyVoice: `llm`, `flow`, `hift`, `stft`, `sample`, `engine` + gate and bench |
| `tts-probe` | 2534 | op-level benchmarks |

## Validation

Fixtures are fp32 CPU reference tensors, so a mismatch is a port bug rather than a precision
difference — which is why both `dump_fixtures.py` scripts run in fp32 even though Audio8
ships in bfloat16.

### Audio8

| check | result |
|---|---|
| codec, 8-frame synthetic codes -> waveform | max\|diff\| **3.4e-6** (rel 3.7e-5) |
| codec, 24-frame e2e codes -> waveform | max\|diff\| **9.6e-6** (rel 4.3e-5) |
| sample count, 24 frames | 49152 = 24 x 2048 exactly |
| prompt token ids | **27/27 identical** |
| slow-AR normed hidden, 27 positions | rel **3.8e-5** |
| slow-AR logits, 27 positions x 4097 rows | rel **3.5e-5** |
| greedy argmax at the final position | identical (2954) |
| **greedy generation, 24 frames x 10 codebooks** | **24/24 frames bit-identical** |

The last row is the one that matters. Greedy decoding is deterministic on both sides, so
identical codes means the whole loop agrees: embedding, RoPE, GQA, KV cache, the discarded
fast-AR priming step, the legacy filter and the argmax. Nothing else in the suite exercises
the fast AR at all.

### CosyVoice

27 checks across three stages; full table in [porting/cosyvoice.md](porting/cosyvoice.md). The two
that carry the weight:

- **Teacher-forced argmax: 105/105 identical**, logits at rel 3.1e-6. Greedy decoding was
  tried first and degenerates to one repeated token within two steps — and a constant is
  something a *wrong* implementation can also produce — so the gate feeds the reference's own
  sampled sequence and compares a dense 105 x 6761 logit surface instead.
- **The vocoder's decoder is exact** from the reference's own mel and source: rel 1.6e-4
  across 100800 samples, including a hand-written 16-point iSTFT.

The DiT's per-block tolerances come from a measured f32 precision floor rather than being
chosen: running the reference decoder in f64 against its own f32 shows it amplifies rounding
error ~500x over 22 blocks, and this port sits *at* that floor.

### And the audio

Neither gate proves the audio is good, so that is measured separately — voice fidelity
against the reference clip, intelligibility against the input text:

| render | median F0 | ΔF0 vs ref | LTAS cos | WER |
|---|---|---|---|---|
| `default_voice.wav` (reference clip) | 179.8 Hz | — | 1.0000 | — |
| Audio8, this port | 177.8 Hz | −2.0 | 0.9969 | 0.008 |
| CosyVoice, this port | 177.8 Hz | −2.0 | 0.9979 | 0.015 |
| CosyVoice, stock PyTorch, same text | 192.8 Hz | +13.0 | 0.9980 | 0.008 |

The last row is the control, and it is the point: in isolation CosyVoice's −13.1 Hz looks
like a defect, but the PyTorch reference on the same text lands +13.0 Hz on the other side
with the same spectral match. ±13 Hz is what the model does as the sampled token sequence
changes, and no userspace RNG reproduces torch's stream, so the sequences necessarily differ.

WER here is `whisper small.en` via `references/cosyvoice/wer.py`. It is **not** comparable to the 0.031
quoted in `performance/quantization-quality.md`, which used WhisperX — only same-recogniser comparisons mean
anything.

## Speed

`examples/senior.txt` — 132 words, 7 segments — M4 / 16 GB, canary-verified stable.

| engine | reference | this port | |
|---|---|---|---|
| `audio8` | 1.307 (PyTorch bf16 MPS, batched) | **0.499** | 2.62x faster |
| `cosyvoice` | 4.370 (stock PyTorch, CPU-only) | **0.697** | 6.27x faster |

| | stage | RTF | share |
|---|---|---|---|
| `audio8` | AR loop (batched) | 0.341 | 68% |
| | codec (GEMM + custom kernels) | 0.158 | 32% |
| `cosyvoice` | LLM (batched) | 0.187 | 27% |
| | flow decoder | 0.474 | 68% |
| | vocoder | 0.037 | 5% |

Two caveats, both load-bearing:

- **The CosyVoice baseline is CPU because upstream has no MPS path at all.**
  `CosyVoiceModel.__init__` hardcodes `cuda if available else cpu`. The adapted service in
  `tts/CosyVoice` reaches RTF ~0.76 on MPS; this port is now within 12% of it. That is the honest number to beat.
- **Audio8's 1.59x is like-for-like**; its reference does run on MPS.

Model load is 1.0-1.5 s for Audio8 including quantizing 417 M params, against 15-17 s for the
PyTorch service's reload path.

## What each port does differently from its reference, on purpose

Every item is measured. Audio8's are detailed in `performance/ar-loop.md` and
`performance/quantization-quality.md`; CosyVoice's in `porting/cosyvoice.md`.

### Audio8

1. **Narrow KV** (5.10x). Attention runs over `0..=pos`. The reference allocates a
   `max_seq_len`-wide cache and masks, doing 16x the necessary work by frame 64 — and its own
   `ArkttsKVCache` supports the narrow form via `return_full=False` and never uses it.
2. **GQA by query reshape** (1.18x). The 7 query heads sharing a KV head fold into the
   matmul's row dimension instead of materialising a 7x copy of K and V per layer per token.
3. **Sliced logit head** (29x on that op). Only 4097 of 155776 rows survive the semantic
   mask. Exact *because* the mask runs first — which is why the port deliberately does not
   expose a caller-supplied `logits_processor` hook that could un-mask other ids.
4. **q8_0 projections** (3.35x). 417 M of 601 M params. Embeddings, heads and norms stay f32.
5. **Alpha folding** in the codec (10.5% of codec time, exact). 13 of 29 snakes shed both
   broadcasts, the other 16 shed one.
6. **Fused RVQ tables** (exact). Ten embeddings and ten conv dispatches become ten gathers.
7. **Depthwise as shifted multiply-accumulates** (19x).
7b. **Causal convs as a single GEMM with a tap-major weight** (1.34x to 1.73x, growing with
    length). This *overturns* `performance/candle-on-metal.md`'s Finding 1, which refuted conv-as-GEMM
    and blamed materialisation traffic. The measurement was right and the attribution was
    wrong: the GEMM is 2.4 TFLOP/s and 8.4x faster than the direct conv on its own, while the
    gather ran at 4.9 GB/s. Reordering the im2col rows from tap-interleaved to tap-major makes
    the gather 3.01x faster, and costs one weight permutation at load. Also applied to
    CosyVoice's vocoder.
8. **f32 Gumbel draw** — a bug fix; see below.
9. **No codec encoder.** Cloning codes ship as a 20 KB asset; 126 encoder tensors stay out.
10. **Batched decoding across segments**, with right-aligned prompts. Per-sequence gain ~1.9x
    at batch 8 — *not* the 11.95x a layer benchmark projected, because a step is 64
    layer-passes plus ten host synchronisations plus a per-sequence sampler, and only the
    projections scale freely. See `performance/ar-loop.md`, including two measurements this
    invalidated and one thermal artifact it produced.

### CosyVoice

1. **Fused SDPA over transposed views** (2.6x, then 2.7x again). The assembled form
   materialises an 81.5 MB scores tensor and touches it four times — 490 MB of traffic for
   2.6 GMAC, which is why attention measured 8.2x a projection while doing 5x *less*
   arithmetic.
2. **Flat 2-D matmul** instead of `broadcast_matmul` (1.53x on a projection, 1.40x on a
   block). Now repo-wide via `tts_nn::matmul_2d`.
3. **Fused affine-free LayerNorm** (5.51x on that op), 660 calls per utterance.
4. **All 72 vocoder snakes shed their reciprocal**, and 36 shed the multiply too — better
   than Audio8's codec, because a HiFTGenerator ResBlock's skip carries the block *input*
   rather than an activation output.
5. **The harmonic source is computed at frame rate**, never materialising the 100800x9
   intermediate, and its phase is accumulated on the host in f64 modulo one cycle. The
   reference's f32 accumulation reaches 1.7e7 radians where one ulp is a full radian.
6. **No attention mask.** Non-streaming, `add_optional_chunk_mask` returns all-ones and the
   subsequent `repeat` and `masked_fill` are no-ops — so the port skips building it 440 times.

## Two bugs in the references worth knowing about

**Audio8's sampler is broken under its own default dtype.** `_sample` is Gumbel-max drawing
noise at `dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct values,
the ratio ordering collapses, and output is unintelligible and never reaches EOS. Diagnosed
by bisection after 168 s of noise with every segment hitting the token cap. Both
implementations here draw in f32.

**CosyVoice's harmonic phase is numerically degenerate in f32** — see item 5 above. Its
`SineGen2.rand_ini` is also dead code: the phase offset it adds to sample 0 is discarded by a
downsample that reads samples 239 and 240 of each block, measured at exactly 0.0.

## Honest negatives

Kept so nobody repeats them.

- **candle's matvec already saturates the bus** (99-139 GB/s on ~120 GB/s). An earlier claim
  of a 3.5x kernel win came from thermally inflated per-layer timings and was retracted.
- **A serial-dependency-chain hypothesis** for the AR loop: 1.01x. **w1/w3 fusion**: 0.94x.
- **Two Audio8 micro-optimizations gained nothing measurable** — a flattened q/k/v split and
  a pre-scaled RoPE table. Both are exact and both are kept because they cost nothing.
- **f16 for CosyVoice's DiT**: 1.08x. Not worth any accuracy risk in a network that already
  amplifies rounding 500x.
- **`grouped_causal_conv1d`'s group loop is 0.97x** at 64 channels per group. The 19x win it
  was written for is specific to *one-channel* groups.
- **Two independent defects in candle 0.10.2's Metal `sdpa`**, both found by trying to use it
  and both caught by a fixture:
  - *A non-zero offset on the head axis gives wrong results.* `narrow(1, 0, 1)` is exact at
    rel 6.7e-7; `narrow(1, 1, 15)` returns rel **1.24**. This killed a real 1.25x on the DiT
    (splitting attention by head group so partial rotary costs nothing). It was benchmarked
    before it was verified, which is how a 1.25x on the wrong function nearly shipped.
  - *An additive mask is wrong for short sequences.* Measured against the naive form,
    `t <= 8` comes back at relative error ~1.5 while `t >= 12` is exact to 2e-7. Swapping the
    fused kernel into Audio8's windowed codec attention passed the 24-frame fixture at 7.6e-6
    and failed the 8-frame one at 6.4e-1. Reverted; that attention stays hand-rolled.

  Both are characterised by `tts-probe --bin dit`. The fused kernel is still used where neither
  applies — the DiT's unmasked, full-width attention — which is where the 2.6x came from.
- **Longer CosyVoice segments are a wash** (1.02x): the flow gains what the LLM gives back.

## What is left

| | value | notes |
|---|---|---|
| Compact interior gaps in a batch | small | only a contiguous *tail* of finished lanes is shed today, because a prefix narrow shares storage; an interior gap would mean rebuilding ~96 MB of KV cache |
| CosyVoice: one flow call per utterance, not per segment | ~1.65x on the flow | each segment re-pays the voice's 588 prompt frames. Changes the output — continuous prosody, no inter-segment gaps — so it is a behavioural decision, not a free win |
| The grouped conv at the DiT's position embedding | ~8% of the flow | 24.6 ms for 3.2 GMAC; both candle's path and the group loop are equally bad. Untried: batched GEMM over a per-group im2col |
| Device-side sampling for CosyVoice | ~1-3% of the LLM | measured: the host readback plus sampling is 1-3% of a decode step at batch 4-7, not the 15% Audio8's loop pays. Much less attractive here than it looks |
| Device-side sampling | ~1.18x on Audio8's AR loop | 15.3% of the loop is the sync plus host-side sampling. Now the largest single item, since the AR is 68% of Audio8's runtime |
| CosyVoice: fuse the DiT's remaining elementwise ops | ~5% of the flow each | the block is now 108 ms at 3192 frames: sdpa 33%, projections 12%, feed-forward 15%, and the rest spread over norms, transposes and residuals. No single large item is left — `gelu_tanh` is 1.4%, the post-attention transpose ~5% |
| CosyVoice: the 588 prompt frames | 18% of every block pass, if it can be exploited at all | the flow runs on prompt+target (3222 frames) and discards the prompt's output, but attention is bidirectional so the prompt's hidden states are genuinely needed at every layer. No exact saving found |
| Streaming, for either engine | — | CosyVoice's vocoder carries a chunk cache and its flow switches to chunked masks; neither is implemented |
| CosyVoice text normalisation | — | an FST normaliser plus number spell-out; not model code, a separate project |
