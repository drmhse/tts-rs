# CosyVoice3 in Rust: ported, validated, measured

`Fun-CosyVoice3-0.5B` as the second engine. Three models, 995 M parameters, no ONNX and no
Python at runtime. All 25 fixture checks pass.

| | Audio8 | CosyVoice3 |
|---|---|---|
| components | 2 | 3 (+ 2 offline) |
| params | 601 M | **995 M** |
| vocoder | one feed-forward pass | 10-step ODE solve, then an iSTFT vocoder |
| Rust LOC | 2654 | **3176** |
| runtime deps beyond weights | none | none (was 997 MB of ONNX) |

```
cargo run -p cosyvoice --release --bin cosyvoice-validate     # the fixture gate, 27 checks
cargo run -p cosyvoice --release --bin cosyvoice-bench        # where the time goes
cargo run -p tts-cli --release -- speak --engine cosyvoice \
    --text-file examples/senior.txt --voice voices/cosy-default-cosyvoice --out out.wav
```

## Validation

Per-stage, against `fixtures/cosyvoice/oracle.safetensors`. Deliberately per-stage: a
whole-pipeline mismatch says nothing about which of three models is wrong, and this port
needed exactly that discrimination twice.

| stage | check | result |
|---|---|---|
| LLM | prompt assembly | **identical**, 0.0 |
| LLM | prefill hidden state, 356 positions | rel **1.7e-6** |
| LLM | prefill logits, 6761 rows | rel **1.7e-6** |
| LLM | **teacher-forced logits, 105 steps x 6761** | rel **3.1e-6** |
| LLM | **teacher-forced argmax ids** | **105/105 identical** |
| flow | speaker affine, token embedding, look-ahead conv, `mu`, `cond` | 0.0 to rel 9.5e-7 |
| flow | timestep embedding, input+position embedding | rel 5.6e-7, 5.0e-7 |
| flow | DiT blocks 0, 1, 10, 21 | rel 8.6e-7, 8.7e-7, 3.6e-5, **6.7e-4** |
| flow | full 10-step solve, and the trimmed mel | rel 5.9e-4, **4.9e-4** |
| vocoder | F0 predictor vs the f32 reference | rel 2.6e-6 |
| vocoder | harmonic source | **1.93e-5** |
| vocoder | **decoder + iSTFT, from the reference's own mel and source** | rel **1.6e-4** |

Two rows carry most of the weight. **105/105 teacher-forced argmax ids** is the CosyVoice
equivalent of Audio8's "24/24 frames bit-identical": greedy decoding was tried first and
degenerates to one repeated token within two steps, and a constant is something a *wrong*
implementation can also produce, so the gate feeds the reference's own sampled sequence
instead and compares a dense 105 x 6761 logit surface. And **the vocoder's decoder is exact
from the reference's inputs**, which is what lets the remaining vocoder difference be
attributed rather than argued about.

### The DiT's tolerances are measured, not chosen

Blocks 0 and 1 agree to rel 1e-6 and block 21 to 6.7e-4 — a 700x growth with depth, which
looks like a bug until you ask how well f32 can resolve this network at all. Running the
*reference* decoder in f64 and comparing against its own f32:

| | block 0 | block 1 | block 10 | block 21 |
|---|---|---|---|---|
| torch f32 vs torch f64 | 1.07e-6 | 1.28e-6 | 3.86e-5 | **5.56e-4** |
| this port vs torch f32 | 9.3e-7 | 8.7e-7 | 3.6e-5 | **6.7e-4** |

The network amplifies rounding error about 500x over 22 blocks. The port is *at* the floor,
not near it. Per-block budgets in the gate are 2x those figures for the early blocks, where
a wrong layer cannot hide; the last block gets 4x, because at that depth the computation is
chaotic enough that swapping one exact attention kernel for another (verified at rel 2.1e-6
in isolation) moved the row from 1.9e-1 to 3.4e-1.

### Is the audio right?

Voice fidelity against the reference clip, and intelligibility against the input text:

| render | median F0 | ΔF0 vs reference | LTAS cos | WER |
|---|---|---|---|---|
| `default_voice.wav` (the reference clip) | 179.8 Hz | — | 1.0000 | — |
| **CosyVoice, this port** | 177.8 Hz | **−2.0** | **0.9979** | 0.015 |
| CosyVoice, stock PyTorch, same text | 192.8 Hz | **+13.0** | 0.9980 | 0.008 |
| Audio8, this port | 177.8 Hz | −2.0 | 0.9969 | 0.008 |

WER is against the input text via `whisper small.en` (`references/cosyvoice/wer.py`), 133 words. Not
comparable to the 0.031 quoted elsewhere in this repo for Audio8, which was measured with
WhisperX — a different recogniser gives a different number and only same-recogniser
comparisons mean anything. On this passage the port transcribed with **zero** errors and the
PyTorch reference with one, which is a tie inside noise; what it rules out is any
intelligibility defect.

The control is the point. In isolation, −13.1 Hz looks like a port defect. The stock
PyTorch reference rendering the same text through the same weights lands +13.0 Hz on the
other side with essentially the same spectral match, so ±13 Hz is what CosyVoice does with
this voice as the sampled token sequence changes — and no userspace RNG can reproduce
torch's stream, so the sequences necessarily differ. The port sits inside the model's own
variation.

## Speed

`examples/senior.txt` — 132 words, 7 segments — on M4 / 16 GB, canary-verified stable:

| | RTF | notes |
|---|---|---|
| stock upstream PyTorch | **4.370** | CPU; see below |
| **this port** | **0.697** | **6.27x faster** |

That comparison needs a caveat stated plainly: `CosyVoiceModel.__init__` hardcodes
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. **Upstream has no MPS path
at all**, so CPU is not a badly-chosen baseline, it is the only thing stock upstream can do
on this machine. The mature service in `tts/CosyVoice` reaches RTF ~0.76 because it was
adapted to MPS; this port is now within 12% of that, having started this session 3.9x behind it.

Stage split, and where the optimization went:

| stage | RTF | share |
|---|---|---|
| LLM (batched) | 0.187 | 27% |
| flow decoder | 0.474 | 68% |
| vocoder | 0.037 | 5% |

The flow dominates, as expected — 10 Euler steps x 2 (guidance) x 22 blocks = **440 DiT
block passes over the full mel length** per utterance, against Audio8's codec which is one
feed-forward pass.

### What was measured, and what it bought

Starting point RTF 2.96; now 0.697. Every figure below is from `tts-probe --bin dit` or
`cosyvoice-bench`, interleaved and canary-checked.

| change | measured | kept |
|---|---|---|
| flat 2-D matmul instead of `broadcast_matmul` | **1.53x** on a projection, **1.40x** on a block | yes |
| fused `candle_nn::ops::sdpa` instead of assembled attention | **2.6x** | yes |
| pass `sdpa` transposed *views*, not `contiguous()` copies | **2.7x** again | yes |
| fused `candle_nn::ops::layer_norm` instead of six primitive passes | **5.51x** on that op | yes |
| vocoder convs as one GEMM with a tap-major weight | **1.30x** on the upsampling decoder | yes |
| **one flow call per utterance, not per segment** | **1.71x on the flow, 1.72x on the vocoder** | yes |
| `slice_set` instead of `slice_assign` for the KV cache | **1.90x on prefill** | yes |
| f16 projections | 1.08x | **no** — not worth any accuracy risk |
| `grouped_causal_conv1d` looping 16 groups | **0.97x** | **no** — see below |
| **split attention by head group**, heads permuted at load | **1.25x on a block** | yes — see below |
| `slice_set` instead of `slice_assign` for the KV cache | **1.90x on prefill, 2.0x on the LLM stage** | yes |
| longer segments (`--max-chars 420`) | 1.02x | **no** — the flow gains, the LLM gives it back |

Attention was the surprise. It costs 8.2x a projection while doing **5x less arithmetic**
(2.6 GMAC against 13.4 per block), because the assembled form materialises a
`[2, 16, 798, 798]` scores tensor — 81.5 MB — and then touches it four times: write, scale,
softmax, read. About 490 MB of traffic to do 2.6 GMAC. The fused kernel never materialises
it.

**A 1.25x, recovered by relabelling the heads.** Only head 0 gets RoPE (trap 1 below), and
attention is independent per head, so the rotated head and the other fifteen should be two
`sdpa` calls over views — avoiding a 6.5 MB rebuild of `q` and again of `k`. Measured 1.25x on
a whole block, landing within 0.25 ms of a variant with the rotary deleted entirely.

The obvious way to write it is wrong, because **candle 0.10.2's Metal `sdpa` mishandles a head
axis narrowed to a non-zero offset**: `narrow(1, 0, 1)` agrees with the naive form to rel
6.7e-7, but `narrow(1, 1, 15)` returns rel **1.24** — a wrong answer, not a precision effect.
Wired in that way it put block 0 off by rel 1.5e-1, and the fixture gate caught it immediately.

The fix is not to work around the bug but to avoid it: **permute the head blocks of `to_q`,
`to_k`, `to_v` and `to_out` at load so the rotated head is last.** The fifteen unrotated heads
are then `narrow(1, 0, 15)` — offset zero, which the kernel handles correctly — and the single
rotated head does carry an offset but is one sixteenth of the tensor, so making it contiguous
first costs almost nothing. Relabelling heads is exact as long as every projection agrees;
`to_out` consumes them along its input columns, so it takes the same permutation on `dim 1`.
Block 0 now validates at **rel 8.56e-7**.

Recorded as a process note too: that variant was benchmarked *before* it was checked for
correctness, which is how a 1.25x on the wrong function nearly became an optimization.

**The vocoder's convolutions do go faster as a GEMM** — but only after fixing why an earlier
attempt at that failed. `../performance/candle-on-metal.md` had refuted conv-as-GEMM and blamed
materialisation traffic; splitting the route in two showed the GEMM is 2.4 TFLOP/s (8.4x faster
than the direct conv on its own) and the *gather* was running at 4.9 GB/s. Reordering the
im2col rows from tap-interleaved to tap-major makes the gather 3.01x faster and costs one weight
permutation at load. The HiFTGenerator's upsampling decoder went 454.0 → 349.7 ms. See
`tts_nn::causal_conv1d_gemm`.

Note the honest limit of that win: **CosyVoice's end-to-end RTF did not move** (1.612 → 1.602,
inside run-to-run noise), because the vocoder is only 16% of this engine and the isolated
decoder is not the whole vocoder stage. The stage benchmark found a real 1.30x; the engine total
cannot see it. Audio8, whose codec is 32% of its runtime, went 0.664 → 0.499 from the same
change.

**The largest win was scheduling, not kernels.** `flow.synthesize` prepends the voice's 294
speech tokens and 588 mel frames on every call, so decoding segment by segment re-pays that
prompt each time: on this passage, **7 x 588 = 4116 prompt mel frames against 2634 generated
ones — 61% of the flow's work was the same prompt, redone**. Decoding the concatenated tokens
in one call pays it once. Attention is O(n^2), so a single 3222-frame solve does more attention
work than seven 964-frame ones, but projections are ~90% of a block and the net is **1.71x on
the flow**. The vocoder gains **1.72x** as well, from one call instead of seven.

That change removes the inter-segment silence, and it costs intelligibility: WER went from
0.000 to **0.023**, three errors in 133 words, from sentence boundaries running together. The
fix does not require giving the speed back — the cut points are exact, because the flow holds
each speech token for two mel frames and the vocoder emits 480 samples per frame, so a segment
of `n` tokens is exactly `n * 960` samples. Cutting the fused waveform there and re-inserting
the gaps brought WER to **0.015**, against 0.008 for the PyTorch reference on the same text.

Two honest notes on that. The remaining gap to the per-segment path's 0.000 is two errors in
133 words and every render here is a different draw, so this is inside the band rather than
provably equal — 0.000, 0.008, 0.015 and 0.023 are all 0-3 errors. And `--set
flow_per_segment=1` restores the old behaviour for anyone who would rather have it.

**`grouped_causal_conv1d` is refuted for this shape.** `tts-nn` loops the groups because
candle's `groups > 1` conv1d is 19x off the hardware for the *depthwise* case (one channel
per group). At the DiT's position embedding — k=31, 16 groups, 64 channels each — the loop
measures **0.97x**, i.e. very slightly worse. The pathology is one-channel groups, not
grouped convolution in general. The op is still 24.6 ms for 3.2 GMAC (0.13 TFLOP/s) and
remains the largest single un-won item in the flow at ~8% of it.

## Nine traps

Six from reading, three from measuring. Every one is silent — the port runs and the audio
sounds like speech.

**1. RoPE reaches head 0 only.** The reference applies `apply_rotary_pos_emb` to the
*pre-reshape* `[b, n, 1024]` projection, and `x_transformers` implements *partial* rotary:
it rotates the first `freqs.shape[-1]` channels and passes the rest through. `freqs` comes
from `RotaryEmbedding(dim_head=64)`, so `rot_dim` is 64 — channels 0..63, which is exactly
head 0 of 16. Heads 1..15 have no positional information. Verified numerically on a tensor
of ones: channels 0..63 change, nothing else does. Applying RoPE per-head to all 16 heads is
what any reasonable implementation would do, and it is a different model.

**2. The two engines' RoPE conventions are opposite.** HF's Qwen2 uses `rotate_half`
(half-split). Audio8's Fish-Speech weights use `torch.polar` (interleaved adjacent pairs).
Same dim, same head count, same head_dim — and both run without error. Reusing `audio8::ar`'s
`rope_i` here put the hidden state off by **rel 0.78**. Caught only because the fixture
existed; there is no way to notice it from the audio.

**3. The flow's initial noise is a fixed tensor, not a draw.**
`CausalConditionalCFM.__init__` calls `set_all_random_seed(0)` and builds
`randn([1, 80, 15000])` once, then slices it on every request. Output is deterministic given
the tokens. A port that samples its own noise looks correct and sounds different, so the
tensor ships as a 4.8 MB asset.

**4. `<|endofprompt|>` is required and nothing in the frontend adds it.** The LLM asserts
token 151646 appears in `prompt_text + text`; the *service* supplies it by prepending
`"You are a helpful assistant.<|endofprompt|>"`. Hit twice — once building the voice asset,
and again writing the PyTorch control render, where its absence collapsed the first segment
to two tokens and failed inside the vocoder on a 3-frame mel. Not an obvious symptom.

**5. The upstream safetensors are the wrong weights.**
`CosyVoice-BlankEN/model.safetensors` (988 MB) sits beside the checkpoints with exactly the
Qwen2 key names, so it looks like the obvious source. It is the base initialisation:
**0 of 290 tensors match** the `llm.model.*` tensors inside `llm.pt`, max relative difference
1.82. `llm.pt` is the fine-tune and the only correct source. Checked rather than assumed,
because the failure mode is fluent speech in the wrong voice with no error anywhere.

**6. The tokenizer needs ~280 special tokens added at construction.** The checkpoint
directory's `added_tokens_decoder` lists three. `CosyVoice3Tokenizer.__init__` registers
`<|endofprompt|>`, the paralinguistic tags (`[breath]`, `[laughter]`, ...) and a full
ARPAbet/pinyin phoneme set on top. Serialising `AutoTokenizer.from_pretrained(dir)` as-is
makes `<|endofprompt|>` tokenize as **nine pieces of literal text**. `convert.py` now
serialises the real wrapper's backend tokenizer and asserts id 151646 survives.

**7. The leaky-ReLU before `conv_post` has slope 0.01, not 0.1.** The reference writes it as
a bare `F.leaky_relu(x)`, taking torch's default, while every sibling in the same loop passes
`self.lrelu_slope`.

**8. The NSF noise is not in the checkpoint, and it is not negligible.**
`SineGen2.sine_waves` is a `torch.rand(1, 300 * 24000, 9)` **plain attribute**, not a
registered buffer — absent from `hift.pt` and redrawn at construction. It is reproducible
at all only because `cosyvoice3.yaml` line 4 calls `torch.manual_seed(1986)` immediately
before building the model. Zeroing it moves the waveform by max **0.164** against a signal
of rms 0.078, so it cannot be ignored. The fixture exports the slice actually used;
synthesis draws its own from the same distribution (`torch.rand`, uniform on `[0, 1)` —
*not* `randn`, so it has a non-zero mean, and reproducing the distribution reproduces that).

**9. The reference's harmonic phase is numerically degenerate in f32.** It accumulates
`cumsum(rad) * 2 * pi * 480`, which on this fixture reaches **1.7e7 radians — where one f32
ulp is 1.0 radian**. Its harmonic phases late in an utterance are rounding noise, and torch's
own f32 result sits rel 5.3e-4 from the same computation in f64. Since `sin` is periodic the
phase only matters modulo one cycle, so this port accumulates it on the host in f64 and folds
each step back into `[0, 1)`: 1890 f64 operations per utterance, and the phase ulp goes from
1.0 radian to about 4e-7. This is the one place the port is deliberately *more* accurate than
the reference — and it moves *closer* to torch's f32 output, not further, because two
independent f32 accumulations disagree by more than either disagrees with exact arithmetic.
The harmonic source then matches to 1.93e-5, which is precisely torch's own f32-versus-f64
error on that tensor.

### Two things that look like traps and are not

Recorded so nobody spends time on them.

- **`SineGen2.rand_ini` is dead code.** It adds a random phase offset to sample 0 of the
  full-rate tensor, which the `scale_factor=1/480` linear downsample then discards — that
  downsample reads samples 239 and 240 of each block. Measured contribution to the source
  signal: **exactly 0.0**. A port that faithfully reproduces it reproduces nothing, and it
  would need an extra asset to do it.
- **`CausalConvRNNF0Predictor` has no recurrence.** Five convolutions and a linear layer.
  The port plan had budgeted for an RNN that does not exist.

A third, in the same spirit: the reference upsamples F0 by 480, takes `(f0 * h / sr) mod 1`,
then *downsamples by 1/480* before the cumulative sum. Those two resamplings look like they
must lose something. With `align_corners=False` the downsample reads samples 239 and 240 of
each constant 480-sample block, so the round trip is exactly "take the frame's value" —
measured at 0.0 difference. The port computes at frame rate and never materialises the
100800x9 intermediate.

## Where the ONNX went

CosyVoice needs a 969 MB whisper-based speech tokenizer and a 28 MB speaker-embedding model
to turn a reference clip into conditioning. Neither is needed at runtime: the conversion
depends only on the clip, so it runs once in Python and the result ships as a voice asset —
the same move that keeps Audio8's codec *encoder* out of its binary. **The Rust runtime for
both engines is safetensors-only.**

The cost, stated plainly: cloning from arbitrary audio at request time is not possible
in-process. Adding a voice is a Python step. For a service with a fixed voice set that is
the right trade; for one that clones per request it is not, and fixing it means porting a
whisper encoder and an FSQ quantizer to Candle.

## What is not ported

- **Streaming.** The reference has it; the vocoder's `hift_cache_dict` carries mel and a
  speech offset between chunks, and the flow decoder switches to chunked attention masks.
  Non-streaming first, on purpose.
- **Text normalisation.** The reference frontend runs WeTextProcessing (an FST-based
  normaliser), number spell-out via `inflect`, and Chinese punctuation rewriting before
  tokenizing. None of it is model code and porting it is a separate project. What is here is
  whitespace collapsing, sentence segmentation and the Qwen2 BPE. Text that arrives already
  normalised is unaffected; text containing digits will be read as the tokenizer sees them.

## Next, in order of measured value

1. **The grouped conv at the DiT's position embedding** — 24.6 ms for 3.2 GMAC at the old
   length, and both candle's grouped path and the group loop are equally bad. Now that the
   tap-major GEMM is known to beat `conv1d` by 1.34-1.73x on dense convs, the same treatment
   per group is the obvious untried form: a batched GEMM of `[16, 64, 1984] @ [16, 1984, L]`.
2. **The LLM is now the joint-largest stage** at ~33-40% of the engine, having been 27%. It
   decodes one token at a time at batch 1, and Audio8's batching work applies almost directly
   — same Qwen2 geometry, same narrow-KV and GQA-by-query-reshape. Segments are already
   decoded in a loop, so the prompts are there to batch.
3. **The `sdpa` offset bug**, upstream — worth 1.25x on every DiT block for free.
4. **Streaming**, which is the one capability the PyTorch service has and this does not.
