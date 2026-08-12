# tts-rs reference

Everything beyond the README: setup, architecture, how it validates, how fast it is, the
traps each port hit, and what did not work.

- [Setup](#setup)
- [Architecture](#architecture)
- [Validation](#validation)
- [Performance](#performance)
- [Porting traps](#porting-traps)
- [Serving and narration](#serving-and-narration)
- [What did not work](#what-did-not-work)

---

## Setup

Three levels. Stop at whichever you need. Voice assets are already in the repo (`voices/`,
~200 KB), so cloning a voice needs no PyTorch — only building a *new* one does.

| level | you get | needs |
|---|---|---|
| 1. Audio8 | `tts speak --engine audio8` | Rust, python ≥ 3.10, ~4 GB |
| 2. CosyVoice / Qwen3-TTS | those engines | their checkpoints, ~4 GB each |
| 3. Regenerate fixtures | the gates rebuilt from source | upstream CosyVoice repo, python 3.10 |

Everything assumes macOS on Apple silicon. `cargo build --no-default-features` drops the
Metal kernels for CPU fallbacks and is the only configuration Linux can build, since candle's
`metal` feature does not exist off Apple platforms.

### 1. Audio8

```sh
./scripts/bootstrap.sh
```

Checks the toolchain, downloads [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
(~2.4 GB), creates `references/audio8/.venv`, folds the codec's `weight_norm` into
`codec.safetensors`, fetches the derived assets, and builds. Idempotent.

The one step that is not a download is folding the codec: Audio8 ships `codec.pth`, a 1.35 GB
pickle where every convolution is wrapped in `weight_norm`, so stored parameters are a
magnitude and a direction rather than a weight. Folding once means the Rust side memory-maps
plain weights and does no reparametrisation at runtime.

### 2. The other two checkpoints

**CosyVoice.** Two artifacts it needs are not in the checkpoint and cannot be produced without
the upstream python package: `rand_noise.safetensors` (the CFM decoder builds it once under
`set_all_random_seed(0)` and slices the same tensor every call) and a consolidated
`tokenizer.json` (`CosyVoice3Tokenizer.__init__` registers ~250 special tokens that
`AutoTokenizer.from_pretrained` alone does not). `scripts/fetch-assets.sh` fetches both, so
**running CosyVoice needs no upstream repo** — only its checkpoint and a plain `torch.load`.

```sh
references/audio8/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B', local_dir='/tmp/Fun-CosyVoice3-0.5B')"

references/audio8/.venv/bin/python references/cosyvoice/convert.py \
    --checkpoints /tmp/Fun-CosyVoice3-0.5B --out references/cosyvoice/weights
```

It prints a tensor inventory — the spec the Rust port was written against. If it does not
match `fixtures/cosyvoice/oracle.json` you have a different model revision and the gates will
fail for a good reason. It also prints `[skip] tokenizer` unless the `cosyvoice` package is
importable; that is expected, since `fetch-assets.sh` already placed the file.

**Qwen3-TTS.** Pip-installable upstream, so no repo and no `PYTHONPATH`:

```sh
mkdir -p references/qwen3tts/weights/speech_tokenizer
B=https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main
for f in config.json generation_config.json preprocessor_config.json \
         tokenizer_config.json vocab.json merges.txt model.safetensors; do
  curl -sSL -C - -o "references/qwen3tts/weights/$f" "$B/$f"
done
for f in config.json configuration.json preprocessor_config.json model.safetensors; do
  curl -sSL -C - -o "references/qwen3tts/weights/speech_tokenizer/$f" "$B/speech_tokenizer/$f"
done
```

There is no `tokenizer.json` in this checkpoint, unlike CosyVoice's — the BPE is built from
`vocab.json` plus `merges.txt` at load. Ten languages only (en, de, es, zh, ja, fr, ko, ru, it,
pt), and `q8_0` is its default because the checkpoint is bf16 and f32 measured 38× slower on a
16 GB machine.

### 3. Regenerating the fixtures

`fetch-assets.sh` pulls ~130 MB of checksummed ground truth from
[`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets), which is what
lets `./scripts/gates.sh` run with no PyTorch at all. Rebuilding from source is the stronger
move if you are auditing the ports rather than using them — a gate that verifies against
tensors somebody else uploaded is only as trustworthy as the upload.

```sh
cd references/audio8 && .venv/bin/python dump_fixtures.py --weights weights --out ../../fixtures/audio8

cd /path/to/CosyVoice            # needs the upstream repo on PYTHONPATH
PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
    /path/to/tts-rs/references/cosyvoice/dump_fixtures.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --voice /path/to/tts-rs/voices/cosy-default-cosyvoice \
    --out /path/to/tts-rs/fixtures/cosyvoice

references/qwen3tts/.venv/bin/python references/qwen3tts/dump_fixtures.py \
    --model references/qwen3tts/weights --voice voices/cosy-default-qwen3tts --out fixtures/qwen3tts
```

Building a new voice uses the same venvs via each engine's `export_voice.py`. The transcript
matters more than it looks: CosyVoice asserts the prompt text contains `<|endofprompt|>` and
nothing in its frontend adds it.

### Disk budget

| path | size | tracked |
|---|---|---|
| `references/*/weights/` | ~3.4–4.3 GB each | no |
| `fixtures/` | ~130 MB | no — fetched |
| `voices/` | ~200 KB | **yes** |
| `target/` | ~1.8 GB | no |

---

## Architecture

Ten crates. Everything shared is `tts-*`; everything engine-specific is named for its engine
and matches the id `--engine` takes.

**Voice assets are the decision that made a second and third engine tractable.** All three
models clone from a reference clip, and in every case turning audio into conditioning needs
machinery the runtime should not carry:

| engine | the clip must become | in-process cost avoided |
|---|---|---|
| `audio8` | `[10, N]` RVQ codes | the codec **encoder** — 126 tensors `convert_codec.py` drops |
| `cosyvoice` | speaker embedding, speech tokens, prompt mel, prompt text tokens | `campplus.onnx` (28 MB) + `speech_tokenizer_v3.onnx` (969 MB), plus an ONNX runtime |
| `qwen3tts` | x-vector, `[T, 16]` RVQ codes, sliced transcript tokens | an ECAPA-TDNN speaker encoder and a Mimi-style RVQ **encoder** |

None of it depends on the text being spoken, so it happens once, offline, in Python, and ships
as a directory of `voice.json` + `voice.safetensors`. `Voice::load` checks the `engine` field
and a mismatch is a hard error rather than a silent substitution.

`Capabilities` is deliberately blunt about what engines genuinely differ on — sample rate,
cloning, streaming, and the quantizations each supports. Neither of the first two models can
use K-quants at all, since those need `k` divisible by 256 and both are 896 wide, so listing
"q8_0" generically would be a lie by omission. Where quantization helps also differs: candle
takes a dedicated matrix-vector kernel only when `dim(-2) == 1`, so quantizing a decode loop
is a 3.35× win while quantizing the DiT — which only runs on full sequences — buys much less.
`cosyvoice` therefore quantizes the LLM and leaves the flow decoder and vocoder dense.

### Adding an engine

1. New crate on `tts-core` + `tts-nn`.
2. `capabilities()` returning `available: false` with a reason — **commit that first**, so the
   identifier exists before the implementation.
3. Register in `tts-engines`.
4. **Before any Rust:** convert weights to safetensors, export a voice asset, dump *per-stage*
   fixtures. This is why Audio8's codec validated at 2.8e-6 and its greedy generation came out
   bit-identical first try, and why CosyVoice's reversed RoPE convention localised to one line
   instead of "the audio sounds wrong". A whole-pipeline mismatch says nothing about which of
   three models is at fault.
5. Implement stage by stage, checking each fixture before starting the next.
6. Flip `available`.
7. Measure through `tts_bench::Harness`, then optimize — and **verify each variant is correct
   before believing its timing**. A 1.25× on the CosyVoice DiT turned out to be computing the
   wrong function.

---

## Validation

```sh
./scripts/gates.sh          # everything below, skipping any tier whose inputs are missing
```

| gate | what it proves |
|---|---|
| `audio8-validate` | 8 checks; greedy generation **bit-identical** to the reference |
| `cosyvoice-validate` | 27 checks; teacher-forced argmax **105/105 identical** |
| `qwen3tts-validate` | 65 rows; argmax codebook 0 identical, predictor **15/15** |
| `cargo test --release` | 32 tests including the `tts_engines` doctest |

Gates compare against per-stage fp32 activations dumped from PyTorch, so a failure localises
to a stage rather than to "the audio sounds wrong". A tier whose inputs are absent is
**skipped and reported**, never silently passed.

Sampled output is deliberately *not* gated on equality: `ras_sampling` draws from torch's
generator, so the sampled sequence is not reproducible across implementations. The gates check
prefill logits and a greedy rollout instead, and quality is checked separately by WER.

---

## Performance

Two fixtures, both tracked so any figure here can be reproduced:

| fixture | | |
|---|---|---|
| `examples/senior.txt` | 132 words, 7 segments, ~50 s of audio | the short-passage case |
| `examples/chapter.txt` | 1612 words, 100 segments, ~11.6 min of audio | the long-form case, where batching engages |

### Short passage

`examples/senior.txt`, M4 / 16 GB, `q8_0`. Median of five samples with the three engines
interleaved in one session.

| engine | reference | this port | spread | |
|---|---|---|---|---|
| `audio8` | 1.307 (PyTorch bf16 MPS, batched) | **0.554** | 0.547–0.562 | 2.36× faster |
| `cosyvoice` | 4.370 (stock PyTorch, CPU-only) | **0.726** | 0.697–0.734 | 6.02× faster |
| `qwen3tts` | — | **0.665** | 0.642–0.687 | the wrong case for it; see below |

### Chapter, and what batching is actually worth

`examples/chapter.txt`, 100 segments. `f16` is the median of three samples.

| engine | weights | RTF | talker/LLM | codec |
|---|---|---|---|---|
| `qwen3tts` | `q8_0` | 0.661 | 0.588 | 0.072 |
| `qwen3tts` | **`f16`** | **0.260** (0.256–0.261) | 0.187 | 0.073 |

**`q8_0` gains nothing from 14× more segments** — 0.665 on seven, 0.661 on a hundred. That is
candle's quantized `mm_t` padding to a large row tile, so batch 8 costs 8× batch 1, measured
end to end rather than inferred from op timings. `f16` gets **2.54×**, and all of it is in the
talker (0.588 → 0.187); the codec is unchanged at 0.072, as it must be, since the weight format
being varied is the talker's.

This is why the narration path uses `f16` and why `qwen3tts` is the default for a book despite
being the slowest of the three on a short passage. Reproduce with:

```sh
cargo run -p tts-cli --release -- speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --quant f16 \
    --text-file examples/chapter.txt --out chapter.wav
```

An RTF of **0.253** was previously quoted for this configuration with no fixture behind it and
no end-to-end render supporting it. The measurement above is what replaced it; the claim turned
out to be close to right, which is luck rather than evidence.

### Where the time goes

On `examples/senior.txt` at `q8_0`, against the same split when each port was first measured.

| | stage | RTF | share | previously |
|---|---|---|---|---|
| `audio8` | AR loop (batched) | 0.340 | 61% | 0.341 |
| | codec | **0.214** | 39% | **0.158** |
| `cosyvoice` | LLM (batched) | 0.183 | 26% | 0.187 |
| | flow decoder | 0.477 | 67% | 0.474 |
| | vocoder | **0.049** | 7% | **0.037** |
| `qwen3tts` | talker | 0.576 | 89% | 0.695 |
| | codec | 0.068 | 11% | 0.168 |

**Open: the two convolution stages regressed and the transformer stages did not.** AR, LLM and
flow all land within 1% of their original measurements; the codec is 35% slower and the vocoder
32%. The suspect is the channels-last conv path added in `tts-nn: decode attention,
channels-last convs, and dense f16 projections`, which was never benchmarked end to end. This
is why `audio8` moved from 0.499 to 0.554. `tts-probe`'s `convgemm` and `upsconv` would isolate
it.

One caveat on the tables. CosyVoice's reference is CPU-only because upstream's
`CosyVoiceModel` hardcodes `cuda if available else cpu` and has no MPS path at all; the adapted
MPS service reaches ~0.76, which this beats by 5% rather than by 6×. And model load is
1.0–1.5 s for Audio8 including quantizing 417 M params, against 15–17 s for the PyTorch
service's reload path.

### How to measure without fooling yourself

An M4 under sustained GPU load drifts **~2×**. This was caught by accident: `dilation.rs`
reported 119.57 ms for a conv that `convopt.rs` had measured at 59.78 ms twenty minutes
earlier. Absolute timings taken on a cool machine early in a session are not comparable to
anything taken later.

1. Interleave variants **in the same run**, never A-then-B-much-later.
2. Report **median and spread of ≥5 samples**, not a 3-iteration mean.
3. Include a **fixed canary workload** so the run's thermal state is recorded with the result.
4. Prefer **ratios within a run** over absolute numbers across runs.
5. Idle-cool between runs, or report the drift.

`crates/tts-bench/src/lib.rs` (`Harness`) implements all five, and
`references/audio8/bench_ar.py` mirrors it so Rust and torch runs land on the same thermal
scale. Two further ways to get it wrong, both learned here: an **unsynchronised stage timer**
once misattributed 13% of a pipeline to the wrong stage, and a **warm A/B loop cannot see
first-touch allocation cost**.

### Memory, and quantization quality

Memory is **not** characterised. An earlier claim that it was "flat by construction" was wrong
— that was an argument, not a measurement. Two attempts to measure it both saturated:
`/usr/bin/time -l` reports RSS, which cannot exceed what is resident, and pinned at 3.59 GB
across a 6× range of input; `phys_footprint_peak` returned exactly 13.00 GB for three
configurations that should differ enormously. What can be said: one process renders 16 minutes
of audio on a 16 GB machine without the system struggling, and two concurrent engines do not
fit. One real find along the way: an unbounded LLM batch at 101 MB per lane, now capped.

`q8_0` costs nothing audible. Token identity is the wrong metric for this — sampled sequences
diverge for reasons unrelated to weight precision, so the question has to be asked of the audio
(WER and speaker similarity), not of the codes.

---

## Porting traps

The traps below cost real time. They are listed because each produced output that was
*plausible but wrong*, which is the worst failure mode a port has.

### Audio8, ranked by cost

1. **RoPE is interleaved, not half-split.** `_apply_rope` reshapes the last dim into adjacent
   `(real, imag)` pairs; candle's default `rotary_emb::rope` uses the half-split convention.
   You need `rope_i`.
2. **RoPE tables are built in bfloat16, then applied in fp32.** Replicate the bf16 round-trip
   on the table rather than computing fresh fp32 sin/cos.
3. **`_fast_step(hidden, 0)`'s result is discarded.** It exists only to prime the fast KV cache
   at position 0. Skip it and codebooks 1..9 are garbage.
4. **The top-k/top-p filter softmaxes *before* temperature** — the opposite of the conventional
   order. Replicate as written.
5. **Residual codebooks are size 1024, not 4096.** The fast head emits 4096-way logits but the
   quantizer clamps rows 1..9 to `0..1023`. Measured: 0 of 216 residual codes exceeded 1023, so
   the clamp is a safety net, not load-bearing — port it anyway, it is two `min` calls.
6. **Sampling is Gumbel-max, not multinomial**: `argmax(softmax(s) / -log(u))`.
7. **RAS draws twice per step**, in that order, substituting the second when the first repeats
   within a 10-token window.
8. **The RAS window initialises to zeros**, so the first 10 steps compare against token id 0
   and never trigger.

Also: **Audio8's reference sampler is broken under its own default dtype.** `_sample` draws
Gumbel noise at `dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct
values and output is unintelligible and never reaches EOS. Both implementations here draw in
f32.

### CosyVoice

1. **RoPE reaches head 0 only** in the reference.
2. **The two engines' RoPE conventions are opposite** — HF's Qwen2 uses `rotate_half`.
3. **The flow's initial noise is a fixed tensor, not a draw.** A port that samples its own
   looks correct and sounds different.
4. **`<|endofprompt|>` is required and nothing in the frontend adds it.** The LLM asserts on it.
5. **The upstream safetensors are the wrong weights.** 0 of 290 tensors match the `llm.model.*`
   tensors inside `llm.pt`, max relative difference 1.82. `llm.pt` is the fine-tune and the only
   correct source — checked rather than assumed.
6. **The tokenizer needs ~280 special tokens added at construction.**
7. **The leaky-ReLU before `conv_post` has slope 0.01, not 0.1.**
8. **The NSF noise is not in the checkpoint and is not negligible.** `SineGen2` holds it as a
   plain attribute, not a registered buffer, so it is redrawn at construction and only
   reproducible because the config calls `torch.manual_seed(1986)` first. Zeroing it moves the
   waveform by max 0.164 against a signal of rms 0.078.
9. **The harmonic phase is numerically degenerate in f32.** The reference accumulates phase to
   1.7e7 radians, where one f32 ulp is a full radian. This port accumulates on the host in f64
   modulo one cycle — the one place it is deliberately *more* accurate than its reference, and
   it moves *closer* to the reference's output, not further.

### Qwen3-TTS

Params come from the safetensors headers, so it reads **2.10 B** rather than the 1.7 B in its
name, which counts only the language backbone. The port was written from the reference and the
checkpoint's tensor shapes *before* any model code — the same idea as fixtures-first, applied
one step earlier, to the traps.

---

## Serving and narration

`tts-serve` replaces the Python FastAPI service and speaks the same protocol, verified side by
side on the same request: identical WAV format (`RIFF`, PCM, mono, 16-bit), identical
`X-Audio-Seconds` / `X-Wall-Seconds` / `X-RTF` / `X-Audio-Format` / `Content-Disposition`
headers, identical auth (`Bearer` or `X-API-Key`, constant-time compare, `503` when no key is
configured), and errors as `{"detail": …}`. The audio differs, as it must — different RNG
streams.

| route | |
|---|---|
| `POST /tts`, `POST /tts/stream`, `GET /health`, `GET /v1/capabilities`, `GET /` | served |
| `/v1/tts-jobs`, `/v1/alignment-jobs`, `/v1/artifacts/…` | **501 with an explanation**, not 404 |

`mode=instruct`, `mode=cross_lingual`, `speed != 1.0` and `instruct_text` also return 501.
**Refusing rather than ignoring is the rule**: returning speed-1.0 audio to a client that asked
for 1.5 would report the request as honoured when it was not. Two optional additions the Python
schema lacks: a per-request `voice`, and a `seed`. An extra `X-Stages` header carries the
per-stage split so a client can see where time went without a second request.

One GPU, so synthesis is serialised behind a semaphore — two requests interleaving on one Metal
queue make both slower and neither faster — and runs on `spawn_blocking` so it never occupies
an async worker. `/tts/stream` is buffered, not incremental.

### Narrating a long document

```sh
scripts/narrate-book.sh --book path/to/document --out narration --engine qwen3tts
scripts/verify-narration.py narration/*.webm
```

One engine load for the whole run, resumable per *stage* (a section with a WAV master is never
re-synthesised), deterministic under a seed. WebM/Opus at 48 kbps for delivery, because
Safari's Opus-in-Ogg support is unreliable and the failure mode is silence.

**Timings are derived from recognising the audio and matching it to the source text**, not from
placing known words into assumed windows. That shortcut measured a median error of **4.8 s per
word** while reporting 99.4% of words "aligned". Every section reports the share of words
carrying a measured time, the longest run it could not measure, and how many cue boundaries
land on a silence `ffmpeg` detected independently.

Text preparation carries most of the quality — stripping what is not speech and rewriting what
the voice reads wrong. Defects were found by aggregating alignment manifests rather than by
listening at random: a word the voice mangles is never recognised, so it shows up as
interpolated in every occurrence, which turns a 146,000-word book into a short list of suspects
for free.

---

## What did not work

| | why |
|---|---|
| **ONNX Runtime** | op coverage gaps, and no path that beat candle on Metal |
| **CoreML** | 19–26 s compilations, and coverage holes on the ops that mattered |
| **A custom q8_0 GEMM** | amortised up to 5.2× where candle's does 1.0×, validated against candle at real widths for m = 1..8 — but no configuration beat MPS f16 in absolute terms |
| **f32 weights for `qwen3tts`** | 7.45× per lane in isolation but RTF **6.25** in a real render at 6.97 GB resident. This is the wall that made `q8_0` the default |
| **Device-side sampling** | moved less than the transfer it saved |
| **f16 codec decoder** | quality loss without a speed win |
| **Padded decode attention** | superseded by fused decode attention |

The one that keeps paying: **candle is 1.70–2.23× slower than torch on this codec**, stable
across every thermal state, which is what motivated the custom Metal kernels in `tts-nn`.
