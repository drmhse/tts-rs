# tts-rs

Local text-to-speech in Rust. Text in, speech out, one process, no Python at runtime —
three voice-cloning speech models ported from PyTorch and validated stage by stage against
fp32 activations dumped from their references.

**Requirements: macOS on Apple silicon.** The custom kernels are Metal, and every number here
was measured on an M4 / 16 GB. It builds and runs correctly elsewhere with
`--no-default-features` (CPU fallbacks, unit-tested), but that path is a portability
guarantee rather than a deployment target — Audio8 measures RTF 2.151 on CPU against 0.554 on
Metal, and it is the codec that suffers most (1.235 of it, against 0.214 on Metal).

```sh
./scripts/bootstrap.sh          # toolchain, all three checkpoints, assets, build

cargo run -p tts-cli --release -- speak \
    --engine audio8 --voice voices/cosy-default \
    --text "Hello from a fresh checkout." --out hello.wav
```

One command, nothing manual: `bootstrap.sh` installs all three engines, their fixtures, and
builds. Budget ~13 GB of disk and a long first run — almost all of it download. Pass
`--audio8-only` to stop after the first engine, at ~4 GB. Every step is skipped if its output
already exists, so re-running is cheap. Details in
**[docs/reference.md](docs/reference.md#setup)**.

## What you can do with it

| | |
|---|---|
| **Clone a voice and speak** | a 10-second reference clip becomes a tracked voice asset; no encoder at runtime |
| **Serve it over HTTP** | `tts-serve`, wire-compatible with the Python FastAPI service it replaces |
| **Use it as a library** | one `Engine` trait, engines chosen by string id at request time |
| **Narrate a whole book** | markdown in, delivery audio plus word-level timings out, resumable per stage |

## The engines — hear them, and what they cost

### ▶ [Listen to all three side by side →](https://drmhse.github.io/tts-rs/)

All three narrating the same 1612-word chapter (`examples/chapter.txt`, in the repo), each in
the configuration it ships in, on one M4 / 16 GB laptop with no Python running. GitHub strips
audio players out of markdown, so the links below download — the
[demo page](https://drmhse.github.io/tts-rs/) plays them in place:

| engine | 45 s sample | full render | wall time | RTF | reach for it when |
|---|---|---|---|---|---|
| `audio8` | [▶](examples/samples/sample-audio8.m4a) | [11:34](https://huggingface.co/datasets/drmhse/tts-rs-assets/resolve/main/samples/chapter-audio8.webm) | 6 min 12 s | **0.536** | you want 44.1 kHz — the highest-fidelity output here |
| `cosyvoice` | [▶](examples/samples/sample-cosyvoice.m4a) | [12:48](https://huggingface.co/datasets/drmhse/tts-rs-assets/resolve/main/samples/chapter-cosyvoice.webm) | 9 min 12 s | **0.718** | you want the widest language coverage |
| `qwen3tts` | [▶](examples/samples/sample-qwen3tts.m4a) | [11:36](https://huggingface.co/datasets/drmhse/tts-rs-assets/resolve/main/samples/chapter-qwen3tts.webm) | **3 min 2 s** | **0.261** | you are narrating something long |

**That bottom row is the point of the project: a chapter becomes 11 minutes of speech in 3
minutes, on a laptop.** A 16-hour book is about 4 hours of compute rather than 12.

Compare the wall-time column, not just RTF. The three do not produce the same duration from
the same text — `cosyvoice` speaks slowest, 12:48 against `audio8`'s 11:34 — and since RTF
divides by audio produced, a slower-speaking engine flatters its own RTF.

```sh
cargo run -p tts-cli --release -- speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --quant f16 \
    --text-file examples/chapter.txt --out chapter.wav
```

`qwen3tts` gets there by batching across sections, which needs `--quant f16` and needs length —
on a 7-segment passage it is the *slowest* of the three at 0.665. It also supports ten languages
only (en, de, es, zh, ja, fr, ko, ru, it, pt). The other two do not batch meaningfully and are
steady at any length; `audio8` is **2.36× its PyTorch reference** like-for-like, that reference
running on MPS too.

Short-passage figures, for comparison — `examples/senior.txt`, 132 words, median of five with
the engines interleaved: `audio8` 0.554, `cosyvoice` 0.726, `qwen3tts` 0.665.

### How it validates

```sh
./scripts/gates.sh          # skips any tier whose inputs are missing, never silently passes
```

| gate | |
|---|---|
| `audio8` | 8 checks; greedy generation **bit-identical** to the reference |
| `cosyvoice` | 27 checks; teacher-forced argmax **105/105 identical**; WER 0.000–0.015 |
| `qwen3tts` | 65 rows; argmax codebook 0 identical, predictor **15/15** |

Fixtures are per stage, so a failure localises to a stage rather than to "the audio sounds
wrong". That is what made Audio8's codec validate at 2.8e-6 first try and localised CosyVoice's
reversed RoPE convention to one line.

Sampled output is deliberately **not** gated on equality: `ras_sampling` draws from torch's
generator, so a free-running sequence is not reproducible across implementations. The gates
check prefill logits and a greedy rollout; quality is checked separately by WER.

## Honest limitations

Read this before quoting any number above.

- **Two stages regressed and it is not diagnosed.** `audio8`'s codec is 35% slower than when
  first measured and `cosyvoice`'s vocoder 32%, while every transformer stage is within 1%.
  Both are convolution-heavy. This is why `audio8` reads 0.554 where this README once
  claimed 0.499.
- **`cosyvoice`'s speedup over PyTorch is 1.05×, not 6×.** Stock upstream measures 4.370 here,
  but only because `CosyVoiceModel` hardcodes `cuda if available else cpu` and has no MPS path
  at all, so CPU is all it can do. The adapted MPS service reaches ~0.76. The 6× figure is
  real and useless.
- **Memory is not characterised.** An earlier claim here that it was "flat by construction" was
  wrong, and two attempts to measure it both saturated. What can be said: one process renders
  16 minutes of audio on a 16 GB machine without the system struggling, and two concurrent
  engines do not fit.
- **Timings on this hardware need a statement of what else was running.** An M4 under sustained
  GPU load drifts ~2×, which once silently invalidated a session of measurements, and an
  unsynchronised stage timer once misattributed 13% of a pipeline to the wrong stage. The
  protocol that catches both is in
  [docs/reference.md](docs/reference.md#how-to-measure-without-fooling-yourself).
- **No durable job queue and no forced alignment** in `tts-serve`. Both answer `501` rather
  than pretending.

Two bugs in the references, worth knowing if you are porting these models yourself. **Audio8's
sampler is broken under its own default dtype** — `_sample` draws Gumbel noise at
`dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct values and output
is unintelligible and never reaches EOS; both implementations here draw in f32. And
**CosyVoice's harmonic phase is numerically degenerate in f32** — the reference accumulates
phase to 1.7e7 radians, where one f32 ulp is a full radian, so this port accumulates on the
host in f64 modulo one cycle. That is the only place either port is deliberately more accurate
than its reference, and it moves *closer* to the reference's output.

## Serving over HTTP

`tts-serve` speaks the same protocol as the Python FastAPI service it replaces — same routes,
request bodies, response headers, auth — so a client switches by pointing at a different
process.

```sh
TTS_API_KEY=… cargo run -p tts-serve --release -- --port 3003
curl -X POST localhost:3003/tts -H "X-API-Key: $TTS_API_KEY" \
     -H 'content-type: application/json' -d '{"text":"Hello."}' -o out.wav
```

Verified side by side on the same request: identical WAV format, headers and auth,
**RTF 0.71 against 0.80** and **3.0 s to load against 15–17 s**.
**[docs/reference.md](docs/reference.md#serving-and-narration)**.

## Narrating long documents

```sh
scripts/narrate-book.sh --book path/to/document --out narration --engine qwen3tts
scripts/verify-narration.py narration/*.webm
```

One engine load for the whole run, resumable per *stage* (a section with a WAV master is never
re-synthesised), deterministic under a seed. A 16-hour document is about **4 hours** of
synthesis at `qwen3tts`'s 0.260, against ~12 at `cosyvoice`'s 0.726, plus an hour of
recognition either way.

Timings come from recognising the audio and matching it to the source text, not from placing
known words into assumed windows — that shortcut measured a **median error of 4.8 s per word**
while reporting 99.4% of words "aligned". Every section reports the share of words carrying a
measured time, the longest run it could not measure, and how many cue boundaries land on a
silence `ffmpeg` detected independently.

## Using it as a library

```rust
use tts_core::{EngineConfig, SynthesisRequest, Voice};

let config = EngineConfig::new(tts_engines::default_root("cosyvoice"));
let engine = tts_engines::load("cosyvoice", &config)?;

let voice = Voice::load("voices/cosy-default-cosyvoice")?;
let request = SynthesisRequest::new("Hello from Rust.").with_voice(voice);
engine.validate(&request)?;                 // rejects a mismatched asset up front

let out = engine.synthesize(&request)?;
tts_core::wav::write("hello.wav", &out.audio)?;
println!("RTF {:.3}", out.stats.rtf(out.audio.seconds()));
```

That snippet is the doctest on `tts_engines`, so it is compiled on every `cargo test` rather
than left to rot. A voice asset built for one engine is refused by another rather than silently
substituted, and an engine that exists but cannot run yet reports `available: false` with a
reason instead of disappearing from the list.

## Documentation

Everything else is one file: **[docs/reference.md](docs/reference.md)**.

| | |
|---|---|
| [Setup](docs/reference.md#setup) | fresh clone to working audio, in three levels |
| [Architecture](docs/reference.md#architecture) | the engine trait, voice assets, adding an engine |
| [Validation](docs/reference.md#validation) | what each gate proves, and what is deliberately not gated |
| [Performance](docs/reference.md#performance) | the numbers, the measurement protocol, and two open regressions |
| [Porting traps](docs/reference.md#porting-traps) | eight Audio8 and nine CosyVoice traps, each of which produced plausible-but-wrong output |
| [Serving and narration](docs/reference.md#serving-and-narration) | the HTTP service, and markdown to audiobook |
| [What did not work](docs/reference.md#what-did-not-work) | ONNX, CoreML, a custom q8_0 GEMM, and four others |

## Layout

```
crates/tts-core/        the Engine trait, voice assets, segmentation, WAV, the PRNG
crates/tts-nn/          shared model machinery + the custom Metal kernels
crates/tts-engines/     the registry — the one place that knows which engines exist
crates/tts-cli/         the `tts` binary: engines / voice / speak
crates/tts-serve/       the HTTP service, wire-compatible with the Python one
crates/tts-bench/       the thermally-honest measurement harness
crates/tts-probe/       op-level benchmarks, one binary per question
crates/{audio8,cosyvoice,qwen3tts}/   one engine each, plus its fixture gate

references/{audio8,cosyvoice,qwen3tts}/   the PyTorch side: conversion, fixtures, quality
fixtures/{audio8,cosyvoice,qwen3tts}/     per-stage ground truth the gates compare against
voices/                 voice assets, one directory each (tracked — they are small)
examples/               senior.txt and chapter.txt, the two benchmark fixtures
scripts/                bootstrap, fetch-assets, gates, render-examples, narration
```

Everything shared is named `tts-*`; everything engine-specific is named for its engine, and
`crates/audio8`, `crates/cosyvoice` and `crates/qwen3tts` match the ids `--engine` takes.

Weights and virtualenvs are not tracked and [docs/reference.md](docs/reference.md#setup) builds
them. Fixtures are fetched rather than regenerated — `scripts/fetch-assets.sh` pulls ~130 MB of
checksummed ground truth from
[`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets), so
`./scripts/gates.sh` runs without any PyTorch.

## Licensing

Apache-2.0 ([LICENSE](LICENSE)). Contains **no model weights** — you download those yourself,
and all three are Apache-2.0 per their model cards:
[Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b),
[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B),
[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

Two kinds of *derived* artifact are distributed: the voice assets in `voices/`, encoded from
reference audio those models ship, and the fixtures in
[`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets). The upstream
licences reach both; attribution and the statement of changes are in [NOTICE](NOTICE).
