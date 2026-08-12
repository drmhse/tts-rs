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

One command, nothing manual. `bootstrap.sh` downloads and converts the checkpoints, fetches
the fixtures for every gate, and builds. **Name the engines you want** — all three is ~13 GB,
one is ~4 GB:

```sh
./scripts/bootstrap.sh --list              # the ids, their models, what each costs
./scripts/bootstrap.sh qwen3tts            # just one
./scripts/bootstrap.sh audio8 cosyvoice    # two
```

Every step is skipped if its output already exists, so re-running is cheap. Details in
**[docs/reference.md](docs/reference.md#setup)**.

## What you can do with it

| | |
|---|---|
| **Clone a voice and speak** | a 10-second reference clip becomes a tracked voice asset; no encoder at runtime |
| **Serve an HTTP API** | `tts-serve` — one engine loaded once, 3.0 s start, per-request voice and seed, cost headers on every response |
| **Use it as a library** | one `Engine` trait, engines chosen by string id at request time |
| **Narrate a whole book** | markdown in, delivery audio plus word-level timings out, resumable per stage |

## The engines — hear them, and what they cost

### ▶ [Listen to all three side by side →](https://drmhse.github.io/tts-rs/)

All three narrating the same 1612-word chapter (`examples/chapter.txt`, in the repo), each in
the configuration it ships in, in two cloned voices, on one M4 / 16 GB laptop with no Python
running. GitHub cannot embed audio in markdown — the
[demo page](https://drmhse.github.io/tts-rs/) plays all six in place:

| engine | RTF | wall time | audio produced | reach for it when |
|---|---|---|---|---|
| `audio8` | 0.527–0.536 | 5m 47s | 11:34 / 10:59 | you want 44.1 kHz — the highest-fidelity output here |
| `cosyvoice` | 0.703–0.718 | 8m 15s | 12:48 / 11:44 | you want the widest language coverage |
| `qwen3tts` | **0.252–0.261** | **2m 40s** | 11:36 / 10:34 | you are narrating something long |

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

## Limitations

- **Ten languages on `qwen3tts`**, a closed list: en, de, es, zh, ja, fr, ko, ru, it, pt. Text
  outside it has no faithful path through that engine.
- **A known performance regression, undiagnosed.** `audio8`'s codec and `cosyvoice`'s vocoder
  are 35% and 32% slower than when first measured, while every transformer stage is unchanged.
  Both are convolution-heavy; the cause is likely the channels-last conv path.
- **Memory is not characterised.** One process renders 16 minutes of audio on a 16 GB machine
  comfortably, and two engines do not fit at once. Beyond that, two attempts to measure it both
  saturated, so there is no peak figure to quote.
- **Speed varies with thermal state.** An M4 under sustained GPU load drifts up to 2×, so
  measure on your own machine, idle, before planning around any number here. The protocol is
  in [docs/reference.md](docs/reference.md#how-to-measure-without-fooling-yourself).
- **`/tts/stream` is buffered**, not incremental — it returns the whole utterance. There is no
  durable job queue and no forced-alignment endpoint; both answer `501`.
- **Off Apple silicon it is correctness, not speed.** The CPU fallbacks are unit-tested against
  the kernels, at RTF 2.151 for `audio8` against 0.554 on Metal.
- **Sampled output is not reproducible across implementations**, because the reference draws
  from torch's RNG. Pass a `seed` for repeatability within this port; use the greedy path if
  you need to compare against PyTorch.

One comparison worth not making: `cosyvoice` looks 6× faster than its PyTorch reference, but
only because upstream hardcodes `cuda if available else cpu` and has no MPS path, so CPU is all
it can do. Against a service that does use MPS it is ahead by about 5%.

## Serving it as an HTTP API

One engine, loaded once, answering requests — no per-request model load, which is where the
**3.0 s start** beats the Python service's 15–17 s. Wire-compatible with the FastAPI service it
replaces, so an existing client switches by changing port.

```sh
TTS_API_KEY=secret cargo run -p tts-serve --release -- --port 3003

curl -X POST localhost:3003/tts -H "X-API-Key: secret" \
     -H 'content-type: application/json' \
     -d '{"text":"Hello from Rust.","voice":"voices/cosy-default-male","seed":7}' \
     -o out.wav -D headers.txt
```

| route | |
|---|---|
| `POST /tts` | WAV body, PCM s16le mono |
| `POST /tts/stream` | same, buffered rather than incremental |
| `GET /v1/capabilities` | engines, sample rates, and the weight formats each supports |
| `GET /health` | liveness |
| `GET /` | lists the live routes **and** the unimplemented ones |

**Every response carries its own cost.** `x-audio-seconds`, `x-wall-seconds`, `x-rtf`, and
`x-stages` with the per-stage split (`llm=10.296,flow=25.129,vocoder=2.898`), so a client sees
where the time went without a second request. **`voice` and `seed` are per request** — the
first selects a voice asset without a restart, the second makes a render reproducible.

## Narrating long documents

```sh
scripts/narrate-book.sh --book path/to/document --out narration --engine qwen3tts
scripts/verify-narration.py narration/*.webm
```

One engine load for the whole run, resumable per *stage* (a section with a WAV master is never
re-synthesised), deterministic under a seed. A 16-hour document is about **4 hours** of
synthesis at `qwen3tts`'s 0.260, against ~12 at `cosyvoice`'s 0.726, plus an hour of
recognition either way.

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
