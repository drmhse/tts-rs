# tts-rs

Local text-to-speech in Rust: text in, speech out, one process, bounded memory. Three
engines behind one interface, all faster than real time on an M4, all gated against
per-stage fp32 fixtures dumped from their PyTorch references.

| engine | model | RTF | vs reference | validation |
|---|---|---|---|---|
| `audio8` | [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b), 601 M, 44.1 kHz | **0.554** | **2.36×** (PyTorch bf16 MPS, 1.307) | greedy generation **bit-identical**, identical WER |
| `cosyvoice` | `FunAudioLLM/Fun-CosyVoice3-0.5B`, 995 M, 24 kHz | **0.726** | **6.02×** (stock PyTorch CPU, 4.370) | teacher-forced argmax **105/105**, WER 0.000–0.015 |
| `qwen3tts` | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), 2.10 B, 24 kHz | **0.665** | no reference render | argmax codebook 0 **identical**, predictor **15/15** |

`examples/senior.txt` (132 words, 7 segments), M4 / 16 GB, `q8_0`, median of five samples with
the engines interleaved in one session. CosyVoice's reference is CPU-only because upstream
hardcodes `cuda if available else cpu` and has no MPS path at all — the adapted MPS service
reaches ~0.76, which this beats by 5% rather than by 6×.

**`qwen3tts`'s row is the configuration nobody runs it in.** Seven segments is too few for the
batching it depends on. On `examples/chapter.txt` — 1612 words, 100 segments, also in the repo —
it measures **RTF 0.260 at `--quant f16`**, against 0.661 for the same chapter at `q8_0`:

```sh
cargo run -p tts-cli --release -- speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --quant f16 \
    --text-file examples/chapter.txt --out chapter.wav
```

`q8_0` gains nothing from 14× more segments (0.665 → 0.661) because candle's quantized `mm_t`
pads to a large row tile; `f16` gets 2.54×, all of it in the talker. That is why the narration
path uses `f16` and why this is the engine to point at a book.

## Quick start

```sh
./scripts/bootstrap.sh          # toolchain, checkpoint, assets, build

cargo run -p tts-cli --release -- speak \
    --engine audio8 --voice voices/cosy-default \
    --text "Hello from a fresh checkout." --out hello.wav
```

`bootstrap.sh` sets up Audio8 and fetches the fixtures for all three gates. CosyVoice and
Qwen3-TTS each need their own checkpoint downloaded — a few more commands, in
**[docs/reference.md](docs/reference.md#setup)**.

```sh
cargo run -p tts-cli --release -- engines                      # what is available
cargo run -p tts-cli --release -- voice voices/cosy-default    # inspect an asset
./scripts/gates.sh                                             # everything verifiable
```

## Three things to know before trusting any of this

**Audio8's reference sampler is broken under its own default dtype.** `_sample` draws its
Gumbel noise at `dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct
values and sampled output is unintelligible and never reaches EOS. Both implementations
here draw in f32.

**CosyVoice's harmonic phase is numerically degenerate in f32.** The reference accumulates
phase to 1.7e7 radians, where one f32 ulp is a full radian. This port accumulates it on the
host in f64 modulo one cycle — the only place either port is deliberately more accurate than
its reference, and it moves *closer* to the reference's output, not further.

**Timings here are only meaningful with a statement of what else was running.** An M4 under
sustained GPU load drifts ~2×, which silently invalidated a session's worth of measurements,
and an unsynchronised stage timer once misattributed 13% of a pipeline to the wrong stage.
Both failure modes and the protocol that catches them are in
[docs/reference.md](docs/reference.md#how-to-measure-without-fooling-yourself).

Memory is **not** characterised — an earlier claim here that it was "flat by construction"
was wrong, and two attempts to measure it both saturated. What can be said: one process
renders 16 minutes of audio on a 16 GB machine without the system struggling, and two
concurrent engines do not fit. [docs/reference.md](docs/reference.md#memory-and-quantization-quality).

## Serving over HTTP

`tts-serve` replaces the Python FastAPI service on its own port and speaks the same protocol
— same routes, request bodies, response headers, auth — so a client switches by pointing at
a different process.

```sh
TTS_API_KEY=… cargo run -p tts-serve --release -- --port 3003
curl -X POST localhost:3003/tts -H "X-API-Key: $TTS_API_KEY" \
     -H 'content-type: application/json' -d '{"text":"Hello."}' -o out.wav
```

Verified side by side against the Python service on the same request: identical WAV format,
headers and auth, **RTF 0.70 against 0.80** and **3–4 s to load against 15–17 s**. The durable
job queue and forced alignment are not implemented and answer `501` rather than pretending.
What it refuses to do, in **[docs/reference.md](docs/reference.md#serving-and-narration)**.

## Narrating long documents

Markdown in, delivery audio plus word-level timings out, for a document of any length:

```sh
scripts/narrate-book.sh --book path/to/document --out narration --engine qwen3tts
scripts/verify-narration.py narration/*.webm
```

One engine load for the whole run, resumable per *stage*, deterministic under a seed, and
gated: every section reports the share of words carrying a time measured from the audio, the
longest run it could not measure, and how many cue boundaries land on a silence `ffmpeg`
detected independently.

Timings come from recognising the audio and matching it to the source text, not from placing
known words into assumed windows — that shortcut measured a **median error of 4.8 s per word**
while reporting 99.4% of words "aligned". `qwen3tts` at `f16` is the default here because
batching amortises across sections: a 16-hour document is about **4 hours** of synthesis at its
measured 0.260, against ~12 at `cosyvoice`'s 0.726, plus an hour of recognition either way. The
reasoning and the text rules that keep the voice from reading markup aloud are in
**[docs/reference.md](docs/reference.md#serving-and-narration)**.

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
than left to rot. Engines are chosen by string id at request time; a voice asset built for one
engine is refused by another rather than silently substituted, and an engine that exists but
cannot run yet reports `available: false` with a reason instead of disappearing from the list.

## Documentation

Everything else is one file: **[docs/reference.md](docs/reference.md)**.

| | |
|---|---|
| [Setup](docs/reference.md#setup) | fresh clone to working audio, in three levels |
| [Architecture](docs/reference.md#architecture) | the engine trait, voice assets, adding an engine |
| [Validation](docs/reference.md#validation) | what each gate proves, and what is deliberately not gated |
| [Performance](docs/reference.md#performance) | the numbers, the measurement protocol, and two open regressions |
| [Porting traps](docs/reference.md#porting-traps) | the eight Audio8 and nine CosyVoice traps, each of which produced plausible-but-wrong output |
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
crates/tts-probe/       op-level benchmarks, one binary per question (see its README)
crates/{audio8,cosyvoice,qwen3tts}/   one engine each, plus its fixture gate

references/{audio8,cosyvoice,qwen3tts}/   the PyTorch side: conversion, fixtures, quality
fixtures/{audio8,cosyvoice,qwen3tts}/     per-stage ground truth the gates compare against
voices/                 voice assets, one directory each (tracked — they are small)
scripts/                bootstrap, fetch-assets, gates, render-examples, narration
docs/                   setup, architecture, status, and the investigation record
```

Everything shared is named `tts-*`; everything engine-specific is named for its engine, and
`crates/audio8`, `crates/cosyvoice` and `crates/qwen3tts` match the ids `--engine` takes, so
there is one name per thing.

Weights and virtualenvs are not tracked and [docs/reference.md](docs/reference.md#setup) builds them. Fixtures are fetched
rather than regenerated — `scripts/fetch-assets.sh` pulls ~130 MB of checksummed ground truth
from [`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets), so
`./scripts/gates.sh` runs without any PyTorch. Rebuilding from source is
[docs/reference.md](docs/reference.md#3-regenerating-the-fixtures).

## Running without a GPU

`--no-default-features` drops the Metal kernels for their CPU fallbacks, which the unit tests
check them against. It is also the only configuration that builds off Apple platforms, since
candle's `metal` feature does not exist there.

```sh
cargo run -p tts-cli --release --no-default-features -- speak --cpu \
    --engine audio8 --voice voices/cosy-default --text "Hello." --out hello.wav
```

Correctness is unchanged; speed is not. Audio8 measures **RTF 3.39** on CPU against 0.554 on
Metal, so treat CPU as a portability guarantee rather than a deployment target.

## Licensing

This repository is Apache-2.0 ([LICENSE](LICENSE)). It contains **no model weights** — you
download those yourself, and all three are Apache-2.0 per their model cards:
[Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b),
[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B),
[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

Two kinds of *derived* artifact are distributed: the voice assets in `voices/`, encoded from
reference audio those models ship, and the fixtures in
[`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets). The upstream
licences reach both; attribution and the statement of changes are in [NOTICE](NOTICE).
