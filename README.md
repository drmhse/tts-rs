# tts-rs

Local text-to-speech in Rust: text in, speech out, one process, bounded memory. Two
engines behind one interface, both faster than real time on an M4, both gated against
per-stage fp32 fixtures dumped from their PyTorch references.

| engine | model | RTF | vs PyTorch | validation |
|---|---|---|---|---|
| `audio8` | [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b), 601 M, 44.1 kHz | **0.499** | 2.62× | greedy generation **bit-identical** |
| `cosyvoice` | `FunAudioLLM/Fun-CosyVoice3-0.5B`, 995 M, 24 kHz | **0.697** | 6.27× | teacher-forced argmax **105/105 identical** |

## Quick start

```sh
./scripts/bootstrap.sh          # toolchain, checkpoint, codec conversion, build

cargo run -p tts-cli --release -- speak \
    --engine audio8 --voice voices/cosy-default \
    --text "Hello from a fresh checkout." --out hello.wav
```

`bootstrap.sh` sets up Audio8 only. CosyVoice needs the upstream CosyVoice repository for
its checkpoints — a few more commands, in **[docs/setup.md](docs/setup.md)**.

```sh
cargo run -p tts-cli --release -- engines                      # what is available
cargo run -p tts-cli --release -- voice voices/cosy-default    # inspect an asset
./scripts/gates.sh                                             # everything verifiable
```

## Serving over HTTP

`tts-serve` replaces the Python FastAPI service on its own port and speaks the same
protocol — same routes, same request bodies, same response headers, same auth — so a
client switches by pointing at a different process.

```sh
TTS_API_KEY=… cargo run -p tts-serve --release -- --port 3003
curl -X POST localhost:3003/tts -H "X-API-Key: $TTS_API_KEY" \
     -H 'content-type: application/json' -d '{"text":"Hello."}' -o out.wav
```

Verified side by side against the Python service on the same request: identical WAV format,
identical headers, identical auth. **RTF 0.70 against 0.80** over HTTP on a 132-word
passage, and **3–4 s to load against 15–17 s**. The durable job queue and forced alignment
are not implemented and answer `501` rather than pretending. Details, and what it refuses
to do, in **[docs/serving.md](docs/serving.md)**.

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

That snippet is the doctest on `tts_engines`, so it is compiled on every `cargo test`
rather than left to rot.

Engines are chosen by string id at request time. A voice asset built for one engine is
refused by another rather than silently substituted, and an engine that exists but cannot
run yet reports `available: false` with a reason instead of disappearing from the list.

## Where it landed

Both on `examples/senior.txt` (132 words, 7 segments), M4 / 16 GB:

| engine | reference | this port | |
|---|---|---|---|
| `audio8` | RTF 1.307 (PyTorch, bf16, MPS, batched) | **RTF 0.499** | **2.62× faster**, identical WER |
| `cosyvoice` | RTF 4.370 (stock PyTorch — CPU-only, see below) | **RTF 0.697** | **6.27× faster**, WER 0.000–0.015 across seeds |

The CosyVoice comparison carries a caveat worth stating up front: upstream's
`CosyVoiceModel` hardcodes `cuda if available else cpu` and **has no MPS path at all**, so
CPU is the only thing stock upstream can do here. The adapted service in `tts/CosyVoice`
reaches RTF ~0.76 on MPS; this port is now faster than that. Details and the remaining
levers are in [docs/porting/cosyvoice.md](docs/porting/cosyvoice.md).

Model load is 1.0–1.5 s for Audio8 (including quantizing 417 M params) against 15–17 s for
the PyTorch service's reload path.

**Memory is not characterised, and the earlier claim here that it was "flat by construction"
was wrong.** There is no MPSGraph cache to reclaim, which is why there is no supervisor or
recycle budget — but that is an argument, not a measurement, and long renders do use more
than a short one. Two attempts to measure it both saturated: `/usr/bin/time -l` reports RSS,
which cannot exceed what is resident and pinned at 3.59 GB across a 6x range of input; and
`phys_footprint_peak` returned exactly 13.00 GB for three configurations that should differ
enormously. What can be said: **one process renders 16 minutes of audio at RTF 0.54 on a
16 GB machine without the system struggling, and two concurrent engines do not fit.**
See [docs/performance/memory.md](docs/performance/memory.md).

## Documentation

| | |
|---|---|
| [docs/setup.md](docs/setup.md) | from a fresh clone to working audio, in three levels |
| [docs/serving.md](docs/serving.md) | running it as an HTTP service, and the markdown-to-audiobook pipeline |
| [docs/status.md](docs/status.md) | **the map** — what exists, how it validates, how fast, what is left |
| [docs/architecture.md](docs/architecture.md) | the engine trait, voice assets, why the crates split as they do |
| [docs/benchmarking.md](docs/benchmarking.md) | how to measure on this hardware without fooling yourself |

<details>
<summary>Per-engine and per-investigation notes</summary>

| | |
|---|---|
| [docs/porting/audio8.md](docs/porting/audio8.md) | architecture and eight ranked traps, written before any code |
| [docs/porting/cosyvoice.md](docs/porting/cosyvoice.md) | the second engine: validation, speed, and nine silent traps |
| [docs/performance/ar-loop.md](docs/performance/ar-loop.md) | the AR loop is 10–20× the codec; the levers, and two wrong conclusions |
| [docs/performance/candle-on-metal.md](docs/performance/candle-on-metal.md) | where the time goes, and what candle does not fuse |
| [docs/performance/memory.md](docs/performance/memory.md) | what is known, and two metrics that saturated |
| [docs/performance/quantization-quality.md](docs/performance/quantization-quality.md) | q8_0 costs nothing audible; why token identity is the wrong metric |
| [docs/rejected/](docs/rejected/) | ONNX, CoreML, and an attic of experiments that did not work out |
| [crates/tts-probe/README.md](crates/tts-probe/README.md) | the measurement binaries, indexed by the question each answers |
| [examples/README.md](examples/README.md) | the renders, and how to reproduce them |

</details>

## Layout

```
crates/tts-core/        the Engine trait, voice assets, segmentation, WAV, the PRNG
crates/tts-nn/          shared model machinery + the custom Metal kernels
crates/tts-engines/     the registry — the one place that knows which engines exist
crates/tts-cli/         the `tts` binary: engines / voice / speak
crates/tts-serve/       the HTTP service, wire-compatible with the Python one
crates/tts-bench/       the thermally-honest measurement harness
crates/tts-probe/       op-level benchmarks, one binary per question (see its README)
crates/audio8/          the Audio8 engine + its fixture gate
crates/cosyvoice/       the CosyVoice engine + its fixture gate and bench

references/audio8/      Audio8's PyTorch side: conversion, fixtures, quality scripts
references/cosyvoice/   CosyVoice's: conversion, voice export, WER
fixtures/{audio8,cosyvoice}/    per-stage ground truth the gates compare against
voices/                 voice assets, one directory each (tracked — they are small)
scripts/                bootstrap, gates, render-examples
docs/                   setup, architecture, status, and the investigation record
```

Everything shared is named `tts-*`; everything engine-specific is named for its engine,
and the two engines' directories are symmetric. `crates/audio8` and `crates/cosyvoice`
match the ids `--engine` takes, so there is one name per thing.

Weights, fixtures and virtualenvs are not tracked; `docs/setup.md` regenerates them.

## Licensing

This repository is Apache-2.0 ([LICENSE](LICENSE)). It contains **no model weights** — you
download those yourself, and they carry their own terms:

| | licence | note |
|---|---|---|
| Audio8-TTS-Preview-0.6b | Apache-2.0 | from the model card on Hugging Face |
| Fun-CosyVoice3-0.5B | Apache-2.0 (upstream CosyVoice) | check the model card before redistributing — model weights and code can differ |

The voice assets in `voices/` are derived from reference audio shipped with those models,
so the same terms reach them.

## Running without a GPU

`--no-default-features` drops the Metal kernels and builds a portable CPU-only binary; it
is also the only configuration that builds off Apple platforms, since candle's `metal`
feature does not exist there.

```sh
cargo run -p tts-cli --release --no-default-features -- speak --cpu \
    --engine audio8 --voice voices/cosy-default --text "Hello." --out hello.wav
```

Correctness is unchanged — the custom kernels each carry a CPU fallback that the unit
tests check them against. Speed is not: Audio8 measures **RTF 3.39** on CPU against 0.499
on Metal, so treat CPU as a portability guarantee rather than a deployment target.

## Three things a reader should know up front

**Audio8's reference sampler is broken under its own default dtype.** `_sample` draws its
Gumbel noise at `dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct
values and sampled output is unintelligible and never reaches EOS. Both implementations
here draw in f32.

**CosyVoice's harmonic phase is numerically degenerate in f32.** The reference accumulates
phase to 1.7e7 radians, where one f32 ulp is a full radian. This port accumulates it on the
host in f64 modulo one cycle — the only place either port is deliberately more accurate
than its reference, and it moves *closer* to the reference's output, not further.

**Timings on this hardware are only meaningful with a statement of what else was
running.** An M4 under sustained GPU load drifts ~2×, which silently invalidated a
session's worth of measurements, and an unsynchronised stage timer once misattributed 13%
of a pipeline to the wrong stage. Both failure modes, and the protocol that catches them,
are in [docs/benchmarking.md](docs/benchmarking.md).
