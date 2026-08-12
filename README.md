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
the configuration it ships in, in two cloned voices, on one M4 / 16 GB laptop with no Python
running. GitHub cannot embed audio in markdown — the
[demo page](https://drmhse.github.io/tts-rs/) plays all six in place:

| engine | RTF | wall time | audio produced | reach for it when |
|---|---|---|---|---|
| `audio8` | 0.527–0.536 | 5m 47s | 11:34 / 10:59 | you want 44.1 kHz — the highest-fidelity output here |
| `cosyvoice` | 0.703–0.718 | 8m 15s | 12:48 / 11:44 | you want the widest language coverage |
| `qwen3tts` | **0.252–0.261** | **2m 40s** | 11:36 / 10:34 | you are narrating something long |
