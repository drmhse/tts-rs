# Setting up from scratch

Three levels, each useful on its own. Stop at whichever one you need.

| level | you get | needs |
|---|---|---|
| [1. Run Audio8](#1-run-audio8) | `tts speak --engine audio8` | Rust, python ≥ 3.10, ~4 GB disk |
| [2. Run CosyVoice](#2-run-cosyvoice) | `tts speak --engine cosyvoice` | the upstream CosyVoice repo, ~4 GB more |
| [3. Regenerate the fixtures](#3-regenerate-the-fixtures) | the correctness gates and the PyTorch baselines | both of the above, plus patience |

Voice assets are **already in the repo** (`voices/`, ~200 KB), so you do not need any
PyTorch to clone a voice — only to build a *new* one.

Everything below assumes macOS on Apple silicon. The engines build and run elsewhere,
but the Metal kernels in `tts-nn` and every performance figure in `docs/` do not apply;
use `--cpu` and expect correctness rather than speed.

---

## 1. Run Audio8

```sh
./scripts/bootstrap.sh
```

That checks your toolchain, downloads
[Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
(~2.4 GB) into `oracle/weights/`, creates `oracle/.venv`, folds the codec's `weight_norm`
into `oracle/weights/codec.safetensors`, and builds the workspace. It is idempotent —
re-running skips whatever is already done.

Then:

```sh
cargo run -p tts-cli --release -- speak \
    --engine audio8 --voice voices/cosy-default \
    --text "Hello from a fresh checkout." --out hello.wav
```

### Doing it by hand

The one step that is not a download is folding the codec. Audio8 ships `codec.pth`, a
1.35 GB pickle in which every convolution is wrapped in `weight_norm`, so the stored
parameters are a magnitude and a direction rather than a weight. Folding once at
conversion time means the Rust side memory-maps plain weights and does no
reparametrisation at runtime:

```sh
cd oracle
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python convert_codec.py --weights weights --out weights/codec.safetensors
```

---

## 2. Run CosyVoice

CosyVoice needs its own environment, and this is not something the bootstrap script can
do for you: converting the checkpoints is a pure `torch.load`, but **dumping its fixtures
imports CosyVoice model code**, which pins python 3.10 and torch 2.3.1. Trying to share
one environment with Audio8's torch 2.13 does not work.

**a. Get the upstream repo and model.** Follow
[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) to clone it, create its
venv, and download `Fun-CosyVoice3-0.5B` into its `pretrained_models/`.

**b. Convert the checkpoints** into this repo's layout. This only calls `torch.load`, so
it runs under either environment:

```sh
/path/to/CosyVoice/.venv/bin/python oracle-cosy/convert.py \
    --checkpoints /path/to/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    --out oracle-cosy/weights
```

It prints a tensor inventory as it goes. That inventory is the spec the Rust port was
written against — if it does not match `fixtures-cosy/oracle.json`, you have a different
model revision and the gates will fail for a good reason.

**c. Fetch the fixed noise tensor.** CosyVoice's vocoder draws NSF noise at construction
from a seeded RNG, so it is not in the checkpoint. `fixtures-cosy/rand_noise.safetensors`
is produced by step 3 below; without it the engine cannot start. If you are not
regenerating fixtures, copy it from a machine that has one.

```sh
cargo run -p tts-cli --release -- speak \
    --engine cosyvoice --voice voices/cosy-default-cosyvoice \
    --text-file examples/senior.txt --out cosy.wav
```

---

## 3. Regenerate the fixtures

The gates compare this port against per-stage fp32 activations dumped from PyTorch. The
tensors are too large to track (`fixtures/` and `fixtures-cosy/` are gitignored), but the
`.json` inventories beside them **are** tracked, so you can always see which tensors a
gate expects and with what shapes.

```sh
# Audio8 — from this repo's oracle venv
cd oracle && .venv/bin/python dump_fixtures.py --weights weights --out ../fixtures

# CosyVoice — from the CosyVoice repo, which must be on PYTHONPATH
cd /path/to/CosyVoice
PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
    /path/to/audio8-rs/oracle-cosy/dump_fixtures.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --voice /path/to/audio8-rs/voices/cosy-default-cosyvoice \
    --out /path/to/audio8-rs/fixtures-cosy
```

Then run everything:

```sh
./scripts/gates.sh
```

`gates.sh` skips any tier whose inputs are missing and says so rather than reporting a
pass it did not earn.

### Building a new voice

A voice asset is the reference clip's speech tokens, its mel, its speaker embedding, and
its transcript — precomputed so that inference needs no encoder. One directory, a
`voice.json` and a `voice.safetensors`, and assets are refused by any engine other than
the one they were built for.

```sh
# Audio8
cd oracle && .venv/bin/python export_voice.py \
    --weights weights --audio /path/to/clip.wav --text "what the clip says" \
    --name my-voice --out ../voices

# CosyVoice — from the CosyVoice repo
PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
    /path/to/audio8-rs/oracle-cosy/export_voice.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --audio /path/to/clip.wav --text "what the clip says" \
    --name my-voice --out /path/to/audio8-rs/voices
```

The transcript matters more than it looks: CosyVoice asserts that the prompt text
contains `<|endofprompt|>`, and nothing in its frontend adds it — the service does. See
trap 2 in [porting/cosyvoice.md](porting/cosyvoice.md).

### Quality measurement

The scripts that produce the WER and speaker-similarity numbers in `docs/` need a whisper
model and torchaudio:

```sh
# intelligibility — needs openai-whisper, which lives in the CosyVoice venv
/path/to/CosyVoice/.venv/bin/python oracle-cosy/wer.py \
    --text-file examples/senior.txt examples/cosy_senior.wav

# speaker similarity: median F0 and long-term average spectrum
/path/to/CosyVoice/.venv/bin/python oracle/verify_voice.py \
    --reference /path/to/reference_clip.wav examples/cosy_senior.wav
```

---

## Disk budget

| path | size | tracked? |
|---|---|---|
| `oracle/weights/` | ~3.4 GB | no — downloaded and converted |
| `oracle-cosy/weights/` | ~3.7 GB | no — converted from the CosyVoice repo |
| `fixtures/` | ~65 MB, plus 8.8 GB if you run the quantization probe | no |
| `fixtures-cosy/` | ~46 MB | no (the `.json` inventory is) |
| `voices/` | ~200 KB | **yes** |
| `target/` | ~1.8 GB | no |

The 8.8 GB is four `fixtures/ar_q*.safetensors` dumps written by
`a8-probe --bin qroundtrip` and read only by `oracle/quality_ar.py`, which asks whether
quantization changes the voice ([performance/quantization-quality.md](performance/quantization-quality.md)).
That question is answered; delete them unless you are re-opening it.
