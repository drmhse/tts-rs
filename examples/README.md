# examples

One passage, both engines, and the PyTorch controls they are measured against.

**The audio here is not tracked** — 34 MB that changes with every sampler, seed and engine
change. `scripts/render-examples.sh` regenerates the Rust ones; the PyTorch controls come
from `references/audio8/synthesize.py` and `references/cosyvoice/reference_render.py` (see "Reproducing"
below). `senior.txt` is tracked, and so is the table of what each render *was*, because
that is the part worth keeping.

`senior.txt` is 132 words in 5 paragraphs, which `tts_core::text::segment` cuts into 7
segments at the default 220-character budget.

## Audio8 — 44.1 kHz

| file | produced by | |
|---|---|---|
| `senior.wav` | PyTorch, batch 4, bf16, Audio8's own voice | 54.54 s, 71.5 s wall → RTF 1.311 |
| `senior_batch8.wav` | PyTorch, all 7 segments in one batch | 67.0 s → RTF 1.260 |
| `senior_cosyvoice.wav` | PyTorch, cloning CosyVoice's default voice | 66.0 s → RTF 1.307 |
| **`senior_rust.wav`** | **Rust, q8_0, batched, same cloned voice** | **50.27 s, 25.1 s wall → RTF 0.499** |

**2.62x faster than the PyTorch path.** F0 within 5.9 Hz of the reference clip, LTAS cosine
0.9971, WER 0.008.

Segments decode eight at a time by default (`--set max_batch=1` for the sequential path).
Batching is worth **1.77x on the AR loop** measured interleaved, and it changes the audio —
per-sequence sampling consumes the RNG in a different order — so a batched render is a
different valid draw, not the same file. Checked rather than assumed: across `max_batch` 1, 2,
4 and 8 the F0 offsets are −2.0, −7.7, +0.0 and −5.9 Hz with LTAS 0.9968-0.9971 and WER 0.008
to 0.015. See `../../docs/performance/ar-loop.md`.

## CosyVoice — 24 kHz

| file | produced by | |
|---|---|---|
| **`cosy_senior.wav`** | **Rust** | **54.86 s, 38.2 s wall → RTF 0.697**, WER 0.000–0.015 |
| `cosy_senior_torch.wav` | stock PyTorch | 52.38 s, 228.9 s wall → RTF 4.370, WER 0.008 |
| `cosy_short.wav` | Rust, one sentence | 4.72 s — the single-utterance case, RTF 2.02 |

**6.27x faster than stock PyTorch**, but read the caveat: upstream's `CosyVoiceModel`
hardcodes `cuda if available else cpu` and has no MPS path, so CPU is the only thing it can do
here. The adapted service in `tts/CosyVoice` reaches RTF ~0.76 on MPS.

The flow decoder runs **once for the whole utterance**, not once per segment — the voice's 588
prompt mel frames were otherwise re-decoded per segment, 61% of that stage's work. The waveform
is cut back into segments afterwards at exact token boundaries so the gaps survive.

`cosy_short.wav` is worth keeping next to the long render because it shows how much of a
single-shot number is Metal shader compilation: the same code is RTF 2.02 on one sentence and
0.70 across seven, with nothing changed but how much work there was to amortise it over.

`cosy_senior_torch.wav` is 32-bit float (torchaudio's default for a float tensor) where
everything else is 16-bit PCM. Cosmetic; it affects no measurement above.

## `diag/*.wav`

The Audio8 sampler bisection — 168 s of noise traced to `_sample` drawing its Gumbel uniforms
at `dtype=probabilities.dtype`, so bfloat16 gives ~256 distinct values and the ratio ordering
collapses. See `../../docs/status.md`.

## Qwen3-TTS — 24 kHz

| file | produced by | |
|---|---|---|
| `senior_qwen3tts.wav` | Rust, q8_0, cloning CosyVoice's default voice | 50.66 s, 43.7 s wall → **RTF 0.863** (talker 0.695, codec 0.168) |

Regenerate:

```sh
cargo run -p tts-cli --release -- speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --text-file examples/senior.txt \
    --out examples/senior_qwen3tts.wav
```

All three engines land within 4.6 s of each other on this passage (50.27 / 54.86 / 50.66 s),
which is the cheapest available check that none of them is truncating or rambling.

A PyTorch control needs the reference venv and is slow (CPU only, RTF ~4.8):

```sh
references/qwen3tts/.venv/bin/python references/qwen3tts/reference_render.py \
    --model references/qwen3tts/weights --audio ../CosyVoice/asset/default_voice.wav \
    --ref-text "<the clip's transcript, verbatim>" \
    --text "Hello from Rust." --out /tmp/ref.wav
```

**The transcript must match the clip.** Pairing that clip's transcript with a 4.72 s excerpt of
it made the port emit 197 frames for a one-sentence input and the PyTorch reference run to its
full 2048-frame cap. See `docs/porting/qwen3tts.md`.

## Reproducing

From the repo root, the Rust renders in one command:

```
./scripts/render-examples.sh
```

or individually:

```
cargo run -p tts-cli --release -- speak --engine audio8 \
    --text-file examples/senior.txt --voice voices/cosy-default \
    --out examples/senior_rust.wav

cargo run -p tts-cli --release -- speak --engine cosyvoice \
    --text-file examples/senior.txt --voice voices/cosy-default-cosyvoice \
    --out examples/cosy_senior.wav
```

The Audio8 PyTorch reference, from `references/audio8/`:

```
# Audio8's own voice
.venv/bin/python synthesize.py --text-file ../examples/senior.txt \
    --out ../examples/senior.wav --batch 4

# CosyVoice's default voice, and encode its codes for the Rust path
C=../../CosyVoice
.venv/bin/python synthesize.py --text-file ../examples/senior.txt \
    --out ../examples/senior_cosyvoice.wav --batch 4 \
    --reference-audio $C/asset/default_voice.wav \
    --reference-text $C/asset/default_voice.txt \
    --save-reference-codes ../fixtures/audio8/default_voice_codes.safetensors
```

`--save-reference-codes` is the only step that needs the codec *encoder*, which is why the
Rust binary does not contain one.

The CosyVoice PyTorch control, from the CosyVoice checkout:

```
PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
    <repo>/references/cosyvoice/reference_render.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --text-file <repo>/examples/senior.txt \
    --prompt-wav asset/default_voice.wav --prompt-text-file asset/default_voice.txt \
    --out <repo>/examples/cosy_senior_torch.wav
```

It segments with the same rule as the Rust engine, so the two renders differ in
implementation and not in how the text was cut up.

Scoring:

```
references/audio8/.venv/bin/python references/audio8/verify_voice.py \
    --reference ../CosyVoice/asset/default_voice.wav examples/*.wav

# from the CosyVoice venv, which has openai-whisper
.venv/bin/python <repo>/references/cosyvoice/wer.py \
    --text-file <repo>/examples/senior.txt <repo>/examples/cosy_senior.wav
```

## Why none of this is bit-reproducible

Sampling is stochastic and **no userspace RNG reproduces torch's Philox stream**, so
re-running any of the above gives a different token sequence and slightly different audio.
Even within PyTorch, `--seed 1234` is fixed but batch size changes how the generator is
consumed, which is why batch 4 and batch 8 render the same text differently.

That is why the fixture gates compare *greedy* and *teacher-forced* output rather than sampled
output, and why the F0 figures here are only quoted alongside a PyTorch control — a single
F0 number for one render says nothing without knowing how much the model itself moves between
runs. For CosyVoice that turned out to be ±13 Hz.
