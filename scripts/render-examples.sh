#!/usr/bin/env bash
#
# Regenerate the audio in examples/.
#
# The renders are not tracked — they are 34 MB and they change whenever the sampler,
# the seed, or the engine does. examples/README.md describes what each one is *for*;
# this produces them.
#
# Needs both engines set up (docs/setup.md) and the PyTorch references for the
# `*_torch` / `senior.wav` comparisons, which this script does not attempt.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
run() { cargo run -q -p tts-cli --release -- speak "$@"; }

say "audio8 -> examples/senior_rust.wav"
run --engine audio8 --voice voices/cosy-default \
    --text-file examples/senior.txt --out examples/senior_rust.wav

say "cosyvoice -> examples/cosy_senior.wav"
run --engine cosyvoice --voice voices/cosy-default-cosyvoice \
    --text-file examples/senior.txt --out examples/cosy_senior.wav

say "cosyvoice, short -> examples/cosy_short.wav"
run --engine cosyvoice --voice voices/cosy-default-cosyvoice \
    --text "The quick brown fox jumps over the lazy dog." \
    --out examples/cosy_short.wav

cat <<'MSG'

Done. The PyTorch reference renders (senior.wav, senior_batch8.wav,
senior_cosyvoice.wav, cosy_senior_torch.wav) come from oracle/synthesize.py and
oracle-cosy/reference_render.py — see examples/README.md.
MSG
