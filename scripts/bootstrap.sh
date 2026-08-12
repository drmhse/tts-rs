#!/usr/bin/env bash
#
# Get this repo from a fresh clone to a working `tts speak`.
#
# What it does:
#   1. checks the toolchain and the platform
#   2. downloads the Audio8 checkpoint (~2.4 GB) from Hugging Face
#   3. creates references/audio8/.venv and folds the codec's weight_norm into safetensors
#   4. fetches the derived assets (~130 MB): the fixture oracles for all three gates, and
#      the two CosyVoice artifacts that need the upstream python package to produce
#
# What it deliberately does *not* do: download the CosyVoice and Qwen3-TTS checkpoints.
# Those are gigabytes each and only two of the three engines need them, so `docs/reference.md#setup`
# has the commands and you choose. CosyVoice no longer needs the upstream *repository*
# though — step 4 fetches the two artifacts that used to require it.
#
# Re-running is safe: every step is skipped if its output already exists. Pass --force
# to redo the conversion.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
FORCE="${1:-}"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ 1. toolchain

say "Checking the toolchain"

command -v cargo >/dev/null || die "cargo not found — install Rust from https://rustup.rs"
cargo --version

if [ "$(uname -s)" != "Darwin" ]; then
  warn "This is not macOS. The engines build and run on CPU, but every performance"
  warn "number in docs/ is Metal, and the custom kernels in tts-nn are Metal-only."
  warn "Expect correctness, not speed."
elif [ "$(uname -m)" != "arm64" ]; then
  warn "Not Apple silicon — Metal will be unavailable or slow. Use --cpu."
fi

VENV="$ROOT/references/audio8/.venv"

# Resolve a python only when one is actually needed. macOS ships 3.9, so demanding >= 3.10
# up front fails on a machine where everything is already built and no python will be run.
need_python() {
  if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
    return 0
  fi
  PY="${PYTHON:-python3}"
  command -v "$PY" >/dev/null || die "$PY not found — set PYTHON=/path/to/python3"
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die \
    "need python >= 3.10, found $("$PY" --version) at $(command -v "$PY").
   macOS ships 3.9. Install a newer one (brew install python@3.12) and either put it on
   PATH or pass it explicitly:  PYTHON=/opt/homebrew/bin/python3.12 $0"
  say "Using $("$PY" --version) at $(command -v "$PY")"
}

# ------------------------------------------------------------------ 2. checkpoint

WEIGHTS="$ROOT/references/audio8/weights"
if [ -f "$WEIGHTS/model.safetensors" ] && [ -f "$WEIGHTS/codec.pth" ]; then
  say "Audio8 checkpoint already present in references/audio8/weights — skipping download"
else
  say "Downloading Audio8/Audio8-TTS-Preview-0.6b (~2.4 GB) into references/audio8/weights"
  need_python
  "$PY" - "$WEIGHTS" <<'PYEOF'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub is not installed.\n"
        "Either `pip install huggingface_hub` into your python, or download\n"
        "https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b into references/audio8/weights by hand."
    )
snapshot_download("Audio8/Audio8-TTS-Preview-0.6b", local_dir=sys.argv[1])
PYEOF
fi

# ------------------------------------------------------------------ 3. python env

if [ -x "$VENV/bin/python" ]; then
  say "references/audio8/.venv already exists — skipping (delete it to rebuild)"
else
  need_python
  say "Creating references/audio8/.venv"
  "$PY" -m venv "$VENV"
  say "Installing requirements.txt (torch and friends — this is the slow part)"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$ROOT/references/audio8/requirements.txt"
fi

# ------------------------------------------------------------------ 4. fold codec

CODEC="$WEIGHTS/codec.safetensors"
if [ -f "$CODEC" ] && [ "$FORCE" != "--force" ]; then
  say "references/audio8/weights/codec.safetensors already built — skipping (pass --force to redo)"
else
  say "Folding the codec's weight_norm into safetensors (~1 GB out)"
  ( cd "$ROOT/references/audio8" && .venv/bin/python convert_codec.py --weights weights --out weights/codec.safetensors )
fi

# ------------------------------------------------------------------ 5. derived assets

# The fixture oracles and the two CosyVoice artifacts that need the upstream python package.
# Cheap (~130 MB) and it is what lets ./scripts/gates.sh run without any PyTorch, so do it
# unconditionally rather than making the gates the user's problem later.
say "Fetching the derived assets"
"$ROOT/scripts/fetch-assets.sh" || warn "asset fetch failed — the gates will report themselves
   skipped, and CosyVoice will not start. Rerun ./scripts/fetch-assets.sh, or regenerate
   from source: docs/reference.md#setup."

# ------------------------------------------------------------------ 6. build

say "Building the workspace"
cargo build --release

say "Done. Audio8 is ready:"
cat <<'EOF'

    cargo run -p tts-cli --release -- speak \
        --engine audio8 --voice voices/cosy-default \
        --text "Hello from a fresh checkout." --out hello.wav
EOF

# Report what is actually still missing rather than assuming a fresh machine. After the asset
# fetch the only things that can be absent are the two checkpoints nobody can download for
# you — so name those, not the fixtures.
missing=0
if [ ! -f "$ROOT/references/cosyvoice/weights/llm.safetensors" ]; then
  printf '\n  * CosyVoice has no checkpoint — download Fun-CosyVoice3-0.5B and convert it.\n'
  printf '    See docs/reference.md#setup. Its other assets are already fetched.\n'
  missing=1
fi
if [ ! -f "$ROOT/references/qwen3tts/weights/model.safetensors" ]; then
  printf '  * Qwen3-TTS has no checkpoint — see docs/reference.md#setup.\n'
  missing=1
fi
for e in audio8 cosyvoice qwen3tts; do
  if [ ! -f "$ROOT/fixtures/$e/oracle.safetensors" ]; then
    printf '  * The %s fixture gate has no fixtures — rerun ./scripts/fetch-assets.sh\n' "$e"
    missing=1
  fi
done
[ "$missing" -eq 0 ] && printf '\nEverything else is present too — all three engines and all three fixture gates.\n'

printf '\nVerify with:  ./scripts/gates.sh\n'
