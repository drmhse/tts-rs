#!/usr/bin/env bash
#
# From a fresh clone to all three engines working. Nothing here is manual.
#
# What it does:
#   1. checks the toolchain and the platform
#   2. downloads the Audio8 checkpoint (~2.4 GB) from Hugging Face
#   3. creates references/audio8/.venv and folds the codec's weight_norm into safetensors
#   4. downloads and converts the CosyVoice and Qwen3-TTS checkpoints (~8 GB together)
#   5. fetches the derived assets (~130 MB): the fixture oracles for all three gates, and
#      the two CosyVoice artifacts that need the upstream python package to produce
#   6. builds the workspace
#
# Budget ~13 GB of disk and a long first run; almost all of it is download. Pass
# --audio8-only to stop after the first engine, which needs ~4 GB.
#
# No engine needs its upstream *repository* any more. CosyVoice's two un-derivable
# artifacts are fetched in step 5, so converting its checkpoint is a plain `torch.load`
# under the venv this script already built.
#
# Re-running is safe: every step is skipped if its output already exists. Pass --force
# to redo the codec conversion, or --audio8-only to skip the other two engines.
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

# ------------------------------------------------- 4b. CosyVoice and Qwen3-TTS checkpoints
#
# Both are plain downloads plus, for CosyVoice, a `torch.load` re-serialisation. Neither
# needs its upstream repo: the two artifacts that used to require one are fetched in step 5.
# Skipped entirely with --audio8-only, since together they are another ~8 GB.

COSY_W="$ROOT/references/cosyvoice/weights"
QWEN_W="$ROOT/references/qwen3tts/weights"

if [ "$FORCE" = "--audio8-only" ]; then
  say "Skipping CosyVoice and Qwen3-TTS (--audio8-only)"
else
  if [ -f "$COSY_W/llm.safetensors" ]; then
    say "CosyVoice checkpoint already converted — skipping"
  else
    say "Downloading Fun-CosyVoice3-0.5B (~4 GB) and converting it"
    need_python
    "$VENV/bin/python" - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B", local_dir="references/cosyvoice/download")
PYEOF
    "$VENV/bin/python" "$ROOT/references/cosyvoice/convert.py" \
        --checkpoints "$ROOT/references/cosyvoice/download" --out "$COSY_W"
    # The .pt files are 4 GB and nothing reads them after conversion.
    if [ -f "$COSY_W/llm.safetensors" ]; then
      say "Removing the raw CosyVoice download (4 GB, no longer needed)"
      rm -rf "$ROOT/references/cosyvoice/download"
    fi
  fi

  if [ -f "$QWEN_W/model.safetensors" ]; then
    say "Qwen3-TTS checkpoint already present — skipping"
  else
    say "Downloading Qwen3-TTS-12Hz-1.7B-Base (~4.3 GB)"
    mkdir -p "$QWEN_W/speech_tokenizer"
    B=https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main
    for f in config.json generation_config.json preprocessor_config.json \
             tokenizer_config.json vocab.json merges.txt model.safetensors; do
      curl -fsSL -C - -o "$QWEN_W/$f" "$B/$f" || die "failed to download $f"
    done
    for f in config.json configuration.json preprocessor_config.json model.safetensors; do
      curl -fsSL -C - -o "$QWEN_W/speech_tokenizer/$f" "$B/speech_tokenizer/$f" \
        || die "failed to download speech_tokenizer/$f"
    done
  fi
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

say "Done."
cat <<'EOF'

    cargo run -p tts-cli --release -- speak \
        --engine audio8 --voice voices/cosy-default \
        --text "Hello from a fresh checkout." --out hello.wav

    cargo run -p tts-cli --release -- engines      # all three, and what each supports
    ./scripts/gates.sh                             # everything verifiable
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
