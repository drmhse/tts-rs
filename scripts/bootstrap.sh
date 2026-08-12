#!/usr/bin/env bash
#
# From a fresh clone to working speech. Nothing here is manual.
#
#   ./scripts/bootstrap.sh                              all three engines, ~13 GB
#   ./scripts/bootstrap.sh qwen3tts                     one engine, ~4.3 GB
#   ./scripts/bootstrap.sh audio8 cosyvoice             two of them
#   ./scripts/bootstrap.sh --list                       what the ids are, and what each costs
#   ./scripts/bootstrap.sh --force audio8               redo a conversion that already ran
#
# Whichever engines you name, it checks the toolchain, downloads and converts their
# checkpoints, fetches the fixtures for every gate (~130 MB, cheap, so always), and builds.
#
# No engine needs its upstream *repository*. CosyVoice's two un-derivable artifacts are
# fetched as assets, so converting its checkpoint is a plain `torch.load`.
#
# Re-running is safe: every step is skipped if its output already exists.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

ALL_ENGINES="audio8 cosyvoice qwen3tts"
FORCE=""
ENGINES=""

usage() {
  cat <<'USAGE'
usage: scripts/bootstrap.sh [--force] [--list] [engine ...]

  engine    audio8 | cosyvoice | qwen3tts   (default: all three)
  --force   redo a conversion whose output already exists
  --list    print the engines, their models and their disk cost, then exit
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --list)
      printf '%-11s %-34s %8s  %s\n' engine model disk notes
      printf '%-11s %-34s %8s  %s\n' audio8    Audio8-TTS-Preview-0.6b   "~4 GB"   "44.1 kHz, highest fidelity"
      printf '%-11s %-34s %8s  %s\n' cosyvoice Fun-CosyVoice3-0.5B       "~4 GB"   "widest language coverage"
      printf '%-11s %-34s %8s  %s\n' qwen3tts  Qwen3-TTS-12Hz-1.7B-Base  "~4.3 GB" "batches; best for long documents"
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    audio8|cosyvoice|qwen3tts) ENGINES="$ENGINES $1"; shift ;;
    --audio8-only) ENGINES="$ENGINES audio8"; shift ;;   # the old spelling
    *) usage >&2; printf '\nunknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[ -n "$ENGINES" ] || ENGINES="$ALL_ENGINES"
# Normalise: the loop above leaves a leading space, which would make `${ENGINES%% *}` empty.
# shellcheck disable=SC2086
set -- $ENGINES
ENGINES="$*"
wants() { case " $ENGINES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

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
if ! wants audio8; then
  say "Skipping the Audio8 checkpoint (not selected)"
elif [ -f "$WEIGHTS/model.safetensors" ] && [ -f "$WEIGHTS/codec.pth" ]; then
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
#
# Needed by Audio8's codec fold and by CosyVoice's conversion. Qwen3-TTS alone needs no
# python at all — its setup is downloads — so building a torch venv for it would be waste.

if ! wants audio8 && ! wants cosyvoice; then
  say "Skipping the python env (only qwen3tts selected, which needs none)"
elif [ -x "$VENV/bin/python" ]; then
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
if ! wants audio8; then
  :
elif [ -f "$CODEC" ] && [ -z "$FORCE" ]; then
  say "references/audio8/weights/codec.safetensors already built — skipping (pass --force to redo)"
else
  say "Folding the codec's weight_norm into safetensors (~1 GB out)"
  ( cd "$ROOT/references/audio8" && .venv/bin/python convert_codec.py --weights weights --out weights/codec.safetensors )
fi

# ------------------------------------------------- 4b. CosyVoice and Qwen3-TTS checkpoints
#
# Both are plain downloads plus, for CosyVoice, a `torch.load` re-serialisation. Neither
# needs its upstream repo: the two artifacts that used to require one are fetched in step 5.
# Each runs only if its engine was asked for.

COSY_W="$ROOT/references/cosyvoice/weights"
QWEN_W="$ROOT/references/qwen3tts/weights"

if ! wants cosyvoice; then
  say "Skipping the CosyVoice checkpoint (not selected)"
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

fi

if ! wants qwen3tts; then
  say "Skipping the Qwen3-TTS checkpoint (not selected)"
else
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

# Show the command for an engine that was actually set up, not always the first one.
first="${ENGINES%% *}"
case "$first" in
  audio8)    voice=voices/cosy-default ;;
  cosyvoice) voice=voices/cosy-default-cosyvoice ;;
  qwen3tts)  voice=voices/cosy-default-qwen3tts ;;
esac

say "Done — set up: $ENGINES"
cat <<EOF

    cargo run -p tts-cli --release -- speak \\
        --engine $first --voice $voice \\
        --text "Hello from a fresh checkout." --out hello.wav

    cargo run -p tts-cli --release -- engines     # what is available, and what each supports
    ./scripts/gates.sh                            # everything verifiable
    TTS_API_KEY=secret cargo run -p tts-serve --release -- --port 3003   # the HTTP API
EOF

# Report what is missing rather than assuming. An engine that was not asked for is not
# missing, so name only the ones that were selected — and the fixture gates, which are
# fetched for every engine because they cost 130 MB and prove the ports are still correct.
missing=0
wants audio8 && [ ! -f "$WEIGHTS/codec.safetensors" ] && {
  printf '\n  * Audio8 has no folded codec — rerun with --force.\n'; missing=1; }
wants cosyvoice && [ ! -f "$COSY_W/llm.safetensors" ] && {
  printf '\n  * CosyVoice has no checkpoint — see docs/reference.md#setup.\n'; missing=1; }
wants qwen3tts && [ ! -f "$QWEN_W/model.safetensors" ] && {
  printf '\n  * Qwen3-TTS has no checkpoint — see docs/reference.md#setup.\n'; missing=1; }
for e in audio8 cosyvoice qwen3tts; do
  if [ ! -f "$ROOT/fixtures/$e/oracle.safetensors" ]; then
    printf '  * The %s fixture gate has no fixtures — rerun ./scripts/fetch-assets.sh\n' "$e"
    missing=1
  fi
done
[ "$missing" -eq 0 ] && printf '\nEverything selected is present, and all three fixture gates.\n'

printf '\nVerify with:  ./scripts/gates.sh\n'
