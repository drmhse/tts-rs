#!/usr/bin/env bash
#
# Fetch the derived assets that are too large to track but too awkward to regenerate:
# the fixture oracles, and the two CosyVoice artifacts that need the upstream python
# package to produce.
#
#   fixtures/{audio8,cosyvoice,qwen3tts}/*.safetensors   ground truth for ./scripts/gates.sh
#   references/cosyvoice/weights/rand_noise.safetensors  the CFM decoder's fixed noise
#   references/cosyvoice/weights/tokenizer.json          the ~250-special-token tokenizer
#
# The last two are why this script exists. Producing them imports CosyVoice model code,
# which pins python 3.10 and torch 2.3.1; fetching them instead means *running* CosyVoice
# needs neither, only the checkpoint and a plain `torch.load`. See docs/setup.md §2.
#
# curl and shasum only — no python, so this runs before any venv exists. Every file is
# checked against the published SHA256SUMS and a mismatch is deleted rather than kept.
#
# Re-running is safe: a file that is already present and hashes correctly is skipped.
# Pass --force to refetch everything.
#
# Regenerating from source instead is documented in docs/setup.md §3, and is the stronger
# move if you are auditing the ports rather than using them — a gate that verifies against
# tensors somebody else uploaded is only as trustworthy as the upload.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="${TTS_ASSETS_REPO:-drmhse/tts-rs-assets}"
BASE="https://huggingface.co/datasets/$REPO/resolve/main"
FORCE="${1:-}"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl not found"
command -v shasum >/dev/null || die "shasum not found"

# A progress bar redirected into a log file is thousands of lines of hashes. Only ask for
# one when something is actually watching.
if [ -t 2 ]; then PROGRESS=(-#); else PROGRESS=(-sS); fi

# Where each path in the asset repo lands in this checkout. The fixtures keep their layout;
# the CosyVoice pair goes next to the converted weights, because that is where the engine
# looks for them.
dest_for() {
  case "$1" in
    cosyvoice/*) printf 'references/cosyvoice/weights/%s' "${1#cosyvoice/}" ;;
    fixtures/*)  printf '%s' "$1" ;;
    *)           die "unexpected path in SHA256SUMS: $1" ;;
  esac
}

say "Fetching the manifest from $REPO"
SUMS="$(mktemp)"
trap 'rm -f "$SUMS"' EXIT
curl -sSfL -o "$SUMS" "$BASE/SHA256SUMS" \
  || die "could not fetch $BASE/SHA256SUMS — is the dataset public, and are you online?"

fetched=0
skipped=0
while read -r want path; do
  [ -n "${path:-}" ] || continue
  path="${path#./}"
  dest="$(dest_for "$path")"

  if [ -f "$dest" ] && [ "$FORCE" != "--force" ]; then
    if [ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" = "$want" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    say "$dest exists but does not match the manifest — refetching"
  fi

  mkdir -p "$(dirname "$dest")"
  say "$path"
  curl "${PROGRESS[@]}" -fL --retry 3 -o "$dest.part" "$BASE/$path" || die "download failed: $path"

  got="$(shasum -a 256 "$dest.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    rm -f "$dest.part"
    die "checksum mismatch for $path
   expected $want
   got      $got
   The upload may have been replaced. Regenerate from source instead (docs/setup.md §3)."
  fi
  mv "$dest.part" "$dest"
  fetched=$((fetched + 1))
done < "$SUMS"

say "Done: $fetched fetched, $skipped already present and verified."

# CosyVoice still needs its checkpoint converted; say so rather than implying it can run.
if [ ! -f references/cosyvoice/weights/llm.safetensors ]; then
  printf '\n  * CosyVoice has its assets but not its weights — convert the checkpoint,\n'
  printf '    see docs/setup.md §2. Everything else here is ready.\n'
fi
