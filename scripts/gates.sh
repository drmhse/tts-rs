#!/usr/bin/env bash
#
# Everything that can say "this port is still correct", in one command.
#
# Three tiers, cheapest first:
#   unit tests    — no model weights needed
#   fixture gates — need converted weights and dumped fixtures (docs/setup.md)
#   renders       — need a voice asset; produce audio for the quality scripts
#
# A tier whose inputs are missing is *skipped and reported*, never silently passed.
# Exit status is non-zero if anything that ran failed.
set -uo pipefail

cd "$(dirname "$0")/.."

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
skip() { printf '\033[33m--- skipped: %s\033[0m\n' "$*"; SKIPPED=$((SKIPPED + 1)); }
fail=0
SKIPPED=0

say "Unit tests"
if cargo test --release --quiet 2>&1 | tail -20; then
  echo "tests ok"
else
  echo "tests FAILED"; fail=1
fi

say "Clippy and formatting"
cargo fmt --check 2>/dev/null || echo "note: cargo fmt would reformat some files"
cargo clippy --release --all-targets --quiet 2>&1 | grep -E "^(warning|error)" | head -5 || true

say "Audio8 fixture gate"
if [ -f fixtures/oracle.safetensors ] && [ -f oracle/weights/codec.safetensors ]; then
  cargo run -q -p a8 --release --bin a8-validate || fail=1
else
  skip "fixtures/oracle.safetensors or oracle/weights/codec.safetensors missing"
fi

say "CosyVoice fixture gate"
if [ -f fixtures-cosy/oracle.safetensors ] && [ -d oracle-cosy/weights ]; then
  cargo run -q -p cosy --release --bin cosy-validate || fail=1
else
  skip "fixtures-cosy/oracle.safetensors or oracle-cosy/weights missing"
fi

say "End-to-end renders"
mkdir -p target/gate
for spec in "audio8:voices/cosy-default" "cosyvoice:voices/cosy-default-cosyvoice"; do
  id="${spec%%:*}"; voice="${spec##*:}"
  if [ -d "$voice" ]; then
    cargo run -q -p tts-cli --release -- speak \
      --engine "$id" --voice "$voice" --text-file examples/senior.txt \
      --out "target/gate/$id.wav" || fail=1
  else
    skip "$id: voice asset $voice missing"
  fi
done

say "Summary"
[ "$SKIPPED" -gt 0 ] && echo "$SKIPPED tier(s) skipped for missing inputs — see docs/setup.md"
if [ "$fail" -eq 0 ]; then
  echo "everything that ran passed"
else
  echo "FAILURES above"
fi
exit "$fail"
