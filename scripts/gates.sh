#!/usr/bin/env bash
#
# Everything that can say "this port is still correct", in one command.
#
# Three tiers, cheapest first:
#   unit tests    — no model weights needed
#   fixture gates — need converted weights and dumped fixtures (docs/reference.md#setup)
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

say "Narration prep tests"
if python3 scripts/test_md_to_narration.py >/dev/null 2>&1; then
  echo "md-to-narration ok"
else
  python3 scripts/test_md_to_narration.py 2>&1 | tail -20
  echo "md-to-narration FAILED"; fail=1
fi

say "Clippy and formatting (advisory)"
if cargo fmt --check >/dev/null 2>&1; then
  echo "formatting clean"
else
  echo "note: \`cargo fmt\` would reformat some files"
fi
lints=$(cargo clippy --release --all-targets 2>&1 | grep -cE "^warning: [a-z]" || true)
echo "clippy: $lints lint(s)"

say "CPU-only build (the portable configuration)"
if cargo build --release --no-default-features --quiet 2>&1 | tail -5; then
  echo "no-default-features builds"
else
  echo "no-default-features FAILED"; fail=1
fi

say "Audio8 fixture gate"
if [ -f fixtures/audio8/oracle.safetensors ] && [ -f references/audio8/weights/codec.safetensors ]; then
  cargo run -q -p audio8 --release --bin audio8-validate || fail=1
else
  skip "fixtures/audio8/oracle.safetensors or references/audio8/weights/codec.safetensors missing"
fi

say "CosyVoice fixture gate"
if [ -f fixtures/cosyvoice/oracle.safetensors ] && [ -d references/cosyvoice/weights ]; then
  cargo run -q -p cosyvoice --release --bin cosyvoice-validate || fail=1
else
  skip "fixtures/cosyvoice/oracle.safetensors or references/cosyvoice/weights missing"
fi

say "Qwen3-TTS gate"
# Two tiers inside one bin: a shape audit that reads only the checkpoint header, then per-stage
# numerics against fixtures/qwen3tts. The numerics tier reports itself as skipped when the
# fixtures are absent. Gated on the talker checkpoint, which is what the audit reads.
if [ -f references/qwen3tts/weights/model.safetensors ]; then
  cargo run -q -p qwen3tts --release --bin qwen3tts-validate || fail=1
else
  skip "references/qwen3tts/weights/model.safetensors missing"
fi

say "End-to-end renders"
mkdir -p target/gate
for spec in "audio8:voices/cosy-default" "cosyvoice:voices/cosy-default-cosyvoice" \
           "qwen3tts:voices/cosy-default-qwen3tts"; do
  id="${spec%%:*}"; voice="${spec##*:}"
  if [ -d "$voice" ]; then
    if ! cargo run -q -p tts-cli --release -- speak \
      --engine "$id" --voice "$voice" --text-file examples/senior.txt \
      --out "target/gate/$id.wav"; then
      echo "render FAILED for $id"; fail=1
    fi
  else
    skip "$id: voice asset $voice missing"
  fi
done

say "HTTP service smoke test"
if [ -d voices/cosy-default-cosyvoice ] && [ -d references/cosyvoice/weights ]; then
  TTS_API_KEY=gate-smoke-key ./target/release/tts-serve --port 3099 >target/gate/serve.log 2>&1 &
  serve_pid=$!
  for _ in $(seq 1 60); do
    curl -sf -o /dev/null http://127.0.0.1:3099/health && break
    sleep 1
  done
  if curl -sf -o /dev/null http://127.0.0.1:3099/health; then
    code=$(curl -s -o target/gate/http.wav -w '%{http_code}' -X POST http://127.0.0.1:3099/tts \
      -H 'content-type: application/json' -H 'X-API-Key: gate-smoke-key' \
      -d '{"text":"Gate smoke test."}')
    unauth=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:3099/tts \
      -H 'content-type: application/json' -d '{"text":"no key"}')
    if [ "$code" = 200 ] && [ "$unauth" = 401 ] && [ -s target/gate/http.wav ]; then
      echo "POST /tts 200, unauthenticated 401, wav non-empty"
    else
      echo "http smoke FAILED (tts=$code unauth=$unauth)"; fail=1
    fi
  else
    echo "http smoke FAILED: /health never came up"; fail=1
  fi
  kill "$serve_pid" 2>/dev/null
  wait "$serve_pid" 2>/dev/null
else
  skip "http smoke: cosyvoice weights or voice asset missing"
fi

say "Summary"
[ "$SKIPPED" -gt 0 ] && echo "$SKIPPED tier(s) skipped for missing inputs — see docs/reference.md#setup"
if [ "$fail" -eq 0 ]; then
  echo "everything that ran passed"
else
  echo "FAILURES above"
fi
exit "$fail"
