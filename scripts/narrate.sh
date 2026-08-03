#!/usr/bin/env bash
#
# Narrate a markdown chapter over HTTP, one engine at a time.
#
#   scripts/narrate.sh <chapter.md> <out-dir> [engine ...]
#
# Why HTTP rather than the CLI: the CLI loads the model per invocation, so narrating a
# book means paying the load on every chapter and holding a fresh copy of the weights each
# time. The service loads once and queues requests behind a single GPU permit.
#
# Why one engine at a time: a single audio8 render peaks around 3.6 GB (the codec's
# conv-as-GEMM materialises `[k*cin, len]` im2col matrices and candle's Metal buffer pool
# retains one per size class). Two engines resident on a 16 GB machine drives it into swap,
# where nothing finishes. This script starts a server, uses it, stops it, and only then
# starts the next.
#
# `--max-chars` is raised well above the service default of 1200. That default mirrors the
# Python service, whose job API existed for long content; here the whole chapter goes in one
# request so the engine's own segmenter and its 90 ms / 320 ms gaps apply across the entire
# text. Chunking client-side and concatenating would break prosody at every seam.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:?usage: narrate.sh <chapter.md> <out-dir> [engine ...]}"
OUT="${2:?usage: narrate.sh <chapter.md> <out-dir> [engine ...]}"
shift 2
ENGINES=("${@:-audio8 cosyvoice}")
[ $# -eq 0 ] && ENGINES=(audio8 cosyvoice)

PORT="${NARRATE_PORT:-3099}"
KEY="${TTS_API_KEY:-narrate-local-key}"
MAX_CHARS="${NARRATE_MAX_CHARS:-40000}"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$OUT"
base=$(basename "${SRC%.*}")
text="$OUT/$base.txt"

say "Converting $SRC"
./scripts/md-to-narration.py "$SRC" -o "$text" --stats
chars=$(wc -c < "$text" | tr -d ' ')
[ "$chars" -gt "$MAX_CHARS" ] && die "$chars chars exceeds NARRATE_MAX_CHARS=$MAX_CHARS"

# The request body, built with python so the text is escaped correctly.
body="$OUT/.$base.request.json"
python3 -c "
import json, sys
json.dump({'text': open(sys.argv[1]).read()}, open(sys.argv[2], 'w'))
" "$text" "$body"

for engine in "${ENGINES[@]}"; do
  case "$engine" in
    audio8)    voice=voices/cosy-default ;;
    cosyvoice) voice=voices/cosy-default-cosyvoice ;;
    *) die "unknown engine '$engine'" ;;
  esac

  # Refuse to start alongside anything else holding the GPU — that is what turns a
  # 10-minute render into a swap storm.
  if pgrep -f "target/release/tts(-serve)? " >/dev/null 2>&1; then
    pgrep -fl "target/release/tts(-serve)? " >&2
    die "another tts process is running; stop it first (see the note above)"
  fi

  say "Starting $engine on :$PORT"
  TTS_API_KEY="$KEY" ./target/release/tts-serve \
    --port "$PORT" --engine "$engine" --voice "$voice" --max-chars "$MAX_CHARS" \
    >"$OUT/.$engine.server.log" 2>&1 &
  pid=$!
  # shellcheck disable=SC2064
  trap "kill $pid 2>/dev/null || true" EXIT

  for _ in $(seq 1 90); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
    kill -0 "$pid" 2>/dev/null || { cat "$OUT/.$engine.server.log" >&2; die "$engine server exited"; }
    sleep 1
  done
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || die "$engine never became healthy"

  say "Rendering (no streaming, so this returns only when the whole chapter is done)"
  code=$(curl -s --max-time 3600 -o "$OUT/$base-$engine.wav" -D "$OUT/.$engine.headers" \
    -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/tts" \
    -H 'content-type: application/json' -H "X-API-Key: $KEY" --data-binary @"$body")
  [ "$code" = 200 ] || { cat "$OUT/$base-$engine.wav" >&2; die "$engine returned HTTP $code"; }

  grep -ihE '^x-(audio-seconds|wall-seconds|rtf|stages)' "$OUT/.$engine.headers" | tr -d '\r' | sed 's/^/    /'

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
  say "Stopped $engine; GPU released before the next one starts"
done

rm -f "$body"
say "Done"
ls -lh "$OUT"/*.wav | awk '{printf "    %-46s %s\n", $9, $5}'
