#!/usr/bin/env bash
#
# Narrate one or more markdown chapters over HTTP.
#
#   scripts/narrate.sh --engine cosyvoice --out narration chapter-*.md
#
# Design notes, each of them the result of getting it wrong first:
#
# * **One server for the whole book.** The engine loads once (~3 s for cosyvoice) and every
#   chapter is a request against it. An earlier version started a server per chapter, which
#   for a twelve-part book is twelve model loads for no reason.
#
# * **One engine resident at a time.** A single render peaks well above what a 16 GB machine
#   can spare alongside a second engine, so this refuses to start when another `tts` process
#   holds the GPU. Matching on `target/release/tts` and not on the cargo wrapper matters:
#   `pkill -f "tts-cli…"` kills the wrapper and orphans the worker, which is exactly how two
#   engines ended up resident and drove the machine into swap.
#
# * **Whole chapter in one request.** `--max-chars` is raised far above the service default
#   so the engine's own segmenter and its 90 ms / 320 ms gaps apply across the entire text.
#   Chunking client-side and concatenating would seam prosody at every join.
#
# * **Resumable.** A chapter whose output already exists is skipped, so an interrupted book
#   picks up where it stopped. Renders are deterministic (seed 1234), so a re-run reproduces
#   the same audio rather than a different valid draw.
#
# * **One failure does not abandon the book.** A chapter that errors is reported and the run
#   continues; the exit status is non-zero so the failure is not silent.
set -uo pipefail
cd "$(dirname "$0")/.."

ENGINE=cosyvoice
OUT=narration
PORT="${NARRATE_PORT:-3099}"
KEY="${TTS_API_KEY:-narrate-local-key}"
MAX_CHARS="${NARRATE_MAX_CHARS:-80000}"
# Opus at 32 kbps mono is transparent for narration and about a quarter the size of the
# 128 kbps MP3s this replaces. Opus resamples to 48 kHz internally; that is normal and not
# a quality loss.
OPUS_BITRATE="${NARRATE_OPUS_BITRATE:-32k}"

FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --bitrate) OPUS_BITRATE="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) FILES+=("$1"); shift ;;
  esac
done
[ ${#FILES[@]} -gt 0 ] || { echo "usage: narrate.sh [--engine E] [--out DIR] <chapter.md ...>" >&2; exit 2; }

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "$ENGINE" in
  audio8)    VOICE=voices/cosy-default ;;
  cosyvoice) VOICE=voices/cosy-default-cosyvoice ;;
  *) die "unknown engine '$ENGINE'" ;;
esac
command -v ffmpeg >/dev/null || die "ffmpeg not found (needed for Opus encoding)"
mkdir -p "$OUT"

if pgrep -f "target/release/tts(-serve)? " >/dev/null 2>&1; then
  pgrep -fl "target/release/tts(-serve)? " >&2
  die "another tts process holds the GPU; stop it first"
fi

say "Starting $ENGINE on :$PORT (loads once, serves every chapter)"
TTS_API_KEY="$KEY" ./target/release/tts-serve \
  --port "$PORT" --engine "$ENGINE" --voice "$VOICE" --max-chars "$MAX_CHARS" \
  >"$OUT/.server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
  kill -0 "$SERVER" 2>/dev/null || { cat "$OUT/.server.log" >&2; die "server exited"; }
  sleep 1
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || die "server never became healthy"
say "Ready. $(( ${#FILES[@]} )) file(s) to narrate."

failed=0
done_n=0
for src in "${FILES[@]}"; do
  base=$(basename "${src%.*}")
  wav="$OUT/$base.wav"
  opus="$OUT/$base.opus"
  txt="$OUT/$base.txt"

  if [ -s "$opus" ]; then
    say "$base — already done, skipping"
    done_n=$((done_n + 1)); continue
  fi

  ./scripts/md-to-narration.py "$src" -o "$txt" --stats 2>&1 | sed 's/^/    /'
  chars=$(wc -c < "$txt" | tr -d ' ')
  if [ "$chars" -gt "$MAX_CHARS" ]; then
    warn "$base: $chars chars exceeds --max-chars $MAX_CHARS; skipped"
    failed=$((failed + 1)); continue
  fi

  body="$OUT/.$base.json"
  python3 -c "
import json, sys
json.dump({'text': open(sys.argv[1]).read()}, open(sys.argv[2], 'w'))" "$txt" "$body"

  say "$base — rendering (returns only when the whole chapter is done)"
  code=$(curl -s --max-time 7200 -o "$wav" -D "$OUT/.$base.headers" -w '%{http_code}' \
    -X POST "http://127.0.0.1:$PORT/tts" -H 'content-type: application/json' \
    -H "X-API-Key: $KEY" --data-binary @"$body")
  rm -f "$body"

  if [ "$code" != 200 ]; then
    warn "$base: HTTP $code — $(head -c 200 "$wav")"
    rm -f "$wav"; failed=$((failed + 1)); continue
  fi
  grep -ihE '^x-(audio-seconds|rtf)' "$OUT/.$base.headers" | tr -d '\r' | sed 's/^/    /'

  # Opus is the deliverable; the WAV is kept as the lossless master to re-encode from.
  if ! ffmpeg -v error -y -i "$wav" -c:a libopus -b:a "$OPUS_BITRATE" -vbr on -ac 1 \
       -metadata title="$base" -metadata album="narration" "$opus"; then
    warn "$base: Opus encoding failed"; failed=$((failed + 1)); continue
  fi

  wd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav")
  od=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$opus")
  drift=$(python3 -c "print(f'{abs($wd - $od):.3f}')")
  printf '    wav %.1fs  opus %.1fs  drift %ss  %s\n' "$wd" "$od" "$drift" \
    "$(du -h "$opus" | cut -f1)"
  # A duration mismatch means the encode dropped or padded audio, which would desync any
  # alignment built against the WAV.
  python3 -c "import sys; sys.exit(0 if abs($wd-$od) < 0.10 else 1)" \
    || warn "$base: duration drift ${drift}s between wav and opus"
  done_n=$((done_n + 1))
done

say "Narrated $done_n, failed $failed"
ls -lh "$OUT"/*.opus 2>/dev/null | awk '{printf "    %-40s %s\n", $9, $5}'
[ "$failed" -eq 0 ] || exit 1
