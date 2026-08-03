#!/usr/bin/env bash
#
# Narrate a book: markdown in, Opus + alignment manifests out.
#
#   scripts/narrate-book.sh --book /path/to/content/books/the-change-interface --out narration
#
# Discovers `introduction.md`, `chapter-N.md` (numerically), `conclusion.md` in that order,
# skipping `_index.md` — a book landing page, not narration content. Pass explicit files
# instead if a book is laid out differently.
#
# Per chapter, in one pass:
#   1. markdown  -> narration text + page-word map   (md-to-narration.py)
#   2. text      -> WAV                               (tts-serve, one server for the book)
#   3. WAV       -> Opus 32 kbps                      (ffmpeg; WAV kept as master)
#   4. Opus+text -> alignment manifest                (align-narration.py)
#
# Resumable: a chapter whose manifest exists is skipped, so an interrupted book continues.
# Renders are deterministic (seed 1234), so re-running reproduces the same audio.
#
# Alignment needs whisperx, which usually lives in a different interpreter from anything
# else here; point ALIGN_PYTHON at it. If it is missing, audio is still produced and the
# alignment step is reported as skipped rather than failing the book.
set -uo pipefail
cd "$(dirname "$0")/.."

BOOK=""
OUT=narration
ENGINE=cosyvoice
FILES=()
PORT="${NARRATE_PORT:-3099}"
KEY="${TTS_API_KEY:-narrate-local-key}"
MAX_CHARS="${NARRATE_MAX_CHARS:-80000}"
BITRATE="${NARRATE_OPUS_BITRATE:-32k}"
ALIGN_PYTHON="${ALIGN_PYTHON:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --book) BOOK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --bitrate) BITRATE="$2"; shift 2 ;;
    --align-python) ALIGN_PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) FILES+=("$1"); shift ;;
  esac
done

say()  { printf '\n\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [ -n "$BOOK" ]; then
  [ -d "$BOOK" ] || die "--book $BOOK is not a directory"
  [ -f "$BOOK/introduction.md" ] && FILES+=("$BOOK/introduction.md")
  while IFS= read -r f; do FILES+=("$f"); done < <(ls "$BOOK"/chapter-*.md 2>/dev/null | sort -V)
  [ -f "$BOOK/conclusion.md" ] && FILES+=("$BOOK/conclusion.md")
fi
[ ${#FILES[@]} -gt 0 ] || die "nothing to narrate; pass --book DIR or explicit files"

case "$ENGINE" in
  audio8)    VOICE=voices/cosy-default ;;
  cosyvoice) VOICE=voices/cosy-default-cosyvoice ;;
  *) die "unknown engine '$ENGINE'" ;;
esac
command -v ffmpeg >/dev/null || die "ffmpeg not found"
[ -x ./target/release/tts-serve ] || die "build first: cargo build --release"
mkdir -p "$OUT"

# Autodetect an interpreter with whisperx if one was not given.
if [ -z "$ALIGN_PYTHON" ]; then
  for cand in \
    "$HOME/Desktop/projects/AI/tts/CosyVoice/.venv-align/bin/python" \
    "$(command -v python3 || true)"; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import whisperx" >/dev/null 2>&1; then ALIGN_PYTHON="$cand"; break; fi
  done
fi
[ -n "$ALIGN_PYTHON" ] || warn "no interpreter with whisperx found; manifests will be skipped"

# Two engines resident will not fit; match the binary, not the cargo wrapper.
if pgrep -f "target/release/tts(-serve)? " >/dev/null 2>&1; then
  pgrep -fl "target/release/tts(-serve)? " >&2
  die "another tts process holds the GPU"
fi

say "Starting $ENGINE on :$PORT — loads once for all ${#FILES[@]} file(s)"
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

ok=0; failed=0; skipped=0
for src in "${FILES[@]}"; do
  base=$(basename "${src%.*}")
  wav="$OUT/$base.wav"; opus="$OUT/$base.opus"
  txt="$OUT/$base.txt"; map="$OUT/$base.map.json"; man="$OUT/$base.manifest.json"

  if [ -s "$man" ]; then say "$base — done already"; skipped=$((skipped+1)); continue; fi
  say "$base"

  ./scripts/md-to-narration.py "$src" -o "$txt" --emit-map "$map" --stats 2>&1 | sed 's/^/    /'
  chars=$(wc -c < "$txt" | tr -d ' ')
  if [ "$chars" -gt "$MAX_CHARS" ]; then
    warn "$base: $chars chars over --max-chars $MAX_CHARS"; failed=$((failed+1)); continue
  fi

  if [ ! -s "$opus" ]; then
    body="$OUT/.$base.json"
    python3 -c "import json,sys; json.dump({'text': open(sys.argv[1]).read()}, open(sys.argv[2],'w'))" "$txt" "$body"
    code=$(curl -s --max-time 7200 -o "$wav" -D "$OUT/.$base.headers" -w '%{http_code}' \
      -X POST "http://127.0.0.1:$PORT/tts" -H 'content-type: application/json' \
      -H "X-API-Key: $KEY" --data-binary @"$body")
    rm -f "$body"
    if [ "$code" != 200 ]; then
      warn "$base: HTTP $code — $(head -c 200 "$wav" 2>/dev/null)"; rm -f "$wav"
      failed=$((failed+1)); continue
    fi
    grep -ihE '^x-(audio-seconds|rtf)' "$OUT/.$base.headers" | tr -d '\r' | sed 's/^/    /'
    ffmpeg -v error -y -i "$wav" -c:a libopus -b:a "$BITRATE" -vbr on -ac 1 \
      -metadata title="$base" "$opus" || { warn "$base: opus encode failed"; failed=$((failed+1)); continue; }
    wd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav")
    od=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$opus")
    python3 -c "import sys; sys.exit(0 if abs($wd-$od)<0.10 else 1)" \
      || warn "$base: wav/opus duration drift $(python3 -c "print(f'{abs($wd-$od):.3f}')")s"
    printf '    opus %s\n' "$(du -h "$opus" | cut -f1)"
  fi

  if [ -n "$ALIGN_PYTHON" ]; then
    "$ALIGN_PYTHON" scripts/align-narration.py --audio "$opus" --text "$txt" \
      --map "$map" --out "$man" --title "$base" 2>&1 | sed 's/^/    /' \
      || { warn "$base: alignment failed"; failed=$((failed+1)); continue; }
  fi
  ok=$((ok+1))
done

say "Done: $ok narrated, $skipped already present, $failed failed"
[ "$failed" -eq 0 ] || exit 1
