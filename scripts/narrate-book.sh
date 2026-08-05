#!/usr/bin/env bash
#
# Narrate a book: markdown in, WebM/Opus + alignment manifests out.
#
#   scripts/narrate-book.sh --book /path/to/content/books/<slug> --out narration
#
# Discovers chapters in either layout this site uses:
#
#   flat    introduction.md, chapter-N.md, conclusion.md
#   nested  part-NN-*/chapter-NNN-slug.md                      (every other book)
#
# `_index.md` is skipped at every level — those are landing pages, not narration content.
# Nested chapters are named `chapter-NNN` on the way out so the published directory matches
# the site's `chapter-NNN` convention regardless of the source filename's slug.
#
# Per chapter, in one pass:
#   1. markdown  -> narration text + page-word map   (md-to-narration.py)
#   2. text      -> WAV master                        (tts-serve, one server for the book)
#   3. WAV       -> WebM/Opus 48k for delivery        (ffmpeg; WAV kept as master)
#   4. WebM+text -> alignment manifest                (align-narration.py)
#
# WebM rather than a bare `.opus`: it is what the site's other books deliver, and Safari's
# support for Opus in an Ogg container is unreliable while Opus in WebM plays. A container
# choice here is a compatibility decision, not a detail — the failure mode is silence.
#
# Resumable per *stage*, which matters when synthesis costs hours and the encode costs
# seconds: a chapter with a WAV is not re-synthesised, one with a manifest is skipped
# entirely. Renders are deterministic (seed 1234), so re-running reproduces the same audio.
#
# Alignment needs faster-whisper, which usually lives in a different interpreter from
# anything else here; point ALIGN_PYTHON at it. If it is missing, audio is still produced
# and alignment is reported as skipped rather than failing the book.
set -uo pipefail
cd "$(dirname "$0")/.."

BOOK=""
OUT=narration
ENGINE=cosyvoice
FILES=()
PORT="${NARRATE_PORT:-3099}"
KEY="${TTS_API_KEY:-narrate-local-key}"
MAX_CHARS="${NARRATE_MAX_CHARS:-80000}"
BITRATE="${NARRATE_OPUS_BITRATE:-48k}"
ALIGN_PYTHON="${ALIGN_PYTHON:-}"
ONLY=""
# Sampling seed. Renders are deterministic, so re-rendering a chapter that came out wrong
# reproduces it exactly — a different seed is the only way to get a different draw. Needed
# because some failures are sampling glitches rather than input problems: one chapter spoke
# "founder" as "Fongder" and "manual" as "ManuArt", and another stopped at a semicolon and
# dropped the clause after it. Neither is detectable from token counts; both are obvious to
# the ASR check in verify-narration.py, and both clear on a different draw.
SEED="${NARRATE_SEED:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --book) BOOK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --bitrate) BITRATE="$2"; shift 2 ;;
    --align-python) ALIGN_PYTHON="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) FILES+=("$1"); shift ;;
  esac
done

say()  { printf '\n\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Output name for a source file: `chapter-NNN` for anything numbered, otherwise the stem.
# Numbering comes from the filename, which both layouts agree on.
out_name() {
  local base; base=$(basename "${1%.*}")
  if [[ "$base" =~ ^chapter-0*([0-9]+) ]]; then
    printf 'chapter-%03d' "${BASH_REMATCH[1]}"
  else
    printf '%s' "$base"
  fi
}

if [ -n "$BOOK" ]; then
  [ -d "$BOOK" ] || die "--book $BOOK is not a directory"
  [ -f "$BOOK/introduction.md" ] && FILES+=("$BOOK/introduction.md")
  # Recursive, numerically sorted by the chapter number in the filename, so nested
  # `part-NN/` directories do not reorder the book.
  while IFS= read -r f; do FILES+=("$f"); done < <(
    find "$BOOK" -name 'chapter-*.md' -not -name '_index.md' \
      | sed -E 's/.*chapter-0*([0-9]+).*/\1 &/' | sort -n -k1,1 | cut -d' ' -f2-
  )
  [ -f "$BOOK/conclusion.md" ] && FILES+=("$BOOK/conclusion.md")
fi
[ ${#FILES[@]} -gt 0 ] || die "nothing to narrate; pass --book DIR or explicit files"

if [ -n "$ONLY" ]; then
  KEEP=()
  for src in "${FILES[@]}"; do
    case ",$ONLY," in *",$(out_name "$src"),"*) KEEP+=("$src") ;; esac
  done
  FILES=("${KEEP[@]}")
  [ ${#FILES[@]} -gt 0 ] || die "--only $ONLY matched nothing"
fi

case "$ENGINE" in
  audio8)    VOICE=voices/cosy-default ;;
  cosyvoice) VOICE=voices/cosy-default-cosyvoice ;;
  *) die "unknown engine '$ENGINE'" ;;
esac
command -v ffmpeg >/dev/null || die "ffmpeg not found"
[ -x ./target/release/tts-serve ] || die "build first: cargo build --release"
mkdir -p "$OUT"

# Autodetect an interpreter with faster-whisper if one was not given.
if [ -z "$ALIGN_PYTHON" ]; then
  for cand in \
    "$HOME/Desktop/projects/AI/tts/CosyVoice/.venv-align/bin/python" \
    "$(command -v python3 || true)"; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import faster_whisper" >/dev/null 2>&1; then ALIGN_PYTHON="$cand"; break; fi
  done
fi
[ -n "$ALIGN_PYTHON" ] || warn "no interpreter with faster-whisper found; manifests skipped"

# Two engines resident will not fit; match the binary, not the cargo wrapper. Killing the
# cargo invocation once orphaned the real process to PID 1 and left two engines on the GPU.
if pgrep -f "target/release/tts(-serve)? " >/dev/null 2>&1; then
  pgrep -fl "target/release/tts(-serve)? " >&2
  die "another tts process holds the GPU"
fi

# Only start a server if something actually needs synthesising, so a pure re-encode or
# re-align run does not pay 3 s of model loading and cannot collide with another render.
NEED_SERVER=0
for src in "${FILES[@]}"; do
  n=$(out_name "$src")
  [ -s "$OUT/$n.manifest.json" ] && continue
  [ -s "$OUT/$n.wav" ] || NEED_SERVER=1
done

SERVER=""
if [ "$NEED_SERVER" = 1 ]; then
  say "Starting $ENGINE on :$PORT — loads once for all ${#FILES[@]} file(s)"
  TTS_API_KEY="$KEY" ./target/release/tts-serve \
    --port "$PORT" --engine "$ENGINE" --voice "$VOICE" --max-chars "$MAX_CHARS" \
    >"$OUT/.server.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
  for _ in $(seq 1 180); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
    kill -0 "$SERVER" 2>/dev/null || { cat "$OUT/.server.log" >&2; die "server exited"; }
    sleep 1
  done
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || die "server never became healthy"
else
  say "Nothing to synthesise; re-encoding and re-aligning only"
fi

ok=0; failed=0; skipped=0; flagged=()
for src in "${FILES[@]}"; do
  base=$(out_name "$src")
  wav="$OUT/$base.wav"; webm="$OUT/$base.webm"
  txt="$OUT/$base.txt"; map="$OUT/$base.map.json"; man="$OUT/$base.manifest.json"

  if [ -s "$man" ]; then say "$base — done already"; skipped=$((skipped+1)); continue; fi
  say "$base  ($(basename "$src"))"

  # Regenerating the text is safe only when there is no audio yet. Deleting a manifest to force
  # a re-align would otherwise re-derive the text with whatever the converter does *today* and
  # align yesterday's audio against it — a manifest describing words the voice never said,
  # which is the exact failure this pipeline exists to prevent. So when a master already exists,
  # convert to a scratch file and refuse to replace the text that produced it.
  if [ -s "$wav" ] && [ -s "$txt" ]; then
    ./scripts/md-to-narration.py "$src" -o "$txt.regen" --emit-map "$map.regen" >/dev/null 2>&1
    if ! cmp -s "$txt.regen" "$txt"; then
      warn "$base: markdown now converts differently than when the audio was made; keeping the
    original text so the manifest still describes the audio. Delete $wav to re-render."
    else
      mv -f "$txt.regen" "$txt"; mv -f "$map.regen" "$map"
    fi
    rm -f "$txt.regen" "$map.regen"
  else
    ./scripts/md-to-narration.py "$src" -o "$txt" --emit-map "$map" --stats 2>&1 | sed 's/^/    /'
  fi
  chars=$(wc -c < "$txt" | tr -d ' ')
  if [ "$chars" -gt "$MAX_CHARS" ]; then
    warn "$base: $chars chars over --max-chars $MAX_CHARS"; failed=$((failed+1)); continue
  fi

  if [ ! -s "$wav" ]; then
    body="$OUT/.$base.json"
    python3 -c "import json,sys; b={'text': open(sys.argv[1]).read()}; s=sys.argv[3] if len(sys.argv)>3 and sys.argv[3] else None; b.update({'seed': int(s)} if s else {}); json.dump(b, open(sys.argv[2],'w'))" "$txt" "$body" "$SEED"
    code=$(curl -s --max-time 14400 -o "$wav" -D "$OUT/.$base.headers" -w '%{http_code}' \
      -X POST "http://127.0.0.1:$PORT/tts" -H 'content-type: application/json' \
      -H "X-API-Key: $KEY" --data-binary @"$body")
    rm -f "$body"
    if [ "$code" != 200 ]; then
      warn "$base: HTTP $code — $(head -c 300 "$wav" 2>/dev/null)"; rm -f "$wav"
      failed=$((failed+1)); continue
    fi
    grep -ihE '^x-(audio-seconds|rtf)' "$OUT/.$base.headers" | tr -d '\r' | sed 's/^/    /'
  fi

  if [ ! -s "$webm" ]; then
    ffmpeg -v error -y -i "$wav" -c:a libopus -b:a "$BITRATE" -vbr on -ac 1 \
      -f webm -metadata title="$base" "$webm" \
      || { warn "$base: webm encode failed"; failed=$((failed+1)); continue; }
    wd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav")
    od=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$webm")
    python3 -c "import sys; sys.exit(0 if abs($wd-$od)<0.10 else 1)" \
      || warn "$base: wav/webm duration drift $(python3 -c "print(f'{abs($wd-$od):.3f}')")s"
    printf '    webm %s\n' "$(du -h "$webm" | cut -f1)"
  fi

  if [ -n "$ALIGN_PYTHON" ]; then
    # A non-zero exit here means a quality gate failed, not that alignment crashed: the
    # manifest is still written. Record it so the book's summary names the chapters to look
    # at rather than only counting them.
    "$ALIGN_PYTHON" scripts/align-narration.py --audio "$webm" --text "$txt" \
      --map "$map" --out "$man" --title "$base" 2>&1 | sed 's/^/    /'
    if [ ! -s "$man" ]; then
      warn "$base: alignment produced no manifest"; failed=$((failed+1)); continue
    fi
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$man'))['quality']['valid'] else 1)"; then
      flagged+=("$base")
    fi
  fi
  ok=$((ok+1))
done

say "Done: $ok narrated, $skipped already present, $failed failed"
if [ ${#flagged[@]} -gt 0 ]; then
  printf '\033[33mquality gates failed:\033[0m %s\n' "${flagged[*]}" >&2
  printf 'Run scripts/verify-narration.py on these to see whether the audio or only the\n'
  printf 'alignment is at fault; a flagged chapter is not necessarily a bad render.\n' >&2
fi
[ "$failed" -eq 0 ] || exit 1
