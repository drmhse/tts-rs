#!/usr/bin/env bash
#
# Narrate a whole book from the swe site and install it, in one command.
#
#   scripts/narrate-swe.sh rust-engineering-handbook
#   scripts/narrate-swe.sh --list                       # what books are there
#
# Rendering only: no verification pass. It produces audio and every piece of narration
# metadata the site needs, then stops. `scripts/verify-narration.py narration-<slug>/*.webm`
# is the separate quality gate and is worth running before you publish for real.
#
# What it does, per book:
#   1. discover chapters                    narrate-book.sh --list  (nested part-NN/ layout)
#   2. markdown -> narration text + map     md-to-narration.py
#   3. text -> WAV -> WebM/Opus             tts-serve + ffmpeg
#   4. word-level alignment manifest        align-narration.py
#   5. install into the site + rewrite paths  publish-narration.py
#
# Steps 1-4 are `narrate-book.sh`, which is resumable per stage: a chapter that already has a
# manifest is skipped, one that has a WAV is only re-encoded. So an interrupted run costs
# nothing to resume, which matters when a big book is hours of synthesis.
#
# Step 5 is what makes the audio appear on the site. `publish-narration.py` copies each
# chapter to `static/audio/books/<slug>/chapter-NNN/chapter.webm`, puts the manifest beside
# it, rewrites `delivery.segments[].file` to the published name, and writes `audio` and
# `audio_duration` into the chapter's front matter. The player renders only when that front
# matter is present, so this step is not optional.
set -uo pipefail
cd "$(dirname "$0")/.."

SITE="${SWE_SITE:-$HOME/Desktop/projects/AI/swe}"
ENGINE="${NARRATE_ENGINE:-qwen3tts}"
OUT=""
SLUG=""
ONLY=""
# Renders are deterministic under a seed, which is what makes a resumed run reproduce the same
# audio instead of a different valid draw. Fixed rather than left to the server's default so a
# chapter re-rendered next week matches the one beside it.
SEED="${NARRATE_SEED:-1234}"
LIST=0
NO_PUBLISH=0
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --site) SITE="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    --no-publish) NO_PUBLISH=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    -*) printf 'unknown flag %s\n' "$1" >&2; exit 2 ;;
    *) SLUG="$1"; shift ;;
  esac
done

say()  { printf '\n\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

BOOKS="$SITE/content/books"
[ -d "$BOOKS" ] || die "no books at $BOOKS (set --site or \$SWE_SITE)"

if [ "$LIST" = 1 ]; then
  printf '%-52s %s\n' "SLUG" "CHAPTERS"
  for d in "$BOOKS"/*/; do
    [ -d "$d" ] || continue
    s=$(basename "$d")
    n=$(find "$d" -name 'chapter-*.md' -not -name '_index.md' | wc -l | tr -d ' ')
    [ "$n" -gt 0 ] && printf '%-52s %s\n' "$s" "$n"
  done
  exit 0
fi

[ -n "$SLUG" ] || die "usage: narrate-swe.sh <book-slug>   (--list to see them)"
BOOK="$BOOKS/$SLUG"
[ -d "$BOOK" ] || die "no book '$SLUG' under $BOOKS — try --list"
OUT="${OUT:-narration-$SLUG}"

# Reuse the renderer's own discovery rather than reimplementing it here: two globs that drift
# apart would silently narrate a different set of chapters than the one that gets published.
# `while read` rather than `mapfile`: macOS ships bash 3.2, which has no mapfile.
ROWS=()
while IFS= read -r row; do [ -n "$row" ] && ROWS+=("$row"); done < <(
  scripts/narrate-book.sh --list --book "$BOOK" ${ONLY:+--only "$ONLY"}
)
[ "${#ROWS[@]}" -gt 0 ] || die "no chapters found in $BOOK"

# The plan, before committing to hours of GPU. Word counts come from the markdown, so they
# include some fencing and front matter the narration text drops — the estimate runs a little
# long rather than a little short, which is the better direction to be wrong in.
words=0; pending=0; pending_words=0
for row in "${ROWS[@]}"; do
  name="${row%%$'\t'*}"; src="${row#*$'\t'}"
  w=$(wc -w < "$src" | tr -d ' ')
  words=$((words + w))
  if [ ! -s "$OUT/$name.manifest.json" ]; then
    pending=$((pending + 1)); pending_words=$((pending_words + w))
  fi
done

# 141 wpm and RTF 0.32 are measured for qwen3tts at f16 on this machine
# (docs/performance/qwen3tts-batching.md). Other engines are slower; the per-chapter ETA that
# narrate-book.sh prints is measured live and supersedes this one.
read -r est_audio est_wall <<EOF
$(python3 -c "
w=$pending_words
rtf = {'qwen3tts': 0.32, 'cosyvoice': 0.71, 'audio8': 0.35}.get('$ENGINE', 0.7)
audio = w / 141.0 * 60.0
print(int(audio), int(audio * rtf))
")
EOF
hms() { printf '%d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

say "$SLUG"
printf '  site      %s\n' "$SITE"
printf '  chapters  %s total, %s to render\n' "${#ROWS[@]}" "$pending"
printf '  words     %s total, %s to render\n' "$words" "$pending_words"
printf '  engine    %s (seed %s)\n' "$ENGINE" "$SEED"
printf '  estimate  %s of audio, ~%s of synthesis\n' "$(hms "$est_audio")" "$(hms "$est_wall")"
printf '  working   %s\n' "$OUT"
printf '  publish   %s\n' "$([ "$NO_PUBLISH" = 1 ] && echo "skipped (--no-publish)" \
  || echo "$SITE/static/audio/books/$SLUG + front matter")"

if [ "$DRY" = 1 ]; then say "Dry run; nothing done"; exit 0; fi
if [ "$pending" = 0 ]; then
  say "All ${#ROWS[@]} chapter(s) already rendered; publishing only"
else
  scripts/narrate-book.sh --book "$BOOK" --out "$OUT" --engine "$ENGINE" --seed "$SEED" \
    ${ONLY:+--only "$ONLY"} \
    || warn "some chapters failed; publishing whatever completed"
fi

if [ "$NO_PUBLISH" = 1 ]; then
  say "Rendered. Not published (--no-publish)."
  exit 0
fi

say "Installing into the site"
scripts/publish-narration.py --narration "$OUT" --site "$SITE" --slug "$SLUG" \
  || die "publish failed; audio is still in $OUT"

say "Done: $SLUG"
printf '  audio     %s\n' "$SITE/static/audio/books/$SLUG"
printf '  masters   %s (WAV, keep these — re-encoding is free, re-synthesis is not)\n' "$OUT"
printf '  verify    scripts/verify-narration.py %s/*.webm\n' "$OUT"
