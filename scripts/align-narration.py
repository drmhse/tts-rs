#!/usr/bin/env python3
"""Word-accurate alignment of narrated audio, and the manifest the web player reads.

Every timing here is measured from the audio. That is the whole design, and two earlier
versions of this file got it wrong in ways worth recording, because in both cases the
failure was invisible to the statistics being reported.

# Mistake one: alignment without recognition

The first version did *forced alignment only*. It skipped recognition, cut the script into
sentence-sized windows, and gave each window a time span in proportion to its character
count. wav2vec2 then placed words inside those windows.

wav2vec2 can only place words *within the span it is handed*, and has no mechanism to
reject a bad span. The spans were bad: proportional-by-character assumes a constant speaking
rate, which narration violates at every heading, paragraph break and inserted gap. Boundary
error was inherited by every word in the window and accumulated down the chapter.

The reported statistics could not see it. `alignedShare 99.4%` and `coverageEndSeconds
1033.6/1033.7` say only that words *received* timestamps spanning the file. A word eight
seconds from its true position scores exactly as well as a correct one. Coverage was
measured and reported as though it were accuracy.

# Mistake two: adding an acoustic pass on top

The rewrite kept wav2vec2 but cut its windows at recognised anchors, on the reasoning that
acoustic onsets must beat recognition timestamps. Measured on chapter 1 it was far worse —
and this time the validation caught it before anything shipped:

| | words placed | drift p90 vs recognition | worst |
|---|---|---|---|
| recognition timestamps alone | 2488/2524 | — | — |
| plus wav2vec2 in anchored windows | 1764/2524 | 4.03 s | 10.9 s |

Subdividing the failing windows made it worse still: 145 windows failing, then 275. The
stage was discarding timings that were already correct. It is not here, deliberately, and
the numbers above are the reason not to reintroduce it.

# What this does

1. **Recognise** the audio — `faster-whisper`, batched, `word_timestamps=True`. Measured
   20.4x realtime, 51 s for a 17-minute chapter. Batching is what makes this affordable:
   whisperx's unbatched pass is 7.8x on the same machine.
2. **Globally align** the recognised word sequence to the book's canonical words, so each
   canonical word takes the time recognition reported for it. The canonical text stays
   authoritative for spelling, so recognition errors never reach the page.
3. **Interpolate only what is left**, marking every such word `interpolated` so the manifest
   never implies a measurement that was not made.
4. **Cut cues at measured pauses**, and check those cuts against silences ffmpeg detected
   independently of everything above.

This is the same construction the reference pipeline for the other books used with Gemini
(`forceAlignWordsToCanonical` in `gemini-transcribe-audio.mjs`). That pipeline's quality came
from measuring boundaries from the audio and validating them — not from the model — so
whisper substitutes cleanly, at 20x realtime and with no API dependency.

# Why cue length is the number that matters

`audio-player.js` does not sweep the highlight from word timings. `updateCumulativeHighlights`
interpolates across the active cue:

    wordProgress = progress * activeWords.length - index

So cue length bounds the visible error however good the word timings are, and a cue must not
span a pause — the words after the pause would highlight during silence, which is what reads
as "the highlight is ahead of the voice". Cues target ~3 s and are cut at pauses that were
measured.

Usage:
    align-narration.py --audio chapter-1.opus --text chapter-1.txt \\
        --map chapter-1.map.json --out chapter-1.manifest.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")

# Player cue length. The highlight is interpolated inside a cue, so this — not word timing
# precision — sets the worst visible error.
TARGET_CUE_SECONDS = 3.0
MAX_CUE_SECONDS = 5.0
MIN_CUE_SECONDS = 1.2

# A gap long enough to call a pause and cut a cue at. Below this, the space between words is
# phrase rhythm rather than a place the voice actually stops.
PAUSE_SECONDS = 0.22

# Quality gates. Failing any of these sets `quality.valid = false` naming the reason, rather
# than shipping a manifest that looks fine and drifts.
MIN_MEASURED_SHARE = 0.95         # canonical words carrying a recognised time

# A matching run this short is not evidence. `SequenceMatcher` happily reports a size-1 match
# on a word like `and`, pairing an occurrence in the script with an unrelated occurrence in the
# recognised stream. On chapter 8 two such matches on `and` pinned 62 canonical words into
# 1.1 s of audio: the anchors were 794.32 s and 795.44 s, and everything between them was
# compressed into the gap. Anchors must be supported by their neighbours.
MIN_ANCHOR_BLOCK = 3

# Narration runs about 2.5-3 words/second. A pair of anchors implying much more than this has
# at least one false member, whatever its block length, so the weaker one is dropped and the
# region becomes an honest hole instead of a compressed lie.
MAX_WORDS_PER_SECOND = 5.0
MAX_INTERPOLATED_RUN = 15         # consecutive unmeasured words before it is a real hole
MIN_CUE_PAUSE_AGREEMENT = 0.50    # cue cuts landing at a detected silence


# Recognition and the book spell the same sound differently, and every such difference used to
# count as an unmeasured word. Aggregating the manifests showed ~100 words per book lost this
# way: the book writes "thirty" where whisper writes "30", and "centre" where it writes
# "center". Neither is a timing problem, but both depressed the measured share and opened
# false holes, so both sides are folded to one form before matching.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
BRITISH = {
    "centre": "center", "centres": "centers", "colour": "color", "colours": "colors",
    "behaviour": "behavior", "behaviours": "behaviors", "organisation": "organization",
    "organisations": "organizations", "organise": "organize", "organised": "organized",
    "recognise": "recognize", "recognised": "recognized", "realise": "realize",
    "realised": "realized", "analyse": "analyze", "analysed": "analyzed",
    "licence": "license", "defence": "defense", "practise": "practice", "favour": "favor",
    "labour": "labor", "programme": "program", "prioritise": "prioritize",
    "modelling": "modeling", "travelled": "traveled", "cancelled": "canceled",
    "fulfil": "fulfill", "theatre": "theater",
}


def raw_normalize(word: str) -> str:
    """Lowercased letters and digits only, with no folding.

    Must match `normalize_word` in md-to-narration.py, which builds the page-word map.
    """
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("\u2019", "'"))


def normalize(word: str) -> str:
    w = re.sub(r"[^a-z0-9']", "", word.lower().replace("\u2019", "'"))
    return NUMBER_WORDS.get(w) or BRITISH.get(w, w)


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


# Containers the site serves, and the MIME type each needs. WebM is the site's convention for
# every other book, and the reason matters: Safari's support for Opus in an Ogg container is
# unreliable, while Opus in WebM plays. Shipping `.opus` therefore risks a silent no-audio
# failure on Safari that no amount of correct alignment would fix.
MIME = {
    "webm": "audio/webm; codecs=opus",
    "ogg": "audio/ogg; codecs=opus",
    "opus": "audio/ogg; codecs=opus",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
}


def probe_delivery(path: Path) -> dict:
    """Codec, container and bitrate of the file being served, read from the file itself.

    Written into the manifest rather than assumed, so the manifest describes what is actually
    delivered even if the encode settings change.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,bit_rate:format=format_name,bit_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    probed = json.loads(out.stdout)
    stream = (probed.get("streams") or [{}])[0]
    fmt = probed.get("format", {})
    container = path.suffix.lstrip(".").lower()
    bits = stream.get("bit_rate") or fmt.get("bit_rate")
    return {
        "codec": stream.get("codec_name", "unknown"),
        "container": container,
        "formatName": fmt.get("format_name", ""),
        "mimeType": MIME.get(container, "application/octet-stream"),
        "bitrate": f"{round(int(bits) / 1000)}k" if bits else None,
    }


def detect_silences(path: Path, noise_db: int = 40, min_seconds: float = 0.20
                    ) -> list[tuple[float, float]]:
    """Silence intervals, from ffmpeg's own analysis of the waveform.

    Deliberately independent of recognition: this is the evidence used to check where cues
    were cut, so it must not share a failure mode with the thing it checks.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-af",
         f"silencedetect=noise=-{noise_db}dB:d={min_seconds}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in re.findall(r"silence_(start|end):\s*(-?[0-9.]+)", out.stderr):
        seconds = float(value)
        if kind == "start":
            start = seconds
        elif start is not None:
            spans.append((start, seconds))
            start = None
    return spans


@dataclass
class Recognised:
    """A word the recogniser heard, and when it heard it."""
    normalized: str
    start: float
    end: float


def recognise(audio: Path, model_name: str, batch_size: int
              ) -> tuple[list[Recognised], float]:
    """Transcribe with batched faster-whisper. Returns words and elapsed seconds."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    pipeline = BatchedInferencePipeline(
        model=WhisperModel(model_name, device="cpu", compute_type="int8")
    )
    t0 = time.time()
    segments, _ = pipeline.transcribe(
        str(audio), language="en", beam_size=1, batch_size=batch_size, word_timestamps=True
    )
    words: list[Recognised] = []
    for segment in segments:
        for word in segment.words or []:
            n = normalize(word.word)
            if n:
                words.append(Recognised(n, float(word.start), float(word.end)))
    return words, time.time() - t0


@dataclass
class Canonical:
    """A word of the book. The book is authoritative for spelling, the audio for timing."""
    text: str
    # Folded form, for matching against recognition: numbers as digits, British spellings
    # as American. See `normalize`.
    normalized: str
    # Unfolded form. The page-word map in `--map` is built by md-to-narration.py with its own
    # normaliser, which does not fold, so comparing a folded word against it desynchronises
    # the cursor and the rest of the chapter never maps. That regression took one chapter from
    # 100% page-mapped to 18.5%.
    raw: str
    char_start: int
    char_end: int
    start: float | None = None
    end: float | None = None
    # False when no recognised time was available and the value was interpolated.
    measured: bool = False
    # Length of the matching run that produced this word's time. Longer runs are stronger
    # evidence, and this is what decides which anchor to discard when two disagree.
    support: int = 0


def canonical_words(text: str) -> list[Canonical]:
    return [
        Canonical(m.group(0), normalize(m.group(0)), raw_normalize(m.group(0)),
                  m.start(), m.end())
        for m in WORD.finditer(text)
        if normalize(m.group(0))
    ]


def attach_times(canon: list[Canonical], heard: list[Recognised]) -> int:
    """Match the two word sequences and copy recognised times onto canonical words.

    `SequenceMatcher` handles the three things recognition does to a script — substitution,
    dropping, insertion — without any of them shifting the words that follow. That is what
    the positional scan in the first version could not do: it desynchronised on the first
    disagreement and mapped 179 of 2511 words.

    `autojunk` must stay off. It classifies tokens appearing in more than 1% of a long
    sequence as junk, which in a 2500-word chapter silently discards `the`, `to`, `of` and
    `a` — several hundred of the most reliable anchors in the text.

    Only runs of at least `MIN_ANCHOR_BLOCK` words are trusted; see that constant for the
    failure a shorter run caused.
    """
    matcher = difflib.SequenceMatcher(
        None, [c.normalized for c in canon], [h.normalized for h in heard], autojunk=False
    )
    measured = 0
    for ci, hi, size in matcher.get_matching_blocks():
        if size < MIN_ANCHOR_BLOCK:
            continue
        for k in range(size):
            word = canon[ci + k]
            word.start, word.end = heard[hi + k].start, heard[hi + k].end
            word.measured = True
            word.support = size
            measured += 1
    return measured


def drop_implausible_anchors(canon: list[Canonical]) -> int:
    """Discard anchors that imply an impossible speaking rate.

    A surviving false anchor shows up as two measured words too close together in time to
    contain the words the script puts between them. Rather than guess which is wrong, drop
    the one with the weaker matching run and re-check, since removing an anchor can only
    widen a hole — never create a new violation.

    Without this, chapter 8's babbled passage compressed 62 words into 1.1 s and the manifest
    reported them as measured.
    """
    dropped = 0
    while True:
        measured = [i for i, c in enumerate(canon) if c.measured]
        worst = None
        for left, right in zip(measured, measured[1:]):
            between = right - left - 1
            if between <= 0:
                continue
            span = canon[right].start - canon[left].end
            rate = between / span if span > 1e-6 else float("inf")
            if rate > MAX_WORDS_PER_SECOND and (worst is None or rate > worst[0]):
                worst = (rate, left, right)
        if worst is None:
            return dropped
        _, left, right = worst
        victim = left if canon[left].support <= canon[right].support else right
        canon[victim].measured = False
        canon[victim].start = canon[victim].end = None
        canon[victim].support = 0
        dropped += 1


def fill_holes(canon: list[Canonical]) -> list[int]:
    """Interpolate words recognition did not place. Returns the size of each hole.

    Every hole is bounded on both sides by a measured word, so error cannot accumulate past
    the next measured word — the property the first version lacked entirely.
    """
    measured = [i for i, c in enumerate(canon) if c.measured]
    if not measured:
        raise SystemExit("recognition and script did not overlap at all")

    runs: list[int] = []
    for i in range(measured[0]):
        canon[i].start = canon[i].end = canon[measured[0]].start
    for i in range(measured[-1] + 1, len(canon)):
        canon[i].start = canon[i].end = canon[measured[-1]].end
    if measured[0]:
        runs.append(measured[0])
    if len(canon) - measured[-1] - 1:
        runs.append(len(canon) - measured[-1] - 1)

    for left, right in zip(measured, measured[1:]):
        if right - left <= 1:
            continue
        runs.append(right - left - 1)
        a, b = canon[left].end, canon[right].start
        step = (b - a) / (right - left)
        for k in range(left + 1, right):
            canon[k].start = a + step * (k - left - 1)
            canon[k].end = a + step * (k - left)
    return runs


def enforce_monotonic(canon: list[Canonical], duration: float) -> int:
    """`syncTranscript` uses `findIndex(s => t >= s.start && t < s.end)`.

    An overlap therefore selects the *earlier* cue, and the highlight jumps backwards. So
    overlapping boundaries are not cosmetic — they are a visible defect. Clamp forward.
    """
    clamped = 0
    for i in range(1, len(canon)):
        if canon[i].start < canon[i - 1].end:
            canon[i].start = canon[i - 1].end
            clamped += 1
        if canon[i].end < canon[i].start:
            canon[i].end = canon[i].start
    for c in canon:
        c.start = max(0.0, min(c.start, duration))
        c.end = max(c.start, min(c.end, duration))
    return clamped


def build_cues(canon: list[Canonical], text: str) -> list[dict]:
    """Group words into ~3 s cues, preferring cuts at a measured pause.

    Because word times are measured, the pauses between them are visible and can be cut on.
    A cue spanning a pause highlights words during silence.
    """
    cues: list[dict] = []
    start = 0
    for i in range(len(canon)):
        last = i + 1 == len(canon)
        span = canon[i].end - canon[start].start
        gap = 0.0 if last else canon[i + 1].start - canon[i].end
        after = "" if last else text[canon[i].char_end : canon[i + 1].char_start]
        sentence = bool(re.search(r"[.!?]", after))
        clause = bool(re.search(r"[,;:]", after))

        cut = last
        if not cut and span >= MIN_CUE_SECONDS and (sentence or gap >= PAUSE_SECONDS):
            cut = True
        elif not cut and span >= TARGET_CUE_SECONDS and (clause or gap >= PAUSE_SECONDS / 2):
            cut = True
        elif not cut and span >= MAX_CUE_SECONDS:
            cut = True

        if cut:
            cues.append({
                "start": round(canon[start].start, 3),
                "end": round(canon[i].end, 3),
                "wordStart": start,
                "wordEnd": i + 1,
                "text": " ".join(text[canon[start].char_start : canon[i].char_end].split()),
            })
            start = i + 1
    for i in range(1, len(cues)):
        if cues[i]["start"] < cues[i - 1]["end"]:
            cues[i]["start"] = cues[i - 1]["end"]
    return cues


def cue_pause_agreement(cues: list[dict], silences: list[tuple[float, float]],
                        tolerance: float = 0.35) -> float:
    """Share of cue boundaries landing at a silence ffmpeg found on its own.

    This is the check that does not depend on the timings it judges. A manifest can be
    perfectly self-consistent and still be shifted against the audio; that shows up here and
    nowhere else, which is precisely what was missing when the first version shipped.
    """
    if len(cues) < 2:
        return 1.0
    hits = sum(
        1 for cue in cues[1:]
        if any(s - tolerance <= cue["start"] <= e + tolerance for s, e in silences)
    )
    return hits / (len(cues) - 1)


def map_page_words(words: list[dict], canon: list[Canonical], map_path: Path) -> int:
    """Attach page-word indices for the DOM highlighting.

    The page tokeniser (`wordPattern` in audio-player.js) and this one can disagree, so the
    cursor resyncs within a bounded window instead of desynchronising for the rest of the
    chapter.
    """
    pm = json.loads(map_path.read_text())
    spoken, s2p = pm["spokenWords"], pm["spokenToPageWord"]
    mapped, cursor = 0, 0
    for index, c in enumerate(canon):
        if cursor >= len(spoken) or spoken[cursor]["normalized"] != c.raw:
            for ahead in range(1, 25):
                if (cursor + ahead < len(spoken)
                        and spoken[cursor + ahead]["normalized"] == c.raw):
                    cursor += ahead
                    break
        if cursor < len(spoken) and spoken[cursor]["normalized"] == c.raw:
            page_index = s2p[cursor]
            if page_index >= 0:
                words[index]["pageWordIndex"] = page_index
                # `explicitPageWordMatches` checks this against the word it finds in the DOM
                # and drops the match if they disagree. Omitting it to save bytes disables
                # the player's only defence against a mis-mapped index, which is a bad trade:
                # a wrong index highlights the wrong word with no way to notice.
                words[index]["pageWordNormalized"] = c.normalized
                mapped += 1
        cursor += 1
    return mapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True, help="the file that will be served")
    ap.add_argument("--text", type=Path, required=True, help="narration text that produced it")
    ap.add_argument("--map", type=Path, help="page-word map from md-to-narration.py --emit-map")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    text = args.text.read_text()
    # Measure the file that will actually be served, so encoder delay is included rather than
    # inherited from a WAV the browser never sees.
    duration = duration_of(args.audio)
    canon = canonical_words(text)
    if not canon:
        sys.exit("no words in narration text")

    t0 = time.time()
    heard, asr_seconds = recognise(args.audio, args.asr_model, args.batch_size)
    measured = attach_times(canon, heard)
    dropped = drop_implausible_anchors(canon)
    measured -= dropped
    holes = fill_holes(canon)
    clamped = enforce_monotonic(canon, duration)
    silences = detect_silences(args.audio)
    cues = build_cues(canon, text)
    delivery = probe_delivery(args.audio)

    words = [
        {"text": c.text, "start": round(c.start, 3), "end": round(c.end, 3)} for c in canon
    ]
    for word, c in zip(words, canon):
        if not c.measured:
            word["interpolated"] = True

    mapped = map_page_words(words, canon, args.map) if args.map and args.map.exists() else 0

    measured_share = measured / len(canon)
    agreement = cue_pause_agreement(cues, silences)
    longest_hole = max(holes, default=0)
    cue_lengths = sorted(c["end"] - c["start"] for c in cues)

    def pct(values: list[float], q: float) -> float:
        return round(values[min(len(values) - 1, int(len(values) * q))], 3) if values else 0.0

    issues = []
    if measured_share < MIN_MEASURED_SHARE:
        issues.append(
            f"only {100 * measured_share:.1f}% of words carry a measured time "
            f"(gate {100 * MIN_MEASURED_SHARE:.0f}%)"
        )
    if longest_hole > MAX_INTERPOLATED_RUN:
        issues.append(
            f"{longest_hole} consecutive interpolated words (gate {MAX_INTERPOLATED_RUN})"
        )
    if agreement < MIN_CUE_PAUSE_AGREEMENT:
        issues.append(
            f"only {100 * agreement:.0f}% of cue cuts land at a detected pause "
            f"(gate {100 * MIN_CUE_PAUSE_AGREEMENT:.0f}%)"
        )
    if cue_lengths and cue_lengths[-1] > MAX_CUE_SECONDS * 2:
        issues.append(f"longest cue {cue_lengths[-1]:.1f}s")

    manifest = {
        "format": "tts-rs-narration-v2",
        "title": args.title or args.audio.stem,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "timing": "faster-whisper word timestamps mapped onto canonical text",
            "asrModel": args.asr_model,
        },
        "delivery": {
            # Matches the shape the site's other books use (`single-opus-webm-v1`), so the
            # asset validator and delivery tooling see a familiar manifest.
            "format": f"single-{delivery['codec']}-{delivery['container']}-v1",
            **{k: v for k, v in delivery.items() if v is not None},
            "totalDurationSeconds": round(duration, 3),
            # One file per chapter. The player sums `startSeconds` across entries, so
            # splitting later means adding entries here and nothing else.
            "segments": [{
                "index": 1,
                "file": args.audio.name,
                "durationSeconds": round(duration, 3),
                "startSeconds": 0,
                "endSeconds": round(duration, 3),
            }],
        },
        "transcript": {
            "segments": cues,
            "words": words,
            "stats": {
                "canonicalWords": len(canon),
                "recognisedWords": len(heard),
                "measuredWords": measured,
                "measuredShare": round(measured_share, 4),
                "interpolatedWords": len(canon) - measured,
                "droppedImplausibleAnchors": dropped,
                "longestInterpolatedRun": longest_hole,
                "pageMappedWords": mapped,
                "segments": len(cues),
                "medianSegmentSeconds": pct(cue_lengths, 0.5),
                "p90SegmentSeconds": pct(cue_lengths, 0.9),
                "longestSegmentSeconds": round(cue_lengths[-1], 3) if cue_lengths else 0,
                "clampedWordStarts": clamped,
                "coverageEndSeconds": round(words[-1]["end"], 3) if words else 0,
                # The independent check: cue cuts against ffmpeg's own silence analysis.
                "detectedSilences": len(silences),
                "cuePauseAgreement": round(agreement, 4),
                "asrSeconds": round(asr_seconds, 1),
                "asrRealtimeFactor": round(duration / max(asr_seconds, 1e-9), 1),
                "totalSeconds": round(time.time() - t0, 1),
            },
        },
        "quality": {
            "valid": not issues,
            "issues": issues,
            "gates": {
                "minMeasuredShare": MIN_MEASURED_SHARE,
                "maxInterpolatedRun": MAX_INTERPOLATED_RUN,
                "minCuePauseAgreement": MIN_CUE_PAUSE_AGREEMENT,
            },
        },
    }
    args.out.write_text(json.dumps(manifest, separators=(",", ":")))

    st = manifest["transcript"]["stats"]
    print(
        f"{args.audio.name}: {len(cues)} cues (median {st['medianSegmentSeconds']}s, "
        f"p90 {st['p90SegmentSeconds']}s, max {st['longestSegmentSeconds']}s), "
        f"{measured}/{len(canon)} measured ({100 * measured_share:.1f}%), "
        f"longest hole {longest_hole}, page-mapped {mapped}, "
        f"cue/pause agreement {100 * agreement:.0f}%, "
        f"asr {st['asrSeconds']}s ({st['asrRealtimeFactor']}x), in {st['totalSeconds']}s",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"  FAIL: {issue}", file=sys.stderr)
    if words and duration - words[-1]["end"] > 5.0:
        print(f"  warning: {duration - words[-1]['end']:.1f}s of audio after the last word",
              file=sys.stderr)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
