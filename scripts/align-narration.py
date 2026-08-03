#!/usr/bin/env python3
"""Forced alignment for narrated audio, and the manifest the web player reads.

This is *forced* alignment, not transcription: the exact script that produced the audio is
known, so wav2vec2 only has to place known words in time. That is both more accurate than
recognition and much cheaper — the ASR pass that a normal whisperx flow spends most of its
time on exists only to supply coarse windows, and a script supplies better ones for free.

Measured on a 17-minute chapter, CPU:

| | total | RTF |
|---|---|---|
| whisperx with its ASR pass | 132.7 s | 0.128 |
| **this, windows from the script** | **26 s** | **0.025** |

# Window granularity is not cosmetic

`audio-player.js` does not sweep the highlight from word timings. `updateCumulativeHighlights`
takes the active *segment's* start/end and interpolates evenly across its words:

    wordProgress = progress * activeWords.length - index

So segment length decides whether the highlight tracks the voice. Paragraph-sized windows
(~50 s) produce perfectly good word timings and a highlight that drifts seconds mid-segment.
Sentences give ~3 s segments, which is what the shipped manifests use (~307 for 1669 s) and
what this emits.

Word timings are still written out — the player uses them to anchor auto-scroll, and they
are what a future version should sweep the highlight from instead of interpolating.

Usage:
    align-narration.py --audio chapter-1.opus --text chapter-1.txt \\
        --map chapter-1.map.json --out chapter-1.manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


# The player interpolates the highlight across a segment's words, so a long segment drifts
# however good the word timings inside it are. The shipped manifests cap cues at 45 s; this
# aims far lower, since median ~3 s is what makes the sweep invisible.
MAX_WINDOW_SECONDS = 12.0


def _split_long(chunk: str, seconds_per_char: float) -> list[str]:
    """Break a chunk that would exceed `MAX_WINDOW_SECONDS` into speakable pieces.

    Splitting on `.!?` alone left a 65.9 s window in one chapter: a short lead-in ending in a
    colon followed by a long block quote is a single "sentence" by that rule. Prefer clause
    punctuation, then fall back to a word count so no window can be unbounded.
    """
    if len(chunk) * seconds_per_char <= MAX_WINDOW_SECONDS:
        return [chunk]
    parts = [p for p in re.split(r"(?<=[,;:])\s+", chunk) if p.strip()]
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_split_long(p, seconds_per_char))
        return out
    # No punctuation to use: cut on words, at roughly the target length.
    words = chunk.split()
    budget = max(4, int(MAX_WINDOW_SECONDS / max(seconds_per_char * 6.0, 1e-6)))
    return [" ".join(words[i : i + budget]) for i in range(0, len(words), budget)] or [chunk]


def sentence_windows(text: str, duration: float) -> list[dict]:
    """Sentence-sized windows, timed in proportion to characters.

    wav2vec2 refines the boundaries, so these only have to contain the right words. Character
    count is a good proxy because the speaking rate is near-constant. The final window is
    forced to the audio end so nothing falls outside every window — an even split once left
    23 s of a chapter unaligned.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    total_chars = sum(len(s) for s in sents) or 1
    per_char = duration / total_chars

    chunks: list[str] = []
    for s in sents:
        chunks.extend(_split_long(s, per_char))

    total = sum(len(c) for c in chunks) or 1
    windows, at = [], 0.0
    for c in chunks:
        end = min(duration, at + len(c) / total * duration)
        windows.append({"start": at, "end": end, "text": c})
        at = end
    if windows:
        windows[-1]["end"] = duration
    return windows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True, help="the file that will be served")
    ap.add_argument("--text", type=Path, required=True, help="narration text that produced it")
    ap.add_argument("--map", type=Path, help="page-word map from md-to-narration.py --emit-map")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()

    import whisperx

    text = args.text.read_text()
    duration = duration_of(args.audio)
    # Align the file that will actually be served, so any encoder delay is baked in rather
    # than inherited from a WAV the browser never sees.
    audio = whisperx.load_audio(str(args.audio))

    model, meta = whisperx.load_align_model(language_code="en", device="cpu")
    t0 = time.time()
    aligned = whisperx.align(
        sentence_windows(text, duration), model, meta, audio, "cpu",
        return_char_alignments=False,
    )
    elapsed = time.time() - t0

    words, segments = [], []
    for seg in aligned["segments"]:
        timed = [w for w in seg.get("words", []) if "start" in w and "end" in w]
        if not timed:
            continue
        start_index = len(words)
        for w in timed:
            # `normalized` is omitted deliberately: the player computes
            # `normalizeWord(word.normalized || word.text)`, so shipping it duplicates
            # ~2900 strings per chapter for nothing. Same for `pageWordNormalized`, which
            # defaults to "" and is only used for build-time verification.
            words.append({
                "text": w["word"],
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
            })
        segments.append({
            "start": round(float(timed[0]["start"]), 3),
            "end": round(float(timed[-1]["end"]), 3),
            "wordStart": start_index,
            "wordEnd": len(words),
            "text": " ".join(w["word"] for w in timed),
        })

    # Attach page-word indices.
    #
    # The aligner emits words split on whitespace, so `self-evident` is one aligned word
    # where the page tokeniser (matching `wordPattern` in audio-player.js) sees two. Matching
    # by normalized string desynchronised on the first such word and then ran off the end —
    # 179 of 2511 mapped. Instead, consume as many script tokens as the aligned word actually
    # contains, and resync within a bounded window if they ever disagree.
    mapped = 0
    if args.map and args.map.exists():
        pm = json.loads(args.map.read_text())
        spoken = pm["spokenWords"]
        s2p = pm["spokenToPageWord"]
        page = pm["pageWords"]
        cursor = 0
        for word in words:
            subs = [normalize_word(t) for t in WORD.findall(word["text"])]  # noqa: E501
            subs = [t for t in subs if t]
            if not subs:
                continue
            if cursor >= len(spoken) or spoken[cursor]["normalized"] != subs[0]:
                # Bounded resync: a genuine mismatch costs a few unmapped words rather than
                # dragging the rest of the chapter out of step.
                for ahead in range(1, 25):
                    if (
                        cursor + ahead < len(spoken)
                        and spoken[cursor + ahead]["normalized"] == subs[0]
                    ):
                        cursor += ahead
                        break
            if cursor < len(spoken) and spoken[cursor]["normalized"] == subs[0]:
                page_index = s2p[cursor]
                if page_index >= 0:
                    word["pageWordIndex"] = page_index
                    mapped += 1
            cursor += len(subs)

    # Enforce the invariants the player assumes.
    #
    # `syncTranscript` picks a segment with `findIndex(s => t >= s.start && t < s.end)`, so
    # overlapping segments make it select the earlier one and the highlight can jump
    # backwards. wav2vec2 occasionally emits adjacent words overlapping by one 20 ms frame —
    # 2 boundaries in 392 on one chapter — which is inaudible but malformed. Clamp forward.
    clamped = 0
    for i in range(1, len(words)):
        if words[i]["start"] < words[i - 1]["end"]:
            words[i]["start"] = words[i - 1]["end"]
            clamped += 1
        if words[i]["end"] < words[i]["start"]:
            words[i]["end"] = words[i]["start"]
    for seg in segments:
        seg["start"] = words[seg["wordStart"]]["start"]
        seg["end"] = words[seg["wordEnd"] - 1]["end"]
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["end"]:
            segments[i]["start"] = segments[i - 1]["end"]

    script_words = len(WORD.findall(text))
    manifest = {
        "format": "tts-rs-narration-v1",
        "title": args.title or args.audio.stem,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "delivery": {
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
            "segments": segments,
            "words": words,
            "stats": {
                "scriptWords": script_words,
                "alignedWords": len(words),
                "alignedShare": round(len(words) / max(1, script_words), 6),
                "pageMappedWords": mapped,
                "segments": len(segments),
                "medianSegmentSeconds": round(
                    sorted(s["end"] - s["start"] for s in segments)[len(segments) // 2], 3
                ) if segments else 0,
                "longestSegmentSeconds": round(
                    max((s["end"] - s["start"] for s in segments), default=0), 3
                ),
                "alignSeconds": round(elapsed, 1),
                "clampedWordStarts": clamped,
                "coverageEndSeconds": round(words[-1]["end"], 3) if words else 0,
            },
        },
    }
    args.out.write_text(json.dumps(manifest, separators=(",", ":")))

    st = manifest["transcript"]["stats"]
    print(
        f"{args.audio.name}: {len(segments)} segments (median {st['medianSegmentSeconds']}s, "
        f"max {st['longestSegmentSeconds']}s), {len(words)}/{script_words} words "
        f"({100 * st['alignedShare']:.1f}%), page-mapped {mapped}, "
        f"covers {st['coverageEndSeconds']}s of {duration:.1f}s, in {elapsed:.0f}s",
        file=sys.stderr,
    )
    # A gap at the end means the highlight stops before the audio does.
    if words and duration - words[-1]["end"] > 5.0:
        print(
            f"  warning: {duration - words[-1]['end']:.1f}s of audio after the last aligned word",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
