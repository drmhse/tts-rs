#!/usr/bin/env python3
"""Does each narrated file actually say the chapter — all of it, not just the end?

Two independent checks, because they catch different failures:

1. **Completeness.** Transcribe the final seconds of the *delivered* file and look for the
   chapter's closing words. This proves the encode did not truncate and that synthesis
   reached the last sentence. It also proves the deliverable decodes.

2. **Integrity.** Find every span the aligner could not measure, transcribe those spans, and
   compare what was heard against what should have been said. This distinguishes the two
   reasons a span goes unmeasured, which matter very differently.

# Why the integrity check exists

The first version of this script checked only the last 35 seconds. Chapter 8 shipped with a
21-second passage of pure babble in the middle of it — the LLM had degenerated into a
repetition loop on a markdown table that `md-to-narration.py` had flattened into a run-on
with no sentence boundaries. Two independent recognisers transcribed it as "tampoligation,
tampolition, sambolition, and tambolion". Nothing in the pipeline noticed: durations matched,
the file decoded, the closing words were present, and the alignment statistics reported
coverage rather than correctness.

A tail check cannot find that, and neither can any measure of whether words *received*
timestamps. What finds it is asking whether the audio in a given span says what the script
says it should.

# Reading the output

    ok        the span was simply missed by recognition; the audio is fine
    BABBLE    the audio does not say the words — synthesis failed, re-render is required

`BABBLE` is an audio defect, not an alignment defect, and re-aligning will not fix it.

Usage:
    verify-narration.py narration/*.opus
    verify-narration.py --tail 45 --words 8 narration/chapter-3.opus
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "  twenty thirty forty fifty sixty seventy eighty ninety".split(" ")
# Only the spellings that actually differ between the book's prose and `small.en` output.
BRITISH = {
    "centre": "center", "centres": "centers", "colour": "color", "colours": "colors",
    "behaviour": "behavior", "behaviours": "behaviors", "organisation": "organization",
    "organisations": "organizations", "organise": "organize", "organised": "organized",
    "recognise": "recognize", "recognised": "recognized", "realise": "realize",
    "realised": "realized", "analyse": "analyze", "analysed": "analyzed",
    "licence": "license", "defence": "defense", "practise": "practice",
    "favour": "favor", "favours": "favors", "labour": "labor", "programme": "program",
    "prioritise": "prioritize", "prioritised": "prioritized", "modelling": "modeling",
    "travelled": "traveled", "cancelled": "canceled", "fulfil": "fulfill",
    "theatre": "theater",
}

# An unmeasured run this long is worth transcribing. Shorter runs are ordinary recognition
# misses on one or two words and are not evidence of anything.
SUSPICIOUS_RUN = 8
# Below this share of the span's words appearing in the transcript, the audio is not saying
# the script. Chapter 8's babbled table scored 0.34; its healthy tables scored 0.93-0.97.
BABBLE_OVERLAP = 0.60


def spell_number(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        t = TENS[n // 10]
        return t if n % 10 == 0 else f"{t} {ONES[n % 10]}"
    if n < 1000:
        head = f"{ONES[n // 100]} hundred"
        return head if n % 100 == 0 else f"{head} {spell_number(n % 100)}"
    return str(n)


def norm(text: str) -> list[str]:
    """Words only, lowercased, numbers spelled out, spellings harmonised.

    A first version compared raw strings and flagged two of twelve correct files: the script
    said `chapter 1` where the voice correctly said `chapter one`, and `centre` where
    `small.en` transcribed `center`. Both were the system working and the checker being naive.
    """
    out: list[str] = []
    for w in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower()):
        if w.isdigit():
            out.extend(spell_number(int(w)).split())
        else:
            out.append(BRITISH.get(w, w))
    return out


def transcribe(model, audio: Path, start: float | None = None,
               seconds: float | None = None, tail: int | None = None) -> str:
    """Transcribe a clip. Either a tail (`-sseof`) or an absolute span."""
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.wav"
        cut = ["-sseof", f"-{tail}"] if tail is not None else ["-ss", f"{start:.3f}"]
        span = [] if seconds is None else ["-t", f"{seconds:.3f}"]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *cut, "-i", str(audio), *span,
             "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(clip)],
            check=True,
        )
        segments, _ = model.transcribe(str(clip), language="en", beam_size=5)
        return "".join(s.text for s in segments)


def unmeasured_runs(manifest: dict, minimum: int) -> list[tuple[int, int]]:
    """Spans of consecutive words the aligner could not measure, as (start index, length)."""
    words = manifest["transcript"]["words"]
    runs: list[tuple[int, int]] = []
    start, length = 0, 0
    for i, word in enumerate(words):
        if word.get("interpolated"):
            if length == 0:
                start = i
            length += 1
        else:
            if length >= minimum:
                runs.append((start, length))
            length = 0
    if length >= minimum:
        runs.append((start, length))
    return runs


def check_integrity(model, audio: Path, manifest: dict, pad: float) -> list[dict]:
    """Transcribe every suspicious span and report whether the audio says the words."""
    words = manifest["transcript"]["words"]
    duration = manifest["delivery"]["totalDurationSeconds"]
    findings = []
    for start, length in unmeasured_runs(manifest, SUSPICIOUS_RUN):
        end = start + length - 1
        t0 = max(0.0, words[start]["start"] - pad)
        t1 = min(duration, words[end]["end"] + pad)
        # A compressed run has almost no time span of its own, so widen to something audible.
        if t1 - t0 < 4.0:
            t1 = min(duration, t0 + 12.0)
        heard = set(norm(transcribe(model, audio, start=t0, seconds=t1 - t0)))
        want = norm(" ".join(w["text"] for w in words[start : start + length]))
        overlap = sum(1 for w in want if w in heard) / max(1, len(want))
        findings.append({
            "wordIndex": start,
            "words": length,
            "start": round(t0, 1),
            "end": round(t1, 1),
            "overlap": round(overlap, 2),
            "babble": overlap < BABBLE_OVERLAP,
            "text": " ".join(w["text"] for w in words[start : start + length])[:90],
        })
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--tail", type=int, default=35, help="seconds of audio to transcribe")
    ap.add_argument("--words", type=int, default=6, help="closing words to report on")
    ap.add_argument("--slack", type=int, default=3,
                    help="script words allowed to go unmatched at the very end; a\n                          homophone costs one or two, truncation costs many")
    ap.add_argument("--pad", type=float, default=3.0, help="seconds around a suspicious span")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--skip-integrity", action="store_true")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"{'file':<18}{'ending':<9}{'spans':<7}{'worst':<8}{'missing':<22}tail")
    print("-" * 104)
    incomplete: list[str] = []
    babbled: list[tuple[str, dict]] = []

    for audio in sorted(args.files):
        script = audio.with_suffix(".txt")
        if not script.exists():
            print(f"{audio.name:<18}{'?':<9}no .txt beside the audio")
            incomplete.append(audio.name)
            continue

        heard = norm(transcribe(model, audio, tail=args.tail))
        script_words = norm(script.read_text())
        # How far into the script the heard tail *reaches*, rather than which closing words
        # are literally present. The presence test asked the wrong question and failed correct
        # files twice in twelve chapters, both times on a homophone the recogniser cannot
        # resolve: "adding work to writes" heard as "rights", and "the system's invariants" as
        # "the systems and variants". Rendering both readings of the latter produced the same
        # transcript for each, so the voice was right and the recogniser cannot tell them
        # apart — and at two per twelve chapters that is a steady stream of investigations
        # into nothing.
        #
        # Truncation and substitution differ in size, which is what makes them separable. A
        # chapter that stops early leaves every remaining word unmatched; a homophone leaves
        # one or two. So align the heard tail against the script's closing words and measure
        # the shortfall past the last match.
        tail_scope = script_words[-max(args.words * 20, 60):]
        blocks = difflib.SequenceMatcher(None, heard, tail_scope,
                                         autojunk=False).get_matching_blocks()
        reached = max((pi + size for _, pi, size in blocks if size), default=0)
        shortfall = len(tail_scope) - reached
        want = script_words[-args.words :]
        missing = [w for w in want if w not in heard]
        if shortfall > args.slack:
            incomplete.append(audio.name)
        elif missing:
            # Reached the end, but not word-for-word. Worth seeing, not worth a re-render.
            missing = [f"~{w}" for w in missing]

        findings: list[dict] = []
        man_path = audio.with_name(audio.stem + ".manifest.json")
        if not args.skip_integrity and man_path.exists():
            findings = check_integrity(model, audio, json.loads(man_path.read_text()), args.pad)
            babbled.extend((audio.name, f) for f in findings if f["babble"])

        worst = min((f["overlap"] for f in findings), default=None)
        print(
            f"{audio.name:<18}{'CHECK' if audio.name in incomplete else 'ok':<9}"
            f"{len(findings):<7}{('-' if worst is None else f'{worst:.2f}'):<8}"
            f"{(','.join(missing) or '-'):<22}...{' '.join(heard[-7:])}"
        )
        for finding in findings:
            if finding["babble"]:
                print(f"    BABBLE {finding['start']}-{finding['end']}s "
                      f"overlap {finding['overlap']}: {finding['text']}")

    print()
    if incomplete:
        print(f"incomplete or unverifiable: {incomplete}")
    if babbled:
        print("AUDIO DEFECTS — these need re-rendering, not re-aligning:")
        for name, finding in babbled:
            print(f"  {name} at {finding['start']}s ({finding['words']} words, "
                  f"overlap {finding['overlap']})")
    if not incomplete and not babbled:
        print(f"all {len(args.files)} files reach their final words and say what they should")
    return 1 if (incomplete or babbled) else 0


if __name__ == "__main__":
    raise SystemExit(main())
