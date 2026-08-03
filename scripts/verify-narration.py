#!/usr/bin/env python3
"""Did every rendered file actually reach the end of its chapter?

Matching durations proves the encode did not truncate; it says nothing about whether
synthesis reached the last sentence. This transcribes the final seconds of each *delivered*
file — so it also proves the deliverable decodes — and checks the chapter's closing words are
present.

# Why the comparison is normalised

A first version compared raw strings and flagged two of twelve correct files:

  * the script said `chapter 1`, the voice correctly said `chapter one`
  * the script said `centre`, whisper's `small.en` transcribed `center`

Both were the system doing the right thing and the checker being naive. Numbers are spelled
out, a small British/American spelling table is applied, and the match is over a bag of the
closing words rather than an exact substring — so word order jitter at a sentence end does
not fail an otherwise complete file.

Usage:
    verify-narration.py narration/*.opus
    verify-narration.py --tail 45 --words 8 narration/chapter-3.opus
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
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
}


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
    """Words only, lowercased, numbers spelled out, spellings harmonised."""
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
    out: list[str] = []
    for w in words:
        if w.isdigit():
            out.extend(spell_number(int(w)).split())
        else:
            out.append(BRITISH.get(w, w))
    return out


def transcribe_tail(model, audio: Path, seconds: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "tail.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-sseof", f"-{seconds}", "-i", str(audio),
             "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(clip)],
            check=True,
        )
        segs, _ = model.transcribe(str(clip), language="en", beam_size=1, batch_size=8)
        return "".join(s.text for s in segs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--tail", type=int, default=35, help="seconds of audio to transcribe")
    ap.add_argument("--words", type=int, default=6, help="closing words that must appear")
    args = ap.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = BatchedInferencePipeline(
        model=WhisperModel("small.en", device="cpu", compute_type="int8")
    )

    print(f"{'file':<20} {'ending':<8} {'missing':<28} tail")
    print("-" * 104)
    bad = []
    for audio in args.files:
        script = audio.with_suffix(".txt")
        if not script.exists():
            print(f"{audio.name:<20} {'?':<8} no .txt beside the audio")
            bad.append(audio.name)
            continue
        heard = norm(transcribe_tail(model, audio, args.tail))
        want = norm(script.read_text())[-args.words :]
        # A bag comparison: order jitter at a sentence end should not fail a complete file.
        missing = [w for w in want if w not in heard]
        ok = not missing
        if not ok:
            bad.append(audio.name)
        print(
            f"{audio.name:<20} {'ok' if ok else 'CHECK':<8} "
            f"{(','.join(missing) or '-'):<28} ...{' '.join(heard[-8:])}"
        )

    print()
    if bad:
        print(f"incomplete or unverifiable: {bad}")
        return 1
    print(f"all {len(args.files)} files reach their chapter's final words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
