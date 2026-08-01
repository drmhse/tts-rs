"""Word error rate against the input text, via Whisper.

The voice checks in `oracle/verify_voice.py` answer "whose voice is this"; they say nothing
about whether the words are right. A cloning path can produce the correct speaker saying
mush, and a sampler bug shows up here long before it shows up in F0 — Audio8's bfloat16
Gumbel defect was unintelligible output at a perfectly normal pitch.

Normalisation is deliberately blunt (case-folded, punctuation stripped, digits left alone)
because the point is to compare renders *against each other* on identical text, not to
report a publishable WER. Note that CosyVoice's reference frontend spells numbers out and
this port does not, so any digit in the input is a source of difference between the two
that is not a defect in either.

Usage (from the CosyVoice venv, which has openai-whisper):
    .venv/bin/python /path/to/oracle-cosy/wer.py --text-file senior.txt a.wav b.wav
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def normalize(s: str) -> list[str]:
    s = s.lower().replace("’", "'").replace("—", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    import whisper

    reference = normalize(Path(args.text_file).read_text())
    model = whisper.load_model(args.model)
    print(f"reference: {len(reference)} words, whisper model {args.model}\n")
    print(f"{'file':<30} {'words':>7} {'errors':>7} {'WER':>7}")
    print("-" * 56)
    for f in args.files:
        hyp = normalize(model.transcribe(f, language="en", fp16=False)["text"])
        errs = edit_distance(reference, hyp)
        print(f"{Path(f).name:<30} {len(hyp):>7} {errs:>7} {errs / len(reference):>7.3f}")


if __name__ == "__main__":
    main()
