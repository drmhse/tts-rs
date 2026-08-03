"""Word error rate against the input text, via Whisper.

The voice checks in `references/audio8/verify_voice.py` answer "whose voice is this"; they say nothing
about whether the words are right. A cloning path can produce the correct speaker saying
mush, and a sampler bug shows up here long before it shows up in F0 — Audio8's bfloat16
Gumbel defect was unintelligible output at a perfectly normal pitch.

Normalisation is deliberately blunt (case-folded, punctuation stripped, digits left alone)
because the point is to compare renders *against each other* on identical text, not to
report a publishable WER. Note that CosyVoice's reference frontend spells numbers out and
this port does not, so any digit in the input is a source of difference between the two
that is not a defect in either.

Usage (from the CosyVoice venv, which has openai-whisper):
    .venv/bin/python /path/to/references/cosyvoice/wer.py --text-file senior.txt a.wav b.wav
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


def load_backend(choice: str, model_name: str):
    """Return `(transcribe_fn, label)`.

    Two implementations of the same model, and the difference is only speed. Measured on a
    949 s chapter, CPU:

    | backend | time |
    |---|---|
    | openai-whisper | ~86 s |
    | faster-whisper, sequential | 72.5 s |
    | faster-whisper, batched | **45.6 s** |

    They also use far less CPU to do it: 34 s of user time against 120 s on the same pair of
    clips, because openai-whisper spreads over every core (915% CPU) for a *longer* wall time.

    **They are close but not identical.** On a 949 s chapter they produced 2528 against 2530
    words with the same tail, and on two 85 s clips they agreed exactly on one (3 errors) and
    differed by one error on the other (4 against 3, so WER 0.018 against 0.014). Treat
    figures from the two backends as comparable to about one error in 220 words, and use
    `--backend openai` to reproduce a recorded number exactly.

    Neither can use the GPU. CTranslate2 is CPU-only on macOS, and openai-whisper on MPS
    dies in `aten::empty.memory_format` on the `SparseMPS` backend, which torch's MPS
    implementation does not provide. `fp16` is therefore off: on CPU it is emulated and
    slower, and torch warns about it on every call.
    """
    if choice in ("auto", "faster"):
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            pipeline = BatchedInferencePipeline(model=model)

            def run(path: str) -> str:
                # beam_size=1 to match openai-whisper's default greedy search; a wider beam
                # made it *slower than* openai-whisper and changed nothing.
                segments, _ = pipeline.transcribe(
                    path, language="en", beam_size=1, batch_size=8
                )
                return "".join(s.text for s in segments)

            return run, "faster-whisper (batched, int8)"
        except ImportError:
            if choice == "faster":
                raise SystemExit(
                    "faster-whisper is not installed in this interpreter. It lives in "
                    "CosyVoice's .venv-align; or use --backend openai."
                )

    import whisper

    model = whisper.load_model(model_name)

    def run(path: str) -> str:
        return model.transcribe(path, language="en", fp16=False)["text"]

    return run, "openai-whisper"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--model", default="small.en")
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "faster", "openai"],
        help="auto prefers faster-whisper, which is ~1.9x quicker on a chapter",
    )
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    reference = normalize(Path(args.text_file).read_text())
    transcribe, backend = load_backend(args.backend, args.model)
    print(f"reference: {len(reference)} words, {backend} model {args.model}\n")
    print(f"{'file':<30} {'words':>7} {'errors':>7} {'WER':>7}")
    print("-" * 56)
    for f in args.files:
        hyp = normalize(transcribe(f))
        errs = edit_distance(reference, hyp)
        print(f"{Path(f).name:<30} {len(hyp):>7} {errs:>7} {errs / len(reference):>7.3f}")


if __name__ == "__main__":
    main()
