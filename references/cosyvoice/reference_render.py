"""Render text through the PyTorch CosyVoice3 service, as the control for the Rust port.

Without this there is no way to tell a *port* defect from a *model* characteristic. The
Rust render of the long passage came out at median F0 166.7 Hz against the reference
clip's 179.8; whether -13.1 Hz is the port or simply what CosyVoice does with this voice
is only answerable by running the same text through the same weights in Python.

Segmentation deliberately matches the Rust engine's (`tts_core::text::segment`) rather
than using the reference frontend's own splitter, so the two renders differ in
implementation and not in how the text was cut up.

Usage:
    cd /path/to/CosyVoice
    PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
        /path/to/references/cosyvoice/reference_render.py \
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
        --text-file /path/to/tts-rs/examples/senior.txt \
        --prompt-wav asset/default_voice.wav \
        --prompt-text-file asset/default_voice.txt \
        --out /path/to/tts-rs/examples/cosy_senior_torch.wav
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
import torchaudio


def segment(text: str, max_chars: int) -> list[str]:
    """The same rule as `tts_core::text::segment`, flattened across paragraphs."""
    out: list[str] = []
    for para in (l.strip() for l in text.splitlines()):
        if not para:
            continue
        para = " ".join(para.split())
        sentences = re.findall(r"[^.!?]*[.!?]|[^.!?]+$", para)
        buf = ""
        for s in (s.strip() for s in sentences):
            if not s:
                continue
            if buf and len(buf) + 1 + len(s) > max_chars:
                out.append(buf)
                buf = ""
            buf = f"{buf} {s}".strip() if buf else s
        if buf:
            out.append(buf)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.common import set_all_random_seed

    cosy = CosyVoice3(args.model_dir, fp16=False)
    # NOTE this fork's `_extract_*` helpers take a *path* and call `load_wav` themselves,
    # so handing them an already-loaded tensor fails inside torchaudio with a confusing
    # "Invalid file: tensor([[...]])".
    prompt = args.prompt_wav
    # The marker the LLM asserts on is added by the *service*, not by
    # `inference_zero_shot` or the frontend — the same trap `export_voice.py` documents.
    # Without it the first segment collapses to a couple of tokens and the vocoder fails
    # on a 3-frame mel, which is not an obvious symptom of a missing prefix.
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + Path(
        args.prompt_text_file
    ).read_text().strip()
    segments = segment(Path(args.text_file).read_text(), args.max_chars)
    print(f"{len(segments)} segments, {sum(len(s) for s in segments)} characters")

    gap = torch.zeros(1, int(0.09 * cosy.sample_rate))
    pieces: list[torch.Tensor] = []
    set_all_random_seed(args.seed)
    t0 = time.time()
    for i, seg in enumerate(segments):
        # `stream=False`, and `text_frontend=False` so the reference's normaliser is out
        # of the comparison — the Rust port does not implement it, and leaving it on
        # would confound "the port is slower" with "the port normalises less".
        for out in cosy.inference_zero_shot(
            seg, prompt_text, prompt, stream=False, text_frontend=False
        ):
            pieces.append(out["tts_speech"])
        print(f"  segment {i + 1}/{len(segments)}: {pieces[-1].shape[1] / cosy.sample_rate:.2f} s")
    wall = time.time() - t0

    stitched = []
    for i, p in enumerate(pieces):
        if i:
            stitched.append(gap)
        stitched.append(p)
    audio = torch.concat(stitched, dim=1)
    seconds = audio.shape[1] / cosy.sample_rate
    torchaudio.save(args.out, audio, cosy.sample_rate)
    print(
        f"\n{seconds:.2f} s of audio in {wall:.1f} s  ->  RTF {wall / seconds:.3f}\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
