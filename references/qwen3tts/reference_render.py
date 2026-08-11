"""Render text with the PyTorch reference, for A/B against the Rust engine.

The fixture gate proves each stage agrees at one position. It cannot tell you whether a
*whole utterance* comes out the same length, because sampling diverges after the first draw.
This is the control for that: same text, same clip, same sampler settings.

Usage:
    references/qwen3tts/.venv/bin/python references/qwen3tts/reference_render.py \\
        --model references/qwen3tts/weights \\
        --audio examples/cosy_short.wav --ref-text "..." \\
        --text "Hello from Rust." --out /tmp/ref.wav
"""
from __future__ import annotations

import argparse
import time

import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--xvec-only", action="store_true")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)

    t0 = time.time()
    wavs, sr = tts.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.xvec_only,
        max_new_tokens=2048,
        do_sample=not args.greedy,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=not args.greedy,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )
    wall = time.time() - t0
    wav = wavs[0]
    seconds = len(wav) / sr
    sf.write(args.out, wav, sr)
    print(f"wrote {args.out}")
    print(f"  {seconds:.2f} s of audio at {sr} Hz ({seconds * 12.5:.0f} frames)")
    print(f"  {wall:.1f} s wall on CPU  ->  RTF {wall / seconds:.2f}")


if __name__ == "__main__":
    main()
