"""Did the reference clip actually change the voice, or just the words?

WhisperX answers "are the words right". It says nothing about *whose voice* said
them, and a cloning path can silently ignore its reference while still producing
clean speech. Two cheap speaker-correlated measures, both computed against the
reference clip:

  - **median F0**, via `torchaudio.functional.detect_pitch_frequency`, over voiced
    frames only. Pitch is the single most audible speaker cue.
  - **LTAS cosine similarity** — the long-term average log-magnitude spectrum,
    which captures vocal-tract colour and is largely independent of what was said.

Neither is a speaker-verification model, so neither proves identity. What they can
show is direction: if the cloned render moves toward the reference on both and the
unconditioned render does not, the reference is being used.

Usage:
    .venv/bin/python verify_voice.py --reference <ref.wav> a.wav b.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from torchaudio.functional import detect_pitch_frequency, resample

SR = 16000


def load(path: Path) -> torch.Tensor:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.mean(axis=1))
    if int(rate) != SR:
        x = resample(x, int(rate), SR)
    return x / x.abs().max().clamp_min(1e-6)


def median_f0(x: torch.Tensor) -> float:
    """Median over voiced frames. Unvoiced frames return junk, so gate on energy."""
    f0 = detect_pitch_frequency(x, SR, frame_time=0.02, freq_low=60, freq_high=400)
    frame = int(SR * 0.02)
    n = min(f0.numel(), x.numel() // frame)
    energy = x[: n * frame].reshape(n, frame).pow(2).mean(-1)
    voiced = energy > energy.median()
    sel = f0[:n][voiced]
    return float(sel.median()) if sel.numel() else float("nan")


def ltas(x: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(
        x, n_fft=1024, hop_length=256,
        window=torch.hann_window(1024), return_complex=True,
    ).abs()
    # Average in the log domain, over frames above the energy median, so silence
    # does not dominate the average.
    power = spec.pow(2).sum(0)
    keep = power > power.median()
    return torch.log(spec[:, keep] + 1e-8).mean(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    ref = load(Path(args.reference))
    ref_f0 = median_f0(ref)
    ref_ltas = ltas(ref)
    print(f"reference: {Path(args.reference).name}  median F0 {ref_f0:.1f} Hz\n")
    print(f"{'file':<26} {'median F0':>10} {'dF0 vs ref':>11} {'LTAS cos':>9}")
    print("-" * 60)
    for f in args.files:
        x = load(Path(f))
        f0 = median_f0(x)
        cos = float(
            torch.nn.functional.cosine_similarity(ltas(x), ref_ltas, dim=0)
        )
        print(f"{Path(f).name:<26} {f0:>10.1f} {f0 - ref_f0:>+11.1f} {cos:>9.4f}")


if __name__ == "__main__":
    main()
