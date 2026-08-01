"""Baseline the PyTorch codec decode against the Candle probe.

The Candle probe projects a codec-only RTF of ~0.36 on Metal. That number is only
interpretable next to what PyTorch gets for the *same* work on the *same* GPU. If
torch lands near 0.35, the codec is simply expensive and Candle is fine. If torch
lands near 0.10, candle's conv kernels are the bottleneck and the port strategy
has to change.

Usage:
    .venv/bin/python bench_codec.py --weights weights --frames 64
"""
from __future__ import annotations

import argparse
import time

import torch


def bench(codec, codes, device, dtype, iters: int, label: str) -> None:
    codec = codec.to(device=device, dtype=dtype)
    codes = codes.to(device)
    sync = (
        torch.mps.synchronize if device.type == "mps"
        else (torch.cuda.synchronize if device.type == "cuda" else lambda: None)
    )
    try:
        with torch.inference_mode():
            warm = codec.decode(codes)
        sync()
    except Exception as exc:  # noqa: BLE001 - reporting beats aborting the sweep
        print(f"{label:<28} FAILED: {type(exc).__name__}: {exc}")
        return

    samples = warm.shape[-1]
    audio_s = samples / codec.sample_rate

    start = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode():
            codec.decode(codes)
        sync()
    elapsed = (time.perf_counter() - start) / iters

    print(
        f"{label:<28} {elapsed * 1000:8.1f} ms   {audio_s:5.2f} s audio   "
        f"RTF {elapsed / audio_s:6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    from transformers import AutoModel

    print("loading model (cpu, fp32)")
    model = AutoModel.from_pretrained(
        args.weights, trust_remote_code=True, dtype=torch.float32
    ).eval()
    codec = model.load_codec(device=torch.device("cpu"))
    config = model.config

    generator = torch.Generator(device="cpu").manual_seed(20260731)
    codes = torch.empty((1, config.num_codebooks, args.frames), dtype=torch.long)
    codes[:, 0] = torch.randint(0, 4096, (1, args.frames), generator=generator)
    codes[:, 1:] = torch.randint(
        0, 1024, (1, config.num_codebooks - 1, args.frames), generator=generator
    )

    print(
        f"\ncodec.decode: {args.frames} frames -> "
        f"{args.frames * config.codec_frame_size} samples @ {config.codec_sample_rate} Hz\n"
    )

    bench(codec, codes, torch.device("cpu"), torch.float32, args.iters, "cpu / fp32")
    if torch.backends.mps.is_available():
        bench(codec, codes, torch.device("mps"), torch.float32, args.iters, "mps / fp32")
        bench(codec, codes, torch.device("mps"), torch.float16, args.iters, "mps / fp16")
        bench(codec, codes, torch.device("mps"), torch.bfloat16, args.iters, "mps / bf16")
    else:
        print("mps unavailable")

    print("\nCompare against PHASE_C_PROBE.md: Candle/Metal projected ~1072 ms")
    print(f"for 2.97 s of audio (RTF ~0.36) at 64 frames, f32.")


if __name__ == "__main__":
    main()
