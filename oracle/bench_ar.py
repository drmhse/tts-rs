"""Baseline the PyTorch DualAR loop against the Candle `arloop` probe.

`arloop` measured Candle at RTF 3.87 for the AR loop — 11.5 s for 2.97 s of audio,
while doing only a few GFLOP. That is dispatch overhead, not compute. But the
number is meaningless without knowing what torch/MPS does with the *same* loop on
the *same* GPU, because torch dispatches per-op too.

This deliberately does not call `generate()`: sampling would stop on EOS at an
unpredictable step, and the RAS/top-k machinery is not what we are measuring. It
drives `_slow_step` and `_generate_codebooks` directly for a fixed step count, so
the work matches `arloop` step for step, and it reports the slow/fast split.

Protocol matches BENCHMARK_VALIDITY.md: interleaved variants, median of >=5,
canary before and after.

Usage:
    .venv/bin/python bench_ar.py --weights weights --prompt 64 --frames 64
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModel

NUM_CODEBOOKS = 10
CANARY_SHAPE = (1, 96, 131072)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def canary(device, dtype):
    """Same fixed workload as the Rust harness: 96ch @ 131072, k7, dilation 9."""
    x = torch.randn(*CANARY_SHAPE, device=device, dtype=dtype)
    w = torch.randn(96, 96, 7, device=device, dtype=dtype) * 0.02
    times = []
    for i in range(4):
        t = time.perf_counter()
        with torch.inference_mode():
            torch.nn.functional.conv1d(torch.nn.functional.pad(x, (54, 0)), w, dilation=9)
        sync(device)
        if i:
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def make_prompt(model, width, device):
    """A [B, num_codebooks+1, T] prompt of the right shape. Row 0 holds semantic
    ids so `_embed` takes its codebook path, which is the expensive one."""
    cfg = model.config
    ids = torch.full((1, NUM_CODEBOOKS + 1, width), cfg.semantic_begin_id, device=device, dtype=torch.long)
    ids[:, 1:] = torch.randint(0, cfg.codebook_size, (1, NUM_CODEBOOKS, width), device=device)
    return ids


def build_step(model, device, prompt_width, frames, with_fast):
    """Returns a zero-arg callable running one full prefill + `frames` decode steps."""
    cfg = model.config
    prompt = make_prompt(model, prompt_width, device)
    mask = torch.ones((1, prompt_width), dtype=torch.long, device=device)
    # `_processed_scores` calls the processor list, so it must be a real
    # LogitsProcessorList and not a bare [].
    empty_processors = model._as_processor_list(None)

    def run():
        with torch.inference_mode():
            model._setup_generation_caches(1, cfg.max_seq_len, next(model.parameters()).dtype)
            cache_position = torch.arange(prompt_width, device=device)
            position_ids = mask.cumsum(-1).sub(1).clamp_min(0)
            logits, slow_hidden = model._slow_step(prompt, cache_position, position_ids, mask)
            step_mask = torch.ones((1, cfg.max_seq_len), dtype=torch.long, device=device)
            for step in range(frames):
                pos = prompt_width + step
                # One semantic token per frame, fed back as the next input row 0.
                nxt = torch.full((1, NUM_CODEBOOKS + 1, 1), cfg.semantic_begin_id, device=device, dtype=torch.long)
                cache_position = torch.tensor([pos], device=device)
                logits, slow_hidden = model._slow_step(
                    nxt, cache_position, cache_position[None], step_mask
                )
                if with_fast:
                    semantic = torch.full((1,), cfg.semantic_begin_id + 5, device=device, dtype=torch.long)
                    model._generate_codebooks(
                        slow_hidden, semantic, empty_processors, 50, 0.9, 0.7, do_sample=False
                    )
        return logits

    return run


def ab(device, label, variants, samples):
    """Interleaved A/B: every variant once per round, `samples` rounds."""
    for _, f in variants:
        f()
    sync(device)
    times = {name: [] for name, _ in variants}
    for _ in range(samples):
        for name, f in variants:
            t = time.perf_counter()
            f()
            sync(device)
            times[name].append((time.perf_counter() - t) * 1000)
    print(f"\n{label}  (n={samples}, interleaved)")
    print(f"{'variant':<34} {'median ms':>10} {'min':>10} {'max':>10} {'spread':>9} {'vs base':>10}")
    print("-" * 88)
    base = None
    out = {}
    for name, _ in variants:
        s = times[name]
        med = statistics.median(s)
        base = base if base is not None else med
        out[name] = med
        print(
            f"{name:<34} {med:>10.3f} {min(s):>10.3f} {max(s):>10.3f} "
            f"{(max(s)/min(s)-1)*100:>8.1f}% {base/med:>9.2f}x"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    c0 = canary(device, torch.float32)
    print(f"canary (96ch@131072 k7 d9): {c0:.2f} ms  [~60 = cool, ~120 = throttled]")

    model = AutoModel.from_pretrained(
        args.weights, trust_remote_code=True, dtype=dtype
    ).to(device).eval()

    frame_rate = model.config.codec_sample_rate / model.config.codec_frame_size
    audio_ms = args.frames / frame_rate * 1000
    print(
        f"torch {args.device}/{args.dtype}: {args.prompt} prompt + {args.frames} frames "
        f"= {audio_ms/1000:.2f} s of audio at {frame_rate:.2f} Hz"
    )

    slow_only = build_step(model, device, args.prompt, args.frames, with_fast=False)
    full = build_step(model, device, args.prompt, args.frames, with_fast=True)

    res = ab(
        device,
        "torch DualAR loop",
        [("slow AR only", slow_only), ("slow + fast AR", full)],
        args.samples,
    )
    slow = res["slow AR only"]
    both = res["slow + fast AR"]
    print(
        f"\nslow AR: {slow:.1f} ms  fast AR: {both - slow:.1f} ms  total: {both:.1f} ms"
        f"  ->  AR-only RTF {both/audio_ms:.3f}"
    )

    c1 = canary(device, torch.float32)
    drift = c1 / c0
    print(
        f"\ncanary: {c0:.2f} ms at start, {c1:.2f} ms at end -> {drift:.2f}x drift "
        f"[{'stable' if drift < 1.15 else 'DRIFTED'}]"
    )


if __name__ == "__main__":
    main()
