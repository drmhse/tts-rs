"""Benchmark the exported codec decoder under ONNX Runtime, CPU vs CoreML.

Answers two questions the Candle probe could not:
  1. Is ORT faster than torch/mps (403.6 ms fp16) or candle/metal (990 ms f32)?
  2. Does CoreML actually take the graph, or does it partition around Snake's Sin
     and thrash device copies between every conv?

Correctness is checked at two *different* frame counts, because the export traced
Python `if difference > 0` branches on shapes -- if any of those baked in a
length-specific decision, a second length exposes it.

Usage:
    .venv/bin/python bench_onnx.py --model ../fixtures/codec_decode.onnx
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file


def make_session(path: str, provider: str, dump: str | None = None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if dump:
        options.optimized_model_filepath = dump
    providers: list = [provider] if provider == "CPUExecutionProvider" else [
        (provider, {}), "CPUExecutionProvider"
    ]
    return ort.InferenceSession(path, sess_options=options, providers=providers)


def verify(session, fixture, codes_key: str, wav_key: str) -> None:
    codes = fixture[codes_key].numpy().astype(np.int64)
    if codes.ndim == 2:
        codes = codes[None]
    expected = fixture[wav_key].numpy()
    got = session.run(["wav"], {"codes": codes})[0]
    if got.shape != expected.shape:
        print(f"  {codes_key:<16} SHAPE MISMATCH got {got.shape} want {expected.shape}")
        return
    diff = np.abs(got - expected)
    rms = float(np.sqrt((expected ** 2).mean()))
    print(
        f"  {codes_key:<16} frames={codes.shape[2]:<4} max|diff| {diff.max():.3e}  "
        f"mean|diff| {diff.mean():.3e}  (signal rms {rms:.3e})"
    )


def bench(session, frames: int, iters: int, label: str) -> None:
    rng = np.random.default_rng(20260731)
    codes = np.empty((1, 10, frames), dtype=np.int64)
    codes[:, 0] = rng.integers(0, 4096, (1, frames))
    codes[:, 1:] = rng.integers(0, 1024, (1, 9, frames))

    try:
        warm = session.run(["wav"], {"codes": codes})[0]
    except Exception as exc:  # noqa: BLE001
        print(f"{label:<34} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
        return
    audio_s = warm.shape[-1] / 44100.0

    start = time.perf_counter()
    for _ in range(iters):
        session.run(["wav"], {"codes": codes})
    elapsed = (time.perf_counter() - start) / iters
    print(
        f"{label:<34} {elapsed * 1000:8.1f} ms   {audio_s:5.2f} s audio   "
        f"RTF {elapsed / audio_s:6.3f}"
    )


def partition_report(dump_path: str) -> None:
    """Count how many nodes CoreML actually claimed in the optimized graph."""
    import onnx

    graph = onnx.load(dump_path, load_external_data=False)
    coreml, other = 0, {}
    for node in graph.graph.node:
        if "CoreML" in node.op_type or "CoreML" in node.domain:
            coreml += 1
        else:
            other[node.op_type] = other.get(node.op_type, 0) + 1
    total_other = sum(other.values())
    print(f"\n  fused CoreML subgraph nodes : {coreml}")
    print(f"  nodes left outside CoreML   : {total_other}")
    if coreml == 0:
        print("  -> CoreML took NOTHING; everything ran on CPU")
    if total_other:
        top = sorted(other.items(), key=lambda kv: -kv[1])[:12]
        print("  most common non-CoreML ops  : "
              + ", ".join(f"{op}x{n}" for op, n in top))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../fixtures/codec_decode.onnx")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    fixture = load_file("../fixtures/audio8/oracle.safetensors")
    print(f"ort {ort.__version__}   providers {ort.get_available_providers()}\n")

    print("=== CPUExecutionProvider ===")
    t0 = time.perf_counter()
    cpu = make_session(args.model, "CPUExecutionProvider")
    print(f"session init {time.perf_counter() - t0:.1f}s")
    print("correctness vs Phase A fixture (two different lengths):")
    verify(cpu, fixture, "codec_syn.codes", "codec_syn.wav")
    verify(cpu, fixture, "e2e.codes", "e2e.wav")
    bench(cpu, args.frames, args.iters, "ort / cpu")
    del cpu

    print("\nCoreML configs are run in separate processes by run_coreml.sh --")
    print("CoreML can hard-abort inside MPSGraph, which would kill this process.")

    print("\n--- reference numbers, same 64 frames / 2.97 s ---")
    print("  torch / mps fp16      403.6 ms   RTF 0.136   <- current best")
    print("  torch / mps f32       568.9 ms   RTF 0.191")
    print("  candle / metal f16    888.3 ms   RTF 0.298")
    print("  candle / metal f32    990.0 ms   RTF 0.332")
    print("  torch / cpu f32      1199.0 ms   RTF 0.403")


if __name__ == "__main__":
    main()
