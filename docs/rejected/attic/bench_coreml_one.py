"""Benchmark ONE CoreML configuration, in its own process.

CoreML can abort the process outright (an assertion failure inside
MetalPerformanceShadersGraph, not a Python exception), so each configuration has
to be isolated or one bad config takes the whole sweep with it.

Usage:
    .venv/bin/python bench_coreml_one.py --format MLProgram --units ALL
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../fixtures/codec_decode.onnx")
    parser.add_argument("--format", default="MLProgram", choices=["MLProgram", "NeuralNetwork"])
    parser.add_argument("--units", default="ALL",
                        choices=["ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly"])
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # WARNING level: shows the GetCapability partition count without the flood of
    # per-partition "Writing CoreML Model" INFO lines.
    options.log_severity_level = 2
    provider_opts = {"ModelFormat": args.format, "MLComputeUnits": args.units}

    print(f"### {args.format} / {args.units}", flush=True)
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        args.model, sess_options=options,
        providers=[("CoreMLExecutionProvider", provider_opts), "CPUExecutionProvider"],
    )
    print(f"session init {time.perf_counter() - t0:.1f}s", flush=True)

    # A statically-exported graph only accepts its export length, so only run the
    # 8-frame fixture check when the frame axis is actually symbolic.
    frame_dim = session.get_inputs()[0].shape[2]
    if isinstance(frame_dim, str):
        fixture = load_file("../fixtures/oracle.safetensors")
        codes = fixture["codec_syn.codes"].numpy().astype(np.int64)
        expected = fixture["codec_syn.wav"].numpy()
        got = session.run(["wav"], {"codes": codes})[0]
        print(f"correctness (8 frames): max|diff| {np.abs(got - expected).max():.3e}", flush=True)
    else:
        print(f"static graph (frames fixed at {frame_dim}); skipping 8-frame check", flush=True)
        args.frames = int(frame_dim)

    rng = np.random.default_rng(20260731)
    bench_codes = np.empty((1, 10, args.frames), dtype=np.int64)
    bench_codes[:, 0] = rng.integers(0, 4096, (1, args.frames))
    bench_codes[:, 1:] = rng.integers(0, 1024, (1, 9, args.frames))

    warm = session.run(["wav"], {"codes": bench_codes})[0]
    audio_s = warm.shape[-1] / 44100.0
    start = time.perf_counter()
    for _ in range(args.iters):
        session.run(["wav"], {"codes": bench_codes})
    elapsed = (time.perf_counter() - start) / args.iters
    print(f"RESULT {args.format}/{args.units}: {elapsed * 1000:.1f} ms  "
          f"{audio_s:.2f} s audio  RTF {elapsed / audio_s:.3f}", flush=True)


if __name__ == "__main__":
    main()
