"""Export the Audio8 codec decode path to ONNX.

Two patches are required before the graph will trace, and both are numerically
exact rather than approximations:

1. `_arktts_snake` is `@torch.jit.script`ed. Replaced with the identical plain
   Python function so the tracer sees through it.

2. `_rope` builds its table with `torch.polar(...).real/.imag`. ONNX has no
   complex tensor type, so this cannot export. `polar(1, phase)` is exactly
   `(cos(phase), sin(phase))`, so the replacement is bit-identical.
   The original then casts the table to bfloat16 and `_apply_rope` multiplies it
   against fp32 activations (promoting back to fp32). We reproduce that by
   rounding through bfloat16 and casting back to fp32 -- same rounding, but no
   bf16 tensors in the exported graph, which CoreML would handle poorly.

Both patches are verified against the Phase A fixture before export, so a silent
numerical change cannot slip through.

Usage:
    .venv/bin/python export_codec.py --weights weights --out ../fixtures/codec_decode.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def patch_module(codec) -> object:
    """Patch the dynamically-loaded remote-code module in place. Returns it."""
    mod = sys.modules[type(codec).__module__]

    def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)

    def rope(length: int, head_dim: int, base: float, device=None) -> torch.Tensor:
        frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, device=device).float()[::2] / head_dim)
        )
        phases = torch.outer(torch.arange(length, device=device).float(), frequencies)
        values = torch.stack((phases.cos(), phases.sin()), dim=-1)
        # bf16 round-trip: identical rounding to the reference, fp32 storage.
        return values.to(torch.bfloat16).to(torch.float32)

    # 3. ArkttsCodecWindowTransformer.forward builds its window mask with an
    #    in-place `mask &= ...`, which the legacy exporter rejects
    #    (aten::__iand_ has no ONNX lowering). Identical logic, non-in-place.
    def window_forward(self, x, x_lens=None):
        del x_lens
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = x.shape[1]
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        keep = column <= row
        if self.window_size is not None:
            keep = keep & (column >= (row - self.window_size + 1).clamp_min(0))
        mask = keep[None, None]
        rope_values = mod._rope(length, self.head_dim, self.rope_base, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x

    # 4. ArkttsCausalConv1d.forward computes its right-hand padding via
    #    `_extra_padding`, which calls math.ceil on x.shape[-1]. Under tracing that
    #    collapses to a constant computed for the *export* length, which is what
    #    locked the first export to 64 frames.
    #
    #    Every ArkttsCausalConv1d in the decode path has stride 1 (the strided
    #    k2/s2 convs are all in the encoder-side `downsample`), and for stride 1:
    #        frames = length - k + (k-1) + 1 = length
    #        ideal  = (length-1)*1 + k - (k-1) = length
    #        right  = ideal - length = 0
    #    So the entire computation is provably zero here and can be dropped,
    #    leaving a plain left-pad and a genuinely dynamic length.
    def causal_conv_forward(self, x):
        if self.stride == 1:
            return self.conv(F.pad(x, (self.padding, 0))).contiguous()
        right = mod._extra_padding(x, self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, right))).contiguous()

    mod._arktts_snake = snake
    mod._rope = rope
    mod.ArkttsCodecWindowTransformer.forward = window_forward
    mod.ArkttsCausalConv1d.forward = causal_conv_forward
    return mod


class DecodeWrapper(nn.Module):
    """codes [B,10,T] int64 -> waveform [B,1,T*2048] float32.

    Mirrors ArkttsCodec.decode == decoder(quantizer.decode(codes)), with the
    per-codebook clamps written functionally instead of via in-place clamp_ on a
    cloned tensor (in-place mutation of a traced input is a trap).
    """

    def __init__(self, codec):
        super().__init__()
        self.quantizer = codec.quantizer
        self.decoder = codec.decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        q = self.quantizer
        semantic_codes = codes[:, :1].clamp(0, q.semantic_quantizer.codebook_size - 1)
        residual_codes = codes[:, 1:].clamp(0, q.quantizer.codebook_size - 1)
        semantic = q.semantic_quantizer.from_codes(semantic_codes)
        residual = q.quantizer.from_codes(residual_codes)
        latent = q.upsample(q.post_module(semantic + residual))
        return self.decoder(latent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--out", default="../fixtures/codec_decode.onnx")
    parser.add_argument("--frames", type=int, default=64, help="frames in the export sample")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static", action="store_true",
                        help="export a fixed-shape graph (no dynamic frame axis). CoreML "
                             "handles dynamic dims poorly; this tests whether that is the "
                             "cause. A static export implies length bucketing at serve time.")
    args = parser.parse_args()

    from transformers import AutoModel

    print("loading codec (cpu, fp32)")
    model = AutoModel.from_pretrained(
        args.weights, trust_remote_code=True, dtype=torch.float32
    ).eval()
    codec = model.load_codec(device=torch.device("cpu"))

    # ---------------------------------------------- verify patches are exact
    fixture = load_file("../fixtures/oracle.safetensors")
    syn_codes = fixture["codec_syn.codes"]
    with torch.inference_mode():
        before = codec.decode(syn_codes)
    patch_module(codec)
    with torch.inference_mode():
        after = codec.decode(syn_codes)

    ref = fixture["codec_syn.wav"]
    d_ref = (before - ref).abs().max().item()
    d_patch = (after - before).abs().max().item()
    print(f"unpatched vs Phase A fixture : max abs diff {d_ref:.3e}")
    print(f"patched   vs unpatched       : max abs diff {d_patch:.3e}")
    if d_patch > 1e-5:
        raise SystemExit(f"patches changed numerics by {d_patch:.3e} -- refusing to export")
    print("patches verified numerically exact\n")

    # ---------------------------------------------------------------- export
    wrapper = DecodeWrapper(codec).eval()
    sample = torch.randint(0, 1024, (1, 10, args.frames), dtype=torch.int64)
    sample[:, 0] = torch.randint(0, 4096, (1, args.frames))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"exporting {args.frames} frames, opset {args.opset}, dynamic frame axis")
    exported = False
    try:
        torch.onnx.export(
            wrapper,
            (sample,),
            str(out_path),
            input_names=["codes"],
            output_names=["wav"],
            dynamic_axes=None if args.static else {"codes": {2: "frames"}, "wav": {2: "samples"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        exported = True
        print("exported with the legacy TorchScript exporter")
    except Exception as exc:  # noqa: BLE001
        print(f"legacy exporter failed: {type(exc).__name__}: {exc}")
        print("retrying with dynamo=True")

    if not exported:
        frames = torch.export.Dim("frames", min=2, max=2048)
        onnx_program = torch.onnx.export(
            wrapper,
            (sample,),
            dynamic_shapes={"codes": {2: frames}},
            opset_version=args.opset,
            dynamo=True,
        )
        onnx_program.optimize()
        onnx_program.save(str(out_path))
        print("exported with the dynamo exporter")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB)")

    # ------------------------------------------------- dynamic-length gate
    # The first export silently baked the trace length into a ConvNeXt residual
    # add and worked ONLY at 64 frames. Never ship that again: run the exported
    # graph at several lengths and fail loudly if any of them break.
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    if args.static:
        print("\nstatic export: skipping the dynamic-length gate "
              f"(this graph only accepts {args.frames} frames)")
        return
    print("\ndynamic-length gate:")
    broken = []
    for frames in (4, 8, 24, 64, 100, 137):
        codes = torch.zeros((1, 10, frames), dtype=torch.int64).numpy()
        try:
            got = session.run(["wav"], {"codes": codes})[0]
            want = frames * 2048
            status = "ok" if got.shape[-1] == want else f"WRONG LEN want {want}"
            if got.shape[-1] != want:
                broken.append(frames)
            print(f"  frames={frames:<5} -> {got.shape}  {status}")
        except Exception as exc:  # noqa: BLE001
            broken.append(frames)
            print(f"  frames={frames:<5} -> FAILED: {str(exc)[-80:]}")
    if broken:
        raise SystemExit(f"\nexport is length-locked; broken at {broken}")
    print("  all lengths ok -- the frame axis is genuinely dynamic")

    # ------------------------------------------------------ graph composition
    import onnx

    graph = onnx.load(str(out_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in graph.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    print(f"\n{sum(counts.values())} nodes, {len(counts)} distinct op types:")
    for op, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {op:<24} {n}")


if __name__ == "__main__":
    main()
