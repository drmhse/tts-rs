"""Convert codec.pth (1.35 GB pickle) to safetensors, folding weight_norm.

Every conv in the codec is wrapped in weight_norm, so the stored parameters are a
magnitude `g` and a direction `v` with `weight = g * v / ||v||`. Folding that once
here means the Rust side loads plain weights via mmap and needs no runtime
reparametrisation at all.

Two key layouts appear in this checkpoint and both must be handled:
  * nn.utils.parametrizations.weight_norm -> `<prefix>.parametrizations.weight.original0` (g)
                                            `<prefix>.parametrizations.weight.original1` (v)
  * legacy nn.utils.weight_norm           -> `<prefix>.weight_g`, `<prefix>.weight_v`
(the quantizer in_proj/out_proj use the legacy form, the convs use the new one)

PyTorch's weight_norm defaults to dim=0, i.e. one magnitude per index of the first
weight dimension, with the norm taken over all remaining dims. That is true for
ConvTranspose1d too, where dim 0 is *in_channels* rather than out_channels.

Usage:
    .venv/bin/python convert_codec.py --weights weights --out weights/codec.safetensors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """weight = g * v / ||v||, norm over every dim except 0 (PyTorch dim=0 default)."""
    dims = tuple(range(1, v.ndim))
    norm = v.float().pow(2).sum(dim=dims, keepdim=True).sqrt()
    return (v.float() * (g.float() / norm)).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--out", default="weights/codec.safetensors")
    parser.add_argument("--keep-encoder", action="store_true",
                        help="keep encoder tensors (needed for voice cloning from raw audio)")
    args = parser.parse_args()

    src = Path(args.weights) / "codec.pth"
    state = torch.load(src, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    state = {k: v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
    print(f"loaded {len(state)} tensors from {src}")

    out: dict[str, torch.Tensor] = {}
    folded = 0
    pending: dict[str, dict[str, torch.Tensor]] = {}

    for key, value in state.items():
        if key.endswith(".parametrizations.weight.original0"):
            pending.setdefault(key[: -len(".parametrizations.weight.original0")], {})["g"] = value
        elif key.endswith(".parametrizations.weight.original1"):
            pending.setdefault(key[: -len(".parametrizations.weight.original1")], {})["v"] = value
        elif key.endswith(".weight_g"):
            pending.setdefault(key[: -len(".weight_g")], {})["g"] = value
        elif key.endswith(".weight_v"):
            pending.setdefault(key[: -len(".weight_v")], {})["v"] = value
        else:
            out[key] = value.float().contiguous()

    for prefix, parts in sorted(pending.items()):
        if set(parts) != {"g", "v"}:
            raise SystemExit(f"{prefix}: incomplete weight_norm pair, got {sorted(parts)}")
        out[f"{prefix}.weight"] = fold_weight_norm(parts["g"], parts["v"])
        folded += 1

    if not args.keep_encoder:
        before = len(out)
        out = {k: v for k, v in out.items() if not k.startswith("encoder.")}
        dropped = before - len(out)
        # The encoder is only needed to turn raw reference audio into codes. Phase C
        # decodes recorded codes, so it can be dropped -- but a shippable service
        # that accepts arbitrary reference wavs needs it back.
        print(f"dropped {dropped} encoder tensors (--keep-encoder to retain)")
    # pre_module and downsample are also encode-only; keep them, they are small and
    # dropping them selectively risks confusing the manifest.

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest))

    manifest = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(out.items())}
    dest.with_suffix(".json").write_text(json.dumps(manifest, indent=2))

    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"folded {folded} weight_norm pairs")
    print(f"wrote {len(out)} tensors, {total / 1e6:.1f} MB -> {dest}")


if __name__ == "__main__":
    main()
