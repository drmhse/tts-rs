"""Bisect what broke long-form generation.

`synthesize.py` produced 168 s of noise with every segment running to the token cap
(no EOS). Three things changed at once relative to the known-good path in
`quality_ar.py` (batch 1, greedy, unpatched, 96 frames): sampling was turned on,
the narrow-KV patch was applied, and generation was batched. This isolates them.
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel, AutoProcessor

from synthesize import patch_f32_sampling, patch_narrow_kv, write_wav

ONE = (
    "Some candidates assume the title is earned by time. They have been employed "
    "for eight or ten years, have shipped many features, and have held the title "
    "before."
)
TWO = "Interviews do not resolve that confusion by counting years. They look for evidence."
THREE = "The word senior creates false confidence and false anxiety."
FOUR = "A senior software engineer is trusted with problem areas, not merely tasks."


def run(model, processor, device, texts, do_sample, max_new, seed=1234):
    batch = processor(text=texts, return_tensors="pt")
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            prefix_input_ids=batch["prefix_input_ids"],
            prefix_attention_mask=batch["prefix_attention_mask"],
            suffix_input_ids=batch["suffix_input_ids"],
            suffix_attention_mask=batch["suffix_attention_mask"],
            do_sample=do_sample,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=max_new,
            generator=gen if do_sample else None,
            return_dict_in_generate=True,
        )
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--out", default="../examples/diag")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.weights, trust_remote_code=True)

    def fresh(patched, f32_sample=False, force_dtype=None):
        dt = force_dtype or dtype
        m = AutoModel.from_pretrained(args.weights, trust_remote_code=True, dtype=dt)
        m = m.to(device).eval()
        if patched:
            patch_narrow_kv(m)
        if f32_sample:
            patch_f32_sampling(m)
        return m

    from pathlib import Path

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # (name, narrow_kv, do_sample, texts, f32_sampling, dtype override)
    cases = [
        ("b1_sample_f32model", False, True, [ONE], False, torch.float32),
        ("b1_sample_f32draw", False, True, [ONE], True, None),
        ("b1_sample_f32draw_narrow", True, True, [ONE], True, None),
        ("b4_sample_f32draw_narrow", True, True, [ONE, TWO, THREE, FOUR], True, None),
    ]

    print(f"{'case':<20} {'frames (per seq)':>26} {'cap?':>6} {'wall s':>8}")
    print("-" * 66)
    for name, patched, sample, texts, f32s, forced in cases:
        model = fresh(patched, f32s, forced)
        codec = model.load_codec(device=device).to(dtype=torch.float32).eval()
        sr = model.config.codec_sample_rate
        out, wall = run(model, processor, device, texts, sample, args.max_new)
        lens = [int(x) for x in out.code_lengths]
        capped = any(x >= args.max_new for x in lens)
        print(f"{name:<20} {str(lens):>26} {str(capped):>6} {wall:>8.1f}")
        # Write the first sequence of each case so it can be listened to / scored.
        n = lens[0]
        if n:
            with torch.inference_mode():
                wav = codec.decode(out.codes[0:1, :, :n].to(device)).float().cpu()
            write_wav(outdir / f"{name}.wav", wav, sr)
        del model, codec
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
