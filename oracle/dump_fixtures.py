"""Phase A: dump reference tensors from the HF implementation.

This is the ground truth the Rust port is validated against. Everything runs on
CPU in float32 by default -- not because that is how the model ships (bfloat16),
but because a port needs a *mathematical* reference. Validating against fp32 CPU
first makes "port bug" and "precision difference" distinguishable; dump bf16
separately to learn the tolerance band the real model operates in.

Usage:
    .venv/bin/python dump_fixtures.py --weights weights --out ../fixtures
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor

# Modules whose outputs we record. Slow-AR layers 0/1/11/23 rather than all 24 --
# a divergence at layer 1 that survives to 23 is the same bug, and the fixture
# file stays small enough to commit.
SLOW_TAPS = ["embeddings", "layers.0", "layers.1", "layers.11", "layers.23", "norm"]
FAST_TAPS = ["fast_project_in", "fast_layers.0", "fast_layers.3", "fast_norm", "fast_output"]
CODEC_TAPS = [
    "quantizer.semantic_quantizer",
    "quantizer.quantizer",
    "quantizer.post_module",
    "quantizer.upsample",
    "decoder.model.0",   # Conv1d(1024 -> 1536, k=7)
    "decoder.model.1",   # DecoderBlock stride 8
    "decoder.model.2",   # DecoderBlock stride 8
    "decoder.model.3",   # DecoderBlock stride 4
    "decoder.model.4",   # DecoderBlock stride 2
]

TEXT = "Welcome to Audio8 TTS."


class Tapper:
    """Records every invocation of the named modules, in call order."""

    def __init__(self, root: torch.nn.Module, names: list[str], prefix: str):
        self.calls: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.prefix = prefix
        self.handles = []
        lookup = dict(root.named_modules())
        for name in names:
            if name not in lookup:
                raise KeyError(f"no module {name!r}; have e.g. {list(lookup)[:8]}")
            self.handles.append(
                lookup[name].register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self.calls[name].append(tensor.detach().float().cpu().contiguous())
        return hook

    def collect(self) -> dict[str, torch.Tensor]:
        out = {}
        for name, tensors in self.calls.items():
            key = f"{self.prefix}.{name}"
            if len(tensors) == 1:
                out[key] = tensors[0]
            else:
                # Ragged call sequences (fast AR grows its input) stay separate.
                same = all(t.shape == tensors[0].shape for t in tensors)
                if same:
                    out[key] = torch.stack(tensors)
                else:
                    for i, t in enumerate(tensors):
                        out[f"{key}.call{i}"] = t
        return out

    def reset(self):
        self.calls.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--out", default="../fixtures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--frames", type=int, default=8, help="synthetic codec frames")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    print(f"loading model  device={device} dtype={dtype}")
    model = AutoModel.from_pretrained(
        args.weights, trust_remote_code=True, dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.weights, trust_remote_code=True)
    config = model.config

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {
        "text": TEXT,
        "device": str(device),
        "dtype": args.dtype,
        "revision": "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17",
    }

    # ---------------------------------------------------------------- prompt
    # Phase B gate: exact token ids out of the processor, no reference voice.
    batch = processor(text=[TEXT], return_tensors="pt")
    prefix = batch["prefix_input_ids"]
    tensors["prompt.prefix_input_ids"] = prefix.cpu().contiguous()
    tensors["prompt.prefix_attention_mask"] = batch["prefix_attention_mask"].cpu().contiguous()
    tensors["prompt.suffix_input_ids"] = batch["suffix_input_ids"].cpu().contiguous()
    print(f"prompt: {prefix.shape[1]} prefix tokens")

    prompt_ids, prompt_mask = model._prepare_prompt(
        prefix_input_ids=prefix,
        prefix_attention_mask=batch["prefix_attention_mask"],
        suffix_input_ids=batch["suffix_input_ids"],
        suffix_attention_mask=batch["suffix_attention_mask"],
    )
    tensors["prompt.input_ids"] = prompt_ids.cpu().contiguous()
    tensors["prompt.attention_mask"] = prompt_mask.cpu().contiguous()
    meta["prompt_width"] = int(prompt_ids.shape[-1])

    # ------------------------------------------------ Phase D: slow AR, forced
    # No sampling, no cache: a single full-width forward pass. Deterministic.
    slow_tap = Tapper(model, SLOW_TAPS, "slow")
    with torch.inference_mode():
        output = model(input_ids=prompt_ids, attention_mask=prompt_mask)
    tensors["slow.logits"] = output.logits.detach().float().cpu().contiguous()
    tensors["slow.hidden_states"] = output.hidden_states.detach().float().cpu().contiguous()
    tensors.update(slow_tap.collect())
    slow_tap.close()
    print(f"slow AR: logits {tuple(output.logits.shape)}")

    # ------------------------------------------------- Phase E: fast AR, greedy
    # Drive _generate_codebooks directly with the recorded slow hidden state so
    # the fixture does not depend on the sampler.
    from transformers.generation import LogitsProcessorList

    slow_hidden = model.norm(output.hidden_states[:, -1:])
    semantic = torch.tensor([config.semantic_begin_id + 123], device=device)
    tensors["fast.slow_hidden"] = slow_hidden.detach().float().cpu().contiguous()
    tensors["fast.semantic"] = semantic.cpu().contiguous()

    model._setup_generation_caches(1, config.max_seq_len, dtype)
    fast_tap = Tapper(model, FAST_TAPS, "fast")
    with torch.inference_mode():
        codebooks = model._generate_codebooks(
            slow_hidden, semantic, LogitsProcessorList(),
            top_k=50, top_p=0.9, temperature=0.7, do_sample=False,
        )
    tensors["fast.codebooks"] = codebooks.cpu().contiguous()
    tensors.update(fast_tap.collect())
    fast_tap.close()
    print(f"fast AR: codebooks {codebooks.tolist()}")

    # ------------------------------------------- Phase C: codec, synthetic codes
    # Seeded pseudo-random codes in the *valid* per-codebook ranges: row 0 is the
    # semantic codebook (4096 entries), rows 1..9 are residual (1024 entries).
    # Using 4096 for rows 1..9 would only exercise the clamp.
    codec = model.load_codec(device=device)
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    synthetic = torch.empty((1, config.num_codebooks, args.frames), dtype=torch.long)
    synthetic[:, 0] = torch.randint(0, 4096, (1, args.frames), generator=generator)
    synthetic[:, 1:] = torch.randint(
        0, 1024, (1, config.num_codebooks - 1, args.frames), generator=generator
    )
    tensors["codec_syn.codes"] = synthetic.contiguous()

    codec_tap = Tapper(codec, CODEC_TAPS, "codec_syn")
    with torch.inference_mode():
        wav_syn = codec.decode(synthetic.to(device))
    tensors["codec_syn.wav"] = wav_syn.detach().float().cpu().contiguous()
    tensors.update(codec_tap.collect())
    print(f"codec synthetic: {args.frames} frames -> {tuple(wav_syn.shape)}"
          f"  (expect {args.frames * config.codec_frame_size} samples)")

    # ------------------------------------ Phase F: end to end, greedy generation
    codec_tap.reset()
    with torch.inference_mode():
        codes = model.generate(
            prefix_input_ids=prefix,
            prefix_attention_mask=batch["prefix_attention_mask"],
            suffix_input_ids=batch["suffix_input_ids"],
            suffix_attention_mask=batch["suffix_attention_mask"],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    tensors["e2e.codes"] = codes.cpu().contiguous()
    print(f"e2e: greedy codes {tuple(codes.shape)}")

    if codes.shape[-1] > 0:
        with torch.inference_mode():
            wav_e2e = codec.decode(codes.to(device))
        tensors["e2e.wav"] = wav_e2e.detach().float().cpu().contiguous()
        tensors.update(codec_tap.collect())
        print(f"e2e: wav {tuple(wav_e2e.shape)}")
    codec_tap.close()

    # ------------------------------------------------------------------- write
    manifest = {
        "meta": meta,
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in sorted(tensors.items())},
    }
    suffix = "" if args.dtype == "float32" else f".{args.dtype}"
    # Some taps legitimately alias: fast_project_in is Identity when fast_dim ==
    # dim, and layers.23's output *is* hidden_states. Clone so safetensors will
    # write them as distinct entries -- the aliasing is itself worth asserting on
    # the Rust side, so we keep both keys.
    tensors = {k: v.clone() for k, v in tensors.items()}
    save_file(tensors, str(out_dir / f"oracle{suffix}.safetensors"))
    (out_dir / f"oracle{suffix}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(tensors)} tensors to {out_dir}/oracle{suffix}.safetensors")


if __name__ == "__main__":
    main()
