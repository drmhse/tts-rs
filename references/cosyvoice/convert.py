"""Phase A for CosyVoice3: checkpoints -> safetensors, plus an inventory.

Same order that made the Audio8 port validate first try. Converting first and
*counting* first means the Rust side is written against known shapes rather than
guessed ones, and the inventory printed here is the spec the port is built from.

Runs under the CosyVoice venv (python 3.10, torch 2.3.1) because it only touches
`torch.load` — no CosyVoice model code is imported, so the pinned transformers is
irrelevant here.

Usage:
    /path/to/CosyVoice/.venv/bin/python convert.py \
        --checkpoints /path/to/pretrained_models/Fun-CosyVoice3-0.5B --out weights
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

# Anything that is a buffer rather than a parameter, or exists only for training.
DROP_SUFFIXES = ("freqs_cis", "causal_mask", "rand_noise")


def load_state(path: Path) -> dict:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    return state


def convert(name: str, src: Path, out_dir: Path) -> dict:
    state = load_state(src)
    kept, dropped = {}, []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            dropped.append((key, "not a tensor"))
            continue
        if key.endswith(DROP_SUFFIXES):
            dropped.append((key, "buffer, recomputed or shipped separately"))
            continue
        # safetensors cannot store shared storage; clone so aliased weights (tied
        # embeddings, weight_norm views) become independent entries.
        kept[key] = value.detach().to(torch.float32).clone().contiguous()

    params = sum(t.numel() for t in kept.values())
    target = out_dir / f"{name}.safetensors"
    save_file(kept, str(target))
    size_mb = target.stat().st_size / 1e6

    print(f"\n=== {name}: {len(kept)} tensors, {params/1e6:.1f} M params, {size_mb:.1f} MB")
    if dropped:
        print(f"    dropped {len(dropped)}: " + ", ".join(k for k, _ in dropped[:6]))

    # Collapse indices so the structure is readable at a glance. This listing is the
    # thing the Rust loader is written against.
    groups: dict[str, list[str]] = {}
    for key in sorted(kept):
        groups.setdefault(re.sub(r"\.\d+\.", ".N.", key), []).append(key)
    print(f"    {len(groups)} distinct shapes:")
    for pattern, keys in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:24]:
        shape = tuple(kept[keys[0]].shape)
        print(f"      {len(keys):4d}  {pattern:<62} {shape}")
    if len(groups) > 24:
        print(f"      ... and {len(groups) - 24} more patterns")

    return {
        "tensors": len(kept),
        "params": params,
        "megabytes": round(size_mb, 1),
        "patterns": {p: {"count": len(k), "shape": list(kept[k[0]].shape)} for p, k in groups.items()},
        "dropped": [k for k, _ in dropped],
    }


def export_tokenizer(src: Path, out_dir: Path) -> None:
    """Write a `tokenizer.json` the Rust `tokenizers` crate can load directly.

    The checkpoint ships `vocab.json` + `merges.txt` + `tokenizer_config.json` but no
    consolidated `tokenizer.json`, so the Rust side has nothing to load.

    It is not enough to serialise `AutoTokenizer.from_pretrained(dir)`: the checkpoint's
    `added_tokens_decoder` lists only three specials, and `CosyVoice3Tokenizer.__init__`
    registers **~250 more at construction** — `<|endofprompt|>`, the paralinguistic tags
    (`[breath]`, `[laughter]`, ...) and a full ARPAbet/pinyin phoneme set. Without them
    `<|endofprompt|>` tokenizes as nine literal-text pieces instead of id 151646, and the
    LLM's assertion fires. Measured, which is why the check below is an assert and not a
    comment.

    Unlike the weight conversion above, this needs the `cosyvoice` package importable.
    """
    tok_dir = src / "CosyVoice-BlankEN"
    if not (tok_dir / "vocab.json").is_file():
        print(f"[skip] no tokenizer at {tok_dir}")
        return
    try:
        from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
    except ImportError:
        # The common path: converting under a plain torch with no upstream repo. The
        # tokenizer is fetched prebuilt by scripts/fetch-assets.sh, so this is not fatal —
        # but say which file is expected, because a missing one fails much later as
        # `<|endofprompt|>` tokenizing to nine pieces and an assertion inside the LLM.
        print(
            f"[skip] tokenizer: the `cosyvoice` package is not importable, so "
            f"{out_dir/'tokenizer.json'} was not rebuilt.\n"
            f"       That file is fetched prebuilt by scripts/fetch-assets.sh; this is only a\n"
            f"       problem if it is absent. To build it here instead, run this script from\n"
            f"       the upstream CosyVoice repo with PYTHONPATH=.:third_party/Matcha-TTS."
        )
        return

    wrapper = CosyVoice3Tokenizer(str(tok_dir))
    hf = wrapper.tokenizer
    target = out_dir / "tokenizer.json"
    hf.backend_tokenizer.save(str(target))

    # The token whose absence is a silent failure downstream.
    ids = wrapper.encode("hello<|endofprompt|>world")
    assert 151646 in ids, f"<|endofprompt|> did not survive serialisation: {ids}"
    n_special = len(wrapper.special_tokens["additional_special_tokens"])
    print(
        f"\n=== tokenizer: wrote {target.name} "
        f"({target.stat().st_size / 1e6:.1f} MB, {n_special} additional special tokens), "
        f"<|endofprompt|> -> 151646 confirmed"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    src = Path(args.checkpoints)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inventory = {}
    for name in ("llm", "flow", "hift"):
        path = src / f"{name}.pt"
        if not path.is_file():
            print(f"[skip] {path} not found")
            continue
        inventory[name] = convert(name, path, out)

    export_tokenizer(src, out)

    total = sum(v["params"] for v in inventory.values())
    print(f"\ntotal {total/1e6:.1f} M params across {len(inventory)} components")
    (out / "inventory.json").write_text(json.dumps(inventory, indent=1) + "\n")
    print(f"wrote {out}/inventory.json")


if __name__ == "__main__":
    main()
