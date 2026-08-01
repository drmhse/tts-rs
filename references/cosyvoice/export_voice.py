"""Turn a reference clip into a `cosyvoice` voice asset.

This is the step that keeps ~997 MB of ONNX out of the Rust binary. CosyVoice needs
four things from a reference clip, and none of them depend on the text being spoken:

  speaker_embedding  [1, 192]      from campplus.onnx        (28 MB)
  speech_tokens      [1, T]        from speech_tokenizer_v3  (969 MB, whisper-based)
  prompt_mel         [1, 2T, 80]   from the mel frontend
  prompt_text_tokens [1, L]        from the Qwen tokenizer

So they are computed once here and shipped as a small asset, exactly as
`references/audio8/export_voice.py` does for Audio8's codec encoder. Only the frontend is
constructed — the 3.4 GB of model weights are not loaded.

Runs under the CosyVoice venv, from the CosyVoice directory (its package must be
importable and the ONNX paths are relative to the checkpoint dir).

Usage:
    cd /path/to/CosyVoice
    .venv/bin/python /path/to/references/cosyvoice/export_voice.py \
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
        --audio asset/default_voice.wav --text asset/default_voice.txt \
        --name cosy-default-cosyvoice --out /path/to/tts-rs/voices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import save_file

ENGINE = "cosyvoice"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True, help="transcript, or a path to it")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    model_dir = Path(args.model_dir)
    with open(model_dir / "cosyvoice3.yaml") as f:
        configs = load_hyperpyyaml(
            f, overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")}
        )
    sample_rate = configs["sample_rate"]

    # Frontend only: tokenizer, mel extractor and the two ONNX sessions. No llm/flow/hift.
    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        str(model_dir / "campplus.onnx"),
        str(model_dir / "speech_tokenizer_v3.onnx"),
        "",
        configs["allowed_special"],
    )

    text_path = Path(args.text)
    text = text_path.read_text(encoding="utf-8").strip() if text_path.is_file() else args.text
    text = " ".join(text.split())
    if not text:
        raise SystemExit("a transcript is required: the LLM prompt interleaves it with the tokens")

    # Every `_extract_*` helper takes a *path* and calls `load_wav` itself, resampling
    # to whatever that stage needs (16 kHz for the token and embedding extractors,
    # 24 kHz for the mel). Passing a tensor fails inside soundfile.
    info = torchaudio.info(args.audio, backend="soundfile")
    seconds = info.num_frames / info.sample_rate
    if seconds > 30:
        raise SystemExit(
            f"the speech tokenizer refuses audio longer than 30 s; this clip is {seconds:.1f} s"
        )

    # CosyVoice3's LLM asserts that token 151646 `<|endofprompt|>` appears in the
    # concatenated prompt_text + text, and the service supplies it by prepending an
    # assistant preamble to the transcript when the caller has not. Reproduce that here
    # or the LLM refuses the prompt outright -- and note this is a *silent* requirement
    # for anyone building an asset by hand.
    prompt_text_str = text
    if "<|endofprompt|>" not in prompt_text_str:
        prompt_text_str = "You are a helpful assistant.<|endofprompt|>" + prompt_text_str

    model_input = frontend.frontend_zero_shot(
        "placeholder text", prompt_text_str, args.audio, sample_rate, ""
    )

    # `llm_embedding` and `flow_embedding` are the same tensor, and the llm/flow prompt
    # tokens are too — store one copy each and let the engine alias them.
    assert torch.equal(model_input["llm_embedding"], model_input["flow_embedding"])
    assert torch.equal(
        model_input["llm_prompt_speech_token"], model_input["flow_prompt_speech_token"]
    )

    tensors = {
        "speaker_embedding": model_input["llm_embedding"].to(torch.float32).cpu().contiguous(),
        "speech_tokens": model_input["llm_prompt_speech_token"].to(torch.int32).cpu().contiguous(),
        "prompt_mel": model_input["prompt_speech_feat"].to(torch.float32).cpu().contiguous(),
        "prompt_text_tokens": model_input["prompt_text"].to(torch.int32).cpu().contiguous(),
    }

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "voice.safetensors"))
    (out / "voice.json").write_text(
        json.dumps(
            {
                "engine": ENGINE,
                "name": args.name,
                "text": text,
                "prompt_text": prompt_text_str,
                "seconds": round(seconds, 3),
                "notes": (
                    f"campplus + speech_tokenizer_v3 + mel from {Path(args.audio).name}; "
                    "exported by references/cosyvoice/export_voice.py so neither ONNX model is "
                    "needed at runtime. prompt_text_tokens encode `prompt_text`, which "
                    "carries the <|endofprompt|> marker the LLM asserts on; `text` is the "
                    "plain transcript"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for k, v in tensors.items():
        print(f"  {k:<20} {tuple(v.shape)} {v.dtype}")
    print(f"\n{args.name}: {seconds:.2f} s of reference audio")
    print(f"wrote {out}/voice.json and voice.safetensors")


if __name__ == "__main__":
    main()
