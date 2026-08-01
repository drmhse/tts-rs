"""Turn a reference clip into an `audio8` voice asset.

This is the offline half of the split described in `crates/tts-core/src/voice.rs`:
converting audio into conditioning needs the codec **encoder**, which
`convert_codec.py` deliberately drops from the Rust weights (126 tensors). Doing it
once here keeps the encoder out of the binary permanently, and adding a voice becomes
a Python step rather than a runtime dependency.

Writes a directory the Rust side reads with `tts_core::Voice::load`:

    voices/<name>/voice.json          engine, name, transcript, duration
    voices/<name>/voice.safetensors    reference_codes [10, T]

Usage:
    .venv/bin/python export_voice.py \
        --audio ../../CosyVoice/asset/default_voice.wav \
        --text  ../../CosyVoice/asset/default_voice.txt \
        --name  cosy-default --out ../voices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel

ENGINE = "audio8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True, help="transcript, or a path to it")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="../voices")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seconds", type=float, help="trim the clip; shorter = cheaper prompts")
    ap.add_argument("--notes")
    args = ap.parse_args()

    import soundfile as sf

    text_path = Path(args.text)
    text = text_path.read_text(encoding="utf-8").strip() if text_path.is_file() else args.text
    text = " ".join(text.split())
    if not text:
        raise SystemExit("a transcript is required: the prompt interleaves it with the codes")

    device = torch.device(args.device)
    model = AutoModel.from_pretrained(args.weights, trust_remote_code=True, dtype=torch.float32)
    model = model.to(device).eval()

    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    if args.seconds:
        audio = audio[: int(rate * args.seconds)]
    target = model.config.codec_sample_rate
    if int(rate) != target:
        from torchaudio.functional import resample

        audio = resample(audio, int(rate), target)
    seconds = audio.numel() / target

    values = audio.reshape(1, 1, -1).to(device)
    lengths = torch.tensor([values.shape[-1]], device=device)
    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(values, lengths)
    n = int(code_lengths[0])
    codes = codes[0, :, :n].to(torch.int32).cpu().contiguous()

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    save_file({"reference_codes": codes}, str(out / "voice.safetensors"))
    manifest = {
        "engine": ENGINE,
        "name": args.name,
        "text": text,
        "seconds": round(seconds, 3),
        "notes": args.notes
        or f"encoded from {Path(args.audio).name} at {target} Hz by oracle/export_voice.py",
    }
    (out / "voice.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"{args.name}: {seconds:.2f} s -> {n} code frames, "
        f"range [{int(codes.min())}, {int(codes.max())}]"
    )
    print(f"wrote {out}/voice.json and voice.safetensors")


if __name__ == "__main__":
    main()
