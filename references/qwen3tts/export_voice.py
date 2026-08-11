"""Turn a reference clip into a `qwen3tts` voice asset.

Runs the checkpoint's ECAPA-TDNN speaker encoder and 12 Hz RVQ encoder once, offline, so the
Rust binary contains neither.

Exports:
  spk_embedding    [1, 2048]  x-vector; width == talker hidden_size, so 0.6B/1.7B assets
                              are not interchangeable
  ref_codes        [T, 16]    the clip's RVQ codes, frames-major (ICL cloning)
  ref_text_tokens  [1, N]     transcript, already sliced

The slicing is the one non-obvious decision: the reference builds prompts from
`ref_ids[:, 3:-2]` and `input_id[:, 3:-5]`, offsets that belong to the processor's chat
template rather than the model. Slicing here avoids reimplementing that template in Rust.

Both cloning modes are supported from one asset — ICL uses all three tensors, x-vector-only
uses just the embedding — so the mode stays a request-time choice.

Usage:
    .venv/bin/python references/qwen3tts/export_voice.py \
        --model references/qwen3tts/weights \
        --audio examples/cosy_short.wav \
        --text "The transcript of that clip, exactly." \
        --name my-voice \
        --out voices/my-voice-qwen3tts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

ENGINE = "qwen3tts"

# The processor applies **no** chat template — `Qwen3TTSModel._build_ref_text` wraps the
# string first, and only then does `generate` slice the affixes back off. Tokenizing raw text
# and slicing [3:-2] silently eats the first three real words.
ROLE_PREFIX_TOKENS = 3
REF_SUFFIX_TOKENS = 2


def build_ref_text(text: str) -> str:
    """`Qwen3TTSModel._build_ref_text` — 3 leading tokens, 2 trailing."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (references/qwen3tts/weights)")
    ap.add_argument("--audio", required=True, help="reference clip, any sample rate")
    ap.add_argument("--text", required=True, help="exact transcript of the clip")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="voice asset directory to create")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    # Imported here so `--help` works without the venv's heavy dependencies resolved.
    from qwen_tts import Qwen3TTSModel

    # CPU float32: runs once per voice, and keeps the embedding comparable with the fp32
    # fixtures.
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    model = tts.model

    if model.tts_model_type != "base":
        raise SystemExit(
            f"voice cloning needs a Base checkpoint; this one is `{model.tts_model_type}`. "
            "CustomVoice and VoiceDesign select or describe a speaker instead of cloning one."
        )

    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        # Mix rather than take channel 0, so a panned speaker keeps level.
        wav = wav.mean(axis=1)
    seconds = len(wav) / sr

    # --- speaker embedding -------------------------------------------------------------
    # The reference resamples before this call, not inside it.
    target_sr = model.speaker_encoder_sample_rate
    if sr != target_sr:
        import librosa

        wav_spk = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    else:
        wav_spk = wav
    with torch.no_grad():
        spk = model.extract_speaker_embedding(audio=wav_spk, sr=target_sr)
    spk = spk.detach().to(torch.float32).reshape(1, -1).contiguous()

    enc_dim = model.config.speaker_encoder_config.enc_dim
    if spk.shape[1] != enc_dim:
        raise SystemExit(f"speaker embedding is {spk.shape[1]}-wide, config says {enc_dim}")
    talker_dim = model.config.talker_config.hidden_size
    if spk.shape[1] != talker_dim:
        # The embedding is one position in the talker's stream — a mismatch means this asset
        # belongs to the other size class.
        raise SystemExit(
            f"speaker embedding is {spk.shape[1]}-wide but the talker's hidden size is "
            f"{talker_dim} — this asset would not fit the checkpoint it was built from"
        )

    # --- reference codes ---------------------------------------------------------------
    with torch.no_grad():
        enc = model.speech_tokenizer.encode(wav, sr=sr)
    codes = enc.audio_codes[0].detach().to(torch.int32).contiguous()
    # [T, Q], frames-major — the orientation `VoiceClonePromptItem.ref_code` documents and
    # `generate_icl_prompt` indexes as `ref_code[:, i:i+1]`. Transposing it is silent.
    if codes.ndim != 2:
        raise SystemExit(f"expected [T, 16] codes, got shape {tuple(codes.shape)}")
    n_q = model.speech_tokenizer.config.encoder_valid_num_quantizers
    if codes.shape[1] != n_q:
        # Encoder config says 32 quantizers, only the first 16 are valid — a real check.
        raise SystemExit(f"expected {n_q} codebooks, got {codes.shape[1]}")

    # --- reference text tokens ---------------------------------------------------------
    ref_ids = tts.processor(
        text=build_ref_text(args.text), return_tensors="pt", padding=True
    )["input_ids"]
    ref_tokens = ref_ids[:, ROLE_PREFIX_TOKENS:-REF_SUFFIX_TOKENS].to(torch.int32).contiguous()
    if ref_tokens.shape[1] == 0:
        raise SystemExit(
            "the transcript tokenized to nothing after the template affixes were removed; "
            "--text must be the clip's actual transcript"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "spk_embedding": spk,
            "ref_codes": codes,
            "ref_text_tokens": ref_tokens,
        },
        str(out / "voice.safetensors"),
    )

    manifest = {
        "engine": ENGINE,
        "name": args.name,
        "text": args.text,
        "seconds": round(seconds, 3),
        "notes": args.notes,
        # So a 0.6B asset handed to a 1.7B engine fails with a sentence, not a shape error.
        "enc_dim": enc_dim,
        "frames": int(codes.shape[0]),
        "codebooks": int(codes.shape[1]),
    }
    (out / "voice.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {out}")
    print(f"  spk_embedding    {tuple(spk.shape)}")
    print(f"  ref_codes        {tuple(codes.shape)}  ({codes.shape[0] / 12.5:.2f} s at 12.5 Hz)")
    print(f"  ref_text_tokens  {tuple(ref_tokens.shape)}")
    print(f"  clip             {seconds:.2f} s at {sr} Hz")
    # Disagreement means the encoder got the resampled waveform or the wrong channel.
    drift = abs(codes.shape[0] / 12.5 - seconds)
    if drift > 0.25:
        print(f"  warning: codes cover {codes.shape[0] / 12.5:.2f} s but the clip is {seconds:.2f} s")

    # A transcript that does not match the clip is the worst failure mode here and nothing
    # downstream detects it: ICL cloning conditions on transcript-and-codes together, so a
    # mismatch makes the talker ramble to its token cap instead of speaking the target text.
    # Measured with an 11.76 s transcript against a 4.72 s clip: 197 frames for a 51-character
    # sentence that should be 50, and the PyTorch reference was worse still at 2047 frames.
    # Token rate is the cheapest available proxy — real speech runs about 2-6 tokens/second.
    rate = ref_tokens.shape[1] / max(seconds, 1e-6)
    if not 1.5 <= rate <= 7.0:
        print(
            f"  WARNING: {ref_tokens.shape[1]} transcript tokens over {seconds:.2f} s is "
            f"{rate:.1f} tokens/s, outside the plausible 1.5-7.0 band.\n"
            f"           --text must be the transcript of *this* clip, verbatim. A mismatch "
            f"makes the talker generate until it hits max_new_tokens."
        )


if __name__ == "__main__":
    main()
