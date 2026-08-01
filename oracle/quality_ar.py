"""Does quantizing the AR weights change the voice?

Speed without degradation is the whole requirement, so this measures degradation
directly rather than inferring it from weight error. Three metrics, in increasing
order of how much they actually matter:

1. **Weight error** — reported by `qroundtrip`. Cheap, and a poor predictor: an 8%
   weight perturbation can leave greedy decoding untouched or derail it.
2. **Token sequence identity.** Greedy generation is deterministic, so if the
   quantized model emits the *same* codes, the decoded waveform is bit-identical
   and degradation is not "small" but literally zero. This is the sharp test.
3. **Waveform / mel distance**, for when the tokens do diverge. Once a code
   sequence differs the audio differs everywhere downstream, so per-sample error
   is meaningless; mel distance and a listen are what count.

Also reports per-step top-1 agreement on the semantic logits under teacher
forcing, which localises where a quantization starts to bite without letting
divergence compound.

The codec stays at f32 throughout — only AR weights are under test.

Usage:
    .venv/bin/python quality_ar.py --variants ../fixtures/ar_q8_0.safetensors \
        ../fixtures/ar_q4_1.safetensors
"""
from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs."
)


def write_wav(path, wav, sr):
    """16-bit PCM, so WhisperX and a pair of ears can both consume it."""
    x = wav.reshape(-1).clamp(-1, 1).mul(32767).short().numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(x.tobytes())


def load_model(weights, device, dtype):
    return (
        AutoModel.from_pretrained(weights, trust_remote_code=True, dtype=dtype)
        .to(device)
        .eval()
    )


def build_prompt(model, processor, device):
    batch = processor(text=[TEXT], return_tensors="pt")
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def generate(model, batch, max_new_tokens):
    with torch.inference_mode():
        return model.generate(
            prefix_input_ids=batch["prefix_input_ids"],
            prefix_attention_mask=batch["prefix_attention_mask"],
            suffix_input_ids=batch["suffix_input_ids"],
            suffix_attention_mask=batch["suffix_attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )


def mel(wav, sr, n_mels=80):
    """Log-mel magnitude. Enough to compare timbre without pulling in torchaudio."""
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(
        wav.reshape(-1), n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    ).abs()
    # Triangular mel filterbank, built here to keep the dependency surface small.
    f_max = sr / 2
    m_min, m_max = 0.0, 2595 * torch.log10(torch.tensor(1 + f_max / 700))
    m_pts = torch.linspace(m_min, float(m_max), n_mels + 2)
    f_pts = 700 * (10 ** (m_pts / 2595) - 1)
    bins = torch.floor((n_fft + 1) * f_pts / sr).long()
    fb = torch.zeros(n_mels, spec.shape[0])
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        fb[i, left:centre] = torch.linspace(0, 1, centre - left)
        fb[i, centre:right] = torch.linspace(1, 0, right - centre)
    return torch.log(fb.to(spec.device) @ spec + 1e-6)


def teacher_forced_agreement(ref, alt, batch, codes, device):
    """Distributional fidelity per step, feeding both models the reference codes.

    Greedy TTS decoding is chaotic: a 0.4 logit nudge flips one near-tied token and
    every frame after it differs, so "codes differ" says nothing about quality and
    neither does waveform or mel distance between two different-but-valid
    renditions. What does carry information is how close the two models'
    *distributions* stay, step by step, with divergence prevented from compounding:

      - top-1 / top-5 agreement on the semantic distribution
      - KL(ref || alt), the information-theoretic distance
      - p_alt(ref's argmax) / max p_alt — how nearly alt would have agreed. Near
        1.0 on a disagreeing step means a coin-flip between near-ties, not a
        confident wrong answer.
    """
    if codes.shape[-1] == 0:
        return None
    prompt, mask = ref._prepare_prompt(
        prefix_input_ids=batch["prefix_input_ids"],
        prefix_attention_mask=batch["prefix_attention_mask"],
        suffix_input_ids=batch["suffix_input_ids"],
        suffix_attention_mask=batch["suffix_attention_mask"],
    )
    agree = 0
    agree5 = 0
    total = 0
    max_abs = 0.0
    kls = []
    margins = []
    with torch.inference_mode():
        for model in (ref, alt):
            model._setup_generation_caches(1, ref.config.max_seq_len, next(model.parameters()).dtype)
        width = prompt.shape[-1]
        cache_position = torch.arange(width, device=device)
        position_ids = mask.cumsum(-1).sub(1).clamp_min(0)
        logits = {}
        for tag, model in (("ref", ref), ("alt", alt)):
            logits[tag], _ = model._slow_step(prompt, cache_position, position_ids, mask)
        step_mask = torch.ones((1, ref.config.max_seq_len), dtype=torch.long, device=device)
        for step in range(codes.shape[-1]):
            a = logits["ref"][0, ref.config.semantic_begin_id : ref.config.semantic_end_id + 1]
            b = logits["alt"][0, ref.config.semantic_begin_id : ref.config.semantic_end_id + 1]
            agree += int(a.argmax() == b.argmax())
            top5 = torch.topk(a, 5).indices
            agree5 += int(b.argmax() in top5)
            max_abs = max(max_abs, (a - b).abs().max().item())
            p = torch.softmax(a.float(), -1)
            q = torch.softmax(b.float(), -1)
            kls.append((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum().item())
            margins.append((q[a.argmax()] / q.max()).item())
            total += 1
            # Feed both the *reference* frame so they stay in lockstep.
            frame = codes[:, :, step]
            nxt = torch.cat(
                [
                    (frame[:, :1] + ref.config.semantic_begin_id),
                    frame[:, :],
                ],
                dim=1,
            )[:, : ref.config.num_codebooks + 1, None]
            pos = torch.tensor([width + step], device=device)
            for tag, model in (("ref", ref), ("alt", alt)):
                logits[tag], _ = model._slow_step(nxt, pos, pos[None], step_mask)
    return {
        "top1": agree / total,
        "top5": agree5 / total,
        "kl": sum(kls) / len(kls),
        "kl_max": max(kls),
        "margin": sum(margins) / len(margins),
        "dlogit": max_abs,
        "n": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="../fixtures/quality")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.weights, trust_remote_code=True)
    ref = load_model(args.weights, device, dtype)
    codec = ref.load_codec(device=device).to(dtype=torch.float32).eval()
    sr = ref.config.codec_sample_rate
    batch = build_prompt(ref, processor, device)

    ref_codes = generate(ref, batch, args.max_new_tokens)
    print(f"reference: {ref_codes.shape[-1]} frames greedy")
    with torch.inference_mode():
        ref_wav = codec.decode(ref_codes.to(device)).float().cpu()
    ref_mel = mel(ref_wav, sr)
    write_wav(out / "ref.wav", ref_wav, sr)

    # One model reloaded per variant; state dicts are swapped in place so the
    # reference stays available for teacher forcing.
    alt = load_model(args.weights, device, dtype)

    print(
        f"\n{'variant':<10} {'frames':>6} {'codes':>10} {'top1':>7} {'top5':>7} "
        f"{'KL mean':>9} {'KL max':>8} {'margin':>7} {'dlogit':>7} {'mel':>7}"
    )
    print("-" * 90)
    for path in args.variants:
        name = Path(path).stem.replace("ar_", "")
        sd = load_file(path)
        alt.load_state_dict(sd, strict=False)
        alt = alt.to(device=device, dtype=dtype).eval()

        alt_codes = generate(alt, batch, args.max_new_tokens)
        n = min(ref_codes.shape[-1], alt_codes.shape[-1])
        same_shape = ref_codes.shape == alt_codes.shape
        identical = same_shape and bool(torch.equal(ref_codes.cpu(), alt_codes.cpu()))
        if identical:
            diverge = "never"
            codes_note = "IDENTICAL"
        else:
            eq = (ref_codes[..., :n] == alt_codes[..., :n]).all(dim=1)[0]
            first = int((~eq).nonzero()[0]) if (~eq).any() else n
            diverge = str(first)
            codes_note = f"{int(eq.sum())}/{n} same"

        tf = teacher_forced_agreement(ref, alt, batch, ref_codes, device)

        if identical:
            mel_dist = 0.0
        else:
            with torch.inference_mode():
                alt_wav = codec.decode(alt_codes.to(device)).float().cpu()
            m = mel(alt_wav, sr)
            k = min(m.shape[-1], ref_mel.shape[-1])
            mel_dist = (ref_mel[..., :k] - m[..., :k]).abs().mean().item()
            write_wav(out / f"{name}.wav", alt_wav, sr)

        print(
            f"{name:<10} {alt_codes.shape[-1]:>6} {codes_note:>10} "
            f"{tf['top1']:>7.3f} {tf['top5']:>7.3f} {tf['kl']:>9.4f} {tf['kl_max']:>8.3f} "
            f"{tf['margin']:>7.3f} {tf['dlogit']:>7.3f} {mel_dist:>7.3f}"
        )

    print(
        "\nHow to read this. `codes IDENTICAL` would mean the waveform is bit-identical\n"
        "to f32 — zero degradation. Short of that, `mel` is NOT a quality measure: two\n"
        "different-but-valid renditions of the same text are far apart in mel space.\n"
        "The columns that carry signal are top1/top5 agreement, KL, and margin\n"
        "(p_alt at ref's choice over p_alt's own max — near 1.0 means near-ties, not\n"
        f"confident errors). Intelligibility is settled by WhisperX on {out}/*.wav."
    )


if __name__ == "__main__":
    main()
