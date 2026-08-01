"""Long-form synthesis with the measured optimizations applied.

This is the reference pipeline, not the Rust port — see the note at the bottom of
`IMPLEMENTATION_STATUS.md`. What it does carry is every finding from
`AR_LOOP_RESULTS.md` that is expressible in PyTorch:

  - **Narrow KV.** `ArkttsKVCache` already supports `return_full=False`, which
    returns `keys[:, :, :pos+1]` instead of the whole `max_seq_len`-wide buffer.
    The reference sets `return_full=True` and then attends over all 2048 positions
    with a mask every step. Flipping it (and narrowing the mask to match) was worth
    5.10x in the Rust probe. Nothing about it is approximate — the extra positions
    are zeros that the mask discards anyway.
  - **Batching.** Segments are generated in batches, because per-sequence cost falls
    up to 11.95x with batch size. Batch 2 and 3 are skipped: they are a per-sequence
    *regression* in candle, and batching wide is free here too.
  - **Segmentation** on sentence boundaries, which is what makes batching possible
    at all and also keeps each prompt far below `max_seq_len`.

Not applied: the logit slice (1.26x) needs the semantic mask to run before the
projection, and rewiring `generate`'s processor chain for it is a Rust-port concern.
Not applied either: q8_0 weights. They are a *candle* win — PyTorch has no quantized
matvec path here, so loading round-tripped q8_0 weights would buy the error without
the speed. The checkpoint's native bfloat16 is the right choice for this path.

Usage:
    .venv/bin/python synthesize.py --text-file ../examples/senior.txt \
        --out ../examples/senior.wav
"""
from __future__ import annotations

import argparse
import re
import time
import wave
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


def patch_narrow_kv(model) -> None:
    """Attend over 0..pos instead of the full max_seq_len cache.

    Two coupled changes: the caches must return only the populated prefix, and
    `_slow_step`'s mask must be built to that same width. Doing one without the
    other is a shape error, which is presumably why the reference does neither.
    """
    original_setup = model._setup_generation_caches

    def setup(batch_size, max_length, dtype):
        original_setup(batch_size, max_length, dtype)
        # Only the slow layers benefit: the fast cache is num_codebooks (10) wide,
        # where full-width costs nothing.
        for layer in model.layers:
            layer.attention.kv_cache.return_full = False

    def slow_step(input_ids, cache_position, position_ids, attention_mask):
        hidden = model._embed(input_ids)
        rope = model.freqs_cis[position_ids]
        key_length = int(cache_position[-1].item()) + 1
        mask = model._causal_mask(attention_mask, cache_position, key_length)
        for layer in model.layers:
            hidden = layer(hidden, rope, mask, cache_position)
        hidden = hidden[:, -1:]
        normalized = model.norm(hidden)
        logits = torch.nn.functional.linear(normalized, model.embeddings.weight)[:, -1]
        fast_hidden = normalized if model.config.norm_fastlayer_input else hidden
        return logits, fast_hidden

    model._setup_generation_caches = setup
    model._slow_step = slow_step


def patch_f32_sampling(model) -> None:
    """Draw the Gumbel noise in float32.

    `_sample` is Gumbel-max: `argmax(softmax(scores) / -log(u))`. It builds `u` with
    `dtype=probabilities.dtype`, so under the checkpoint's native bfloat16 both the
    probabilities and the uniforms carry an 8-bit mantissa — about 256 distinct
    values across [0, 1). That is far too coarse for the ratio ordering to survive:
    sampled output is unintelligible and never reaches EOS, while greedy decoding
    from the identical model is clean. Computing the draw in f32 costs nothing (it is
    one vector per token) and leaves the bf16 matmuls untouched.
    """

    def sample(scores, generator=None):
        probabilities = torch.softmax(scores.float(), dim=-1)
        random = torch.rand(
            probabilities.shape,
            dtype=torch.float32,
            device=probabilities.device,
            generator=generator,
        )
        # u == 0 would give -log(0) = inf, which merely makes that entry
        # unselectable, but clamping keeps the ratio finite and well defined.
        noise = -torch.log(random.clamp_min(torch.finfo(torch.float32).tiny))
        return torch.argmax(probabilities / noise, dim=-1)

    model._sample = sample


def load_reference(model, processor, path, text, device, seconds=None, save=None):
    """Encode a reference clip once, and return codes to reuse for every segment.

    `_prepare_prompt` will happily take `reference_audio_values` and encode per call,
    but the encoder pass is identical every time, so it is done once here. Encoding
    also has a second payoff: **the Rust port then never needs the codec encoder at
    all.** `convert_codec.py` drops the 126 encoder tensors, and cloning from raw
    audio would have required putting them back. Shipping the reference *codes* as an
    asset keeps the encoder out of the port permanently.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    if seconds:
        audio = audio[: int(rate * seconds)]
    target = model.config.codec_sample_rate
    if int(rate) != target:
        from torchaudio.functional import resample

        audio = resample(audio, int(rate), target)
    values = audio.reshape(1, 1, -1).to(device)
    lengths = torch.tensor([values.shape[-1]], device=device)
    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(values, lengths)
    n = int(code_lengths[0])
    codes = codes[0, :, :n].cpu()
    print(
        f"reference: {path.name}, {audio.numel() / target:.2f} s at {target} Hz "
        f"-> {n} code frames, range [{int(codes.min())}, {int(codes.max())}]"
    )
    if save:
        from safetensors.torch import save_file

        save_file({"reference_codes": codes.contiguous()}, str(save))
        print(f"  saved codes to {save} (the port can load these; no encoder needed)")
    return codes, text


def segment(text: str, max_chars: int) -> list[list[str]]:
    """Split into paragraphs, then into segments of whole sentences.

    Returns a list of paragraphs, each a list of segments, so silence can be longer
    between paragraphs than between segments of one paragraph.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    out = []
    for para in paragraphs:
        # Collapse the stray ".." in the source and normalise whitespace.
        para = re.sub(r"\.\.+", ".", " ".join(para.split()))
        sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", para)
        sentences = [s.strip() for s in sentences if s.strip()]
        segments, current = [], ""
        for s in sentences:
            if current and len(current) + 1 + len(s) > max_chars:
                segments.append(current)
                current = s
            else:
                current = f"{current} {s}".strip()
        if current:
            segments.append(current)
        if segments:
            out.append(segments)
    return out


def write_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    x = wav.reshape(-1).clamp(-1, 1).mul(32767).short().numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(x.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=4, help="1, or >=4 (2 and 3 regress)")
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gap-ms", type=int, default=90, help="silence between segments")
    ap.add_argument("--para-gap-ms", type=int, default=320)
    ap.add_argument("--no-narrow-kv", action="store_true")
    ap.add_argument("--no-f32-sampling", action="store_true")
    ap.add_argument("--reference-audio", help="clip to clone, e.g. CosyVoice asset/default_voice.wav")
    ap.add_argument("--reference-text", help="transcript of the clip, or a path to it")
    ap.add_argument("--reference-seconds", type=float, help="trim the clip; shorter = cheaper prompts")
    ap.add_argument("--save-reference-codes", help="write the encoded codes for the Rust port")
    args = ap.parse_args()

    if args.batch in (2, 3):
        raise SystemExit(
            "batch 2 and 3 are a per-sequence regression (candle takes its dedicated "
            "matvec path only at batch 1). Use 1 or >=4."
        )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    text = Path(args.text_file).read_text(encoding="utf-8")
    paragraphs = segment(text, args.max_chars)
    flat = [(pi, s) for pi, para in enumerate(paragraphs) for s in para]
    print(f"{len(paragraphs)} paragraphs -> {len(flat)} segments (<= {args.max_chars} chars)")

    model = AutoModel.from_pretrained(args.weights, trust_remote_code=True, dtype=dtype)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.weights, trust_remote_code=True)
    if not args.no_narrow_kv:
        patch_narrow_kv(model)
        print("narrow-KV lever: on")
    if not args.no_f32_sampling:
        patch_f32_sampling(model)
        print("f32 Gumbel sampling: on")
    codec = model.load_codec(device=device).to(dtype=torch.float32).eval()
    sr = model.config.codec_sample_rate
    frame_rate = sr / model.config.codec_frame_size

    # `_sample` draws with `torch.rand(..., device=scores.device)`, so the
    # generator has to live on the same device.
    ref_codes = None
    ref_text = None
    if args.reference_audio:
        rt = args.reference_text
        if rt and Path(rt).is_file():
            rt = Path(rt).read_text(encoding="utf-8").strip()
        if not rt:
            raise SystemExit(
                "--reference-text is required with --reference-audio: the prompt "
                "interleaves the clip's transcript with its codes."
            )
        ref_codes, ref_text = load_reference(
            model, processor, Path(args.reference_audio), rt, device,
            args.reference_seconds, args.save_reference_codes,
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    pieces: list[tuple[int, torch.Tensor]] = []
    total_frames = 0
    t0 = time.perf_counter()

    for start in range(0, len(flat), args.batch):
        chunk = flat[start : start + args.batch]
        texts = [s for _, s in chunk]
        kw = {}
        if ref_codes is not None:
            # Same clip for every row in the batch.
            kw = {
                "reference_codes": [ref_codes] * len(texts),
                "reference_text": [ref_text] * len(texts),
            }
        batch = processor(text=texts, return_tensors="pt", **kw)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.inference_mode():
            out = model.generate(
                prefix_input_ids=batch["prefix_input_ids"],
                prefix_attention_mask=batch["prefix_attention_mask"],
                suffix_input_ids=batch["suffix_input_ids"],
                suffix_attention_mask=batch["suffix_attention_mask"],
                reference_codes=batch.get("reference_codes"),
                reference_code_lengths=batch.get("reference_code_lengths"),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                generator=generator,
                return_dict_in_generate=True,
            )
        for i, (pi, seg) in enumerate(chunk):
            n = int(out.code_lengths[i].item())
            total_frames += n
            if n == 0:
                print(f"  [warn] segment produced no frames: {seg[:50]!r}")
                continue
            codes = out.codes[i : i + 1, :, :n]
            with torch.inference_mode():
                wav = codec.decode(codes.to(device)).float().cpu().reshape(-1)
            pieces.append((pi, wav))
            print(f"  seg {start + i + 1}/{len(flat)}: {n} frames, {n / frame_rate:.2f} s")

    elapsed = time.perf_counter() - t0

    # Stitch: short silence inside a paragraph, longer between paragraphs.
    gap = torch.zeros(int(sr * args.gap_ms / 1000))
    para_gap = torch.zeros(int(sr * args.para_gap_ms / 1000))
    joined: list[torch.Tensor] = []
    prev_para = None
    for pi, wav in pieces:
        if prev_para is not None:
            joined.append(para_gap if pi != prev_para else gap)
        joined.append(wav)
        prev_para = pi
    audio = torch.cat(joined) if joined else torch.zeros(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, audio, sr)

    seconds = audio.numel() / sr
    print(
        f"\n{seconds:.2f} s of audio at {sr} Hz in {elapsed:.1f} s wall"
        f"  ->  RTF {elapsed / seconds:.3f}"
    )
    print(f"{total_frames} frames total, batch {args.batch}, {args.dtype}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
