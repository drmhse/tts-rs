"""Per-stage fp32 CPU ground truth for the Rust port.

fp32 on CPU so a mismatch is a port bug, not a precision difference. Stages are taken apart
rather than captured end to end: a whole-pipeline diff says nothing about which of the prompt
assembly, the talker, the predictor or the codec is wrong.

What is dumped, and why each boundary:

  prompt.embeds        `talker_input_embeds` — the assembled prompt. The single most likely
                       thing to be wrong, and everything downstream depends on it.
  prompt.trailing      `trailing_text_hiddens` — the one-token-per-frame text stream (trap 3).
  talker.hidden        prefill's last normed hidden state.
  talker.logits        `codec_head` over it — what picks codebook 0.
  predictor.logits_N   the 15 residual heads for frame 0, teacher-forced on the reference's
                       own codebook-0 code so a wrong code0 does not cascade.
  predictor.codes      the argmax codes those logits give.
  step1.input          the loop's next-frame input: 16 embeddings summed plus the next
                       text token. `step1.hidden` / `step1.logits` follow from it.
  codec.codes          the reference clip's full [T, 16] code block — past the 72-frame
                       sliding window, so the window mask is actually exercised.
  codec.long.*         the same codes tiled past the 300-frame chunk boundary.
  codec.quantized      split-RVQ output, before pre_conv.
  codec.pre_tf         the pre-transformer's output, after output_proj.
  codec.wav            the waveform.

Deliberately teacher-forced rather than sampled: the reference draws from torch's generator,
so a sampled sequence is not reproducible across implementations. Argmax is.

Usage:
    references/qwen3tts/.venv/bin/python references/qwen3tts/dump_fixtures.py \\
        --model references/qwen3tts/weights \\
        --voice voices/cosy-default-qwen3tts \\
        --out fixtures/qwen3tts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TEXT = "Hello from Rust."
LANGUAGE = "english"
# Frames of codec codes to check the decoder on.
#
# 40 was wrong and hid a real gap: the pre-transformer's sliding window is **72 frames** and
# `chunked_decode` splits at **300**, so a 40-frame fixture exercises neither. Real segments
# average ~88 frames, i.e. past the window. Two blocks now:
#   codec.*        every frame of the reference clip (147) — past the window, one chunk
#   codec.long.*   the clip's codes tiled to 400 — past the chunk boundary too
# Synthetic tiling is fine here: the decoder's length behaviour is what is under test, not
# whether the codes are a plausible utterance.
CODEC_FRAMES = None   # None = all of them
CODEC_LONG_FRAMES = 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    model = tts.model
    talker = model.talker
    cfg = model.config
    tcfg = cfg.talker_config

    voice = Path(args.voice)
    assets = load_file(str(voice / "voice.safetensors"))
    manifest = json.loads((voice / "voice.json").read_text())
    spk = assets["spk_embedding"].to(torch.float32)
    ref_code = assets["ref_codes"].to(torch.long)          # [T, 16]
    ref_text_tokens = assets["ref_text_tokens"].to(torch.long)  # [1, N]

    out: dict[str, torch.Tensor] = {}
    inventory: dict[str, object] = {"text": TEXT, "language": LANGUAGE, "voice": manifest["name"]}

    # The processor applies no chat template; `_build_assistant_text` does, and only then does
    # `generate` slice the affixes off. Feeding raw text here makes `[3:-5]` empty and the role
    # prefix the first three words of the sentence.
    assistant = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    full = tts.processor(text=assistant, return_tensors="pt", padding=True)["input_ids"]
    text_id = full[:, 3:-5]
    assert text_id.shape[1] > 0, "the template slice is empty — affix widths changed"
    out["tokens.full"] = full.to(torch.int32)
    out["tokens.text"] = text_id.to(torch.int32)
    out["tokens.role"] = full[:, :3].to(torch.int32)
    inventory["role_ids"] = full[0, :3].tolist()
    inventory["text_ids"] = text_id[0].tolist()

    with torch.no_grad():
        # ---------------------------------------------------------------- prompt
        # Mirror `generate`'s assembly for the ICL path with an x-vector.
        tts_bos, tts_eos, tts_pad = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]])
            )
        ).chunk(3, dim=1)

        language_id = tcfg.codec_language_id[LANGUAGE]
        codec_prefill = torch.tensor(
            [[tcfg.codec_think_id, tcfg.codec_think_bos_id, language_id, tcfg.codec_think_eos_id]]
        )
        codec_0 = talker.get_input_embeddings()(codec_prefill)
        codec_1 = talker.get_input_embeddings()(
            torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]])
        )
        codec_input = torch.cat([codec_0, spk.view(1, 1, -1), codec_1], dim=1)

        role = talker.text_projection(talker.get_text_embeddings()(full[:, :3]))
        prefix = (
            torch.cat([tts_pad.expand(-1, codec_input.shape[1] - 2, -1), tts_bos], dim=1)
            + codec_input[:, :-1]
        )
        icl, trailing = model.generate_icl_prompt(
            text_id=text_id,
            ref_id=ref_text_tokens,
            ref_code=ref_code,
            tts_pad_embed=tts_pad,
            tts_eos_embed=tts_eos,
            non_streaming_mode=False,
        )
        embeds = torch.cat([role, prefix, icl], dim=1)
        out["prompt.embeds"] = embeds
        out["prompt.trailing"] = trailing
        out["prompt.codec_input"] = codec_input
        out["prompt.tts_pad"] = tts_pad
        out["prompt.tts_bos"] = tts_bos
        out["prompt.tts_eos"] = tts_eos
        inventory["prompt_width"] = int(embeds.shape[1])
        inventory["trailing_width"] = int(trailing.shape[1])

        # ---------------------------------------------------------------- talker prefill
        result = talker.model(
            input_ids=None,
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long),
            position_ids=torch.arange(embeds.shape[1]).view(1, -1).expand(3, 1, -1),
            use_cache=True,
        )
        hidden = result.last_hidden_state[:, -1:, :]
        logits = talker.codec_head(hidden)
        out["talker.hidden"] = hidden
        out["talker.logits"] = logits
        code0 = int(logits[0, -1].argmax())
        inventory["talker_argmax_code0"] = code0

        # ---------------------------------------------------------------- predictor
        # Teacher-forced on the reference's own code0, and on its own argmax at each step.
        code0_embed = talker.get_input_embeddings()(torch.tensor([[code0]]))
        pred_in = torch.cat([hidden, code0_embed], dim=1)
        pred = talker.code_predictor
        h = pred.model(
            input_ids=None,
            inputs_embeds=pred.small_to_mtp_projection(pred_in),
            use_cache=True,
        )
        past = h.past_key_values
        last = h.last_hidden_state[:, -1:, :]
        codes = []
        for step in range(tcfg.num_code_groups - 1):
            step_logits = pred.lm_head[step](last)
            out[f"predictor.logits_{step}"] = step_logits
            code = int(step_logits[0, -1].argmax())
            codes.append(code)
            if step + 1 < tcfg.num_code_groups - 1:
                embed = pred.model.get_input_embeddings()[step](torch.tensor([[code]]))
                h = pred.model(
                    input_ids=None,
                    inputs_embeds=pred.small_to_mtp_projection(embed),
                    past_key_values=past,
                    use_cache=True,
                )
                past = h.past_key_values
                last = h.last_hidden_state[:, -1:, :]
        out["predictor.codes"] = torch.tensor(codes, dtype=torch.int32)
        inventory["predictor_argmax_codes"] = codes

        # ------------------------------------------------------- one decode step
        # The loop update (trap 3) is the only part the prefill fixtures do not cover: the
        # next input is the frame's 16 codebook embeddings summed, plus the next text token.
        frame0 = [code0] + codes
        step_embed = talker.get_input_embeddings()(torch.tensor([[frame0[0]]]))
        for i, c in enumerate(frame0[1:]):
            step_embed = step_embed + pred.model.get_input_embeddings()[i](torch.tensor([[c]]))
        # generation_step 0 takes trailing[:, 0]; here trailing is one pad wide.
        step_embed = step_embed + trailing[:, 0].unsqueeze(1)
        out["step1.input"] = step_embed

        step1 = talker.model(
            input_ids=None,
            inputs_embeds=torch.cat([embeds, step_embed], dim=1),
            attention_mask=torch.ones((1, embeds.shape[1] + 1), dtype=torch.long),
            position_ids=torch.arange(embeds.shape[1] + 1).view(1, -1).expand(3, 1, -1),
            use_cache=True,
        )
        step1_hidden = step1.last_hidden_state[:, -1:, :]
        out["step1.hidden"] = step1_hidden
        out["step1.logits"] = talker.codec_head(step1_hidden)
        inventory["step1_argmax_code0"] = int(out["step1.logits"][0, -1].argmax())

        # ---------------------------------------------------------------- codec
        # The reference clip's own codes, so the block is real rather than synthetic.
        codec_codes = (
            ref_code if CODEC_FRAMES is None else ref_code[:CODEC_FRAMES]
        ).contiguous()
        out["codec.codes"] = codec_codes.to(torch.int32)
        # `speech_tokenizer` is the inference wrapper; the module is `.model`.
        dec = model.speech_tokenizer.model.decoder
        # `[1, 16, T]` — `decode` transposes `[B, T, Q]` into this before `chunked_decode`.
        codes_t = codec_codes.transpose(0, 1).unsqueeze(0).contiguous()
        quantized = dec.quantizer.decode(codes_t)
        out["codec.quantized"] = quantized
        pre = dec.pre_conv(quantized).transpose(1, 2)
        pre_tf = dec.pre_transformer(inputs_embeds=pre).last_hidden_state
        out["codec.pre_tf"] = pre_tf
        # One `forward` over the whole block, and the chunked path the engine actually takes.
        out["codec.wav"] = dec(codes_t)
        out["codec.wav_chunked"] = dec.chunked_decode(codes_t)
        inventory["codec_frames"] = int(codec_codes.shape[0])
        inventory["codec_samples"] = int(out["codec.wav"].shape[-1])

        # Past the 300-frame chunk boundary, so `chunked_decode`'s left-context trimming is
        # actually checked rather than assumed.
        reps = -(-CODEC_LONG_FRAMES // codec_codes.shape[0])
        long_codes = codec_codes.repeat(reps, 1)[:CODEC_LONG_FRAMES].contiguous()
        long_t = long_codes.transpose(0, 1).unsqueeze(0).contiguous()
        out["codec.long.codes"] = long_codes.to(torch.int32)
        out["codec.long.wav"] = dec(long_t)
        out["codec.long.wav_chunked"] = dec.chunked_decode(long_t)
        inventory["codec_long_frames"] = int(long_codes.shape[0])

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    save_file(
        # .clone(): trailing is literally tts_pad when the codec side is the longer of the
        # two, and safetensors refuses aliased storage.
        {k: (v.to(torch.float32) if v.is_floating_point() else v).contiguous().clone()
         for k, v in out.items()},
        str(outdir / "oracle.safetensors"),
    )
    inventory["tensors"] = {k: list(v.shape) for k, v in out.items()}
    (outdir / "oracle.json").write_text(json.dumps(inventory, indent=2) + "\n")

    print(f"wrote {outdir}/oracle.safetensors")
    for k, v in out.items():
        print(f"  {k:<24} {tuple(v.shape)}")
    print(f"\ntalker argmax code0: {code0}")
    print(f"predictor argmax codes: {codes}")


if __name__ == "__main__":
    main()
