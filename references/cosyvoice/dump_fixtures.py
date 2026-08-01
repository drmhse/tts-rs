"""Phase A fixtures for CosyVoice3: per-stage ground truth for the Rust port.

The Audio8 port validated first try — codec at 2.8e-6, greedy generation bit-identical
— because fixtures existed before any Rust was written. This is the same step for
CosyVoice, and the three stages are deliberately taken apart rather than captured as
one end-to-end pair: a whole-pipeline mismatch tells you nothing about which of the
LLM, the flow decoder or the vocoder is wrong.

What is deterministic and therefore directly checkable:

  * **LLM logits and greedy tokens** for a fixed prompt. The *sampled* sequence is not
    reproducible across implementations (`ras_sampling` draws from torch's generator),
    so the port is gated on prefill logits and on a greedy rollout, exactly as the
    Audio8 slow AR was. The sampled sequence is still exported, because the flow stage
    needs *some* fixed token sequence to be validated against.
  * **flow mel** given a token sequence. Deterministic because `CausalConditionalCFM`
    does not draw noise: it built `rand_noise` once with `set_all_random_seed(0)` and
    slices the same tensor every call. That tensor is exported as an asset — a port
    that samples its own noise looks correct and sounds different.
  * **vocoder waveform** given a mel — but only once the NSF noise is pinned; see below.

Two things this script exports that reading the reference does not obviously demand,
both established by measurement (`scratchpad/nsf.py`, reproduced in COSYVOICE_PORT.md):

  * `hift.nsf_noise`. `SineGen2(causal=True)` holds `self.sine_waves = torch.rand(1,
    300*24000, 9)` as a **plain attribute, not a registered buffer**, so it is absent
    from `hift.pt` and redrawn at construction. It is only reproducible at all because
    `cosyvoice3.yaml` line 4 calls `torch.manual_seed(1986)` immediately before
    building the model. It is not negligible: zeroing it moves the waveform by
    max 0.164 against a signal of rms 0.078. So the slice actually used is exported and
    the Rust vocoder is validated with it injected.
  * `hift.f0`, `hift.source`. The stage boundary inside the vocoder, so a mismatch
    separates "the F0 predictor is wrong" from "the harmonic source is wrong" from
    "the upsampling decoder is wrong".

And one thing it deliberately does *not* export: `SineGen2.rand_ini`. The initial phase
offset is added to sample 0 only and then discarded by the `scale_factor=1/480` linear
downsample, which reads samples 239 and 240 of each block. Measured contribution to the
source signal: exactly 0.0. It is dead code in the reference and the port omits it.

Runs under the CosyVoice venv, from the CosyVoice directory.

Usage:
    cd /path/to/CosyVoice
    PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
        /path/to/references/cosyvoice/dump_fixtures.py \
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
        --voice /path/to/tts-rs/voices/cosy-default-cosyvoice \
        --out /path/to/tts-rs/fixtures/cosyvoice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TEXT = "Welcome to CosyVoice three, running on Apple Silicon."

# How many greedy LLM steps to record. Greedy decoding is deterministic on both sides,
# so this is the check that actually proves the loop agrees — embedding, RoPE, GQA, KV
# cache and the head all at once. 32 is enough to catch drift and cheap to run on CPU.
GREEDY_STEPS = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--voice", required=True, help="voice asset directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.common import set_all_random_seed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading CosyVoice3 (llm + flow + hift) ...")
    cosy = CosyVoice3(args.model_dir, fp16=False)
    model = cosy.model
    device = torch.device("cpu")
    llm = model.llm.to(device).eval()
    flow = model.flow.to(device).eval()
    hift = model.hift.to(device).eval()

    voice = load_file(str(Path(args.voice) / "voice.safetensors"))
    manifest = json.loads((Path(args.voice) / "voice.json").read_text())
    prompt_text = voice["prompt_text_tokens"].to(torch.int32)
    speech_token = voice["speech_tokens"].to(torch.int32)
    prompt_mel = voice["prompt_mel"].to(torch.float32)
    embedding = voice["speaker_embedding"].to(torch.float32)
    print(
        f"voice `{manifest['name']}`: {speech_token.shape[1]} speech tokens, "
        f"{prompt_mel.shape[1]} mel frames"
    )

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {
        "text": TEXT,
        "voice": manifest["name"],
        "voice_text": manifest["text"],
        "seed": args.seed,
        "device": "cpu",
        "dtype": "float32",
        "greedy_steps": GREEDY_STEPS,
    }

    def put(name: str, t: torch.Tensor) -> None:
        tensors[name] = t.detach().float().cpu().contiguous()

    # ------------------------------------------------------------------ prompt
    text_token, text_token_len = cosy.frontend._extract_text_token(TEXT)
    text_token, text_token_len = text_token.to(device), text_token_len.to(device)
    tensors["prompt.text_tokens"] = text_token.to(torch.int32).cpu().contiguous()
    tensors["prompt.prompt_text_tokens"] = prompt_text.cpu().contiguous()
    tensors["prompt.speech_tokens"] = speech_token.cpu().contiguous()
    put("prompt.speaker_embedding", embedding)
    put("prompt.prompt_mel", prompt_mel)
    print(f"prompt: {text_token.shape[1]} text tokens for the target text")

    # ------------------------------------------------------------------- LLM
    # Build the prompt exactly as `CosyVoice3LM.inference` does, so the port can be
    # checked on the assembled input rather than on a re-derivation of it.
    full_text = torch.concat([prompt_text, text_token], dim=1)
    assert 151646 in full_text, "<|endofprompt|> missing — see trap 2 in COSYVOICE_PORT.md"
    text_emb = llm.llm.model.model.embed_tokens(full_text)
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    prompt_speech_emb = llm.speech_embedding(speech_token)
    lm_input = torch.concat([sos_emb, text_emb, task_emb, prompt_speech_emb], dim=1)
    put("llm.lm_input", lm_input)
    print(f"llm: prefill width {lm_input.shape[1]}")

    # Prefill logits at the last position: the single most diagnostic tensor, because
    # every weight in the stack contributes to it.
    with torch.inference_mode():
        y_pred, cache = llm.llm.forward_one_step(
            lm_input,
            masks=torch.tril(
                torch.ones((1, lm_input.shape[1], lm_input.shape[1]), dtype=torch.bool)
            ),
            cache=None,
        )
        put("llm.prefill_hidden", y_pred[:, -1])
        logits = llm.llm_decoder(y_pred[:, -1])
        put("llm.prefill_logits", logits)

        # Greedy rollout. Deterministic on both sides, but a weak gate on this model:
        # it degenerates to a repeated token within a step or two, and a constant is
        # something a *wrong* implementation can also produce. Recorded because it is
        # nearly free, gated on only as a smoke test — the real check is teacher
        # forcing below.
        greedy, step_in = [], lm_input
        cache = None
        for _ in range(GREEDY_STEPS):
            y_pred, cache = llm.llm.forward_one_step(
                step_in,
                masks=torch.tril(
                    torch.ones((1, step_in.shape[1], step_in.shape[1]), dtype=torch.bool)
                ),
                cache=cache,
            )
            tok = int(llm.llm_decoder(y_pred[:, -1]).argmax(dim=-1).item())
            greedy.append(tok)
            if tok >= llm.speech_token_size:
                break
            step_in = llm.speech_embedding.weight[tok].reshape(1, 1, -1)
    tensors["llm.greedy_tokens"] = torch.tensor([greedy], dtype=torch.int32)
    meta["llm_greedy_tokens"] = len(greedy)
    print(f"llm: {len(greedy)} greedy tokens (degenerate: {len(set(greedy)) == 1})")

    # The sampled sequence, for the flow stage to consume. Reproducible for *this*
    # reference at this seed, not across implementations.
    set_all_random_seed(args.seed)
    with torch.inference_mode():
        speech_tokens = [
            int(t)
            for t in llm.inference(
                text=text_token,
                text_len=text_token_len,
                prompt_text=prompt_text.to(device),
                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
                prompt_speech_token=speech_token.to(device),
                prompt_speech_token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32),
                embedding=embedding.to(device),
            )
        ]
    generated = torch.tensor([speech_tokens], dtype=torch.int32)
    tensors["llm.speech_tokens"] = generated.cpu().contiguous()
    meta["llm_tokens"] = len(speech_tokens)
    print(f"llm: {len(speech_tokens)} sampled speech tokens (seed {args.seed})")

    # Teacher forcing over that sequence: the gate that actually proves the decode loop
    # agrees. Feeding known, non-degenerate tokens exercises the KV cache, RoPE at real
    # positions, GQA and the head, and compares a dense [n, 6761] logit surface rather
    # than a constant. This is the CosyVoice equivalent of Audio8's "24/24 frames
    # bit-identical" row.
    with torch.inference_mode():
        tf_logits, step_in, cache = [], lm_input, None
        for i in range(len(speech_tokens)):
            y_pred, cache = llm.llm.forward_one_step(
                step_in,
                masks=torch.tril(
                    torch.ones((1, step_in.shape[1], step_in.shape[1]), dtype=torch.bool)
                ),
                cache=cache,
            )
            tf_logits.append(llm.llm_decoder(y_pred[:, -1]))
            step_in = llm.speech_embedding.weight[speech_tokens[i]].reshape(1, 1, -1)
        tf = torch.concat(tf_logits, dim=0)
    put("llm.tf_logits", tf)
    tensors["llm.tf_argmax"] = tf.argmax(dim=-1).to(torch.int32).reshape(1, -1).cpu().contiguous()
    print(
        f"llm: teacher-forced logits {tuple(tf.shape)}, "
        f"{len(set(tf.argmax(dim=-1).tolist()))} distinct argmax ids"
    )

    # ----------------------------------------------------------- the flow decoder
    # Reproduce the pre-DiT conditioning explicitly so the port can be gated on `mu`,
    # `cond` and `spks` before the 440-block-pass solver is ever run. A DiT that is
    # fed the wrong `mu` produces plausible mel and is undebuggable end to end.
    from cosyvoice.utils.mask import make_pad_mask

    with torch.inference_mode():
        spks = F.normalize(embedding, dim=1)
        spks = flow.spk_embed_affine_layer(spks)
        put("flow.spks", spks)

        tok_all = torch.concat([speech_token, generated], dim=1)
        tok_len = torch.tensor([tok_all.shape[1]], dtype=torch.int32)
        tmask = (~make_pad_mask(tok_len)).unsqueeze(-1).to(spks)
        tok_emb = flow.input_embedding(torch.clamp(tok_all, min=0)) * tmask
        put("flow.token_emb", tok_emb)

        h = flow.pre_lookahead_layer(tok_emb)
        put("flow.lookahead", h)
        h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_mel.shape[1], h.shape[1] - prompt_mel.shape[1]
        mu = h.transpose(1, 2).contiguous()
        put("flow.mu", mu)

        conds = torch.zeros([1, mel_len1 + mel_len2, flow.output_size]).to(h.dtype)
        conds[:, :mel_len1] = prompt_mel
        put("flow.cond", conds.transpose(1, 2))
        meta["flow_mel_len1"] = int(mel_len1)
        meta["flow_mel_len2"] = int(mel_len2)
        print(f"flow: mu {tuple(mu.shape)}  prompt {mel_len1} + generated {mel_len2} frames")

        # One DiT evaluation at the solver's first timestep, on the doubled CFG batch.
        # This isolates "the DiT block is wrong" from "the solver is wrong".
        cfm = flow.decoder
        n_steps = 10
        t_span = torch.linspace(0, 1, n_steps + 1)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        put("flow.t_span", t_span)
        z = cfm.rand_noise[:, :, : mu.size(2)]
        put("flow.z", z)
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h).unsqueeze(1)
        n = mu.size(2)
        x_in = torch.zeros([2, 80, n])
        x_in[:] = z
        mask_in = torch.zeros([2, 1, n])
        mask_in[:] = mask
        mu_in = torch.zeros([2, 80, n])
        mu_in[0] = mu
        t_in = torch.zeros([2])
        t_in[:] = t_span[0]
        spks_in = torch.zeros([2, 80])
        spks_in[0] = spks
        cond_in = torch.zeros([2, 80, n])
        cond_in[0] = conds.transpose(1, 2)
        # Taps inside the DiT, so an accumulated difference can be told apart from a
        # wrong layer. Without these, "the DiT is off by rel 4e-4" is unactionable.
        est = cfm.estimator
        taps: dict[str, torch.Tensor] = {}
        h_time = est.time_embed(t_in)
        taps["flow.dit_time"] = h_time
        h_in = est.input_embed(
            x_in.transpose(1, 2),
            cond_in.transpose(1, 2),
            mu_in.transpose(1, 2),
            spks_in,
        )
        taps["flow.dit_input"] = h_in
        rope = est.rotary_embed.forward_from_seq_len(h_in.shape[1])
        h = h_in
        for bi, block in enumerate(est.transformer_blocks):
            h = block(h, h_time, mask=None, rope=rope)
            if bi in (0, 1, 10, len(est.transformer_blocks) - 1):
                taps[f"flow.dit_block{bi}"] = h
        for name, t in taps.items():
            put(name, t)
        print(f"flow: DiT taps {sorted(taps)}")

        put("flow.dit_step0", cfm.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False))

        mel_full, _ = cfm(
            mu=mu,
            mask=mask,
            spks=spks,
            cond=conds.transpose(1, 2),
            n_timesteps=n_steps,
            streaming=False,
        )
        put("flow.mel_full", mel_full)

        mel, _ = flow.inference(
            token=generated.to(device),
            token_len=torch.tensor([generated.shape[1]], dtype=torch.int32),
            prompt_token=speech_token.to(device),
            prompt_token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_mel.to(device),
            prompt_feat_len=torch.tensor([prompt_mel.shape[1]], dtype=torch.int32),
            embedding=embedding.to(device),
            streaming=False,
            finalize=True,
        )
    put("flow.mel", mel)
    print(f"flow: mel {tuple(mel.shape)} (full {tuple(mel_full.shape)})")

    # ---------------------------------------------------------------- the vocoder
    voc_mel = mel
    sg = hift.m_source.l_sin_gen

    # The NSF noise slice, before anything mutates it. See the module docstring: this is
    # not in the checkpoint and is only reproducible via the yaml's manual_seed(1986).
    n_samples = voc_mel.shape[2] * int(
        torch.prod(torch.tensor(hift.upsample_rates)).item()
    ) * hift.istft_params["hop_len"]
    put("hift.nsf_noise", sg.sine_waves[:, :n_samples])
    meta["hift_upsample_total"] = int(n_samples / voc_mel.shape[2])

    with torch.inference_mode():
        # F0 in both precisions. The reference moves the predictor to float64 with a
        # comment that precision is crucial; Metal has no f64, so the port needs to
        # know whether that matters here. Exported so the gap is measured, not assumed.
        import copy

        p32 = copy.deepcopy(hift.f0_predictor).float()
        f0_32 = p32(voc_mel.float())
        p64 = copy.deepcopy(hift.f0_predictor).double()
        f0_64 = p64(voc_mel.double())
        put("hift.f0", f0_64)
        put("hift.f0_f32", f0_32)
        print(
            f"hift: f0 {tuple(f0_64.shape)}  f32-vs-f64 max|d| "
            f"{(f0_32 - f0_64.float()).abs().max().item():.3e} Hz"
        )

        # The harmonic source, the vocoder's other input.
        s_up = hift.f0_upsamp(f0_64.float()[:, None]).transpose(1, 2)
        source, _, _ = hift.m_source(s_up)
        source = source.transpose(1, 2)
        put("hift.source", source)

        wav, _ = hift.inference(speech_feat=voc_mel, finalize=True)
    put("hift.mel_in", voc_mel)
    put("hift.wav", wav)
    print(f"hift: source {tuple(source.shape)}  wav {tuple(wav.shape)} "
          f"({wav.shape[-1] / cosy.sample_rate:.2f} s)")

    # --------------------------------------------- the fixed noise, as an asset
    # `CausalConditionalCFM.__init__` does set_all_random_seed(0) then
    # randn([1, 80, 15000]). Reproducing torch's Philox stream from Rust is not
    # practical, so ship the tensor.
    rand_noise = flow.decoder.rand_noise.float().cpu().contiguous()
    save_file({"rand_noise": rand_noise}, str(out / "rand_noise.safetensors"))
    print(
        f"rand_noise: {tuple(rand_noise.shape)} -> rand_noise.safetensors "
        f"({rand_noise.numel() * 4 / 1e6:.1f} MB)"
    )

    save_file(tensors, str(out / "oracle.safetensors"))
    manifest_out = {
        "meta": meta,
        "sample_rate": cosy.sample_rate,
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()
        },
    }
    (out / "oracle.json").write_text(json.dumps(manifest_out, indent=1) + "\n")
    print(f"\nwrote {out}/oracle.safetensors ({len(tensors)} tensors) and oracle.json")


if __name__ == "__main__":
    main()
