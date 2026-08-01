# Audio8-TTS-Preview-0.6b — Rust port spec

Derived by reading the vendored `modeling_arktts.py`, `modeling_arktts_codec.py`,
`configuration_arktts.py` at revision `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`
(published 2026-07-29). Pin this revision; the port is coupled to this exact
architecture and weight layout.

Everything below is read off the source, not inferred from the model card.

---

## 1. Shape of the thing

Two autoregressive transformers plus a residual-VQ codec.

```
text tokens ──> slow AR (24 layers) ──> semantic token  (= codebook 0)
                     │                        │
                     └── hidden ──> fast AR (4 layers) ──> codebooks 1..9
                                                              │
                    codes [10, T] ──> codec decoder ──> 44.1 kHz waveform
```

**One code frame = 2048 audio samples**, so the frame rate is
44100 / 2048 = **21.53 Hz**. That is remarkably low; a 10 s clip is only ~215
frames, which is why `max_seq_len` of 2048 is enough for long-form.

### Slow AR

Geometry is identical to Qwen2.5-0.5B: `dim 896`, `n_layer 24`, `n_head 14`,
`n_local_heads 2` (GQA, 7× repeat), `head_dim 64`, `intermediate_size 4864`,
`norm_eps 1e-6`, `rope_base 1e6`, tied embeddings, qkv bias **on**, o bias
**off**, no qk-norm.

**But the weight layout is not Qwen2's.** This is the Fish-Speech / llama-from-
scratch layout: a single fused `wqkv` producing `(n_head + 2·n_local_heads)·head_dim
= (14+4)·64 = 1152` columns, plus `wo`, `feed_forward.w1/w2/w3`, and
`attention_norm` / `ffn_norm`. So `candle_transformers::models::qwen2` cannot be
dropped in — the arithmetic matches, the parameter names and fusion do not.
Either split `wqkv` at conversion time or implement the fused form directly
(prefer fused: one matmul is faster).

### Fast AR

`n_fast_layer 4`, `fast_dim 896`, `fast_n_head 14`, `fast_n_local_heads 2`,
`fast_head_dim 64`, `fast_intermediate_size 4864`, qkv bias **off**.
`use_sdpa=False` — it uses the manual `scores @ v` path with a boolean mask.
Its RoPE table is precomputed for only `num_codebooks = 10` positions.

### Embedding trick

`forward` takes `input_ids` of shape `[B, num_codebooks+1, T]` — **row 0 is the
text/semantic token, rows 1..10 are the codebook tokens**.

```
hidden = embeddings(ids[:,0]) + Σ_{i=0..9} codebook_embeddings(ids[:,i+1] + i*4096)
```

`codebook_embeddings` is one `Embedding(4096*10, 896)` table with a per-codebook
offset. The codebook sum is **zeroed** wherever row 0 is not in the semantic
range — i.e. text positions get text embeddings only.

### Codebook 0 is the semantic token

In `_generate_codebooks`: `current = semantic - semantic_begin_id`, and that
value is `codebooks[0]`. The fast AR therefore only *samples* codebooks 1..9;
codebook 0 comes straight from the slow AR's semantic token. Conversely
`_prepare_prompt` adds `semantic_begin_id` back to reference codebook 0.
Semantic ids occupy `151678..155773` = exactly 4096 slots.

---

## 2. Port traps (ranked by how much time they will cost)

1. **RoPE is interleaved, not half-split.** `_apply_rope` reshapes the last dim
   into adjacent `(real, imag)` pairs. Candle's default `rotary_emb::rope` uses
   the half-split convention; you need `rope_i`. Getting this wrong produces
   audio that is *plausible but wrong*, which is the worst failure mode.
2. **RoPE tables are built in bfloat16, then applied in fp32.**
   `_precompute_rope` ends `.to(torch.bfloat16)`; `_apply_rope` then does
   `x.float()`. To match the oracle tightly you must replicate the bf16
   round-trip on the table, not compute fresh fp32 sin/cos.
3. **`_fast_step(hidden, 0)` result is discarded** (`modeling_arktts.py:519`).
   It exists solely to prime the fast KV cache at position 0 with the projected
   slow hidden state. Skip it and codebooks 1..9 are garbage.
4. **The top-k/top-p filter softmaxes *before* temperature.** `_processed_scores`
   runs the legacy filter (sort → softmax → cumsum → drop `cum > top_p` or
   `pos >= top_k`, always keep rank 0) and *then* divides by temperature. The
   conventional order is the opposite. Replicate as written.
5. **Residual codebooks are size 1024, not 4096.** The fast head emits 4096-way
   logits for every codebook, but the codec's residual quantizer codebooks hold
   only 1024 entries; `ArkttsDownsampleQuantizer.decode` **clamps** rows 1..9 to
   `0..1023` (row 0 to `0..4095`). *Measured*: over a real greedy generation, 0 of
   216 residual codes exceeded 1023 (observed range 1..1014) — the model learned
   to stay in range, so the clamp is a safety net, not load-bearing. Port it
   anyway (it is two `min` calls) but do not treat it as a correctness risk.
6. **Sampling is Gumbel-max, not multinomial**: `argmax(softmax(s) / -log(u))`
   with one uniform per vocab entry. Structurally far easier to mirror than a
   CDF walk — but see §4 on reproducibility.
7. **RAS draws twice per step.** `_sample_semantic` samples once at
   `(top_p, temperature)` and again at `(ras_top_p=0.9, ras_temperature=1.0)`,
   then substitutes the second when the first repeats within a 10-token window.
   Two generator draws per step, in that order.
8. **The RAS window initialises to zeros**, so the first 10 steps compare against
   token id 0, which is outside the semantic range and therefore never triggers.

---

## 3. Free performance sitting in the Python implementation

These are inefficiencies in the reference code, not framework costs, and the
Rust port gets them for free. They are the reason the port may beat the Python
service on RTF even if Candle/Metal only matches torch/MPS per-kernel.

- **`ArkttsKVCache` is allocated at full `max_seq_len` (2048) with
  `return_full=True`**, so every decode step attends over all 2048 positions and
  masks the unused ones — from step 1, regardless of actual length. A 200-frame
  clip does ~10× the attention work it needs. Attending `0..pos` is the obvious
  fix.
- **`sdpa_kernel(SDPBackend.MATH)` is forced** for every decode step
  (`modeling_arktts.py:723`), i.e. the slow materialised-scores path, because the
  full-width boolean mask defeats the fused kernels.
- **`repeat_interleave` materialises K and V** to full head count on every step
  instead of broadcasting GQA.
- **The `[B,1,1,2048]` mask is rebuilt every step** from scratch.

---

## 4. Reproducibility: the guarantee changes

The existing CosyVoice service promises every clip is regenerable from its
recorded seed. That cannot hold *across* engines: different RNG streams and
different reduction orders. Within `audio8-rs` seeds will be reproducible; across
Python↔Rust they will not be.

This precedent already exists in the project — the fp16 LLM switch broke
byte-exactness against fp32-era artifacts. The requirement is that **engine
identity and version become part of the artifact metadata** in the job store, so
a clip records what can actually regenerate it. Decide this before the job store
is written, not after.

Bit-exactness against Python is technically reachable (Gumbel-max over
`torch.rand` on a CPU MT19937 generator is implementable) but is not worth the
constraint it puts on the whole pipeline. Use the WhisperX gate for equivalence
instead: match and coverage, not byte diffs.

---

## 5. Codec — the Phase C target

**Good news: no float64, no STFT, no pitch predictor.** CosyVoice's vocoder had
to run on CPU because of a float64 pitch predictor with no MPS kernel. Nothing in
this codec has that problem. Total op inventory for the decode path:

| op | where | Candle risk |
|---|---|---|
| `ConvTranspose1d`, stride 2/4/8/8, k=2·stride | decoder blocks, upsample | **highest** — Metal coverage and perf |
| depthwise `Conv1d` groups=1024, k=7 | `ConvNeXtBlock.dwconv` | medium |
| `Conv1d` k=1/2/3/7, dilation 1/3/9 | everywhere | low |
| embedding lookup | `decode_code` | none |
| Snake: `x + (α+1e-9)⁻¹·sin(αx)²` | every residual unit | none (elementwise) |
| RMSNorm (f32 accum), LayerNorm eps 1e-6 | transformers, ConvNeXt | none |
| LayerScale (γ multiply) | codec transformer blocks | none |
| SDPA + windowed causal mask (128/512) | pre/post modules | low |
| left `pad`, `Tanh`, slicing | length fixups | none |

**Fold weight_norm at conversion time.** Every conv is wrapped in `weight_norm`,
so `weight = g · v/‖v‖`. Compute that once during `.pth → safetensors` conversion
and ship plain weights — no runtime reparametrisation. Note **two key layouts**:
`nn.utils.parametrizations.weight_norm` (`parametrizations.weight.original0/1`)
for the convs, and legacy `weight_norm` (`weight_g` / `weight_v`) for the
quantizer `in_proj`/`out_proj`. Handle both.

### Decode path, exactly

```
codes [B,10,T], clamp row0→0..4095, rows1..9→0..1023
  semantic = semantic_quantizer.from_codes(codes[:,:1])     # 1 cb × 4096 × dim 8
  residual = quantizer.from_codes(codes[:,1:])              # 9 cb × 1024 × dim 8
    each: embedding(dim 8) → out_proj: Conv1d(8→1024, k=1)  (weight-normed)
  z = upsample(post_module(semantic + residual))
    post_module: 8-layer windowed transformer, dim 1024, 16 heads / 8 kv,
                 head_dim 64, ffn 1216, window 128, causal, rope_base 1e4,
                 norm_eps 1e-5, channels_first, in/out proj = Identity
    upsample: 2 × [ConvTranspose1d(1024→1024, k=2, s=2) + ConvNeXtBlock(1024)]
  wav = decoder(z)
    Conv1d(1024→1536, k=7)
    4 × DecoderBlock, strides (8,8,4,2), dims 1536→768→384→192→96:
       Snake → ConvTranspose1d(k=2·stride, stride) → 3× ResidualUnit(dil 1,3,9)
       ResidualUnit = Snake → Conv1d(k=7, dilated) → Snake → Conv1d(k=1) → +skip
    Snake(96) → Conv1d(96→1, k=7) → Tanh
```

Upsampling: `2·2 · 8·8·4·2 = 4 · 512 = 2048` = `codec_frame_size`. Consistent.

**Causality is exact**, so chunked/streaming decode is natural. It will also be
necessary: the decoder runs at full audio rate, and at 192 channels × 441 000
samples × 4 B that is ~339 MB for a single activation on a 10 s clip. Chunk it.

### The encoder is not optional

Voice cloning goes through `encode_audio` → `codec.encode`, so cloning a
reference wav needs the **encoder** too (`Conv1d(1→64,k=7)`, 4 encoder blocks
strides 2/4/8/8, a 4-layer window-512 transformer at the last block, then
`Conv1d(→1024,k=3)`), plus `pre_module`, `downsample`, and the VQ *search*
(`decode_latents`: L2-normalise, cosine distance against the codebook, argmax).
Phase C can defer this by caching reference codes from the oracle, but a shippable
service needs it. Alternatively: precompute and store codes per voice, and only
port the encoder when arbitrary user-supplied reference audio is required.
