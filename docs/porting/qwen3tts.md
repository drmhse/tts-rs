# Porting Qwen3-TTS-12Hz-1.7B-Base

Status as of 2026-08-10: **ported and synthesizing.**

`cargo run -q -p qwen3tts --release --bin qwen3tts-validate` passes **63 rows, 0 failures** —
a shape audit of every `cfg` constant against the checkpoint header, then per-stage numerics
against `fixtures/qwen3tts/oracle.safetensors`:

| row | result |
|---|---|
| `prompt.embeds` | rel 1.4e-7 |
| `talker.hidden` / `talker.logits` | rel 9.2e-7 / 1.7e-6 |
| talker argmax codebook 0 | **identical (1226)** |
| predictor argmax codes 1..15 | **15/15 identical** |
| `step1.input` (the loop update) | **exact, max abs 0.0** |
| `step1.hidden` / `step1.logits` | rel 7.1e-7 / 1.6e-6 |
| `codec.quantized` | **exact, max abs 0.0** |
| `codec.pre_tf` | rel 1.3e-6 |
| `codec.wav`, `codec.wav_chunked` | rel 5.6e-6 |

Measured on this Mac (16 GB, Metal, q8_0) over `examples/senior.txt`, 7 segments:
**RTF 0.863** — talker 0.695, codec 0.168. It produces 50.66 s of audio for that passage
against Audio8's 50.27 s and CosyVoice's 54.86 s, so the three agree on duration.
CosyVoice is RTF 0.710 on the same run with a trunk a third the size.

Time the runs uncontended. The same render measured 1.051 while a CPU job was resident — this
repo has been fooled by consecutive timings on this Mac before.

The order here is deliberate: this document was written *before* any model code, from the
reference implementation and the checkpoint's own tensor shapes. The Audio8 port validated
first try because fixtures existed before any Rust did; this is the same idea applied one
step earlier, to the traps.

## Why a third engine

| | CosyVoice3-0.5B | Qwen3-TTS-12Hz-1.7B |
|---|---|---|
| frame rate | 25 Hz | **12.5 Hz** |
| waveform path | 22-block DiT, 10 Euler steps, CFG-doubled batch | RVQ dequant, 8-layer windowed transformer, causal conv stack |
| trunk | Qwen2-0.5B, 24 layers at 896 | Qwen3, 28 layers at 2048 |
| per frame | 1 LLM step | 1 talker step **+ 15 predictor steps** |
| seed-tts WER en / zh | 1.45 / **0.71** | **1.24** / 0.77 |
| languages | broader | **ten, closed list** |

The flow decoder is 65-68% of CosyVoice's RTF and this model has no equivalent. Against
that: 3.4x the trunk parameters, and a depth transformer that is not free (trap 2).

The closed language list is the reason this is an addition and not a replacement. There is
no `codec_language_id` outside those ten, so text outside them has no faithful prefill —
not a quality tradeoff, an absence. CosyVoice stays the default.

## Architecture

Per audio frame, at 12.5 Hz:

```
text tokens ──┐
              ├─> talker (28L @ 2048) ─> codec_head ─> codebook 0
prev frame's ─┘         │
16 embeddings           └─ last hidden ─> code_predictor (5L @ 1024)
   (summed @ 2048)                          └─ 15 sequential AR steps ─> codebooks 1..15
                                                        │
       16 codes ──> RVQ dequant ──> pre_conv ──> 8L transformer (win 72)
                        └──> 2x2 upsample ──> conv stack [8,5,4,3] ──> 24 kHz
```

The talker's input embedding at each step is a **sum** of two streams — the previous frame's
16 codebook embeddings, and the next text token's projected hidden — which is what makes the
model streamable and is the source of trap 3.

## The three findings that cost the most

Neither was in the reference's model code, and neither would have been found by reading.

**1. f32 on a bf16 checkpoint is a 38x slowdown, not a 2x memory cost.** `Weights::get` casts
to f32, so the f32 talker is 6.3 GB of projections; with a 1.09 GB KV cache on a 16 GB machine
that thrashes. Measured per frame: **f32 1994 ms, q8_0 52 ms.** The talker alone went
1331 ms -> 16 ms. So `qwen3tts` defaults to **q8_0** where the other two engines default to
f32, and three other footprint sources were fixed with it: the cache capacity (5120 -> 1536
positions), `text_embedding` (1.24 GB as f32, kept raw bf16 and cast per selected row), and the
15 predictor tables (503 MB as f32, same treatment). The lesson generalises past this repo:
a "memory" problem showed up as a pure throughput number, and the first instinct was to look
for a slow kernel.

**2. The processor applies no chat template — the model wrapper does.**
`Qwen3TTSModel._build_assistant_text` wraps text as
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` *before* tokenizing, and
only then does `generate` slice the affixes back off with `[:, :3]` and `[:, 3:-5]`
(`[:, 3:-2]` for the reference transcript). Tokenizing raw text and slicing anyway silently
eats the first three words and leaves the text slice empty. Both `export_voice.py` and the
first `dump_fixtures.py` did exactly that: the voice asset's `ref_text_tokens` came out 20
tokens where it should be 25, and the fixture's role prefix was the first three words of the
sentence. The Rust side was right the whole time; the *fixture* was wrong.

That is the argument for splitting prompt assembly from the model in the gate. The failing run
showed `prompt.embeds` at rel 5.3e-2 with every model stage passing — one row, pointing at one
thing. Had the gate only compared end-to-end audio, it would have said "everything is broken".

**3. A wrong ICL transcript makes the model ramble to its token cap, and nothing downstream
detects it.** The first voice asset paired an 11.76 s transcript with a 4.72 s excerpt of the
same recording. Result: 197 frames (15.76 s) for a 51-character sentence that should be 50
frames — and the PyTorch reference on the identical input was *worse*, running to its full
`max_new_tokens=2048` for 163.76 s of audio. Rebuilding the asset from the 11.76 s clip with its
verbatim transcript gave exactly 50 frames / 4.00 s.

Two lessons. ICL conditions on transcript **and** codes jointly, so a mismatch is not a mild
quality loss — it destroys the stopping behaviour. And "my port generates too much" is not
evidence of a port bug: the control has to be run before concluding anything, and here the
control was further off than the port. `export_voice.py` now warns when the transcript's token
rate falls outside 1.5-7.0 tokens/second of audio, which is the cheapest proxy available
without an ASR pass.

## Traps

Nine, all silent. Numbered as in the crate docs.

1. **The config asks for M-RoPE; the model does not use it.** `rope_scaling` gives
   `mrope_section: [24, 20, 20]` with `interleaved: true`, and
   `apply_multimodal_rotary_pos_emb` (modeling_qwen3_tts.py:660) implements it properly. But
   `get_rope_index` (:1746) is degenerate: it builds `cumsum(attention_mask) - 1` and
   `.unsqueeze(0).expand(3, -1, -1)`, so all three sections are the same row at every
   position. `apply_interleaved_rope` (:694) then starts from `x[0].clone()` and overwrites
   strided slices with values from `x[1]` and `x[2]` — equal to `x[0]` — and returns it
   unchanged. `cat([...] * 2)` and `rotate_half` follow, so the result is **ordinary
   half-split RoPE at theta 1e6**. Implementing the config costs a day and changes nothing.
   `position_id_per_seconds: 13` is dead too — no uses in the package, and 13 is not this
   model's frame rate.
2. **The "MTP block" is 15 sequential AR steps.** The technical report says the residual
   codebooks are predicted "in parallel". `Qwen3TTSTalkerForConditionalGeneration::forward`
   (:1668) calls `code_predictor.generate(max_new_tokens=num_code_groups - 1)` — a real
   autoregressive loop with its own cache. The true per-frame cost is one 28-layer pass plus
   **15 five-layer passes**. Any RTF estimate from the report's wording is wrong by most of
   the predictor.
3. **Text is consumed at one token per audio frame.** `inputs_embeds` per step is
   `codec_hiddens.sum(1)` plus `trailing_text_hidden[:, generation_step]`, falling back to
   `tts_pad_embed` when the text runs out (:1686). Consequence: a segment whose audio stops
   early leaves **unspoken text in the buffer**, and unlike CosyVoice — where truncation had
   to be inferred from a speech-tokens-per-character ratio — the shortfall here is exactly
   `len(trailing_text) - generation_step`. Use that, not a ratio heuristic.
4. **Codebooks are stored divided.** `EuclideanCodebook` (tokenizer_v2:661) keeps
   `embedding_sum [2048, 256]` and `cluster_usage [2048]`; `decode` computes
   `embedding_sum / cluster_usage.clamp(min=1e-5)[:, None]`. These are training EMA
   accumulators, so `cluster_usage` is not all-ones. Using `embedding_sum` directly is a
   per-row scale error on every code that loads, runs and produces audio. Fold at load, as
   `tts_nn::Weights::get_weight_norm` does for weight-norm.
5. **`semantic_codebook_size: 4096` is false.** Zero uses in the reference; every codebook
   tensor in the checkpoint is `[2048, 256]`. This port believed the config, wrote the
   assertion, and the shape audit failed it on the first run — which is the entire argument
   for auditing shapes before writing model code. Separately, the *encoder* does have 31
   acoustic quantizers with only the first 16 valid (`encoder_valid_num_quantizers`), so a
   voice asset built from all of them is silently wrong from codebook 16 on.
6. **The codec decoder is not only a ConvNet.** An 8-layer transformer at hidden 512 with a
   **live** 72-frame sliding window sits between quantizer and convolutions, with
   `input_proj [512, 1024]` and `output_proj [1024, 512]`. The talker's `sliding_window` is
   null; this one is not. Shared attention code that ignores the window is wrong here only.
7. **head_dim is decoupled from hidden_size in the predictor.** Hidden 1024, 16 heads,
   head_dim 128, so `q_proj` is `[2048, 1024]` and `o_proj` `[1024, 2048]`. Deriving
   `head_dim = hidden / heads` gives 64; every shape still divides and the model runs.
8. **The 15 codebook embedding tables are at talker width.** Each `[2048, 2048]`, narrowed
   by `small_to_mtp_projection [1024, 2048]`. Dual-use: the talker sums all 16 of a frame's
   embeddings at 2048 to form its own next input, so the tables cannot live at 1024.
9. **No attention biases — except `text_projection` has them.** `attention_bias: false`
   throughout both transformers, while `Qwen3TTSTalkerResizeMLP` is built `bias=True`. One
   rule for the whole checkpoint drops two bias vectors, and a dropped bias passes every
   shape check.

Two things that look like traps and are not, recorded so nobody spends time on them:
`VectorQuantization.project_out` is an `Identity` (codebook_dim == dim), so its absence from
the checkpoint is correct; and unlike CosyVoice — which hardcodes `cuda if available else
cpu` and has **no MPS path at all** — this reference takes a standard HF `device_map`, so
the reference side runs on CPU without patching.

## What transfers from the existing engines

- **Narrow KV** and **GQA by query reshape** from `cosyvoice/llm.rs`: 8 KV heads, 2 query
  heads each. Worth 5.10x and unchanged in arithmetic.
- **In-place `slice_set`, never `slice_assign`,** for the KV cache. `slice_assign`
  reallocated ~100 MB per generated token and cost 2.0x on CosyVoice's LLM stage.
- **Longest-first lane sorting** if segments are batched: only a contiguous *tail* of
  finished lanes can be shed, so unsorted batching was slower than no batching at all.
- **`tts_nn`'s Metal kernels** for the conv stack — tap-major im2col (3.98-6.45x) and fused
  snake, and the decoder uses `SnakeBeta`. Crucially the codec's `chunked_decode`
  (chunk 300 frames, left context 25) means the stack is called **many times per utterance**,
  which is the call pattern where candle's size-pooled Metal buffers recycle. CosyVoice's
  `hift.forward` is called once and took none of that win; this should take it.

## What does not transfer

- **QK-norm** — new code; neither existing engine has it.
- **The dual-track prompt.** CosyVoice is prompt-then-generate; this interleaves text and
  audio per frame. The prefill assembly is genuinely different and is where trap 3 lives.
- **The sampler.** `top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05` for the
  talker and `top_k=50, top_p=1.0, temperature=0.9` for the predictor, plus `suppress_tokens`
  over the top 1024 of the talker's vocabulary except `codec_eos`. Not CosyVoice's
  `ras_sampling` and not Audio8's Gumbel-max.

## Remaining work

1. **Where the time is.** The talker is 80% of RTF and the per-frame breakdown says host
   reads are the largest single component: 16 `to_vec1` round trips per frame (one for
   codebook 0, 15 for the predictor) measured at 33-44 ms/frame against 16 ms for the talker
   step itself. Sampling on device, or batching the predictor's head reads, is the obvious
   next lever. `QWEN3TTS_TIMING=1` on `qwen3tts-probe` prints the split.
2. **Segment batching.** Unlike CosyVoice's LLM, each frame's input depends on that lane's own
   text cursor, so batching needs per-lane trailing text. CosyVoice got 1.21x from this.
3. **Streaming.** The architecture is natively streaming (dual-track text, chunked codec) and
   `Capabilities::streaming` is still false because the trait has no method for it.
4. **`--set language=`** is honoured but there is no per-request language field; a mixed-language
   document cannot switch mid-run.
