# Does quantization change the voice? q8_0: no, measurably.

The requirement is speed at near-zero voice degradation, so degradation was
measured rather than inferred. Probes: `qroundtrip` (Rust, candle's own quantizer),
`references/audio8/quality_ar.py` (torch, greedy generation + distributional metrics),
faster-whisper `small.en` from the CosyVoice align venv for intelligibility.

## Verdict

**Use q8_0.** It is 3.35× faster than f32 and transcribes *identically* to the f32
reference. q4_1 buys only 1.42× more and does degrade.

| | q8_0 | q5_0 | q4_1 | q4_0 |
|---|---|---|---|---|
| bytes / param | 1.06 | 0.69 | 0.62 | 0.56 |
| speed vs f32 | **3.35×** | 4.16× | 4.76× | 4.35× |
| relative weight error | **0.55%** | 4.4% | 8.0% | 8.8% |
| semantic top-1 agreement | **95.8%** | 82.3% | 76.0% | 81.2% |
| semantic top-5 agreement | **100%** | 100% | 100% | 99.0% |
| KL(ref‖alt), mean | **0.0005** | 0.0219 | 0.1007 | 0.1033 |
| KL, worst step | **0.005** | 0.105 | 0.463 | 0.489 |
| margin at ref's choice | **0.998** | 0.965 | 0.920 | 0.931 |
| max logit delta | **0.42** | 2.45 | 5.52 | 6.14 |
| WER vs f32 reference audio | **0.000** | 0.071 | 0.071 | 0.214 |

## What was quantized, and what deliberately was not

Only the 28 transformer layers' five projections — 417 M of 601 M params (69%), and
~94% of the bytes a decode step reads. Left at f32:

- **`embeddings.weight` [155776, 896]** — it is both the input embedding and, tied,
  the logit head, so quantizing it perturbs token *choice* most directly. And it is
  nearly free to keep: the logit slice reduces the head to 4097 of 155776 rows, so
  full precision there costs 3.7 M params of reads per token instead of 139.6 M.
  **The logit slice and this decision reinforce each other** — the slice is what
  makes full-precision embeddings affordable.
- `codebook_embeddings`, `fast_embeddings`, `fast_output` — gathers and one small
  head. Negligible bandwidth, direct effect on sampling.
- all `*_norm.weight` and `wqkv.bias` — tiny, and an RMSNorm scale error propagates
  through everything downstream of it.

## Why the obvious metrics are the wrong ones

**Token identity fails as a test, for every variant including q8_0.** All four emit
a different code sequence from f32 within the first frame. That sounds fatal and is
not: greedy decoding over 4096 near-tied semantic codes is chaotic, so a 0.42 logit
nudge flips one choice and every subsequent frame differs. Divergence is expected;
it does not measure quality.

**Mel and waveform distance are equally useless here.** Once the code sequence
differs the audio differs everywhere, and two different-but-valid renditions of the
same sentence sit far apart in mel space. The measured mel distances (1.34–2.50) do
not even order the variants correctly — q4_1 scores *better* than q5_0.

So the metrics that carry signal are the ones taken **under teacher forcing**, with
both models fed the reference's codes so divergence cannot compound:

- **top-1 / top-5 agreement** on the semantic distribution,
- **KL(ref‖alt)**, and
- **margin** = `p_alt(ref's argmax) / max p_alt`. This is the one that settles it: at
  0.998, the 4.2% of steps where q8_0 picks a different token are steps where its own
  probability at the reference's choice is 99.8% of its maximum. Those are coin-flips
  between near-ties, not confident errors.

And then intelligibility, which is what a listener actually notices: **WhisperX
transcribes q8_0 word-for-word identically to the f32 reference** (WER 0.000).
q4_0's transcript is audibly truncated mid-phrase ("Pack my-").

## Caveats worth carrying

- `qroundtrip` quantizes with candle and dequantizes back to f32, then runs the
  **reference torch model** with those weights. Candle's Metal matvec instead
  dequantizes blocks on the fly and accumulates in f32, so this is a proxy, not
  bit-exact to inference. The difference is accumulation order — second order next
  to the quantization error itself — but the port should re-run this gate against
  its own output once it exists.
- One prompt, 96 frames (4.46 s), greedy. Enough to separate q8_0 from q4_0
  decisively; not enough to certify q8_0 across voices, languages, or sampled
  (non-greedy) decoding, where RAS and top-p change which near-ties get taken.
- The K-quants are unavailable at all: Q4K/Q5K/Q6K need `k` divisible by 256 and
  `dim` is 896. Only block-32 types apply, which is why q5_0 rather than q5_K is the
  middle option.
- Nothing here tests the **codec** under quantization. It stayed at f32 throughout.
  The codec is a deterministic decoder rather than a sampler, so it should tolerate
  quantization far better — but that is an assumption, not a measurement.

## Where the speed actually comes from

Worth stating plainly, because it changes how much the quantization choice matters:
quantization is 3.35×, but **batching at ≥4 is up to 11.95× per sequence** and
carries no numerics risk at all beyond accumulation order. The three exact AR levers
(narrow KV, GQA by query reshape, logit slice) are another 7.57× and are also free.

Ranked by speed-per-unit-of-risk, q8_0 is third. It is worth taking — it is the
difference between AR RTF 0.88 and 0.26 at batch 1 — but the port should not be
tempted into q4 to chase a further 1.4×, because the free levers are larger.

See [ar-loop.md](ar-loop.md) for those measurements.
