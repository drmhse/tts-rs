# Architecture: two models, one interface

The requirement is that a caller picks the engine at request time. That is a small API and a
large set of decisions about where model-specific knowledge is allowed to live.

```
tts-cli          `tts engines` / `tts voice` / `tts speak --engine ...`
   |
tts-engines      the registry: the ONLY place that knows which engines exist
   |        \
  audio8  cosyvoice  qwen3tts   engines: one crate each, no knowledge of each other
   |    \    /    |
   |     tts-nn   |        shared model machinery: convs, activations, norms, RoPE, Proj
   \      /       /
    tts-core             Engine trait, Voice assets, segmentation, WAV, the PRNG

tts-bench                the measurement harness, used by tts-probe and cosyvoice-bench
```

Five rules, each of which earned its place.

**`tts-core` knows nothing about any model.** No candle model code, no weights, no engine ids
beyond what a `Capabilities` struct carries. If a type in `tts-core` needs to know whether it
is talking to Audio8, the abstraction is wrong.

**The registry is one file.** Adding an engine touches `tts-engines/src/lib.rs` and nothing
else. `tts-core` cannot depend on its implementations (that would be circular) and the CLI
should not enumerate them (a service would have to duplicate the list).

**Engines do not know about each other.** `audio8` and `cosyvoice` both depend on `tts-core` and
`tts-nn`, and neither depends on the other. This is why `tts-nn` exists: when the CosyVoice
port needed causal convolutions, snake activations, RoPE tables and the quantized projection
wrapper, the choice was to lift them out of `audio8::nn` rather than let `cosyvoice` reach into `audio8`.
`audio8` re-exports the crate as `audio8::nn` so its own call sites read unchanged.

**Shared does not mean identical.** `tts-nn` carries *both* RoPE conventions
(`rope_table` bf16-rounded and interleaved for Audio8, `rope_table_f32` for CosyVoice's DiT)
and both GELU variants, because the two models genuinely differ and a single "sensible"
choice would silently be wrong for one of them. Using Audio8's convention in the CosyVoice
LLM put its hidden state off by rel 0.78 — the sort of error that runs fine and produces
speech. Where a shared helper has a fast form and a readable form, both stay:
`layer_norm_plain` is the reference that `LayerNormPlain`'s fused kernel is checked against.

**Unavailable engines are visible.** `Capabilities` carries `available` and a `reason`, so an
engine can be registered before it works. All three engines are now available; the mechanism
stays, because the alternative — hiding an engine, or falling back to another — hands back
audio in the wrong voice from the wrong model with no indication anything unusual happened.

## The decision that makes a second and third engine tractable: voice assets

Both models clone from a reference clip, and in both cases turning audio into conditioning
needs machinery the runtime should not carry.

| engine | the clip must become | in-process cost avoided |
|---|---|---|
| `audio8` | `[10, N]` RVQ codes | the codec **encoder** — 126 tensors `convert_codec.py` drops |
| `cosyvoice` | speaker embedding, speech tokens, prompt mel, prompt text tokens | `campplus.onnx` (28 MB) + `speech_tokenizer_v3.onnx` (969 MB), plus an ONNX runtime |
| `qwen3tts` | x-vector, `[T, 16]` RVQ codes, sliced transcript tokens | an ECAPA-TDNN speaker encoder and a Mimi-style RVQ **encoder**, both inside the talker checkpoint |

None of it depends on the text being spoken. So it happens once, offline, in Python, and
ships as a directory:

```
voices/cosy-default-cosyvoice/
  voice.json          { engine, name, text, prompt_text, seconds, notes }
  voice.safetensors    engine-specific tensors
```

`Voice::load` reads it; the `engine` field is checked and a mismatch is a hard error:

```
$ tts speak --engine audio8 --voice voices/cosy-default-cosyvoice --text hi --out x.wav
Error: voice `cosy-default-cosyvoice` was built for engine `cosyvoice`, but `audio8` was
requested — voice assets are not interchangeable between engines
```

That check matters because the tensors are not merely differently shaped, they mean different
things, and a silent fallback would produce a plausible voice that is not the one asked for.
`cosyvoice` adds a second check of its own: the prompt mel must be exactly
`speech_tokens * TOKEN_MEL_RATIO` frames, because the flow decoder holds each token for two
mel frames and a mismatch would misalign the prompt against its conditioning without
erroring.

**Voice assets load on the host, and the engine pulls them onto its device.** Not an
implementation detail — `Voice::load` deliberately takes no `Device`. A caller loads a voice
*before* it loads an engine, so a device passed there would be a second handle to the same
GPU, and candle compares device identity rather than hardware: mixing tensors from two
handles fails with `device mismatch in matmul` naming the same `gpu_id` twice, and
`to_device` between them is not implemented at all. Both were hit. Keeping assets on the host
and having engines call `Voice::get_on` makes the whole class of error unreachable.

**The consequence: the Rust runtime for both engines is safetensors-only.** No ONNX, no
encoders, ~1.1 GB of model surface removed from the binary across the two engines. ONNX
survives only in `references/cosyvoice/export_voice.py`, where it runs in a venv that already has
onnxruntime and its speed is irrelevant — the right place for it, given `rejected/onnx.md`
measured ORT as the slowest runtime on this machine.

**The cost, stated plainly:** cloning from arbitrary audio at request time is not possible
in-process. Adding a voice is a Python step. For a service with a fixed voice set that is the
right trade; for one that clones per request it is not, and fixing it means porting a whisper
encoder and an FSQ quantizer to Candle.

## What `Capabilities` is for

It is deliberately blunt about the things engines genuinely differ on, because a caller
choosing between a 44.1 kHz model and a 24 kHz one needs to know before it commits:

```
engine       sample rate   cloning   stream     state  weights
audio8         44100 Hz     asset       no     ready  f32, q8_0, q5_0, q4_1, q4_0
cosyvoice      24000 Hz     asset       no     ready  f32, q8_0, q5_0, q4_1, q4_0
```

`quantization` is listed per engine because the constraints differ — neither model can use
the K-quants at all, since those need `k` divisible by 256 and both are 896 wide, so listing
"q8_0" generically would be a lie by omission. What the table cannot express is *where*
quantization helps: candle takes a dedicated matrix-vector kernel only when `dim(-2) == 1`,
so quantizing a decode loop is a 3.35x win while quantizing the DiT — which only ever runs on
full sequences — buys much less. `cosyvoice` therefore quantizes the LLM and leaves the flow
decoder and vocoder dense, and says so in `tts_nn::Proj`'s documentation rather than leaving
it to be rediscovered.

## Stage timings are a named list, not fixed fields

`Stats::stages` is `Vec<(&'static str, f64)>`. Audio8 reports `ar` / `codec`; CosyVoice
reports `llm` / `flow` / `vocoder`. Forcing the second into the first's shape would flatten
the distinction that matters most for optimization — and it is the distinction this project
has twice been wrong about, so the CLI prints whatever the engine names without knowing which
engine it is talking to.

## Where the segment loop lives

In the engine, not the CLI. Segmentation itself is in `tts-core::text` because it is a
property of the request — every model has a context limit and every model degrades holding
prosody over too long a span — but *iterating* segments, timing the stages and stitching with
silence is per-engine.

This is also why the standalone `audio8` binary was deleted (it lives in `docs/rejected/attic/`): it and the
CLI were two code paths to the same synthesis.

## Adding an engine

1. New crate depending on `tts-core` and `tts-nn`.
2. `capabilities()` returning `available: false` with a reason — commit that first, so the
   identifier exists before the implementation does.
3. Register in `tts-engines`.
4. Phase A **before any Rust**: convert weights to safetensors, export a voice asset, dump
   **per-stage** fixtures. This is the step that made Audio8's codec validate at 2.8e-6 and
   its greedy generation come out bit-identical first try, and it is what localised
   CosyVoice's reversed RoPE convention to one line instead of "the audio sounds wrong".
   Per-stage matters: a whole-pipeline mismatch says nothing about which of three models is
   at fault.
5. Implement stage by stage, checking each against its fixture before starting the next.
6. Flip `available` to true.
7. Measure through `tts_bench::Harness`, then optimize — and **verify each variant is
   correct before believing its timing**. A 1.25x on the CosyVoice DiT turned out to be
   computing the wrong function; see `porting/cosyvoice.md`. See `benchmarking.md` for
   what happens to the numbers themselves without the harness.
