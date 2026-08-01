//! CosyVoice3 (`Fun-CosyVoice3-0.5B`) in Rust: text in, 24 kHz speech out.
//!
//! Three models in series, 995 M parameters, no ONNX and no Python at runtime:
//!
//! | stage | params | what it is |
//! |---|---|---|
//! | [`llm`] | 642 M | Qwen2-0.5B trunk with a 6761-row speech head, RAS sampling |
//! | [`flow`] | 332 M | token embedding, speaker affine, look-ahead conv, then a 22-block DiT under a 10-step Euler solve with classifier-free guidance |
//! | [`hift`] | 21 M | F0 predictor, NSF harmonic source, upsampling decoder, iSTFT |
//!
//! Conditioning — speaker embedding, speech tokens, prompt mel, prompt text tokens — is
//! precomputed into a voice asset by `oracle-cosy/export_voice.py`, which is what keeps
//! 997 MB of ONNX (a whisper-based speech tokenizer and a speaker-embedding model) out of
//! this crate entirely.
//!
//! # Validation
//!
//! `cargo run -p cosy --release --bin cosy-validate` checks all three stages against
//! `fixtures-cosy/oracle.safetensors`. The two rows that matter most:
//!
//! - **LLM: 105/105 teacher-forced argmax ids identical**, logits at rel 3.1e-6. Teacher
//!   forcing rather than greedy decoding, because greedy decoding on this model
//!   degenerates to one repeated token within two steps — and a constant is something a
//!   *wrong* implementation can also produce.
//! - **Vocoder: the decoder is exact** (rel 1.6e-4 from the reference's own mel and
//!   source), and the harmonic source matches to 1.93e-5, which is precisely the
//!   reference's own f32-versus-f64 error on that tensor.
//!
//! The DiT's tolerances are derived from a measured precision floor rather than chosen:
//! running the reference decoder in f64 and comparing against its own f32 shows it
//! amplifies rounding error ~500x over 22 blocks, reaching rel 5.6e-4 at the last block.
//! This port sits at 6.7e-4 there, so it is *at* the floor rather than merely near it.
//!
//! # Eight traps, each of which produces a port that runs and is wrong
//!
//! Six were found by reading, two by measuring. All are silent.
//!
//! 1. **RoPE reaches head 0 only.** The reference applies `apply_rotary_pos_emb` to the
//!    pre-reshape `[b, n, 1024]` projection, and `x_transformers` does *partial* rotary:
//!    `rot_dim` is 64, so channels 0..63 rotate and 64..1023 pass through. Heads 1..15 get
//!    no positional information. Applying RoPE per-head to all 16 is what any reasonable
//!    implementation would do, and it is a different model. See [`flow`].
//! 2. **The two engines' RoPE conventions are opposite.** HF's Qwen2 uses `rotate_half`
//!    (half-split); Audio8's Fish-Speech weights use `torch.polar` (interleaved pairs).
//!    Same geometry, same shapes, both run. Getting it backwards here put the hidden state
//!    off by rel 0.78 — caught only because the fixture existed. See [`llm`].
//! 3. **The flow's initial noise is a fixed tensor, not a draw.** `CausalConditionalCFM`
//!    seeds torch at construction and slices the same `randn([1, 80, 15000])` every call,
//!    so output is deterministic given the tokens. Ships as an asset.
//! 4. **`<|endofprompt|>` is required and nothing in the frontend adds it.** The *service*
//!    prepends `"You are a helpful assistant.<|endofprompt|>"`. A voice asset built from
//!    the raw transcript makes the LLM refuse the prompt outright.
//! 5. **The upstream safetensors are the wrong weights.**
//!    `CosyVoice-BlankEN/model.safetensors` sits beside the checkpoints with exactly the
//!    Qwen2 key names, but 0 of its 290 tensors match `llm.pt`'s (max relative difference
//!    1.82). It is the base initialisation; `llm.pt` is the fine-tune.
//! 6. **The tokenizer needs ~280 special tokens added at construction.** The checkpoint
//!    directory's `added_tokens_decoder` lists three. Serialising it as-is makes
//!    `<|endofprompt|>` tokenize as nine pieces of literal text.
//! 7. **The leaky-ReLU before `conv_post` has slope 0.01, not 0.1** — a bare
//!    `F.leaky_relu(x)` taking torch's default while every sibling passes `lrelu_slope`.
//! 8. **The NSF noise is not in the checkpoint**, and it is not negligible: zeroing it
//!    moves the waveform by 0.164 against a signal of rms 0.078. It is a plain
//!    `torch.rand` attribute, reproducible only because the yaml seeds torch just before
//!    construction. See [`hift`], which also documents the *ninth* thing — that the
//!    reference's phase accumulation is numerically degenerate in f32, reaching 1.7e7
//!    radians where one ulp is a full radian.
//!
//! Two more things that look like traps and are not, recorded so nobody spends time on
//! them: `SineGen2.rand_ini` is dead code (the downsample discards it — measured at
//! exactly 0.0), and `CausalConvRNNF0Predictor` contains no recurrence despite its name.
//!
//! # What is not ported
//!
//! Streaming, and the reference's text normalisation (an FST-based normaliser plus number
//! spell-out). Both are stated in [`engine::capabilities`]'s documentation rather than
//! left to be discovered. See `docs/porting/cosyvoice.md`.

pub mod cfg;
pub mod engine;
pub mod flow;
pub mod hift;
pub mod llm;
pub mod sample;
pub mod stft;

pub use engine::{capabilities, CosyVoiceEngine, ID};
