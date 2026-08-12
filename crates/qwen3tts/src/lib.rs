//! Qwen3-TTS (`Qwen3-TTS-12Hz-1.7B-Base`): text in, 24 kHz speech out.
//!
//! | stage | what it is |
//! |---|---|
//! | [`talker`] | Qwen3 trunk, 28 layers at 2048, QK-norm; emits codebook 0 |
//! | [`predictor`] | 5-layer depth transformer at 1024; emits codebooks 1..15 |
//! | [`codec`] | RVQ dequant, 8-layer windowed transformer, causal conv upsampler |
//!
//! No diffusion and no mel anywhere. Conditioning is precomputed into a voice asset by
//! `references/qwen3tts/export_voice.py`, keeping the checkpoint's ECAPA-TDNN speaker
//! encoder out of this crate.
//!
//! Validated against `fixtures/qwen3tts/oracle.safetensors`: 63 rows, 0 failures. Argmax
//! codebook 0 identical, predictor 15/15 identical, the loop update and the quantizer exact,
//! waveform rel 4.0e-6. RTF 0.863 on this Mac at q8_0 (talker 0.695, codec 0.168).
//!
//! # Why a third engine
//!
//! It drops CosyVoice's dominant stage: the flow decoder is 65-68% of that engine's RTF and
//! this model has no equivalent. It also runs at 12.5 Hz against 25 Hz. Costs going the
//! other way: 3.4x the trunk parameters, and 15 predictor steps per frame (trap 2).
//!
//! The hard limit is **ten languages, closed list** ([`cfg::talker::LANGUAGES`]) — text
//! outside it has no faithful prefill, so CosyVoice stays the default.
//!
//! # Traps
//!
//! All silent. Found by reading the reference and auditing shapes before writing any port.
//! Full detail with reference line numbers in `docs/reference.md#porting-traps`.
//!
//! 1. **M-RoPE is configured and unused.** `get_rope_index` gives all three sections the
//!    same `cumsum(mask) - 1`, so `apply_interleaved_rope` is identity and the result is
//!    plain half-split RoPE at theta 1e6. `position_id_per_seconds` is dead too.
//! 2. **The "MTP block" is 15 sequential AR steps, not one parallel pass.** Per frame: one
//!    28-layer talker step plus 15 five-layer predictor steps.
//! 3. **Text is consumed one token per audio frame.** Each step's input is the previous
//!    frame's 16 codebook embeddings summed, plus the next text token. So audio that ends
//!    early leaves unspoken text; the shortfall is exact, not a ratio heuristic.
//! 4. **Codebooks are stored divided:** `embedding_sum / cluster_usage.clamp(1e-5)`. Using
//!    `embedding_sum` directly is a per-row scale error that still produces audio.
//! 5. **`semantic_codebook_size: 4096` is false** — zero uses in the reference, every
//!    codebook tensor is `[2048, 256]`. The shape audit caught this on its first run. The
//!    *encoder* does have 31 acoustic quantizers with only 16 valid.
//! 6. **The codec decoder is not only a ConvNet:** 8 layers at 512 with a live 72-frame
//!    sliding window, projected in and out. The talker's window is null; this one is not.
//! 7. **`HEADS * HEAD_DIM != DIM` in the predictor** (2048 vs 1024). Deriving head_dim from
//!    hidden size gives 64 and every shape still divides.
//! 8. **The 15 codebook embedding tables are at talker width** `[2048, 2048]`, narrowed by
//!    `small_to_mtp_projection`. Dual-use: the talker sums them for its own next input.
//! 9. **No attention biases, but `text_projection` has them.** One rule for the whole
//!    checkpoint drops two bias vectors, which passes every shape check.
//!
//! # Validation
//!
//! `cargo run -p qwen3tts --release --bin qwen3tts-validate`. Tier 1 audits every [`cfg`]
//! constant against the checkpoint header (cheap — header only, so it runs even mid-download).
//! Tier 2 compares each stage against the fixtures, on **CPU at f32**: the engine defaults to
//! q8_0 because an f32 talker is 6.3 GB and fails Metal allocation on a 16 GB machine.
//!
//! **q8_0 is the default weight format here, unlike the other two engines.** f32 measured
//! 1994 ms/frame against q8_0's 52 ms — memory pressure, not arithmetic. See
//! `docs/reference.md#porting-traps`.

pub mod cfg;
pub mod codec;
pub mod engine;
pub mod qwen3;
pub mod talker;

pub use engine::{capabilities, Qwen3TtsEngine, ID};
