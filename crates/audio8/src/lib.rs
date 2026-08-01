//! Audio8-TTS in Rust: text -> 44.1 kHz waveform, single process, bounded memory.
//!
//! Layout mirrors the reference (`references/audio8/weights/modeling_arktts*.py`) closely enough
//! that the two can be diffed by eye, because every divergence has to be explainable
//! against the fixtures in `fixtures/`.
//!
//! - [`nn`] — layers shared by both halves: causal convs, snake, norms, RoPE.
//! - [`codec`] — the RVQ + transformer + upsampling decoder, codes to audio.
//! - [`ar`] — the DualAR model, prompt to codes, with the measured levers built in.
//! - [`sample`] — Gumbel-max, the legacy top-k/top-p ordering, and RAS.
//! - [`prompt`] — tokenizer and prompt construction, including reference codes.
//! - [`engine`] — the adapter that makes this selectable via `tts_core::Engine`.
//!
//! What is deliberately absent: the codec *encoder*. Cloning needs a reference clip
//! turned into codes, but that is a one-off offline step, so the codes ship as an
//! asset (`fixtures/audio8/default_voice_codes.safetensors`) and 126 encoder tensors stay
//! out of the binary. See `docs/status.md`.
//!
//! Segments decode in batches ([`ar::Model::generate_batch`], `--set max_batch=<n>`).
//! Two things about that are worth knowing before touching it: prompts are
//! **right-aligned**, which is exact because attention scores depend only on relative
//! position — but only in real arithmetic, since the bf16-rounded RoPE table satisfies
//! `R(p)^T R(j) = R(p - j)` to about 4e-3. And the per-sequence gain is **~1.9x, not the
//! 11.95x** a layer benchmark projected: a decode step is 64 layer-passes plus ten host
//! synchronisations plus a sampler that runs once per sequence, and only the projections
//! scale freely with batch. `docs/performance/ar-loop.md` has the numbers and the two measurements
//! that turned out to be measuring something else.

pub mod ar;
pub mod codec;
pub mod engine;
pub mod prompt;
pub mod sample;

/// Re-exported from `tts-core`: segmentation and WAV writing are engine-neutral and
/// live there now so a second engine does not duplicate them.
pub use tts_core::{text, wav};

/// Re-exported from `tts-nn`: the convolutions, activations, norms, RoPE tables and
/// quantized projection this port established, now shared with the CosyVoice engine.
/// Kept under the original name so `crate::nn::` reads the same as it always did.
pub use tts_nn as nn;

/// Model geometry, from `references/audio8/weights/config.json`. Hard-coded rather than parsed:
/// the port is validated against one pinned revision
/// (`1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`), and a config field silently changing
/// shape underneath it would be worse than a compile-time constant.
pub mod cfg {
    pub const DIM: usize = 896;
    pub const N_HEAD: usize = 14;
    pub const N_KV: usize = 2;
    pub const HEAD_DIM: usize = 64;
    pub const FFN: usize = 4864;
    pub const N_LAYER: usize = 24;
    pub const N_FAST_LAYER: usize = 4;
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_BASE: f64 = 1e6;
    pub const MAX_SEQ_LEN: usize = 2048;

    pub const VOCAB: usize = 155776;
    pub const NUM_CODEBOOKS: usize = 10;
    pub const CODEBOOK_SIZE: usize = 4096;
    /// The residual codebooks are 1024 entries, not 4096 — the clamp in
    /// `ArkttsDownsampleQuantizer.decode` is what enforces it.
    pub const RESIDUAL_CODEBOOK_SIZE: usize = 1024;
    pub const SEMANTIC_BEGIN_ID: u32 = 151678;
    pub const SEMANTIC_END_ID: u32 = 155773;
    pub const EOS_TOKEN_ID: u32 = 151645;
    pub const PAD_TOKEN_ID: u32 = 151643;
    /// 4096 semantic ids plus eos: the only rows the semantic mask leaves finite.
    pub const REACHABLE: usize = 4097;

    pub const RAS_WINDOW_SIZE: usize = 10;
    pub const RAS_TOP_P: f32 = 0.9;
    pub const RAS_TEMPERATURE: f32 = 1.0;

    pub const SAMPLE_RATE: usize = 44100;
    pub const CODEC_FRAME_SIZE: usize = 2048;
    /// Codec transformer blocks use a different epsilon from the AR stack.
    pub const CODEC_NORM_EPS: f32 = 1e-5;
    pub const CODEC_ROPE_BASE: f64 = 10000.0;
    pub const CODEC_POST_N_LAYER: usize = 8;
    pub const CODEC_POST_N_HEAD: usize = 16;
    pub const CODEC_POST_N_KV: usize = 8;
    pub const CODEC_POST_FFN: usize = 1216;
    pub const CODEC_WINDOW: usize = 128;
    pub const CODEC_DIM: usize = 1024;
    pub const CODEBOOK_DIM: usize = 8;

    pub fn frame_rate() -> f64 {
        SAMPLE_RATE as f64 / CODEC_FRAME_SIZE as f64
    }
}
