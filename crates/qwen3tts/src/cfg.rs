//! Geometry for `Qwen3-TTS-12Hz-1.7B-Base`, hard-coded like the other engines' and checked
//! against the checkpoint's tensor shapes by `qwen3tts-validate`.
//!
//! This config carries an unusual amount of **dead configuration** — fields that are
//! present, plausible, and never read. Each is flagged where it appears.

// ------------------------------------------------------------------ shared

pub const SAMPLE_RATE: usize = 24_000;

/// Frames per second — half CosyVoice's 25 Hz. Not an integer, hence not `usize`.
pub const FRAME_RATE: f64 = 12.5;

/// `decode_upsample_rate`: 2*2 * 8*5*4*3. Asserted against the codec's stacks.
pub const SAMPLES_PER_FRAME: usize = 1920;

/// `num_code_groups`. The talker predicts codebook 0, the predictor the other 15.
pub const CODE_GROUPS: usize = 16;

// ------------------------------------------------------------------ talker

/// The 28-layer trunk that turns text into codebook-0 codes: `talker_config`.
pub mod talker {
    /// Qwen3-1.7B-shaped. `HEADS * HEAD_DIM == DIM` here, which the predictor breaks
    /// deliberately — do not rely on it structurally.
    pub const DIM: usize = 2048;
    pub const LAYERS: usize = 28;
    pub const HEADS: usize = 16;
    pub const N_KV: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const FFN: usize = 6144;
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_BASE: f64 = 1_000_000.0;
    /// How many query heads share one KV head.
    pub const GQA: usize = HEADS / N_KV;

    /// **No attention biases**, unlike CosyVoice's Qwen2 which biases q/k/v. Carrying that
    /// loader over asks for tensors that do not exist — a loud failure, the good case.
    pub const ATTENTION_BIAS: bool = false;

    /// **QK-norm**: RMS over the head dim only, *before* RoPE. Two `[HEAD_DIM]` weights per
    /// layer. Neither existing engine has this, so it is new code.
    pub const QK_NORM: bool = true;

    /// Text is embedded by `talker.model.text_embedding` and projected into `DIM` by
    /// `talker.text_projection` — `linear_fc1`, SiLU, `linear_fc2`, both `[2048, 2048]`.
    pub const TEXT_VOCAB: usize = 151_936;
    pub const TEXT_DIM: usize = 2048;

    /// **`text_projection` is biased even though attention is not** (`ResizeMLP` is built
    /// `bias=True`). One rule for the whole checkpoint drops two bias vectors, and a dropped
    /// bias passes every shape check.
    pub const TEXT_PROJECTION_BIAS: bool = true;

    /// Codec-side vocabulary of the talker's own embedding table and `codec_head`: real
    /// codes, then a block of control ids.
    pub const VOCAB: usize = 3072;
    /// Codes the talker may emit: `generate` suppresses the top 1024 of `VOCAB` except
    /// `CODEC_EOS`, so the live range is `0..2048` plus that id.
    pub const CODES: usize = 2048;

    // Control ids, from `talker_config`. All sit above `CODES`.
    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
    pub const CODEC_EOS: u32 = 2150;
    pub const CODEC_THINK: u32 = 2154;
    pub const CODEC_NOTHINK: u32 = 2155;
    pub const CODEC_THINK_BOS: u32 = 2156;
    pub const CODEC_THINK_EOS: u32 = 2157;

    /// Text-side control ids, from the top-level config rather than `talker_config`.
    pub const TTS_PAD: u32 = 151_671;
    pub const TTS_BOS: u32 = 151_672;
    pub const TTS_EOS: u32 = 151_673;
    pub const IM_START: u32 = 151_644;
    pub const IM_END: u32 = 151_645;
    pub const ASSISTANT: u32 = 77_091;

    /// The language tag prefilled into the codec stream. `None` takes the `CODEC_NOTHINK`
    /// path, which is what `language: "auto"` does.
    pub fn language_id(name: &str) -> Option<u32> {
        Some(match name {
            "english" => 2050,
            "german" => 2053,
            "spanish" => 2054,
            "chinese" => 2055,
            "japanese" => 2058,
            "french" => 2061,
            "korean" => 2064,
            "russian" => 2069,
            "italian" => 2070,
            "portuguese" => 2071,
            _ => return None,
        })
    }

    /// Ten, and closed: no `language_id` exists for anything else, so text outside this set
    /// has no faithful prefill. Notably absent: Swahili and every other African language.
    pub const LANGUAGES: &[&str] = &[
        "english",
        "german",
        "spanish",
        "chinese",
        "japanese",
        "french",
        "korean",
        "russian",
        "italian",
        "portuguese",
    ];

    /// The reference's talker defaults, from `generate`'s signature.
    pub const TOP_K: usize = 50;
    pub const TOP_P: f32 = 1.0;
    pub const TEMPERATURE: f32 = 0.9;
    pub const REPETITION_PENALTY: f32 = 1.05;
    /// `min_new_tokens=2`, `max_new_tokens=4096`.
    pub const MIN_NEW_TOKENS: usize = 2;
    pub const MAX_NEW_TOKENS: usize = 4096;

    // ---------------------------------------------------------- dead configuration

    /// `rope_scaling.mrope_section`, deliberately **not** implemented.
    ///
    /// `get_rope_index` builds all three sections from the same `cumsum(mask) - 1`, so
    /// `apply_interleaved_rope` returns `x[0]` unchanged and what reaches `rotate_half` is
    /// ordinary half-split RoPE. Kept as a constant because the config is the first thing a
    /// reader checks. Trap 1 in `docs/reference.md#porting-traps`.
    pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];

    /// Dead: no uses in the reference, and 13 is not this model's frame rate anyway.
    pub const POSITION_ID_PER_SECONDS: usize = 13;

    /// `sliding_window: null` with `use_sliding_window: false`. Full causal attention.
    pub const SLIDING_WINDOW: Option<usize> = None;
}

// ------------------------------------------------------------------ code predictor

/// The depth transformer for codebooks 1..15: `code_predictor_config`.
///
/// Called the "sub-talker" upstream and an "MTP block" in the report; both oversell it as
/// parallel. `forward` calls `code_predictor.generate(max_new_tokens=num_code_groups - 1)` —
/// **15 sequential AR steps per frame**, each a full 5-layer pass. The most important cost
/// fact about this engine. Trap 2.
pub mod predictor {
    /// Half the talker's width; `small_to_mtp_projection` resizes into it.
    pub const DIM: usize = 1024;
    pub const LAYERS: usize = 5;
    pub const HEADS: usize = 16;
    pub const N_KV: usize = 8;
    /// **`HEADS * HEAD_DIM` is 2048, not `DIM`** — `q_proj` is `[2048, 1024]`. Deriving
    /// `head_dim = DIM / HEADS` gives 64, every shape still divides, and the model runs.
    pub const HEAD_DIM: usize = 128;
    pub const FFN: usize = 3072;
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_BASE: f64 = 1_000_000.0;
    pub const GQA: usize = HEADS / N_KV;
    pub const QK_NORM: bool = true;
    pub const ATTENTION_BIAS: bool = false;

    /// 15 heads, not 16: codebook 0 comes from the talker's `codec_head`. Each
    /// `lm_head.{i}.weight` is `[VOCAB, DIM]`.
    pub const VOCAB: usize = 2048;
    pub const HEADS_OUT: usize = super::CODE_GROUPS - 1;

    /// **The 15 embedding tables are `[VOCAB, 2048]` — the *talker's* width.**
    ///
    /// Dual-use: the predictor reads them through `small_to_mtp_projection`, but the talker
    /// also sums all 16 of a frame's embeddings at full width for its own next input
    /// (`codec_hiddens.sum(1)`). Sizing them at [`DIM`] is a load-time shape error.
    pub const EMBED_DIM: usize = super::talker::DIM;

    /// The reference's sub-talker defaults.
    pub const TOP_K: usize = 50;
    pub const TOP_P: f32 = 1.0;
    pub const TEMPERATURE: f32 = 0.9;

    /// `layer_types` lists five `full_attention` entries and `sliding_window` is null.
    pub const SLIDING_WINDOW: Option<usize> = None;
}

// ------------------------------------------------------------------ codec decoder

/// The RVQ decoder: codes to 24 kHz audio, from `decoder_config`.
///
/// The report calls this a "lightweight causal ConvNet". It does replace DiT + BigVGAN, but
/// an **8-layer sliding-window transformer** sits between quantizer and convolutions —
/// small at hidden 512, but not a ConvNet and not free.
pub mod codec {
    /// Codebooks, split 1 semantic + 15 acoustic by `SplitResidualVectorQuantizer`.
    pub const QUANTIZERS: usize = super::CODE_GROUPS;
    pub const SEMANTIC_QUANTIZERS: usize = 1;
    pub const ACOUSTIC_QUANTIZERS: usize = QUANTIZERS - SEMANTIC_QUANTIZERS;

    /// Entries per codebook — **all sixteen, semantic included**.
    ///
    /// The config's `semantic_codebook_size: 4096` has zero uses in the reference, and every
    /// codebook tensor in the checkpoint is `[2048, 256]`. The gate asserts the semantic
    /// table matches the acoustic ones rather than the config.
    pub const CODEBOOK: usize = 2048;

    /// Each codebook's vector width.
    ///
    /// Both stacks are built `dimension = codebook_dim // 2, force_projection=True`: a 1x1
    /// conv narrows 512 -> 256, lookup at 256, then widens back. `VectorQuantization`'s own
    /// `project_out` is an `Identity` (codebook_dim == dim), so it has no tensors.
    pub const QUANTIZER_INNER: usize = 256;
    /// `codebook_dim`: the width the two quantizer stacks are summed at, and what `pre_conv`
    /// consumes.
    pub const CODEBOOK_DIM: usize = 512;

    /// **The codebook is stored divided.** `EuclideanCodebook` keeps `embedding_sum` and
    /// `cluster_usage`, and `decode` looks up
    /// `embedding_sum / cluster_usage.clamp(min=1e-5)[:, None]`. These are training EMA
    /// accumulators, so `cluster_usage` is not all-ones: using `embedding_sum` directly is a
    /// per-row scale error on every code that still produces audio. Fold at load, as
    /// `tts_nn::Weights::get_weight_norm` does for weight-norm.
    pub const CLUSTER_USAGE_EPSILON: f64 = 1e-5;

    /// The encoder is trained deeper than used: 1 semantic + 31 acoustic in the checkpoint,
    /// `encoder_valid_num_quantizers: 16`. This crate never runs it, so it matters in one
    /// place — a voice asset built from all 32 is silently wrong from codebook 16 on.
    pub const ENCODER_QUANTIZERS: usize = 32;
    pub const ENCODER_VALID_QUANTIZERS: usize = 16;

    /// `latent_dim`: what `pre_conv` widens the quantizer output to, and the width the
    /// pre-transformer and the upsample stages run at.
    pub const LATENT_DIM: usize = 1024;
    /// `pre_conv` is a causal conv with this kernel.
    pub const PRE_CONV_KERNEL: usize = 3;

    /// 8 layers at hidden 512 with a 72-frame sliding window.
    ///
    /// Settled by the shape audit, not by reading: `pre_conv` emits `LATENT_DIM` but the
    /// checkpoint has `input_proj [512, 1024]` and `output_proj [1024, 512]`, so the
    /// transformer runs at 512 and is projected in and out.
    pub const TF_LAYERS: usize = 8;
    pub const TF_DIM: usize = 512;
    pub const TF_HEADS: usize = 16;
    pub const TF_N_KV: usize = 16;
    pub const TF_HEAD_DIM: usize = 64;
    pub const TF_FFN: usize = 1024;
    pub const TF_NORM_EPS: f32 = 1e-5;
    pub const TF_ROPE_BASE: f64 = 10_000.0;
    /// **Live**, unlike the talker's null window. 72 frames is 5.76 s at 12.5 Hz.
    pub const TF_SLIDING_WINDOW: usize = 72;
    /// Every residual branch is scaled by a learned per-channel weight. A missed
    /// `LayerScale` reads as a plain residual and runs.
    pub const LAYER_SCALE_INIT: f64 = 0.01;

    /// Two stages of causal transposed conv (stride = kernel = factor) plus a ConvNeXt
    /// block, both at `LATENT_DIM`.
    pub const UPSAMPLING_RATIOS: [usize; 2] = [2, 2];

    /// Waveform stack: causal conv `LATENT_DIM -> DECODER_DIM` (k=7), one decoder block per
    /// rate halving channels, then `SnakeBeta` and a causal conv to 1 channel (k=7).
    pub const DECODER_DIM: usize = 1536;
    pub const UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
    pub const OUT_CONV_KERNEL: usize = 7;

    /// Channels after each decoder block: 1536/2, /4, /8, /16.
    pub const fn stage_channels(i: usize) -> usize {
        DECODER_DIM >> (i + 1)
    }

    /// The width `SnakeBeta` and the final conv see: `DECODER_DIM >> 4` = 96.
    pub const OUT_CHANNELS: usize = DECODER_DIM >> UPSAMPLE_RATES.len();

    /// `chunked_decode`'s defaults; the context's audio is discarded after.
    ///
    /// Worth taking over one whole-utterance `forward`: candle's Metal device pools buffers
    /// by size, so a decoder called once pays every allocation cold. CosyVoice's
    /// `hift.forward` is called once and took **none** of the im2col win that Audio8's
    /// seven-calls-per-utterance codec took in full.
    pub const CHUNK_FRAMES: usize = 300;
    pub const CHUNK_LEFT_CONTEXT: usize = 25;

    /// A hard clamp, not Audio8's `AUDIO_LIMIT` scaling.
    pub const CLAMP: f32 = 1.0;
}

// ------------------------------------------------------------------ speaker encoder

/// The ECAPA-TDNN speaker encoder: `speaker_encoder_config`.
///
/// In the checkpoint, unlike CosyVoice's separate `campplus.onnx`, but still not in this
/// crate — `export_voice.py` runs it once and the embedding ships in the voice asset.
pub mod speaker {
    /// The one place 0.6B and 1.7B differ structurally: 1024 there, 2048 here, matching
    /// each talker's `hidden_size` — the embedding is one position in the codec stream.
    pub const ENC_DIM: usize = 2048;
    pub const SAMPLE_RATE: usize = 24_000;
}
