//! Geometry, hard-coded against `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`.
//!
//! Hard-coded rather than parsed, for the same reason Audio8's is: the yaml is a
//! hyperpyyaml document that constructs Python objects, so "parsing" it means running
//! it. Every constant here is checked against the checkpoint's tensor shapes by
//! `cosy-validate`, which is the property that actually matters — a wrong constant
//! shows up as a shape mismatch at load rather than as quiet garbage.

// ------------------------------------------------------------------ shared

pub const SAMPLE_RATE: usize = 24_000;
/// Speech tokens per second.
pub const TOKEN_RATE: usize = 25;
/// Mel frames per speech token.
pub const TOKEN_MEL_RATIO: usize = 2;
/// Mel channels.
pub const MEL_DIM: usize = 80;

// ------------------------------------------------------------------ LLM

pub mod llm {
    /// Qwen2-0.5B. Note this is the *same* geometry as Audio8's slow AR — dim 896,
    /// 24 layers, 14/2 GQA, head_dim 64, ffn 4864, eps 1e-6, rope base 1e6 — which is
    /// why `a8::ar`'s attention shape transfers without change.
    pub const DIM: usize = 896;
    pub const LAYERS: usize = 24;
    pub const N_HEADS: usize = 14;
    pub const N_KV: usize = 2;
    pub const HEAD_DIM: usize = 64;
    pub const FFN: usize = 4864;
    pub const NORM_EPS: f32 = 1e-6;
    pub const ROPE_BASE: f64 = 1_000_000.0;
    /// How many query heads share one KV head.
    pub const GQA: usize = N_HEADS / N_KV;

    /// Speech-token vocabulary proper; ids at or above this are control tokens.
    pub const SPEECH_TOKENS: usize = 6561;
    /// `llm_decoder` and `speech_embedding` are both `[6761, 896]`: the speech tokens
    /// plus 200 control slots.
    pub const VOCAB: usize = SPEECH_TOKENS + 200;

    pub const SOS: usize = SPEECH_TOKENS; // 6561
    pub const EOS: usize = SPEECH_TOKENS + 1; // 6562
    pub const TASK_ID: usize = SPEECH_TOKENS + 2; // 6563

    /// `<|endofprompt|>`. The LLM asserts this appears in the concatenated prompt text;
    /// see trap 2 in `docs/porting/cosyvoice.md`.
    pub const ENDOFPROMPT: u32 = 151_646;

    /// `ras_sampling(top_p=0.8, top_k=25, win_size=10, tau_r=0.1)`.
    pub const TOP_P: f32 = 0.8;
    pub const TOP_K: usize = 25;
    pub const RAS_WIN: usize = 10;
    pub const RAS_TAU: f32 = 0.1;

    /// Generated tokens per text token, the reference's stopping bounds.
    pub const MIN_TOKEN_RATIO: usize = 2;
    pub const MAX_TOKEN_RATIO: usize = 20;
}

// ------------------------------------------------------------------ flow

pub mod flow {
    /// `input_embedding` is `[6561, 80]` — speech tokens only, no control rows.
    pub const TOKEN_VOCAB: usize = 6561;
    pub const SPK_EMBED_DIM: usize = 192;
    /// `spk_embed_affine_layer` projects 192 -> 80.
    pub const SPK_DIM: usize = 80;
    pub const PRE_LOOKAHEAD_LEN: usize = 3;
    pub const PRE_LOOKAHEAD_CHANNELS: usize = 1024;

    /// DiT.
    pub const DIM: usize = 1024;
    pub const DEPTH: usize = 22;
    pub const HEADS: usize = 16;
    pub const HEAD_DIM: usize = 64;
    /// `ff_mult = 2`, so the feed-forward inner width is 2048.
    pub const FF_INNER: usize = DIM * 2;
    /// `SinusPositionEmbedding(256)` feeding a 256 -> 1024 -> 1024 MLP.
    pub const TIME_EMBED_DIM: usize = 256;
    /// The scale `SinusPositionEmbedding.forward` multiplies the timestep by.
    pub const TIME_SCALE: f64 = 1000.0;
    pub const LAYER_NORM_EPS: f64 = 1e-6;
    /// `x_transformers.RotaryEmbedding(dim_head)` default base.
    pub const ROPE_BASE: f64 = 10_000.0;
    /// The convolutional position embedding: two grouped convs, k=31, groups=16.
    pub const CONV_POS_KERNEL: usize = 31;
    pub const CONV_POS_GROUPS: usize = 16;

    /// Euler steps. **Ten**, not six — `flow.inference` passes `n_timesteps=10` and the
    /// `cfm_params` default of 10 is what the yaml leaves in place.
    pub const N_TIMESTEPS: usize = 10;
    /// Classifier-free guidance rate, applied on a doubled batch.
    pub const CFG_RATE: f64 = 0.7;
    /// `rand_noise` is `randn([1, 80, 50 * 300])`.
    pub const RAND_NOISE_FRAMES: usize = 15_000;
}

// ------------------------------------------------------------------ vocoder

pub mod hift {
    pub const IN_CHANNELS: usize = 80;
    pub const BASE_CHANNELS: usize = 512;
    /// `nb_harmonics = 8`, so the harmonic stack is 9 wide (fundamental + 8).
    pub const HARMONICS: usize = 9;
    pub const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
    pub const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
    pub const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
    pub const SOURCE_RESBLOCK_KERNELS: [usize; 3] = [7, 7, 11];
    pub const DILATIONS: [usize; 3] = [1, 3, 5];
    pub const N_FFT: usize = 16;
    pub const HOP: usize = 4;
    /// `n_fft + 2` — magnitude and phase channels packed together.
    pub const SPEC_CHANNELS: usize = N_FFT + 2;
    /// `prod(UPSAMPLE_RATES) * HOP` = 480 samples per mel frame.
    pub const UPSAMPLE_TOTAL: usize = 8 * 5 * 3 * HOP;
    /// The leaky-ReLU slope used *inside the upsampling loop*.
    pub const LRELU_SLOPE: f64 = 0.1;
    /// The slope of the leaky-ReLU immediately before `conv_post`, which the reference
    /// writes as a bare `F.leaky_relu(x)` — so it takes torch's default 0.01, not the
    /// configured 0.1. See trap 7 in `docs/porting/cosyvoice.md`.
    pub const LRELU_SLOPE_POST: f64 = 0.01;
    pub const AUDIO_LIMIT: f64 = 0.99;
    /// `conv_pre` looks 4 frames forward.
    pub const CONV_PRE_LOOKAHEAD: usize = 4;
    /// F0 below this is treated as unvoiced.
    pub const VOICED_THRESHOLD: f64 = 10.0;
    /// The sample rate as a float, for the phase arithmetic.
    pub const SAMPLE_RATE_F: f64 = super::SAMPLE_RATE as f64;
    /// Sine amplitude (`nsf_alpha`) and the voiced-region noise std (`nsf_sigma`).
    pub const SINE_AMP: f64 = 0.1;
    pub const NOISE_STD: f64 = 0.003;
    pub const F0_COND_CHANNELS: usize = 512;
    /// `Snake.no_div_by_zero` — the reference divides by `alpha + 1e-9`, and folding
    /// has to use the same denominator or the fold is not exact.
    pub const SNAKE_EPS: f64 = 1e-9;

    /// Channels after each upsample stage: 512/2, 512/4, 512/8.
    pub const fn stage_channels(i: usize) -> usize {
        BASE_CHANNELS >> (i + 1)
    }
}
