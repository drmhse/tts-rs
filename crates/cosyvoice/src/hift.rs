//! `CausalHiFTGenerator`: mel -> 24 kHz waveform. Neural source filter plus iSTFTNet.
//!
//! Three stages, and the fixture set is split along the same seams so a mismatch says
//! which one is wrong:
//!
//! 1. **F0 predictor** — five weight-normalised causal convs with ELU, then a linear
//!    layer and an absolute value. `mel [1, 80, T] -> f0 [1, T]`.
//! 2. **Harmonic source** — nine sine harmonics at multiples of F0, noise-mixed by a
//!    voiced/unvoiced decision, collapsed to one channel by a `tanh(Linear(9, 1))`.
//!    `f0 [1, T] -> s [1, 1, T * 480]`.
//! 3. **Upsampling decoder** — `conv_pre`, three upsample stages each fused with a
//!    downsampled view of the source's STFT and averaged over three ResBlocks, then
//!    `conv_post` and an iSTFT.
//!
//! # What this port does differently, and why each is exact
//!
//! **The harmonic source is computed at frame rate, not sample rate.** The reference
//! upsamples F0 by 480 to get `[1, 100800, 9]`, takes `(f0 * h / sr) mod 1`, then
//! *downsamples that by 1/480 with `mode='linear'`* before the cumulative sum. Those two
//! resamplings look like they must lose something. They do not: with `align_corners=
//! False` the downsample reads samples 239 and 240 of each 480-sample block with weight
//! 0.5 each, and both lie inside the block, which is constant. So the round trip is
//! exactly "take the frame's value" — measured at 0.0 difference, not approximately. The
//! port computes `rad` per frame and never materialises the 100800x9 intermediate.
//!
//! **`SineGen2.rand_ini` is dead code and is not implemented.** It adds a random phase
//! offset to sample 0 of the full-rate tensor, which the downsample above then discards
//! — samples 239 and 240 are what get read. Measured contribution: exactly 0.0. A port
//! that faithfully reproduces it is reproducing nothing, and it would need an extra
//! asset to do it.
//!
//! **The NSF noise is a caller-supplied tensor.** `SineGen2.sine_waves` is a
//! `torch.rand(1, 300 * 24000, 9)` *plain attribute* — not a registered buffer, so it is
//! absent from `hift.pt` and redrawn at construction, reproducible only because
//! `cosyvoice3.yaml` seeds torch immediately beforehand. It is not negligible: zeroing
//! it moves the waveform by max 0.164 against a signal of rms 0.078. So [`Hift::decode`]
//! takes a [`Noise`] and validation injects the reference slice, while synthesis draws
//! its own. That substitution is *distributionally* exact, because the reference draws
//! uniform `[0, 1)` and so does this — see [`Noise::Draw`].
//!
//! **Both snake scalings are folded into neighbouring convolutions.** `snake(x) =
//! a^-1 (u + sin^2 u)` with `u = a x`. In a ResBlock the first activation's output feeds
//! only `convs1` and the second's only `convs2` — the skip connection carries the
//! block's *input*, not an activation — so unlike Audio8's codec, **every one of the 72
//! snakes here can shed its reciprocal**, and the 36 second activations can shed the
//! multiply too by folding `alpha` into the preceding conv's output weights and bias.
//! What is left per group is one `broadcast_mul`, down from four broadcasts.
//!
//! **F0 is predicted in f32, not f64.** The reference casts the predictor to `float64`
//! with a comment that precision is crucial, then immediately casts the result back to
//! f32 and does the phase accumulation in f32 anyway — so the f64 buys precision in the
//! convolutions only. Metal has no f64. Measured difference on this fixture: 2.8e-3 Hz
//! on an F0 range of 0-347 Hz, and the resulting waveform difference is reported by
//! `cosyvoice-validate` rather than assumed. See trap 8 in `docs/reference.md#porting-traps`.
//!
//! **The leaky-ReLU before `conv_post` uses slope 0.01, not 0.1.** The reference writes
//! it as a bare `F.leaky_relu(x)`, taking torch's default, while every other one in the
//! loop passes `self.lrelu_slope`. Trap 7.

use crate::cfg::hift as k;
use crate::stft::{reflect_pad_1d, Stft};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use tts_core::rng::Rng;
use tts_nn::{
    causal_conv1d, causal_conv1d_gemm, elu, fused, leaky_relu, lookahead_conv1d, tap_major_weight,
    upsample_nearest1d, Weights,
};

/// Where the NSF noise comes from.
///
/// The reference's noise is not in the checkpoint (see the module docstring), so this is
/// a choice the caller has to make rather than a detail the vocoder can hide.
pub enum Noise<'a> {
    /// The reference's own slice, `[1, samples, 9]`. For fixture validation, where the
    /// point is to prove the rest of the graph exactly.
    Reference(&'a Tensor),
    /// Draw uniform `[0, 1)`, which is the distribution the reference draws from —
    /// `torch.rand`, not `torch.randn`. Worth noting because uniform noise has a
    /// non-zero mean, so it contributes a small positive offset as well as noise; that
    /// is the reference's behaviour and reproducing the *distribution* reproduces it.
    Draw(&'a mut Rng),
    /// No noise at all. Not the reference's behaviour — kept only so the noise's
    /// contribution can be measured, which is how the 0.164 figure above was obtained.
    Silent,
}

/// A weight-normalised convolution with its parametrisation already resolved.
struct Conv {
    /// `[out, in, k]`.
    w: Tensor,
    b: Tensor,
    dilation: usize,
    /// `[out, k * in]` for the GEMM route; `None` for `k == 1`. Built by [`Conv::finalize`]
    /// *after* any alpha folding, since folding changes the weight.
    w_tap: Option<Tensor>,
}

impl Conv {
    fn load(w: &Weights, prefix: &str, dilation: usize) -> Result<Self> {
        Ok(Self {
            w: w.get_weight_norm(prefix)?,
            b: w.get(&format!("{prefix}.bias"))?,
            dilation,
            w_tap: None,
        })
    }

    /// Precompute the tap-major weight. Must run after all folding.
    fn finalize(&mut self) -> Result<()> {
        if self.w.dim(2)? > 1 {
            self.w_tap = Some(tap_major_weight(&self.w)?);
        }
        Ok(())
    }

    /// Scale the input channels — folds a preceding `alpha^-1` in.
    fn fold_input(&mut self, scale: &Tensor) -> Result<()> {
        let c_in = self.w.dim(1)?;
        self.w = self
            .w
            .broadcast_mul(&scale.reshape((1, c_in, 1))?)?
            .contiguous()?;
        Ok(())
    }

    /// Scale the output channels and bias — folds a following `alpha` in.
    fn fold_output(&mut self, scale: &Tensor) -> Result<()> {
        let c_out = self.w.dim(0)?;
        self.w = self
            .w
            .broadcast_mul(&scale.reshape((c_out, 1, 1))?)?
            .contiguous()?;
        self.b = self.b.mul(scale)?;
        Ok(())
    }

    /// Left-causal. Takes the GEMM route when a tap-major weight exists — measured 1.34x to
    /// 1.73x over candle's `conv1d` across this channel/length regime; see
    /// [`tts_nn::causal_conv1d_gemm`].
    fn causal(&self, x: &Tensor) -> Result<Tensor> {
        match &self.w_tap {
            Some(w_tap) => {
                causal_conv1d_gemm(x, w_tap, Some(&self.b), self.w.dim(2)?, self.dilation)
            }
            None => causal_conv1d(x, &self.w, Some(&self.b), self.dilation),
        }
    }

    fn lookahead(&self, x: &Tensor) -> Result<Tensor> {
        lookahead_conv1d(x, &self.w, Some(&self.b), self.dilation)
    }
}

/// One dilation group of a ResBlock, with both snake scalings folded away.
///
/// The reference is
/// ```text
/// xt = snake1(x); xt = convs1(xt); xt = snake2(xt); xt = convs2(xt); x = xt + x
/// ```
/// and after folding `alpha1^-1` into `convs1`'s input, `alpha2` into `convs1`'s output
/// and `alpha2^-1` into `convs2`'s input, it is
/// ```text
/// u = alpha1 * x; h = u + sin^2 u; u2 = convs1'(h); h2 = u2 + sin^2 u2; x = convs2'(h2) + x
/// ```
/// One broadcast multiply remains, because `x` also feeds the skip.
struct ResGroup {
    /// `[1, c, 1]`, ready to broadcast.
    alpha1: Tensor,
    conv1: Conv,
    conv2: Conv,
}

struct ResBlock {
    groups: Vec<ResGroup>,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str, kernel: usize) -> Result<Self> {
        let _ = kernel; // the kernel width is carried by the stored weight's shape
        let mut groups = Vec::with_capacity(k::DILATIONS.len());
        for (i, &d) in k::DILATIONS.iter().enumerate() {
            let a1 = w.get(&format!("{prefix}.activations1.{i}.alpha"))?;
            let a2 = w.get(&format!("{prefix}.activations2.{i}.alpha"))?;
            // The reference divides by `alpha + 1e-9`; folding has to use the same
            // denominator or it is not the same function.
            let r1 = recip(&a1)?;
            let r2 = recip(&a2)?;

            let mut conv1 = Conv::load(w, &format!("{prefix}.convs1.{i}"), d)?;
            let mut conv2 = Conv::load(w, &format!("{prefix}.convs2.{i}"), 1)?;
            conv1.fold_input(&r1)?;
            conv1.fold_output(&a2)?;
            conv2.fold_input(&r2)?;
            conv1.finalize()?;
            conv2.finalize()?;

            let c = a1.elem_count();
            groups.push(ResGroup {
                alpha1: a1.reshape((1, c, 1))?,
                conv1,
                conv2,
            });
        }
        Ok(Self { groups })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for g in &self.groups {
            let h = fused::snake_alpha(&x, &g.alpha1)?;
            let u2 = g.conv1.causal(&h)?;
            // `u2` already carries `alpha2`, so the half-folded snake applies directly.
            let h2 = fused::snake_folded(&u2)?;
            x = (g.conv2.causal(&h2)? + x)?;
        }
        Ok(x)
    }
}

fn recip(alpha: &Tensor) -> Result<Tensor> {
    Ok((alpha + k::SNAKE_EPS)?.recip()?)
}

/// The F0 predictor: `CausalConvRNNF0Predictor`.
///
/// Named "RNN" upstream but there is no recurrence in it — five convolutions and a
/// linear layer. Worth stating because the port plan budgeted for an RNN that does not
/// exist.
struct F0Predictor {
    /// The first conv looks *forward* (`causal_type='right'`); the rest look back.
    convs: Vec<Conv>,
    classifier_w: Tensor,
    classifier_b: Tensor,
}

impl F0Predictor {
    fn load(w: &Weights) -> Result<Self> {
        let mut convs = Vec::with_capacity(5);
        // nn.Sequential alternates conv and ELU, so the convs sit at 0, 2, 4, 6, 8.
        for i in 0..5 {
            convs.push(Conv::load(
                w,
                &format!("f0_predictor.condnet.{}", i * 2),
                1,
            )?);
        }
        Ok(Self {
            convs,
            classifier_w: w.get("f0_predictor.classifier.weight")?,
            classifier_b: w.get("f0_predictor.classifier.bias")?,
        })
    }

    /// `[1, 80, T] -> [1, T]`.
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = elu(&self.convs[0].lookahead(mel)?)?;
        for c in &self.convs[1..] {
            x = elu(&c.causal(&x)?)?;
        }
        // [1, C, T] -> [1, T, C] -> [1, T, 1] -> [1, T], then abs.
        let y = x
            .transpose(1, 2)?
            .contiguous()?
            .broadcast_matmul(&self.classifier_w.t()?)?
            .broadcast_add(&self.classifier_b)?;
        Ok(y.squeeze(2)?.abs()?)
    }
}

pub struct Hift {
    f0: F0Predictor,
    /// `[1, 9]` collapsing the harmonics, and its bias.
    source_w: Tensor,
    source_b: Tensor,
    conv_pre: Conv,
    /// `(conv, stride)` per upsample stage. These run at the decoder's widest lengths
    /// (k=16/11/7 after upsampling by 8/5/3), so they take the GEMM route like every
    /// other conv here — calling `causal_conv1d` directly left the single largest
    /// convolution in the vocoder on the slow path.
    ups: Vec<(Conv, usize)>,
    /// `(conv, stride)` — stride 1 stages are plain causal convs.
    source_downs: Vec<(Conv, usize)>,
    source_resblocks: Vec<ResBlock>,
    /// Nine blocks: three per upsample stage.
    resblocks: Vec<ResBlock>,
    conv_post: Conv,
    stft: Stft,
    device: Device,
}

impl Hift {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;

        let mut ups = Vec::with_capacity(3);
        for i in 0..3 {
            let mut conv = Conv::load(&w, &format!("ups.{i}"), 1)?;
            conv.finalize()?;
            ups.push((conv, k::UPSAMPLE_RATES[i]));
        }

        // downsample_rates = [1] + reversed(upsample_rates)[:-1] = [1, 3, 5];
        // cumprod = [1, 3, 15]; reversed = [15, 3, 1].
        let down_strides = [15usize, 3, 1];
        let mut source_downs = Vec::with_capacity(3);
        let mut source_resblocks = Vec::with_capacity(3);
        for (i, &stride) in down_strides.iter().enumerate() {
            let mut down = Conv::load(&w, &format!("source_downs.{i}"), 1)?;
            // Only the stride-1 stage goes through `causal`; the strided ones call `conv1d`
            // directly with a stride, which the GEMM route does not cover.
            if stride == 1 {
                down.finalize()?;
            }
            source_downs.push((down, stride));
            source_resblocks.push(ResBlock::load(
                &w,
                &format!("source_resblocks.{i}"),
                k::SOURCE_RESBLOCK_KERNELS[i],
            )?);
        }

        let mut resblocks = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                resblocks.push(ResBlock::load(
                    &w,
                    &format!("resblocks.{}", i * 3 + j),
                    k::RESBLOCK_KERNELS[j],
                )?);
            }
        }

        let hift = Self {
            f0: F0Predictor::load(&w)?,
            source_w: w.get("m_source.l_linear.weight")?,
            source_b: w.get("m_source.l_linear.bias")?,
            conv_pre: Conv::load(&w, "conv_pre", 1)?,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post: {
                let mut c = Conv::load(&w, "conv_post", 1)?;
                c.finalize()?;
                c
            },
            stft: Stft::new(k::N_FFT, k::HOP, device)?,
            device: device.clone(),
        };
        hift.check_geometry()?;
        Ok(hift)
    }

    /// Cross-check `cfg` against the checkpoint. A wrong constant should fail here, at
    /// load, rather than as quiet garbage 100 k samples later.
    fn check_geometry(&self) -> Result<()> {
        if self.conv_pre.w.dim(2)? != k::CONV_PRE_LOOKAHEAD + 1 {
            bail!(
                "conv_pre kernel is {} but CONV_PRE_LOOKAHEAD implies {}",
                self.conv_pre.w.dim(2)?,
                k::CONV_PRE_LOOKAHEAD + 1
            );
        }
        if self.conv_post.w.dim(0)? != k::SPEC_CHANNELS {
            bail!(
                "conv_post emits {} channels, expected n_fft + 2 = {}",
                self.conv_post.w.dim(0)?,
                k::SPEC_CHANNELS
            );
        }
        if self.source_w.dim(1)? != k::HARMONICS {
            bail!(
                "m_source.l_linear takes {} harmonics, expected {}",
                self.source_w.dim(1)?,
                k::HARMONICS
            );
        }
        for (i, (conv, stride)) in self.ups.iter().enumerate() {
            let w = &conv.w;
            let want_in = if i == 0 {
                k::BASE_CHANNELS
            } else {
                k::stage_channels(i - 1)
            };
            if w.dim(1)? != want_in || w.dim(0)? != k::stage_channels(i) {
                bail!(
                    "ups.{i} is [{}, {}] but the config implies [{}, {}]",
                    w.dim(0)?,
                    w.dim(1)?,
                    k::stage_channels(i),
                    want_in
                );
            }
            if *stride != k::UPSAMPLE_RATES[i] {
                bail!("ups.{i} stride {stride} != {}", k::UPSAMPLE_RATES[i]);
            }
        }
        Ok(())
    }

    /// Total samples per mel frame — 480 at this configuration.
    pub fn samples_per_frame(&self) -> usize {
        k::UPSAMPLE_TOTAL
    }

    /// mel `[1, 80, T]` -> waveform `[1, T * 480]`.
    pub fn forward(&self, mel: &Tensor, noise: Noise<'_>) -> Result<Tensor> {
        let f0 = self.f0.forward(mel)?;
        let source = self.source(&f0, noise)?;
        self.decode(mel, &source)
    }

    /// The F0 predictor alone, exposed so the fixture gate can bisect the stages.
    pub fn predict_f0(&self, mel: &Tensor) -> Result<Tensor> {
        self.f0.forward(mel)
    }

    /// F0 `[1, T]` -> harmonic source `[1, 1, T * 480]`.
    ///
    /// Computed at frame rate up to the final `sin`; see the module docstring for why
    /// that is exact rather than an approximation.
    pub fn source(&self, f0: &Tensor, noise: Noise<'_>) -> Result<Tensor> {
        let frames = f0.dim(1)?;
        let up = k::UPSAMPLE_TOTAL;
        let samples = frames * up;

        // The phase, accumulated on the host in f64 and reduced modulo one cycle.
        //
        // This is the one place the port is deliberately *more* accurate than the
        // reference rather than equal to it. The reference accumulates
        // `cumsum(rad) * 2 pi * 480` in f32, which on this fixture reaches 1.7e7 radians
        // — where a single f32 ulp is 1.0 radian. Its harmonic phases at the tail of an
        // utterance are therefore rounding noise, and torch's own f32 result sits rel
        // 5.3e-4 away from the same computation in f64.
        //
        // Since `sin` is periodic the phase only matters modulo one cycle, so reducing
        // it is exact in real arithmetic. Doing that on the host costs `frames * 9`
        // f64 operations — 1890 here, once per utterance — and brings the phase from an
        // ulp of 1.0 radian to about 4e-7. It also moves the port *closer* to torch's
        // f32 output, not further, because torch's error is bounded and independent of
        // ours: two f32 accumulations disagree by more than either disagrees with exact.
        let f0_host = f0
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut frac = vec![0f32; k::HARMONICS * frames];
        for h in 0..k::HARMONICS {
            // Accumulate in cycles rather than radians, and fold each step back into
            // [0, 1) so the running value never loses low-order bits.
            let mut acc = 0f64;
            let harmonic = (h + 1) as f64;
            for (t, &v) in f0_host.iter().enumerate() {
                let rad = (v as f64 * harmonic / k::SAMPLE_RATE_F).fract();
                acc = (acc + rad * up as f64).fract();
                frac[h * frames + t] = acc as f32;
            }
        }
        // [1, 9, frames] -> hold each frame's phase across its 480 samples.
        let phase = Tensor::from_vec(frac, (1, k::HARMONICS, frames), &self.device)?;
        let phase = upsample_nearest1d(&phase, up)?;
        let sines = phase
            .affine(2.0 * std::f64::consts::PI, 0.0)?
            .sin()?
            .affine(k::SINE_AMP, 0.0)?;

        // Voiced/unvoiced from F0, held across the frame the same way.
        let uv = f0
            .gt(k::VOICED_THRESHOLD as f32)?
            .to_dtype(DType::F32)?
            .reshape((1, 1, frames))?;
        let uv = upsample_nearest1d(&uv, up)?;
        // noise_amp = uv * sigma + (1 - uv) * sine_amp / 3
        let noise_amp = (uv.affine(k::NOISE_STD - k::SINE_AMP / 3.0, k::SINE_AMP / 3.0))?;

        let mut waves = sines.broadcast_mul(&uv)?;
        match noise {
            Noise::Silent => {}
            Noise::Reference(t) => {
                // The reference stores `[1, samples, 9]`; this pipeline is channels-first.
                let n = t.narrow(1, 0, samples)?.transpose(1, 2)?.contiguous()?;
                waves = (waves + n.broadcast_mul(&noise_amp)?)?;
            }
            Noise::Draw(rng) => {
                let n = rng.uniform_tensor((1, k::HARMONICS, samples), &self.device)?;
                waves = (waves + n.broadcast_mul(&noise_amp)?)?;
            }
        }

        // tanh(Linear(9 -> 1)): a weighted sum over the harmonic axis.
        let w = self.source_w.reshape((1, k::HARMONICS, 1))?;
        let merged = waves
            .broadcast_mul(&w)?
            .sum_keepdim(1)?
            .broadcast_add(&self.source_b.reshape((1, 1, 1))?)?;
        Ok(merged.tanh()?)
    }

    /// mel `[1, 80, T]` plus source `[1, 1, T * 480]` -> waveform `[1, T * 480]`.
    pub fn decode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor> {
        let s_spec = self.stft.forward(&source.reshape((1, ()))?)?;
        let mut x = self.conv_pre.lookahead(mel)?;

        for i in 0..3 {
            x = leaky_relu(&x, k::LRELU_SLOPE)?;
            let (conv, stride) = &self.ups[i];
            // CausalConv1dUpsample: nearest upsample, then a causal conv at stride 1.
            x = upsample_nearest1d(&x, *stride)?;
            x = conv.causal(&x)?;

            if i == 2 {
                // ReflectionPad1d((1, 0)) — one frame of left context by reflection,
                // which is also what makes the lengths line up with the source's STFT.
                x = reflect_pad_1d(&x, 1, 0)?;
            }

            let (conv, down_stride) = &self.source_downs[i];
            let si = if *down_stride == 1 {
                conv.causal(&s_spec)?
            } else {
                // CausalConv1dDownSample: left-pad `stride - 1`, then a strided conv.
                let padded = s_spec.pad_with_zeros(2, down_stride - 1, 0)?;
                padded
                    .conv1d(&conv.w, 0, *down_stride, 1, 1)?
                    .broadcast_add(&conv.b.reshape((1, conv.b.elem_count(), 1))?)?
            };
            let si = self.source_resblocks[i].forward(&si)?;
            x = (x + si)?;

            // Average the three ResBlocks for this stage.
            let mut acc: Option<Tensor> = None;
            for j in 0..3 {
                let y = self.resblocks[i * 3 + j].forward(&x)?;
                acc = Some(match acc {
                    None => y,
                    Some(a) => (a + y)?,
                });
            }
            x = (acc.expect("three resblocks per stage") / 3.0)?;
        }

        // Trap 7: a bare `F.leaky_relu(x)` here, so torch's default slope of 0.01.
        x = leaky_relu(&x, k::LRELU_SLOPE_POST)?;
        x = self.conv_post.causal(&x)?;

        let bins = self.stft.bins();
        let magnitude = x.narrow(1, 0, bins)?.exp()?;
        // The reference clips the magnitude before the inverse.
        let magnitude = magnitude.clamp(f32::NEG_INFINITY, 1e2)?;
        let phase = x.narrow(1, bins, bins)?.sin()?;
        let real = (&magnitude * phase.cos()?)?;
        let imag = (&magnitude * phase.sin()?)?;

        let wav = self
            .stft
            .inverse(&Tensor::cat(&[real, imag], 1)?.contiguous()?)?;
        Ok(wav.clamp(-k::AUDIO_LIMIT as f32, k::AUDIO_LIMIT as f32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recip_uses_the_references_epsilon() -> Result<()> {
        let d = Device::Cpu;
        let a = Tensor::from_vec(vec![1f32, 2.0], 2, &d)?;
        let r = recip(&a)?.to_vec1::<f32>()?;
        for (i, v) in [1f32, 2.0].iter().enumerate() {
            let want = 1.0 / (v + k::SNAKE_EPS as f32);
            assert!((r[i] - want).abs() < 1e-9);
        }
        Ok(())
    }
}
