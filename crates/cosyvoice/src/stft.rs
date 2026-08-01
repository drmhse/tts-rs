//! A 16-point STFT and its inverse, as dense matrices.
//!
//! The vocoder's `n_fft` is 16 with a hop of 4. At that size a general FFT is the wrong
//! tool: the transform is a `[16, 9]` matrix and the whole thing is two GEMMs plus an
//! overlap-add. Candle has no FFT and no `torch.stft`, so this would have to be written
//! either way; writing it as a matmul makes it one dispatch instead of a kernel.
//!
//! Both directions have to agree with torch bit-for-bit-ish, which means reproducing
//! three conventions that are easy to get wrong:
//!
//! - **`center=True`**, torch's default: the signal is reflect-padded by `n_fft / 2` on
//!   both sides before framing, and the inverse trims the same amount. Skipping this
//!   shifts the output by 8 samples and shortens it by 16 — audible as a click at the
//!   segment boundary rather than as an obvious failure.
//! - **A periodic Hann window** (`get_window("hann", 16, fftbins=True)`), not the
//!   symmetric one. The two differ in every sample but the first.
//! - **Window-square normalisation on the way back.** `torch.istft` divides the
//!   overlap-added result by the overlap-added squared window. With hop = `n_fft / 4`
//!   Hann satisfies COLA in the interior, so the divisor is constant there and only the
//!   first and last few samples differ — which is exactly the region `center=True`
//!   trims, so getting it wrong is invisible until you compare against a fixture.
//!
//! One deliberate departure from the reference, and it is not an approximation:
//! `torch.istft` on a one-sided spectrum builds the full spectrum by Hermitian symmetry,
//! which discards the imaginary parts of bins 0 and `n_fft / 2`. The matrix here has
//! zero columns for those two, so the discard is structural rather than something the
//! caller has to remember.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::f64::consts::PI;

/// Precomputed analysis and synthesis matrices for one `(n_fft, hop)` pair.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    bins: usize,
    /// `[n_fft, bins * 2]`: frames -> (real | imag).
    analysis: Tensor,
    /// `[bins * 2, n_fft]`: (real | imag) -> frames, including the `1/n_fft` scaling.
    synthesis: Tensor,
    /// `[1, n_fft]`, the periodic Hann window.
    window: Tensor,
    /// `[n_fft]`, the window squared — for the inverse's normalisation.
    window_sq: Vec<f32>,
    device: Device,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize, device: &Device) -> Result<Self> {
        let bins = n_fft / 2 + 1;
        // Periodic Hann: 0.5 - 0.5 cos(2 pi n / N). scipy's `fftbins=True`.
        let win: Vec<f32> = (0..n_fft)
            .map(|n| (0.5 - 0.5 * (2.0 * PI * n as f64 / n_fft as f64).cos()) as f32)
            .collect();

        // Analysis: X[k] = sum_n x[n] e^{-i 2 pi k n / N}, so the real part takes
        // cos and the imaginary part takes -sin.
        let mut analysis = vec![0f32; n_fft * bins * 2];
        for n in 0..n_fft {
            for k in 0..bins {
                let th = 2.0 * PI * (k * n) as f64 / n_fft as f64;
                analysis[n * bins * 2 + k] = th.cos() as f32;
                analysis[n * bins * 2 + bins + k] = -th.sin() as f32;
            }
        }

        // Synthesis: the one-sided inverse real FFT. Bin 0 and bin N/2 appear once and
        // contribute only their real part; bins in between appear twice, which is where
        // the factor of 2 comes from.
        let mut synthesis = vec![0f32; bins * 2 * n_fft];
        let scale = 1.0 / n_fft as f64;
        for k in 0..bins {
            let edge = k == 0 || k == n_fft / 2;
            let mult = if edge { 1.0 } else { 2.0 };
            for n in 0..n_fft {
                let th = 2.0 * PI * (k * n) as f64 / n_fft as f64;
                synthesis[k * n_fft + n] = (mult * scale * th.cos()) as f32;
                // Hermitian symmetry drops the imaginary part of the two edge bins.
                synthesis[(bins + k) * n_fft + n] = if edge {
                    0.0
                } else {
                    (-mult * scale * th.sin()) as f32
                };
            }
        }

        Ok(Self {
            n_fft,
            hop,
            bins,
            analysis: Tensor::from_vec(analysis, (n_fft, bins * 2), device)?,
            synthesis: Tensor::from_vec(synthesis, (bins * 2, n_fft), device)?,
            window: Tensor::from_vec(win.clone(), (1, n_fft), device)?,
            window_sq: win.iter().map(|w| w * w).collect(),
            device: device.clone(),
        })
    }

    /// Frames for a signal of `len` samples under `center=True`.
    pub fn frames_for(&self, len: usize) -> usize {
        len / self.hop + 1
    }

    /// `[1, len] -> [1, bins * 2, frames]`, real channels then imaginary.
    ///
    /// Matches `torch.stft(..., return_complex=True)` followed by
    /// `cat([real, imag], dim=1)`, which is what the vocoder does with it.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let len = x.dim(1)?;
        let half = self.n_fft / 2;
        let padded = reflect_pad_1d(&x.reshape((1, 1, len))?, half, half)?.reshape((1, ()))?;
        let frames = self.frames_for(len);

        // Frame the signal. Each tap `n` reads a stride-`hop` subsequence, which is a
        // reshape away: view the tail as `[frames, hop]` and keep column 0.
        let need = frames * self.hop;
        let padded = padded
            .pad_with_zeros(1, 0, need + self.n_fft)?
            .flatten_all()?;
        let mut taps = Vec::with_capacity(self.n_fft);
        for n in 0..self.n_fft {
            taps.push(
                padded
                    .narrow(0, n, need)?
                    .reshape((frames, self.hop))?
                    .narrow(1, 0, 1)?,
            );
        }
        // [frames, n_fft], windowed.
        let framed = Tensor::cat(&taps, 1)?.broadcast_mul(&self.window)?;
        // [frames, bins * 2] -> [1, bins * 2, frames].
        let spec = framed.matmul(&self.analysis)?;
        Ok(spec.t()?.unsqueeze(0)?.contiguous()?)
    }

    /// `[1, bins * 2, frames] -> [1, len]`, the inverse with window normalisation.
    ///
    /// `magnitude` and `phase` are the vocoder's own parameterisation; this takes the
    /// already-combined real and imaginary channels.
    pub fn inverse(&self, spec: &Tensor) -> Result<Tensor> {
        let frames = spec.dim(2)?;
        // [frames, bins * 2] @ [bins * 2, n_fft] -> [frames, n_fft], then window.
        let framed = spec
            .squeeze(0)?
            .t()?
            .contiguous()?
            .matmul(&self.synthesis)?
            .broadcast_mul(&self.window)?;

        let total = (frames - 1) * self.hop + self.n_fft;
        let sum = self.overlap_add(&framed, frames, total)?;

        // Normalise by the overlap-added squared window. Constant in the interior, and
        // the ends are trimmed by `center=True`, but computed exactly all the same.
        let env = self.window_envelope(frames, total)?;
        let y = sum.broadcast_div(&env)?;

        let half = self.n_fft / 2;
        Ok(y.narrow(0, half, total - self.n_fft)?.reshape((1, ()))?)
    }

    /// Scatter `[frames, n_fft]` back to a `total`-length signal at hop spacing.
    ///
    /// The transpose of the framing above: tap `n` of every frame lands at
    /// `t * hop + n`, so writing column `n` into the first slot of a `[frames, hop]`
    /// view and shifting right by `n` places all of them at once.
    fn overlap_add(&self, framed: &Tensor, frames: usize, total: usize) -> Result<Tensor> {
        let stride_len = frames * self.hop;
        let pad = Tensor::zeros((frames, self.hop - 1), DType::F32, &self.device)?;
        let mut acc: Option<Tensor> = None;
        for n in 0..self.n_fft {
            let col = framed.narrow(1, n, 1)?;
            let spread = Tensor::cat(&[col, pad.clone()], 1)?.reshape(stride_len)?;
            // Shift right by `n`, then trim or pad to `total`.
            let placed = spread.pad_with_zeros(0, n, 0)?;
            let plen = placed.dim(0)?;
            let placed = if plen >= total {
                placed.narrow(0, 0, total)?
            } else {
                placed.pad_with_zeros(0, 0, total - plen)?
            };
            acc = Some(match acc {
                None => placed,
                Some(a) => (a + placed)?,
            });
        }
        Ok(acc.expect("n_fft > 0"))
    }

    /// The overlap-added squared window, `[total]`, floored away from zero the way
    /// `torch.istft` floors it.
    fn window_envelope(&self, frames: usize, total: usize) -> Result<Tensor> {
        let mut env = vec![0f32; total];
        for t in 0..frames {
            for n in 0..self.n_fft {
                env[t * self.hop + n] += self.window_sq[n];
            }
        }
        // torch guards the division with a small epsilon rather than masking.
        for v in env.iter_mut() {
            if *v < 1e-11 {
                *v = 1.0;
            }
        }
        Ok(Tensor::from_vec(env, total, &self.device)?)
    }

    pub fn bins(&self) -> usize {
        self.bins
    }
}

/// Reflect padding along the last dimension of a `[b, c, len]` tensor.
///
/// Reflection excludes the edge sample, matching torch: padding by 1 on the left
/// prepends `x[1]`, not `x[0]`.
pub fn reflect_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let len = x.dim(2)?;
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        // x[left], x[left-1], ..., x[1]
        let idx: Vec<u32> = (1..=left).rev().map(|i| i as u32).collect();
        let idx = Tensor::from_vec(idx, left, x.device())?;
        parts.push(x.index_select(&idx, 2)?);
    }
    parts.push(x.clone());
    if right > 0 {
        // x[len-2], x[len-3], ..., x[len-1-right]
        let idx: Vec<u32> = (1..=right).map(|i| (len - 1 - i) as u32).collect();
        let idx = Tensor::from_vec(idx, right, x.device())?;
        parts.push(x.index_select(&idx, 2)?);
    }
    Ok(Tensor::cat(&parts, 2)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_nn::max_abs_diff;

    #[test]
    fn reflect_pad_excludes_the_edge_sample() -> Result<()> {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 4), &d)?;
        let y = reflect_pad_1d(&x, 2, 2)?;
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        Ok(())
    }

    #[test]
    fn stft_round_trip_reconstructs_the_signal() -> Result<()> {
        // The property that matters: analysis then synthesis is the identity, including
        // the centring and the window normalisation. If any of the three conventions in
        // the module docstring is wrong this fails.
        let d = Device::Cpu;
        let st = Stft::new(16, 4, &d)?;
        let n = 512;
        let x: Vec<f32> = (0..n)
            .map(|i| (0.017 * i as f64).sin() as f32 * 0.5 + (0.3 * i as f64).cos() as f32 * 0.2)
            .collect();
        let x = Tensor::from_vec(x, (1, n), &d)?;
        let spec = st.forward(&x)?;
        assert_eq!(spec.dims(), &[1, 18, n / 4 + 1]);
        let y = st.inverse(&spec)?;
        assert_eq!(y.dims(), &[1, n]);
        let err = max_abs_diff(&y, &x)?;
        assert!(err < 2e-5, "round trip error {err}");
        Ok(())
    }

    #[test]
    fn a_constant_signal_transforms_to_the_window_spectrum() -> Result<()> {
        // A constant signal is not spectrally flat once windowed — what comes out is the
        // window's own transform. Periodic Hann is `0.5 - 0.5 cos(2 pi n / N)`, whose
        // DFT is `N/2` at bin 0 and `-N/4` at bin 1, real, and zero everywhere else.
        // That pins the window convention, the scaling and the real/imaginary split all
        // at once, which is why this is worth asserting exactly.
        let d = Device::Cpu;
        let st = Stft::new(16, 4, &d)?;
        let x = Tensor::ones((1, 128), DType::F32, &d)?;
        let spec = st.forward(&x)?;
        let mid = spec.narrow(2, 10, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let bins = st.bins();
        let mut want = vec![0f32; bins * 2];
        want[0] = 8.0;
        want[1] = -4.0;
        for (k, (got, exp)) in mid.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "{} {} = {got}, expected {exp}",
                if k < bins { "real bin" } else { "imag bin" },
                k % bins
            );
        }
        Ok(())
    }
}
