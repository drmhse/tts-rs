//! A reproducible userspace PRNG, shared by the engines.
//!
//! xoshiro256** — small, fast, and reproducible across runs and machines, which matters
//! more here than statistical pedigree.
//!
//! The limit worth stating up front, because it shapes how both engines are validated:
//! **no userspace RNG reproduces torch's Philox stream.** So sampled output can never be
//! compared token-for-token against the Python reference, and every gate in this repo is
//! built on something deterministic instead — greedy decoding, teacher forcing, or a
//! fixed noise tensor shipped as an asset. Reaching for "seed both sides the same" is
//! the trap; it does not exist.

use anyhow::Result;
use candle_core::{Device, Shape, Tensor};

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 to spread a single seed over the state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1) — open at both ends so `-ln(u)` is always finite.
    ///
    /// Audio8's Gumbel-max sampler depends on that: a drawn 0 would give an infinite
    /// score and a drawn 1 a zero one.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        (bits as f32 + 0.5) / 16_777_216.0
    }

    /// Fill a host buffer with uniforms.
    pub fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next_f32();
        }
    }

    /// A tensor of uniforms on `device`.
    ///
    /// Drawn on the host and uploaded rather than using candle's device RNG, because
    /// reproducibility here has to survive a change of backend — the same seed must give
    /// the same audio on Metal and on CPU.
    pub fn uniform_tensor<S: Into<Shape>>(&mut self, shape: S, device: &Device) -> Result<Tensor> {
        let shape: Shape = shape.into();
        let mut v = vec![0f32; shape.elem_count()];
        self.fill(&mut v);
        Ok(Tensor::from_vec(v, shape, device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let a: Vec<f32> = {
            let mut r = Rng::new(7);
            (0..64).map(|_| r.next_f32()).collect()
        };
        let b: Vec<f32> = {
            let mut r = Rng::new(7);
            (0..64).map(|_| r.next_f32()).collect()
        };
        assert_eq!(a, b);
        let c: Vec<f32> = {
            let mut r = Rng::new(8);
            (0..64).map(|_| r.next_f32()).collect()
        };
        assert_ne!(a, c);
    }

    #[test]
    fn draws_stay_strictly_inside_the_unit_interval() {
        // Load-bearing for Gumbel-max: `-ln(u)` must be finite and non-zero.
        let mut r = Rng::new(1);
        for _ in 0..100_000 {
            let u = r.next_f32();
            assert!(u > 0.0 && u < 1.0, "{u}");
        }
    }

    #[test]
    fn the_mean_of_many_draws_is_near_one_half() {
        // The NSF noise substitution in the CosyVoice vocoder relies on this being a
        // uniform on [0, 1) — including its non-zero mean, which the reference's
        // `torch.rand` also has.
        let mut r = Rng::new(3);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| r.next_f32() as f64).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }
}
