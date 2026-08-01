//! Thermally-honest benchmarking.
//!
//! Rules, from `docs/benchmarking.md`:
//!
//! 1. Interleave every variant in the same run, alternating round by round, so a
//!    clock change hits all variants roughly equally and cancels in the ratio.
//! 2. Report median and spread of >=5 samples, not a 3-iteration mean.
//! 3. Run a fixed canary workload before and after, so each run records its own
//!    thermal state and any drift *during* the run is visible.
//! 4. Prefer ratios within a run to absolute numbers across runs.
//!
//! The canary is the `96ch @ 131072` k7/d9 conv from `convopt.rs`, chosen because
//! it is the shape whose drift exposed the problem. Its absolute time is the
//! run's thermal fingerprint: ~60 ms is a cool machine, ~120 ms is a throttled
//! one. Quote it whenever you quote a millisecond figure.

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use std::time::Instant;

/// Median of a sample set. Even counts take the lower midpoint — with an odd
/// `samples` default this rarely matters, and it avoids inventing a value that
/// was never measured.
pub fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// A single variant's measured distribution.
pub struct Stat {
    pub name: String,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Stat {
    /// Max/min as a percentage. Above ~15% means the run was unstable and the
    /// medians should not be trusted to fine resolution.
    pub fn spread_pct(&self) -> f64 {
        (self.max / self.min - 1.0) * 100.0
    }
}

pub struct Harness {
    dev: Device,
    samples: usize,
    canary_x: Tensor,
    canary_w: Tensor,
    canary_first: f64,
    canary_last: f64,
}

impl Harness {
    pub fn new(dev: &Device, samples: usize) -> Result<Self> {
        if samples < 5 {
            bail!("the protocol requires >=5 samples, got {samples}");
        }
        let canary_x = Tensor::randn(0f32, 1.0, (1, 96, 131072), dev)?;
        let canary_w = Tensor::randn(0f32, 0.02, (96, 96, 7), dev)?;
        let mut h = Self {
            dev: dev.clone(),
            samples,
            canary_x,
            canary_w,
            canary_first: 0.0,
            canary_last: 0.0,
        };
        h.canary_first = h.canary()?;
        h.canary_last = h.canary_first;
        println!(
            "canary (96ch@131072 k7 d9): {:.2} ms  [~60 = cool, ~120 = throttled]",
            h.canary_first
        );
        Ok(h)
    }

    /// The fixed reference workload. Median of 3 — it only needs to place the
    /// machine on the thermal scale, not resolve small differences.
    pub fn canary(&mut self) -> Result<f64> {
        let x = self.canary_x.clone();
        let w = self.canary_w.clone();
        let mut s = Vec::new();
        for i in 0..4 {
            let t = Instant::now();
            let out = x.pad_with_zeros(2, 6 * 9, 0)?.conv1d(&w, 0, 1, 9, 1)?;
            self.dev.synchronize()?;
            drop(out);
            if i > 0 {
                s.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let m = median(&s);
        self.canary_last = m;
        Ok(m)
    }

    /// Time variants against each other, interleaved. `variants` are run in
    /// order once per round for `samples` rounds, after one untimed warmup
    /// round. Returns stats in the order given; `[0]` is the baseline the ratio
    /// column is relative to.
    pub fn ab(
        &mut self,
        label: &str,
        variants: &mut [(&str, &mut dyn FnMut() -> candle_core::Result<()>)],
    ) -> Result<Vec<Stat>> {
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];

        // Warmup: first-touch allocation, kernel compilation, cache population.
        for (_, f) in variants.iter_mut() {
            f()?;
        }
        self.dev.synchronize()?;

        for _ in 0..self.samples {
            for (i, (_, f)) in variants.iter_mut().enumerate() {
                let t = Instant::now();
                f()?;
                self.dev.synchronize()?;
                samples[i].push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }

        let stats: Vec<Stat> = variants
            .iter()
            .zip(samples.iter())
            .map(|((name, _), s)| Stat {
                name: (*name).to_string(),
                median: median(s),
                min: s.iter().cloned().fold(f64::MAX, f64::min),
                max: s.iter().cloned().fold(0.0, f64::max),
            })
            .collect();

        let base = stats[0].median;
        println!("\n{label}  (n={}, interleaved)", self.samples);
        println!(
            "{:<34} {:>10} {:>10} {:>10} {:>9} {:>10}",
            "variant", "median ms", "min", "max", "spread", "vs base"
        );
        println!("{}", "-".repeat(88));
        for s in &stats {
            println!(
                "{:<34} {:>10.3} {:>10.3} {:>10.3} {:>8.1}% {:>9.2}x",
                s.name,
                s.median,
                s.min,
                s.max,
                s.spread_pct(),
                base / s.median
            );
        }
        Ok(stats)
    }

    /// Call at the end of a run. A large first/last gap means the run drifted
    /// under itself and even the ratios deserve suspicion.
    pub fn report_drift(&mut self) -> Result<()> {
        let end = self.canary()?;
        let drift = end / self.canary_first;
        println!(
            "\ncanary: {:.2} ms at start, {:.2} ms at end  ->  {:.2}x drift during run  [{}]",
            self.canary_first,
            end,
            drift,
            if drift < 1.15 {
                "stable, ratios trustworthy"
            } else {
                "DRIFTED — interleaving absorbs most of this, but treat absolutes as junk"
            }
        );
        Ok(())
    }

    pub fn dev(&self) -> &Device {
        &self.dev
    }
}
