//! `ras_sampling`: nucleus sampling with a repetition guard.
//!
//! Nothing exotic, but four details are load-bearing and each is easy to normalise into
//! something that sounds subtly different:
//!
//! 1. **The top-p cut-off is taken over the top-k *only*, and the test is
//!    `cum_prob < top_p` evaluated *before* adding.** So the surviving set always
//!    contains at least one token, and it contains the first token whose running sum
//!    reaches `top_p` — the threshold is crossed, not stopped short of. Writing the loop
//!    the other way round gives a set one token smaller almost every step.
//! 2. **The survivors are sampled with their *unnormalised* probabilities.**
//!    `prob.multinomial(1)` on a truncated slice — torch's `multinomial` normalises
//!    internally, so this is equivalent to renormalising, and the port does that
//!    explicitly.
//! 3. **The repetition guard counts over the last `win_size` emitted tokens and fires at
//!    `rep_num >= win_size * tau_r`** — with `win_size = 10` and `tau_r = 0.1` that is
//!    `>= 1.0`, so *a single* repeat within the last ten tokens triggers it. The guard is
//!    therefore not a rare fallback: it fires constantly, and it is what stops this model
//!    degenerating the way its greedy decode does.
//! 4. **When it fires, the offending token is set to `-inf` and the fallback samples from
//!    the *whole* distribution**, not from the nucleus. That widening is the point.
//!
//! The draw cannot match torch's, so sampled output is never compared token-wise against
//! the reference; see `tts_core::rng`. The gate is teacher forcing instead.

use crate::cfg::llm as k;
use tts_core::rng::Rng;

/// Scratch buffers, so a decode loop does not allocate per step.
pub struct Sampler {
    probs: Vec<f32>,
    order: Vec<u32>,
}

impl Sampler {
    pub fn new(vocab: usize) -> Self {
        Self {
            probs: vec![0.0; vocab],
            order: (0..vocab as u32).collect(),
        }
    }
}

/// Softmax in place into `sampler.probs`, guarding the fully-masked case.
fn softmax(sampler: &mut Sampler, scores: &[f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (p, &s) in sampler.probs.iter_mut().zip(scores.iter()) {
        // `exp(-inf - max)` is 0, which is what a mask wants, but `-inf - -inf` is NaN.
        let e = if s == f32::NEG_INFINITY || !max.is_finite() {
            0.0
        } else {
            (s - max).exp()
        };
        *p = e;
        sum += e;
    }
    if sum > 0.0 {
        for p in sampler.probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Draw an index from `weights[..len]`, treated as unnormalised.
fn multinomial(weights: &[f32], idx: &[u32], rng: &mut Rng) -> u32 {
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return idx[0];
    }
    let target = rng.next_f32() * total;
    let mut acc = 0f32;
    for (w, &i) in weights.iter().zip(idx.iter()) {
        acc += w;
        if acc >= target {
            return i;
        }
    }
    // Only reachable through floating-point drift at the very top of the range.
    *idx.last().expect("non-empty")
}

/// Top-p over the top-k, as described in the module docstring.
///
/// Returns the sampled id. `scores` are logits (or log-probabilities — softmax is
/// shift-invariant, so it makes no difference).
fn nucleus(sampler: &mut Sampler, scores: &[f32], rng: &mut Rng) -> u32 {
    softmax(sampler, scores);
    let top_k = k::TOP_K.min(scores.len());
    // Partial sort: only the top-k ordering matters.
    let probs = &sampler.probs;
    sampler.order.select_nth_unstable_by(top_k - 1, |&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let probs = &sampler.probs;
    let head = &mut sampler.order[..top_k];
    head.sort_unstable_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // The cut-off: test before adding, so at least one survives and the threshold is
    // crossed rather than approached.
    let mut cum = 0f32;
    let mut count = 0usize;
    for &i in head.iter() {
        if cum < k::TOP_P {
            cum += sampler.probs[i as usize];
            count += 1;
        } else {
            break;
        }
    }

    let chosen: Vec<f32> = head[..count]
        .iter()
        .map(|&i| sampler.probs[i as usize])
        .collect();
    let ids: Vec<u32> = head[..count].to_vec();
    multinomial(&chosen, &ids, rng)
}

/// Nucleus sampling, then the repetition guard.
///
/// `scores` is mutated when the guard fires — matching the reference, which writes
/// `-inf` into the offending entry before resampling.
pub fn ras_sampling(
    sampler: &mut Sampler,
    scores: &mut [f32],
    decoded: &[u32],
    rng: &mut Rng,
) -> u32 {
    let top = nucleus(sampler, scores, rng);
    let window = decoded.len().saturating_sub(k::RAS_WIN);
    let repeats = decoded[window..].iter().filter(|&&t| t == top).count();
    if (repeats as f32) < k::RAS_WIN as f32 * k::RAS_TAU {
        return top;
    }
    // Fired: mask the repeat and sample from the *whole* distribution, not the nucleus.
    scores[top as usize] = f32::NEG_INFINITY;
    softmax(sampler, scores);
    let all: Vec<u32> = (0..scores.len() as u32).collect();
    let probs = sampler.probs[..scores.len()].to_vec();
    multinomial(&probs, &all, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(v: &[f32]) -> Vec<f32> {
        v.to_vec()
    }

    #[test]
    fn a_dominant_token_is_always_chosen() {
        // One token at probability ~1: the nucleus is a single token and the draw is
        // forced regardless of the RNG.
        let mut s = Sampler::new(8);
        let mut rng = Rng::new(1);
        let sc = logits(&[0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for _ in 0..50 {
            assert_eq!(ras_sampling(&mut s, &mut sc.clone(), &[], &mut rng), 2);
        }
    }

    #[test]
    fn the_cutoff_crosses_top_p_rather_than_stopping_short() {
        // Probabilities 0.5, 0.3, 0.2 with top_p 0.8: testing before adding keeps
        // {0.5, 0.3} — two tokens, because after adding 0.5 the sum 0.5 is still below
        // 0.8 so the second is admitted, and then 0.8 is not < 0.8 so the third is not.
        // The off-by-one form would keep only the first.
        let mut s = Sampler::new(3);
        let mut rng = Rng::new(2);
        let sc = logits(&[0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()]);
        let mut seen = [0usize; 3];
        for _ in 0..4000 {
            seen[ras_sampling(&mut s, &mut sc.clone(), &[], &mut rng) as usize] += 1;
        }
        assert!(
            seen[0] > 0 && seen[1] > 0,
            "both survivors should appear: {seen:?}"
        );
        assert_eq!(
            seen[2], 0,
            "the third token is outside the nucleus: {seen:?}"
        );
        // Renormalised, the split should be about 5:3.
        let ratio = seen[0] as f64 / seen[1] as f64;
        assert!((ratio - 5.0 / 3.0).abs() < 0.2, "ratio {ratio}");
    }

    #[test]
    fn a_single_repeat_in_the_window_trips_the_guard() {
        // win_size * tau_r = 10 * 0.1 = 1.0, so one repeat is enough. This is the detail
        // that makes the guard the common path rather than a rare fallback.
        let mut s = Sampler::new(4);
        let mut rng = Rng::new(3);
        // Token 1 dominates, but it was just emitted.
        let sc = logits(&[0.0, 40.0, 0.0, 0.0]);
        let mut got_other = false;
        for _ in 0..50 {
            let t = ras_sampling(&mut s, &mut sc.clone(), &[1], &mut rng);
            assert_ne!(t, 1, "the guard must exclude the repeated token");
            got_other = true;
        }
        assert!(got_other);
    }

    #[test]
    fn the_guard_stays_quiet_when_nothing_repeats() {
        let mut s = Sampler::new(4);
        let mut rng = Rng::new(4);
        let sc = logits(&[0.0, 40.0, 0.0, 0.0]);
        // Token 1 is not in the recent window.
        assert_eq!(
            ras_sampling(&mut s, &mut sc.clone(), &[0, 2, 3], &mut rng),
            1
        );
    }

    #[test]
    fn the_window_only_looks_back_ras_win_tokens() {
        let mut s = Sampler::new(4);
        let mut rng = Rng::new(5);
        let sc = logits(&[0.0, 40.0, 0.0, 0.0]);
        // Token 1 appears, but eleven tokens ago — outside a window of ten.
        let mut history = vec![1u32];
        history.extend(std::iter::repeat_n(0, k::RAS_WIN));
        assert_eq!(ras_sampling(&mut s, &mut sc.clone(), &history, &mut rng), 1);
    }
}
