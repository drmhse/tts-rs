//! Sampling, ported exactly — including the parts that look like bugs.
//!
//! Three details here are load-bearing and each is easy to "fix" into a port that
//! sounds subtly wrong:
//!
//! 1. **The legacy filter softmaxes before temperature.** `_processed_scores` runs
//!    top-k/top-p first and divides by temperature *after*, so temperature does not
//!    influence which candidates survive — only how the survivors are weighted.
//!    Applying temperature first is the natural implementation and it is not this
//!    model.
//! 2. **Selection is Gumbel-max, not inverse-CDF.** `argmax(softmax(s) / -log(u))`,
//!    with one uniform per vocabulary entry rather than one per step.
//! 3. **The draw must be f32.** The reference builds its uniforms with
//!    `dtype=probabilities.dtype`, so under the checkpoint's native bfloat16 both the
//!    probabilities and the noise carry an 8-bit mantissa and the ratio ordering
//!    collapses — measured: unintelligible output that never reaches EOS. See
//!    `docs/reference.md`. Doing this in f32 is a deliberate divergence from
//!    the reference's behaviour under bf16, and it matches its behaviour under f32.

/// Re-exported from `tts-core`: the PRNG is engine-neutral and both engines draw from
/// it. Its documented limit applies here — it cannot reproduce torch's Philox stream,
/// which is why the fixture gate compares greedy output rather than sampled output.
pub use tts_core::rng::Rng;

fn softmax_into(scores: &[f32], out: &mut [f32]) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (o, &s) in out.iter_mut().zip(scores.iter()) {
        // exp(-inf - max) is 0, which is what the mask wants, but -inf - -inf is NaN,
        // so the fully-masked case is handled explicitly rather than arithmetically.
        let e = if s == f32::NEG_INFINITY || !max.is_finite() {
            0.0
        } else {
            (s - max).exp()
        };
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    }
}

/// `ArkttsLegacyTopKTopPLogitsProcessor` followed by the temperature divide.
///
/// Operates in place. `scores` may already contain `-inf` entries (the semantic mask);
/// they sort last, contribute nothing to the cumulative sum, and stay masked.
/// Reusable buffers for the sampler.
///
/// Not premature: the AR loop calls [`processed_scores`] and [`gumbel_argmax`] **eleven
/// times per frame per sequence** — twice for the semantic token's two RAS draws, nine times
/// for the residual codebooks — over ~4096 entries each. At batch 8 that is 88 calls a step,
/// and this is the one part of a decode step that scales linearly with the batch and
/// amortises nothing, which is precisely what caps the batching gain at ~2x. Three heap
/// allocations per call was a measurable share of it.
pub struct Scratch {
    probs: Vec<f32>,
    order: Vec<u32>,
    remove: Vec<bool>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Self {
            probs: vec![0.0; n],
            order: (0..n as u32).collect(),
            remove: vec![false; n],
        }
    }

    fn fit(&mut self, n: usize) {
        if self.probs.len() != n {
            self.probs.resize(n, 0.0);
            self.order = (0..n as u32).collect();
            self.remove.resize(n, false);
        }
    }
}

/// `ArkttsLegacyTopKTopPLogitsProcessor` followed by the temperature divide, using a
/// caller-supplied scratch.
///
/// **Only the top `top_k` ranks can survive**, because the reference removes every entry at
/// `rank >= top_k` unconditionally. So the full `n log n` sort the obvious implementation
/// does is wasted: a linear-time selection of the top `top_k` followed by sorting just those
/// is the same function. With `n = 4096` and `top_k = 50` that replaces ~49000 indirect
/// comparisons with ~4100 plus a 50-element sort.
///
/// Ordering is by value descending, index ascending — a total order, so unlike the previous
/// `sort_unstable_by` the result no longer depends on how the sort happened to break ties.
pub fn processed_scores_with(
    scratch: &mut Scratch,
    scores: &mut [f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
) {
    let n = scores.len();
    scratch.fit(n);
    softmax_into(scores, &mut scratch.probs);

    let keep = top_k.clamp(1, n);
    let by_score = |a: &u32, b: &u32| {
        let (x, y) = (scores[*a as usize], scores[*b as usize]);
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if keep < n {
        scratch.order.select_nth_unstable_by(keep - 1, by_score);
    }
    scratch.order[..keep].sort_unstable_by(by_score);

    // Everything outside the top-k is removed; inside it, removal follows the cumulative
    // sum. `remove_sorted[..., 0] = False` — the top candidate always survives, even if its
    // own probability already exceeds top_p.
    scratch.remove.iter_mut().for_each(|r| *r = true);
    let mut cumulative = 0f32;
    for (rank, &i) in scratch.order[..keep].iter().enumerate() {
        cumulative += scratch.probs[i as usize];
        if rank == 0 || cumulative <= top_p {
            scratch.remove[i as usize] = false;
        }
    }

    let t = temperature.max(1e-5);
    for (s, &drop) in scores.iter_mut().zip(scratch.remove.iter()) {
        if drop {
            *s = f32::NEG_INFINITY;
        } else {
            *s /= t;
        }
    }
}

/// Allocating convenience wrapper, for callers outside the hot loop.
pub fn processed_scores(scores: &mut [f32], top_k: usize, top_p: f32, temperature: f32) {
    let mut scratch = Scratch::new(scores.len());
    processed_scores_with(&mut scratch, scores, top_k, top_p, temperature);
}

/// Gumbel-max: `argmax(softmax(scores) / -ln(u))`.
///
/// **Draws only for entries that can win.** After [`processed_scores`] all but at most
/// `top_k` entries are `-inf`, hence probability exactly zero, hence `p / -ln(u) == 0` for
/// any draw — they cannot beat a surviving candidate and they cannot beat each other. So the
/// ~4046 draws and `ln()` calls the obvious loop spends on them buy nothing, and skipping
/// them selects from an identical distribution.
///
/// It does mean the RNG stream advances differently, so audio rendered before this change is
/// not reproducible after it. That was already true across any change to sampling order, and
/// it cannot be compared against the reference's stream regardless — see `tts_core::rng`.
/// Greedy decoding draws nothing, so the fixture gate is untouched.
pub fn gumbel_argmax_with(scratch: &mut Scratch, scores: &[f32], rng: &mut Rng) -> usize {
    scratch.fit(scores.len());
    softmax_into(scores, &mut scratch.probs);
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &p) in scratch.probs.iter().enumerate() {
        if p == 0.0 {
            // Cannot win: `0 / noise` is 0, and index 0 already holds the running best at
            // -inf, so the outcome is the same as drawing and discarding.
            continue;
        }
        let v = p / -rng.next_f32().ln();
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Allocating convenience wrapper.
pub fn gumbel_argmax(scores: &[f32], rng: &mut Rng) -> usize {
    let mut scratch = Scratch::new(scores.len());
    gumbel_argmax_with(&mut scratch, scores, rng)
}

/// Plain argmax, for greedy decoding.
pub fn argmax(scores: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best_val {
            best_val = s;
            best = i;
        }
    }
    best
}
