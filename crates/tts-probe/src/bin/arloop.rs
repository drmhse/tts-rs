//! The unmeasured half: the DualAR loop.
//!
//! Everything measured so far has been the codec. The AR loop is the other half
//! of the RTF and had exactly one data point (a 6.72 ms logit projection). This
//! builds the whole thing with random weights at the real geometry — shapes
//! determine cost, so correctness is irrelevant here — and A/B tests the three
//! levers identified in docs/porting/audio8.md / docs/rejected/coreml-and-op-coverage.md:
//!
//!   1. **Logit slice.** The slow AR projects 896 -> 155776 every token, then
//!      `ArkttsSemanticLogitsProcessor` sets all but ids 151678..155773 + eos to
//!      -inf. Only 4097 rows are reachable, so 151679 of them are pure waste.
//!   2. **KV width.** The reference allocates a `max_seq_len`-wide (2048) cache
//!      and attends over all of it with a mask every step, instead of narrowing
//!      to the 0..pos that actually holds keys.
//!   3. **GQA expansion.** The reference does `repeat_interleave(7, dim=1)` on K
//!      and V, materialising a 7x copy of the whole cache every layer every
//!      token. Reshaping the *query* to [b, kv_heads, q_per_kv, d] instead needs
//!      no copy at all.
//!
//! Geometry (config.json): slow = 24 layers, dim 896, 14 heads / 2 KV heads,
//! head_dim 64, ffn 4864, fused wqkv [1152, 896]. fast = 4 layers, same dim,
//! 10 positions per frame (position 0 is the discarded priming step that seeds
//! the fast KV cache — trap 3), fast_output 896 -> 4096.
//!
//! 64 frames at 21.53 Hz = 2.97 s of audio, matching every codec measurement.
//!
//! Run:  cargo run -p tts-probe --release --bin arloop

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{rms_norm, softmax};
use candle_nn::rotary_emb::rope_i;
use tts_probe::bench::Harness;

const DIM: usize = 896;
const N_HEAD: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const FFN: usize = 4864;
const N_LAYER: usize = 24;
const N_FAST_LAYER: usize = 4;
const NUM_CODEBOOKS: usize = 10;
const CODEBOOK_SIZE: usize = 4096;
const VOCAB: usize = 155776;
/// ids 151678..=155773 plus eos — the only rows the semantic mask can leave finite.
const REACHABLE: usize = 4097;
const MAX_SEQ: usize = 2048;
const EPS: f64 = 1e-6;

const PROMPT: usize = 64;
const FRAMES: usize = 64;
const SAMPLES: usize = 5;

/// How K/V get widened from 2 KV heads to 14 query heads.
#[derive(Copy, Clone, PartialEq)]
enum Gqa {
    /// What the reference does: materialise a 7x copy of K and V.
    RepeatInterleave,
    /// Reshape the query instead; no copy.
    QueryReshape,
}

struct Layer {
    // Stored pre-transposed: a port would never transpose per token.
    wqkv_t: Tensor, // [896, 1152]
    bqkv: Tensor,   // [1152]
    wo_t: Tensor,   // [896, 896]
    w1_t: Tensor,   // [896, 4864]
    w3_t: Tensor,   // [896, 4864]
    w2_t: Tensor,   // [4864, 896]
    attn_norm: Tensor,
    ffn_norm: Tensor,
}

impl Layer {
    fn new(dev: &Device, dt: DType) -> Result<Self> {
        let r =
            |a, b| -> Result<Tensor> { Ok(Tensor::randn(0f32, 0.02, (a, b), dev)?.to_dtype(dt)?) };
        Ok(Self {
            wqkv_t: r(DIM, (N_HEAD + 2 * N_KV) * HEAD_DIM)?,
            bqkv: Tensor::randn(0f32, 0.02, (N_HEAD + 2 * N_KV) * HEAD_DIM, dev)?.to_dtype(dt)?,
            wo_t: r(N_HEAD * HEAD_DIM, DIM)?,
            w1_t: r(DIM, FFN)?,
            w3_t: r(DIM, FFN)?,
            w2_t: r(FFN, DIM)?,
            attn_norm: Tensor::ones(DIM, dt, dev)?,
            ffn_norm: Tensor::ones(DIM, dt, dev)?,
        })
    }
}

struct Cache {
    k: Tensor, // [1, N_KV, cap, HEAD_DIM]
    v: Tensor,
}

impl Cache {
    fn new(dev: &Device, dt: DType, cap: usize) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros((1, N_KV, cap, HEAD_DIM), dt, dev)?,
            v: Tensor::zeros((1, N_KV, cap, HEAD_DIM), dt, dev)?,
        })
    }
    fn zero(&mut self, dev: &Device, dt: DType, cap: usize) -> Result<()> {
        self.k = Tensor::zeros((1, N_KV, cap, HEAD_DIM), dt, dev)?;
        self.v = Tensor::zeros((1, N_KV, cap, HEAD_DIM), dt, dev)?;
        Ok(())
    }
}

/// One decode step through one layer. `x` is [1, 1, DIM].
#[allow(clippy::too_many_arguments)]
fn layer_step(
    l: &Layer,
    x: &Tensor,
    cache: &Cache,
    pos: usize,
    attend: usize,
    mask_row: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    gqa: Gqa,
) -> candle_core::Result<Tensor> {
    let h = rms_norm(x, &l.attn_norm, EPS as f32)?;
    let qkv = h
        .reshape((1, DIM))?
        .matmul(&l.wqkv_t)?
        .broadcast_add(&l.bqkv.reshape((1, ()))?)?;
    let qs = N_HEAD * HEAD_DIM;
    let kvs = N_KV * HEAD_DIM;
    let q = qkv
        .narrow(1, 0, qs)?
        .reshape((1, N_HEAD, 1, HEAD_DIM))?
        .contiguous()?;
    let k = qkv
        .narrow(1, qs, kvs)?
        .reshape((1, N_KV, 1, HEAD_DIM))?
        .contiguous()?;
    let v = qkv
        .narrow(1, qs + kvs, kvs)?
        .reshape((1, N_KV, 1, HEAD_DIM))?
        .contiguous()?;

    let cp = cos.narrow(0, pos, 1)?;
    let sp = sin.narrow(0, pos, 1)?;
    let q = rope_i(&q, &cp, &sp)?;
    let k = rope_i(&k, &cp, &sp)?;

    cache.k.slice_set(&k, 2, pos)?;
    cache.v.slice_set(&v, 2, pos)?;
    let kk = cache.k.narrow(2, 0, attend)?;
    let vv = cache.v.narrow(2, 0, attend)?;

    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let ctx = match gqa {
        Gqa::RepeatInterleave => {
            // [1,KV,L,D] -> [1,KV,1,L,D] -> expand 7 -> [1,14,L,D]. The
            // contiguous() is the copy the reference pays.
            let rep = N_HEAD / N_KV;
            let kk = kk
                .unsqueeze(2)?
                .expand((1, N_KV, rep, attend, HEAD_DIM))?
                .contiguous()?
                .reshape((1, N_HEAD, attend, HEAD_DIM))?;
            let vv = vv
                .unsqueeze(2)?
                .expand((1, N_KV, rep, attend, HEAD_DIM))?
                .contiguous()?
                .reshape((1, N_HEAD, attend, HEAD_DIM))?;
            let mut s = (q.matmul(&kk.transpose(2, 3)?.contiguous()?)? * scale)?;
            if let Some(m) = mask_row {
                s = s.broadcast_add(m)?;
            }
            softmax(&s, D::Minus1)?.matmul(&vv)?
        }
        Gqa::QueryReshape => {
            // Fold the 7 query heads that share a KV head into the "rows" of the
            // per-KV-head matmul: [1,KV,7,D] @ [1,KV,D,L] -> [1,KV,7,L].
            let rep = N_HEAD / N_KV;
            let q = q.reshape((1, N_KV, rep, HEAD_DIM))?;
            let mut s = (q.matmul(&kk.transpose(2, 3)?.contiguous()?)? * scale)?;
            if let Some(m) = mask_row {
                s = s.broadcast_add(m)?;
            }
            softmax(&s, D::Minus1)?
                .matmul(&vv.contiguous()?)?
                .reshape((1, N_HEAD, 1, HEAD_DIM))?
        }
    };

    let attn = ctx
        .transpose(1, 2)?
        .contiguous()?
        .reshape((1, qs))?
        .matmul(&l.wo_t)?
        .reshape((1, 1, DIM))?;
    let x = (x + attn)?;

    let h = rms_norm(&x, &l.ffn_norm, EPS as f32)?.reshape((1, DIM))?;
    let gate = candle_nn::ops::silu(&h.matmul(&l.w1_t)?)?;
    let up = h.matmul(&l.w3_t)?;
    let ffn = (gate * up)?.matmul(&l.w2_t)?.reshape((1, 1, DIM))?;
    &x + ffn
}

struct Model {
    slow: Vec<Layer>,
    fast: Vec<Layer>,
    norm: Tensor,
    fast_norm: Tensor,
    lm_head_t: Tensor,   // [896, 155776], tied embeddings
    lm_head_cut: Tensor, // [896, 4097]
    fast_out_t: Tensor,  // [896, 4096]
    cos: Tensor,
    sin: Tensor,
    fast_cos: Tensor,
    fast_sin: Tensor,
    mask: Tensor, // [MAX_SEQ, MAX_SEQ] additive
    fast_mask: Tensor,
}

/// Additive causal mask, `[n, n]`: row i is 0 for j<=i, -inf otherwise.
fn causal_mask(dev: &Device, dt: DType, n: usize) -> Result<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            v[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(v, (n, n), dev)?.to_dtype(dt)?)
}

/// Interleaved-pair RoPE tables, `[n, HEAD_DIM/2]`.
fn rope_tables(dev: &Device, dt: DType, n: usize, base: f64) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let mut c = vec![0f32; n * half];
    let mut s = vec![0f32; n * half];
    for t in 0..n {
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f64 / HEAD_DIM as f64);
            let a = t as f64 * freq;
            c[t * half + i] = a.cos() as f32;
            s[t * half + i] = a.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(c, (n, half), dev)?.to_dtype(dt)?,
        Tensor::from_vec(s, (n, half), dev)?.to_dtype(dt)?,
    ))
}

impl Model {
    fn new(dev: &Device, dt: DType) -> Result<Self> {
        let mut slow = Vec::new();
        for _ in 0..N_LAYER {
            slow.push(Layer::new(dev, dt)?);
        }
        let mut fast = Vec::new();
        for _ in 0..N_FAST_LAYER {
            fast.push(Layer::new(dev, dt)?);
        }
        let lm_head_t = Tensor::randn(0f32, 0.02, (DIM, VOCAB), dev)?.to_dtype(dt)?;
        let lm_head_cut = lm_head_t
            .narrow(1, VOCAB - REACHABLE, REACHABLE)?
            .contiguous()?;
        let (cos, sin) = rope_tables(dev, dt, MAX_SEQ, 1e6)?;
        let (fast_cos, fast_sin) = rope_tables(dev, dt, NUM_CODEBOOKS, 1e6)?;
        Ok(Self {
            slow,
            fast,
            norm: Tensor::ones(DIM, dt, dev)?,
            fast_norm: Tensor::ones(DIM, dt, dev)?,
            lm_head_t,
            lm_head_cut,
            fast_out_t: Tensor::randn(0f32, 0.02, (DIM, CODEBOOK_SIZE), dev)?.to_dtype(dt)?,
            cos,
            sin,
            fast_cos,
            fast_sin,
            mask: causal_mask(dev, dt, MAX_SEQ)?,
            fast_mask: causal_mask(dev, dt, NUM_CODEBOOKS)?,
        })
    }
}

#[derive(Copy, Clone)]
struct Cfg {
    /// Attend over the whole `max_seq_len` cache (reference) vs 0..=pos.
    full_width_kv: bool,
    gqa: Gqa,
    /// Project to all 155776 logits (reference) vs the 4097 reachable rows.
    full_logits: bool,
    /// Run the fast AR's 10 positions per frame.
    with_fast: bool,
}

/// One full generation: PROMPT tokens of prefill-as-decode-steps, then FRAMES
/// frames. Prefill is run token-by-token rather than as one batched forward —
/// that understates the reference slightly, but it is identical across variants
/// so the ratios are unaffected, and it keeps the harness to one code path.
fn generate(
    m: &Model,
    dev: &Device,
    dt: DType,
    caches: &mut [Cache],
    fc: &mut [Cache],
    cfg: Cfg,
) -> Result<()> {
    for c in caches.iter_mut() {
        c.zero(dev, dt, MAX_SEQ)?;
    }
    let mut x = Tensor::randn(0f32, 1.0, (1, 1, DIM), dev)?.to_dtype(dt)?;
    for pos in 0..(PROMPT + FRAMES) {
        let attend = if cfg.full_width_kv { MAX_SEQ } else { pos + 1 };
        let mask_row = if cfg.full_width_kv {
            Some(m.mask.narrow(0, pos, 1)?.reshape((1, 1, 1, MAX_SEQ))?)
        } else {
            None
        };
        let mut h = x.clone();
        for (l, c) in m.slow.iter().zip(caches.iter()) {
            h = layer_step(
                l,
                &h,
                c,
                pos,
                attend,
                mask_row.as_ref(),
                &m.cos,
                &m.sin,
                cfg.gqa,
            )?;
        }
        let normed = rms_norm(&h, &m.norm, EPS as f32)?.reshape((1, DIM))?;
        let head = if cfg.full_logits {
            &m.lm_head_t
        } else {
            &m.lm_head_cut
        };
        let logits = normed.matmul(head)?;
        // Stand in for sampling: a reduction that forces the logits to exist.
        let _ = logits.max(D::Minus1)?;

        if pos >= PROMPT && cfg.with_fast {
            for c in fc.iter_mut() {
                c.zero(dev, dt, NUM_CODEBOOKS)?;
            }
            // fast_project_in is Identity (fast_dim == dim), verified by the
            // fixture aliasing in dump_fixtures.py.
            let mut fh = normed.reshape((1, 1, DIM))?;
            for p in 0..NUM_CODEBOOKS {
                let mr = m
                    .fast_mask
                    .narrow(0, p, 1)?
                    .reshape((1, 1, 1, NUM_CODEBOOKS))?;
                let mut t = fh.clone();
                for (l, c) in m.fast.iter().zip(fc.iter()) {
                    t = layer_step(
                        l,
                        &t,
                        c,
                        p,
                        NUM_CODEBOOKS,
                        Some(&mr),
                        &m.fast_cos,
                        &m.fast_sin,
                        cfg.gqa,
                    )?;
                }
                let out = rms_norm(&t, &m.fast_norm, EPS as f32)?
                    .reshape((1, DIM))?
                    .matmul(&m.fast_out_t)?;
                let _ = out.max(D::Minus1)?;
                fh = t;
            }
        }
        x = Tensor::randn(0f32, 1.0, (1, 1, DIM), dev)?.to_dtype(dt)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let dt = match std::env::args().nth(1).as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => DType::F32,
    };
    println!("arloop probe — dtype {dt:?}, {PROMPT} prompt + {FRAMES} frames (2.97 s of audio)");
    println!(
        "slow: {N_LAYER} layers x {} steps = {} layer-steps",
        PROMPT + FRAMES,
        N_LAYER * (PROMPT + FRAMES)
    );
    println!(
        "fast: {N_FAST_LAYER} layers x {NUM_CODEBOOKS} positions x {FRAMES} frames = {} layer-steps",
        N_FAST_LAYER * NUM_CODEBOOKS * FRAMES
    );

    let m = Model::new(&dev, dt)?;
    let mut h = Harness::new(&dev, SAMPLES)?;

    let reference = Cfg {
        full_width_kv: true,
        gqa: Gqa::RepeatInterleave,
        full_logits: true,
        with_fast: true,
    };

    // Slow AR only first, so each lever is visible without the fast AR's
    // constant cost diluting it.
    let variants: Vec<(&str, Cfg)> = vec![
        (
            "reference (2048 KV, repeat, 155776)",
            Cfg {
                with_fast: false,
                ..reference
            },
        ),
        (
            "+ narrow KV to 0..pos",
            Cfg {
                with_fast: false,
                full_width_kv: false,
                ..reference
            },
        ),
        (
            "+ GQA via query reshape",
            Cfg {
                with_fast: false,
                full_width_kv: false,
                gqa: Gqa::QueryReshape,
                ..reference
            },
        ),
        (
            "+ logits sliced to 4097",
            Cfg {
                with_fast: false,
                full_width_kv: false,
                gqa: Gqa::QueryReshape,
                full_logits: false,
            },
        ),
    ];

    {
        let mut fns: Vec<Box<dyn FnMut() -> candle_core::Result<()>>> = Vec::new();
        for (_, cfg) in &variants {
            let cfg = *cfg;
            let mp = &m;
            let dp = &dev;
            // Each closure gets its own caches so nothing is shared mutably.
            let mut cs: Vec<Cache> = (0..N_LAYER)
                .map(|_| Cache::new(dp, dt, MAX_SEQ).unwrap())
                .collect();
            let mut fs: Vec<Cache> = (0..N_FAST_LAYER)
                .map(|_| Cache::new(dp, dt, NUM_CODEBOOKS).unwrap())
                .collect();
            fns.push(Box::new(move || {
                generate(mp, dp, dt, &mut cs, &mut fs, cfg)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))
            }));
        }
        let mut refs: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
        for ((name, _), f) in variants.iter().zip(fns.iter_mut()) {
            refs.push((name, f.as_mut()));
        }
        h.ab("slow AR, 128 steps — stacked levers", &mut refs)?;
    }

    // Now the whole DualAR loop, best config vs reference, to size the AR half
    // against the codec's measured cost.
    {
        let full_ref = reference;
        let full_best = Cfg {
            full_width_kv: false,
            gqa: Gqa::QueryReshape,
            full_logits: false,
            with_fast: true,
        };
        let mut cs1: Vec<Cache> = (0..N_LAYER)
            .map(|_| Cache::new(&dev, dt, MAX_SEQ).unwrap())
            .collect();
        let mut fs1: Vec<Cache> = (0..N_FAST_LAYER)
            .map(|_| Cache::new(&dev, dt, NUM_CODEBOOKS).unwrap())
            .collect();
        let mut cs2: Vec<Cache> = (0..N_LAYER)
            .map(|_| Cache::new(&dev, dt, MAX_SEQ).unwrap())
            .collect();
        let mut fs2: Vec<Cache> = (0..N_FAST_LAYER)
            .map(|_| Cache::new(&dev, dt, NUM_CODEBOOKS).unwrap())
            .collect();
        let (mp, dp) = (&m, &dev);
        let mut f1 = move || {
            generate(mp, dp, dt, &mut cs1, &mut fs1, full_ref)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };
        let mut f2 = move || {
            generate(mp, dp, dt, &mut cs2, &mut fs2, full_best)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };
        let stats = h.ab(
            "full DualAR loop (slow + fast)",
            &mut [
                (
                    "reference",
                    &mut f1 as &mut dyn FnMut() -> candle_core::Result<()>,
                ),
                ("all levers", &mut f2),
            ],
        )?;
        let best = stats[1].median;
        println!(
            "\nAR loop for 2.97 s of audio: {:.1} ms reference, {:.1} ms optimised -> RTF {:.3} / {:.3}",
            stats[0].median,
            best,
            stats[0].median / 2970.0,
            best / 2970.0
        );
    }

    h.report_drift()?;
    Ok(())
}
