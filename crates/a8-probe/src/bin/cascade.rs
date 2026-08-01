//! Full Audio8 codec decode cascade in Candle, with random weights.
//!
//! The op-by-op probe (`a8-probe`) extrapolated a total by arithmetic. That is not
//! good enough to decide a port against, because PyTorch/MPS gives a single
//! measured number (568.9 ms f32 / 403.6 ms fp16 for 64 frames). This builds the
//! *entire* decode graph at the real shapes and times it end to end, so the two
//! are directly comparable.
//!
//! Correctness is irrelevant here -- weights are random. Only shapes matter, and
//! shapes determine cost.
//!
//! Run:  cargo run -p a8-probe --release --bin cascade

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const FRAMES: usize = 64;
const ITERS: usize = 3;

/// Depthwise conv as shifted broadcast-muls: 19x faster than candle's grouped
/// conv on Metal (37.78 ms -> 1.96 ms). See docs/rejected/coreml-and-op-coverage.md.
fn depthwise_k7(x: &Tensor, taps: &[Tensor]) -> candle_core::Result<Tensor> {
    let len = x.dim(2)?;
    let padded = x.pad_with_zeros(2, 6, 0)?;
    let mut acc: Option<Tensor> = None;
    for (k, tap) in taps.iter().enumerate() {
        let term = padded.narrow(2, k, len)?.broadcast_mul(tap)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    acc.unwrap().contiguous()
}

fn snake(x: &Tensor, alpha: &Tensor) -> candle_core::Result<Tensor> {
    let recip = (alpha + 1e-9)?.recip()?;
    x + x
        .broadcast_mul(alpha)?
        .sin()?
        .sqr()?
        .broadcast_mul(&recip)?
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let v = x.sqr()?.mean_keepdim(x.rank() - 1)?;
    x.broadcast_div(&(v + eps)?.sqrt()?)?.broadcast_mul(w)
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let last = x.rank() - 1;
    let mean = x.mean_keepdim(last)?;
    let centred = x.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(last)?;
    centred
        .broadcast_div(&(var + 1e-6)?.sqrt()?)?
        .broadcast_mul(w)?
        .broadcast_add(b)
}

/// Causal padding for stride-1 convs: left-pad by (k-1)*dilation so output length
/// equals input length, matching ArkttsCausalConv1d for stride 1.
fn causal_conv(x: &Tensor, w: &Tensor, k: usize, dilation: usize) -> candle_core::Result<Tensor> {
    let pad = (k - 1) * dilation;
    let x = if pad > 0 {
        x.pad_with_zeros(2, pad, 0)?
    } else {
        x.clone()
    };
    x.conv1d(w, 0, 1, dilation, 1)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift64*, host-side only, so no Date/rand dependency
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / 16_777_216.0) - 0.5
    }
}

fn rand_tensor(shape: &[usize], dt: DType, dev: &Device, rng: &mut Rng) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next() * 0.05).collect();
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dt)?)
}

struct ResidualUnit {
    alpha1: Tensor,
    alpha2: Tensor,
    w7: Tensor,
    w1: Tensor,
    dilation: usize,
}

impl ResidualUnit {
    fn new(ch: usize, dilation: usize, dt: DType, dev: &Device, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            alpha1: Tensor::ones((1, ch, 1), dt, dev)?,
            alpha2: Tensor::ones((1, ch, 1), dt, dev)?,
            w7: rand_tensor(&[ch, ch, 7], dt, dev, rng)?,
            w1: rand_tensor(&[ch, ch, 1], dt, dev, rng)?,
            dilation,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = snake(x, &self.alpha1)?;
        let h = causal_conv(&h, &self.w7, 7, self.dilation)?;
        let h = snake(&h, &self.alpha2)?;
        let h = h.conv1d(&self.w1, 0, 1, 1, 1)?;
        x + h
    }
}

struct ConvNeXt {
    taps: Vec<Tensor>,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Tensor,
    pw2: Tensor,
    gamma: Tensor,
}

impl ConvNeXt {
    fn new(dim: usize, dt: DType, dev: &Device, rng: &mut Rng) -> Result<Self> {
        let dw = rand_tensor(&[dim, 1, 7], dt, dev, rng)?;
        let taps = (0..7)
            .map(|k| dw.narrow(2, k, 1)?.reshape((1, dim, 1)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            taps,
            ln_w: Tensor::ones(dim, dt, dev)?,
            ln_b: Tensor::zeros(dim, dt, dev)?,
            pw1: rand_tensor(&[dim, 4 * dim], dt, dev, rng)?,
            pw2: rand_tensor(&[4 * dim, dim], dt, dev, rng)?,
            gamma: Tensor::ones(dim, dt, dev)?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = depthwise_k7(x, &self.taps)?.transpose(1, 2)?.contiguous()?;
        let h = layer_norm(&h, &self.ln_w, &self.ln_b)?;
        let h = h.broadcast_matmul(&self.pw1)?.gelu_erf()?;
        let h = h.broadcast_matmul(&self.pw2)?.broadcast_mul(&self.gamma)?;
        x + h.transpose(1, 2)?.contiguous()?
    }
}

/// One block of the codec's windowed transformer (dim 1024, 16 heads / 8 kv).
struct CodecBlock {
    wqkv: Tensor,
    wo: Tensor,
    w1: Tensor,
    w3: Tensor,
    w2: Tensor,
    n_attn: Tensor,
    n_ffn: Tensor,
    ls_attn: Tensor,
    ls_ffn: Tensor,
}

impl CodecBlock {
    fn new(dt: DType, dev: &Device, rng: &mut Rng) -> Result<Self> {
        let (dim, heads, kv, hd, ffn) = (1024usize, 16usize, 8usize, 64usize, 1216usize);
        Ok(Self {
            wqkv: rand_tensor(&[dim, (heads + 2 * kv) * hd], dt, dev, rng)?,
            wo: rand_tensor(&[heads * hd, dim], dt, dev, rng)?,
            w1: rand_tensor(&[dim, ffn], dt, dev, rng)?,
            w3: rand_tensor(&[dim, ffn], dt, dev, rng)?,
            w2: rand_tensor(&[ffn, dim], dt, dev, rng)?,
            n_attn: Tensor::ones(dim, dt, dev)?,
            n_ffn: Tensor::ones(dim, dt, dev)?,
            ls_attn: Tensor::ones(dim, dt, dev)?,
            ls_ffn: Tensor::ones(dim, dt, dev)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (heads, kv, hd) = (16usize, 8usize, 64usize);
        let h = rms_norm(x, &self.n_attn, 1e-5)?;
        let qkv = h.broadcast_matmul(&self.wqkv)?;
        let q = qkv
            .narrow(2, 0, heads * hd)?
            .reshape((b, t, heads, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, heads * hd, kv * hd)?
            .reshape((b, t, kv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, (heads + kv) * hd, kv * hd)?
            .reshape((b, t, kv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        // GQA by repeat: 16 query heads over 8 kv heads.
        let k = Tensor::cat(&[&k, &k], 1)?;
        let v = Tensor::cat(&[&v, &v], 1)?;
        let scores = (q.matmul(&k.transpose(2, 3)?)? / (hd as f64).sqrt())?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
        let attn = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, heads * hd))?;
        let x = (x + attn
            .broadcast_matmul(&self.wo)?
            .broadcast_mul(&self.ls_attn)?)?;

        let h = rms_norm(&x, &self.n_ffn, 1e-5)?;
        let gate = candle_nn::ops::silu(&h.broadcast_matmul(&self.w1)?)?;
        let ffn = (gate * h.broadcast_matmul(&self.w3)?)?.broadcast_matmul(&self.w2)?;
        x + ffn.broadcast_mul(&self.ls_ffn)?
    }
}

fn window_mask(len: usize, window: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; len * len];
    for i in 0..len {
        let lo = (i as i64 - window as i64 + 1).max(0) as usize;
        for j in 0..len {
            if j > i || j < lo {
                m[i * len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, len, len), dev)?.to_dtype(dt)?)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut rng = Rng(0x9E3779B97F4A7C15);

    for dt in [DType::F32, DType::F16] {
        println!("\n=== dtype {dt:?} ===");
        let t0 = Instant::now();

        // ---- rvq from_codes: 10 codebooks -> summed 1024-channel latent
        let codes = Tensor::ones((10, FRAMES), DType::U32, &dev)?;
        let mut books = Vec::new();
        for i in 0..10 {
            let size = if i == 0 { 4096 } else { 1024 };
            books.push((
                rand_tensor(&[size, 8], dt, &dev, &mut rng)?,
                rand_tensor(&[1024, 8, 1], dt, &dev, &mut rng)?,
            ));
        }

        // ---- post_module: 8 layers, window 128
        let post: Vec<CodecBlock> = (0..8)
            .map(|_| CodecBlock::new(dt, &dev, &mut rng))
            .collect::<Result<Vec<_>>>()?;
        let post_norm = Tensor::ones(1024, dt, &dev)?;
        let mask = window_mask(FRAMES, 128, dt, &dev)?;

        // ---- quantizer upsample: 2 x (convT s2 + ConvNeXt)
        let up: Vec<(Tensor, ConvNeXt)> = (0..2)
            .map(|_| -> Result<_> {
                Ok((
                    rand_tensor(&[1024, 1024, 2], dt, &dev, &mut rng)?,
                    ConvNeXt::new(1024, dt, &dev, &mut rng)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // ---- decoder: entry conv, 4 blocks (strides 8,8,4,2), tail
        let entry = rand_tensor(&[1536, 1024, 7], dt, &dev, &mut rng)?;
        let mut blocks = Vec::new();
        let mut ch = 1536usize;
        for stride in [8usize, 8, 4, 2] {
            let out = ch / 2;
            let alpha = Tensor::ones((1, ch, 1), dt, &dev)?;
            let convt = rand_tensor(&[ch, out, 2 * stride], dt, &dev, &mut rng)?;
            let units = [1usize, 3, 9]
                .iter()
                .map(|d| ResidualUnit::new(out, *d, dt, &dev, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            blocks.push((alpha, convt, stride, units));
            ch = out;
        }
        let tail_alpha = Tensor::ones((1, ch, 1), dt, &dev)?;
        let tail = rand_tensor(&[1, ch, 7], dt, &dev, &mut rng)?;
        println!("built weights in {:.1}s", t0.elapsed().as_secs_f64());

        let run = || -> candle_core::Result<Tensor> {
            // rvq
            let mut z: Option<Tensor> = None;
            for (i, (book, proj)) in books.iter().enumerate() {
                let idx = codes.narrow(0, i, 1)?.flatten_all()?;
                let e = book
                    .index_select(&idx, 0)?
                    .reshape((1, FRAMES, 8))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let term = e.conv1d(proj, 0, 1, 1, 1)?;
                z = Some(match z {
                    None => term,
                    Some(a) => (a + term)?,
                });
            }
            let z = z.unwrap();

            // post_module (channels_first -> transpose in and out)
            let mut h = z.transpose(1, 2)?.contiguous()?;
            for block in &post {
                h = block.forward(&h, &mask)?;
            }
            let h = rms_norm(&h, &post_norm, 1e-5)?;
            let mut x = h.transpose(1, 2)?.contiguous()?;

            // upsample
            for (convt, cnx) in &up {
                x = x.conv_transpose1d(convt, 0, 0, 2, 1, 1)?;
                x = cnx.forward(&x)?;
            }

            // decoder
            let mut x = causal_conv(&x, &entry, 7, 1)?;
            for (alpha, convt, stride, units) in &blocks {
                x = snake(&x, alpha)?;
                x = x.conv_transpose1d(convt, 0, 0, *stride, 1, 1)?;
                for unit in units {
                    x = unit.forward(&x)?;
                }
            }
            let x = snake(&x, &tail_alpha)?;
            causal_conv(&x, &tail, 7, 1)?.tanh()
        };

        let warm = run()?;
        dev.synchronize()?;
        let samples = warm.dim(2)?;
        let audio_s = samples as f64 / 44100.0;
        println!("output {:?} = {:.2} s audio", warm.dims(), audio_s);
        drop(warm);

        let start = Instant::now();
        for _ in 0..ITERS {
            let out = run()?;
            dev.synchronize()?;
            drop(out);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
        println!(
            "candle/metal full decode: {ms:8.1} ms   RTF {:.3}",
            ms / 1000.0 / audio_s
        );
        println!("  torch/mps reference:      568.9 ms f32 / 403.6 ms fp16  (RTF 0.191 / 0.136)");
    }
    Ok(())
}
