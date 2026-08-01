//! `audio8` — text to 44.1 kHz speech, one process, no supervisor.
//!
//! Run:
//!   cargo run -p audio8 --release --bin a8 -- \
//!       --text-file examples/senior.txt --out examples/senior_rust.wav \
//!       --reference-codes fixtures/audio8/default_voice_codes.safetensors \
//!       --reference-text ../CosyVoice/asset/default_voice.txt

use audio8::ar::{GenConfig, Model};
use audio8::cfg;
use audio8::codec::Codec;
use audio8::prompt::{load_reference_codes, segment, PromptBuilder};
use audio8::sample::Rng;
use audio8::wav;
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Audio8-TTS in Rust", long_about = None)]
struct Args {
    /// Text to speak, or use --text-file.
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    text_file: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,

    #[arg(long, default_value = "references/audio8/weights/model.safetensors")]
    weights: String,
    #[arg(long, default_value = "references/audio8/weights/codec.safetensors")]
    codec: String,
    #[arg(long, default_value = "references/audio8/weights/tokenizer.json")]
    tokenizer: String,

    /// Reference codes from `synthesize.py --save-reference-codes`.
    #[arg(long)]
    reference_codes: Option<String>,
    /// Transcript of the reference clip, or a path to it.
    #[arg(long)]
    reference_text: Option<String>,

    /// Weight format for the 28 layers' projections: q8_0, q5_0, q4_1, q4_0, or none.
    #[arg(long, default_value = "q8_0")]
    quant: String,

    #[arg(long, default_value_t = 220)]
    max_chars: usize,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 50)]
    top_k: usize,
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    /// Decode greedily instead of sampling.
    #[arg(long)]
    greedy: bool,

    #[arg(long, default_value_t = 90)]
    gap_ms: usize,
    #[arg(long, default_value_t = 320)]
    para_gap_ms: usize,

    #[arg(long)]
    cpu: bool,
}

fn parse_quant(s: &str) -> Result<Option<GgmlDType>> {
    Ok(match s {
        "none" | "f32" => None,
        "q8_0" => Some(GgmlDType::Q8_0),
        "q5_0" => Some(GgmlDType::Q5_0),
        "q4_1" => Some(GgmlDType::Q4_1),
        "q4_0" => Some(GgmlDType::Q4_0),
        // The K-quants need k divisible by 256 and dim is 896, so they cannot load at
        // all. Fail with the reason rather than a shape error 30 seconds in.
        "q4_K" | "q5_K" | "q6_K" => {
            anyhow::bail!("{s} uses 256-element blocks; every k=896 projection fails. Use q8_0.")
        }
        other => anyhow::bail!("unknown quant {other}"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = match (&args.text, &args.text_file) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => std::fs::read_to_string(p)?,
        (None, None) => anyhow::bail!("pass --text or --text-file"),
    };

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let quant = parse_quant(&args.quant)?;

    let paragraphs = segment(&text, args.max_chars);
    let flat: Vec<(usize, String)> = paragraphs
        .iter()
        .enumerate()
        .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s.clone())))
        .collect();
    anyhow::ensure!(!flat.is_empty(), "no text to speak");
    println!(
        "{} paragraphs -> {} segments (<= {} chars)",
        paragraphs.len(),
        flat.len(),
        args.max_chars
    );

    let t_load = Instant::now();
    let builder = PromptBuilder::load(&args.tokenizer)?;
    let model = Model::load(&args.weights, &device, quant)?;
    let codec = Codec::load(&args.codec, &device)?;
    println!(
        "loaded in {:.1} s  (projections: {}, embeddings and heads: f32)",
        t_load.elapsed().as_secs_f64(),
        args.quant
    );

    // Reference voice: codes were encoded offline, so the codec encoder is not in this
    // binary at all.
    let ref_codes = match &args.reference_codes {
        None => None,
        Some(p) => Some(load_reference_codes(p)?),
    };
    let ref_text = match &args.reference_text {
        None => None,
        Some(t) => {
            let path = PathBuf::from(t);
            Some(if path.is_file() {
                std::fs::read_to_string(path)?
            } else {
                t.clone()
            })
        }
    };
    if let Some(codes) = &ref_codes {
        anyhow::ensure!(
            ref_text.is_some(),
            "--reference-text is required with --reference-codes: the prompt \
             interleaves the clip's transcript with its codes"
        );
        println!(
            "reference voice: {} code frames ({:.2} s)",
            codes[0].len(),
            codes[0].len() as f64 / cfg::frame_rate()
        );
    }

    let gen = GenConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        do_sample: !args.greedy,
    };
    let mut rng = Rng::new(args.seed);

    let mut pieces: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut total_frames = 0usize;
    let mut ar_total = 0f64;
    let mut codec_total = 0f64;
    let t0 = Instant::now();
    for (i, (pi, seg)) in flat.iter().enumerate() {
        let reference = match (&ref_codes, &ref_text) {
            (Some(c), Some(t)) => Some((c.as_slice(), t.as_str())),
            _ => None,
        };
        let prompt = builder.build(seg, reference)?;
        let t_seg = Instant::now();
        let codes = model.generate(&prompt, &gen, &mut rng)?;
        let ar_s = t_seg.elapsed().as_secs_f64();
        let frames = codes[0].len();
        if frames == 0 {
            println!("  [warn] segment {} produced no frames: {seg:?}", i + 1);
            continue;
        }
        total_frames += frames;
        let t_codec = Instant::now();
        let audio = codec.decode(&codes)?;
        let samples = audio.flatten_all()?.to_vec1::<f32>()?;
        let codec_s = t_codec.elapsed().as_secs_f64();
        ar_total += ar_s;
        codec_total += codec_s;
        println!(
            "  seg {}/{}: {frames} frames, {:.2} s audio  ar {:.2} s  codec {:.2} s",
            i + 1,
            flat.len(),
            frames as f64 / cfg::frame_rate(),
            ar_s,
            codec_s
        );
        pieces.push((*pi, samples));
    }
    let elapsed = t0.elapsed().as_secs_f64();

    // Short silence inside a paragraph, longer between paragraphs.
    let gap = wav::silence(cfg::SAMPLE_RATE, args.gap_ms);
    let para_gap = wav::silence(cfg::SAMPLE_RATE, args.para_gap_ms);
    let mut audio: Vec<f32> = Vec::new();
    let mut prev: Option<usize> = None;
    for (pi, samples) in &pieces {
        if let Some(p) = prev {
            audio.extend_from_slice(if *pi != p { &para_gap } else { &gap });
        }
        audio.extend_from_slice(samples);
        prev = Some(*pi);
    }
    anyhow::ensure!(!audio.is_empty(), "no audio was produced");

    wav::write_mono(&args.out, &audio, cfg::SAMPLE_RATE as u32)?;
    let seconds = audio.len() as f64 / cfg::SAMPLE_RATE as f64;
    println!(
        "\n{seconds:.2} s of audio at {} Hz in {elapsed:.1} s  ->  RTF {:.3}",
        cfg::SAMPLE_RATE,
        elapsed / seconds
    );
    println!(
        "  AR    {ar_total:>7.1} s ({:>5.1}%)  RTF {:.3}\n  codec {codec_total:>7.1} s ({:>5.1}%)  RTF {:.3}",
        100.0 * ar_total / elapsed,
        ar_total / seconds,
        100.0 * codec_total / elapsed,
        codec_total / seconds
    );
    println!("{total_frames} frames total, {} weights", args.quant);
    println!("wrote {}", args.out.display());
    Ok(())
}
