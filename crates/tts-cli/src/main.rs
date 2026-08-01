//! `tts` — synthesize with a chosen engine.
//!
//! ```text
//! tts engines                                   # what exists, and what works
//! tts speak --engine audio8 --text "hello" --out out.wav
//! tts speak --engine audio8 --text-file book.txt --voice voices/cosy-default --out out.wav
//! ```

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tts_core::{Cloning, EngineConfig, Gaps, Sampling, SynthesisRequest, Voice};

#[derive(Parser)]
#[command(name = "tts", about = "Local text-to-speech, pick your engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

// `Speak` carries every synthesis flag and dwarfs the other two variants. Boxing it
// would buy a few bytes on a value constructed once per process.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// List engines and their capabilities.
    Engines,
    /// Describe a voice asset without synthesizing.
    Voice { path: PathBuf },
    /// Synthesize speech.
    Speak(Speak),
}

#[derive(Args)]
struct Speak {
    /// Engine id; omit for the first available one.
    #[arg(long)]
    engine: Option<String>,

    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    text_file: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,

    /// Voice asset directory (`voice.json` + `voice.safetensors`).
    #[arg(long)]
    voice: Option<PathBuf>,

    /// Engine model root; defaults per engine.
    #[arg(long)]
    model_root: Option<PathBuf>,
    /// Override a specific file, e.g. `--set codec=oracle/weights/codec.safetensors`.
    #[arg(long = "set", value_parser = parse_override)]
    overrides: Vec<(String, PathBuf)>,

    /// Weight format; engine-specific, see `tts engines`.
    #[arg(long)]
    quant: Option<String>,
    #[arg(long)]
    cpu: bool,

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
    #[arg(long)]
    greedy: bool,
    #[arg(long, default_value_t = 90)]
    gap_ms: usize,
    #[arg(long, default_value_t = 320)]
    para_gap_ms: usize,
}

fn parse_override(s: &str) -> Result<(String, PathBuf), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected key=path, got {s:?}"))?;
    Ok((k.to_string(), PathBuf::from(v)))
}

fn cmd_engines() -> Result<()> {
    println!(
        "{:<12} {:>10} {:>9} {:>8} {:>9}  weights",
        "engine", "sample rate", "cloning", "stream", "state"
    );
    println!("{}", "-".repeat(92));
    for c in tts_engines::catalogue() {
        println!(
            "{:<12} {:>10} {:>9} {:>8} {:>9}  {}",
            c.id,
            format!("{} Hz", c.sample_rate),
            match c.cloning {
                Cloning::None => "no",
                Cloning::PrecomputedAsset => "asset",
            },
            if c.streaming { "yes" } else { "no" },
            if c.available { "ready" } else { "staged" },
            c.quantization.join(", ")
        );
        println!("             {}", c.description);
        if let Some(reason) = c.reason {
            println!("             unavailable: {reason}");
        }
    }
    println!("\ndefault engine: {}", tts_engines::default_id());
    Ok(())
}

fn cmd_voice(path: &Path) -> Result<()> {
    let voice = Voice::load(path)?;
    println!("name    {}", voice.name);
    println!("engine  {}", voice.engine);
    if let Some(s) = voice.seconds {
        println!("length  {s:.2} s of reference audio");
    }
    println!("text    {:?}", voice.text);
    println!("tensors {}", voice.keys().join(", "));
    let caps = tts_engines::catalogue();
    match caps.iter().find(|c| c.id == voice.engine) {
        Some(c) if c.available => println!("\nusable with engine `{}`", c.id),
        Some(c) => println!("\nengine `{}` is staged: {}", c.id, c.reason.unwrap_or("")),
        None => println!("\nno engine in this build claims `{}`", voice.engine),
    }
    Ok(())
}

fn cmd_speak(args: &Speak) -> Result<()> {
    let id = args
        .engine
        .clone()
        .unwrap_or_else(|| tts_engines::default_id().to_string());
    let text = match (&args.text, &args.text_file) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => {
            std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?
        }
        (None, None) => anyhow::bail!("pass --text or --text-file"),
    };

    let root = args
        .model_root
        .clone()
        .unwrap_or_else(|| PathBuf::from(tts_engines::default_root(&id)));
    let mut overrides = BTreeMap::new();
    for (k, v) in &args.overrides {
        overrides.insert(k.clone(), v.clone());
    }
    let config = EngineConfig {
        model_root: root,
        quant: args.quant.clone(),
        cpu: args.cpu,
        overrides,
    };

    // Voice assets load on the host; the engine pulls them onto its own device. See
    // `Voice::load` for why the CLI must not open a second device handle here.
    let voice = match &args.voice {
        None => None,
        Some(p) => Some(Voice::load(p)?),
    };

    // Fail on an engine/voice mismatch before spending 1-2 s loading weights.
    if let Some(v) = &voice {
        if v.engine != id {
            anyhow::bail!(
                "voice `{}` was built for engine `{}`, but `{id}` was requested — voice \
                 assets are not interchangeable between engines",
                v.name,
                v.engine
            );
        }
    }

    let engine = tts_engines::load(&id, &config)?;
    let caps = engine.capabilities();
    println!("engine {} — {}", caps.id, caps.description);

    let request = SynthesisRequest {
        text,
        voice,
        sampling: Sampling {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
            greedy: args.greedy,
        },
        max_chars: args.max_chars,
        max_new_tokens: args.max_new_tokens,
        gaps: Gaps {
            segment_ms: args.gap_ms,
            paragraph_ms: args.para_gap_ms,
        },
    };
    if let Some(v) = &request.voice {
        println!(
            "voice  {} ({:?})",
            v.name,
            v.seconds.map(|s| format!("{s:.2} s"))
        );
    }

    let out = engine.synthesize(&request)?;
    let seconds = out.audio.seconds();
    tts_core::wav::write_mono(&args.out, &out.audio.samples, out.audio.sample_rate)?;

    let s = &out.stats;
    println!(
        "\n{:.2} s of audio at {} Hz from {} segments ({} frames) in {:.1} s  ->  RTF {:.3}",
        seconds,
        out.audio.sample_rate,
        s.segments,
        s.frames,
        s.total_s,
        s.rtf(seconds)
    );
    // The stage names come from the engine, so this prints a two-stage Audio8 split and
    // a three-stage CosyVoice one without knowing which it is talking to.
    for (name, secs, rtf, share) in s.breakdown(seconds) {
        println!("  {name:<14} {secs:>7.1} s ({share:>4.1}%)  RTF {rtf:.3}");
    }
    println!("wrote {}", args.out.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Engines => cmd_engines(),
        Command::Voice { path } => cmd_voice(path),
        Command::Speak(args) => cmd_speak(args),
    }
}
