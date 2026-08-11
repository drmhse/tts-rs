//! The registry: the one place that knows which engines exist.
//!
//! Kept separate from `tts-core` so the trait has no dependency on its
//! implementations, and separate from the CLI so a service can reuse selection without
//! inheriting an argument parser. A caller does:
//!
//! ```no_run
//! use tts_core::{EngineConfig, SynthesisRequest, Voice};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Enumerate first: unavailable engines are listed too, with a reason.
//! for caps in tts_engines::catalogue() {
//!     println!("{:<10} {}", caps.id, if caps.available { "ready" } else { "unavailable" });
//! }
//!
//! let id = "cosyvoice";
//! let config = EngineConfig::new(tts_engines::default_root(id));
//! let engine = tts_engines::load(id, &config)?;
//!
//! // Voice assets load on the host; the engine pulls them onto its own device.
//! let voice = Voice::load("voices/cosy-default-cosyvoice")?;
//! let request = SynthesisRequest::new("Hello from Rust.").with_voice(voice);
//!
//! // `validate` rejects a mismatched asset up front rather than at the first tensor.
//! engine.validate(&request)?;
//! let out = engine.synthesize(&request)?;
//!
//! tts_core::wav::write("hello.wav", &out.audio)?;
//! let secs = out.audio.seconds();
//! println!("{secs:.2} s at RTF {:.3}", out.stats.rtf(secs));
//! for (stage, s, rtf, share) in out.stats.breakdown(secs) {
//!     println!("  {stage:<8} {s:.1} s ({share:.0}%) RTF {rtf:.3}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Unavailable engines are listed rather than hidden, so a client can discover an
//! identifier before it works and get an error that says why instead of silently
//! getting a different voice from a different model.

use anyhow::Result;
use tts_core::{Capabilities, Engine, EngineConfig};

/// Every engine known to this build, in preference order.
///
/// Order is preference, and [`default_id`] takes the first *available* entry — so an
/// unfinished engine may sit anywhere without becoming the default. `qwen3tts` is last
/// for a reason that will outlast its port: it supports ten languages and the list is
/// closed, so it cannot be the fallback for a caller that did not choose it.
pub fn catalogue() -> Vec<Capabilities> {
    vec![
        audio8::engine::capabilities(),
        cosyvoice::capabilities(),
        qwen3tts::capabilities(),
    ]
}

/// Ids a caller may pass to [`load`].
pub fn ids() -> Vec<&'static str> {
    catalogue().into_iter().map(|c| c.id).collect()
}

/// The first engine that can actually synthesize — what a client gets when it does not
/// care which.
pub fn default_id() -> &'static str {
    catalogue()
        .into_iter()
        .find(|c| c.available)
        .map(|c| c.id)
        .unwrap_or(audio8::engine::ID)
}

pub fn load(id: &str, config: &EngineConfig) -> Result<Box<dyn Engine>> {
    match id {
        audio8::engine::ID => Ok(Box::new(audio8::engine::Audio8Engine::load(config)?)),
        cosyvoice::ID => Ok(Box::new(cosyvoice::CosyVoiceEngine::load(config)?)),
        qwen3tts::ID => Ok(Box::new(qwen3tts::Qwen3TtsEngine::load(config)?)),
        other => anyhow::bail!("unknown engine `{other}`; available: {}", ids().join(", ")),
    }
}

/// Default model root per engine, relative to the repo. Conventions rather than
/// configuration, overridable through [`EngineConfig::overrides`].
pub fn default_root(id: &str) -> &'static str {
    match id {
        audio8::engine::ID => "references/audio8/weights",
        cosyvoice::ID => "references/cosyvoice/weights",
        qwen3tts::ID => "references/qwen3tts/weights",
        _ => ".",
    }
}
