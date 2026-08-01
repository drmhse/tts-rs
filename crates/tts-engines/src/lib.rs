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
pub fn catalogue() -> Vec<Capabilities> {
    vec![a8::engine::capabilities(), cosy::capabilities()]
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
        .unwrap_or(a8::engine::ID)
}

pub fn load(id: &str, config: &EngineConfig) -> Result<Box<dyn Engine>> {
    match id {
        a8::engine::ID => Ok(Box::new(a8::engine::Audio8Engine::load(config)?)),
        cosy::ID => Ok(Box::new(cosy::CosyVoiceEngine::load(config)?)),
        other => anyhow::bail!("unknown engine `{other}`; available: {}", ids().join(", ")),
    }
}

/// Default model root per engine, relative to the repo. Conventions rather than
/// configuration, overridable through [`EngineConfig::overrides`].
pub fn default_root(id: &str) -> &'static str {
    match id {
        a8::engine::ID => "oracle/weights",
        cosy::ID => "oracle-cosy/weights",
        _ => ".",
    }
}
