//! Engine-neutral TTS types: what every engine must offer a client, and nothing else.
//!
//! The goal this crate exists to serve is that a caller picks an engine at request
//! time. That only works if the things engines genuinely differ on are named
//! explicitly rather than papered over, so [`Capabilities`] is deliberately blunt:
//! sample rate, frame rate, whether cloning is supported, whether streaming is, and
//! which weight formats load. A caller that ignores it will produce a request some
//! engine cannot honour, and it should get a clear error rather than a silent
//! substitution.
//!
//! The design decision that makes a second engine tractable is [`voice::Voice`]. Both
//! models clone from a reference clip, and both need heavy machinery to turn audio
//! into conditioning — Audio8 needs the codec *encoder*, CosyVoice needs a 969 MB
//! speech tokenizer and a 28 MB speaker-embedding model, both ONNX. But that
//! conversion is a function of the clip alone, so it happens once, offline, in Python,
//! and the result ships as a small asset. Neither encoder is in the Rust binary.

pub mod engine;
pub mod rng;
pub mod text;
pub mod voice;
pub mod wav;

pub use engine::{
    Audio, Capabilities, Cloning, Engine, EngineConfig, Gaps, Sampling, Stats, Synthesis,
    SynthesisRequest,
};
pub use voice::Voice;
