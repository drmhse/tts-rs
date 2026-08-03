//! Minimal 16-bit PCM WAV encoding.
//!
//! Hand-rolled rather than pulled from a crate: the header is 44 bytes and this keeps
//! the dependency list short enough to audit.
//!
//! **There is exactly one float-to-`i16` conversion here, and everything goes through it.**
//! That is not tidiness. `tts-serve` originally carried its own copy for building an HTTP
//! body, and it truncated (`as i16`) where this rounds — so the server and the CLI produced
//! audio differing by 1 LSB on 48% of samples from identical tokens. Inaudible at 71 dB
//! below signal, and still wrong: two code paths that should agree, silently disagreeing.
//! Any new consumer wants [`to_bytes`] or [`pcm_s16le`], not a fourth encoder.

use crate::engine::Audio;
use anyhow::Result;
use std::io::Write;
use std::path::Path;

/// Write an [`Audio`] — what every engine returns — to a path.
///
/// The sample rate travels with the samples, so this cannot be called with the two out
/// of step, which `write_mono` allows and which produces a file that plays at the wrong
/// speed rather than failing.
pub fn write(path: impl AsRef<Path>, audio: &Audio) -> Result<()> {
    write_mono(path.as_ref(), &audio.samples, audio.sample_rate)
}

/// Mono 16-bit PCM samples, `[-1, 1]` clamped and **rounded** to nearest.
///
/// Clamped rather than wrapped: the decoders end in `tanh` so samples should already be in
/// range, but a wrap would turn a tiny overshoot into a loud click. Rounded rather than
/// truncated because truncation biases every sample toward zero by up to 1 LSB.
pub fn pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A complete WAV file in memory — for an HTTP body, or anything else that is not a path.
pub fn to_bytes(audio: &Audio) -> Vec<u8> {
    encode(&audio.samples, audio.sample_rate)
}

fn encode(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let pcm = pcm_s16le(samples);
    let data_bytes = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    out.extend_from_slice(&pcm);
    out
}

/// Write mono 16-bit PCM. Samples outside `[-1, 1]` are clamped, not wrapped — the
/// decoder ends in `tanh` so they should already be in range, but a wrap would turn a
/// tiny overshoot into a loud click.
pub fn write_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&encode(samples, sample_rate))?;
    Ok(())
}

/// `sample_rate * seconds` worth of silence, for gaps between segments.
pub fn silence(sample_rate: usize, millis: usize) -> Vec<f32> {
    vec![0f32; sample_rate * millis / 1000]
}
