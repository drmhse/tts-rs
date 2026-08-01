//! Minimal 16-bit PCM WAV writer.
//!
//! Hand-rolled rather than pulled from a crate: the header is 44 bytes and this keeps
//! the dependency list short enough to audit.

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

/// Write mono 16-bit PCM. Samples outside `[-1, 1]` are clamped, not wrapped — the
/// decoder ends in `tanh` so they should already be in range, but a wrap would turn a
/// tiny overshoot into a loud click.
pub fn write_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let data_bytes = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
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
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&out)?;
    Ok(())
}

/// `sample_rate * seconds` worth of silence, for gaps between segments.
pub fn silence(sample_rate: usize, millis: usize) -> Vec<f32> {
    vec![0f32; sample_rate * millis / 1000]
}
