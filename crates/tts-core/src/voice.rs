//! Voice assets: conditioning precomputed offline, consumed cheaply at runtime.
//!
//! All three engines clone from a reference clip, and in every case turning audio into
//! conditioning needs machinery the runtime should not carry:
//!
//! | engine | what the clip must become | what that would cost in-process |
//! |---|---|---|
//! | `audio8` | 10 x N RVQ codes | the codec **encoder**, 126 tensors dropped by `convert_codec.py` |
//! | `cosyvoice` | speaker embedding, speech tokens, prompt mel, prompt text tokens | `campplus.onnx` (28 MB) + `speech_tokenizer_v3.onnx` (969 MB), and an ONNX runtime |
//! | `qwen3tts` | x-vector, `[T, 16]` RVQ codes, sliced transcript tokens | an ECAPA-TDNN encoder and an RVQ encoder, both in the talker checkpoint |
//!
//! None of that depends on the text being spoken, so it runs once in Python and ships
//! as an asset. The Rust binary contains neither encoder.
//!
//! Layout is a directory:
//!
//! ```text
//! voices/cosy-default/
//!   voice.json         { engine, name, text, notes }
//!   voice.safetensors  engine-specific tensors
//! ```
//!
//! The `engine` field is checked on load. Assets are not interchangeable — the
//! tensors mean entirely different things — and a mismatch is a hard error rather
//! than a fallback, because falling back would silently produce the wrong speaker.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VoiceManifest {
    /// Engine id this asset was built for.
    pub engine: String,
    pub name: String,
    /// Transcript of the reference clip. Both engines interleave it with the clip's
    /// tokens, so it is part of the asset rather than a separate argument.
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub notes: Option<String>,
    /// Seconds of reference audio, for reporting.
    #[serde(default)]
    pub seconds: Option<f64>,
}

/// `Clone` is cheap and intentional: candle tensors are `Arc`-backed, so a clone bumps
/// refcounts rather than copying the asset. A long-lived server hands the same voice to
/// every request, and that should not mean re-reading it from disk or wrapping it in a
/// second layer of `Arc` at each call site.
#[derive(Clone)]
pub struct Voice {
    pub engine: String,
    pub name: String,
    pub text: String,
    pub seconds: Option<f64>,
    tensors: HashMap<String, Tensor>,
}

impl Voice {
    /// Load a voice asset. Tensors land on the CPU.
    ///
    /// Deliberately not parameterised by device. A caller loads a voice *before* it loads
    /// an engine, so a device passed here would be a second handle to the same GPU — and
    /// candle compares device identity rather than hardware, so mixing tensors from the
    /// two fails with `device mismatch in matmul` naming the same `gpu_id` twice, and
    /// `to_device` between them is not implemented at all. Keeping assets on the host and
    /// having engines pull them across with [`Voice::get_on`] makes that unreachable.
    /// The assets are a few hundred kilobytes, so there is nothing to gain by doing
    /// otherwise.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_on(dir.as_ref(), &Device::Cpu)
    }

    fn load_on(dir: &Path, device: &Device) -> Result<Self> {
        let manifest_path = dir.join("voice.json");
        let raw = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: VoiceManifest = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", manifest_path.display()))?;
        let tensor_path = dir.join("voice.safetensors");
        let tensors = candle_core::safetensors::load(&tensor_path, device)
            .with_context(|| format!("loading {}", tensor_path.display()))?;
        Ok(Self {
            engine: manifest.engine,
            name: manifest.name,
            text: manifest.text,
            seconds: manifest.seconds,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).with_context(|| {
            format!(
                "voice `{}` (engine {}) has no tensor `{name}`; it has: {}",
                self.name,
                self.engine,
                self.keys().join(", ")
            )
        })
    }

    /// Fetch a tensor placed on `device` — what an engine should call.
    pub fn get_on(&self, name: &str, device: &Device) -> Result<Tensor> {
        Ok(self.get(name)?.to_device(device)?)
    }

    /// Fetch a `[rows, len]` integer tensor as row-major `u32`, which is how both
    /// engines want their token grids.
    ///
    /// The cast happens on the CPU: Metal has no `I32 -> U32` conversion kernel, and the
    /// result is a host `Vec` regardless, so there is nothing to gain from doing it on
    /// device. Exporters are free to write whatever integer width is natural.
    pub fn get_rows_u32(&self, name: &str) -> Result<Vec<Vec<u32>>> {
        let t = self
            .get(name)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?;
        let (rows, len) = t
            .dims2()
            .with_context(|| format!("tensor `{name}` in voice `{}` is not 2-D", self.name))?;
        let flat = t.flatten_all()?.to_vec1::<u32>()?;
        Ok((0..rows)
            .map(|i| flat[i * len..(i + 1) * len].to_vec())
            .collect())
    }

    pub fn keys(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    }
}
