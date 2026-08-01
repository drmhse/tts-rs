//! Benchmarks for the Audio8 engine.
//!
//! The measurement protocol itself lives in `tts-bench`, so the CosyVoice benchmarks can
//! use the same harness — a thermally-honest number is not an Audio8-specific concern.
//! Re-exported here under the original name so the existing probes keep compiling.
pub use tts_bench as bench;
