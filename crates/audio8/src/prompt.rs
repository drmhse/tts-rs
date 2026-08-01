//! Tokenizer and prompt construction, mirroring `ArkttsProcessor`.
//!
//! The prompt is a `[num_codebooks + 1, T]` grid, not a flat sequence. Row 0 carries
//! text token ids and, when cloning, the reference clip's semantic codes offset into
//! the semantic id range. Rows 1..=10 carry that clip's residual codes, aligned under
//! the semantic ones and zero everywhere else. `Model::embed` reassembles it.
//!
//! Each chat segment is tokenized *separately* and concatenated, which is not the same
//! as tokenizing the joined string: a merge spanning a boundary would change the ids.

use crate::cfg;
use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// Whitespace collapsing lives in `tts-core` because every engine needs it.
pub use tts_core::text::clean_text;

/// `<|speaker:N|>` prefix, added only when the caller has not supplied one.
fn format_reference_text(text: &str) -> String {
    let cleaned = clean_text(text);
    if has_speaker_tag(&cleaned) {
        cleaned
    } else {
        format!("<|speaker:0|>{cleaned}")
    }
}

/// Hand-rolled equivalent of `re.search(r"<\|speaker:\d+\|>", s)`, to avoid pulling in
/// a regex engine for one pattern.
fn has_speaker_tag(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut from = 0usize;
    while from < s.len() {
        let Some(off) = s[from..].find("<|speaker:") else {
            return false;
        };
        let start = from + off;
        let mut j = start + "<|speaker:".len();
        let digits_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > digits_start && s[j..].starts_with("|>") {
            return true;
        }
        from = start + 1;
    }
    false
}

pub struct PromptBuilder {
    tokenizer: Tokenizer,
}

/// A prompt ready for the model: row-major `[num_codebooks + 1, T]`.
#[derive(Clone)]
pub struct Prompt {
    pub rows: Vec<Vec<u32>>,
    pub len: usize,
}

impl PromptBuilder {
    pub fn load(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading {tokenizer_path}: {e}"))?;
        Ok(Self { tokenizer })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing {text:?}: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn encode_parts(&self, parts: &[&str]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for p in parts {
            out.extend(self.encode(p)?);
        }
        Ok(out)
    }

    /// The `has_reference == false` branch: everything in the prefix, empty suffix.
    pub fn segments_plain(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let target = clean_text(text);
        anyhow::ensure!(!target.is_empty(), "text must not be empty");
        let prefix = self.encode_parts(&[
            "<|im_start|>system\n",
            "convert the provided text to speech",
            "<|im_end|>\n",
            "<|im_start|>user\n",
            &target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])?;
        Ok((prefix, Vec::new()))
    }

    /// The cloning branch: the reference transcript goes in the *system* turn, and the
    /// clip's codes are spliced between prefix and suffix.
    pub fn segments_reference(
        &self,
        text: &str,
        reference_text: &str,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let target = clean_text(text);
        anyhow::ensure!(!target.is_empty(), "text must not be empty");
        anyhow::ensure!(
            !clean_text(reference_text).is_empty(),
            "reference_text is required when a reference voice is provided"
        );
        let formatted = format_reference_text(reference_text);
        let prefix = self.encode_parts(&[
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            &formatted,
            "\n\nSpeech:\n",
        ])?;
        let suffix = self.encode_parts(&[
            "<|im_end|>\n",
            "<|im_start|>user\n",
            &target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])?;
        Ok((prefix, suffix))
    }

    /// Assemble the `[num_codebooks + 1, T]` grid.
    ///
    /// `reference_codes` is `[num_codebooks, R]`; its row 0 becomes semantic ids in row
    /// 0 of the prompt, and all ten rows are written under that span.
    pub fn build(&self, text: &str, reference: Option<(&[Vec<u32>], &str)>) -> Result<Prompt> {
        let (prefix, suffix, ref_codes) = match reference {
            None => {
                let (p, s) = self.segments_plain(text)?;
                (p, s, None)
            }
            Some((codes, ref_text)) => {
                let (p, s) = self.segments_reference(text, ref_text)?;
                (p, s, Some(codes))
            }
        };

        let ref_len = ref_codes.map(|c| c[0].len()).unwrap_or(0);
        let total = prefix.len() + ref_len + suffix.len();
        anyhow::ensure!(
            total < cfg::MAX_SEQ_LEN,
            "prompt of {total} tokens does not fit in max_seq_len {}",
            cfg::MAX_SEQ_LEN
        );

        let mut rows = vec![vec![0u32; total]; cfg::NUM_CODEBOOKS + 1];
        let mut row0 = Vec::with_capacity(total);
        row0.extend_from_slice(&prefix);
        if let Some(codes) = ref_codes {
            for &c in &codes[0] {
                row0.push(c + cfg::SEMANTIC_BEGIN_ID);
            }
        }
        row0.extend_from_slice(&suffix);
        rows[0] = row0;
        if let Some(codes) = ref_codes {
            for (i, row) in codes.iter().enumerate() {
                anyhow::ensure!(row.len() == ref_len, "ragged reference codes at row {i}");
                rows[i + 1][prefix.len()..prefix.len() + ref_len].copy_from_slice(row);
            }
        }
        Ok(Prompt { rows, len: total })
    }
}

/// Load `[num_codebooks, T]` reference codes written by
/// `synthesize.py --save-reference-codes`.
pub fn load_reference_codes(path: &str) -> Result<Vec<Vec<u32>>> {
    use candle_core::{DType, Device};
    let map = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {path}"))?;
    let t = map
        .get("reference_codes")
        .context("no `reference_codes` tensor in file")?
        .to_dtype(DType::U32)?;
    let (n, len) = t.dims2()?;
    anyhow::ensure!(
        n == cfg::NUM_CODEBOOKS,
        "expected {} codebooks, got {n}",
        cfg::NUM_CODEBOOKS
    );
    let flat = t.flatten_all()?.to_vec1::<u32>()?;
    Ok((0..n)
        .map(|i| flat[i * len..(i + 1) * len].to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_tag_detection() {
        assert!(has_speaker_tag("<|speaker:0|>hello"));
        assert!(has_speaker_tag("x <|speaker:12|> y"));
        assert!(!has_speaker_tag("<|speaker:|>"));
        assert!(!has_speaker_tag("<|speaker:0|"));
        assert!(!has_speaker_tag("no tag here"));
    }

    #[test]
    fn clean_collapses_whitespace() {
        assert_eq!(clean_text("  a \n b\tc "), "a b c");
    }
}
