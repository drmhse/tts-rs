//! Text handling shared by every engine.
//!
//! Segmentation lives here rather than in an engine because it is a property of the
//! request, not the model: every engine has a context limit, every engine degrades if
//! asked to hold prosody over too long a span, and batching has nothing to batch
//! without it.

/// `" ".join(str(text).strip().split())` — collapses every run of whitespace.
pub fn clean_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split text into paragraphs, then into segments of whole sentences within a character
/// budget. Segmentation keeps prompts far below `max_seq_len` and bounds how much audio
/// a single AR run has to stay coherent over.
pub fn segment(text: &str, max_chars: usize) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for para in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let para = collapse_dots(&clean_text(para));
        let mut sentences: Vec<String> = Vec::new();
        let mut current = String::new();
        for ch in para.chars() {
            current.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                sentences.push(std::mem::take(&mut current));
            }
        }
        if !current.trim().is_empty() {
            sentences.push(current);
        }
        let mut segments: Vec<String> = Vec::new();
        let mut buf = String::new();
        for s in sentences {
            let s = s.trim();
            if s.is_empty() {
                continue;
            }
            if !buf.is_empty() && buf.len() + 1 + s.len() > max_chars {
                segments.push(std::mem::take(&mut buf));
            }
            if buf.is_empty() {
                buf.push_str(s);
            } else {
                buf.push(' ');
                buf.push_str(s);
            }
        }
        if !buf.is_empty() {
            segments.push(buf);
        }
        if !segments.is_empty() {
            out.push(segments);
        }
    }
    out
}

/// Collapse `..` and longer runs to a single period. The source text for the first
/// example had one, and it becomes an audible stumble.
fn collapse_dots(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut dots = 0usize;
    for ch in s.chars() {
        if ch == '.' {
            dots += 1;
            if dots == 1 {
                out.push('.');
            }
        } else {
            dots = 0;
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_and_collapse() {
        assert_eq!(clean_text("  a \n b\tc "), "a b c");
        assert_eq!(collapse_dots("found it.. next"), "found it. next");
    }

    #[test]
    fn segmentation_splits_on_sentences() {
        let segs = segment("One. Two. Three.", 10);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0], vec!["One. Two.", "Three."]);
    }

    #[test]
    fn paragraphs_stay_separate() {
        let segs = segment("First para.\nSecond para.", 200);
        assert_eq!(segs.len(), 2);
    }
}
