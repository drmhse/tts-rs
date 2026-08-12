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

/// Break a sentence that exceeds the budget on its own.
///
/// Sentence boundaries alone do not bound segment length, and for a long time this was
/// unenforced: `segment` only checked the budget when *merging* sentences, so a single
/// sentence longer than `max_chars` became one over-long segment. The book narrated with
/// that behaviour contained segments of 566 characters against a 220 budget — roughly 38
/// seconds of speech, about 950 speech tokens, against a `max_new_tokens` of 512. The AR
/// loop stopped at the cap and the rest of the sentence was never spoken: 24 segments
/// across one book, about two minutes of missing audio, and nothing reported it.
///
/// Clause punctuation is preferred over word count because the inserted segment gap then
/// lands where the voice would already pause. Word splitting is the last resort, so no
/// segment can be unbounded whatever the punctuation looks like.
fn split_long(sentence: &str, max_chars: usize) -> Vec<String> {
    if sentence.len() <= max_chars {
        return vec![sentence.to_string()];
    }

    // Atoms end *after* clause punctuation, so the mark stays with the clause it closes.
    let mut atoms: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in sentence.chars() {
        current.push(ch);
        if matches!(ch, ',' | ';' | ':' | '\u{2014}' | '\u{2013}') {
            atoms.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        atoms.push(current);
    }

    // A clause can still be too long by itself; fall back to words.
    let mut pieces: Vec<String> = Vec::new();
    for atom in atoms {
        if atom.len() <= max_chars {
            pieces.push(atom);
            continue;
        }
        let mut buf = String::new();
        for word in atom.split_whitespace() {
            if !buf.is_empty() && buf.len() + 1 + word.len() > max_chars {
                pieces.push(std::mem::take(&mut buf));
            }
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(word);
        }
        if !buf.is_empty() {
            pieces.push(buf);
        }
    }

    // Repack, so splitting on an early comma does not leave a trail of tiny segments.
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for piece in pieces {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        if !buf.is_empty() && buf.len() + 1 + piece.len() > max_chars {
            out.push(std::mem::take(&mut buf));
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(piece);
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Split text into paragraphs, then into segments within a character budget.
///
/// Segments are whole sentences where sentences fit, and clause-sized pieces where one does
/// not — see [`split_long`]. Every returned segment is at most `max_chars` bytes, which is
/// what keeps prompts below `max_seq_len` and, more importantly, keeps a segment's speech
/// inside the AR loop's `max_new_tokens`.
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
        // A sentence over budget is broken up before merging, so `max_chars` bounds every
        // segment rather than only the merged ones.
        let sentences = sentences
            .iter()
            .flat_map(|s| split_long(s.trim(), max_chars));
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

    /// The invariant that was missing. A sentence longer than the budget used to pass
    /// through whole, and the AR loop silently truncated its speech at `max_new_tokens`.
    #[test]
    fn no_segment_exceeds_the_budget() {
        let long = "Change that crosses a product boundary needs a group whose legitimacy \
            comes from different places, because the decision needs several kinds of \
            knowledge that no single team holds: product and engineering owners who can \
            change scope, support and field engineers who see failure first, respected \
            practitioners from outside, and where the change touches data retention, \
            contracts or compliance, security, legal and operations.";
        assert!(
            long.len() > 220,
            "fixture must exceed the budget to be meaningful"
        );
        for para in segment(long, 220) {
            for seg in para {
                assert!(seg.len() <= 220, "segment of {} bytes: {seg}", seg.len());
            }
        }
    }

    #[test]
    fn long_sentence_prefers_clause_boundaries() {
        // Two clauses of ~30 bytes each with a 40-byte budget: the split must land on the
        // comma rather than mid-clause.
        let pieces = split_long(
            "alpha beta gamma delta epsilon, zeta eta theta iota kappa",
            40,
        );
        assert!(pieces.len() >= 2);
        assert!(
            pieces[0].ends_with(','),
            "expected a clause break, got {:?}",
            pieces[0]
        );
    }

    #[test]
    fn unpunctuated_run_still_bounded() {
        let words = "alpha ".repeat(200);
        for piece in split_long(words.trim(), 60) {
            assert!(
                piece.len() <= 60,
                "unbounded piece of {} bytes",
                piece.len()
            );
        }
    }

    #[test]
    fn short_text_is_untouched() {
        assert_eq!(split_long("Short enough.", 220), vec!["Short enough."]);
    }
}
