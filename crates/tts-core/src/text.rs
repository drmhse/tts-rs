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
        let chars: Vec<char> = para.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            current.push(ch);
            i += 1;
            if matches!(ch, '.' | '!' | '?') {
                // A closing quote or bracket belongs to the sentence the mark ends, so the
                // boundary is looked for past it and the run is taken with it.
                if let Some(end) = closer_end(&chars, i) {
                    while i < end {
                        current.push(chars[i]);
                        i += 1;
                    }
                    sentences.push(std::mem::take(&mut current));
                }
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

/// Words whose period is part of the word, not the end of a sentence.
///
/// The narration prep expands these (`md-to-narration.py`, `ABBREVIATIONS`), so a book that
/// goes through it never reaches here with one. Text handed straight to an engine does: an
/// API caller's paragraph, a fixture, a probe. "Fig. 2" split there is a full stop's fall and
/// pause inside a noun phrase, which is the same defect as the one `12.4` used to have and
/// costs nothing to rule out.
const ABBREVIATIONS: &[&str] = &[
    "fig", "figs", "eq", "eqs", "tab", "sec", "secs", "ch", "chs", "no", "nos", "vol", "vols",
    "pp", "p", "al", "cf", "eg", "ie", "dr", "mr", "mrs", "ms", "prof", "st", "approx", "ca",
    "etc", "vs", "resp", "inc", "ltd", "co", "est", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
];

/// Is the word ending at `i` (exclusive) one of them?
fn is_abbreviation(chars: &[char], i: usize) -> bool {
    let start = chars[..i]
        .iter()
        .rposition(|c| !c.is_alphanumeric())
        .map_or(0, |p| p + 1);
    if start == i || i - start > 5 {
        return false;
    }
    let word: String = chars[start..i].iter().collect::<String>().to_lowercase();
    ABBREVIATIONS.contains(&word.as_str())
}

/// Where a sentence ending at `i` stops, if it ends there at all.
///
/// A bare `.` is not a boundary: "12.4" split into "12." and "4", the engine spoke them as
/// two segments, and the listener heard "twelve" — pause — "four". Version strings, file
/// names and initialisms fail the same way. A real boundary is followed by whitespace or
/// the end of the paragraph, optionally past a closing quote or bracket — which the
/// sentence keeps, so `Some(j)` is the index one past the last character of the sentence.
fn closer_end(chars: &[char], i: usize) -> Option<usize> {
    // "Fig. 2" and "et al. (2021)" — the period belongs to the word in front of it.
    if i > 0 && chars[i - 1] == '.' && is_abbreviation(chars, i - 1) {
        return None;
    }
    let mut j = i;
    while matches!(
        chars.get(j),
        Some('"' | '\'' | ')' | ']' | '\u{201d}' | '\u{2019}')
    ) {
        j += 1;
    }
    match chars.get(j) {
        None => Some(j),
        // A lower-case word after the gap continues the sentence whatever the mark was. This
        // is what catches the abbreviations the list above does not name — "e.g. the", "no.
        // seven" — and English does not start a sentence in lower case, so nothing real is
        // merged by it. An over-long merge is bounded by `split_long` in any case; a false
        // boundary is not recoverable once it has been spoken.
        Some(c) if c.is_whitespace() => match chars[j..].iter().find(|c| !c.is_whitespace()) {
            Some(next) if next.is_lowercase() => None,
            _ => Some(j),
        },
        Some(_) => None,
    }
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
    fn an_abbreviation_is_not_a_sentence_boundary() {
        let segs = segment("See Fig. 2 and Tab. 1. It holds.", 200);
        assert_eq!(segs[0], vec!["See Fig. 2 and Tab. 1. It holds."]);
        // The generic rule: whatever the mark, a lower-case continuation is one sentence.
        let segs = segment("Shown in Sec. 4 e.g. the second run. Done.", 200);
        assert_eq!(segs[0], vec!["Shown in Sec. 4 e.g. the second run. Done."]);
    }

    /// A capital after the gap is still a boundary — "10 MWh of demand at A. The model..."
    /// is two sentences, and welding them was the cost of a rule that guessed the other way.
    #[test]
    fn a_capital_after_the_gap_still_breaks() {
        let segs = segment("Demand at A. The model solves it.", 25);
        assert_eq!(segs[0], vec!["Demand at A.", "The model solves it."]);
    }

    /// A decimal is one token. Splitting on its dot put "12." and "4" in separate segments
    /// and the gap between them was audible.
    #[test]
    fn decimals_are_not_sentence_boundaries() {
        let segs = segment("Latency rose to 12.4 seconds. Then it fell.", 200);
        assert_eq!(segs[0], vec!["Latency rose to 12.4 seconds. Then it fell."]);
        let segs = segment("Version 1.2.3 shipped.", 200);
        assert_eq!(segs[0], vec!["Version 1.2.3 shipped."]);
    }

    #[test]
    fn quoted_sentence_still_breaks() {
        let segs = segment("He said \"go.\" She left.", 20);
        assert_eq!(segs[0], vec!["He said \"go.\"", "She left."]);
    }

    #[test]
    fn short_text_is_untouched() {
        assert_eq!(split_long("Short enough.", 220), vec!["Short enough."]);
    }
}
