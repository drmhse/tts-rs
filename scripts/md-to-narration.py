#!/usr/bin/env python3
"""Markdown -> narration text.

Feeding raw markdown to a TTS engine reads the syntax aloud: "plus plus plus",
"title equals", "hash hash". This strips the notation and keeps the prose, making the
choices a narrator would:

  * **Front matter is dropped.** TOML (`+++`) and YAML (`---`) both.
  * **Headings become sentences.** A heading is a spoken section break, so it keeps its
    text and gains a full stop. That matters because both engines insert a longer gap at
    a paragraph boundary than at a sentence one (320 ms against 90 ms), so a heading on
    its own paragraph is what produces an audible section pause.
  * **Blockquotes lose the marker, keep the words.** In this book they are example copy
    meant to be read.
  * **Emphasis markers are removed, not spoken.** `*word*` is "word".
  * **List items become sentences** so they do not run together into one long clause.
  * **Em dashes become commas.** An em dash is a prosodic pause in print; a comma is the
    same pause in speech, and the engines' tokenizers handle it more predictably.
  * **Code blocks are dropped by default** (`--keep-code` to narrate them). Reading shell
    syntax aloud is noise, not content.
  * **HTML comments are dropped.** Not cosmetic: this book carries
    `<!-- ILLUSTRATION-PLACEHOLDER ... -->` blocks holding several hundred words of art
    direction, and an earlier version of this script narrated them ("Editorial line art in
    the book's shared visual language, rounded rectangles for artifacts...").
  * **Hugo shortcodes are dropped, but a `caption=` is kept.** `{{< chapter-figure >}}`
    marks an image the listener cannot see, and its caption is usually the one sentence
    stating what the figure was for. Keeping it is the better audiobook default; pass
    `--no-captions` to omit.

# Page-word mapping
#
# The site's player highlights words in the *rendered page*, and it discovers those words by
# walking the DOM (`collectPageWords` in `audio-player.js`). So a manifest entry's
# `pageWordIndex` has to index into rendered-prose word order, not into the narration text.
#
# The two sequences are nearly identical — the transformations above change punctuation and
# structure, not words — so `--emit-map` writes the page words in rendered order alongside a
# `spokenToPageWord` array produced by aligning the two token streams. Where they diverge
# (a figure's `alt` text is never rendered; a heading gains a full stop) the alignment
# absorbs it rather than drifting.
#
# Emitting explicit indices matters: the player prefers them and only falls back to
# `legacyGreedyPageWordMatches`, a fuzzy DOM match, when they are absent.

Usage:
    scripts/md-to-narration.py chapter-1.md -o chapter-1.txt
    scripts/md-to-narration.py chapter-1.md --emit-map chapter-1.map.json
    scripts/md-to-narration.py chapter-1.md --stats
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def strip_front_matter(text: str) -> str:
    for fence in ("+++", "---"):
        if text.lstrip().startswith(fence):
            body = text.lstrip()
            end = body.find(f"\n{fence}", len(fence))
            if end != -1:
                return body[end + len(fence) + 1 :]
    return text


def drop_code_blocks(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def drop_html_comments(text: str) -> str:
    """`<!-- ... -->`, including the multi-line art-direction blocks."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def resolve_shortcodes(text: str, keep_captions: bool = True) -> str:
    """Drop `{{< ... >}}` / `{{% ... %}}`, optionally promoting `caption=` to prose."""

    def repl(m: re.Match) -> str:
        if not keep_captions:
            return ""
        caption = re.search(r'caption\s*=\s*"([^"]*)"', m.group(0))
        if not caption or not caption.group(1).strip():
            return ""
        text = caption.group(1).strip()
        return "\n\n" + (text if ends_sentence(text) else text + ".") + "\n\n"

    return re.sub(r"\{\{[<%].*?[>%]\}\}", repl, text, flags=re.S)


def clean_inline(line: str) -> str:
    line = re.sub(r"`([^`]*)`", r"\1", line)              # inline code
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)      # images
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # links keep their text
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)        # bold
    line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)  # italic
    line = re.sub(r"_([^_]+)_", r"\1", line)
    line = line.replace("~~", "")
    # An em dash is a pause in print; a comma is the same pause in speech.
    line = re.sub(r"\s*[—–]\s*", ", ", line)
    line = line.replace("…", "...")
    # Smart quotes confuse some tokenizers and add nothing when spoken.
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]:
        line = line.replace(a, b)
    return re.sub(r"[ \t]+", " ", line).strip()


def ends_sentence(s: str) -> bool:
    return s.endswith((".", "!", "?", ":", ";", '"'))


TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE = re.compile(r"^\s*\|[\s|:\-]+\|\s*$")


def render_tables(text: str) -> str:
    """Rewrite markdown tables as speakable sentences.

    Without this, table lines fall through to the paragraph buffer and a table becomes one
    long run-on with no sentence boundaries — the separator row included. That is not merely
    ugly to listen to: on chapter 8 it made the LLM degenerate into a repetition loop and
    emit babble for the whole table, which two independent recognisers transcribed as
    "tampoligation, tampolition, sambolition, and tambolion". The audio, not the alignment,
    was wrong, and it shipped.

    Each row becomes its own short sentences, so the engine's sentence segmentation keeps
    every prompt small:

        | Partner | DevRel provides    | DevRel requires |
        | Product | field synthesis... | roadmap context... |

        -> "Product. DevRel provides: field synthesis... DevRel requires: roadmap context..."

    A single-column or header-only table degrades to its cells as sentences, which is still
    speakable.
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not TABLE_ROW.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < len(lines) and TABLE_ROW.match(lines[i]):
            block.append(lines[i])
            i += 1

        def cells(row: str) -> list[str]:
            return [c.strip() for c in row.strip().strip("|").split("|")]

        rows = [cells(r) for r in block if not TABLE_RULE.match(r)]
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        if not body:  # header-only: just speak the cells
            body, header = [header], []
        out.append("")
        for row in body:
            parts: list[str] = []
            label = row[0] if row else ""
            if label:
                parts.append(label if ends_sentence(label) else label + ".")
            for index, value in enumerate(row[1:], start=1):
                if not value:
                    continue
                name = header[index] if index < len(header) else ""
                clause = f"{name}: {value}" if name else value
                parts.append(clause if ends_sentence(clause) else clause + ".")
            if parts:
                out.append(" ".join(parts))
                out.append("")
    return "\n".join(out)


def convert(text: str, keep_code: bool = False, keep_captions: bool = True) -> str:
    text = strip_front_matter(text)
    text = drop_html_comments(text)
    text = resolve_shortcodes(text, keep_captions=keep_captions)
    # Before paragraph assembly, so table rows never reach the run-on buffer.
    text = render_tables(text)
    if not keep_code:
        text = drop_code_blocks(text)

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue

        if line.lstrip().startswith("#"):
            # A section break: its own paragraph, so the engine's paragraph gap applies.
            flush()
            heading = clean_inline(line.lstrip("#").strip())
            if heading:
                paragraphs.append(heading if ends_sentence(heading) else heading + ".")
            continue

        if re.match(r"^\s*>", line):
            line = re.sub(r"^\s*>\s?", "", line)

        bullet = re.match(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush()
            item = clean_inline(bullet.group(1))
            if item:
                paragraphs.append(item if ends_sentence(item) else item + ".")
            continue

        if set(line.strip()) <= {"-", "=", "*", "_"} and len(line.strip()) >= 3:
            continue  # horizontal rule

        cleaned = clean_inline(line)
        if cleaned:
            buffer.append(cleaned)

    flush()
    return "\n\n".join(p for p in paragraphs if p.strip()) + "\n"


# Matches `wordPattern` in audio-player.js so both sides tokenise identically.
WORD = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")


def normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("\u2019", "'"))


def page_text(md: str) -> str:
    """The prose a browser would show, in DOM order.

    Same removals as narration except the ones that only affect *speech*: no comma
    substitution for em dashes, no full stop appended to headings. A figure shortcode
    contributes only its caption, which is what actually renders as text.
    """
    md = strip_front_matter(md)
    md = drop_html_comments(md)
    md = resolve_shortcodes(md, keep_captions=True)
    md = drop_code_blocks(md)
    out = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line)
        if set(line) <= {"-", "=", "*", "_"} and len(line) >= 3:
            continue
        # clean_inline without the em-dash rewrite: it changes punctuation, not words.
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"_([^_]+)_", r"\1", line)
        out.append(line)
    return " ".join(out)


def align_tokens(spoken: list[str], page: list[str]) -> list[int]:
    """`spoken[i]` -> index into `page`, or -1.

    A two-pointer walk with bounded lookahead. The sequences agree almost everywhere, so
    this only has to recover from short local differences; a full edit-distance table over
    ~3000 tokens would be needless. Lookahead is bounded so a genuine mismatch costs a few
    unmapped words instead of dragging the whole alignment out of step.
    """
    out, j = [], 0
    for token in spoken:
        if j < len(page) and page[j] == token:
            out.append(j)
            j += 1
            continue
        hit = -1
        for ahead in range(1, 25):
            if j + ahead < len(page) and page[j + ahead] == token:
                hit = j + ahead
                break
        if hit >= 0:
            out.append(hit)
            j = hit + 1
        else:
            out.append(-1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="defaults to stdout")
    ap.add_argument("--keep-code", action="store_true")
    ap.add_argument(
        "--no-captions",
        action="store_true",
        help="omit figure captions instead of narrating them",
    )
    ap.add_argument(
        "--emit-map",
        type=Path,
        help="write {pageWords, spokenToPageWord} JSON for the site's highlighting",
    )
    ap.add_argument("--stats", action="store_true", help="report length and est. duration")
    args = ap.parse_args()

    out = convert(
        args.input.read_text(),
        keep_code=args.keep_code,
        keep_captions=not args.no_captions,
    )

    if args.stats:
        words = len(out.split())
        # ~155 wpm is what these engines produce at speed 1.0 on this material.
        print(
            f"{args.input.name}: {words} words, {len(out)} chars, "
            f"{out.count(chr(10) * 2) + 1} paragraphs, ~{words / 155:.1f} min of audio",
            file=sys.stderr,
        )

    if args.emit_map:
        import json

        source = args.input.read_text()
        page_tokens = WORD.findall(page_text(source))
        spoken_tokens = WORD.findall(out)
        page_norm = [normalize_word(w) for w in page_tokens]
        spoken_norm = [normalize_word(w) for w in spoken_tokens]
        mapping = align_tokens(spoken_norm, page_norm)
        mapped = sum(1 for m in mapping if m >= 0)
        args.emit_map.write_text(
            json.dumps(
                {
                    "pageWords": [
                        {"text": t, "normalized": n}
                        for t, n in zip(page_tokens, page_norm)
                    ],
                    "spokenWords": [
                        {"text": t, "normalized": n}
                        for t, n in zip(spoken_tokens, spoken_norm)
                    ],
                    "spokenToPageWord": mapping,
                    "pageMapping": {
                        "algorithm": "two-pointer-lookahead-24",
                        "spokenWordCount": len(spoken_tokens),
                        "pageWordCount": len(page_tokens),
                        "mappedSpokenWordCount": mapped,
                        "unmappedSpokenWordCount": len(spoken_tokens) - mapped,
                        "mappedShare": round(mapped / max(1, len(spoken_tokens)), 6),
                    },
                },
                separators=(",", ":"),
            )
        )
        if args.stats:
            print(
                f"  page words {len(page_tokens)}, spoken {len(spoken_tokens)}, "
                f"mapped {100 * mapped / max(1, len(spoken_tokens)):.2f}%",
                file=sys.stderr,
            )

    if args.out:
        args.out.write_text(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
