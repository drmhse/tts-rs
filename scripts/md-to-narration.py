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

Usage:
    scripts/md-to-narration.py chapter-1.md -o chapter-1.txt
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


def convert(text: str, keep_code: bool = False, keep_captions: bool = True) -> str:
    text = strip_front_matter(text)
    text = drop_html_comments(text)
    text = resolve_shortcodes(text, keep_captions=keep_captions)
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

    if args.out:
        args.out.write_text(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
