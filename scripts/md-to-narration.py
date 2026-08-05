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
  * **Inline code spans are verbalised, not dropped.** A span sits inside a sentence, so
    deleting it leaves a hole the listener hears as a mistake ("a conditional update can
    protect a simple state transition: proves the precondition"). `speak_code` turns the
    span into the words a narrator would actually say: identifiers keep their words,
    operators become English, and CamelCase event names are split. See its docstring.
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
import difflib
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


# SQL keywords, which this book writes in upper case. Upper case is a pronunciation
# instruction to this voice and it obeys it: `WHERE id = ?` was spoken as "WHAE IED", and
# `UPDATE ... SET` as "UPDAT ... SET" — the letters, not the words. The same statement in
# lower case reads correctly. Confirmed against a rendered A/B of the identical clause.
SQL_KEYWORDS = {
    "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "SET", "ORDER", "BY", "GROUP",
    "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AND",
    "OR", "NOT", "NULL", "VALUES", "INTO", "AS", "DISTINCT", "COUNT", "SUM", "EXISTS",
    "BETWEEN", "LIKE", "IN", "IS", "CREATE", "TABLE", "INDEX", "UNIQUE", "PRIMARY", "KEY",
}


def _split_camel(token: str) -> str:
    """`OrderCancellationAccepted` -> "Order cancellation accepted".

    The split is what makes the voice read words instead of one fused token. Lowercasing the
    tail is what keeps it from mangling them: "Order Cancellation Accepted" came back as
    "order, scancelation accepted", while the same phrase in sentence case was clean. Every
    interior capital in an identifier is a word boundary, not an emphasis.
    """
    parts = re.split(r"(?<=[a-z0-9])(?=[A-Z])", token)
    if len(parts) == 1:
        return token
    return parts[0] + " " + " ".join(p.lower() for p in parts[1:])


def speak_code(inner: str) -> str:
    """The contents of one inline code span, as a narrator would read it aloud.

    Dropping the span is not an option: it is a sentence constituent, and removing it leaves
    a sentence with a hole in it. Reading it literally is not an option either — the raw
    characters give "seats set remaining equals sign remaining minus sign one", or worse,
    silence where the symbol was.

    Most spans in a systems book are identifiers, and those need almost nothing: the
    snake_case rule downstream already says "tenant id". The two that need work are

      * CamelCase domain names — `PaymentCaptured`, `OrderCancellationAccepted`. Split at
        the internal capital so the voice reads two words. Done here rather than globally
        because prose is full of CamelCase that must survive intact: SaaS, NoSQL, GraphQL.
        Being inside backticks is the signal that a token is an identifier.
      * operators — a handful of spans are query fragments, and an operator carries the
        meaning of the clause. `remaining > 0` is a precondition; drop the `>` and the
        sentence asserts nothing.

    These words are spoken but do not exist on the page, so they align to nothing and cost
    a little mapping coverage. Across this book that is roughly sixty words in 158,000, and
    the alternative is a sentence the listener cannot follow.
    """
    s = inner.strip()
    if not s:
        return ""
    # CamelCase, but not an all-caps run: `PaymentCaptured` splits, `SLO` and `ID` do not.
    s = re.sub(r"[A-Za-z][A-Za-z0-9]*", lambda m: _split_camel(m.group(0)), s)
    # An all-caps SQL keyword is spelled out rather than read; see SQL_KEYWORDS.
    s = re.sub(r"\b[A-Z]+\b",
               lambda m: m.group(0).lower() if m.group(0) in SQL_KEYWORDS else m.group(0), s)
    # SCREAMING_SNAKE identifiers — the graph edge labels `OBSERVED_AT_IP`, `USED_DEVICE`,
    # `CONTACTED_FROM`. Upper case sends this voice into spelling mode and it does not even
    # spell accurately: `OBSERVED_AT_IP, USED_DEVICE, USED_CARD` was rendered "OBS or VAT.
    # ATIP, USE device, USE card". The same labels in lower case came back verbatim. An
    # underscore is what distinguishes an identifier from an acronym like `API` or `SLO`,
    # which must keep its capitals to be spelled on purpose.
    # No `\b` around the run: an underscore is itself a word character, so `\bOBSERVED\b`
    # never matches inside `OBSERVED_AT_IP`.
    #
    # Two triggers, because one label in the same list of edge types has no underscore at
    # all (`REFERRED`) and would otherwise have been the only one spelled out. A run of four
    # or more capitals inside backticks is a word, not an initialism: every acronym this book
    # puts in a code span is a SQL keyword, and every acronym that must keep its capitals to
    # be spelled deliberately — API, SLO, RPO, TTL — appears in prose, which never reaches
    # this function.
    if "_" in inner:
        s = re.sub(r"[A-Z]{2,}", lambda m: m.group(0).lower(), s)
    s = re.sub(r"[A-Z]{4,}", lambda m: m.group(0).lower(), s)
    # Call and tuple syntax. `PaymentCaptured(provider_transaction_id)` reads as an event
    # and the field it carries; a comma is the pause that says so.
    s = re.sub(r"\s*\(\s*", ", ", s)
    s = re.sub(r"\s*\)\s*", " ", s)
    # `product:{id}` is one key template, not two things.
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s*:\s*", " ", s)
    # Two-character operators first, or `>=` becomes "greater than equals".
    for pat, word in (
        (r">=", " at least "), (r"<=", " at most "), (r"!=", " not equal to "),
        (r"=", " equals "), (r">", " greater than "), (r"<", " less than "),
        (r"\+", " plus "),
    ):
        s = re.sub(r"\s*" + pat + r"\s*", word, s)
    # Only a spaced hyphen is arithmetic. An unspaced one belongs to `B-tree`, `us-east-2a`
    # or an illustrative identifier like `customer-84`.
    s = re.sub(r"(?<=\s)-(?=\s)", "minus", s)
    # A bound parameter. Naming it is what a narrator does; "question mark" is not.
    s = re.sub(r"\s*\?", " a parameter", s)
    # A path or a slash-joined pair inside code: the separator is not spoken.
    s = re.sub(r"\s*/\s*", " ", s)
    s = s.replace(",", ", ")
    # Abbreviations that are read, not spelled, when they appear in an index definition.
    s = re.sub(r"\bDESC\b", "descending", s, flags=re.I)
    s = re.sub(r"\bASC\b", "ascending", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" ,")


# Compound acronyms and slash pairs that are spoken as their parts, not as "slash". Anything
# else keeps a spoken separator, chosen below by whether the source spaced the slash.
SLASH_PAIRS = re.compile(r"\b(I/O|pub/sub|read/write|write/read|and/or)\b", re.I)

# A currency amount with the scale word that may follow it. Both are needed together: the
# unit is spoken after a scale ("4.76 million dollars") but before the fraction ("4 dollars
# 82 cents"), and no rule reading only the digits can tell those apart.
CURRENCY = re.compile(
    r"(\b(?:a|an|the)\s+(?:[a-z]+\s+){0,2})?"
    r"\$(\d+(?:,\d{3})*)(?:\.(\d+))?(\s+(?:thousand|million|billion|trillion))?"
    r"(\s+[a-z]+)?\b", re.I
)

# Words that, following an amount, mean the amount is the noun rather than modifying one.
# "credit $12 to platform fee revenue" keeps the plural; "a $12 platform fee" does not.
NOT_A_NOUN = {
    "to", "from", "and", "or", "for", "in", "on", "of", "at", "than", "while", "unless",
    "per", "by", "is", "was", "were", "after", "before", "with", "under", "over", "but",
}


def _speak_currency(m: re.Match) -> str:
    det, whole, frac, scale, tail = m.groups()
    tail = tail or ""
    # Attributive use: "a $12 platform fee" is spoken "a 12 dollar platform fee", singular.
    # An article ahead of the amount and a noun behind it are the two cheap signals; either
    # one alone is wrong often enough to matter, since an article can belong to an earlier
    # clause and a following lowercase word is as likely to be a preposition as a noun.
    if det and not scale and not frac and tail.strip() and tail.strip() not in NOT_A_NOUN:
        return f"{det}{whole} dollar{tail}"
    det = det or ""
    if scale:
        # "4.76 million dollars". Cents make no sense at this magnitude, and the decimal is
        # part of the quantity rather than a separate unit.
        amount = f"{whole}.{frac}" if frac else whole
        return f"{det}{amount}{scale} dollars{tail}"
    if frac and len(frac) == 2:
        return f"{det}{whole} dollars {frac} cents{tail}"
    if frac:
        return f"{det}{whole} point {frac} dollars{tail}"
    return f"{det}{whole} dollars{tail}"


def clean_inline(line: str) -> str:
    line = re.sub(r"`([^`]*)`", lambda m: speak_code(m.group(1)), line)  # inline code
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)      # images
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # links keep their text
    # Markdown task-list markers. `- [ ] item` keeps its bullet stripped elsewhere, but the
    # checkbox itself stayed, so the voice read the brackets: 24 of them in one chapter, 34 in
    # another. They carry no spoken meaning.
    line = re.sub(r"^\s*\[[ xX]?\]\s*", "", line)
    line = re.sub(r"(?<=\s)\[[ xX]?\]\s*", "", line)
    # Bare square brackets are template notation — "It helps [specific customer] move from
    # [painful current state]". They are visual: a narrator reads the words inside and drops
    # the brackets, which is also what the page's own tokeniser does, since brackets are not
    # word characters. Links were already unwrapped above, so nothing here can be a link.
    line = re.sub(r"\[([^\[\]]*)\]", r"\1", line)
    # `<name>` is a fill-in placeholder in the source, not a tag; the word inside is what
    # should be spoken. Real tags are dropped entirely.
    line = re.sub(r"</?(?:br|em|strong|span|div|a|img|sup|sub|code|pre|p|ul|ol|li|hr)\b[^>]*/?>",
                  "", line, flags=re.I)
    line = re.sub(r"<([A-Za-z][\w -]*)>", r"\1", line)
    # Bold-italic first: `***x***` defeats both rules below. `\*\*` matches, then `[^*]+`
    # fails on the third asterisk, and the italic rule's `(?<!\*)` refuses to start there.
    # So the markers survived into the narration, and the voice read them out — chapter 10
    # of one book says "asterisk, asterisk, asterisk local design" and then degenerates into
    # "asterisksisks", because a repeated nonsense token is exactly what makes the LLM loop.
    line = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", line)    # bold italic
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)        # bold
    line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)  # italic
    # Underscore italics, but only when the underscores delimit a word rather than sit inside
    # one. The unanchored form matched *across* identifiers: given
    # "`agency_account` -> `client` -> `source_export`", it paired the underscore in
    # agency_account with the one in source_export, deleted both, and produced "agencyaccount"
    # and "sourceexport". The voice then read those fused non-words, which is both wrong and
    # the kind of nonsense token that sends the LLM into a repetition loop.
    line = re.sub(r"(?<![A-Za-z0-9])_([^\s_][^_]*?)_(?![A-Za-z0-9])", r"\1", line)
    # snake_case is spoken as separate words: "agency account", not "agencyaccount" and not
    # "agency underscore account".
    line = re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", " ", line)
    # Arrows carry the meaning of a chain and were being dropped, leaving a list of nouns with
    # no relationship between them — "agency account client source export change digest".
    line = re.sub(r"\s*(?:->|=>|\u2192|\u21d2)\s*", ", then ", line)
    # A fill-in blank in a template sentence: "We chose this system because ___." The rules
    # above leave it alone (the italic rule needs non-underscore content between the markers,
    # the snake_case rule needs word characters on both sides), so it reached the voice, and
    # a run of identical characters is the input most likely to send the LLM into a loop.
    # Saying "blank" is what a narrator reading a worksheet aloud does.
    line = re.sub(r"_{2,}", "blank", line)
    # Currency. `$120` is otherwise read as "120" with the unit silently dropped, which in a
    # cost chapter changes the sentence's meaning rather than its polish. Where the unit goes
    # depends on what follows the number, and getting that wrong is worse than dropping it:
    # a first attempt turned `$4.8 million` into "4 dollars.8 million", which is both wrong
    # and a sentence boundary the segmenter would have believed.
    line = CURRENCY.sub(_speak_currency, line)
    line = re.sub(r"(\d)\s*%", r"\1 percent", line)
    # An identifier of the form `I-2048` \u2014 an illustrative invoice or order number. The
    # hyphen between a letter and digits is not spoken; the space keeps the letter separate
    # so it is not swallowed into the number.
    line = re.sub(r"\b([A-Za-z])-(\d)", r"\1 \2", line)
    # Slashes. A spaced slash comes from a table cell listing alternatives, where a comma is
    # the reading. An unspaced one joins two alternatives in prose ("warehouse/lakehouse"),
    # where "or" is the reading. The compound acronyms are neither.
    line = SLASH_PAIRS.sub(lambda m: m.group(1).replace("/", " "), line)
    # A slash with nothing after it: a table cell ended on an open alternation
    # ("versioned amendment /"). Nothing to separate, so nothing to say.
    line = re.sub(r"\s*/\s*(?=[.,;:!?]|$)", "", line)
    line = re.sub(r"\s+/\s+", ", ", line)
    line = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " or ", line)
    # `B+tree` in prose rather than in a code span, where the plus is part of the name and
    # is spoken. Only between word characters, so a stray plus is still dropped downstream.
    line = re.sub(r"(?<=[A-Za-z0-9])\+(?=[A-Za-z0-9])", " plus ", line)
    line = line.replace("~~", "")
    # Catch-all. Any asterisk still here is unmatched or nested in a way the rules above did
    # not anticipate, and there is no reading of one that belongs in speech. Removing it
    # unconditionally is safer than another special case, because the failure mode of a
    # missed marker is not a small blemish — it is a degenerate loop that destroys the
    # passage. The `--stats` scan reports anything else that looks like surviving markup.
    line = line.replace("*", "")
    # A semicolon joins independent clauses, and a listener has no way to hear the join. Read
    # each clause as its own sentence instead, which is also what keeps segments short: this
    # book puts semicolon lists inside table cells, and one such row reached the voice as a
    # single 218-character segment carrying eight noun phrases. It came out as "QH, WEF codes,
    # and paid boot and fulfilled orders" for "queue age, webhook retries, and
    # paid-but-unfulfilled orders". Split into sentences, the same clause renders correctly.
    #
    # The damage is not confined to that row. Across this book there are 562 semicolons and
    # 249 sentences over the engine's 220-character segment budget, so a segment packed with
    # disconnected clauses is the normal case rather than an outlier.
    # The optional quote is not decoration: this book's semicolon clauses often open on a
    # quoted term ('is vague; "complete through the previous UTC day" is not'), and a rule
    # expecting a word character right after the semicolon skips every one of them.
    # Smart quotes are still smart at this point — they are folded to ASCII further down — so
    # the class has to list them, or the four clauses that open on a curly quote are missed.
    line = re.sub(r"\s*;\s+([\"'“‘]?)(\w)",
                  lambda m: ". " + m.group(1) + m.group(2).upper(), line)
    line = re.sub(r"\s*;\s*$", ".", line)
    # A hyphen between two lower-case words is a compound modifier. It is not spoken, and
    # leaving it in makes the voice fuse the parts: `paid-but-unfulfilled` was rendered "paid
    # button fulfilled". Spacing it also improves page-word mapping, because the page's own
    # tokeniser already treats a hyphen as a boundary.
    line = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", line)
    # An em dash is a pause in print; a comma is the same pause in speech.
    line = re.sub(r"\s*[—–]\s*", ", ", line)
    line = line.replace("…", "...")
    # Smart quotes confuse some tokenizers and add nothing when spoken.
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]:
        line = line.replace(a, b)
    line = split_mangled(line)
    return re.sub(r"[ \t]+", " ", line).strip()


# Compounds this voice cannot pronounce as one token, with the spacing that fixes them.
#
# Found by aggregating the alignment manifests: a word the voice mangles is never recognised,
# so it shows up as interpolated in every occurrence. Most words that flag that way are
# innocent — "countermetric", "tradeoff", "codebase" and "quickstart" are all spoken correctly
# and merely transcribed as two words — so each candidate was listened to before being added
# here. These two genuinely fail: "timezone" was spoken as "Heideheb" and "signup" as
# "SignGen".
MANGLED_COMPOUNDS = {
    "timezone": "time zone",
    "timezones": "time zones",
    "signup": "sign up",
    "signups": "sign ups",
}


def split_mangled(line: str) -> str:
    def sub(m):
        word = m.group(0)
        fixed = MANGLED_COMPOUNDS[word.lower()]
        return fixed.capitalize() if word[:1].isupper() else fixed
    return re.sub(
        r"\b(?:" + "|".join(sorted(MANGLED_COMPOUNDS, key=len, reverse=True)) + r")\b",
        sub, line, flags=re.I,
    )


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

        | Partner | Provides           | Requires        |
        | Product | field synthesis... | roadmap context... |

        -> "Product. Provides: field synthesis... Requires: roadmap context..."

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
    # Speech has no case, but the model's pronunciation is sensitive to it: a paragraph
    # beginning with a lowercase word came out mangled — "founder intervention recorded" was
    # spoken as "Sharpen intervention recorded", and reproducibly so across two sampling
    # seeds, which is what ruled out a sampling glitch. Lowercase openings arise naturally
    # here because snake_case identifiers (`founder_intervention`) become ordinary words and
    # each list item is its own paragraph.
    paragraphs = [p[:1].upper() + p[1:] if p[:1].islower() else p for p in paragraphs]
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
        # Same checkbox and placeholder handling as `clean_inline`, so the two tokenisers
        # agree; a difference here shows up as unmapped words rather than as a warning.
        line = re.sub(r"\[[ xX]?\]\s*", "", line)
        line = re.sub(r"\[([^\[\]]*)\]", r"\1", line)
        line = re.sub(r"</?(?:br|em|strong|span|div|a|img|sup|sub|code|pre|p|ul|ol|li|hr)\b[^>]*/?>",
                      "", line, flags=re.I)
        line = re.sub(r"<([A-Za-z][\w -]*)>", r"\1", line)
        if set(line) <= {"-", "=", "*", "_"} and len(line) >= 3:
            continue
        # clean_inline without the em-dash rewrite: it changes punctuation, not words.
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"(?<![A-Za-z0-9])_([^\s_][^_]*?)_(?![A-Za-z0-9])", r"\1", line)
        line = re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", " ", line)
        line = re.sub(r"\s*(?:->|=>|\u2192|\u21d2)\s*", ", then ", line)
        out.append(line)
    return " ".join(out)


def align_tokens(spoken: list[str], page: list[str]) -> list[int]:
    """`spoken[i]` -> index into `page`, or -1.

    `SequenceMatcher` rather than a two-pointer walk with bounded lookahead. The walk was
    adequate while narration and page text differed only locally, but rendering a table as
    one sentence per row repeats the column headers on every row — dozens of insertions, each
    beyond a 24-token lookahead. It desynchronised and stayed desynchronised: chapter 9 fell
    from 100% mapped to 47.8%.

    A matcher handles insertions, deletions and substitutions without any of them shifting
    what follows, which is the same reason `align-narration.py` uses one. `autojunk` must
    stay off or the most common words in a long chapter are discarded as noise.
    """
    matcher = difflib.SequenceMatcher(None, spoken, page, autojunk=False)
    out = [-1] * len(spoken)
    for si, pi, size in matcher.get_matching_blocks():
        for k in range(size):
            out[si + k] = pi + k
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

    # Markup that reaches the voice is not cosmetic. A `***` that survived cleaning made one
    # chapter read "asterisk, asterisk, asterisk" and then degenerate into a repetition loop,
    # destroying the passage. Report anything that still looks like markup, always — a silent
    # converter is how that shipped.
    residue = {
        "asterisk": r"\*",
        "pipe (table)": r"\|",
        "backtick": r"`",
        "bracket": r"[\[\]]",
        "heading hash": r"(?:^|\s)#{1,6}\s",
        "blockquote": r"(?:^|\n)\s*>",
        "html tag": r"<[a-zA-Z/][^>]*>",
        "shortcode": r"\{\{",
    }
    for label, pattern in residue.items():
        hits = re.findall(pattern, out, flags=re.MULTILINE)
        if hits:
            sample = re.search(pattern, out, flags=re.MULTILINE)
            context = out[max(0, sample.start() - 40) : sample.start() + 40].replace("\n", " ")
            print(
                f"warning: {args.input.name}: {len(hits)} surviving {label} in narration "
                f"text — the voice will read it: ...{context}...",
                file=sys.stderr,
            )

    # A single over-long sentence is spoken as one segment, and a segment the AR loop cannot
    # finish is silently cut short (see `tts_core::text::split_long`). The engine now splits
    # and retries, but flagging it here points at the sentence rather than the symptom.
    long_sentences = [s for s in re.split(r"(?<=[.!?])\s+", out) if len(s) > 400]
    if long_sentences:
        print(
            f"warning: {args.input.name}: {len(long_sentences)} sentence(s) over 400 chars; "
            f"longest {max(len(s) for s in long_sentences)}: {long_sentences[0][:70]}...",
            file=sys.stderr,
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
