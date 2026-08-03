#!/usr/bin/env python3
"""Install narration output into a Hugo book site, in the layout the site expects.

The site's convention, from `layouts/partials/chapter-audio-player.html`:

    audio = "/audio/books/<slug>/<dir>/chapter.<ext>"
    data-audio-manifest = replaceRE `[^/]+$` `manifest.json` $audio

So the manifest must sit *beside* the audio, and the audio must be named `chapter.<ext>` —
the player resolves `delivery.segments[].file` relative to the manifest URL, so that field
has to be rewritten to the published filename rather than the working one.

Also writes `audio` and `audio_duration` into each chapter's TOML front matter, which is
what makes the player appear at all (`{{- if $audio -}}`).

Usage:
    publish-narration.py --narration narration --site ../swe-verify \\
        --slug the-change-interface [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Reading order, and the directory each becomes on the site. `chapter-NNN` is zero-padded to
# three digits to match the existing books; `introduction` and `conclusion` keep their names
# because nothing globs these directories numerically — the only globber in the site's
# scripts matches `chapter-part-\d{3}.mp3` *inside* a directory, not the directories.
def site_dir(stem: str) -> str:
    m = re.fullmatch(r"chapter-(\d+)", stem)
    return f"chapter-{int(m.group(1)):03d}" if m else stem


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def set_front_matter(md: Path, audio: str, duration: str, dry: bool) -> str:
    """Set `audio` and `audio_duration` in TOML front matter, preserving everything else."""
    text = md.read_text()
    if not text.startswith("+++"):
        return "no TOML front matter — skipped"
    end = text.find("\n+++", 3)
    if end < 0:
        return "unterminated front matter — skipped"
    head, body = text[3:end], text[end:]

    lines = [ln for ln in head.splitlines() if not re.match(r"\s*audio(_duration)?\s*=", ln)]
    # Insert after `chapter`/`title` so the block stays readable.
    at = len(lines)
    for i, ln in enumerate(lines):
        if re.match(r"\s*(chapter|weight)\s*=", ln):
            at = i + 1
    lines[at:at] = [f'audio_duration = "{duration}"', f'audio = "{audio}"']
    out = "+++" + "\n".join(lines) + body
    if not dry:
        md.write_text(out)
    return "front matter updated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--narration", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True, help="repo root of the Hugo site")
    ap.add_argument("--slug", required=True, help="book slug under content/books")
    ap.add_argument("--ext", default="opus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    content = args.site / "content" / "books" / args.slug
    if not content.is_dir():
        sys.exit(f"no such book: {content}")
    audio_root = args.site / "static" / "audio" / "books" / args.slug

    manifests = sorted(args.narration.glob("*.manifest.json"))
    if not manifests:
        sys.exit(f"no manifests in {args.narration}")

    print(f"{'chapter':<16} {'-> site dir':<18} {'audio':>9} {'manifest':>9}  front matter")
    published = 0
    for man_path in manifests:
        stem = man_path.name.removesuffix(".manifest.json")
        src_audio = args.narration / f"{stem}.{args.ext}"
        md = content / f"{stem}.md"
        if not src_audio.exists():
            print(f"{stem:<16} missing {src_audio.name}")
            continue
        if not md.exists():
            print(f"{stem:<16} no markdown at {md.name} — skipped")
            continue

        dest_dir = audio_root / site_dir(stem)
        dest_audio = dest_dir / f"chapter.{args.ext}"
        dest_man = dest_dir / "manifest.json"

        manifest = json.loads(man_path.read_text())
        # The player resolves this relative to the manifest, so it must be the *published*
        # name. Leaving the working name here is the one mistake that breaks silently: the
        # page renders, the manifest loads, and audio 404s.
        for seg in manifest["delivery"]["segments"]:
            seg["file"] = f"chapter.{args.ext}"
        duration = mmss(manifest["delivery"]["totalDurationSeconds"])
        web_path = f"/audio/books/{args.slug}/{site_dir(stem)}/chapter.{args.ext}"

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_audio, dest_audio)
            dest_man.write_text(json.dumps(manifest, separators=(",", ":")))
        note = set_front_matter(md, web_path, duration, args.dry_run)
        print(
            f"{stem:<16} {site_dir(stem):<18} "
            f"{src_audio.stat().st_size // 1024:>8}K {len(man_path.read_text()) // 1024:>8}K  {note}"
        )
        published += 1

    print(f"\n{published} chapter(s) {'would be ' if args.dry_run else ''}published to {audio_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
