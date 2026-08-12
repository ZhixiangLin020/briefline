"""Validate local Markdown targets and heading anchors without network access."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _without_fenced_code(text: str) -> str:
    kept: list[str] = []
    active_fence: str | None = None
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if active_fence is None:
                active_fence = marker
            elif marker == active_fence:
                active_fence = None
            continue
        if active_fence is None:
            kept.append(line)
    return "\n".join(kept)


def _github_slug(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("`", "").strip().lower()
    characters = []
    for character in value:
        category = unicodedata.category(character)
        if character in {"-", "_", " "} or not category.startswith(("P", "S")):
            characters.append(character)
    return re.sub(r"\s+", "-", "".join(characters))


def _heading_anchors(path: Path) -> set[str]:
    counts: Counter[str] = Counter()
    anchors: set[str] = set()
    text = _without_fenced_code(path.read_text(encoding="utf-8"))
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match is None:
            continue
        base = _github_slug(match.group(1))
        duplicate_index = counts[base]
        counts[base] += 1
        anchors.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return anchors


def _markdown_files(requested_paths: Sequence[str]) -> list[Path]:
    if requested_paths:
        paths = [(PROJECT_ROOT / value).resolve() for value in requested_paths]
    else:
        paths = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
    return [path for path in paths if path.is_file()]


def validate(paths: Iterable[Path]) -> list[str]:
    failures: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for source in paths:
        text = _without_fenced_code(source.read_text(encoding="utf-8"))
        for match in MARKDOWN_LINK_RE.finditer(text):
            raw_target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if raw_target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = raw_target.partition("#")
            target = source if not path_part else (source.parent / unquote(path_part)).resolve()
            source_label = source.relative_to(PROJECT_ROOT)
            if not target.exists():
                failures.append(f"{source_label}: missing target {raw_target}")
                continue
            if anchor and target.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(target, _heading_anchors(target))
                if unquote(anchor) not in anchors:
                    failures.append(f"{source_label}: missing anchor {raw_target}")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check repository-local Markdown links and heading anchors."
    )
    parser.add_argument("paths", nargs="*", help="Markdown paths relative to the repository root.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _markdown_files(args.paths)
    failures = validate(paths)
    if failures:
        print("Markdown link validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"Markdown link validation passed for {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
