"""Markdown-aware text chunker for the knowledge corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str       # filename stem, e.g. "paged_attention"
    section: str      # nearest heading above this chunk
    char_start: int


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Strip YAML frontmatter and return (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end]
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text[end + 4:].lstrip()


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split on H2/H3 headings. Returns list of (heading, content) pairs."""
    pattern = re.compile(r"^(#{2,3} .+)$", re.MULTILINE)
    positions = [(m.start(), m.group()) for m in pattern.finditer(text)]

    if not positions:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for i, (pos, heading) in enumerate(positions):
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[pos + len(heading):next_pos].strip()
        sections.append((heading.lstrip("#").strip(), content))
    return sections


def chunk_document(
    path: Path,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Split a Markdown document into overlapping text chunks.

    chunk_size and overlap are measured in whitespace-split words, not chars.
    """
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    source = meta.get("source", path.stem)
    sections = _split_by_headings(body)

    chunks: list[Chunk] = []
    for section_title, section_text in sections:
        words = section_text.split()
        step = chunk_size - overlap
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 30:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    section=section_title,
                    char_start=start,
                ))
            start += step
            if end == len(words):
                break

    return chunks


def chunk_directory(
    directory: Path,
    glob: str = "*.md",
    **kwargs,
) -> list[Chunk]:
    """Chunk all Markdown files in a directory."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob(glob)):
        chunks.extend(chunk_document(path, **kwargs))
    return chunks
