"""Unit tests for the Markdown chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferops.rag.chunker import Chunk, chunk_document, chunk_directory


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_chunk_document_returns_chunks(tmp_path):
    doc = _write(tmp_path, "test.md", """\
---
source: Test Doc
section: Intro
---

## Introduction

Word " * 500

""".replace('"', "word ") + "word " * 500)

    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_document_uses_frontmatter_source(tmp_path):
    doc = _write(tmp_path, "paged.md", """\
---
source: PagedAttention Paper
---

## Core Algorithm

""" + "word " * 120)

    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert all(c.source == "PagedAttention Paper" for c in chunks)


def test_chunk_document_falls_back_to_stem_when_no_frontmatter(tmp_path):
    doc = _write(tmp_path, "myfile.md", "## Section\n" + "word " * 50)
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert all(c.source == "myfile" for c in chunks)


def test_chunk_document_splits_by_headings(tmp_path):
    doc = _write(tmp_path, "multi.md", """\
## Section One
""" + "word " * 60 + """

## Section Two
""" + "word " * 60)

    chunks = chunk_document(doc, chunk_size=50, overlap=5)
    sections = {c.section for c in chunks}
    assert "Section One" in sections
    assert "Section Two" in sections


def test_chunk_directory_processes_all_md_files(tmp_path):
    for i in range(3):
        _write(tmp_path, f"doc{i}.md", "## Sec\n" + "word " * 80)

    chunks = chunk_directory(tmp_path, chunk_size=60, overlap=5)
    assert len(chunks) >= 3


def test_chunk_directory_skips_short_chunks(tmp_path):
    _write(tmp_path, "tiny.md", "## Sec\nHi.")
    chunks = chunk_document(tmp_path / "tiny.md", chunk_size=100, overlap=10)
    # The two-word chunk "Hi." should be skipped (< 30 chars)
    assert all(len(c.text.strip()) > 30 for c in chunks)
