"""Build the RAG knowledge corpus index.

Downloads/copies Markdown docs → chunks → embeds → stores in Chroma.
Run once (or after adding new corpus documents).

Usage:
    python scripts/build_corpus.py                        # default: data/corpus → data/chroma
    python scripts/build_corpus.py --corpus-dir my/docs  # custom corpus dir
    python scripts/build_corpus.py --reset               # wipe and rebuild index
    python scripts/build_corpus.py --dry-run             # print chunk plan only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from inferops.rag.chunker import chunk_directory
from inferops.rag.embedder import embed_texts
from inferops.rag.store import build_index, collection_size

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG corpus index")
    parser.add_argument("--corpus-dir", default="data/corpus", help="Directory of Markdown files")
    parser.add_argument("--db-path", default="data/chroma", help="Chroma DB output path")
    parser.add_argument("--chunk-size", type=int, default=400, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collection")
    parser.add_argument("--dry-run", action="store_true", help="Show chunk plan, skip embedding")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        console.print(f"[red]Corpus directory not found: {corpus_dir}[/]")
        sys.exit(1)

    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        console.print(f"[red]No .md files found in {corpus_dir}[/]")
        sys.exit(1)

    console.print(f"\n[bold]Chunking[/] {len(md_files)} documents from [cyan]{corpus_dir}[/]")
    t0 = time.time()

    chunks = chunk_directory(corpus_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    console.print(f"  → {len(chunks)} chunks (size≤{args.chunk_size} words, overlap={args.overlap})")

    if args.dry_run:
        t = Table(show_header=True, box=None, padding=(0, 1))
        t.add_column("source", style="cyan")
        t.add_column("section", style="dim")
        t.add_column("words", justify="right")
        t.add_column("preview")
        for c in chunks[:20]:
            words = len(c.text.split())
            preview = c.text[:60].replace("\n", " ") + "…"
            t.add_row(c.source[:30], c.section[:30], str(words), preview)
        console.print(t)
        if len(chunks) > 20:
            console.print(f"  [dim]… {len(chunks) - 20} more chunks[/]")
        console.print("[yellow]Dry run — skipping embedding and indexing.[/]")
        return

    console.print(f"\n[bold]Embedding[/] {len(chunks)} chunks with bge-base-zh-v1.5 …")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    console.print(f"  → embedding dim: {len(embeddings[0])}")

    console.print(f"\n[bold]Indexing[/] → [cyan]{args.db_path}[/] (reset={args.reset})")
    build_index(chunks, embeddings, db_path=args.db_path, reset=args.reset)

    n = collection_size(args.db_path)
    elapsed = time.time() - t0
    console.print(f"\n[green]Done.[/] {n} chunks indexed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
