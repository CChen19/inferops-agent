"""Chroma vector store interface for the knowledge corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferops.rag.chunker import Chunk

_COLLECTION_NAME = "inferops_corpus"
_DEFAULT_DB_PATH = "data/chroma"


def _client(db_path: str = _DEFAULT_DB_PATH):
    import chromadb
    return chromadb.PersistentClient(path=db_path)


def build_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    db_path: str = _DEFAULT_DB_PATH,
    reset: bool = False,
) -> None:
    """Upsert chunks + embeddings into the Chroma collection."""
    client = _client(db_path)
    if reset:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c.text for c in chunks]
    metadatas: list[dict[str, Any]] = [
        {"source": c.source, "section": c.section} for c in chunks
    ]

    col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query(
    query_embedding: list[float],
    top_k: int = 5,
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Return top_k chunks as dicts with keys: text, source, section, score.

    score is cosine distance (lower = more similar). Converted to similarity = 1 - distance.
    """
    client = _client(db_path)
    try:
        col = client.get_collection(_COLLECTION_NAME)
    except Exception:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
            "score": round(1.0 - dist, 4),
        })
    return hits


def collection_size(db_path: str = _DEFAULT_DB_PATH) -> int:
    """Return number of indexed chunks, or 0 if collection does not exist."""
    try:
        client = _client(db_path)
        col = client.get_collection(_COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0
