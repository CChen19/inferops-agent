"""Chroma vector store interface for the knowledge corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferops.rag.chunker import Chunk

_COLLECTION_NAME = "inferops_corpus"
_DEFAULT_DB_PATH = "data/chroma"
CORPUS_VERSION = "inferops-corpus-1"
INDEX_INCOMPATIBLE_MESSAGE = (
    "knowledge index version is incompatible with this build — "
    "rebuild with `scripts/build_corpus.py` (CORPUS_VERSION mismatch or missing)"
)


def _client(db_path: str = _DEFAULT_DB_PATH):
    import chromadb
    return chromadb.PersistentClient(path=db_path)


def build_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    db_path: str = _DEFAULT_DB_PATH,
    reset: bool = False,
) -> None:
    """Upsert chunks + embeddings into the Chroma collection.

    Persists ``CORPUS_VERSION`` on both collection metadata and each chunk.
    """
    client = _client(db_path)
    if reset:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "corpus_version": CORPUS_VERSION},
    )
    # Refresh collection-level version even when the collection already existed.
    try:
        col.modify(metadata={"hnsw:space": "cosine", "corpus_version": CORPUS_VERSION})
    except Exception:
        pass

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c.text for c in chunks]
    metadatas: list[dict[str, Any]] = [
        {"source": c.source, "section": c.section, "version": CORPUS_VERSION}
        for c in chunks
    ]

    col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def _collection_corpus_version(col) -> str:
    meta = getattr(col, "metadata", None) or {}
    if isinstance(meta, dict):
        return str(meta.get("corpus_version") or meta.get("version") or "")
    return ""


def index_version_compatible(db_path: str = _DEFAULT_DB_PATH) -> tuple[bool, str]:
    """Return (ok, reason). Empty/unbuilt index is not incompatible — just empty.

    Incompatible = collection has chunks but collection/chunk version is missing
    or != CORPUS_VERSION.
    """
    try:
        client = _client(db_path)
        col = client.get_collection(_COLLECTION_NAME)
    except Exception:
        return True, ""  # not built — caller's empty path
    count = col.count()
    if count <= 0:
        return True, ""
    coll_ver = _collection_corpus_version(col)
    if coll_ver and coll_ver != CORPUS_VERSION:
        return False, INDEX_INCOMPATIBLE_MESSAGE
    # Peek one chunk for version when collection metadata is missing/empty.
    try:
        peek = col.peek(limit=1)
        metas = (peek.get("metadatas") or [None])[0]
        chunk_ver = ""
        if isinstance(metas, dict):
            chunk_ver = str(metas.get("version") or "")
        elif isinstance(metas, list) and metas and isinstance(metas[0], dict):
            chunk_ver = str(metas[0].get("version") or "")
    except Exception:
        chunk_ver = ""
    effective = coll_ver or chunk_ver
    if not effective or effective != CORPUS_VERSION:
        return False, INDEX_INCOMPATIBLE_MESSAGE
    return True, ""


def query(
    query_embedding: list[float],
    top_k: int = 5,
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Return top_k chunks with their Chroma id, corpus version, text, source, section, and score.

    score is cosine distance (lower = more similar). Converted to similarity = 1 - distance.
    Returns [] when the index is missing or version-incompatible (safe empty-RAG).
    """
    client = _client(db_path)
    try:
        col = client.get_collection(_COLLECTION_NAME)
    except Exception:
        return []

    if col.count() <= 0:
        return []

    ok, _reason = index_version_compatible(db_path)
    if not ok:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        version = str((meta or {}).get("version") or "")
        if version != CORPUS_VERSION:
            # Per-chunk mismatch: skip inventing sources from stale chunks.
            continue
        hits.append({
            "chunk_id": chunk_id,
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
            "version": version,
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
