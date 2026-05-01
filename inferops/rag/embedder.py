"""Sentence-transformer embedding wrapper (lazy-loaded to avoid import cost)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
_model: "SentenceTransformer | None" = None


def get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Force CPU so the embedder doesn't compete with vLLM for VRAM
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return normalized embeddings for a list of strings."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
