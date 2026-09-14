"""Tool: knowledge_retriever — semantic search over the vLLM knowledge corpus."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.observability import span
from inferops.rag.embedder import embed_query
from inferops.rag.store import (
    INDEX_INCOMPATIBLE_MESSAGE,
    collection_size,
    index_version_compatible,
    query,
)

_DEFAULT_DB_PATH = "data/chroma"


class KnowledgeRetrieverInput(BaseModel):
    query: str = Field(description="Natural language query, e.g. 'chunked prefill effect on TTFT'")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return")
    db_path: str = Field(default=_DEFAULT_DB_PATH, description="Path to the Chroma DB directory")


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    section: str
    version: str
    score: float


class KnowledgeRetrieverOutput(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_found: int
    index_empty: bool = False
    index_incompatible: bool = False
    message: str = ""


def knowledge_retriever(inp: KnowledgeRetrieverInput) -> KnowledgeRetrieverOutput:
    """
    Retrieve the most relevant knowledge chunks for a given query.

    Embeds the query with bge-base-zh-v1.5, searches the Chroma corpus index,
    and returns the top-k chunks with their id, source document, section heading,
    and corpus version.
    Returns index_empty=True (with no chunks) if the index has not been built yet.
    Returns index_incompatible=True (with no chunks) if the index has chunks but
    a missing/mismatched CORPUS_VERSION — safe empty-RAG, no invented sources.

    Use the returned chunks as grounding for hypothesis rationales. Each hypothesis
    rationale MUST include a [source: <source_field>] citation from a returned chunk.
    """
    with span("tool.knowledge_retriever", {"query": inp.query[:80], "top_k": str(inp.top_k)}):
        if collection_size(inp.db_path) == 0:
            return KnowledgeRetrieverOutput(
                query=inp.query,
                chunks=[],
                total_found=0,
                index_empty=True,
            )

        ok, reason = index_version_compatible(inp.db_path)
        if not ok:
            return KnowledgeRetrieverOutput(
                query=inp.query,
                chunks=[],
                total_found=0,
                index_incompatible=True,
                message=reason or INDEX_INCOMPATIBLE_MESSAGE,
            )

        q_emb = embed_query(inp.query)
        hits = query(q_emb, top_k=inp.top_k, db_path=inp.db_path)

    return KnowledgeRetrieverOutput(
        query=inp.query,
        chunks=[RetrievedChunk(**h) for h in hits],
        total_found=len(hits),
    )
