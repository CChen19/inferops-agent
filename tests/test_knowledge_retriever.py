"""Unit tests for knowledge_retriever tool — mocks Chroma and embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferops.citations import sources_from_context
from inferops.rag.chunker import Chunk
from inferops.rag.store import (
    CORPUS_VERSION,
    INDEX_INCOMPATIBLE_MESSAGE,
    build_index,
    index_version_compatible,
    query as query_store,
)
from inferops.tools.knowledge_retriever import (
    KnowledgeRetrieverInput,
    KnowledgeRetrieverOutput,
    knowledge_retriever,
)


def _mock_store_query(hits):
    return patch("inferops.tools.knowledge_retriever.query", return_value=hits)


def _mock_collection_size(n):
    return patch("inferops.tools.knowledge_retriever.collection_size", return_value=n)


def _mock_embed():
    return patch("inferops.tools.knowledge_retriever.embed_query", return_value=[0.1] * 768)


def test_knowledge_retriever_returns_chunks():
    hits = [
        {
            "chunk_id": "chunk_0",
            "text": "PagedAttention reduces fragmentation.",
            "source": "paged_attention",
            "section": "Algorithm",
            "version": "inferops-corpus-1",
            "score": 0.92,
        },
        {
            "chunk_id": "chunk_1",
            "text": "Chunked prefill splits long prompts.",
            "source": "chunked_prefill",
            "section": "When to Enable",
            "version": "inferops-corpus-1",
            "score": 0.88,
        },
    ]
    with (
        _mock_collection_size(10),
        patch(
            "inferops.tools.knowledge_retriever.index_version_compatible",
            return_value=(True, ""),
        ),
        _mock_embed(),
        _mock_store_query(hits),
    ):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="prefill optimization", top_k=2))

    assert isinstance(out, KnowledgeRetrieverOutput)
    assert out.total_found == 2
    assert out.chunks[0].source == "paged_attention"
    assert out.chunks[0].chunk_id == "chunk_0"
    assert out.chunks[0].version == "inferops-corpus-1"
    assert out.chunks[1].score == 0.88
    assert not out.index_empty
    assert not out.index_incompatible


def test_knowledge_retriever_returns_index_empty_when_no_index():
    with _mock_collection_size(0):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="anything"))

    assert out.index_empty is True
    assert out.index_incompatible is False
    assert out.chunks == []
    assert out.total_found == 0


def test_knowledge_retriever_incompatible_index_returns_no_sources():
    with (
        _mock_collection_size(5),
        patch(
            "inferops.tools.knowledge_retriever.index_version_compatible",
            return_value=(False, INDEX_INCOMPATIBLE_MESSAGE),
        ),
    ):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="anything"))

    assert out.index_incompatible is True
    assert out.index_empty is False
    assert out.chunks == []
    assert out.total_found == 0
    assert "rebuild" in out.message.lower() or "incompatible" in out.message.lower()
    # Planner surface must not invent parseable sources
    from inferops.agent.planner import _retrieve_knowledge

    with patch(
        "inferops.tools.knowledge_retriever.knowledge_retriever",
        return_value=out,
    ):
        ctx = _retrieve_knowledge("compute-bound", "chat_short")
    assert "incompatible" in ctx.lower() or "rebuild" in ctx.lower()
    assert sources_from_context(ctx) == set()


def test_store_build_index_records_corpus_version_on_collection_and_chunks():
    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    chunk = Chunk(text="scheduler guidance", source="doc", section="Scheduling", char_start=0)

    with patch("inferops.rag.store._client", return_value=client):
        build_index([chunk], [[0.1, 0.2]], db_path="unused")

    create_meta = client.get_or_create_collection.call_args.kwargs["metadata"]
    assert create_meta.get("corpus_version") == CORPUS_VERSION
    kwargs = collection.upsert.call_args.kwargs
    assert kwargs["ids"] == ["chunk_0"]
    assert kwargs["metadatas"] == [
        {"source": "doc", "section": "Scheduling", "version": CORPUS_VERSION}
    ]


def test_store_query_returns_chunk_id_and_version():
    collection = MagicMock()
    collection.count.return_value = 1
    collection.metadata = {"corpus_version": CORPUS_VERSION}
    collection.query.return_value = {
        "ids": [["chunk_0"]],
        "documents": [["scheduler guidance"]],
        "metadatas": [[
            {"source": "doc", "section": "Scheduling", "version": CORPUS_VERSION}
        ]],
        "distances": [[0.1]],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with (
        patch("inferops.rag.store._client", return_value=client),
        patch("inferops.rag.store.index_version_compatible", return_value=(True, "")),
    ):
        hits = query_store([0.1, 0.2], top_k=1, db_path="unused")

    assert hits == [
        {
            "chunk_id": "chunk_0",
            "text": "scheduler guidance",
            "source": "doc",
            "section": "Scheduling",
            "version": CORPUS_VERSION,
            "score": 0.9,
        }
    ]


def test_index_version_compatible_missing_version_is_incompatible():
    collection = MagicMock()
    collection.count.return_value = 3
    collection.metadata = {}  # no corpus_version
    collection.peek.return_value = {"metadatas": [{"source": "doc", "section": "x"}]}
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("inferops.rag.store._client", return_value=client):
        ok, reason = index_version_compatible("unused")

    assert ok is False
    assert "incompatible" in reason.lower() or "rebuild" in reason.lower()


def test_index_version_compatible_empty_collection_is_not_incompatible():
    collection = MagicMock()
    collection.count.return_value = 0
    collection.metadata = {}
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("inferops.rag.store._client", return_value=client):
        ok, reason = index_version_compatible("unused")

    assert ok is True
    assert reason == ""


def test_knowledge_retriever_limits_top_k():
    hits = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"chunk {i}",
            "source": "doc",
            "section": "sec",
            "version": "inferops-corpus-1",
            "score": 0.9 - i * 0.1,
        }
        for i in range(3)
    ]
    with (
        _mock_collection_size(5),
        patch(
            "inferops.tools.knowledge_retriever.index_version_compatible",
            return_value=(True, ""),
        ),
        _mock_embed(),
        _mock_store_query(hits[:2]),
    ):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="test", top_k=2))

    assert out.total_found == 2
