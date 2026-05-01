"""Unit tests for knowledge_retriever tool — mocks Chroma and embedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        {"text": "PagedAttention reduces fragmentation.", "source": "paged_attention", "section": "Algorithm", "score": 0.92},
        {"text": "Chunked prefill splits long prompts.", "source": "chunked_prefill", "section": "When to Enable", "score": 0.88},
    ]
    with _mock_collection_size(10), _mock_embed(), _mock_store_query(hits):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="prefill optimization", top_k=2))

    assert isinstance(out, KnowledgeRetrieverOutput)
    assert out.total_found == 2
    assert out.chunks[0].source == "paged_attention"
    assert out.chunks[1].score == 0.88
    assert not out.index_empty


def test_knowledge_retriever_returns_index_empty_when_no_index():
    with _mock_collection_size(0):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="anything"))

    assert out.index_empty is True
    assert out.chunks == []
    assert out.total_found == 0


def test_knowledge_retriever_limits_top_k():
    hits = [
        {"text": f"chunk {i}", "source": "doc", "section": "sec", "score": 0.9 - i * 0.1}
        for i in range(3)
    ]
    with _mock_collection_size(5), _mock_embed(), _mock_store_query(hits[:2]):
        out = knowledge_retriever(KnowledgeRetrieverInput(query="test", top_k=2))

    assert out.total_found == 2
