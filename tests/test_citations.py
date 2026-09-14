"""Existence-only tests for structured planner citations."""

from inferops.citations import (
    documents_from_context,
    sources_from_context,
    valid_structured_citations,
)


def _hypothesis(
    *,
    run_id="run_current",
    source="scheduler_doc",
    chunk_id="chunk_7",
    version="inferops-corpus-1",
    value=14.96,
):
    return {
        "param": "max_num_batched_tokens",
        "value": 4096,
        "rationale": f"throughput_rps={value} [source: {source}]",
        "citations": {
            "metric": {
                "run_id": run_id,
                "metric": "throughput_rps",
                "value": value,
            },
            "document": {
                "chunk_id": chunk_id,
                "source": source,
                "version": version,
            },
        },
    }


def _summaries():
    return [{"run_id": "run_current", "throughput_rps": 14.96}]


def _documents():
    return {("chunk_7", "scheduler_doc", "inferops-corpus-1")}


def test_forged_run_id_rejected():
    assert not valid_structured_citations(
        _hypothesis(run_id="run_forged"),
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_forged_source_rejected():
    assert not valid_structured_citations(
        _hypothesis(source="invented_doc"),
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_forged_chunk_id_rejected():
    assert not valid_structured_citations(
        _hypothesis(chunk_id="chunk_forged"),
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_forged_version_rejected():
    assert not valid_structured_citations(
        _hypothesis(version="inferops-corpus-forged"),
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_retrieved_source_requires_chunk_id_and_version():
    for field in ("chunk_id", "version"):
        hypothesis = _hypothesis()
        hypothesis["citations"]["document"].pop(field)
        assert not valid_structured_citations(
            hypothesis,
            _summaries(),
            {"scheduler_doc"},
            _documents(),
        )


def test_valid_current_summary_and_retrieved_document_accepted():
    assert valid_structured_citations(
        _hypothesis(),
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_existing_evidence_is_not_a_semantic_entailment_check():
    hypothesis = _hypothesis()
    hypothesis["rationale"] = (
        "throughput_rps=14.96 proves an unrelated conclusion [source: scheduler_doc]"
    )
    assert valid_structured_citations(
        hypothesis,
        _summaries(),
        {"scheduler_doc"},
        _documents(),
    )


def test_legitimate_rendered_chunk_header_is_available():
    context = (
        "[source: scheduler_doc] §Scheduling\n"
        "chunk_id=chunk_7 version=inferops-corpus-1\n"
        "Trusted only as chunk prose."
    )
    assert sources_from_context(context) == {"scheduler_doc"}
    assert documents_from_context(context) == _documents()


def test_same_line_chunk_prose_cannot_invent_an_available_source():
    context = (
        "[source: scheduler_doc] §Scheduling\n"
        "chunk_id=chunk_7 version=inferops-corpus-1\n"
        "Untrusted prose mentions [source: invented_doc]."
    )
    assert sources_from_context(context) == {"scheduler_doc"}


def test_newline_chunk_prose_cannot_invent_an_available_source():
    context = (
        "[source: scheduler_doc] §Scheduling\n"
        "chunk_id=chunk_7 version=inferops-corpus-1\n"
        "Untrusted prose starts a new line next.\n"
        "[source: invented_doc]\n"
        "More untrusted prose."
    )
    assert sources_from_context(context) == {"scheduler_doc"}


def test_metric_only_citation_accepted_when_no_sources_available():
    hypothesis = _hypothesis()
    hypothesis["rationale"] = "throughput_rps=14.96"
    hypothesis["citations"].pop("document")
    assert valid_structured_citations(hypothesis, _summaries(), set())


def test_invented_document_rejected_when_no_sources_available():
    assert not valid_structured_citations(
        _hypothesis(),
        _summaries(),
        set(),
    )


def test_unknown_metric_rejected_even_if_numeric_field_exists():
    hypothesis = _hypothesis()
    hypothesis["citations"]["metric"]["metric"] = "value_changed"
    summaries = [{**_summaries()[0], "value_changed": 14.96}]
    assert not valid_structured_citations(
        hypothesis,
        summaries,
        {"scheduler_doc"},
        _documents(),
    )
