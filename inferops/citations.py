"""Validation helpers for planner evidence citations.

This gate checks only that cited evidence exists in the planner's current
inputs. It deliberately does not judge whether that evidence supports the
hypothesis prose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]\r\n]+?)\s*\]", re.IGNORECASE)
_CONTEXT_SOURCE_RE = re.compile(
    r"^\[source:\s*([^\]\r\n]+?)\s*\][ \t]+§[^\r\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_CITABLE_METRICS = frozenset(
    {
        "throughput_rps",
        "tokens_per_second",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "e2e_p50_ms",
        "vs_baseline_pct",
        "error_rate",
    }
)


def source_tags(text: str) -> set[str]:
    """Return the exact source names cited by ``[source: ...]`` tags."""
    return {match.group(1).strip() for match in _SOURCE_TAG_RE.finditer(text)}


def sources_from_context(context: str) -> set[str]:
    """Return source names from rendered chunk headers, not chunk prose."""
    return {match.group(1).strip() for match in _CONTEXT_SOURCE_RE.finditer(context)}


def _same_numeric_value(actual: Any, cited: Any) -> bool:
    if (
        isinstance(actual, bool)
        or isinstance(cited, bool)
        or not isinstance(actual, Real)
        or not isinstance(cited, Real)
    ):
        return False
    return math.isclose(float(actual), float(cited), rel_tol=1e-12, abs_tol=1e-12)


def valid_structured_citations(
    hypothesis: Mapping[str, Any],
    summaries: Iterable[Mapping[str, Any]],
    available_sources: set[str],
) -> bool:
    """Check metric and document citation existence against current inputs."""
    citations = hypothesis.get("citations")
    if not isinstance(citations, Mapping):
        return False

    metric_citation = citations.get("metric")
    document_citation = citations.get("document", {})
    if not isinstance(metric_citation, Mapping) or not isinstance(document_citation, Mapping):
        return False

    run_id = metric_citation.get("run_id")
    metric = metric_citation.get("metric")
    value = metric_citation.get("value")
    if not isinstance(run_id, str) or not run_id:
        return False
    if not isinstance(metric, str) or metric not in _CITABLE_METRICS:
        return False

    metric_exists = any(
        summary.get("run_id") == run_id
        and metric in summary
        and _same_numeric_value(summary.get(metric), value)
        for summary in summaries
    )
    if not metric_exists:
        return False

    rationale = hypothesis.get("rationale")
    if not isinstance(rationale, str):
        return False
    cited_source_tags = source_tags(rationale)
    source = document_citation.get("source")
    if not available_sources:
        return source is None and not cited_source_tags
    if not isinstance(source, str) or source not in available_sources:
        return False
    if source not in cited_source_tags:
        return False
    return True
