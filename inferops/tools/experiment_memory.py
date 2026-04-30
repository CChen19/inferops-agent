"""Tool: query_experiment_memory — look up past experiment results from SQLite."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.memory.db import query_results
from inferops.observability import span


class QueryMemoryInput(BaseModel):
    workload_name: str | None = Field(
        default=None,
        description="Filter by workload. One of 'chat_short', 'long_context_qa', or null for all.",
    )
    sort_by: str = Field(
        default="throughput_rps",
        description="Metric to rank results by. One of: throughput_rps, ttft_p50_ms, "
                    "ttft_p99_ms, e2e_p50_ms, e2e_p99_ms.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return.")


class MemoryQueryOutput(BaseModel):
    results: list[dict[str, Any]]
    total_found: int
    sort_by: str
    workload_filter: str | None


def query_experiment_memory(inp: QueryMemoryInput) -> MemoryQueryOutput:
    """
    Query the experiment history database for past benchmark results.

    Returns the top-k experiments ranked by the chosen metric. Use this before
    proposing a new config to understand which configurations have already been
    tried and what their outcomes were. Avoids redundant experimentation.
    """
    with span("tool.query_experiment_memory", {"workload": str(inp.workload_name), "sort_by": inp.sort_by}):
        rows = query_results(
            workload_name=inp.workload_name,
            sort_by=inp.sort_by,
            top_k=inp.top_k,
        )

    return MemoryQueryOutput(
        results=rows,
        total_found=len(rows),
        sort_by=inp.sort_by,
        workload_filter=inp.workload_name,
    )
