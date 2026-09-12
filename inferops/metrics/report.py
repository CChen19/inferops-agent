"""Consistent report / recalculation entrypoint for summaries (P0-④).

Reports MUST call `report_from_ledger` or `report_from_result` rather than
re-deriving aggregates ad hoc. Missing metrics stay `n/a` — never 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferops.metrics.aggregate import (
    AggregateMetrics,
    format_aggregate_report,
    recalculate_from_ledger,
)
from inferops.metrics.ledger import RequestLedger, load_ledger


def ledger_from_result(result: Any) -> RequestLedger | None:
    """Rebuild a RequestLedger from an ExperimentResult if rows were stored."""
    payload = getattr(result, "request_ledger", None)
    run_id = getattr(result, "run_id", "") or ""
    path = getattr(result, "ledger_path", None)
    if isinstance(payload, dict) and payload.get("records"):
        return RequestLedger.model_validate(payload)
    if isinstance(payload, list) and payload and run_id:
        # Legacy: records-only list (window unknown → derive from timestamps).
        return RequestLedger.model_validate(
            {
                "run_id": run_id,
                "records": payload,
                "window_start_s": None,
                "window_end_s": None,
            }
        )
    if path and Path(path).is_file():
        return load_ledger(path)
    return None


def report_from_ledger(
    ledger: RequestLedger,
    *,
    gpu_utilization_pct: float | None = None,
    gpu_memory_used_gb: float | None = None,
    cost_usd: float | None = None,
) -> tuple[AggregateMetrics, str]:
    """Single entrypoint: recalculate + Markdown. Used by reports / CLI."""
    agg = recalculate_from_ledger(
        ledger,
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_memory_used_gb=gpu_memory_used_gb,
        cost_usd=cost_usd,
    )
    return agg, format_aggregate_report(agg)


def report_from_result(result: Any) -> tuple[AggregateMetrics | None, str]:
    """Recalculate from a persisted ExperimentResult's ledger, or explain why not."""
    ledger = ledger_from_result(result)
    if ledger is None:
        return None, (
            f"No request ledger for run_id=`{getattr(result, 'run_id', '')}`. "
            "Aggregates cannot be independently recalculated."
        )
    gpu_u = getattr(result, "gpu_utilization_pct", None)
    gpu_m = getattr(result, "gpu_memory_used_gb", None)
    cost = getattr(result, "cost_usd", None)
    return report_from_ledger(
        ledger,
        gpu_utilization_pct=gpu_u,
        gpu_memory_used_gb=gpu_m,
        cost_usd=cost,
    )
