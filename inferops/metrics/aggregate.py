"""Aggregate metric recalculation from a RequestLedger.

Single entrypoint: `recalculate_from_ledger`. Reports / MLflow / ExperimentResult
must use these numbers (or call this) so aggregates stay consistent.
Missing metrics are None — never defaulted to 0.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.metrics.definitions import (
    E2E_SAMPLE_SCOPE,
    TPOT_SAMPLE_SCOPE,
    TTFT_SAMPLE_SCOPE,
    is_error_for_rate,
    is_latency_eligible,
    is_tpot_eligible,
    is_ttft_eligible,
    measured_records,
    percentiles,
)
from inferops.metrics.ledger import RequestLedger, RequestOutcome


class LatencyStat(BaseModel):
    """Percentiles with explicit sample scope (never invent zeros)."""

    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    sample_n: int = 0
    sample_scope: str = ""

    @classmethod
    def from_samples(cls, samples: list[float], scope: str) -> LatencyStat:
        if not samples:
            return cls(sample_n=0, sample_scope=scope)
        pct = percentiles(samples)
        return cls(
            p50=pct["p50"],
            p90=pct["p90"],
            p95=pct["p95"],
            p99=pct["p99"],
            sample_n=len(samples),
            sample_scope=scope,
        )

    def as_latency_percentiles_dict(self) -> dict[str, float | None]:
        return {
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "sample_n": self.sample_n,
            "sample_scope": self.sample_scope,
        }


class AggregateMetrics(BaseModel):
    """Full recalculable summary for one run_id."""

    run_id: str
    total_requests: int  # measured non-warmup denominator
    successful_requests: int  # SUCCESS + TRUNCATE
    failed_requests: int  # error-rate numerator
    error_rate: float | None  # failed / total; None if total==0

    # Throughput window
    window_start_s: float | None = None
    window_end_s: float | None = None
    total_time_s: float | None = None  # window_end - window_start
    throughput_rps: float | None = None  # successful / total_time_s
    tokens_per_second: float | None = None  # sum(output tokens of successes) / time
    total_output_tokens: int = 0
    total_input_tokens: int | None = None

    ttft: LatencyStat = Field(default_factory=LatencyStat)
    tpot: LatencyStat = Field(default_factory=LatencyStat)
    e2e: LatencyStat = Field(default_factory=LatencyStat)

    # Optional — only when evidence exists (never invent)
    gpu_utilization_pct: float | None = None
    gpu_memory_used_gb: float | None = None
    cost_usd: float | None = None

    outcome_counts: dict[str, int] = Field(default_factory=dict)


def _window_seconds(ledger: RequestLedger) -> tuple[float | None, float | None, float | None]:
    """Resolve throughput time window.

    Prefer explicit ledger window (load wall clock). Else derive from measured
    request timestamps (first t_start → last t_end).
    """
    start = ledger.window_start_s
    end = ledger.window_end_s
    measured = measured_records(ledger.records)
    if start is None and measured:
        start = min(r.t_start_s for r in measured)
    if end is None and measured:
        ends = [r.t_end_s for r in measured if r.t_end_s is not None]
        end = max(ends) if ends else None
    total = None
    if start is not None and end is not None and end >= start:
        total = end - start
    return start, end, total


def recalculate_from_ledger(
    ledger: RequestLedger,
    *,
    gpu_utilization_pct: float | None = None,
    gpu_memory_used_gb: float | None = None,
    cost_usd: float | None = None,
) -> AggregateMetrics:
    """Independent recalculation entrypoint — source of truth for reports."""
    measured = measured_records(ledger.records)
    total = len(measured)

    outcome_counts: dict[str, int] = {}
    for r in measured:
        key = r.outcome.value
        outcome_counts[key] = outcome_counts.get(key, 0) + 1

    successful = [
        r
        for r in measured
        if r.outcome in (RequestOutcome.SUCCESS, RequestOutcome.TRUNCATE)
    ]
    failed_n = sum(1 for r in measured if is_error_for_rate(r))
    error_rate = (failed_n / total) if total > 0 else None

    start, end, total_time = _window_seconds(ledger)

    total_out = sum(r.output_tokens for r in successful)
    input_vals = [r.input_tokens for r in measured if r.input_tokens is not None]
    total_in = sum(input_vals) if input_vals else None

    throughput = None
    tok_s = None
    if total_time is not None and total_time > 0:
        throughput = len(successful) / total_time
        tok_s = total_out / total_time

    ttft_samples = [r.ttft_ms for r in measured if is_ttft_eligible(r) and r.ttft_ms is not None]
    tpot_samples = [r.tpot_ms for r in measured if is_tpot_eligible(r) and r.tpot_ms is not None]
    e2e_samples = [
        r.e2e_ms
        for r in measured
        if is_latency_eligible(r) and r.e2e_ms is not None
    ]

    return AggregateMetrics(
        run_id=ledger.run_id,
        total_requests=total,
        successful_requests=len(successful),
        failed_requests=failed_n,
        error_rate=error_rate,
        window_start_s=start,
        window_end_s=end,
        total_time_s=total_time,
        throughput_rps=throughput,
        tokens_per_second=tok_s,
        total_output_tokens=total_out,
        total_input_tokens=total_in,
        ttft=LatencyStat.from_samples(ttft_samples, TTFT_SAMPLE_SCOPE),
        tpot=LatencyStat.from_samples(tpot_samples, TPOT_SAMPLE_SCOPE),
        e2e=LatencyStat.from_samples(e2e_samples, E2E_SAMPLE_SCOPE),
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_memory_used_gb=gpu_memory_used_gb,
        cost_usd=cost_usd,
        outcome_counts=outcome_counts,
    )


def ledger_to_experiment_fields(agg: AggregateMetrics) -> dict[str, Any]:
    """Map AggregateMetrics → ExperimentResult constructor kwargs."""
    from inferops.schemas import LatencyPercentiles

    def _lp(stat: LatencyStat) -> LatencyPercentiles:
        return LatencyPercentiles(
            p50=stat.p50,
            p90=stat.p90,
            p95=stat.p95,
            p99=stat.p99,
            sample_n=stat.sample_n,
            sample_scope=stat.sample_scope,
        )

    return {
        "total_requests": agg.total_requests,
        "successful_requests": agg.successful_requests,
        "total_time_s": agg.total_time_s if agg.total_time_s is not None else 0.0,
        "throughput_rps": agg.throughput_rps,
        "tokens_per_second": agg.tokens_per_second,
        "error_rate": agg.error_rate,
        "ttft": _lp(agg.ttft),
        "tpot": _lp(agg.tpot),
        "e2e_latency": _lp(agg.e2e),
        "raw_ttft_ms": [],  # filled by caller from ledger if needed
        "raw_e2e_ms": [],
        "gpu_utilization_pct": agg.gpu_utilization_pct,
        "gpu_memory_used_gb": agg.gpu_memory_used_gb,
    }


def apply_aggregate_to_result_fields(agg: AggregateMetrics) -> dict[str, Any]:
    """Alias kept as a stable name for item ⑤."""
    return ledger_to_experiment_fields(agg)


def format_aggregate_report(agg: AggregateMetrics) -> str:
    """Markdown snippet: aggregates + sample scopes (no invented numbers)."""
    def fmt(v: float | None, digits: int = 3) -> str:
        if v is None:
            return "n/a"
        return f"{v:.{digits}f}"

    def fmt_lat(stat: LatencyStat) -> str:
        if stat.sample_n == 0:
            return f"n/a (n=0, scope=`{stat.sample_scope}`)"
        return (
            f"p50={fmt(stat.p50, 1)} p90={fmt(stat.p90, 1)} "
            f"p95={fmt(stat.p95, 1)} p99={fmt(stat.p99, 1)} "
            f"(n={stat.sample_n}, scope=`{stat.sample_scope}`)"
        )

    lines = [
        f"### Metrics for `run_id={agg.run_id}`",
        "",
        f"- **total_requests** (error-rate denominator): {agg.total_requests}",
        f"- **successful_requests** (SUCCESS+TRUNCATE): {agg.successful_requests}",
        f"- **failed_requests** (error numerator): {agg.failed_requests}",
        f"- **error_rate**: {fmt(agg.error_rate, 4)}",
        f"- **throughput window (s)**: {fmt(agg.total_time_s, 4)}",
        f"- **throughput_rps** (successful / window): {fmt(agg.throughput_rps)}",
        f"- **tokens_per_second** (success output tokens / window): {fmt(agg.tokens_per_second)}",
        f"- **TTFT (ms)**: {fmt_lat(agg.ttft)}",
        f"- **TPOT (ms)**: {fmt_lat(agg.tpot)}",
        f"- **E2E (ms)**: {fmt_lat(agg.e2e)}",
        f"- **outcome_counts**: `{agg.outcome_counts}`",
    ]
    if agg.gpu_utilization_pct is not None:
        lines.append(f"- **gpu_utilization_pct** (sampled): {fmt(agg.gpu_utilization_pct, 1)}")
    else:
        lines.append("- **gpu_utilization_pct**: n/a (not sampled)")
    if agg.gpu_memory_used_gb is not None:
        lines.append(f"- **gpu_memory_used_gb** (sampled): {fmt(agg.gpu_memory_used_gb, 2)}")
    else:
        lines.append("- **gpu_memory_used_gb**: n/a (not sampled)")
    if agg.cost_usd is not None:
        lines.append(f"- **cost_usd**: {fmt(agg.cost_usd, 4)}")
    else:
        lines.append("- **cost_usd**: n/a (no pricing assumptions)")
    lines.append("")
    return "\n".join(lines)
