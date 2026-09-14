"""Tool: compare_experiments — bootstrap confidence interval comparison of two runs."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

from inferops.memory.db import get_result_by_id
from inferops.observability import span

MetricName = Literal[
    "throughput_rps", "tokens_per_second",
    "ttft_p50_ms", "ttft_p99_ms",
    "e2e_p50_ms", "e2e_p99_ms",
]

_LOWER_IS_BETTER = {"ttft_p50_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p99_ms"}
_THROUGHPUT_METRICS = {"throughput_rps", "tokens_per_second"}

_CI_UNAVAILABLE_THROUGHPUT = (
    "throughput metrics are single-run aggregates; per-request samples are not "
    "available for bootstrap CI"
)
_CI_UNAVAILABLE_NO_RAW_TTFT = (
    "raw_ttft_ms missing; refusing to synthesize samples from percentiles"
)
_CI_UNAVAILABLE_NO_RAW_E2E = (
    "raw_e2e_ms missing; refusing to synthesize samples from percentiles"
)
_WITHIN_RUN_CI_NOTE = (
    "CI reflects uncertainty of this run's request samples, not repeated-run CI."
)


class CompareExperimentsInput(BaseModel):
    experiment_id_a: str = Field(description="Baseline experiment ID.")
    experiment_id_b: str = Field(description="Candidate experiment ID to compare against baseline.")
    metric: MetricName = Field(
        default="throughput_rps",
        description="Metric to compare. Use throughput_rps / tokens_per_second for capacity, "
                    "ttft_*/e2e_* for latency.",
    )
    n_bootstrap: int = Field(default=2000, ge=200, le=10000, description="Bootstrap resampling iterations.")
    confidence: float = Field(default=0.95, ge=0.80, le=0.99, description="Confidence level for the interval.")


class ComparisonResult(BaseModel):
    experiment_id_a: str
    experiment_id_b: str
    metric: str
    value_a: float
    value_b: float
    delta_pct: float
    winner: Literal["a", "b", "tie"]
    ci_low_pct: float | None = None
    ci_high_pct: float | None = None
    significant: bool
    interpretation: str
    ci_unavailable_reason: str | None = None


def compare_experiments(inp: CompareExperimentsInput) -> ComparisonResult:
    """
    Compare two experiments on a chosen metric.

    Latency metrics with stored raw per-request samples (raw_ttft_ms / raw_e2e_ms)
    use bootstrap CIs over those samples. Throughput aggregates and latency without
    raw samples report point deltas only — CI is explicitly unavailable (never
    synthesized).
    """
    with span("tool.compare_experiments", {"metric": inp.metric, "a": inp.experiment_id_a, "b": inp.experiment_id_b}):
        res_a = get_result_by_id(inp.experiment_id_a)
        res_b = get_result_by_id(inp.experiment_id_b)

    if res_a is None:
        raise ValueError(f"Experiment '{inp.experiment_id_a}' not found")
    if res_b is None:
        raise ValueError(f"Experiment '{inp.experiment_id_b}' not found")

    def _status_note(res) -> str:
        status = getattr(res, "status", None)
        value = status.value if hasattr(status, "value") else (status or "insufficient_evidence")
        return value

    def _point_value(res, metric: str) -> float:
        if metric == "ttft_p50_ms":
            val = res.ttft.p50
        elif metric == "ttft_p99_ms":
            val = res.ttft.p99
        elif metric == "e2e_p50_ms":
            val = res.e2e_latency.p50
        elif metric == "e2e_p99_ms":
            val = res.e2e_latency.p99
        else:
            val = getattr(res, metric, None)
        if val is None:
            raise ValueError(f"Metric '{metric}' not available (missing / n/a)")
        return float(val)

    def _raw_latency_samples(res, metric: str) -> tuple[list[float] | None, str | None]:
        if metric in ("ttft_p50_ms", "ttft_p99_ms"):
            if res.raw_ttft_ms:
                return list(res.raw_ttft_ms), None
            return None, _CI_UNAVAILABLE_NO_RAW_TTFT
        if metric in ("e2e_p50_ms", "e2e_p99_ms"):
            if res.raw_e2e_ms:
                return list(res.raw_e2e_ms), None
            return None, _CI_UNAVAILABLE_NO_RAW_E2E
        return None, _CI_UNAVAILABLE_THROUGHPUT

    def _stat(samples: list[float], metric: str) -> float:
        if "p99" in metric:
            s = sorted(samples)
            return s[min(int(0.99 * len(s)), len(s) - 1)]
        if "p50" in metric:
            s = sorted(samples)
            return s[int(0.50 * len(s))]
        return sum(samples) / len(samples)

    def _delta_pct(sb: float, sa: float) -> float:
        if sa == 0:
            raise ValueError(
                f"Cannot compare '{inp.metric}': baseline denominator is 0 — "
                "refusing silent 0% gain"
            )
        return (sb - sa) / sa * 100

    status_a = _status_note(res_a)
    status_b = _status_note(res_b)
    validity_note = (
        f" Validity: a=`{status_a}` (run_id={getattr(res_a, 'run_id', '')}), "
        f"b=`{status_b}` (run_id={getattr(res_b, 'run_id', '')}). "
        "Metric deltas do not imply config was applied."
    )

    lower_is_better = inp.metric in _LOWER_IS_BETTER

    def _winner(delta: float) -> Literal["a", "b", "tie"]:
        # Point comparison only (±2% heuristic) — not a statistical claim.
        if lower_is_better:
            return "b" if delta < -2 else ("a" if delta > 2 else "tie")
        return "b" if delta > 2 else ("a" if delta < -2 else "tie")

    def _point_phrase(delta: float) -> str:
        if lower_is_better:
            better_word = "lower" if delta < 0 else "higher"
        else:
            better_word = "higher" if delta > 0 else "lower"
        return (
            f"{inp.experiment_id_b} has {abs(delta):.1f}% {better_word} {inp.metric}"
        )

    # Throughput: single aggregates only — never synthesize jitter samples.
    if inp.metric in _THROUGHPUT_METRICS:
        val_a = _point_value(res_a, inp.metric)
        val_b = _point_value(res_b, inp.metric)
        delta_pct = _delta_pct(val_b, val_a)
        winner = _winner(delta_pct)
        reason = _CI_UNAVAILABLE_THROUGHPUT
        interp = (
            f"{_point_phrase(delta_pct)} (point comparison). "
            f"Confidence interval unavailable ({reason}). "
            f"Interval significance cannot be judged."
            f"{validity_note}"
        )
        return ComparisonResult(
            experiment_id_a=inp.experiment_id_a,
            experiment_id_b=inp.experiment_id_b,
            metric=inp.metric,
            value_a=round(val_a, 3),
            value_b=round(val_b, 3),
            delta_pct=round(delta_pct, 2),
            winner=winner,
            ci_low_pct=None,
            ci_high_pct=None,
            significant=False,
            interpretation=interp,
            ci_unavailable_reason=reason,
        )

    samples_a, reason_a = _raw_latency_samples(res_a, inp.metric)
    samples_b, reason_b = _raw_latency_samples(res_b, inp.metric)

    # Latency without raw samples: point values from stored percentiles; no CI.
    if samples_a is None or samples_b is None:
        val_a = _point_value(res_a, inp.metric)
        val_b = _point_value(res_b, inp.metric)
        delta_pct = _delta_pct(val_b, val_a)
        winner = _winner(delta_pct)
        reason = reason_a or reason_b or "raw latency samples unavailable"
        interp = (
            f"{_point_phrase(delta_pct)} (point comparison). "
            f"Confidence interval unavailable ({reason}). "
            f"Interval significance cannot be judged."
            f"{validity_note}"
        )
        return ComparisonResult(
            experiment_id_a=inp.experiment_id_a,
            experiment_id_b=inp.experiment_id_b,
            metric=inp.metric,
            value_a=round(val_a, 3),
            value_b=round(val_b, 3),
            delta_pct=round(delta_pct, 2),
            winner=winner,
            ci_low_pct=None,
            ci_high_pct=None,
            significant=False,
            interpretation=interp,
            ci_unavailable_reason=reason,
        )

    val_a = _stat(samples_a, inp.metric)
    val_b = _stat(samples_b, inp.metric)

    # Bootstrap the delta distribution over real per-request samples.
    rng = random.Random(42)
    delta_dist: list[float] = []
    for _ in range(inp.n_bootstrap):
        boot_a = rng.choices(samples_a, k=len(samples_a))
        boot_b = rng.choices(samples_b, k=len(samples_b))
        sa = _stat(boot_a, inp.metric)
        sb = _stat(boot_b, inp.metric)
        delta_dist.append(_delta_pct(sb, sa))

    delta_dist.sort()
    alpha = (1 - inp.confidence) / 2
    ci_low = delta_dist[int(alpha * inp.n_bootstrap)]
    ci_high = delta_dist[int((1 - alpha) * inp.n_bootstrap)]

    delta_pct = _delta_pct(val_b, val_a)
    winner = _winner(delta_pct)

    # Significant if CI doesn't straddle zero
    significant = not (ci_low <= 0 <= ci_high)

    sig_clause = (
        "Statistically significant."
        if significant
        else "Not significant — may be noise."
    )
    interp = (
        f"{_point_phrase(delta_pct)} "
        f"(CI [{ci_low:.1f}%, {ci_high:.1f}%]). "
        f"{sig_clause} {_WITHIN_RUN_CI_NOTE}"
        f"{validity_note}"
    )

    return ComparisonResult(
        experiment_id_a=inp.experiment_id_a,
        experiment_id_b=inp.experiment_id_b,
        metric=inp.metric,
        value_a=round(val_a, 3),
        value_b=round(val_b, 3),
        delta_pct=round(delta_pct, 2),
        winner=winner,
        ci_low_pct=round(ci_low, 2),
        ci_high_pct=round(ci_high, 2),
        significant=significant,
        interpretation=interp,
        ci_unavailable_reason=None,
    )
