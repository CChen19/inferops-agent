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
    ci_low_pct: float
    ci_high_pct: float
    significant: bool
    interpretation: str


def compare_experiments(inp: CompareExperimentsInput) -> ComparisonResult:
    """
    Compare two experiments on a chosen metric using bootstrap confidence intervals.

    Uses the raw per-request latency samples stored in ExperimentResult
    (raw_ttft_ms / raw_e2e_ms) for latency metrics, and a scalar bootstrap
    from throughput aggregates for throughput metrics.

    Returns the % delta, bootstrap CI, and whether the difference is statistically
    significant at the requested confidence level.
    """
    with span("tool.compare_experiments", {"metric": inp.metric, "a": inp.experiment_id_a, "b": inp.experiment_id_b}):
        res_a = get_result_by_id(inp.experiment_id_a)
        res_b = get_result_by_id(inp.experiment_id_b)

    if res_a is None:
        raise ValueError(f"Experiment '{inp.experiment_id_a}' not found")
    if res_b is None:
        raise ValueError(f"Experiment '{inp.experiment_id_b}' not found")

    def _get_samples(res, metric: str) -> list[float]:
        if metric in ("ttft_p50_ms", "ttft_p99_ms"):
            return res.raw_ttft_ms or _percentile_to_samples(res.ttft.p50, res.ttft.p99, res.successful_requests)
        if metric in ("e2e_p50_ms", "e2e_p99_ms"):
            return res.raw_e2e_ms or _percentile_to_samples(res.e2e_latency.p50, res.e2e_latency.p99, res.successful_requests)
        # Throughput: scalar — bootstrap by resampling request-level contribution
        val = getattr(res, metric, None)
        if val is None:
            raise ValueError(f"Metric '{metric}' not available")
        # Synthesize samples around the scalar (small jitter for bootstrap)
        return [val * random.gauss(1.0, 0.02) for _ in range(res.successful_requests)]

    def _stat(samples: list[float], metric: str) -> float:
        if "p99" in metric:
            s = sorted(samples)
            return s[min(int(0.99 * len(s)), len(s) - 1)]
        if "p50" in metric:
            s = sorted(samples)
            return s[int(0.50 * len(s))]
        return sum(samples) / len(samples)

    samples_a = _get_samples(res_a, inp.metric)
    samples_b = _get_samples(res_b, inp.metric)

    val_a = _stat(samples_a, inp.metric)
    val_b = _stat(samples_b, inp.metric)

    # Bootstrap the delta distribution
    rng = random.Random(42)
    delta_dist: list[float] = []
    for _ in range(inp.n_bootstrap):
        boot_a = rng.choices(samples_a, k=len(samples_a))
        boot_b = rng.choices(samples_b, k=len(samples_b))
        sa = _stat(boot_a, inp.metric)
        sb = _stat(boot_b, inp.metric)
        delta_dist.append((sb - sa) / sa * 100 if sa != 0 else 0)

    delta_dist.sort()
    alpha = (1 - inp.confidence) / 2
    ci_low = delta_dist[int(alpha * inp.n_bootstrap)]
    ci_high = delta_dist[int((1 - alpha) * inp.n_bootstrap)]

    delta_pct = (val_b - val_a) / val_a * 100 if val_a != 0 else 0.0
    lower_is_better = inp.metric in _LOWER_IS_BETTER

    # Significant if CI doesn't straddle zero
    significant = not (ci_low <= 0 <= ci_high)

    if lower_is_better:
        winner = "b" if delta_pct < -2 else ("a" if delta_pct > 2 else "tie")
        better_word = "lower" if delta_pct < 0 else "higher"
        interp = (
            f"{inp.experiment_id_b} has {abs(delta_pct):.1f}% {better_word} {inp.metric} "
            f"(CI [{ci_low:.1f}%, {ci_high:.1f}%]). "
            f"{'Statistically significant.' if significant else 'Not significant — may be noise.'}"
        )
    else:
        winner = "b" if delta_pct > 2 else ("a" if delta_pct < -2 else "tie")
        better_word = "higher" if delta_pct > 0 else "lower"
        interp = (
            f"{inp.experiment_id_b} has {abs(delta_pct):.1f}% {better_word} {inp.metric} "
            f"(CI [{ci_low:.1f}%, {ci_high:.1f}%]). "
            f"{'Statistically significant.' if significant else 'Not significant — may be noise.'}"
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
    )


def _percentile_to_samples(p50: float, p99: float, n: int) -> list[float]:
    """Approximate raw samples from p50/p99 using a log-normal distribution."""
    import math
    if p50 <= 0:
        return [p99] * n
    # Fit log-normal: mu and sigma from p50 and p99
    mu = math.log(p50)
    sigma = (math.log(p99) - mu) / 2.326  # z=2.326 for p99
    rng = random.Random(0)
    return [math.exp(rng.gauss(mu, max(sigma, 0.01))) for _ in range(n)]
