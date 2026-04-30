"""Outcome, efficiency, and composite metrics for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary metric per workload → (field_name, "max"|"min")
WORKLOAD_PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "chat_short":                 ("throughput_rps",    "max"),
    "long_context_qa":            ("throughput_rps",    "max"),
    "high_concurrency_short_out": ("throughput_rps",    "max"),
    "long_generation":            ("tokens_per_second", "max"),
    "mixed_traffic":              ("throughput_rps",    "max"),
}

# Secondary metrics always tracked: (field_name, direction, display_label)
SECONDARY_METRICS: list[tuple[str, str, str]] = [
    ("ttft_p99_ms", "min", "TTFT p99"),
    ("e2e_p50_ms",  "min", "E2E p50"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OutcomeMetrics:
    workload_name: str
    primary_metric: str
    ground_truth_value: float
    agent_value: float
    gap_pct: float                            # >0 = agent worse; <0 = agent better than GT
    secondary: dict[str, float] = field(default_factory=dict)  # metric → gap_pct


@dataclass
class EfficiencyMetrics:
    n_experiments: int
    wall_clock_s: float
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0


@dataclass
class WorkloadScore:
    workload_name: str
    outcome: OutcomeMetrics
    efficiency: EfficiencyMetrics
    trajectory_score: float | None = None    # 0–1 from LLM judge
    composite: float = 0.0                   # 0–1, higher is better


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_outcome(
    ground_truth: dict[str, Any],
    agent_result: dict[str, Any],
) -> OutcomeMetrics:
    """
    Compute the gap between the agent's best result and the ground-truth optimum.

    Both ground_truth (loaded from JSON) and agent_result are flat dicts with keys
    like throughput_rps, ttft_p99_ms, tokens_per_second, etc.
    """
    workload_name = ground_truth["workload_name"]
    metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]

    gt_val = float(ground_truth["best_value"])
    agent_val = float(agent_result.get(metric, 0.0))

    # Positive gap = agent is worse than ground truth
    if direction == "max":
        gap_pct = (gt_val - agent_val) / gt_val * 100 if gt_val != 0 else 0.0
    else:
        gap_pct = (agent_val - gt_val) / gt_val * 100 if gt_val != 0 else 0.0

    # Secondary metrics: find ground truth values from the best experiment row
    secondary: dict[str, float] = {}
    best_row = next(
        (e for e in ground_truth.get("experiments", [])
         if e["experiment_id"] == ground_truth["best_experiment_id"]),
        None,
    )
    if best_row:
        for sec_metric, sec_dir, _ in SECONDARY_METRICS:
            gt_sec = best_row.get(sec_metric)
            agent_sec = agent_result.get(sec_metric)
            if gt_sec and agent_sec:
                if sec_dir == "min":
                    secondary[sec_metric] = (float(agent_sec) - float(gt_sec)) / float(gt_sec) * 100
                else:
                    secondary[sec_metric] = (float(gt_sec) - float(agent_sec)) / float(gt_sec) * 100

    return OutcomeMetrics(
        workload_name=workload_name,
        primary_metric=metric,
        ground_truth_value=gt_val,
        agent_value=agent_val,
        gap_pct=round(gap_pct, 2),
        secondary={k: round(v, 2) for k, v in secondary.items()},
    )


def compute_efficiency(
    n_experiments: int,
    wall_clock_s: float,
    llm_tokens_in: int = 0,
    llm_tokens_out: int = 0,
) -> EfficiencyMetrics:
    return EfficiencyMetrics(
        n_experiments=n_experiments,
        wall_clock_s=wall_clock_s,
        llm_tokens_in=llm_tokens_in,
        llm_tokens_out=llm_tokens_out,
    )


def composite_score(
    outcome: OutcomeMetrics,
    efficiency: EfficiencyMetrics,
    budget_experiments: int = 12,
    trajectory_score: float | None = None,
) -> float:
    """
    Composite 0–1 score (higher is better).

    Weights:
      50%  quality   — how close to ground truth (1 − gap_pct/100, floored at 0)
      30%  efficiency — fraction of budget NOT used  (1 − n_experiments/budget)
      20%  trajectory — LLM judge score, or quality if unavailable
    """
    quality = max(0.0, min(1.0, 1.0 - outcome.gap_pct / 100.0))
    eff_score = max(0.0, 1.0 - efficiency.n_experiments / max(budget_experiments, 1))
    traj = trajectory_score if trajectory_score is not None else quality

    return round(0.50 * quality + 0.30 * eff_score + 0.20 * traj, 4)


def aggregate_scores(scores: list[WorkloadScore]) -> dict[str, float]:
    """Return mean metrics across all workload scores."""
    if not scores:
        return {}
    n = len(scores)
    return {
        "mean_gap_pct":        round(sum(s.outcome.gap_pct for s in scores) / n, 2),
        "mean_n_experiments":  round(sum(s.efficiency.n_experiments for s in scores) / n, 1),
        "mean_wall_clock_min": round(sum(s.efficiency.wall_clock_s for s in scores) / n / 60, 1),
        "mean_composite":      round(sum(s.composite for s in scores) / n, 4),
    }
