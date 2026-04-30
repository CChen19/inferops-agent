"""Baseline agent simulators for eval harness comparisons.

These baselines run against ground-truth sweep rows, so they are deterministic,
fast, and safe for CI. Real vLLM runs remain the responsibility of the benchmark
scripts and self-hosted/manual runners.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC

_KNOBS = ("max_num_batched_tokens", "enable_chunked_prefill", "enable_prefix_caching")


@dataclass
class BaselineRun:
    agent_name: str
    workload_name: str
    best_result: dict[str, Any]
    n_experiments: int
    trajectory: list[dict[str, Any]]


def run_random_agent(
    ground_truth: dict[str, Any],
    budget: int = 6,
    seed: int = 42,
) -> BaselineRun:
    """Select configs uniformly at random from the ground-truth sweep rows."""
    rows = list(ground_truth.get("experiments", []))
    rng = random.Random(seed)
    rng.shuffle(rows)
    tried = rows[:max(0, min(budget, len(rows)))]
    metric, direction = WORKLOAD_PRIMARY_METRIC[ground_truth["workload_name"]]
    best = _best_row(tried, metric, direction) if tried else {}

    trajectory = [
        _trajectory_step(i, "random_agent", row, metric, "randomly sampled config")
        for i, row in enumerate(tried, 1)
    ]
    return BaselineRun("random_agent", ground_truth["workload_name"], best, len(tried), trajectory)


def run_greedy_agent(
    ground_truth: dict[str, Any],
    budget: int = 6,
) -> BaselineRun:
    """Greedy local search that only considers one-knob neighbors of the latest config."""
    rows = list(ground_truth.get("experiments", []))
    metric, direction = WORKLOAD_PRIMARY_METRIC[ground_truth["workload_name"]]
    if not rows or budget <= 0:
        return BaselineRun("greedy_agent", ground_truth["workload_name"], {}, 0, [])

    current = _default_row(rows) or rows[0]
    tried = [current]
    trajectory = [
        _trajectory_step(
            1,
            "greedy_agent",
            current,
            metric,
            "start from default config",
        )
    ]

    while len(tried) < budget:
        candidates = [
            row for row in rows
            if row not in tried and _diff_count(row, current) == 1
        ]
        if not candidates:
            candidates = [row for row in rows if row not in tried]
        if not candidates:
            break

        candidate = _best_row(candidates, metric, direction)
        tried.append(candidate)
        improved = _is_better(candidate, current, metric, direction)
        reasoning = (
            f"latest {metric}={current.get(metric, 0):.3f}; "
            f"try best one-knob neighbor {candidate['experiment_id']}"
        )
        trajectory.append(
            _trajectory_step(len(tried), "greedy_agent", candidate, metric, reasoning)
        )
        if improved:
            current = candidate

    best = _best_row(tried, metric, direction)
    return BaselineRun("greedy_agent", ground_truth["workload_name"], best, len(tried), trajectory)


def _best_row(rows: list[dict[str, Any]], metric: str, direction: str) -> dict[str, Any]:
    if direction == "max":
        return max(rows, key=lambda row: float(row.get(metric, 0.0)))
    return min(rows, key=lambda row: float(row.get(metric, float("inf"))))


def _default_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("max_num_batched_tokens") == 2048
            and row.get("enable_chunked_prefill") is False
            and row.get("enable_prefix_caching") is False
        ):
            return row
    return None


def _diff_count(a: dict[str, Any], b: dict[str, Any]) -> int:
    return sum(a.get(k) != b.get(k) for k in _KNOBS)


def _is_better(
    candidate: dict[str, Any],
    current: dict[str, Any],
    metric: str,
    direction: str,
) -> bool:
    cand = float(candidate.get(metric, 0.0))
    cur = float(current.get(metric, 0.0))
    return cand > cur if direction == "max" else cand < cur


def _trajectory_step(
    step: int,
    agent_name: str,
    row: dict[str, Any],
    metric: str,
    reasoning: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "node": agent_name,
        "workload": row.get("workload_name"),
        "action": f"select_config({row.get('experiment_id', 'unknown')})",
        "experiment_id": row.get("experiment_id", ""),
        "reasoning": reasoning,
        "result": {
            metric: row.get(metric, 0.0),
            "ttft_p99_ms": row.get("ttft_p99_ms", 0.0),
            "e2e_p50_ms": row.get("e2e_p50_ms", 0.0),
        },
    }
