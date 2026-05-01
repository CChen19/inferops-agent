"""Unit tests for simulated baseline agents."""

from __future__ import annotations

import json
from pathlib import Path

from inferops.eval.baselines import run_greedy_agent, run_random_agent


def _gt(name: str = "chat_short") -> dict:
    return json.loads((Path("tests/fixtures/ground_truth") / f"{name}.json").read_text())


def test_random_agent_is_seeded_and_budget_limited():
    run_a = run_random_agent(_gt(), budget=2, seed=7)
    run_b = run_random_agent(_gt(), budget=2, seed=7)

    assert run_a.n_experiments == 2
    assert run_a.best_result == run_b.best_result
    assert [s["experiment_id"] for s in run_a.trajectory] == [
        s["experiment_id"] for s in run_b.trajectory
    ]


def test_greedy_agent_starts_from_default_and_finds_neighbor_improvement():
    gt = _gt()
    metric = gt["primary_metric"]
    exps = {e["experiment_id"]: e for e in gt["experiments"]}
    # Default: lowest max_num_batched_tokens, both flags off
    min_batch = min(e["max_num_batched_tokens"] for e in gt["experiments"])
    default_exp = next(
        e for e in gt["experiments"]
        if e["max_num_batched_tokens"] == min_batch
        and not e["enable_chunked_prefill"]
        and not e["enable_prefix_caching"]
    )

    run = run_greedy_agent(gt, budget=2)

    assert run.n_experiments == 2
    assert run.trajectory[0]["experiment_id"] == default_exp["experiment_id"]
    # Best must be at least as good as the starting point
    assert run.best_result[metric] >= default_exp[metric]
    assert run.best_result["experiment_id"] in exps


def test_greedy_agent_handles_zero_budget():
    run = run_greedy_agent(_gt(), budget=0)

    assert run.n_experiments == 0
    assert run.best_result == {}
    assert run.trajectory == []
