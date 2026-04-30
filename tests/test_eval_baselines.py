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
    run = run_greedy_agent(_gt(), budget=2)

    assert run.n_experiments == 2
    assert run.trajectory[0]["experiment_id"] == "grid_chat_short_t2048_c0_p0"
    assert run.best_result["experiment_id"] == "grid_chat_short_t4096_c0_p0"
    assert run.best_result["throughput_rps"] == 17.2


def test_greedy_agent_handles_zero_budget():
    run = run_greedy_agent(_gt(), budget=0)

    assert run.n_experiments == 0
    assert run.best_result == {}
    assert run.trajectory == []
