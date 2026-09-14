"""Tests for the fair eval protocol (Stage B slice 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.eval.protocol import (
    BudgetPolicy,
    HiddenResultFixture,
    Observation,
    RunScore,
    SLOPolicy,
    TrialLedger,
    row_to_observation,
    score_run,
)
from inferops.eval.strategies import (
    run_all_strategies,
    run_default_strategy,
    run_online_local_search,
    run_random_strategy,
)


def _gt(name: str = "chat_short") -> dict:
    return json.loads((Path("tests/fixtures/ground_truth") / f"{name}.json").read_text())


def _fixture_with_contract_fields(rows: list[dict]) -> HiddenResultFixture:
    enriched = [
        {
            **row,
            "validity_status": "valid",
            "error_rate": 0.01,
            "has_config_evidence": True,
            "bottleneck": "compute-bound",
        }
        for row in rows
    ]
    return HiddenResultFixture.from_rows(enriched)


def test_online_local_search_does_not_pick_neighbor_by_unread_gt_score():
    """Fair local search picks by stable order, not clairvoyant best-metric neighbor."""
    gt = _gt()
    fixture = _fixture_with_contract_fields(gt["experiments"])
    budget = BudgetPolicy(total_slots=2)

    run = run_online_local_search(
        fixture,
        budget,
        workload_name=gt["workload_name"],
    )

    assert run.ledger.records[0].kind == "baseline"
    assert run.ledger.records[0].config["max_num_batched_tokens"] == 2048

    second = run.ledger.records[1]
    # Clairvoyant greedy would jump to t4096 (throughput 17.2); fair search tries
    # the first sorted one-knob neighbor: chunked prefill on (throughput 16.0).
    assert second.config["enable_chunked_prefill"] is True
    assert second.config["max_num_batched_tokens"] == 2048
    assert second.config["max_num_batched_tokens"] != 4096


def test_all_strategies_share_budget_policy_charging():
    gt = _gt()
    fixture = _fixture_with_contract_fields(gt["experiments"])
    total_slots = 3

    default = run_default_strategy(
        fixture, BudgetPolicy(total_slots=total_slots), workload_name=gt["workload_name"]
    )
    random_run = run_random_strategy(
        fixture,
        BudgetPolicy(total_slots=total_slots),
        workload_name=gt["workload_name"],
        seed=1,
    )
    local = run_online_local_search(
        fixture,
        BudgetPolicy(total_slots=total_slots),
        workload_name=gt["workload_name"],
    )

    assert default.score["n_paid"] == 1
    assert random_run.score["n_paid"] == total_slots
    assert local.score["n_paid"] == total_slots

    dup_budget = BudgetPolicy(total_slots=2)
    dup_budget.charge_baseline()
    assert dup_budget.charge_trial(duplicate=True) is True
    assert dup_budget.n_paid == 1

    confirm_budget = BudgetPolicy(total_slots=3)
    confirm_budget.charge_baseline()
    assert confirm_budget.charge_trial(confirmation=True) is True
    assert confirm_budget.n_paid == 3
    assert confirm_budget.confirmation_charged is True


def test_missing_error_rate_fails_slo():
    obs = Observation(
        metrics={"throughput_rps": 10.0},
        validity_status="valid",
        error_rate=None,
        config_evidence=True,
        bottleneck="unknown",
    )
    result = SLOPolicy.check(obs)
    assert result["ok"] is False
    assert result["reason"] == "error_rate_missing_fail_closed"


def test_score_run_reports_decomposed_fields_not_composite_only():
    gt = _gt()
    rows = [
        {
            **gt["experiments"][0],
            "validity_status": "valid",
            "error_rate": 0.01,
            "has_config_evidence": True,
            "bottleneck": "compute-bound",
        },
        {
            **gt["experiments"][1],
            "validity_status": "valid",
            "error_rate": 0.02,
            "has_config_evidence": True,
            "bottleneck": "compute-bound",
        },
    ]
    fixture = HiddenResultFixture.from_rows(rows)
    budget = BudgetPolicy(total_slots=2)
    run = run_online_local_search(fixture, budget, workload_name=gt["workload_name"])

    scored = score_run(
        run.ledger,
        workload_name=gt["workload_name"],
        gt_optimum=gt,
    )
    assert isinstance(scored, RunScore)
    assert scored.first_valid_n == 1
    assert scored.wasted_trials >= 0
    assert scored.n_paid == 2
    assert scored.gap_pct is not None
    assert "composite" not in scored.__dict__
    assert set(run.score.keys()) == {
        "success_in_budget",
        "first_valid_n",
        "confirmed_gain",
        "wasted_trials",
        "n_paid",
        "gap_pct",
    }


def test_hidden_fixture_reveals_metrics_only_via_observe():
    gt = _gt()
    fixture = HiddenResultFixture.from_ground_truth(gt)
    default = fixture.search_space.default_config()
    obs = fixture.observe(default)
    assert "throughput_rps" in obs.metrics
    assert not hasattr(obs, "gt_score")
    assert "gt_score" not in obs.to_summary()


def test_run_all_strategies_wrapper():
    gt = _gt()
    fixture = _fixture_with_contract_fields(gt["experiments"])
    runs = run_all_strategies(
        fixture,
        BudgetPolicy(total_slots=2),
        workload_name=gt["workload_name"],
        gt_optimum=gt,
    )
    assert set(runs) == {"default", "random", "online_local_search"}
    for name, run in runs.items():
        assert run.strategy_name == name
        assert "first_valid_n" in run.score


def test_row_to_observation_has_no_hidden_gt_field():
    row = {"throughput_rps": 1.0, "validity_status": "valid", "error_rate": 0.0}
    obs = row_to_observation(row)
    summary = obs.to_summary()
    assert summary["throughput_rps"] == 1.0
    assert "gt_score" not in summary
