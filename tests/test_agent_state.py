"""Unit tests for agent state helpers."""

from __future__ import annotations

from inferops.agent.state import (
    initial_state,
    is_duplicate,
    pending_hypotheses,
    summary_from_result,
)


def test_initial_state_sets_budget_and_defaults():
    state = initial_state("chat_short", "sess_", max_experiments=4)

    assert state["workload_name"] == "chat_short"
    assert state["session_prefix"] == "sess_"
    assert state["experiments_remaining"] == 4
    assert state["current_bottleneck"] == "unknown"
    assert state["hypotheses"] == []
    assert state["messages"] == []


def test_pending_hypotheses_filters_only_pending():
    state = initial_state("chat_short", "sess_")
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_seqs",
            "value": 64,
            "rationale": "rps=1",
            "status": "pending",
            "experiment_id": None,
        },
        {
            "id": "h2",
            "param": "max_num_seqs",
            "value": 128,
            "rationale": "rps=1",
            "status": "failed",
            "experiment_id": "e2",
        },
    ]

    assert [h["id"] for h in pending_hypotheses(state)] == ["h1"]


def test_is_duplicate_compares_param_value_pairs():
    state = initial_state("chat_short", "sess_")
    state["experiment_summaries"] = [
        {
            "experiment_id": "e1",
            "param_changed": "max_num_batched_tokens",
            "value_changed": 4096,
            "throughput_rps": 1.0,
            "tokens_per_second": 10.0,
            "ttft_p50_ms": 1.0,
            "ttft_p99_ms": 2.0,
            "e2e_p50_ms": 3.0,
            "bottleneck": "compute-bound",
            "vs_baseline_pct": 0.0,
        }
    ]

    assert is_duplicate(state, "max_num_batched_tokens", "4096") is True
    assert is_duplicate(state, "max_num_batched_tokens", 2048) is False


def test_summary_from_result_uses_primary_metric(result_b):
    summary = summary_from_result(
        result_b,
        param_changed="max_num_batched_tokens",
        value_changed=4096,
        baseline_primary=2.0,
        primary_metric="throughput_rps",
    )

    assert summary["experiment_id"] == result_b.experiment_id
    assert summary["param_changed"] == "max_num_batched_tokens"
    assert summary["value_changed"] == 4096
    assert summary["throughput_rps"] == 2.38
    assert summary["vs_baseline_pct"] == 19.0
