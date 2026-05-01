"""Unit tests for graph assembly helpers without running vLLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.graph import _run_baseline, make_llm, prepare_initial_state, run_agent


def test_make_llm_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        make_llm("bogus")


def test_run_baseline_loads_existing_result(result):
    analysis = MagicMock(bottleneck="compute-bound")

    with patch("inferops.agent.graph.get_result_by_id", return_value=result), \
         patch("inferops.agent.graph.analyze_bottleneck", return_value=analysis), \
         patch("inferops.agent.graph.save_result") as mock_save:
        summary, bottleneck = _run_baseline("chat_short", "sess_")

    mock_save.assert_not_called()
    assert summary["experiment_id"] == "sess_baseline"
    assert summary["throughput_rps"] == result.throughput_rps
    assert bottleneck == "compute-bound"


def test_run_agent_initializes_state_and_invokes_graph():
    baseline = {
        "experiment_id": "sess_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 2.0,
        "tokens_per_second": 128.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
    }
    captured = {}

    class FakeGraph:
        def invoke(self, state):
            captured["state"] = state
            state["should_stop"] = True
            state["stop_reason"] = "unit_test"
            return state

    with patch("inferops.agent.graph.init_db"), \
         patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")), \
         patch("inferops.agent.graph.build_graph", return_value=FakeGraph()), \
         patch("inferops.agent.graph._print_run_summary"):
        final_state = run_agent(
            workload_name="chat_short",
            llm=object(),
            max_experiments=5,
            session_prefix="sess_",
        )

    assert captured["state"]["experiments_remaining"] == 4
    assert captured["state"]["baseline_summary"] == baseline
    assert final_state["stop_reason"] == "unit_test"


def test_prepare_initial_state_includes_baseline_and_best():
    baseline = {
        "experiment_id": "sess_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 2.0,
        "tokens_per_second": 128.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
    }

    with patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")):
        state = prepare_initial_state("chat_short", "sess_", max_experiments=5)

    assert state["baseline_summary"] == baseline
    assert state["best_summary"] == baseline
    assert state["experiment_summaries"] == [baseline]
    assert state["tried_experiment_ids"] == ["sess_baseline"]
    assert state["current_bottleneck"] == "compute-bound"
    assert state["experiments_remaining"] == 4
