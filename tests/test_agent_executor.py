"""Unit tests for the executor node with all external tools mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferops.agent.executor import executor_node
from inferops.agent.state import AgentState, initial_state
from inferops.tools.run_benchmark import RunBenchmarkOutput


def _state_with_baseline() -> AgentState:
    state = initial_state("chat_short", "sess_", max_experiments=5)
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
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline]
    state["tried_experiment_ids"] = ["sess_baseline"]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 4
    return state


def test_executor_returns_empty_patch_when_no_pending():
    assert executor_node(_state_with_baseline()) == {}


def test_executor_skips_duplicate_without_spending_budget():
    state = _state_with_baseline()
    state["experiment_summaries"].append({
        "experiment_id": "sess_max_num_seqs_256",
        "param_changed": "max_num_seqs",
        "value_changed": 256,
        "throughput_rps": 2.1,
        "tokens_per_second": 130.0,
        "ttft_p50_ms": 45.0,
        "ttft_p99_ms": 65.0,
        "e2e_p50_ms": 850.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 5.0,
    })
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_seqs",
            "value": 256,
            "rationale": "rps=2.0 suggests more parallelism",
            "status": "pending",
            "experiment_id": None,
        }
    ]

    patch = executor_node(state)

    assert patch["hypotheses"][0]["status"] == "skipped"
    assert "experiments_remaining" not in patch


def test_executor_uses_existing_result_and_updates_best(result_b):
    state = _state_with_baseline()
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "rps=2.0 suggests batching could help",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    analysis = MagicMock(bottleneck="compute-bound")
    comparison = MagicMock(delta_pct=19.0)

    with patch("inferops.agent.executor.get_result_by_id", return_value=result_b), \
         patch("inferops.agent.executor.run_benchmark") as mock_run, \
         patch("inferops.agent.executor.analyze_bottleneck", return_value=analysis), \
         patch("inferops.agent.executor.compare_experiments", return_value=comparison):
        patch_out = executor_node(state)

    mock_run.assert_not_called()
    assert patch_out["hypotheses"][0]["status"] == "success"
    assert patch_out["best_summary"]["experiment_id"] == "sess_max_num_batched_tokens_4096"
    assert patch_out["experiment_summaries"][-1]["vs_baseline_pct"] == 19.0
    assert patch_out["experiments_remaining"] == 3


def test_executor_marks_failed_when_benchmark_raises():
    state = _state_with_baseline()
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "enable_chunked_prefill",
            "value": True,
            "rationale": "TTFT p99=70ms suggests scheduling variance",
            "status": "pending",
            "experiment_id": None,
        }
    ]

    with patch("inferops.agent.executor.get_result_by_id", return_value=None), \
         patch("inferops.tools.propose_config.propose_config_patch"), \
         patch("inferops.agent.executor.run_benchmark", side_effect=RuntimeError("boom")):
        patch_out = executor_node(state)

    assert patch_out["hypotheses"][0]["status"] == "failed"
    assert patch_out["tried_experiment_ids"] == [
        "sess_baseline",
        "sess_enable_chunked_prefill_True",
    ]
    assert patch_out["experiments_remaining"] == 3


def test_executor_runs_benchmark_when_no_existing_result(result_b):
    state = _state_with_baseline()
    state["hypotheses"] = [
        {
            "id": "h1",
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "rps=2.0 suggests batching could help",
            "status": "pending",
            "experiment_id": None,
        }
    ]
    bench_out = RunBenchmarkOutput(
        experiment_id="sess_max_num_batched_tokens_4096",
        workload_name="chat_short",
        throughput_rps=2.38,
        tokens_per_second=152.0,
        ttft_p50_ms=52.0,
        ttft_p99_ms=66.0,
        e2e_p50_ms=780.0,
        e2e_p99_ms=870.0,
        gpu_util_pct=88.0,
        gpu_mem_gb=3.8,
        success_rate="10/10",
        mlflow_run_id=None,
    )

    with patch("inferops.agent.executor.get_result_by_id", side_effect=[None, result_b]), \
         patch("inferops.tools.propose_config.propose_config_patch"), \
         patch("inferops.agent.executor.run_benchmark", return_value=bench_out), \
         patch(
             "inferops.agent.executor.analyze_bottleneck",
             return_value=MagicMock(bottleneck="compute-bound"),
         ), \
         patch(
             "inferops.agent.executor.compare_experiments",
             return_value=MagicMock(delta_pct=19.0),
         ):
        patch_out = executor_node(state)

    assert patch_out["hypotheses"][0]["status"] == "success"
    assert patch_out["experiment_summaries"][-1]["throughput_rps"] == 2.38
