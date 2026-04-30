"""Unit tests for run_benchmark tool — mocks bench_runner so no vLLM needed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark


def test_run_benchmark_success(result, tmp_db):
    with patch("inferops.tools.run_benchmark.run_experiment", return_value=result) as mock_run, \
         patch("inferops.tools.run_benchmark.save_result") as mock_save, \
         patch("inferops.tools.run_benchmark.get_prompts", return_value=["prompt"] * 20):
        inp = RunBenchmarkInput(
            experiment_id="test_run",
            config_patch={"max_num_batched_tokens": 4096},
            workload_name="chat_short",
            persist=True,
        )
        out = run_benchmark(inp)

    assert out.experiment_id == result.experiment_id
    assert out.throughput_rps == result.throughput_rps
    assert out.success_rate == "10/10"
    mock_run.assert_called_once()
    mock_save.assert_called_once()


def test_run_benchmark_no_persist(result):
    with patch("inferops.tools.run_benchmark.run_experiment", return_value=result), \
         patch("inferops.tools.run_benchmark.save_result") as mock_save, \
         patch("inferops.tools.run_benchmark.get_prompts", return_value=["p"] * 20):
        out = run_benchmark(RunBenchmarkInput(
            experiment_id="no_persist",
            workload_name="chat_short",
            persist=False,
        ))

    mock_save.assert_not_called()
    assert out.experiment_id == result.experiment_id


def test_run_benchmark_unsafe_param():
    with pytest.raises(ValueError, match="outside safe range"):
        run_benchmark(RunBenchmarkInput(
            experiment_id="bad",
            config_patch={"gpu_memory_utilization": 0.99},  # exceeds 0.85 limit
            workload_name="chat_short",
        ))


def test_run_benchmark_rejects_unknown_patch_key():
    with pytest.raises(ValueError, match="Unknown config_patch key"):
        run_benchmark(RunBenchmarkInput(
            experiment_id="bad_key",
            config_patch={"not_a_vllm_knob": 123},
            workload_name="chat_short",
        ))


def test_run_benchmark_unknown_workload():
    with pytest.raises(ValueError, match="Unknown workload"):
        run_benchmark(RunBenchmarkInput(
            experiment_id="bad_wl",
            workload_name="nonexistent",
        ))
