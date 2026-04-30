"""Unit tests for eval runner DB and ground-truth plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from inferops.eval.runner import _best_agent_result, evaluate, load_ground_truth


def _write_ground_truth(path: Path) -> None:
    path.write_text(json.dumps({
        "workload_name": "chat_short",
        "primary_metric": "throughput_rps",
        "higher_is_better": True,
        "best_experiment_id": "gt_best",
        "best_value": 10.0,
        "experiments": [
            {
                "experiment_id": "gt_best",
                "throughput_rps": 10.0,
                "ttft_p99_ms": 50.0,
                "e2e_p50_ms": 100.0,
            }
        ],
    }))


def test_load_ground_truth_reads_json(tmp_path):
    gt_path = tmp_path / "chat_short.json"
    _write_ground_truth(gt_path)

    data = load_ground_truth("chat_short", tmp_path)

    assert data["workload_name"] == "chat_short"
    assert data["best_value"] == 10.0


def test_load_ground_truth_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ground truth not found"):
        load_ground_truth("chat_short", tmp_path)


def test_best_agent_result_filters_prefix_and_max_metric():
    rows = [
        {"experiment_id": "other_1", "throughput_rps": 99.0},
        {"experiment_id": "agent_1", "throughput_rps": 5.0},
        {"experiment_id": "agent_2", "throughput_rps": 7.0},
    ]

    with patch("inferops.eval.runner.query_results", return_value=rows):
        best = _best_agent_result("agent_", "chat_short", "throughput_rps", "max")

    assert best["experiment_id"] == "agent_2"


def test_best_agent_result_returns_none_without_matching_prefix():
    with patch("inferops.eval.runner.query_results", return_value=[{"experiment_id": "other"}]):
        assert _best_agent_result("agent_", "chat_short", "throughput_rps", "max") is None


def test_evaluate_scores_matching_agent_results(tmp_path):
    _write_ground_truth(tmp_path / "chat_short.json")
    rows = [
        {
            "experiment_id": "agent_1",
            "workload_name": "chat_short",
            "throughput_rps": 8.0,
            "ttft_p99_ms": 60.0,
            "e2e_p50_ms": 120.0,
        },
        {
            "experiment_id": "other_1",
            "workload_name": "chat_short",
            "throughput_rps": 9.0,
            "ttft_p99_ms": 55.0,
            "e2e_p50_ms": 110.0,
        },
    ]

    with patch("inferops.eval.runner.query_results", return_value=rows):
        scores = evaluate("agent_", tmp_path, ["chat_short"])

    assert len(scores) == 1
    assert scores[0].workload_name == "chat_short"
    assert scores[0].outcome.agent_value == 8.0
    assert scores[0].efficiency.n_experiments == 1
