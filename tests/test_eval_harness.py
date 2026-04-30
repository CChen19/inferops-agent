"""Unit tests for commit-level eval harness output."""

from __future__ import annotations

from pathlib import Path

from inferops.eval.harness import (
    render_markdown_report,
    run_mock_eval,
    run_session_eval,
    write_eval_outputs,
)


FIXTURES = Path("tests/fixtures/ground_truth")


def test_run_mock_eval_scores_baseline_strategies():
    report = run_mock_eval(
        commit_sha="abc123",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short", "long_generation"],
        budget=2,
        seed=7,
    )

    assert report["commit_sha"] == "abc123"
    assert set(report["strategies"]) == {"random_agent", "greedy_agent"}
    assert len(report["strategies"]["greedy_agent"]) == 2
    assert "mean_composite" in report["aggregates"]["greedy_agent"]
    assert (
        report["aggregates"]["greedy_agent"]["mean_composite"]
        > report["aggregates"]["random_agent"]["mean_composite"]
    )


def test_render_markdown_report_contains_dashboard_tables():
    report = run_mock_eval("abc123", FIXTURES, workloads=["chat_short"], budget=2)

    text = render_markdown_report(report)

    assert "# InferOps Eval Report" in text
    assert "random_agent" in text
    assert "greedy_agent" in text
    assert "| Strategy | Mean gap %" in text


def test_write_eval_outputs_writes_markdown_and_json(tmp_path):
    report = run_mock_eval("abc123", FIXTURES, workloads=["chat_short"], budget=2)

    md_path, json_path = write_eval_outputs(report, tmp_path)

    assert md_path.exists()
    assert json_path.exists()
    assert md_path.name == "abc123.md"
    assert json_path.name == "abc123.json"


def test_run_session_eval_wraps_runner_scores(monkeypatch):
    from inferops.eval.metrics import OutcomeMetrics, WorkloadScore, compute_efficiency

    outcome = OutcomeMetrics(
        workload_name="chat_short",
        primary_metric="throughput_rps",
        ground_truth_value=10.0,
        agent_value=9.0,
        gap_pct=10.0,
    )
    score = WorkloadScore(
        workload_name="chat_short",
        outcome=outcome,
        efficiency=compute_efficiency(3, 120.0),
        trajectory_score=None,
        composite=0.8,
    )
    monkeypatch.setattr("inferops.eval.harness.evaluate", lambda **kwargs: [score])

    report = run_session_eval(
        commit_sha="abc123",
        prefix="agent_",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short"],
        wall_clock_s=120.0,
    )

    assert report["mode"] == "session"
    assert report["prefix"] == "agent_"
    assert report["strategies"]["agent_session"][0]["gap_pct"] == 10.0
