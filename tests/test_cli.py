"""Unit tests for the Typer CLI entry points."""

from __future__ import annotations

from typer.testing import CliRunner

from inferops.cli import app

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "agent" in result.output
    assert "grid-sweep" in result.output
    assert "memory" in result.output


def test_agent_rejects_unknown_workload():
    result = runner.invoke(app, ["agent", "--workload", "not_a_workload"])

    assert result.exit_code != 0
    assert "Unknown workload" in result.output


def test_memory_command_prints_rows(monkeypatch):
    def fake_query_results(workload_name=None, sort_by="throughput_rps", top_k=10):
        assert workload_name == "chat_short"
        assert sort_by == "throughput_rps"
        assert top_k == 1
        return [{
            "experiment_id": "e1",
            "workload_name": "chat_short",
            "throughput_rps": 2.5,
            "ttft_p99_ms": 70.0,
            "e2e_p50_ms": 900.0,
        }]

    monkeypatch.setattr("inferops.memory.db.query_results", fake_query_results)

    result = runner.invoke(app, [
        "memory",
        "--workload",
        "chat_short",
        "--top-k",
        "1",
    ])

    assert result.exit_code == 0
    assert "e1" in result.output
    assert "chat_short" in result.output


def test_eval_command_exits_when_no_scores(monkeypatch, tmp_path):
    monkeypatch.setattr("inferops.eval.runner.evaluate", lambda *args, **kwargs: [])

    result = runner.invoke(app, ["eval", "--prefix", "agent_", "--ground-truth", str(tmp_path)])

    assert result.exit_code == 1
