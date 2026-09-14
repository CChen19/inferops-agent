"""Unit tests for graph assembly helpers without running vLLM."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from inferops.agent.graph import (
    _print_run_summary,
    _run_baseline,
    build_graph,
    make_llm,
    prepare_initial_state,
    production_checkpointer,
    run_agent,
)
from inferops.schemas import HardwareInfo, InferenceEngine
from inferops.task import default_task_for_workload

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_LEAKS = (
    "inferops_memory.db",
    "inferops_memory.db-wal",
    "inferops_memory.db-shm",
)


def _assert_no_repo_root_sqlite() -> None:
    for name in _SQLITE_LEAKS:
        assert not (_REPO_ROOT / name).exists(), f"leaked {name} into repo root"


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
    assert summary["validity_status"] == "insufficient_evidence"


def test_run_agent_initializes_state_and_invokes_graph(tmp_path):
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
        "run_id": "aa",
        "validity_status": "valid",
        "mlflow_run_id": "m0",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
    }
    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            captured["config"] = config
            state["should_stop"] = True
            state["stop_reason"] = "unit_test"
            return state

    conns: list[sqlite3.Connection] = []
    orig = production_checkpointer

    @contextmanager
    def tracking(db_path=Path("inferops_memory.db")):
        with orig(db_path) as saver:
            conns.append(saver.conn)
            yield saver

    with patch("inferops.agent.graph.production_checkpointer", tracking), \
         patch("inferops.agent.graph.init_db"), \
         patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")), \
         patch("inferops.agent.graph.build_graph", return_value=FakeGraph()), \
         patch("inferops.agent.graph._print_run_summary"):
        final_state = run_agent(
            workload_name="chat_short",
            llm=object(),
            max_experiments=5,
            session_prefix="sess_",
            db_path=tmp_path / "memory.db",
        )

    assert captured["state"]["experiments_remaining"] == 4
    assert captured["state"]["baseline_summary"] == baseline
    assert captured["state"]["memory_db_path"] == str(tmp_path / "memory.db")
    assert captured["config"]["configurable"]["thread_id"] == "sess"
    assert final_state["stop_reason"] == "unit_test"
    assert conns
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conns[0].execute("SELECT 1")
    _assert_no_repo_root_sqlite()


def test_production_checkpointer_closes_connection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with production_checkpointer() as saver:
        saver.setup()
        conn = saver.conn
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    _assert_no_repo_root_sqlite()


def test_prepare_initial_state_passes_confirmed_task_engine_to_fingerprint(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_collect_hardware_info(**kwargs):
        captured.update(kwargs)
        return HardwareInfo(
            model_name=kwargs["model_name"],
            engine=kwargs["engine"],
        )

    monkeypatch.setattr(
        "inferops.memory.hardware.collect_hardware_info",
        fake_collect_hardware_info,
    )
    task = default_task_for_workload("chat_short", 2).model_copy(
        update={"engine": InferenceEngine.OLLAMA}
    )
    baseline = {"experiment_id": "sess_baseline", "promotable": False}

    with patch(
        "inferops.agent.graph._run_baseline",
        return_value=(baseline, "compute-bound"),
    ):
        state = prepare_initial_state(
            "chat_short",
            "sess_",
            task=task,
            db_path=tmp_path / "memory.db",
        )

    assert captured == {
        "model_name": task.model_name,
        "engine": "ollama",
        "probe_nvidia": True,
    }
    assert state["hardware_fingerprint"] is None
    _assert_no_repo_root_sqlite()


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
        "run_id": "aa",
        "validity_status": "valid",
        "mlflow_run_id": "m0",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
    }

    with patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")):
        state = prepare_initial_state("chat_short", "sess_", max_experiments=5)

    assert state["baseline_summary"] == baseline
    assert state["best_summary"] == baseline
    assert state["experiment_summaries"] == [baseline]
    assert state["tried_experiment_ids"] == ["sess_baseline"]
    assert state["current_bottleneck"] == "compute-bound"
    assert state["experiments_remaining"] == 4
    assert "memory_db_path" not in state


def test_prepare_initial_state_skips_best_when_baseline_unevidenced():
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
        "run_id": "aa",
        "validity_status": "insufficient_evidence",
        "mlflow_run_id": None,
        "has_config_evidence": False,
        "promotable": False,
        "failure_reason": "",
    }

    with patch("inferops.agent.graph._run_baseline", return_value=(baseline, "compute-bound")):
        state = prepare_initial_state("chat_short", "sess_", max_experiments=5)

    assert state["baseline_summary"] == baseline
    assert state["best_summary"] is None

def _summary(*, experiment_id: str, vs: float | None, rps: float = 15.565) -> dict:
    return {
        "experiment_id": experiment_id,
        "param_changed": "max_num_seqs",
        "value_changed": 64,
        "throughput_rps": rps,
        "tokens_per_second": 1900.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": vs,
        "run_id": "run",
        "validity_status": "valid",
        "mlflow_run_id": "m",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
    }


def test_print_run_summary_vs_baseline_at_stored_precision(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr(
        "inferops.agent.graph.console",
        Console(file=buf, width=200, no_color=True, highlight=False),
    )
    best = _summary(experiment_id="sess_max_num_seqs_64", vs=3.77)
    baseline = _summary(experiment_id="sess_baseline", vs=0.0, rps=15.0)
    baseline["param_changed"] = None
    baseline["value_changed"] = None
    _print_run_summary({
        "workload_name": "chat_short",
        "stop_reason": "budget_exhausted",
        "tried_experiment_ids": ["sess_baseline", "sess_max_num_seqs_64"],
        "baseline_summary": baseline,
        "best_summary": best,
        "experiment_summaries": [baseline, best],
    })
    text = buf.getvalue()
    assert "+3.77%" in text
    assert "+3.8%" not in text


def test_print_run_summary_missing_vs_baseline_is_not_zero(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr(
        "inferops.agent.graph.console",
        Console(file=buf, width=200, no_color=True, highlight=False),
    )
    missing = _summary(experiment_id="sess_unknown", vs=None)
    _print_run_summary({
        "workload_name": "chat_short",
        "stop_reason": "budget_exhausted",
        "tried_experiment_ids": ["sess_unknown"],
        "baseline_summary": None,
        "best_summary": None,
        "experiment_summaries": [missing],
    })
    text = buf.getvalue()
    assert "n/a" in text
    assert "+0.0%" not in text
    assert "+3.8%" not in text


def test_build_graph_can_render_mermaid():
    graph = build_graph(object())

    mermaid = graph.get_graph().draw_mermaid()

    assert "planner" in mermaid
    assert "executor" in mermaid
    assert "reflector" in mermaid
