"""CPU tests for Stage D compatible prior-session history."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from inferops.agent.planner import _validate_hypotheses, planner_node
from inferops.agent.state import initial_state, is_duplicate
from inferops.memory.db import save_result
from inferops.memory.history import CLAIM_LEVEL, query_compatible_history
from inferops.schemas import ExperimentValidityStatus
from inferops.task import default_task_for_workload

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_OTHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _assert_no_repo_root_sqlite() -> None:
    for name in (
        "inferops_memory.db",
        "inferops_memory.db-wal",
        "inferops_memory.db-shm",
    ):
        assert not (_REPO_ROOT / name).exists(), f"leaked {name} into repo root"


def _row(
    result,
    *,
    experiment_id,
    session_id,
    run_id,
    status,
    notes="",
    throughput_rps=None,
    **cfg_update,
):
    cfg = result.config.model_copy(update={"experiment_id": experiment_id, **cfg_update})
    update = {
        "experiment_id": experiment_id,
        "config": cfg,
        "session_id": session_id,
        "run_id": run_id,
        "status": status,
        "notes": notes,
    }
    if throughput_rps is not None:
        update["throughput_rps"] = throughput_rps
    return result.model_copy(update=update)


def test_query_includes_compatible_prior_and_excludes_other_model_and_current(
    result, workload, tmp_path
):
    db = tmp_path / "history.db"
    compatible = _row(
        result,
        experiment_id="prior_max_num_batched_tokens_4096",
        session_id="prior_",
        run_id="run_prior_ok",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        workload=workload,
    )
    other_model = _row(
        result,
        experiment_id="alien_max_num_seqs_256",
        session_id="alien_",
        run_id="run_alien",
        status=ExperimentValidityStatus.VALID,
        model_name=_OTHER_MODEL,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        workload=workload,
    )
    current = _row(
        result,
        experiment_id="current_max_num_seqs_64",
        session_id="current_",
        run_id="run_current",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        workload=workload,
    )
    save_result(compatible, db_path=db)
    save_result(other_model, db_path=db)
    save_result(current, db_path=db)

    rows = query_compatible_history(
        model_name=_MODEL,
        workload_name="chat_short",
        exclude_session_id="current_",
        db_path=db,
    )
    run_ids = {r["run_id"] for r in rows}
    assert "run_prior_ok" in run_ids
    assert "run_alien" not in run_ids
    assert "run_current" not in run_ids
    prior = next(r for r in rows if r["run_id"] == "run_prior_ok")
    assert prior["claim_level"] == CLAIM_LEVEL
    assert prior["param"] == "max_num_batched_tokens"
    assert prior["value"] == 4096
    _assert_no_repo_root_sqlite()


def test_citing_history_run_id_as_metric_fails_this_session_gate(result, tmp_path, workload):
    db = tmp_path / "history.db"
    prior = _row(
        result,
        experiment_id="prior_max_num_seqs_256",
        session_id="prior_",
        run_id="run_prior_metric",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        throughput_rps=9.5,
        workload=workload,
    )
    save_result(prior, db_path=db)
    history = query_compatible_history(
        model_name=_MODEL,
        workload_name="chat_short",
        exclude_session_id="now_",
        db_path=db,
    )
    assert any(r["run_id"] == "run_prior_metric" for r in history)

    state = initial_state("chat_short", "now_")
    state["compatible_history"] = history
    state["experiment_summaries"] = [
        {
            "experiment_id": "now_baseline",
            "run_id": "run_this",
            "param_changed": None,
            "value_changed": None,
            "throughput_rps": 14.96,
            "tokens_per_second": 1916.0,
            "ttft_p50_ms": 48.0,
            "ttft_p99_ms": 69.0,
            "e2e_p50_ms": 1015.0,
            "bottleneck": "compute-bound",
            "vs_baseline_pct": 0.0,
        }
    ]
    raw = [
        {
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "throughput_rps=9.5 from a prior session",
            "citations": {
                "metric": {
                    "run_id": "run_prior_metric",
                    "metric": "throughput_rps",
                    "value": 9.5,
                }
            },
        }
    ]
    assert _validate_hypotheses(raw, state) == []
    _assert_no_repo_root_sqlite()


def test_failed_history_pair_is_duplicate(result, workload, tmp_path):
    db = tmp_path / "history.db"
    failed = _row(
        result,
        experiment_id="prior_max_num_seqs_256",
        session_id="prior_",
        run_id="run_prior_fail",
        status=ExperimentValidityStatus.FAILED,
        notes="CUDA out of memory",
        model_name=_MODEL,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        workload=workload,
    )
    save_result(failed, db_path=db)
    history = query_compatible_history(
        model_name=_MODEL,
        workload_name="chat_short",
        exclude_session_id="now_",
        db_path=db,
    )
    state = initial_state("chat_short", "now_")
    state["compatible_history"] = history
    assert is_duplicate(state, "max_num_seqs", 256) is True
    assert is_duplicate(state, "max_num_seqs", 64) is False
    _assert_no_repo_root_sqlite()


def test_planner_skips_history_without_memory_db_path(monkeypatch):
    called = {"n": 0}

    def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("must not query history without memory_db_path")

    monkeypatch.setattr("inferops.memory.history.query_compatible_history", boom)
    state = initial_state("chat_short", "test_", max_experiments=6)
    state["baseline_summary"] = {
        "experiment_id": "test_baseline",
        "run_id": "run_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 14.96,
        "tokens_per_second": 1916.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 69.0,
        "e2e_p50_ms": 1015.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
    }
    state["best_summary"] = state["baseline_summary"]
    state["experiment_summaries"] = [state["baseline_summary"]]
    state["current_bottleneck"] = "compute-bound"
    llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({"analysis": "rps=14.96", "hypotheses": []})
    resp.usage_metadata = {}
    llm.invoke.return_value = resp
    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="[source: vllm_scheduler] §Scheduling\ntext",
    ):
        planner_node(state, llm)
    assert called["n"] == 0
    _assert_no_repo_root_sqlite()


def test_planner_prompt_includes_prior_history_and_does_not_cite_it(
    result, workload, tmp_path
):
    db = tmp_path / "history.db"
    prior = _row(
        result,
        experiment_id="prior_max_num_batched_tokens_4096",
        session_id="prior_",
        run_id="run_prior_hint",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        workload=workload,
    )
    save_result(prior, db_path=db)
    task = default_task_for_workload("chat_short", 6, model_name=_MODEL)
    state = initial_state(
        "chat_short",
        "now_",
        max_experiments=6,
        optimization_task=task.model_dump(mode="json"),
    )
    state["memory_db_path"] = str(db)
    state["baseline_summary"] = {
        "experiment_id": "now_baseline",
        "run_id": "run_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 14.96,
        "tokens_per_second": 1916.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 69.0,
        "e2e_p50_ms": 1015.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
    }
    state["best_summary"] = state["baseline_summary"]
    state["experiment_summaries"] = [state["baseline_summary"]]
    state["current_bottleneck"] = "compute-bound"

    captured: dict[str, str] = {}

    def capture(messages):
        if "user" not in captured:
            captured["user"] = messages[1].content
        resp = MagicMock()
        hyp = {
            "param": "max_num_batched_tokens",
            "value": 3072,
            "rationale": "throughput_rps=14.96 [source: vllm_scheduler]",
            "citations": {
                "metric": {
                    "run_id": "run_prior_hint",
                    "metric": "throughput_rps",
                    "value": 2.0,
                },
                "document": {"source": "vllm_scheduler"},
            },
        }
        resp.content = json.dumps({"analysis": "cite prior", "hypotheses": [hyp]})
        resp.usage_metadata = {}
        return resp

    llm = MagicMock()
    llm.invoke.side_effect = capture
    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="[source: vllm_scheduler] §Scheduling\nscheduler guidance",
    ):
        patch_out = planner_node(state, llm)

    assert "PRIOR COMPATIBLE HISTORY" in captured["user"]
    assert "citations.metric.run_id" in captured["user"]
    assert "run_prior_hint" in captured["user"]
    assert patch_out["hypotheses"] == []
    assert patch_out["compatible_history"]
    assert all(r["claim_level"] == CLAIM_LEVEL for r in patch_out["compatible_history"])
    _assert_no_repo_root_sqlite()
