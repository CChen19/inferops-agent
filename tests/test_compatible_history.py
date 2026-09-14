"""CPU tests for Stage D compatible prior-session history + hardware fingerprint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from inferops.agent.planner import _validate_hypotheses, planner_node
from inferops.agent.state import initial_state, is_duplicate
from inferops.memory.db import save_result
from inferops.memory.hardware import HardwareFingerprint
from inferops.memory.history import CLAIM_LEVEL, _recover_param_value, query_compatible_history
from inferops.schemas import ExperimentValidityStatus, HardwareInfo
from inferops.task import default_task_for_workload

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_OTHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_FP: HardwareFingerprint = {
    "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "gpu_memory_total_gb": 6.0,
    "model_name": _MODEL,
    "engine": "vllm",
    "vllm_version": "0.6.0",
}
_FP_OTHER: HardwareFingerprint = {
    **_FP,
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_memory_total_gb": 40.0,
}


def _assert_no_repo_root_sqlite() -> None:
    for name in (
        "inferops_memory.db",
        "inferops_memory.db-wal",
        "inferops_memory.db-shm",
    ):
        assert not (_REPO_ROOT / name).exists(), f"leaked {name} into repo root"


def _hw(**overrides) -> HardwareInfo:
    base = dict(
        model_name=_MODEL,
        engine="vllm",
        vllm_version="0.6.0",
        gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
        gpu_memory_total_gb=6.0,
    )
    base.update(overrides)
    return HardwareInfo(**base)


def _row(
    result,
    *,
    experiment_id,
    session_id,
    run_id,
    status,
    notes="",
    throughput_rps=None,
    hardware=None,
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
        "hardware": hardware if hardware is not None else _hw(model_name=cfg.model_name),
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
        hardware=_hw(model_name=_OTHER_MODEL),
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
        current_fingerprint=_FP,
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


def test_hardware_mismatch_and_unknown_excluded_from_ranking(result, workload, tmp_path):
    db = tmp_path / "history.db"
    match = _row(
        result,
        experiment_id="prior_max_num_seqs_64",
        session_id="prior_",
        run_id="run_match",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=64,
        workload=workload,
    )
    mismatch = _row(
        result,
        experiment_id="prior_a100_max_num_seqs_64",
        session_id="a100_",
        run_id="run_mismatch",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=64,
        workload=workload,
        hardware=_hw(gpu_name="NVIDIA A100-SXM4-40GB", gpu_memory_total_gb=40.0),
    )
    legacy = _row(
        result,
        experiment_id="prior_legacy_max_num_seqs_64",
        session_id="legacy_",
        run_id="run_legacy",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=64,
        workload=workload,
        hardware=_hw(gpu_name=None, gpu_memory_total_gb=None, vllm_version=None),
    )
    save_result(match, db_path=db)
    save_result(mismatch, db_path=db)
    save_result(legacy, db_path=db)

    rows = query_compatible_history(
        model_name=_MODEL,
        workload_name="chat_short",
        exclude_session_id="now_",
        db_path=db,
        current_fingerprint=_FP,
    )
    run_ids = {r["run_id"] for r in rows}
    assert run_ids == {"run_match"}

    # Incomplete current fingerprint → nothing ranked
    assert (
        query_compatible_history(
            model_name=_MODEL,
            workload_name="chat_short",
            exclude_session_id="now_",
            db_path=db,
            current_fingerprint=None,
        )
        == []
    )
    _assert_no_repo_root_sqlite()


def test_multi_knob_diff_does_not_guess_param(result, workload, tmp_path):
    db = tmp_path / "history.db"
    # Two non-default knobs; experiment_id lacks _max_num_seqs_ token.
    multi = _row(
        result,
        experiment_id="prior_session_trial_r3",
        session_id="prior_",
        run_id="run_multi",
        status=ExperimentValidityStatus.FAILED,
        notes="CUDA out of memory",
        model_name=_MODEL,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        workload=workload,
    )
    save_result(multi, db_path=db)
    rows = query_compatible_history(
        model_name=_MODEL,
        workload_name="chat_short",
        exclude_session_id="now_",
        db_path=db,
        current_fingerprint=_FP,
    )
    assert len(rows) == 1
    assert rows[0]["param"] is None
    assert rows[0]["value"] is None
    state = initial_state("chat_short", "now_")
    state["compatible_history"] = rows
    # Unknown param must not suppress an untried pair
    assert is_duplicate(state, "max_num_seqs", 256) is False
    assert is_duplicate(state, "max_num_batched_tokens", 4096) is False
    _assert_no_repo_root_sqlite()


def test_single_knob_diff_still_recovers():
    cfg = {
        "model_name": _MODEL,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 2048,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    param, value = _recover_param_value("anything", cfg)
    assert param == "max_num_seqs"
    assert value == 256


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
        current_fingerprint=_FP,
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
        current_fingerprint=_FP,
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


def test_planner_uses_state_fingerprint_without_env(
    result, workload, tmp_path, monkeypatch
):
    """Default path: no env vars; injected run-start fingerprint ranks matching history."""
    for key in ("INFEROPS_GPU_NAME", "INFEROPS_GPU_MEM_GB", "VLLM_VERSION"):
        monkeypatch.delenv(key, raising=False)

    db = tmp_path / "history.db"
    prior = _row(
        result,
        experiment_id="prior_max_num_batched_tokens_4096",
        session_id="prior_",
        run_id="run_prior_wired",
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
    state["hardware_fingerprint"] = dict(_FP)
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

    # Fallback collect must not be required when state fingerprint is present.
    def boom_collect(**kwargs):
        raise AssertionError("must use state hardware_fingerprint, not collect")

    monkeypatch.setattr("inferops.memory.hardware.collect_hardware_info", boom_collect)

    llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({"analysis": "ok", "hypotheses": []})
    resp.usage_metadata = {}
    llm.invoke.return_value = resp
    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="[source: vllm_scheduler] §Scheduling\ntext",
    ):
        patch_out = planner_node(state, llm)

    assert any(r["run_id"] == "run_prior_wired" for r in patch_out["compatible_history"])
    _assert_no_repo_root_sqlite()


def test_planner_incomplete_fingerprint_ranks_zero_history(
    result, workload, tmp_path, monkeypatch
):
    for key in ("INFEROPS_GPU_NAME", "INFEROPS_GPU_MEM_GB", "VLLM_VERSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "inferops.memory.hardware.collect_hardware_info",
        lambda **kw: HardwareInfo(model_name=_MODEL, engine="vllm"),
    )

    db = tmp_path / "history.db"
    prior = _row(
        result,
        experiment_id="prior_max_num_seqs_64",
        session_id="prior_",
        run_id="run_prior_hidden",
        status=ExperimentValidityStatus.VALID,
        model_name=_MODEL,
        max_num_seqs=64,
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
    state["hardware_fingerprint"] = None  # incomplete / unknown current
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

    llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps({"analysis": "ok", "hypotheses": []})
    resp.usage_metadata = {}
    llm.invoke.return_value = resp
    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="(knowledge index not built — run scripts/build_corpus.py)",
    ):
        patch_out = planner_node(state, llm)

    assert patch_out["compatible_history"] == []
    _assert_no_repo_root_sqlite()


def test_prepare_initial_state_stores_hardware_fingerprint(tmp_path, monkeypatch):
    from inferops.agent.graph import prepare_initial_state

    monkeypatch.setenv("INFEROPS_GPU_NAME", _FP["gpu_name"])
    monkeypatch.setenv("INFEROPS_GPU_MEM_GB", "6.0")
    monkeypatch.setenv("VLLM_VERSION", _FP["vllm_version"])
    monkeypatch.setattr(
        "inferops.agent.graph._run_baseline",
        lambda *a, **k: (
            {
                "experiment_id": "s_baseline",
                "run_id": "r0",
                "param_changed": None,
                "value_changed": None,
                "throughput_rps": 1.0,
                "tokens_per_second": 1.0,
                "ttft_p50_ms": 1.0,
                "ttft_p99_ms": 1.0,
                "e2e_p50_ms": 1.0,
                "bottleneck": "compute-bound",
                "vs_baseline_pct": 0.0,
                "validity_status": "valid",
                "has_config_evidence": True,
                "promotable": True,
                "failure_reason": "",
                "error_rate": 0.0,
                "mlflow_run_id": None,
            },
            "compute-bound",
        ),
    )
    task = default_task_for_workload("chat_short", 3, model_name=_MODEL)
    state = prepare_initial_state(
        "chat_short",
        "s_",
        max_experiments=3,
        task=task,
        db_path=tmp_path / "mem.db",
    )
    assert state.get("memory_db_path")
    fp = state.get("hardware_fingerprint")
    assert isinstance(fp, dict)
    assert fp["gpu_name"] == _FP["gpu_name"]
    assert fp["vllm_version"] == _FP["vllm_version"]
    _assert_no_repo_root_sqlite()


def test_planner_prompt_includes_prior_history_and_does_not_cite_it(
    result, workload, tmp_path, monkeypatch
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
    state["hardware_fingerprint"] = dict(_FP)
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
    assert "hardware fingerprint" in captured["user"]
    assert "citations.metric.run_id" in captured["user"]
    assert "run_prior_hint" in captured["user"]
    assert patch_out["hypotheses"] == []
    assert patch_out["compatible_history"]
    assert all(r["claim_level"] == CLAIM_LEVEL for r in patch_out["compatible_history"])
    _assert_no_repo_root_sqlite()
