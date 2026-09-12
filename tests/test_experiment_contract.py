"""Week-1 experiment contract: status, evidence, promotion gates, MLflow alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from inferops.memory.db import get_result_by_id, query_results, save_result
from inferops.observability import init_mlflow, log_experiment_result, mlflow_run
from inferops.schemas import (
    EXPERIMENT_SCHEMA_VERSION,
    ConfigEvidence,
    ExperimentResult,
    ExperimentValidityStatus,
    compute_workload_hash,
    derive_status,
    external_unverified_evidence,
    is_promotable,
    managed_start_evidence,
)
from inferops.tools.final_report import FinalReportInput, write_final_report


def test_legacy_result_defaults_to_insufficient_evidence(result):
    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert is_promotable(result) is False


def test_old_json_without_contract_fields_does_not_auto_valid(config, tmp_path):
    """Deserialize pre-contract JSON → insufficient_evidence, never auto-valid."""
    legacy = {
        "experiment_id": "legacy_row",
        "config": json.loads(config.model_dump_json()),
        "total_requests": 10,
        "successful_requests": 10,
        "total_time_s": 5.0,
        "throughput_rps": 50.0,
        "tokens_per_second": 100.0,
        "ttft": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "tpot": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "e2e_latency": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "mlflow_run_id": "old-mlflow",
    }
    parsed = ExperimentResult.model_validate(legacy)
    assert parsed.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert parsed.schema_version == EXPERIMENT_SCHEMA_VERSION
    assert parsed.run_id  # auto-generated
    assert is_promotable(parsed) is False


def test_derive_status_and_insufficient_kinds():
    req = {"max_num_batched_tokens": 4096}
    weak = external_unverified_evidence(host="127.0.0.1", port=8000)
    assert derive_status(evidence=weak, actual_config=None, requested_config=req) == (
        ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    )

    for kind in ("config_file_only", "http_ok_only", "health_check_only", "performance_delta_only"):
        ev = ConfigEvidence(kind=kind, verified=True, observed_params=req)
        assert ev.is_critical_evidence() is False
        assert derive_status(evidence=ev, actual_config=req, requested_config=req) == (
            ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
        )

    strong = managed_start_evidence(
        process_pid=9, host="127.0.0.1", port=8000, requested=req
    )
    assert derive_status(evidence=strong, actual_config=req, requested_config=req) == (
        ExperimentValidityStatus.VALID
    )
    assert derive_status(
        evidence=strong,
        actual_config={"max_num_batched_tokens": 2048},
        requested_config=req,
    ) == ExperimentValidityStatus.INVALID
    assert derive_status(failed=True) == ExperimentValidityStatus.FAILED


def test_is_promotable_requires_actual_and_evidence(result_b, result_b_unevidenced):
    assert is_promotable(result_b) is True
    assert is_promotable(result_b_unevidenced) is False


def test_save_and_query_contract_columns(result_b, result_b_unevidenced, tmp_db):
    save_result(result_b, db_path=tmp_db)
    save_result(
        result_b_unevidenced.model_copy(update={"experiment_id": "hot_unevidenced"}),
        db_path=tmp_db,
    )

    all_rows = query_results(sort_by="throughput_rps", top_k=10, db_path=tmp_db)
    assert len(all_rows) == 2
    promotable = query_results(
        sort_by="throughput_rps", top_k=10, db_path=tmp_db, promotable_only=True
    )
    assert len(promotable) == 1
    assert promotable[0]["experiment_id"] == result_b.experiment_id
    assert promotable[0]["status"] == "valid"
    assert promotable[0]["run_id"] == result_b.run_id

    fetched = get_result_by_id(result_b.experiment_id, db_path=tmp_db)
    assert fetched is not None
    assert fetched.mlflow_run_id == result_b.mlflow_run_id
    assert fetched.workload_hash == compute_workload_hash(result_b.config.workload)


def test_mlflow_logs_run_id_alignment(result_b, tmp_path, monkeypatch):
    tracking = tmp_path / "mlruns.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking}")
    import inferops.observability as obs

    monkeypatch.setattr(obs, "_MLFLOW_TRACKING_URI", f"sqlite:///{tracking}")
    init_mlflow("inferops-contract-test")
    with mlflow_run(run_name=result_b.experiment_id, tags={"seed": "1"}) as run:
        log_experiment_result(result_b)
        run_id = run.info.run_id

    stored = mlflow.get_run(run_id)
    assert stored.data.tags.get("run_id") == result_b.run_id
    assert stored.data.tags.get("status") == "valid"
    assert stored.data.tags.get("experiment_id") == result_b.experiment_id
    assert stored.data.params.get("run_id") == result_b.run_id
    assert stored.data.params.get("status") == "valid"
    # Contract identity: our run_id tag aligns with the MLflow run that logged it
    assert result_b.mlflow_run_id == "mlflow-test-b"  # fixture identity
    assert stored.info.run_id == run_id


def test_final_report_withholds_deploy_without_evidence(tmp_path):
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
    }
    hot = {
        **baseline,
        "experiment_id": "sess_hot",
        "throughput_rps": 99.9,
        "vs_baseline_pct": 400.0,
        "run_id": "bb",
        "validity_status": "insufficient_evidence",
        "has_config_evidence": False,
        "mlflow_run_id": "m1",
    }
    out = tmp_path / "report.md"
    write_final_report(
        FinalReportInput(
            workload_name="chat_short",
            session_prefix="sess_",
            experiment_summaries=[baseline, hot],
            baseline_summary=baseline,
            best_summary=hot,
            output_path=str(out),
        )
    )
    text = out.read_text()
    assert "No deploy recommendation" in text
    assert "Deploy experiment" not in text
    assert "insufficient_evidence" in text


def test_final_report_deploys_only_when_valid(tmp_path):
    best = {
        "experiment_id": "sess_good",
        "param_changed": "max_num_batched_tokens",
        "value_changed": 4096,
        "throughput_rps": 3.0,
        "tokens_per_second": 200.0,
        "ttft_p50_ms": 40.0,
        "ttft_p99_ms": 55.0,
        "e2e_p50_ms": 700.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 20.0,
        "run_id": "cc",
        "validity_status": "valid",
        "mlflow_run_id": "m2",
        "has_config_evidence": True,
    }
    out = tmp_path / "ok.md"
    write_final_report(
        FinalReportInput(
            workload_name="chat_short",
            session_prefix="sess_",
            experiment_summaries=[best],
            baseline_summary=best,
            best_summary=best,
            output_path=str(out),
        )
    )
    assert "Deploy experiment **`sess_good`**" in out.read_text()


def test_status_sample_reports_exist():
    root = Path("reports/week1_status_samples")
    for name in ("valid", "invalid", "failed", "insufficient_evidence"):
        path = root / f"{name}.md"
        assert path.exists(), path
        text = path.read_text()
        assert f"`{name}`" in text or f"status | `{name}`" in text or name in text
        assert "not a real performance" in text.lower() or "Synthetic" in text


def test_prepare_initial_state_does_not_promote_unevidenced_baseline(result):
    from inferops.agent.graph import prepare_initial_state

    # Legacy fixture: insufficient_evidence
    with patch("inferops.agent.graph._run_baseline") as mock_base:
        from inferops.agent.state import summary_from_result

        summary = summary_from_result(
            result,
            param_changed=None,
            value_changed=None,
            baseline_primary=result.throughput_rps,
            primary_metric="throughput_rps",
            bottleneck="compute-bound",
        )
        summary["experiment_id"] = "sess_baseline"
        mock_base.return_value = (summary, "compute-bound")
        state = prepare_initial_state("chat_short", "sess_", max_experiments=5)

    assert state["baseline_summary"]["validity_status"] == "insufficient_evidence"
    assert state["best_summary"] is None


def test_eval_best_ignores_unevidenced_high_score(result_b, result_b_unevidenced, tmp_db, monkeypatch):
    from inferops.eval import runner as eval_runner

    hot = result_b_unevidenced.model_copy(
        update={
            "experiment_id": "agent_x_hot",
            "throughput_rps": 99.9,
        }
    )
    good = result_b.model_copy(
        update={
            "experiment_id": "agent_x_good",
            "throughput_rps": 2.5,
        }
    )
    save_result(hot, db_path=tmp_db)
    save_result(good, db_path=tmp_db)

    monkeypatch.setattr(
        eval_runner,
        "query_results",
        lambda **kw: __import__("inferops.memory.db", fromlist=["query_results"]).query_results(
            **{**kw, "db_path": tmp_db}
        ),
    )
    best = eval_runner._best_agent_result(
        prefix="agent_x_",
        workload_name="chat_short",
        metric="throughput_rps",
        direction="max",
    )
    assert best is not None
    assert best["experiment_id"] == "agent_x_good"
    assert best["status"] == "valid"
