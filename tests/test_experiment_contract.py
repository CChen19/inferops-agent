"""Week-1 experiment contract: status, evidence, promotion gates, MLflow alignment.

Correction-round regressions for P1-1..P1-4 and P2-5..P2-6.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from inferops.agent.executor import executor_node
from inferops.agent.graph import prepare_initial_state
from inferops.agent.state import (
    initial_state,
    is_promotable_summary,
    summary_from_result,
)
from inferops.bench_runner import BenchmarkError, OOMError, run_experiment
from inferops.eval import runner as eval_runner
from inferops.memory.db import get_result_by_id, query_results, save_result
from inferops.observability import init_mlflow, log_experiment_result, mlflow_run
from inferops.schemas import (
    EXPERIMENT_SCHEMA_VERSION,
    ConfigEvidence,
    ExperimentResult,
    ExperimentValidityStatus,
    LatencyPercentiles,
    MANAGED_CLI_EVIDENCED_KEYS,
    actual_covers_requested,
    compute_workload_hash,
    config_knobs,
    derive_status,
    empty_latency,
    external_unverified_evidence,
    is_promotable,
    managed_cli_actual_config,
    managed_start_evidence,
    stable_legacy_run_id,
)
from inferops.tools.final_report import FinalReportInput, write_final_report
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_lat():
    return empty_latency()


def _make_failed_load_result(config, **kwargs):
    base = dict(
        experiment_id=config.experiment_id,
        config=config,
        total_requests=10,
        successful_requests=0,
        total_time_s=1.0,
        throughput_rps=0.0,
        tokens_per_second=0.0,
        ttft=_zero_lat(),
        tpot=_zero_lat(),
        e2e_latency=_zero_lat(),
        run_id="deadbeefdeadbeefdeadbeefdeadbeef",
        status=ExperimentValidityStatus.FAILED,
        requested_config=config_knobs(config),
        actual_config=None,
        config_evidence=None,
    )
    base.update(kwargs)
    return ExperimentResult(**base)


def _executor_state():
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
        "run_id": "ffffffffffffffffffffffffffffffff",
        "validity_status": "valid",
        "mlflow_run_id": "mlflow-baseline",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
    }
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline]
    state["tried_experiment_ids"] = ["sess_baseline"]
    state["current_bottleneck"] = "compute-bound"
    state["experiments_remaining"] = 4
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
    return state


# ---------------------------------------------------------------------------
# Core gate / derive_status
# ---------------------------------------------------------------------------

def test_legacy_result_defaults_to_insufficient_evidence(result):
    assert result.status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert is_promotable(result) is False


def test_old_json_without_contract_fields_does_not_auto_valid(config):
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
    assert parsed.run_id == stable_legacy_run_id("legacy_row", "old-mlflow")
    assert is_promotable(parsed) is False


def test_legacy_run_id_stable_across_rereads(config):
    """P2-6: missing run_id must not mint a new UUID per deserialize."""
    legacy = {
        "experiment_id": "legacy_stable",
        "config": json.loads(config.model_dump_json()),
        "total_requests": 10,
        "successful_requests": 10,
        "total_time_s": 5.0,
        "throughput_rps": 1.0,
        "tokens_per_second": 10.0,
        "ttft": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "tpot": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "e2e_latency": {"p50": 1, "p90": 2, "p95": 3, "p99": 4},
        "mlflow_run_id": "mlf-legacy",
    }
    a = ExperimentResult.model_validate(legacy)
    b = ExperimentResult.model_validate(legacy)
    assert a.run_id == b.run_id
    assert a.run_id == stable_legacy_run_id("legacy_stable", "mlf-legacy")


def test_legacy_run_id_backfilled_in_db(config, tmp_db):
    """P2-6: DB re-read of legacy row keeps identical run_id after backfill."""
    legacy = ExperimentResult(
        experiment_id="legacy_db",
        config=config.model_copy(update={"experiment_id": "legacy_db"}),
        total_requests=10,
        successful_requests=10,
        total_time_s=5.0,
        throughput_rps=1.0,
        tokens_per_second=10.0,
        ttft=_zero_lat(),
        tpot=_zero_lat(),
        e2e_latency=_zero_lat(),
        run_id="",  # force stable fill
        mlflow_run_id="mlf-db",
    )
    # Simulate pre-contract JSON (no run_id key)
    raw = json.loads(legacy.model_dump_json())
    del raw["run_id"]
    from inferops.memory.db import init_db, _connect
    init_db(tmp_db)
    with _connect(tmp_db) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, workload_name, config_hash, config_json, result_json)
            VALUES (?,?,?,?,?)
            """,
            ("legacy_db", "chat_short", "x", "{}", json.dumps(raw)),
        )
        conn.commit()

    first = get_result_by_id("legacy_db", db_path=tmp_db)
    second = get_result_by_id("legacy_db", db_path=tmp_db)
    assert first is not None and second is not None
    assert first.run_id == second.run_id
    assert first.run_id == stable_legacy_run_id("legacy_db", "mlf-db")


def test_empty_actual_with_critical_evidence_not_valid_or_promotable(config):
    """P1-2: empty actual + critical evidence must NOT be valid/promotable."""
    req = config_knobs(config)
    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=req
    )
    assert derive_status(
        evidence=ev, actual_config={}, requested_config=req
    ) == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE
    assert derive_status(
        evidence=ev, actual_config=None, requested_config=req
    ) == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE

    result = ExperimentResult(
        experiment_id="empty_actual",
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=1.0,
        throughput_rps=99.0,
        tokens_per_second=100.0,
        ttft=_zero_lat(),
        tpot=_zero_lat(),
        e2e_latency=_zero_lat(),
        run_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        requested_config=req,
        actual_config={},
        config_evidence=ev,
        status=ExperimentValidityStatus.VALID,  # wrongly stamped
    )
    # Even if status was incorrectly set to valid, full gate rejects
    assert actual_covers_requested(req, {}) is False
    assert is_promotable(result) is False


def test_partial_cli_actual_insufficient_evidence(config):
    """P1-1: managed CLI-only actual cannot claim valid while non-CLI keys remain."""
    req = config_knobs(config)
    actual = managed_cli_actual_config(req)
    assert "scheduler_policy" not in actual
    assert "tensor_parallel_size" not in actual
    assert set(actual) <= MANAGED_CLI_EVIDENCED_KEYS

    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=actual
    )
    status = derive_status(evidence=ev, actual_config=actual, requested_config=req)
    assert status == ExperimentValidityStatus.INSUFFICIENT_EVIDENCE

    result = ExperimentResult(
        experiment_id="cli_only",
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=1.0,
        throughput_rps=5.0,
        tokens_per_second=50.0,
        ttft=_zero_lat(),
        tpot=_zero_lat(),
        e2e_latency=_zero_lat(),
        run_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        requested_config=req,
        actual_config=actual,
        config_evidence=ev,
        status=status,
    )
    assert is_promotable(result) is False


def test_mismatched_actual_is_invalid_not_promotable(config):
    req = config_knobs(config)
    actual = dict(req)
    actual["max_num_seqs"] = req["max_num_seqs"] + 1
    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=actual
    )
    assert derive_status(
        evidence=ev, actual_config=actual, requested_config=req
    ) == ExperimentValidityStatus.INVALID


def test_zero_successful_requests_failed_not_promotable(config):
    """P1-4: all-failed workload → failed / not promotable."""
    req = config_knobs(config)
    actual = dict(req)
    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=actual
    )
    status = derive_status(
        evidence=ev,
        actual_config=actual,
        requested_config=req,
        successful_requests=0,
    )
    assert status == ExperimentValidityStatus.FAILED
    result = _make_failed_load_result(
        config,
        requested_config=req,
        actual_config=actual,
        config_evidence=ev,
        status=status,
    )
    assert is_promotable(result) is False


def test_is_promotable_requires_actual_and_evidence(result_b, result_b_unevidenced):
    assert is_promotable(result_b) is True
    assert is_promotable(result_b_unevidenced) is False


# ---------------------------------------------------------------------------
# Unified gate across executor / baseline / eval / report (P1-3)
# ---------------------------------------------------------------------------

def test_unified_gate_rejects_valid_looking_partial_everywhere(config, result_b, tmp_db, tmp_path):
    """Same counterexample must fail executor, baseline, eval, AND report."""
    req = config_knobs(config)
    actual = managed_cli_actual_config(req)  # missing non-CLI keys
    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=actual
    )
    # Intentionally stamp status=valid to simulate inconsistent older writers
    hot = ExperimentResult(
        experiment_id="sess_max_num_batched_tokens_4096",
        config=config.model_copy(update={
            "experiment_id": "sess_max_num_batched_tokens_4096",
            "max_num_batched_tokens": 4096,
        }),
        total_requests=10,
        successful_requests=10,
        total_time_s=1.0,
        throughput_rps=99.9,
        tokens_per_second=999.0,
        ttft=LatencyPercentiles(p50=1, p90=2, p95=3, p99=4),
        tpot=LatencyPercentiles(p50=1, p90=2, p95=3, p99=4),
        e2e_latency=LatencyPercentiles(p50=1, p90=2, p95=3, p99=4),
        run_id="cccccccccccccccccccccccccccccccc",
        requested_config=req,
        actual_config=actual,
        config_evidence=ev,
        status=ExperimentValidityStatus.VALID,
        mlflow_run_id="mlf-hot",
    )
    assert is_promotable(hot) is False
    summary = summary_from_result(
        hot, "max_num_batched_tokens", 4096, 2.0, "throughput_rps"
    )
    assert summary["promotable"] is False
    assert is_promotable_summary(summary) is False

    # --- executor ---
    state = _executor_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=hot), \
         patch("inferops.agent.executor.run_benchmark"), \
         patch("inferops.agent.executor.analyze_bottleneck",
               return_value=MagicMock(bottleneck="compute-bound")), \
         patch("inferops.agent.executor.compare_experiments",
               return_value=MagicMock(delta_pct=400.0)):
        out = executor_node(state)
    assert out["best_summary"]["experiment_id"] == "sess_baseline"

    # --- baseline seeding ---
    with patch("inferops.agent.graph._run_baseline", return_value=(summary, "unknown")):
        # force experiment_id for baseline shape
        summary = {**summary, "experiment_id": "sess_baseline", "vs_baseline_pct": 0.0}
    with patch("inferops.agent.graph._run_baseline", return_value=(summary, "unknown")):
        st = prepare_initial_state("chat_short", "sess_", max_experiments=3)
    assert st["best_summary"] is None

    # --- DB / eval ---
    save_result(hot, db_path=tmp_db)
    save_result(result_b.model_copy(update={"experiment_id": "agent_x_good"}), db_path=tmp_db)
    prom = query_results(top_k=10, db_path=tmp_db, promotable_only=True)
    assert all(r["experiment_id"] != hot.experiment_id for r in prom)

    # --- final report ---
    fake_best = {
        **summary,
        "experiment_id": hot.experiment_id,
        "promotable": False,
        "failure_reason": "",
    }
    out_path = tmp_path / "r.md"
    write_final_report(FinalReportInput(
        workload_name="chat_short",
        session_prefix="sess_",
        experiment_summaries=[fake_best],
        baseline_summary=fake_best,
        best_summary=fake_best,
        output_path=str(out_path),
    ))
    text = out_path.read_text()
    assert "No deploy recommendation" in text
    assert "Deploy experiment" not in text


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
    assert promotable[0]["promotable"] == 1
    assert promotable[0]["run_id"] == result_b.run_id

    fetched = get_result_by_id(result_b.experiment_id, db_path=tmp_db)
    assert fetched is not None
    assert fetched.mlflow_run_id == result_b.mlflow_run_id
    assert fetched.workload_hash == compute_workload_hash(result_b.config.workload)


def test_eval_best_ignores_unevidenced_high_score(result_b, result_b_unevidenced, tmp_db, monkeypatch):
    hot = result_b_unevidenced.model_copy(
        update={"experiment_id": "agent_x_hot", "throughput_rps": 99.9}
    )
    good = result_b.model_copy(
        update={"experiment_id": "agent_x_good", "throughput_rps": 2.5}
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
    assert best["promotable"] == 1


def test_executor_does_not_promote_zero_success(config):
    """P1-4 via executor path."""
    req = config_knobs(config)
    actual = dict(req)
    ev = managed_start_evidence(
        process_pid=1, host="127.0.0.1", port=8000, observed_params=actual
    )
    failed = _make_failed_load_result(
        config.model_copy(update={"experiment_id": "sess_max_num_batched_tokens_4096"}),
        requested_config=req,
        actual_config=actual,
        config_evidence=ev,
        status=ExperimentValidityStatus.FAILED,
        throughput_rps=0.0,
    )
    state = _executor_state()
    with patch("inferops.agent.executor.get_result_by_id", return_value=failed), \
         patch("inferops.agent.executor.run_benchmark"), \
         patch("inferops.agent.executor.analyze_bottleneck",
               return_value=MagicMock(bottleneck="unknown")), \
         patch("inferops.agent.executor.compare_experiments",
               return_value=MagicMock(delta_pct=-100.0)):
        out = executor_node(state)
    assert out["best_summary"]["experiment_id"] == "sess_baseline"
    assert out["experiment_summaries"][-1]["validity_status"] == "failed"
    assert out["experiment_summaries"][-1]["promotable"] is False


# ---------------------------------------------------------------------------
# Failed start persistence (P2-5)
# ---------------------------------------------------------------------------

def test_start_failure_persists_failed_contract_row(config, tmp_db, monkeypatch):
    """P2-5: OOM/start failure persists failed ExperimentResult with stable run_id."""
    from inferops.bench_runner import run_experiment as real_run

    failed_result = _make_failed_load_result(
        config.model_copy(update={"experiment_id": "fail_start"}),
        notes="vLLM OOM during startup",
        mlflow_run_id="mlf-fail-real",
        run_id="f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1",
    )

    def boom(*args, **kwargs):
        raise OOMError("vLLM OOM during startup", result=failed_result)

    monkeypatch.setattr("inferops.tools.run_benchmark.run_experiment", boom)
    monkeypatch.setattr(
        "inferops.tools.run_benchmark.save_result",
        lambda r: save_result(r, db_path=tmp_db),
    )
    monkeypatch.setattr(
        "inferops.tools.run_benchmark.get_prompts",
        lambda w: ["p"] * 5,
    )

    with pytest.raises(OOMError) as ei:
        run_benchmark(RunBenchmarkInput(
            experiment_id="fail_start",
            workload_name="chat_short",
            persist=True,
        ))
    assert ei.value.result is not None
    assert ei.value.result.status == ExperimentValidityStatus.FAILED
    assert ei.value.result.run_id == "f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1"

    stored = get_result_by_id("fail_start", db_path=tmp_db)
    assert stored is not None
    assert stored.status == ExperimentValidityStatus.FAILED
    assert stored.run_id == "f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1"
    assert stored.mlflow_run_id == "mlf-fail-real"
    assert is_promotable(stored) is False


def test_bench_runner_oom_builds_failed_result_with_mlflow_id(config):
    """Direct bench_runner path: failed result carries run_id + mlflow_run_id."""
    prompts = ["hi"]
    fake_run = MagicMock()
    fake_run.info.run_id = "mlflow-active-xyz"

    class FakeProc:
        log_path = Path("/tmp/fake.log")
        _pid = 42
        start_token = "tok"

        @property
        def pid(self):
            return self._pid

        def start(self):
            return None

        def wait_ready_verbose(self, log):
            return False

        def oom_in_log(self):
            return True

        def is_crashed(self):
            return False

        def stop(self):
            self._pid = None

        def identity(self):
            from inferops.tools.vllm_process import InstanceIdentity
            return InstanceIdentity(
                host="127.0.0.1", port=8000, pid=self._pid, start_token=self.start_token
            )

        def evidenced_actual_config(self):
            return {}

    with patch("inferops.bench_runner.init_mlflow"), \
         patch("inferops.bench_runner.mlflow_run") as mock_mlf, \
         patch("inferops.bench_runner.log_experiment_result") as mock_log, \
         patch("inferops.bench_runner.VLLMProcess", return_value=FakeProc()), \
         patch("inferops.bench_runner.probe_live_instance") as mock_probe:
        from inferops.tools.vllm_process import LiveProbe
        mock_probe.return_value = LiveProbe(healthy=False)
        mock_mlf.return_value.__enter__.return_value = fake_run
        mock_mlf.return_value.__exit__.return_value = None

        with pytest.raises(OOMError) as ei:
            run_experiment(config, prompts)

    assert ei.value.result is not None
    fr = ei.value.result
    assert fr.status == ExperimentValidityStatus.FAILED
    assert fr.mlflow_run_id == "mlflow-active-xyz"
    assert fr.run_id  # stable for this attempt
    assert fr.successful_requests == 0
    mock_log.assert_called_once()
    logged = mock_log.call_args[0][0]
    assert logged.run_id == fr.run_id
    assert logged.mlflow_run_id == "mlflow-active-xyz"


# ---------------------------------------------------------------------------
# MLflow alignment
# ---------------------------------------------------------------------------

def test_mlflow_logs_run_id_alignment(result_b, tmp_path, monkeypatch):
    tracking = tmp_path / "mlruns.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking}")
    import inferops.observability as obs

    monkeypatch.setattr(obs, "_MLFLOW_TRACKING_URI", f"sqlite:///{tracking}")
    init_mlflow("inferops-contract-test")
    with mlflow_run(run_name=result_b.experiment_id, tags={"seed": "1"}) as run:
        # Align result.mlflow_run_id with the *active* MLflow run (not a fixture fake)
        aligned = result_b.model_copy(update={"mlflow_run_id": run.info.run_id})
        log_experiment_result(aligned)
        active_id = run.info.run_id

    assert aligned.mlflow_run_id == active_id
    stored = mlflow.get_run(active_id)
    assert stored.data.tags.get("run_id") == aligned.run_id
    assert stored.data.tags.get("status") == "valid"
    assert stored.data.tags.get("experiment_id") == aligned.experiment_id
    assert stored.data.params.get("run_id") == aligned.run_id

    # Flows through DB
    from inferops.memory.db import save_result as _save
    db = tmp_path / "mem.db"
    _save(aligned, db_path=db)
    fetched = get_result_by_id(aligned.experiment_id, db_path=db)
    assert fetched is not None
    assert fetched.mlflow_run_id == active_id
    assert fetched.run_id == aligned.run_id

    # Flows through report
    summary = summary_from_result(
        aligned, "max_num_batched_tokens", 4096, 2.0, "throughput_rps"
    )
    assert summary["mlflow_run_id"] == active_id
    assert summary["run_id"] == aligned.run_id
    assert summary["promotable"] is True


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
        "promotable": True,
        "failure_reason": "",
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
        "promotable": False,
        "failure_reason": "",
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


def test_final_report_deploys_only_when_promotable(tmp_path):
    baseline = {
        "experiment_id": "sess_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 2.5,
        "tokens_per_second": 160.0,
        "ttft_p50_ms": 45.0,
        "ttft_p99_ms": 60.0,
        "e2e_p50_ms": 750.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
        "run_id": "bb",
        "validity_status": "valid",
        "mlflow_run_id": "m1",
        "has_config_evidence": True,
        "promotable": True,
        "failure_reason": "",
        "error_rate": 0.0,
        "actual_config": {"max_num_batched_tokens": 2048},
    }
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
        "promotable": True,
        "failure_reason": "",
        "error_rate": 0.0,
        "actual_config": {"max_num_batched_tokens": 4096},
    }
    out = tmp_path / "ok.md"
    write_final_report(
        FinalReportInput(
            workload_name="chat_short",
            session_prefix="sess_",
            experiment_summaries=[baseline, best],
            baseline_summary=baseline,
            best_summary=best,
            output_path=str(out),
        )
    )
    text = out.read_text()
    assert "sess_good" in text
    assert "confirmed_and_meets_goals" in text
    assert "No deploy recommendation" not in text


def test_final_report_rejects_valid_without_promotable_flag(tmp_path):
    """P1-3: status=valid + evidence alone is not enough without promotable=True."""
    best = {
        "experiment_id": "sess_spoof",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 99.0,
        "tokens_per_second": 1.0,
        "ttft_p50_ms": 1.0,
        "ttft_p99_ms": 1.0,
        "e2e_p50_ms": 1.0,
        "bottleneck": "unknown",
        "vs_baseline_pct": 100.0,
        "run_id": "dd",
        "validity_status": "valid",
        "mlflow_run_id": "m3",
        "has_config_evidence": True,
        "promotable": False,
        "failure_reason": ""
    }
    out = tmp_path / "spoof.md"
    write_final_report(FinalReportInput(
        workload_name="chat_short",
        session_prefix="sess_",
        experiment_summaries=[best],
        best_summary=best,
        output_path=str(out),
    ))
    assert "No deploy recommendation" in out.read_text()


def test_status_sample_reports_exist():
    root = Path("reports/week1_status_samples")
    for name in ("valid", "invalid", "failed", "insufficient_evidence"):
        path = root / f"{name}.md"
        assert path.exists(), path


def test_prepare_initial_state_does_not_promote_unevidenced_baseline(result):
    summary = summary_from_result(
        result,
        param_changed=None,
        value_changed=None,
        baseline_primary=result.throughput_rps,
        primary_metric="throughput_rps",
        bottleneck="compute-bound",
    )
    summary["experiment_id"] = "sess_baseline"
    with patch("inferops.agent.graph._run_baseline", return_value=(summary, "compute-bound")):
        state = prepare_initial_state("chat_short", "sess_", max_experiments=5)

    assert state["baseline_summary"]["validity_status"] == "insufficient_evidence"
    assert state["baseline_summary"]["promotable"] is False
    assert state["best_summary"] is None


def test_summary_from_result_sets_promotable(result_b):
    summary = summary_from_result(
        result_b, "max_num_batched_tokens", 4096, 2.0, "throughput_rps"
    )
    assert summary["promotable"] is True
    assert summary["validity_status"] == "valid"
    assert is_promotable_summary(summary) is True


# ---------------------------------------------------------------------------
# P2-5 correction round 2: executor surfaces failed contract rows in report
# ---------------------------------------------------------------------------

def test_executor_benchmark_error_appends_failed_summary_and_trajectory(config):
    """BenchmarkError(exc.result) → failed summary + trajectory (not silent drop)."""
    from inferops.bench_runner import OOMError

    failed = _make_failed_load_result(
        config.model_copy(update={"experiment_id": "sess_max_num_batched_tokens_4096"}),
        notes="vLLM OOM during startup — config: sess_max_num_batched_tokens_4096",
        mlflow_run_id="mlf-oom-exec",
        run_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    state = _executor_state()

    with patch(
        "inferops.agent.executor.get_result_by_id", return_value=None
    ), patch("inferops.tools.propose_config.propose_config_patch"), patch(
        "inferops.agent.executor.run_benchmark",
        side_effect=OOMError("vLLM OOM during startup", result=failed),
    ):
        out = executor_node(state)

    assert out["hypotheses"][0]["status"] == "failed"
    assert out["experiments_remaining"] == 3
    assert len(out["experiment_summaries"]) == len(state["experiment_summaries"]) + 1
    summary = out["experiment_summaries"][-1]
    assert summary["experiment_id"] == "sess_max_num_batched_tokens_4096"
    assert summary["validity_status"] == "failed"
    assert summary["run_id"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert summary["mlflow_run_id"] == "mlf-oom-exec"
    assert "OOM" in summary["failure_reason"]
    assert summary["promotable"] is False
    # best unchanged
    assert out["best_summary"]["experiment_id"] == "sess_baseline"
    traj = out["trajectory"][-1]
    assert traj["validity_status"] == "failed"
    assert traj["run_id"] == summary["run_id"]
    assert traj["result"]["failure_reason"]
    assert traj["result"]["promoted_to_best"] is False


def test_oom_run_benchmark_executor_final_report_e2e(config, tmp_path, tmp_db, monkeypatch):
    """OOM → run_benchmark persists → executor summary → final_report shows fields."""
    from inferops.bench_runner import OOMError

    eid = "sess_max_num_batched_tokens_4096"
    failed = _make_failed_load_result(
        config.model_copy(update={"experiment_id": eid}),
        notes="vLLM OOM during startup — config: " + eid,
        mlflow_run_id="mlf-oom-e2e",
        run_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )

    def boom(*args, **kwargs):
        raise OOMError("vLLM OOM during startup", result=failed)

    monkeypatch.setattr("inferops.tools.run_benchmark.run_experiment", boom)
    monkeypatch.setattr(
        "inferops.tools.run_benchmark.save_result",
        lambda r: save_result(r, db_path=tmp_db),
    )
    monkeypatch.setattr(
        "inferops.tools.run_benchmark.get_prompts", lambda w: ["p"] * 3
    )

    state = _executor_state()
    # Use real run_benchmark (which persists then re-raises) via executor
    with patch("inferops.agent.executor.get_result_by_id", return_value=None), \
         patch("inferops.tools.propose_config.propose_config_patch"):
        # Don't mock run_benchmark — let it call our boom via run_experiment
        out = executor_node(state)

    stored = get_result_by_id(eid, db_path=tmp_db)
    assert stored is not None
    assert stored.status == ExperimentValidityStatus.FAILED
    assert stored.run_id == failed.run_id
    assert stored.mlflow_run_id == "mlf-oom-e2e"

    summary = out["experiment_summaries"][-1]
    assert summary["validity_status"] == "failed"
    assert summary["run_id"] == failed.run_id
    assert summary["mlflow_run_id"] == "mlf-oom-e2e"
    assert "OOM" in summary["failure_reason"]

    report_path = tmp_path / "final.md"
    write_final_report(FinalReportInput(
        workload_name="chat_short",
        session_prefix="sess_",
        experiment_summaries=out["experiment_summaries"],
        baseline_summary=state["baseline_summary"],
        best_summary=out.get("best_summary"),
        output_path=str(report_path),
    ))
    text = report_path.read_text()
    assert "Failed attempts" in text
    assert eid in text
    assert failed.run_id in text
    assert "mlf-oom-e2e" in text
    assert "failed" in text
    assert "OOM" in text


def test_run_experiment_zero_success_not_promotable(config):
    """Nice-to-have: successful=0 after load → failed / not promotable."""
    from types import SimpleNamespace

    from inferops.metrics.ledger import (
        RequestLedger,
        RequestOutcome,
        RequestRecord,
        TerminationReason,
    )
    from inferops.tools.traffic import LoadResult
    from inferops.tools.vllm_process import InstanceIdentity, LiveProbe

    run_id = "zero_success_run_id_000000000001"
    ledger = RequestLedger(run_id=run_id, window_start_s=100.0, window_end_s=101.0)
    for i in range(10):
        ledger.add(
            RequestRecord(
                run_id=run_id,
                request_id=f"req-{i:04d}",
                t_start_s=100.0 + i * 0.01,
                t_end_s=100.5 + i * 0.01,
                e2e_ms=500.0,
                output_tokens=0,
                outcome=RequestOutcome.FAIL,
                termination_reason=TerminationReason.ERROR,
                error="synthetic",
            )
        )
    load = LoadResult(
        total_requests=10,
        successful=0,
        total_time_s=1.0,
        throughput_rps=0.0,
        tokens_per_second=0.0,
        ttft_ms=[],
        e2e_ms=[],
        error_rate=1.0,
        ledger=ledger,
    )
    # samples>0 so GPU is recorded; zero successes still fail the gate
    gpu_summary = SimpleNamespace(max_mem_used_gb=1.0, avg_util_pct=10.0, samples=2)
    fake_run = MagicMock()
    fake_run.info.run_id = "mlf-zero-success"

    req = config_knobs(config)

    class FakeProc:
        log_path = None
        _pid = 7
        start_token = "tok"

        @property
        def pid(self):
            return self._pid

        def start(self):
            return None

        def wait_ready_verbose(self, log):
            return True

        def stop(self):
            return None

        def identity(self):
            return InstanceIdentity(
                host="127.0.0.1", port=8000, pid=self._pid, start_token=self.start_token
            )

        def evidenced_actual_config(self):
            return dict(req)

        def oom_in_log(self):
            return False

        def is_crashed(self):
            return False

    class FakeGPU:
        def start(self):
            return None

        def stop(self):
            return gpu_summary

    with patch("inferops.bench_runner.init_mlflow"), \
         patch("inferops.bench_runner.mlflow_run") as mock_mlf, \
         patch("inferops.bench_runner.log_experiment_result"), \
         patch("inferops.bench_runner.VLLMProcess", return_value=FakeProc()), \
         patch("inferops.bench_runner.GPUMonitor", return_value=FakeGPU()), \
         patch("inferops.bench_runner._run_load_with_cleanup_workaround", return_value=load), \
         patch("inferops.bench_runner.probe_live_instance", return_value=LiveProbe(healthy=False)), \
         patch("inferops.bench_runner.assert_listener_bound_to_child", return_value=7), \
         patch("inferops.bench_runner.config_knobs", return_value=dict(req)), \
         patch("inferops.bench_runner.persist_ledger"):
        mock_mlf.return_value.__enter__.return_value = fake_run
        mock_mlf.return_value.__exit__.return_value = None
        with patch("inferops.bench_runner.uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = run_id
            result = run_experiment(config, ["p"])

    assert result.successful_requests == 0
    assert result.status == ExperimentValidityStatus.FAILED
    assert result.mlflow_run_id == "mlf-zero-success"
    assert is_promotable(result) is False
    assert result.error_rate == 1.0
    assert result.ttft.p50 is None  # no inventing zeros
    assert result.tpot.p50 is None


def test_promotable_column_backfill_for_preexisting_valid_rows(result_b, tmp_db):
    """Nice-to-have: NULL promotable on legacy valid rows is recomputed on init_db."""
    from inferops.memory.db import init_db, _connect

    # Insert a fully promotable result_json but leave promotable NULL (pre-column)
    init_db(tmp_db)
    with _connect(tmp_db) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, workload_name, config_hash, config_json, result_json,
                 throughput_rps, status, promotable)
            VALUES (?,?,?,?,?,?,?,NULL)
            """,
            (
                result_b.experiment_id,
                result_b.config.workload.name,
                "hash",
                result_b.config.model_dump_json(),
                result_b.model_dump_json(),
                result_b.throughput_rps,
                "valid",
            ),
        )
        conn.commit()

    # Re-init triggers backfill
    init_db(tmp_db)
    rows = query_results(top_k=10, db_path=tmp_db, promotable_only=True)
    assert any(r["experiment_id"] == result_b.experiment_id for r in rows)
    with _connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT promotable FROM experiments WHERE experiment_id = ?",
            (result_b.experiment_id,),
        ).fetchone()
    assert row["promotable"] == 1
