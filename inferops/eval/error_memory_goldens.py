"""Week-3 ⑦ closeout: error-memory goldens + deterministic CPU gate.

Consumes Week-1 ``is_promotable`` / ``derive_status`` and experiment memory
(``save_result``, ``query_results``, ``get_promotable_result``). Does not
invent a second memory schema or rewrite Tune Reflect. Incomplete applyable
actual still cannot promote.

CPU / fixture only. GPU-not-run is not a pass. An empty or skipped
fixture set is a fail.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferops.eval.measurement_goldens import promotable_stub_result
from inferops.memory.db import (
    get_promotable_result,
    get_result_by_id,
    query_results,
    save_result,
)
from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    InferenceEngine,
    LatencyPercentiles,
    ModelSize,
    SchedulerPolicy,
    WorkloadSpec,
    config_knobs,
    derive_status,
    is_promotable,
    managed_cli_actual_config,
    managed_start_evidence,
    stable_legacy_run_id,
)

GOLDEN_SCHEMA = "inferops.error_memory_goldens.v1"
GPU_QUEUE_ENV = "INFEROPS_GPU_GOLDENS"
INVENTED_GPU_FIELDS = ("gpu_utilization_pct", "gpu_memory_used_gb", "cost_usd")

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/error_memory_goldens")

REQUIRED_GOLDEN_IDS: tuple[str, ...] = (
    "failed_row_remembered_not_promotable",
    "unevidenced_high_score_not_promotable",
    "invalid_actual_mismatch_not_promotable",
    "insufficient_evidence_cli_only_not_promotable",
    "zero_success_failed_not_promotable",
    "legacy_incomplete_row_not_promotable",
    "duplicate_failed_config_stays_remembered",
    "oom_failed_row_not_best",
    "promotable_only_excludes_errors",
)


@dataclass
class GoldenCaseResult:
    golden_id: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ErrorMemoryGateResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    cases: list[GoldenCaseResult]
    gpu_status: str  # "not_run" | "queued"

    def report(self) -> str:
        lines = [
            "### Error-memory golden gate",
            "",
            f"- **passed**: `{self.passed}`",
            f"- **gpu_status**: `{self.gpu_status}` "
            "(GPU-not-run is not a pass by itself)",
            f"- **cases**: {len(self.cases)}",
            "",
        ]
        for case in self.cases:
            mark = "ok" if case.ok else "FAIL"
            lines.append(f"- `{case.golden_id}`: {mark}")
            for fail in case.failures:
                lines.append(f"  - {fail}")
        if self.failures:
            lines.append("")
            lines.append("Gate failures:")
            for fail in self.failures:
                lines.append(f"- {fail}")
        if self.warnings:
            lines.append("")
            for warn in self.warnings:
                lines.append(f"- warning: {warn}")
        lines.append("")
        return "\n".join(lines)


def fixture_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_FIXTURE_DIR


def load_catalog(root: str | Path | None = None) -> dict[str, Any]:
    path = fixture_dir(root) / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != GOLDEN_SCHEMA:
        raise ValueError(
            f"golden catalog schema {data.get('schema')!r} is not {GOLDEN_SCHEMA!r}"
        )
    return data


def load_golden_specs(root: str | Path | None = None) -> list[dict[str, Any]]:
    directory = fixture_dir(root)
    specs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "catalog.json":
            continue
        specs.append(json.loads(path.read_text(encoding="utf-8")))
    return specs


def gpu_goldens_queued() -> bool:
    return os.environ.get(GPU_QUEUE_ENV) == "1"


def required_ids_floor(catalog: dict[str, Any] | None = None) -> tuple[str, ...]:
    extras: list[str] = []
    for gid in (catalog or {}).get("required_ids") or ():
        if gid not in REQUIRED_GOLDEN_IDS and gid not in extras:
            extras.append(str(gid))
    return tuple([*REQUIRED_GOLDEN_IDS, *extras])


def catalog_shrunk_below_floor(catalog: dict[str, Any]) -> list[str]:
    if "required_ids" not in catalog:
        return []
    listed = {str(gid) for gid in (catalog.get("required_ids") or [])}
    return [gid for gid in REQUIRED_GOLDEN_IDS if gid not in listed]


def _walk_numeric_gpu_claims(node: Any, *, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if key in INVENTED_GPU_FIELDS and value is not None:
                found.append((child, value))
            found.extend(_walk_numeric_gpu_claims(value, path=child))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_walk_numeric_gpu_claims(item, path=f"{path}[{i}]"))
    return found


def _refuse_invented_gpu_numbers(spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sampled = bool(spec.get("gpu_sampled"))
    queued = gpu_goldens_queued()
    if sampled and not queued:
        failures.append(
            f"{spec.get('id')}: gpu_sampled=true but {GPU_QUEUE_ENV} is unset; "
            "GPU-not-run ≠ pass"
        )
    if sampled and queued:
        return failures
    for path, value in _walk_numeric_gpu_claims(spec):
        failures.append(
            f"{spec.get('id')}: invented {path}={value!r} (gpu_sampled=false)"
        )
    return failures


def _refuse_result_gpu_claims(result: ExperimentResult, golden_id: str) -> list[str]:
    failures: list[str] = []
    if result.gpu_utilization_pct is not None:
        failures.append(f"{golden_id}: invented gpu_utilization_pct")
    if result.gpu_memory_used_gb is not None:
        failures.append(f"{golden_id}: invented gpu_memory_used_gb")
    if result.cost_usd is not None:
        failures.append(f"{golden_id}: invented cost_usd")
    return failures


def _base_config(experiment_id: str, *, max_num_batched_tokens: int = 2048) -> ExperimentConfig:
    workload = WorkloadSpec(
        name="chat_short",
        prompt_template="",
        num_requests=10,
        concurrency=4,
        input_len=64,
        output_len=64,
    )
    return ExperimentConfig(
        experiment_id=experiment_id,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_size=ModelSize.HALF_B,
        engine=InferenceEngine.VLLM,
        max_num_seqs=64,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=1024,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        scheduler_policy=SchedulerPolicy.FCFS,
        workload=workload,
    )


def _missing_lat() -> LatencyPercentiles:
    return LatencyPercentiles()


def _failed_unrun_result(
    experiment_id: str,
    *,
    notes: str,
    run_id: str,
    mlflow_run_id: str = "mlf-err-mem",
) -> ExperimentResult:
    """Failed attempt that never produced a measured window — metrics stay None."""
    config = _base_config(experiment_id)
    knobs = config_knobs(config)
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=0,
        successful_requests=0,
        total_time_s=0.0,
        throughput_rps=None,
        tokens_per_second=None,
        error_rate=None,
        ttft=_missing_lat(),
        tpot=_missing_lat(),
        e2e_latency=_missing_lat(),
        run_id=run_id,
        session_id="errmem_",
        mlflow_run_id=mlflow_run_id,
        requested_config=knobs,
        actual_config=None,
        config_evidence=None,
        status=ExperimentValidityStatus.FAILED,
        notes=notes,
    )


def _zero_success_result(experiment_id: str) -> ExperimentResult:
    config = _base_config(experiment_id)
    knobs = config_knobs(config)
    evidence = managed_start_evidence(
        process_pid=1,
        host="127.0.0.1",
        port=8000,
        observed_params=managed_cli_actual_config(knobs),
    )
    status = derive_status(
        evidence=evidence,
        actual_config=dict(knobs),
        requested_config=knobs,
        successful_requests=0,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=10,
        successful_requests=0,
        total_time_s=5.0,
        throughput_rps=0.0,
        tokens_per_second=None,
        error_rate=1.0,
        ttft=_missing_lat(),
        tpot=_missing_lat(),
        e2e_latency=_missing_lat(),
        run_id="00000000000000000000000000000000",
        session_id="errmem_",
        mlflow_run_id="mlf-zero-success",
        requested_config=knobs,
        actual_config=dict(knobs),
        config_evidence=evidence,
        status=status,
        notes="all requests failed",
    )


def _unevidenced_high_score(experiment_id: str) -> ExperimentResult:
    stub = promotable_stub_result()
    return stub.model_copy(
        update={
            "experiment_id": experiment_id,
            "config": stub.config.model_copy(update={"experiment_id": experiment_id}),
            "throughput_rps": 99.9,
            "tokens_per_second": 999.0,
            "status": ExperimentValidityStatus.INSUFFICIENT_EVIDENCE,
            "actual_config": None,
            "config_evidence": None,
            "run_id": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "mlflow_run_id": "mlf-unevidenced",
        }
    )


def _invalid_mismatch(experiment_id: str) -> ExperimentResult:
    config = _base_config(experiment_id, max_num_batched_tokens=4096)
    knobs = config_knobs(config)
    actual = dict(knobs)
    actual["max_num_batched_tokens"] = 2048
    evidence = managed_start_evidence(
        process_pid=2,
        host="127.0.0.1",
        port=8000,
        observed_params=managed_cli_actual_config(actual),
    )
    status = derive_status(
        evidence=evidence,
        actual_config=actual,
        requested_config=knobs,
        successful_requests=10,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=4.0,
        throughput_rps=3.0,
        tokens_per_second=128.0,
        error_rate=0.0,
        ttft=LatencyPercentiles(p50=50.0, p90=60.0, p95=65.0, p99=70.0),
        tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
        e2e_latency=LatencyPercentiles(p50=800.0, p90=850.0, p95=870.0, p99=900.0),
        run_id="11111111111111111111111111111111",
        session_id="errmem_",
        mlflow_run_id="mlf-invalid",
        requested_config=knobs,
        actual_config=actual,
        config_evidence=evidence,
        status=status,
    )


def _cli_only_insufficient(experiment_id: str) -> ExperimentResult:
    """Incomplete even among CLI-evidenced keys — still insufficient."""
    config = _base_config(experiment_id)
    knobs = config_knobs(config)
    actual = managed_cli_actual_config(knobs)
    actual.pop("max_num_seqs", None)
    evidence = managed_start_evidence(
        process_pid=3,
        host="127.0.0.1",
        port=8000,
        observed_params=actual,
    )
    status = derive_status(
        evidence=evidence,
        actual_config=actual,
        requested_config=knobs,
        successful_requests=10,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=5.0,
        throughput_rps=2.5,
        tokens_per_second=128.0,
        error_rate=0.0,
        ttft=LatencyPercentiles(p50=50.0, p90=60.0, p95=65.0, p99=70.0),
        tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
        e2e_latency=LatencyPercentiles(p50=900.0, p90=950.0, p95=970.0, p99=1000.0),
        run_id="22222222222222222222222222222222",
        session_id="errmem_",
        mlflow_run_id="mlf-cli-only",
        requested_config=knobs,
        actual_config=actual,
        config_evidence=evidence,
        status=status,
    )


def _legacy_incomplete(experiment_id: str) -> ExperimentResult:
    config = _base_config(experiment_id)
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=5.0,
        throughput_rps=4.0,
        tokens_per_second=200.0,
        ttft=LatencyPercentiles(p50=40.0, p90=50.0, p95=55.0, p99=60.0),
        tpot=LatencyPercentiles(p50=5.0, p90=6.0, p95=6.5, p99=7.0),
        e2e_latency=LatencyPercentiles(p50=700.0, p90=750.0, p95=770.0, p99=800.0),
        mlflow_run_id="mlf-legacy",
    )


def _memory_view(db_path: Path, experiment_id: str) -> dict[str, Any]:
    stored = get_result_by_id(experiment_id, db_path=db_path)
    promotable_row = get_promotable_result(experiment_id, db_path=db_path)
    all_rows = query_results(top_k=50, db_path=db_path)
    promo_rows = query_results(top_k=50, db_path=db_path, promotable_only=True)
    return {
        "stored": stored,
        "promotable_row": promotable_row,
        "all_ids": [r["experiment_id"] for r in all_rows],
        "promo_ids": [r["experiment_id"] for r in promo_rows],
        "promo_flags": {r["experiment_id"]: r.get("promotable") for r in all_rows},
    }


def _check_common(
    *,
    golden_id: str,
    result: ExperimentResult,
    view: dict[str, Any],
    expect: dict[str, Any],
) -> list[str]:
    failures = _refuse_result_gpu_claims(result, golden_id)
    got_promo = is_promotable(result)
    if "is_promotable" in expect and got_promo != bool(expect["is_promotable"]):
        failures.append(
            f"{golden_id}: is_promotable={got_promo}, expected {expect['is_promotable']}"
        )
    if expect.get("is_promotable") is True:
        failures.append(
            f"{golden_id}: this error-memory set must not claim is_promotable=true"
        )
    if "status" in expect:
        status = result.status.value if hasattr(result.status, "value") else str(result.status)
        if status != expect["status"]:
            failures.append(f"{golden_id}: status={status!r}, expected {expect['status']!r}")
    stored = view["stored"]
    if expect.get("remembered") is True and stored is None:
        failures.append(f"{golden_id}: row was not remembered in experiment memory")
    if expect.get("get_promotable_result_none") and view["promotable_row"] is not None:
        failures.append(f"{golden_id}: get_promotable_result returned a row")
    if expect.get("promotable_only_includes") is False:
        if result.experiment_id in view["promo_ids"]:
            failures.append(f"{golden_id}: promotable_only included a non-promotable row")
    if expect.get("promotable_flag") is not None:
        flag = view["promo_flags"].get(result.experiment_id)
        if int(flag or 0) != int(expect["promotable_flag"]):
            failures.append(
                f"{golden_id}: denormalized promotable={flag}, "
                f"expected {expect['promotable_flag']}"
            )
    return failures


def _drive_failed_row(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    result = _failed_unrun_result(
        "errmem_failed",
        notes="generic tool failure — no measured window",
        run_id="f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0",
    )
    save_result(result, db_path=db_path)
    return result, _memory_view(db_path, result.experiment_id), []


def _drive_unevidenced(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    hot = _unevidenced_high_score("errmem_unevidenced")
    good = promotable_stub_result().model_copy(
        update={"experiment_id": "errmem_good_neighbor"}
    )
    save_result(hot, db_path=db_path)
    save_result(good, db_path=db_path)
    view = _memory_view(db_path, hot.experiment_id)
    notes = [f"neighbor_promotable={is_promotable(good)}"]
    extra: list[str] = []
    if "errmem_good_neighbor" not in view["promo_ids"]:
        extra.append("unevidenced: Week-1-valid neighbor missing from promotable_only")
    if hot.experiment_id in view["promo_ids"]:
        extra.append("unevidenced: high-score unevidenced row leaked into promotable_only")
    if is_promotable(hot):
        extra.append("unevidenced: high score became promotable (① loosened)")
    return hot, view, extra


def _drive_invalid(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    result = _invalid_mismatch("errmem_invalid")
    extra: list[str] = []
    if result.status != ExperimentValidityStatus.INVALID:
        extra.append(
            f"invalid: derive_status produced {result.status!r}, expected invalid"
        )
    save_result(result, db_path=db_path)
    return result, _memory_view(db_path, result.experiment_id), extra


def _drive_cli_only(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    result = _cli_only_insufficient("errmem_cli_only")
    extra: list[str] = []
    if result.status != ExperimentValidityStatus.INSUFFICIENT_EVIDENCE:
        extra.append(
            f"cli_only: derive_status produced {result.status!r}, "
            "expected insufficient_evidence"
        )
    if is_promotable(result):
        extra.append("cli_only: incomplete CLI actual became promotable (① loosened)")
    save_result(result, db_path=db_path)
    return result, _memory_view(db_path, result.experiment_id), extra


def _drive_zero_success(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    result = _zero_success_result("errmem_zero_success")
    extra: list[str] = []
    if result.status != ExperimentValidityStatus.FAILED:
        extra.append(
            f"zero_success: derive_status produced {result.status!r}, expected failed"
        )
    save_result(result, db_path=db_path)
    return result, _memory_view(db_path, result.experiment_id), extra


def _drive_legacy(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    result = _legacy_incomplete("errmem_legacy")
    extra: list[str] = []
    if result.status != ExperimentValidityStatus.INSUFFICIENT_EVIDENCE:
        extra.append(
            f"legacy: default status={result.status!r}, expected insufficient_evidence"
        )
    expected_run = stable_legacy_run_id(result.experiment_id, result.mlflow_run_id)
    if result.run_id != expected_run:
        extra.append(
            f"legacy: run_id={result.run_id!r} != stable_legacy_run_id {expected_run!r}"
        )
    save_result(result, db_path=db_path)
    view = _memory_view(db_path, result.experiment_id)
    stored = view["stored"]
    if stored is not None and stored.run_id != result.run_id:
        extra.append("legacy: reload minted a new run_id")
    if stored is not None and is_promotable(stored):
        extra.append("legacy: reloaded incomplete row became promotable")
    return result, view, extra


def _drive_duplicate_failed(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    first = _failed_unrun_result(
        "errmem_dup_failed",
        notes="first failure",
        run_id="d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1",
        mlflow_run_id="mlf-dup-1",
    )
    save_result(first, db_path=db_path)
    second = first.model_copy(update={"notes": "retried same config — still failed"})
    save_result(second, db_path=db_path)
    view = _memory_view(db_path, first.experiment_id)
    extra: list[str] = []
    stored = view["stored"]
    if stored is None:
        extra.append("duplicate: upsert dropped the failed row")
    elif stored.notes != "retried same config — still failed":
        extra.append("duplicate: upsert did not keep the latest failed notes")
    rows = query_results(top_k=50, db_path=db_path)
    same_id = [r for r in rows if r["experiment_id"] == first.experiment_id]
    if len(same_id) != 1:
        extra.append(f"duplicate: expected one remembered row, got {len(same_id)}")
    return second, view, extra


def _drive_oom(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    oom = _failed_unrun_result(
        "errmem_oom",
        notes="vLLM OOM during startup",
        run_id="0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        mlflow_run_id="mlf-oom",
    )
    good = promotable_stub_result().model_copy(update={"experiment_id": "errmem_oom_neighbor"})
    save_result(oom, db_path=db_path)
    save_result(good, db_path=db_path)
    view = _memory_view(db_path, oom.experiment_id)
    extra: list[str] = []
    if view["promotable_row"] is not None:
        extra.append("oom: failed OOM row returned by get_promotable_result")
    if "errmem_oom" in view["promo_ids"]:
        extra.append("oom: OOM row treated as best/promotable")
    if "errmem_oom_neighbor" not in view["promo_ids"]:
        extra.append("oom: valid neighbor missing from promotable_only")
    return oom, view, extra


def _drive_promotable_only(db_path: Path) -> tuple[ExperimentResult, dict[str, Any], list[str]]:
    failed = _failed_unrun_result(
        "errmem_mix_failed",
        notes="failed mix",
        run_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    unevidenced = _unevidenced_high_score("errmem_mix_unevidenced")
    invalid = _invalid_mismatch("errmem_mix_invalid")
    good = promotable_stub_result().model_copy(update={"experiment_id": "errmem_mix_good"})
    for row in (failed, unevidenced, invalid, good):
        save_result(row, db_path=db_path)
    view = _memory_view(db_path, failed.experiment_id)
    extra: list[str] = []
    if view["promo_ids"] != ["errmem_mix_good"]:
        extra.append(
            f"promotable_only: expected ['errmem_mix_good'], got {view['promo_ids']!r}"
        )
    if not is_promotable(good):
        extra.append("promotable_only: Week-1-valid control is not promotable")
    return failed, view, extra


DRIVERS = {
    "failed_row_remembered_not_promotable": _drive_failed_row,
    "unevidenced_high_score_not_promotable": _drive_unevidenced,
    "invalid_actual_mismatch_not_promotable": _drive_invalid,
    "insufficient_evidence_cli_only_not_promotable": _drive_cli_only,
    "zero_success_failed_not_promotable": _drive_zero_success,
    "legacy_incomplete_row_not_promotable": _drive_legacy,
    "duplicate_failed_config_stays_remembered": _drive_duplicate_failed,
    "oom_failed_row_not_best": _drive_oom,
    "promotable_only_excludes_errors": _drive_promotable_only,
}


def evaluate_golden(spec: dict[str, Any], *, db_path: Path | None = None) -> GoldenCaseResult:
    golden_id = str(spec.get("id") or "unknown")
    failures = _refuse_invented_gpu_numbers(spec)
    notes: list[str] = []
    if spec.get("skip") or spec.get("skipped"):
        failures.append(f"{golden_id}: skipped golden is not a pass")
    if spec.get("schema") != GOLDEN_SCHEMA:
        failures.append(f"{golden_id}: schema {spec.get('schema')!r} != {GOLDEN_SCHEMA!r}")
    if spec.get("synthetic") is not True:
        failures.append(f"{golden_id}: fixture must be synthetic")
    if spec.get("gpu_sampled") is True and not gpu_goldens_queued():
        failures.append(f"{golden_id}: GPU-not-run ≠ pass")
    if (spec.get("expect") or {}).get("is_promotable") is True:
        failures.append(f"{golden_id}: this set must not claim is_promotable=true")
    if (spec.get("expect") or {}).get("confirmed_promotable") is True:
        failures.append(f"{golden_id}: this set must not claim confirmed_promotable=true")

    driver_name = str(spec.get("driver") or golden_id)
    driver = DRIVERS.get(driver_name)
    if driver is None:
        failures.append(f"{golden_id}: unknown driver {driver_name!r}")
        return GoldenCaseResult(golden_id=golden_id, ok=False, failures=failures)

    own_tmp: tempfile.TemporaryDirectory[str] | None = None
    resolved = db_path
    if resolved is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="inferops_errmem_")
        resolved = Path(own_tmp.name) / "error_memory.db"
    try:
        result, view, extra = driver(resolved)
        failures.extend(extra)
        failures.extend(
            _check_common(
                golden_id=golden_id,
                result=result,
                view=view,
                expect=spec.get("expect") or {},
            )
        )
        notes.append(f"remembered={view['stored'] is not None}")
        notes.append(f"promotable_only={view['promo_ids']}")
    except Exception as exc:  # noqa: BLE001 — gate records the failure
        failures.append(f"{golden_id}: driver raised {type(exc).__name__}: {exc}")
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()

    return GoldenCaseResult(
        golden_id=golden_id,
        ok=not failures,
        failures=failures,
        notes=notes,
    )


def error_memory_golden_gate(root: str | Path | None = None) -> ErrorMemoryGateResult:
    """Deterministic CI gate. Empty / skipped / GPU-not-run is not a pass."""
    failures: list[str] = []
    warnings: list[str] = []
    gpu_status = "queued" if gpu_goldens_queued() else "not_run"

    catalog = load_catalog(root)
    if catalog.get("gpu_queued") and gpu_status != "queued":
        failures.append(
            "catalog.gpu_queued=true but GPU goldens were not queued; "
            "GPU-not-run ≠ pass"
        )
    if gpu_status == "not_run" and catalog.get("cpu_only") is not True:
        failures.append("CPU-only catalog required while GPU is not queued")
    shrunk = catalog_shrunk_below_floor(catalog)
    if shrunk:
        failures.append(f"catalog required_ids shrunk below floor: {shrunk}")

    required = required_ids_floor(catalog)
    specs = load_golden_specs(root)
    by_id = {str(spec.get("id")): spec for spec in specs}
    missing = [gid for gid in required if gid not in by_id]
    if missing:
        failures.append(f"required goldens missing: {missing}")
    if not specs:
        failures.append("GPU-not-run ≠ pass: no CPU goldens evaluated")

    extra = sorted(set(by_id) - set(required) - set(REQUIRED_GOLDEN_IDS))
    if extra:
        warnings.append(f"extra golden ids (allowed): {extra}")

    cases = [evaluate_golden(spec) for spec in specs]
    for case in cases:
        failures.extend(case.failures)

    return ErrorMemoryGateResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        cases=cases,
        gpu_status=gpu_status,
    )
