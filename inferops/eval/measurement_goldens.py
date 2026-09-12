"""Week-2 P0-⑦: measurement-trust goldens + deterministic CPU gate.

Consumes ④ ledger / ⑤ confirmation / ⑥ Reflect conclusions already on
master. Does not invent a second metrics schema, loosen Week-1
``is_promotable``, or redefine Reflect actions.

CPU / fixture only. GPU-not-run is not a pass and invents no GPU/perf
numbers. A GPU golden is legal only when Chris queues
``INFEROPS_GPU_GOLDENS=1``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferops.agent.reflect_constraints import conclude_experiment
from inferops.metrics.aggregate import recalculate_from_ledger
from inferops.metrics.confirm import (
    DEFAULT_MIN_PAIRS,
    DEFAULT_MIN_REL_DELTA,
    RepeatPhase,
    is_confirmed_promotable,
    verdict_from_ledgers,
)
from inferops.metrics.definitions import compute_tpot_ms
from inferops.metrics.ledger import (
    LEDGER_SCHEMA_VERSION,
    RequestLedger,
    RequestOutcome,
    RequestRecord,
    RunConditions,
    TerminationReason,
    TokenCountSource,
)
from inferops.schemas import (
    ConfigEvidence,
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    InferenceEngine,
    LatencyPercentiles,
    ModelSize,
    SchedulerPolicy,
    WorkloadSpec,
    config_knobs,
    is_promotable,
)

GOLDEN_SCHEMA = "inferops.measurement_goldens.v1"
GPU_QUEUE_ENV = "INFEROPS_GPU_GOLDENS"
INVENTED_GPU_FIELDS = ("gpu_utilization_pct", "gpu_memory_used_gb", "cost_usd")

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/measurement_goldens")

REQUIRED_GOLDEN_IDS: tuple[str, ...] = (
    "missing_requests",
    "failures_not_dropped",
    "tpot_na",
    "search_win_unconfirmed",
    "too_noisy",
    "no_reliable_improvement",
)

CONDITIONS = RunConditions(
    workload_name="chat_short",
    num_requests=10,
    concurrency=4,
    input_len_target=64,
    output_len_target=64,
    distribution="uniform",
    arrival_rps=None,
    warmup_requests=0,
    stream_response=True,
    sampling_temperature=0.0,
    cache_enabled=False,
)


@dataclass
class GoldenCaseResult:
    golden_id: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class MeasurementGateResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    cases: list[GoldenCaseResult]
    gpu_status: str  # "not_run" | "queued"

    def report(self) -> str:
        lines = [
            "### Measurement-trust golden gate",
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
    """Fixtures must not smuggle GPU/perf numbers when GPU was not sampled."""
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


def _record_from_spec(run_id: str, raw: dict[str, Any]) -> RequestRecord:
    source = raw.get("token_count_source", "missing")
    return RequestRecord(
        run_id=run_id,
        request_id=raw["request_id"],
        is_warmup=bool(raw.get("is_warmup", False)),
        t_start_s=float(raw["t_start_s"]),
        t_first_token_s=raw.get("t_first_token_s"),
        t_end_s=raw.get("t_end_s"),
        input_tokens=raw.get("input_tokens"),
        output_tokens=raw.get("output_tokens"),
        token_count_source=TokenCountSource(source),
        outcome=RequestOutcome(raw["outcome"]),
        termination_reason=TerminationReason(raw["termination_reason"]),
        error=str(raw.get("error") or ""),
    )


def ledger_from_explicit(spec: dict[str, Any]) -> RequestLedger:
    run_id = spec["run_id"]
    ledger = RequestLedger(
        run_id=run_id,
        schema_version=LEDGER_SCHEMA_VERSION,
        conditions=CONDITIONS,
        window_start_s=spec.get("window_start_s"),
        window_end_s=spec.get("window_end_s"),
    )
    for raw in spec.get("records") or []:
        ledger.add(_record_from_spec(run_id, raw))
    return ledger


def rps_ledger(
    run_id: str,
    *,
    rps: float | None,
    n_success: int = 10,
    n_error: int = 0,
) -> RequestLedger:
    """Deterministic throughput fixture. ``rps is None`` → empty (missing) ledger."""
    if rps is None:
        return RequestLedger(run_id=run_id, conditions=CONDITIONS)
    window_s = n_success / rps if rps else 1.0
    ledger = RequestLedger(
        run_id=run_id,
        conditions=CONDITIONS,
        window_start_s=1000.0,
        window_end_s=1000.0 + window_s,
    )
    t0 = 1000.0
    for i in range(n_success):
        ledger.add(
            RequestRecord(
                run_id=run_id,
                request_id=f"req-{i:04d}",
                t_start_s=t0 + i * 0.01,
                t_first_token_s=t0 + i * 0.01 + 0.05,
                t_end_s=t0 + i * 0.01 + 0.2,
                output_tokens=16,
                input_tokens=8,
                token_count_source=TokenCountSource.USAGE,
                outcome=RequestOutcome.SUCCESS,
                termination_reason=TerminationReason.STOP,
            )
        )
    for i in range(n_error):
        ledger.add(
            RequestRecord(
                run_id=run_id,
                request_id=f"err-{i:04d}",
                t_start_s=t0 + 0.5 + i * 0.01,
                t_end_s=t0 + 0.6 + i * 0.01,
                output_tokens=None,
                token_count_source=TokenCountSource.MISSING,
                outcome=RequestOutcome.TIMEOUT,
                termination_reason=TerminationReason.TIMEOUT,
                error="timeout",
            )
        )
    return ledger


def _ledgers_from_pairs(spec: dict[str, Any]) -> tuple[list[RequestLedger], list[RequestLedger]]:
    baseline: list[RequestLedger] = []
    candidate: list[RequestLedger] = []
    for pair in spec.get("pairs") or []:
        b = pair["baseline"]
        c = pair["candidate"]
        baseline.append(
            rps_ledger(
                b["run_id"],
                rps=b.get("rps"),
                n_success=int(b.get("n_success") or 10),
                n_error=int(b.get("n_error") or 0),
            )
        )
        candidate.append(
            rps_ledger(
                c["run_id"],
                rps=c.get("rps"),
                n_success=int(c.get("n_success") or 10),
                n_error=int(c.get("n_error") or 0),
            )
        )
    return baseline, candidate


def promotable_stub_result(*, run_id: str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") -> ExperimentResult:
    """Week-1-valid result so goldens can prove ⑤ still refuses promotion."""
    workload = WorkloadSpec(
        name="chat_short",
        prompt_template="",
        num_requests=10,
        concurrency=4,
        input_len=64,
        output_len=64,
    )
    config = ExperimentConfig(
        experiment_id="golden_promotable",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_size=ModelSize.HALF_B,
        engine=InferenceEngine.VLLM,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        scheduler_policy=SchedulerPolicy.FCFS,
        workload=workload,
    )
    knobs = config_knobs(config)
    lp = LatencyPercentiles(p50=50.0, p90=60.0, p95=65.0, p99=70.0)
    return ExperimentResult(
        experiment_id="golden_promotable",
        config=config,
        total_requests=10,
        successful_requests=10,
        total_time_s=5.0,
        throughput_rps=2.0,
        tokens_per_second=128.0,
        error_rate=0.0,
        ttft=lp,
        tpot=LatencyPercentiles(p50=6.0, p90=7.0, p95=7.5, p99=8.0),
        e2e_latency=LatencyPercentiles(p50=900.0, p90=950.0, p95=970.0, p99=1000.0),
        run_id=run_id,
        session_id="golden_",
        mlflow_run_id="mlflow-golden",
        requested_config=knobs,
        actual_config=dict(knobs),
        config_evidence=ConfigEvidence(
            kind="managed_process_start",
            verified=True,
            instance_id="127.0.0.1:8000:pid=1",
            process_pid=1,
            observed_params=dict(knobs),
        ),
        status=ExperimentValidityStatus.VALID,
    )


def _values_close(got: Any, expected: Any) -> bool:
    if expected is None:
        return got is None
    if got is None:
        return False
    if isinstance(expected, float) or isinstance(got, float):
        return abs(float(got) - float(expected)) < 1e-9
    return got == expected


def _aggregate_view(agg) -> dict[str, Any]:
    return {
        "total_requests": agg.total_requests,
        "successful_requests": agg.successful_requests,
        "failed_requests": agg.failed_requests,
        "error_rate": agg.error_rate,
        "throughput_rps": agg.throughput_rps,
        "tokens_per_second": agg.tokens_per_second,
        "total_output_tokens": agg.total_output_tokens,
        "tpot_p50": agg.tpot.p50,
        "tpot_sample_n": agg.tpot.sample_n,
        "ttft_p50": agg.ttft.p50,
        "gpu_utilization_pct": agg.gpu_utilization_pct,
        "gpu_memory_used_gb": agg.gpu_memory_used_gb,
        "cost_usd": agg.cost_usd,
        "outcome_counts": agg.outcome_counts,
    }


def _check_aggregate(
    role: str,
    view: dict[str, Any],
    expected: dict[str, Any],
    *,
    null_must_not_become_zero: list[str],
) -> list[str]:
    failures: list[str] = []
    for key, want in expected.items():
        got = view.get(key)
        if not _values_close(got, want):
            failures.append(f"{role}.{key}: got {got!r}, expected {want!r}")
    for key in null_must_not_become_zero:
        if key in expected and expected[key] is None and view.get(key) == 0:
            failures.append(
                f"{role}.{key}: missing became 0 — measurement-trust loosened"
            )
        if key in expected and expected[key] is None and view.get(key) is not None:
            failures.append(
                f"{role}.{key}: missing became {view.get(key)!r} (must stay None)"
            )
    if view.get("gpu_utilization_pct") is not None:
        failures.append(f"{role}: GPU-not-run invented gpu_utilization_pct")
    if view.get("gpu_memory_used_gb") is not None:
        failures.append(f"{role}: GPU-not-run invented gpu_memory_used_gb")
    if view.get("cost_usd") is not None:
        failures.append(f"{role}: invented cost_usd")
    return failures


def _candidate_run_id(spec: dict[str, Any]) -> str:
    pairs = spec.get("pairs") or []
    if pairs:
        return str(pairs[-1]["candidate"]["run_id"])
    return "golden_candidate"


def _reflect_conclusion(spec: dict[str, Any], decision) -> Any:
    latest = {
        "experiment_id": "golden_cand",
        "param_changed": "max_num_batched_tokens",
        "value_changed": 4096,
        "validity_status": "valid",
        "error_rate": 0.0,
        "vs_baseline_pct": 0.5,
        "run_id": _candidate_run_id(spec),
        "bottleneck": "compute-bound",
        "failure_reason": "",
    }
    baseline = {
        "experiment_id": "golden_baseline",
        "param_changed": None,
        "value_changed": None,
        "validity_status": "valid",
        "error_rate": 0.0,
        "vs_baseline_pct": 0.0,
        "run_id": "golden_baseline",
        "bottleneck": "compute-bound",
    }
    return conclude_experiment(
        experiments_remaining=3,
        no_improvement_streak=0,
        current_bottleneck="compute-bound",
        latest=latest,
        baseline=baseline,
        best=baseline,
        summaries=[baseline, latest],
        confirmation_decision=decision,
        primary_metric=spec.get("metric") or "throughput_rps",
    )


def evaluate_golden(
    spec: dict[str, Any],
    *,
    result: ExperimentResult | None = None,
) -> GoldenCaseResult:
    golden_id = str(spec.get("id") or "unnamed")
    failures = _refuse_invented_gpu_numbers(spec)
    notes: list[str] = []
    expect = spec.get("expect") or {}
    null_guard = list(expect.get("null_must_not_become_zero") or [])
    stub = result or promotable_stub_result()

    if not spec.get("synthetic", True):
        failures.append(f"{golden_id}: goldens must be synthetic fixture/CPU rows")

    ledgers_by_role: dict[str, RequestLedger] = {}
    for raw_ledger in spec.get("ledgers") or []:
        ledger = ledger_from_explicit(raw_ledger)
        ledgers_by_role[raw_ledger["role"]] = ledger
        agg = recalculate_from_ledger(ledger)
        view = _aggregate_view(agg)
        wanted = (expect.get("aggregates") or {}).get(raw_ledger["role"])
        if wanted:
            failures.extend(
                _check_aggregate(
                    f"{golden_id}/{raw_ledger['role']}",
                    view,
                    wanted,
                    null_must_not_become_zero=null_guard,
                )
            )
        if expect.get("incomplete_is_not_success"):
            incomplete = sum(
                1 for rec in ledger.measured() if rec.outcome == RequestOutcome.INCOMPLETE
            )
            if incomplete and agg.successful_requests >= agg.total_requests:
                failures.append(
                    f"{golden_id}: incomplete packaged as success "
                    f"(successful={agg.successful_requests})"
                )
            if incomplete and agg.failed_requests < incomplete:
                failures.append(
                    f"{golden_id}: incomplete dropped from error denominator"
                )

    for check in spec.get("tpot_checks") or []:
        got = compute_tpot_ms(
            e2e_ms=check["e2e_ms"],
            ttft_ms=check["ttft_ms"],
            output_tokens=check["output_tokens"],
            token_count_source=check.get("token_count_source"),
        )
        if got != check.get("tpot_ms"):
            failures.append(
                f"{golden_id}: compute_tpot_ms({check}) -> {got!r}, "
                f"expected {check.get('tpot_ms')!r}"
            )
        if expect.get("tpot_never_zero") and got == 0:
            failures.append(f"{golden_id}: TPOT became 0 (must stay N/A)")

    decision = None
    if spec.get("pairs"):
        baseline, candidate = _ledgers_from_pairs(spec)
        decision = verdict_from_ledgers(
            baseline,
            candidate,
            metric=spec.get("metric") or "throughput_rps",
            phase=RepeatPhase(spec.get("phase") or "confirmation"),
            min_pairs=int(spec.get("min_pairs") or DEFAULT_MIN_PAIRS),
            min_rel_delta=float(spec.get("min_rel_delta") or DEFAULT_MIN_REL_DELTA),
        )
        if expect.get("verdict") and decision.verdict.value != expect["verdict"]:
            failures.append(
                f"{golden_id}: verdict={decision.verdict.value!r}, "
                f"expected {expect['verdict']!r}"
            )
        if expect.get("numeric_signal") and (
            decision.numeric_signal.value != expect["numeric_signal"]
        ):
            failures.append(
                f"{golden_id}: numeric_signal={decision.numeric_signal.value!r}, "
                f"expected {expect['numeric_signal']!r}"
            )
        if "search_winner" in expect and decision.search_winner != bool(
            expect["search_winner"]
        ):
            failures.append(
                f"{golden_id}: search_winner={decision.search_winner}, "
                f"expected {expect['search_winner']}"
            )
        if expect.get("reason") and decision.reason != expect["reason"]:
            failures.append(
                f"{golden_id}: reason={decision.reason!r}, expected {expect['reason']!r}"
            )
        if expect.get("reason_contains") and expect["reason_contains"] not in (
            decision.reason or ""
        ):
            failures.append(
                f"{golden_id}: reason {decision.reason!r} does not contain "
                f"{expect['reason_contains']!r}"
            )
        if decision.verdict.value == "confirmed_improvement" and spec.get("phase") == "search":
            failures.append(f"{golden_id}: search phase minted confirmed_improvement")

        if "confirmed_promotable" in expect:
            if not is_promotable(stub):
                failures.append(f"{golden_id}: stub result is not Week-1 promotable")
            got_gate = is_confirmed_promotable(stub, decision)
            if got_gate != bool(expect["confirmed_promotable"]):
                failures.append(
                    f"{golden_id}: is_confirmed_promotable={got_gate}, "
                    f"expected {expect['confirmed_promotable']}"
                )
            if got_gate and decision.search_winner:
                failures.append(
                    f"{golden_id}: search winner became confirmed_promotable "
                    "(measurement-trust loosened)"
                )

        reflect_expect = expect.get("reflect")
        if reflect_expect:
            conclusion = _reflect_conclusion(spec, decision)
            if conclusion.next_action != reflect_expect.get("next_action"):
                failures.append(
                    f"{golden_id}: Reflect next_action={conclusion.next_action!r}, "
                    f"expected {reflect_expect.get('next_action')!r}"
                )
            if conclusion.stop_reason != reflect_expect.get("stop_reason", ""):
                failures.append(
                    f"{golden_id}: Reflect stop_reason={conclusion.stop_reason!r}, "
                    f"expected {reflect_expect.get('stop_reason')!r}"
                )
            if conclusion.promote != bool(reflect_expect.get("promote")):
                failures.append(
                    f"{golden_id}: Reflect promote={conclusion.promote}, "
                    f"expected {reflect_expect.get('promote')}"
                )
            notes.append(
                f"reflect next_action={conclusion.next_action} "
                f"stop_reason={conclusion.stop_reason or '-'}"
            )

    if expect.get("confirmed_promotable") is False and decision is None:
        # Ledger-only goldens must not imply a confirm.
        notes.append("no confirmation pairs; confirm gate stays closed")

    return GoldenCaseResult(
        golden_id=golden_id,
        ok=not failures,
        failures=failures,
        notes=notes,
    )


def measurement_trust_gate(
    root: str | Path | None = None,
    *,
    result: ExperimentResult | None = None,
) -> MeasurementGateResult:
    """Deterministic CI gate. Empty / GPU-not-run is not a pass."""
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

    required = tuple(catalog.get("required_ids") or REQUIRED_GOLDEN_IDS)
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

    cases = [evaluate_golden(spec, result=result) for spec in specs]
    for case in cases:
        failures.extend(case.failures)

    # Week-1 gate must stay closed for these goldens (none are a confirmed promote).
    if any(
        (spec.get("expect") or {}).get("confirmed_promotable") is True for spec in specs
    ):
        failures.append(
            "this thin golden set must not claim confirmed_promotable=true; "
            "do not loosen ① / ⑤ gates"
        )

    return MeasurementGateResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        cases=cases,
        gpu_status=gpu_status,
    )
