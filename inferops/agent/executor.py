"""Executor node — picks one pending hypothesis, runs the experiment, updates state.

Deduplication: if a (param, value) pair was already tried, marks the hypothesis
as "skipped" and moves on without spending an experiment slot.

Tool call chain per hypothesis:
  1. propose_config_patch  — validates param/value against safe ranges
  2. run_benchmark         — starts vLLM, runs load, persists result to DB
  3. analyze_bottleneck    — classifies bottleneck from the stored result
  4. compare_experiments   — bootstrap CI vs baseline

Offline / real-graph eval may inject stubs ONLY at tool boundaries via
`tool_boundary_overrides` (including confirmation `run_arm`) — production
planner/executor/reflector nodes stay. Remeasure default is per-slot
`run_benchmark`, not “campaign unavailable”.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from rich.console import Console

from inferops.agent.confirm_campaign import (
    campaign_to_repeat_ledgers,
    candidate_fingerprint,
    clear_confirmation_fields,
    decision_binds_to_result,
    fingerprints_match,
    production_slot_run_arm,
    run_confirmation_campaign,
    search_winner_state_pack,
)
from inferops.agent.recovery import (
    CODE_ACK_LOST,
    STAGE_ANALYZE,
    STAGE_BENCHMARK,
    STAGE_COMPARE,
    STAGE_CONFIRM_CAMPAIGN,
    STAGE_CONFIRM_SLOT,
    STAGE_PROPOSE,
    AckLostError,
    clear_stale_attempt_fields,
    contract_status_value,
    is_ack_lost,
    is_failed_contract_row,
    is_unconfirmable_contract_row,
    keep_failed_recovery_semantics,
    recovery_event,
    reraise_hard_control,
    tool_unavailable,
    trajectory_audit_fields,
    unconfirmable_contract_result,
)
from inferops.agent.reflect_constraints import MAX_REMEASURES
from inferops.agent.state import (
    AgentState,
    ExperimentSummary,
    Hypothesis,
    is_duplicate,
    is_promotable_summary,
    model_name_of,
    pending_hypotheses,
    primary_metric_of,
    summary_from_result,
    task_of,
)
from inferops.task import task_conditions
from inferops.bench_runner import BenchmarkError
from inferops.memory.db import get_result_by_id, save_result
from inferops.schemas import ExperimentResult, ExperimentValidityStatus, is_promotable
from inferops.tools.analyze_bottleneck import AnalyzeBottleneckInput, analyze_bottleneck
from inferops.tools.compare_experiments import CompareExperimentsInput, compare_experiments
from inferops.tools.run_benchmark import RunBenchmarkInput, run_benchmark

console = Console()

# Optional eval stubs (tool edges only). None → production callables.
_run_benchmark_override: Callable[[RunBenchmarkInput], Any] | None = None
_propose_config_override: Callable[..., Any] | None = None
_confirmation_run_arm_override: Callable[..., Any] | None = None


@contextmanager
def tool_boundary_overrides(
    *,
    run_benchmark_fn: Callable[[RunBenchmarkInput], Any] | None = None,
    propose_config_fn: Callable[..., Any] | None = None,
    confirmation_run_arm_fn: Callable[..., Any] | None = None,
) -> Iterator[None]:
    """Temporarily replace benchmark / propose / confirmation-run_arm at tool edges."""
    global _run_benchmark_override, _propose_config_override, _confirmation_run_arm_override
    prev_bench, prev_propose = _run_benchmark_override, _propose_config_override
    prev_arm = _confirmation_run_arm_override
    if run_benchmark_fn is not None:
        _run_benchmark_override = run_benchmark_fn
    if propose_config_fn is not None:
        _propose_config_override = propose_config_fn
    if confirmation_run_arm_fn is not None:
        _confirmation_run_arm_override = confirmation_run_arm_fn
    try:
        yield
    finally:
        _run_benchmark_override = prev_bench
        _propose_config_override = prev_propose
        _confirmation_run_arm_override = prev_arm


@contextmanager
def confirmation_run_arm_override(run_arm: Callable[..., Any]) -> Iterator[None]:
    """Offline / fixture ⑤ run_arm — no GPU. Preferred CI confirmation path."""
    global _confirmation_run_arm_override
    prev = _confirmation_run_arm_override
    _confirmation_run_arm_override = run_arm
    try:
        yield
    finally:
        _confirmation_run_arm_override = prev


def _benchmark_input(
    state: AgentState,
    experiment_id: str,
    config_patch: dict[str, Any],
) -> RunBenchmarkInput:
    """Build a RunBenchmarkInput from the confirmed task when present."""
    task = task_of(state)
    return RunBenchmarkInput(
        experiment_id=experiment_id,
        config_patch=config_patch,
        workload_name=state["workload_name"],
        persist=True,
        session_id=state["session_prefix"],
        model_name=model_name_of(state),
        workload=task.workload if task is not None else None,
    )


def executor_node(state: AgentState) -> dict:
    """
    Pick the first pending hypothesis, run the experiment, and return a state patch.

    If the hypothesis is a duplicate, mark it "skipped" and return immediately
    without decrementing the budget.
    """
    pending = pending_hypotheses(state)
    if not pending:
        # Nothing to execute — signal planner to generate more
        return {}

    hyp = pending[0]
    primary_metric = primary_metric_of(state)

    # --- Deduplication check (remeasure of the same hyp is not a skip) ---
    remeasuring = state.get("next_action") == "remeasure"
    same_target = fingerprints_match(
        state.get("confirmation_target"), hyp["param"], hyp["value"]
    )
    if not (remeasuring and same_target):
        # New hyp / different candidate: drop stale ⑤ state so A cannot promote B.
        stale_clear = clear_confirmation_fields()
    else:
        stale_clear = {}

    if remeasuring:
        # Always a new ⑤ campaign for this hyp — do not re-apply stale_clear after.
        return _execute_confirmation_campaign(state, hyp, primary_metric)
    if not remeasuring and is_duplicate(state, hyp["param"], hyp["value"]):
        console.print(f"  [dim]executor: skip duplicate ({hyp['param']}={hyp['value']})[/dim]")
        updated_hyps = _set_status(state["hypotheses"], hyp["id"], "skipped", None)
        traj_step = {
            "step": len(state["trajectory"]) + 1,
            "node": "executor",
            "workload": state["workload_name"],
            "action": f"skip_duplicate({hyp['param']}={hyp['value']})",
            "hypothesis": {
                "id": hyp["id"],
                "param": hyp["param"],
                "value": hyp["value"],
                "text": hyp.get("rationale") or "",
            },
            "result": {"skip_reason": "duplicate_candidate", "promoted_to_best": False},
            **trajectory_audit_fields(
                state, budget_consumed=False, next_action="", stop_reason=""
            ),
        }
        return {
            "hypotheses": updated_hyps,
            "last_skip_reason": "duplicate_candidate",
            "last_skipped_hypothesis_id": hyp["id"],
            "last_result": None,
            "last_recovery": None,
            "trajectory": state["trajectory"] + [traj_step],
            **stale_clear,
        }

    # --- Build experiment ID ---
    remasure_n = int(state.get("remeasure_count") or 0)
    if remeasuring and remasure_n > 0:
        eid = f"{state['session_prefix']}{hyp['param']}_{hyp['value']}_r{remasure_n}"
    else:
        eid = f"{state['session_prefix']}{hyp['param']}_{hyp['value']}"

    # --- Check DB for existing result (in case of resume) ---
    existing = get_result_by_id(eid)
    if existing is not None:
        if keep_failed_recovery_semantics(existing):
            console.print(
                f"  [dim]executor: reused persisted failure {eid} "
                f"({contract_status_value(existing)})[/dim]"
            )
            return _failure_from_persisted_row(
                state,
                hyp,
                experiment_id=eid,
                row=existing,
                primary_metric=primary_metric,
                extra_clear=stale_clear,
            )
        console.print(f"  [dim]executor: loaded from DB: {eid}[/dim]")
        result = existing
        bench_dict = _result_to_bench_dict(existing)
    else:
        # --- propose_config_patch: validate ranges ---
        try:
            from inferops.tools.propose_config import ProposeConfigInput, propose_config_patch
            base_eid = state["baseline_summary"]["experiment_id"] if state["baseline_summary"] else ""
            propose_fn = _propose_config_override or propose_config_patch
            propose_fn(ProposeConfigInput(
                base_experiment_id=base_eid,
                param=hyp["param"],
                value=hyp["value"],
                rationale=hyp["rationale"],
                new_experiment_id=eid,
            ))
        except ValueError as exc:
            console.print(f"  [red]executor: propose rejected ({exc})[/red]")
            return _attempt_failure_patch(
                state,
                hyp,
                experiment_id=eid,
                stage=STAGE_PROPOSE,
                reason=str(exc),
                code="propose_rejected",
                result_persisted=False,
                budget_consumed=False,
                retryable=False,
                next_action="rollback",
                persist_result=None,
                primary_metric=primary_metric,
                extra_clear=stale_clear,
            )
        except Exception as exc:
            reraise_hard_control(exc)
            console.print(f"  [red]executor: propose failed ({exc})[/red]")
            return _attempt_failure_patch(
                state,
                hyp,
                experiment_id=eid,
                stage=STAGE_PROPOSE,
                reason=str(exc),
                code="propose_tool_error",
                result_persisted=False,
                budget_consumed=False,
                retryable=False,
                next_action="rollback",
                persist_result=None,
                primary_metric=primary_metric,
                extra_clear=stale_clear,
            )

        # --- run_benchmark (production or eval stub at tool boundary) ---
        console.print(f"  executor: running {eid} ({hyp['param']}={hyp['value']}) …")
        try:
            bench_fn = _run_benchmark_override or run_benchmark
            bench_out = bench_fn(_benchmark_input(
                state,
                eid,
                {hyp["param"]: hyp["value"]},
            ))
        except Exception as exc:
            reraise_hard_control(exc)
            recovered = _factcheck_persisted_attempt(eid)
            if recovered is not None:
                if keep_failed_recovery_semantics(recovered, exc):
                    console.print(
                        f"  [dim]executor: fact-check reused failed row {eid}[/dim]"
                    )
                    return _failure_from_persisted_row(
                        state,
                        hyp,
                        experiment_id=eid,
                        row=recovered,
                        exc=exc,
                        primary_metric=primary_metric,
                        extra_clear=stale_clear,
                    )
                # Startup-ok / receipt lost: reuse a completed row by id.
                # Do not re-start or re-benchmark.
                console.print(
                    f"  [dim]executor: ack/receipt lost — reused persisted {eid}[/dim]"
                )
                result = recovered
                bench_dict = _result_to_bench_dict(recovered)
            else:
                return _benchmark_unconfirmed_failure(
                    state,
                    hyp,
                    experiment_id=eid,
                    exc=exc,
                    primary_metric=primary_metric,
                    extra_clear=stale_clear,
                )
        else:
            bench_dict = bench_out.model_dump() if hasattr(bench_out, "model_dump") else {}
            result = _experiment_result_from_tool(eid, bench_out)

    # --- analyze_bottleneck (may degrade; never silent 0 / promotion change) ---
    bottleneck = "unknown"
    tool_status: dict[str, Any] = {}
    try:
        ba = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=eid))
        bottleneck = ba.bottleneck
        tool_status[STAGE_ANALYZE] = "ok"
    except Exception as exc:
        reraise_hard_control(exc)
        tool_status[STAGE_ANALYZE] = tool_unavailable(exc)

    # --- compare_experiments vs baseline ---
    vs_from_compare: float | None = None
    if state["baseline_summary"] is not None:
        try:
            cmp = compare_experiments(CompareExperimentsInput(
                experiment_id_a=state["baseline_summary"]["experiment_id"],
                experiment_id_b=eid,
                metric=primary_metric,
                n_bootstrap=1000,
            ))
            # For "max" metrics: positive delta_pct = b is better
            vs_from_compare = cmp.delta_pct
            tool_status[STAGE_COMPARE] = "ok"
        except Exception as exc:
            reraise_hard_control(exc)
            tool_status[STAGE_COMPARE] = tool_unavailable(exc)

    # --- Build ExperimentSummary (contract fields from result when present) ---
    baseline_primary = (
        state["baseline_summary"][primary_metric]
        if state["baseline_summary"] else 0.0
    )
    if result is not None:
        summary = summary_from_result(
            result,
            param_changed=hyp["param"],
            value_changed=hyp["value"],
            baseline_primary=baseline_primary,
            primary_metric=primary_metric,
            bottleneck=bottleneck,
        )
        # Prefer bootstrap comparison when available — never overwrite with silent 0.
        if vs_from_compare is not None:
            summary["vs_baseline_pct"] = round(vs_from_compare, 2)
    else:
        summary = ExperimentSummary(
            experiment_id=eid,
            param_changed=hyp["param"],
            value_changed=hyp["value"],
            throughput_rps=bench_dict.get("throughput_rps", 0.0),
            tokens_per_second=bench_dict.get("tokens_per_second", 0.0),
            ttft_p50_ms=bench_dict.get("ttft_p50_ms", 0.0),
            ttft_p99_ms=bench_dict.get("ttft_p99_ms", 0.0),
            e2e_p50_ms=bench_dict.get("e2e_p50_ms", 0.0),
            bottleneck=bottleneck,
            vs_baseline_pct=round(vs_from_compare, 2) if vs_from_compare is not None else None,
            run_id=bench_dict.get("run_id") or "",
            validity_status=bench_dict.get("status") or "insufficient_evidence",
            mlflow_run_id=bench_dict.get("mlflow_run_id"),
            has_config_evidence=False,
            promotable=False,
            failure_reason="",
            error_rate=bench_dict.get("error_rate"),
        )

    # --- Best is owned by Reflect (⑥) via is_confirmed_promotable ---
    # Search-phase scores / Week-1 is_promotable alone must not promote.
    current_primary = bench_dict.get(primary_metric, 0.0)
    new_best = state["best_summary"]
    week1_ok = (
        is_promotable(result) if result is not None else is_promotable_summary(summary)
    )
    if not week1_ok:
        console.print(
            f"  [yellow]executor: not promoting {eid} to best "
            f"(status={summary.get('validity_status')}, "
            f"evidence={summary.get('has_config_evidence')})[/yellow]"
        )
    else:
        console.print(
            f"  [dim]executor: search result recorded; Reflect owns promotion "
            f"(is_confirmed_promotable)[/dim]"
        )

    # --- Mark hypothesis done ---
    # Compare unavailable is not an exec fail; None vs must not look like 0-gain fail.
    if vs_from_compare is None:
        status = "success"
    else:
        status = "success" if vs_from_compare >= 0 else "failed"
    updated_hyps = _set_status(state["hypotheses"], hyp["id"], status, eid)

    # --- Trajectory ---
    traj_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "executor",
        "workload": state["workload_name"],
        "action": f"run_benchmark({hyp['param']}={hyp['value']})",
        "experiment_id": eid,
        "run_id": summary.get("run_id"),
        "validity_status": summary.get("validity_status"),
        "reasoning": hyp["rationale"],
        "tools": tool_status,
        "result": {
            primary_metric: current_primary,
            "ttft_p99_ms": summary["ttft_p99_ms"],
            "bottleneck": bottleneck,
            "vs_baseline_pct": summary.get("vs_baseline_pct"),
            "promoted_to_best": False,
            "tools": tool_status,
        },
        **trajectory_audit_fields(
            state,
            budget_consumed=True,
            next_action="",
            stop_reason="",
        ),
    }
    task = task_of(state)
    if task is not None:
        traj_step["task_conditions"] = task_conditions(task)

    vs_log = summary.get("vs_baseline_pct")
    vs_txt = f"{vs_log:+.1f}%" if vs_log is not None else "unavailable"
    primary_txt = (
        f"{current_primary:.3f}" if current_primary is not None else "n/a"
    )
    console.print(
        f"  executor: done — {primary_metric}={primary_txt}  "
        f"bottleneck={bottleneck}  vs_baseline={vs_txt}  "
        f"status={summary.get('validity_status')}"
    )

    # Genuine ⑤ search winner (④ ledgers on baseline + candidate) queues remasure.
    # No ledger → no invented search_winner (streak / metric-only path stays).
    # Overlay AFTER stale_clear so a new hyp's search pack is not wiped.
    search_pack = _search_winner_pack_if_ledgers(state, result, hyp, primary_metric)

    return {
        "hypotheses":            updated_hyps,
        "tried_experiment_ids":  state["tried_experiment_ids"] + [eid],
        "experiment_summaries":  state["experiment_summaries"] + [summary],
        "best_summary":          new_best,
        "current_bottleneck":    bottleneck,
        "experiments_remaining": state["experiments_remaining"] - 1,
        "last_result":           result,
        "last_skip_reason":      "",
        "last_recovery":         None,
        "confirmation_blocked":  False,
        "trajectory":            state["trajectory"] + [traj_step],
        **stale_clear,
        **search_pack,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attempt_failure_patch(
    state: AgentState,
    hyp: Hypothesis,
    *,
    experiment_id: str,
    stage: str,
    reason: str,
    code: str,
    result_persisted: bool,
    budget_consumed: bool,
    retryable: bool,
    next_action: str,
    persist_result: Any,
    primary_metric: str,
    extra_clear: dict[str, Any] | None = None,
    cited_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """This-attempt failure: emit recovery fact, never forge a success row.

    ``BenchmarkError`` with a persisted contract row keeps that row.
    Generic tool exceptions do not invent run_id / perf / GPU / ledger.
    Stale ``last_result`` / ⑤ bind fields are cleared so Reflect cannot
    treat a prior success as the current attempt.
    """
    event = recovery_event(
        experiment_id=experiment_id,
        hypothesis=hyp,
        stage=stage,
        reason=reason,
        code=code,
        result_persisted=result_persisted,
        budget_consumed=budget_consumed,
        retryable=retryable,
        next_action=next_action,
        cited_run_ids=cited_run_ids,
        validity_status=(
            contract_status_value(persist_result)
            or (
                ExperimentValidityStatus.INSUFFICIENT_EVIDENCE.value
                if code == CODE_ACK_LOST
                else ""
            )
        ),
        retry_count=int(state.get("remeasure_count") or 0),
        incomplete=code == CODE_ACK_LOST,
    )
    updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", experiment_id)
    patch: dict[str, Any] = {
        "hypotheses": updated_hyps,
        "last_recovery": event,
        "last_skip_reason": "",
        "best_summary": state.get("best_summary"),
        **clear_stale_attempt_fields(),
        **(extra_clear or {}),
    }
    if budget_consumed:
        patch["tried_experiment_ids"] = list(state["tried_experiment_ids"]) + [experiment_id]
        patch["experiments_remaining"] = int(state["experiments_remaining"]) - 1

    failed_summary = None
    if persist_result is not None:
        baseline_primary = (
            state["baseline_summary"][primary_metric]
            if state.get("baseline_summary") else 0.0
        )
        failed_summary = summary_from_result(
            persist_result,
            param_changed=hyp["param"],
            value_changed=hyp["value"],
            baseline_primary=baseline_primary,
            primary_metric=primary_metric,
            bottleneck="unknown",
        )
        if not failed_summary.get("failure_reason"):
            failed_summary["failure_reason"] = reason
        patch["experiment_summaries"] = list(state["experiment_summaries"]) + [failed_summary]
        patch["last_result"] = persist_result
        event["cited_run_ids"] = [
            rid for rid in (
                list(cited_run_ids or []) + [str(failed_summary.get("run_id") or "")]
            ) if rid
        ]
        patch["last_recovery"] = event
    else:
        # Do not append a forged summary / run_id / perf / GPU / ledger.
        patch["last_result"] = None

    traj_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "executor",
        "workload": state["workload_name"],
        "action": f"{stage}({hyp['param']}={hyp['value']})",
        "experiment_id": experiment_id,
        "run_id": (failed_summary or {}).get("run_id") if failed_summary else None,
        "validity_status": (
            (failed_summary or {}).get("validity_status")
            or event.get("validity_status")
            or "failed"
        ),
        "mlflow_run_id": (failed_summary or {}).get("mlflow_run_id") if failed_summary else None,
        "reasoning": hyp["rationale"],
        "recovery": event,
        "hypothesis": {
            "id": hyp["id"],
            "param": hyp["param"],
            "value": hyp["value"],
            "text": hyp.get("rationale") or "",
        },
        "result": {
            "status": "failed",
            "failure_reason": reason,
            "code": code,
            "result_persisted": result_persisted,
            "promoted_to_best": False,
            "this_attempt_failed": True,
        },
        **trajectory_audit_fields(
            state,
            budget_consumed=budget_consumed,
            next_action=next_action,
            stop_reason="",
        ),
    }
    patch["trajectory"] = list(state["trajectory"]) + [traj_step]
    return patch


def _factcheck_persisted_attempt(experiment_id: str) -> Any:
    """Lookup by stable attempt/experiment id — never invent a row."""
    if not experiment_id:
        return None
    return get_result_by_id(experiment_id)


def _unconfirmable_config(state: AgentState, hyp: Hypothesis, experiment_id: str):
    """Requested config snapshot for an unconfirmable ① row (no GPU numbers)."""
    from configs.search_space import make_configs
    from workloads.definitions import ALL_WORKLOADS

    workload = {w.name: w for w in ALL_WORKLOADS}[state["workload_name"]]
    task = task_of(state)
    if task is not None:
        workload = task.workload
    base = make_configs(workload, model_name=model_name_of(state))[0]
    update: dict[str, Any] = {"experiment_id": experiment_id}
    param = hyp.get("param")
    if param:
        update[str(param)] = hyp.get("value")
    return base.model_copy(update=update)


def _try_persist_unconfirmable(
    state: AgentState,
    hyp: Hypothesis,
    *,
    experiment_id: str,
    reason: str,
) -> tuple[ExperimentResult, bool]:
    """Write Week-1 insufficient_evidence. ``persisted`` is True only if save worked."""
    row = unconfirmable_contract_result(
        experiment_id=experiment_id,
        config=_unconfirmable_config(state, hyp, experiment_id),
        session_id=state.get("session_prefix"),
        reason=reason,
    )
    try:
        save_result(row)
    except Exception as exc:
        reraise_hard_control(exc)
        console.print(f"  [red]executor: unconfirmable persist failed ({exc})[/red]")
        return row, False
    return row, True


def _ack_lost_reason(exc: BaseException) -> str:
    return (
        f"incomplete: {exc}. startup succeeded but receipt/ack lost; "
        "cannot confirm completion"
    )


def _failure_from_persisted_row(
    state: AgentState,
    hyp: Hypothesis,
    *,
    experiment_id: str,
    row: Any,
    primary_metric: str,
    extra_clear: dict[str, Any] | None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    """Keep failed / unconfirmable contract semantics — never a success patch."""
    persist = row
    if persist is None and isinstance(exc, BenchmarkError):
        persist = exc.result
    unconfirmable = is_unconfirmable_contract_row(persist)
    code = CODE_ACK_LOST if unconfirmable else "benchmark_error"
    reason = str(getattr(persist, "notes", "") or "") or (
        _ack_lost_reason(exc) if exc and is_ack_lost(exc) else str(exc or "persisted failed contract row")
    )
    return _attempt_failure_patch(
        state,
        hyp,
        experiment_id=experiment_id,
        stage=STAGE_BENCHMARK,
        reason=reason,
        code=code,
        result_persisted=persist is not None,
        budget_consumed=True,
        retryable=False,
        next_action="rollback",
        persist_result=persist,
        primary_metric=primary_metric,
        extra_clear=extra_clear,
    )


def _benchmark_unconfirmed_failure(
    state: AgentState,
    hyp: Hypothesis,
    *,
    experiment_id: str,
    exc: BaseException,
    primary_metric: str,
    extra_clear: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bench error with no persisted row: ack-lost → explicit ① status; else legacy fail."""
    if is_ack_lost(exc):
        reason = _ack_lost_reason(exc)
        console.print(f"  [red]executor: ack/receipt lost, unconfirmable ({exc})[/red]")
        incomplete, persisted = _try_persist_unconfirmable(
            state, hyp, experiment_id=experiment_id, reason=reason
        )
        return _attempt_failure_patch(
            state,
            hyp,
            experiment_id=experiment_id,
            stage=STAGE_BENCHMARK,
            reason=reason,
            code=CODE_ACK_LOST,
            result_persisted=persisted,
            budget_consumed=True,
            retryable=False,
            next_action="rollback",
            persist_result=incomplete if persisted else None,
            primary_metric=primary_metric,
            extra_clear=extra_clear,
        )
    if isinstance(exc, BenchmarkError):
        console.print(f"  [red]executor: benchmark failed ({exc})[/red]")
        return _attempt_failure_patch(
            state,
            hyp,
            experiment_id=experiment_id,
            stage=STAGE_BENCHMARK,
            reason=str(exc),
            code="benchmark_error",
            result_persisted=exc.result is not None,
            budget_consumed=True,
            retryable=False,
            next_action="rollback",
            persist_result=exc.result,
            primary_metric=primary_metric,
            extra_clear=extra_clear,
        )
    console.print(f"  [red]executor: benchmark failed ({exc})[/red]")
    return _attempt_failure_patch(
        state,
        hyp,
        experiment_id=experiment_id,
        stage=STAGE_BENCHMARK,
        reason=str(exc),
        code="tool_exception",
        result_persisted=False,
        budget_consumed=True,
        retryable=False,
        next_action="rollback",
        persist_result=None,
        primary_metric=primary_metric,
        extra_clear=extra_clear,
    )


def _experiment_result_from_tool(experiment_id: str, bench_out: Any) -> Any:
    """Prefer the persisted row; accept an ExperimentResult returned at the tool edge."""
    result = get_result_by_id(experiment_id)
    if result is not None:
        return result
    if isinstance(bench_out, ExperimentResult):
        return bench_out
    return None


def _search_winner_pack_if_ledgers(
    state: AgentState,
    result: Any,
    hyp: Hypothesis,
    primary_metric: str,
) -> dict[str, Any]:
    """Record a ⑤ search winner only when both arms have real request ledgers."""
    from inferops.metrics import ledger_from_result

    if result is None or ledger_from_result(result) is None:
        return {}
    baseline_eid = (state.get("baseline_summary") or {}).get("experiment_id")
    baseline_result = get_result_by_id(baseline_eid) if baseline_eid else None
    return search_winner_state_pack(
        baseline=baseline_result,
        candidate=result,
        hyp=hyp,
        primary_metric=primary_metric,
    )


def _production_confirm_run_arm(state: AgentState, hyp: Hypothesis):
    """Per-slot runner: ``run_benchmark`` (or tool-boundary stub) → ④ ledger.

    Slot ids include remasure/attempt identity. An already-persisted slot
    row is reused — no second benchmark for that id.
    """
    remasure_n = int(state.get("remeasure_count") or 0)

    def _run_slot(eid: str, config: dict[str, Any]) -> Any:
        existing = get_result_by_id(eid)
        if existing is not None:
            console.print(f"  [dim]executor: confirmation slot reused: {eid}[/dim]")
            return existing
        bench_fn = _run_benchmark_override or run_benchmark
        try:
            out = bench_fn(_benchmark_input(state, eid, config))
        except Exception as exc:
            reraise_hard_control(exc)
            reused = _factcheck_persisted_attempt(eid)
            if reused is not None:
                if is_failed_contract_row(reused):
                    raise
                if is_unconfirmable_contract_row(reused):
                    raise AckLostError(
                        str(getattr(reused, "notes", "") or exc),
                        experiment_id=eid,
                        result=reused,
                        result_persisted=True,
                    ) from exc
                console.print(
                    f"  [dim]executor: confirmation slot ack-lost reuse: {eid}[/dim]"
                )
                return reused
            if is_ack_lost(exc):
                reason = _ack_lost_reason(exc)
                row, persisted = _try_persist_unconfirmable(
                    state, hyp, experiment_id=eid, reason=reason
                )
                raise AckLostError(
                    reason,
                    experiment_id=eid,
                    result=row if persisted else None,
                    result_persisted=persisted,
                ) from exc
            raise
        result = _experiment_result_from_tool(eid, out)
        if result is None:
            raise RuntimeError(
                f"confirmation slot {eid} produced no ExperimentResult"
            )
        return result

    return production_slot_run_arm(
        hypothesis=hyp,
        session_prefix=state["session_prefix"],
        remasure_count=remasure_n,
        run_slot=_run_slot,
    )


def _execute_confirmation_campaign(
    state: AgentState,
    hyp: Hypothesis,
    primary_metric: str,
) -> dict[str, Any]:
    """Drive ⑤ interleave + evaluate_campaign via production or fixture run_arm."""
    from inferops.metrics import DEFAULT_MIN_PAIRS, RepeatPhase

    run_arm = _confirmation_run_arm_override or _production_confirm_run_arm(state, hyp)
    expected = None
    rl = state.get("repeat_ledgers") or {}
    if rl.get("conditions") is not None:
        expected = rl["conditions"]
    elif rl.get("baseline"):
        expected = rl["baseline"][0].conditions

    target = candidate_fingerprint(hyp["param"], hyp["value"])
    completed_run_ids: list[str] = []
    failed_stage = STAGE_CONFIRM_CAMPAIGN

    def _collecting_run_arm(arm, slot):
        nonlocal failed_stage
        failed_stage = STAGE_CONFIRM_SLOT
        ledger = run_arm(arm, slot)
        rid = getattr(ledger, "run_id", None)
        if rid:
            completed_run_ids.append(str(rid))
        return ledger

    remasure_n = int(state.get("remeasure_count") or 0)
    budget_left = int(state.get("experiments_remaining") or 0)
    retryable = remasure_n < MAX_REMEASURES and budget_left > 1
    try:
        campaign, decision = run_confirmation_campaign(
            _collecting_run_arm,
            # Confirmation always uses the ⑤ default pair floor — do not inherit
            # a search-phase min_pairs=1 from the queued search winner.
            n_pairs=DEFAULT_MIN_PAIRS,
            metric=rl.get("metric") or primary_metric,
            phase=RepeatPhase.CONFIRMATION,
            expected_conditions=expected,
        )
    except Exception as exc:
        reraise_hard_control(exc)
        console.print(f"  [red]executor: confirmation campaign failed ({exc})[/red]")
        next_action = "remeasure" if retryable else "rollback"
        attempt_id = (
            f"{state['session_prefix']}confirm_{hyp['param']}_"
            f"{hyp['value']}_r{remasure_n}"
        )
        ack_lost = is_ack_lost(exc)
        persist_result = getattr(exc, "result", None) if ack_lost else None
        persisted = bool(getattr(exc, "result_persisted", False)) if ack_lost else False
        if ack_lost and not persisted:
            slot_eid = str(getattr(exc, "experiment_id", "") or attempt_id)
            row, persisted = _try_persist_unconfirmable(
                state, hyp, experiment_id=slot_eid, reason=_ack_lost_reason(exc)
            )
            persist_result = row if persisted else None
        if ack_lost:
            code = CODE_ACK_LOST
            validity = ExperimentValidityStatus.INSUFFICIENT_EVIDENCE.value
        else:
            code = (
                "confirmation_slot_failed"
                if failed_stage == STAGE_CONFIRM_SLOT
                else "confirmation_campaign_failed"
            )
            validity = ""
        event = recovery_event(
            experiment_id=str(
                getattr(persist_result, "experiment_id", None)
                or hyp.get("experiment_id")
                or ""
            ),
            attempt_id=attempt_id,
            hypothesis=hyp,
            stage=failed_stage,
            reason=str(exc),
            code=code,
            result_persisted=persisted,
            budget_consumed=True,
            retryable=retryable,
            next_action=next_action,
            cited_run_ids=list(completed_run_ids),
            retry_count=remasure_n,
            validity_status=validity,
            incomplete=ack_lost,
        )
        updated_hyps = _set_status(
            state["hypotheses"], hyp["id"], "failed", hyp.get("experiment_id")
        )
        summaries = list(state.get("experiment_summaries") or [])
        last_result = None
        if persisted and persist_result is not None:
            last_result = persist_result
            baseline_primary = (
                state["baseline_summary"][primary_metric]
                if state.get("baseline_summary") else 0.0
            )
            summaries = summaries + [
                summary_from_result(
                    persist_result,
                    param_changed=hyp["param"],
                    value_changed=hyp["value"],
                    baseline_primary=baseline_primary,
                    primary_metric=primary_metric,
                    bottleneck=state.get("current_bottleneck") or "unknown",
                )
            ]
        traj_step = {
            "step": len(state["trajectory"]) + 1,
            "node": "executor",
            "workload": state["workload_name"],
            "action": "confirmation_campaign",
            "hypothesis": {
                "id": hyp["id"],
                "param": hyp["param"],
                "value": hyp["value"],
                "text": hyp.get("rationale") or "",
            },
            "cited_run_ids": list(completed_run_ids),
            "recovery": event,
            "validity_status": validity or None,
            "result": {
                "status": "failed",
                "reason": code,
                "stage": failed_stage,
                "promoted_to_best": False,
                "this_attempt_failed": True,
            },
            **trajectory_audit_fields(
                state,
                budget_consumed=True,
                next_action=next_action,
                stop_reason="",
                retry_count=remasure_n,
            ),
        }
        patch = {
            "hypotheses": updated_hyps,
            "next_action": next_action,
            "confirmation_target": target if retryable else None,
            "confirmation_decision": None,
            "repeat_ledgers": None,
            "confirmation_bound_run_ids": None,
            "confirmation_blocked": True,
            "last_result": last_result,
            "last_recovery": event,
            "last_skip_reason": "",
            "best_summary": state.get("best_summary"),
            "experiments_remaining": max(0, budget_left - 1),
            "trajectory": state["trajectory"] + [traj_step],
        }
        if last_result is not None:
            patch["experiment_summaries"] = summaries
        return patch

    bound_ids = [lg.run_id for lg in campaign.candidate_ledgers]
    last_result = getattr(run_arm, "last_candidate_result", None)
    if last_result is None or not decision_binds_to_result(
        decision, last_result, bound_run_ids=bound_ids, bound_target=target
    ):
        prev = state.get("last_result")
        if decision_binds_to_result(
            decision, prev, bound_run_ids=bound_ids, bound_target=target
        ):
            last_result = prev
        else:
            last_result = None

    summaries = list(state.get("experiment_summaries") or [])
    hyp_eid = hyp.get("experiment_id")
    if last_result is not None:
        baseline_primary = (
            state["baseline_summary"][primary_metric]
            if state.get("baseline_summary") else 0.0
        )
        summary = summary_from_result(
            last_result,
            param_changed=hyp["param"],
            value_changed=hyp["value"],
            baseline_primary=baseline_primary,
            primary_metric=primary_metric,
            bottleneck=state.get("current_bottleneck") or "unknown",
        )
        summaries = summaries + [summary]
        hyp_eid = getattr(last_result, "experiment_id", None) or hyp_eid

    updated_hyps = _set_status(state["hypotheses"], hyp["id"], "success", hyp_eid)

    cited = []
    for lg in list(campaign.baseline_ledgers) + list(campaign.candidate_ledgers):
        if lg.run_id not in cited:
            cited.append(lg.run_id)

    traj_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "executor",
        "workload": state["workload_name"],
        "action": "confirmation_campaign",
        "hypothesis": {
            "id": hyp["id"],
            "param": hyp["param"],
            "value": hyp["value"],
            "text": hyp.get("rationale") or "",
        },
        "cited_run_ids": cited,
        "result": {
            "phase": decision.phase.value,
            "verdict": decision.verdict.value,
            "search_winner": decision.search_winner,
            "promoted_to_best": False,
        },
        **trajectory_audit_fields(
            state,
            budget_consumed=True,
            next_action="",
            stop_reason="",
            retry_count=remasure_n,
        ),
    }
    console.print(
        f"  executor: confirmation campaign — verdict={decision.verdict.value} "
        f"phase={decision.phase.value} pairs={decision.usable_pairs}/{decision.pair_count}"
    )
    tried = list(state["tried_experiment_ids"])
    if hyp_eid and hyp_eid not in tried:
        tried = tried + [hyp_eid]
    return {
        "hypotheses": updated_hyps,
        "confirmation_decision": decision,
        "repeat_ledgers": campaign_to_repeat_ledgers(
            campaign, metric=rl.get("metric") or primary_metric
        ),
        "confirmation_target": target,
        "confirmation_bound_run_ids": bound_ids,
        "confirmation_blocked": False,
        "last_result": last_result,
        "last_recovery": None,
        "experiment_summaries": summaries,
        "last_skip_reason": "",
        "best_summary": state.get("best_summary"),
        "tried_experiment_ids": tried,
        # Same attempt accounting as the confirmation-failure path.
        "experiments_remaining": max(0, budget_left - 1),
        "trajectory": state["trajectory"] + [traj_step],
    }


def _set_status(
    hypotheses: list[Hypothesis],
    hyp_id: str,
    status: str,
    experiment_id: str | None,
) -> list[Hypothesis]:
    return [
        {**h, "status": status, "experiment_id": experiment_id}
        if h["id"] == hyp_id else h
        for h in hypotheses
    ]


def _result_to_bench_dict(result) -> dict[str, Any]:
    status = getattr(result, "status", None)
    status_value = status.value if hasattr(status, "value") else (status or "insufficient_evidence")
    return {
        "throughput_rps":   result.throughput_rps,
        "tokens_per_second": result.tokens_per_second,
        "ttft_p50_ms":      result.ttft.p50,
        "ttft_p99_ms":      result.ttft.p99,
        "e2e_p50_ms":       result.e2e_latency.p50,
        "run_id":           getattr(result, "run_id", ""),
        "status":           status_value,
        "mlflow_run_id":    getattr(result, "mlflow_run_id", None),
        "error_rate":       getattr(result, "error_rate", None),
    }
