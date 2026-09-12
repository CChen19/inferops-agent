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
from inferops.agent.state import (
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    ExperimentSummary,
    Hypothesis,
    is_duplicate,
    is_promotable_summary,
    pending_hypotheses,
    summary_from_result,
)
from inferops.bench_runner import BenchmarkError
from inferops.memory.db import get_result_by_id
from inferops.schemas import ExperimentResult, is_promotable
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
    primary_metric = WORKLOAD_PRIMARY_METRIC[state["workload_name"]]

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
        }
        return {
            "hypotheses": updated_hyps,
            "last_skip_reason": "duplicate_candidate",
            "last_skipped_hypothesis_id": hyp["id"],
            "last_result": None,
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
            updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", eid)
            return {"hypotheses": updated_hyps}

        # --- run_benchmark (production or eval stub at tool boundary) ---
        console.print(f"  executor: running {eid} ({hyp['param']}={hyp['value']}) …")
        try:
            bench_fn = _run_benchmark_override or run_benchmark
            bench_out = bench_fn(RunBenchmarkInput(
                experiment_id=eid,
                config_patch={hyp["param"]: hyp["value"]},
                workload_name=state["workload_name"],
                persist=True,
                session_id=state["session_prefix"],
            ))
        except BenchmarkError as exc:
            # P2-5: failed contract row already built/persisted — surface it in
            # experiment_summaries + trajectory so final_report can show the attempt.
            console.print(f"  [red]executor: benchmark failed ({exc})[/red]")
            updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", eid)
            patch: dict[str, Any] = {
                "hypotheses": updated_hyps,
                "tried_experiment_ids": state["tried_experiment_ids"] + [eid],
                "experiments_remaining": state["experiments_remaining"] - 1,
            }
            if exc.result is not None:
                baseline_primary = (
                    state["baseline_summary"][primary_metric]
                    if state["baseline_summary"] else 0.0
                )
                failed_summary = summary_from_result(
                    exc.result,
                    param_changed=hyp["param"],
                    value_changed=hyp["value"],
                    baseline_primary=baseline_primary,
                    primary_metric=primary_metric,
                    bottleneck="unknown",
                )
                # Prefer explicit exception message if notes empty
                if not failed_summary.get("failure_reason"):
                    failed_summary["failure_reason"] = str(exc)
                traj_step = {
                    "step": len(state["trajectory"]) + 1,
                    "node": "executor",
                    "workload": state["workload_name"],
                    "action": f"run_benchmark({hyp['param']}={hyp['value']})",
                    "experiment_id": eid,
                    "run_id": failed_summary.get("run_id"),
                    "validity_status": failed_summary.get("validity_status"),
                    "mlflow_run_id": failed_summary.get("mlflow_run_id"),
                    "reasoning": hyp["rationale"],
                    "result": {
                        "status": "failed",
                        "failure_reason": failed_summary.get("failure_reason", ""),
                        "promoted_to_best": False,
                    },
                }
                patch["experiment_summaries"] = (
                    state["experiment_summaries"] + [failed_summary]
                )
                patch["trajectory"] = state["trajectory"] + [traj_step]
                # best_summary unchanged — failed rows are never promotable
                patch["best_summary"] = state["best_summary"]
                patch["last_result"] = exc.result
            return patch
        except Exception as exc:
            console.print(f"  [red]executor: benchmark failed ({exc})[/red]")
            updated_hyps = _set_status(state["hypotheses"], hyp["id"], "failed", eid)
            return {
                "hypotheses": updated_hyps,
                "tried_experiment_ids": state["tried_experiment_ids"] + [eid],
                "experiments_remaining": state["experiments_remaining"] - 1,
            }
        bench_dict = bench_out.model_dump() if hasattr(bench_out, "model_dump") else {}
        result = _experiment_result_from_tool(eid, bench_out)

    # --- analyze_bottleneck ---
    bottleneck = "unknown"
    try:
        ba = analyze_bottleneck(AnalyzeBottleneckInput(experiment_id=eid))
        bottleneck = ba.bottleneck
    except Exception:
        pass

    # --- compare_experiments vs baseline ---
    vs_baseline_pct = 0.0
    if state["baseline_summary"] is not None:
        try:
            cmp = compare_experiments(CompareExperimentsInput(
                experiment_id_a=state["baseline_summary"]["experiment_id"],
                experiment_id_b=eid,
                metric=primary_metric,
                n_bootstrap=1000,
            ))
            # For "max" metrics: positive delta_pct = b is better
            vs_baseline_pct = cmp.delta_pct
        except Exception:
            pass

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
        # Prefer bootstrap comparison when available
        summary["vs_baseline_pct"] = round(vs_baseline_pct, 2)
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
            vs_baseline_pct=round(vs_baseline_pct, 2),
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
    status = "success" if vs_baseline_pct >= 0 else "failed"
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
        "result": {
            primary_metric: current_primary,
            "ttft_p99_ms": summary["ttft_p99_ms"],
            "bottleneck": bottleneck,
            "vs_baseline_pct": vs_baseline_pct,
            "promoted_to_best": False,
        },
    }

    console.print(
        f"  executor: done — {primary_metric}={current_primary:.3f}  "
        f"bottleneck={bottleneck}  vs_baseline={vs_baseline_pct:+.1f}%  "
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
        "trajectory":            state["trajectory"] + [traj_step],
        **stale_clear,
        **search_pack,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Per-slot runner: ``run_benchmark`` (or tool-boundary stub) → ④ ledger."""

    def _run_slot(eid: str, config: dict[str, Any]) -> Any:
        bench_fn = _run_benchmark_override or run_benchmark
        out = bench_fn(RunBenchmarkInput(
            experiment_id=eid,
            config_patch=config,
            workload_name=state["workload_name"],
            persist=True,
            session_id=state["session_prefix"],
        ))
        result = _experiment_result_from_tool(eid, out)
        if result is None:
            raise RuntimeError(
                f"confirmation slot {eid} produced no ExperimentResult"
            )
        return result

    return production_slot_run_arm(
        hypothesis=hyp,
        session_prefix=state["session_prefix"],
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

    try:
        campaign, decision = run_confirmation_campaign(
            run_arm,
            # Confirmation always uses the ⑤ default pair floor — do not inherit
            # a search-phase min_pairs=1 from the queued search winner.
            n_pairs=DEFAULT_MIN_PAIRS,
            metric=rl.get("metric") or primary_metric,
            phase=RepeatPhase.CONFIRMATION,
            expected_conditions=expected,
        )
    except Exception as exc:
        console.print(f"  [red]executor: confirmation campaign failed ({exc})[/red]")
        updated_hyps = _set_status(
            state["hypotheses"], hyp["id"], "failed", hyp.get("experiment_id")
        )
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
            "cited_run_ids": [],
            "result": {
                "status": "failed",
                "reason": "confirmation_campaign_failed",
                "promoted_to_best": False,
            },
        }
        return {
            "hypotheses": updated_hyps,
            "next_action": "remeasure",
            "confirmation_target": target,
            "confirmation_decision": None,
            "repeat_ledgers": None,
            "confirmation_bound_run_ids": None,
            "confirmation_blocked": True,
            "last_result": None,
            "last_skip_reason": "",
            "trajectory": state["trajectory"] + [traj_step],
        }

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
        "experiment_summaries": summaries,
        "last_skip_reason": "",
        "best_summary": state.get("best_summary"),
        "tried_experiment_ids": tried,
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
