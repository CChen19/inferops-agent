"""Reflector node — judges each experiment and controls loop flow.

Decision logic is **deterministic code** (``reflect_constraints``):
  continue / remeasure / rollback / stop

  - Budget exhausted                         → stop
  - Duplicate candidate                      → continue (do not promote)
  - OOM / exec fail                          → rollback
  - invalid / insufficient_evidence          → rollback (fail-closed)
  - SLO breach or missing error_rate         → rollback (fail-closed)
  - ⑤ ``too_noisy``                          → remeasure (cap) or stop
  - ⑤ ``no_diff`` / ``regression``           → stop (no_reliable_improvement)
  - ⑤ search winner                          → remeasure (queue confirmation)
  - ⑤ ``confirmed_improvement``              → promote **only if**
                                               ``is_confirmed_promotable``
  - Search-only: improvement >5% resets streak;
    streak ≥ 3                               → stop (no_reliable_improvement)
  - Bottleneck type changed                  → skip remaining pending hyps

LLM (planner) may propose hypotheses / explanations. Reflect never lets an
LLM decide validity, SLO, budget, duplicate identity, or confirmation.

Routing (``route_after_reflector``):
  - stop / should_stop=True     → END
  - remeasure                   → executor (same hyp re-pended)
  - rollback / continue         → executor if pending else planner
"""

from __future__ import annotations

from typing import Any, Literal

from inferops.agent.confirm_campaign import (
    candidate_fingerprint,
    clear_confirmation_fields,
)
from inferops.agent.recovery import (
    current_attempt_latest,
    hypothesis_fact,
    trajectory_audit_fields,
)
from inferops.agent.reflect_constraints import (
    LLM_MUST_NOT_OWN,
    conclude_experiment,
    hypothesis_record,
    maybe_promote_best,
    optional_llm_explanation,
    resolve_confirmation,
    summarize_config_diff,
)
from inferops.agent.state import AgentState, WORKLOAD_PRIMARY_METRIC, pending_hypotheses

# Re-export legacy names so existing tests keep importing them.
from inferops.agent.reflect_constraints import (  # noqa: F401
    IMPROVEMENT_THRESHOLD_PCT as _IMPROVEMENT_THRESHOLD_PCT,
    MAX_STREAK as _MAX_STREAK,
)


def _latest_hypothesis(
    state: AgentState,
    latest: dict[str, Any] | None,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    hyps = list(state.get("hypotheses") or [])
    rec_hyp = (recovery or {}).get("hypothesis") if recovery else None
    if rec_hyp and rec_hyp.get("id"):
        for h in reversed(hyps):
            if h.get("id") == rec_hyp.get("id"):
                return h
    if latest:
        eid = latest.get("experiment_id")
        for h in reversed(hyps):
            if eid and h.get("experiment_id") == eid:
                return h
        param, value = latest.get("param_changed"), latest.get("value_changed")
        for h in reversed(hyps):
            if h.get("param") == param and str(h.get("value")) == str(value):
                return h
    skip_id = state.get("last_skipped_hypothesis_id")
    if skip_id:
        for h in hyps:
            if h.get("id") == skip_id:
                return h
    for h in reversed(hyps):
        if h.get("status") == "skipped":
            return h
    return hyps[-1] if hyps else None


def _repend_hypothesis(hypotheses: list[dict[str, Any]], hyp: dict[str, Any] | None) -> list:
    if not hyp:
        return hypotheses
    return [
        {**h, "status": "pending"} if h.get("id") == hyp.get("id") else h
        for h in hypotheses
    ]


def reflector_node(state: AgentState) -> dict:
    """Evaluate the most recent experiment and update control flags."""

    summaries = list(state.get("experiment_summaries") or [])
    recovery = state.get("last_recovery")
    latest = current_attempt_latest(summaries, recovery)
    this_attempt_failed = bool(recovery and recovery.get("this_attempt_failed"))
    # Budget-only stop is still recorded even with no summaries.
    if state["experiments_remaining"] <= 0:
        conclusion = conclude_experiment(
            experiments_remaining=state["experiments_remaining"],
            no_improvement_streak=state.get("no_improvement_streak") or 0,
            current_bottleneck=state.get("current_bottleneck") or "unknown",
            latest=latest,
            baseline=state.get("baseline_summary"),
            best=state.get("best_summary"),
            summaries=summaries,
            last_skip_reason=state.get("last_skip_reason") or "",
            remeasure_count=int(state.get("remeasure_count") or 0),
            last_result=state.get("last_result") if not this_attempt_failed or latest is not None else None,
            confirmation_decision=state.get("confirmation_decision"),
            repeat_ledgers=state.get("repeat_ledgers"),
            primary_metric=WORKLOAD_PRIMARY_METRIC[state["workload_name"]],
            bound_run_ids=state.get("confirmation_bound_run_ids"),
            bound_target=state.get("confirmation_target"),
            last_recovery=recovery,
        )
        return _apply_conclusion(state, conclusion, latest, recovery)

    if latest is None and not this_attempt_failed:
        return {}

    conclusion = conclude_experiment(
        experiments_remaining=state["experiments_remaining"],
        no_improvement_streak=state.get("no_improvement_streak") or 0,
        current_bottleneck=state.get("current_bottleneck") or "unknown",
        latest=latest,
        baseline=state.get("baseline_summary"),
        best=state.get("best_summary"),
        summaries=summaries,
        last_skip_reason=state.get("last_skip_reason") or "",
        remeasure_count=int(state.get("remeasure_count") or 0),
        last_result=state.get("last_result") if not this_attempt_failed or latest is not None else None,
        confirmation_decision=state.get("confirmation_decision"),
        repeat_ledgers=state.get("repeat_ledgers"),
        primary_metric=WORKLOAD_PRIMARY_METRIC[state["workload_name"]],
        bound_run_ids=state.get("confirmation_bound_run_ids"),
        bound_target=state.get("confirmation_target"),
        last_recovery=recovery,
    )
    return _apply_conclusion(state, conclusion, latest, recovery)


def _apply_conclusion(
    state: AgentState,
    conclusion,
    latest: dict[str, Any] | None,
    recovery: dict[str, Any] | None = None,
) -> dict:
    hyp = _latest_hypothesis(state, latest, recovery)
    current = state.get("best_summary") or state.get("baseline_summary")
    config_diff = summarize_config_diff(current, latest)

    updated_hyps = list(state.get("hypotheses") or [])
    if conclusion.skip_pending:
        updated_hyps = [
            {**h, "status": "skipped"} if h.get("status") == "pending" else h
            for h in updated_hyps
        ]
    if conclusion.next_action == "remeasure":
        updated_hyps = _repend_hypothesis(updated_hyps, hyp)

    new_best = state.get("best_summary")
    if conclusion.promote:
        decision = resolve_confirmation(
            repeat_ledgers=state.get("repeat_ledgers"),
            confirmation_decision=state.get("confirmation_decision"),
            metric=WORKLOAD_PRIMARY_METRIC[state["workload_name"]],
        )
        new_best, promoted = maybe_promote_best(
            state.get("last_result"),
            decision,
            latest,
            new_best,
            bound_run_ids=state.get("confirmation_bound_run_ids"),
            bound_target=state.get("confirmation_target"),
        )
        if not promoted:
            new_best = state.get("best_summary")

    remasure_count = int(state.get("remeasure_count") or 0)
    if conclusion.next_action == "remeasure":
        remasure_count += 1

    explanation = optional_llm_explanation(conclusion)
    # LLM_MUST_NOT_OWN is the documented split; explanation never overrides action.
    assert conclusion.next_action in ("continue", "remeasure", "rollback", "stop")
    assert not (LLM_MUST_NOT_OWN & {"explanation"})

    traj_step = {
        "step": len(state.get("trajectory") or []) + 1,
        "node": "reflector",
        "workload": state["workload_name"],
        "action": "reflect",
        "hypothesis": hypothesis_record(hyp) or hypothesis_fact((recovery or {}).get("hypothesis")),
        "config_diff": config_diff,
        "cited_run_ids": list(conclusion.cited_run_ids),
        "constraint_checks": dict(conclusion.constraint_checks),
        "next_action": conclusion.next_action,
        "stop_reason": conclusion.stop_reason,
        "reasoning": explanation,
        "recovery": dict(recovery) if recovery else None,
        "result": {
            "vs_baseline_pct": (latest or {}).get("vs_baseline_pct"),
            "streak": conclusion.no_improvement_streak,
            "bottleneck_switched": conclusion.skip_pending,
            "promoted_to_best": conclusion.promote,
            "confirmation": conclusion.confirmation,
            "this_attempt_failed": bool(recovery and recovery.get("this_attempt_failed")),
        },
        **trajectory_audit_fields(
            state,
            budget_consumed=bool((recovery or {}).get("budget_consumed")),
            next_action=conclusion.next_action,
            stop_reason=conclusion.stop_reason or "",
            retry_count=remasure_count,
        ),
    }

    patch: dict[str, Any] = {
        "no_improvement_streak": conclusion.no_improvement_streak,
        "should_stop": conclusion.should_stop,
        "stop_reason": conclusion.stop_reason,
        "next_action": conclusion.next_action,
        "hypotheses": updated_hyps,
        "trajectory": list(state.get("trajectory") or []) + [traj_step],
        "last_skip_reason": "",
        "last_skipped_hypothesis_id": "",
        "remeasure_count": remasure_count,
        "best_summary": new_best,
        # Consumed this cycle; the auditable copy lives on the trajectory step.
        "last_recovery": None,
    }
    # Stale ⑤ decisions must not outlive this candidate. Keep only while remasuring.
    if conclusion.next_action == "remeasure" and hyp:
        patch["confirmation_target"] = candidate_fingerprint(hyp.get("param"), hyp.get("value"))
    else:
        patch.update(clear_confirmation_fields())
        if conclusion.next_action != "remeasure":
            patch["remeasure_count"] = 0
    return patch


def route_after_reflector(
    state: AgentState,
) -> Literal["planner", "executor", "__end__"]:
    """Conditional edge: maps next_action onto the existing graph."""
    next_action = state.get("next_action") or (
        "stop" if state.get("should_stop") else "continue"
    )
    if state.get("should_stop") or next_action == "stop":
        return "__end__"
    if next_action == "remeasure":
        return "executor" if pending_hypotheses(state) else "planner"
    # continue / rollback: restore-current already applied; same as before
    if pending_hypotheses(state):
        return "executor"
    return "planner"
