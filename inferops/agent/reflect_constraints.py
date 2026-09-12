"""Deterministic Reflect constraints — Tune ⑥ adapter over ① / ④ / ⑤.

Code owns: validity, SLO, budget, duplicate identity, confirmation verdicts,
and best-promotion eligibility. LLM must never decide these.

Does not invent a metrics schema or mint ``confirmed_improvement``.
Confirmation comes only from ``verdict_from_ledgers`` / ``evaluate_campaign``.
Promotion requires ``is_confirmed_promotable(result, decision)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from inferops.agent.confirm_campaign import (
    decision_applies_to_latest,
    decision_binds_to_result,
)
from inferops.metrics import (
    DEFAULT_MIN_PAIRS,
    DEFAULT_MIN_REL_DELTA,
    ConfirmationDecision,
    ConfirmationVerdict,
    RepeatPhase,
    is_confirmed_promotable,
    verdict_from_ledgers,
)

NextAction = Literal["continue", "remeasure", "rollback", "stop"]

# Agent-local SLO hook (④ ``error_rate`` only — not a new metrics schema).
# 5% matches ⑤ DEFAULT_MIN_REL_DELTA / the search improvement bar.
MAX_ERROR_RATE = 0.05


def slo_max_error_rate() -> float:
    raw = os.environ.get("INFEROPS_REFLECT_MAX_ERROR_RATE")
    if raw is None or raw == "":
        return MAX_ERROR_RATE
    return float(raw)
# Cap remasures so too_noisy cannot loop forever (⑤ default pair floor).
MAX_REMEASURES = DEFAULT_MIN_PAIRS
IMPROVEMENT_THRESHOLD_PCT = 5.0
MAX_STREAK = 3

LLM_MAY_PROPOSE = frozenset({"hypothesis_text", "explanation"})
LLM_MUST_NOT_OWN = frozenset(
    {
        "validity_status",
        "slo_ok",
        "budget",
        "is_duplicate",
        "confirmation_verdict",
        "confirmed_promotable",
        "next_action",
    }
)


@dataclass(frozen=True)
class ReflectConclusion:
    """Pure experiment conclusion. Control flow reads this; LLM does not."""

    next_action: NextAction
    should_stop: bool
    stop_reason: str
    no_improvement_streak: int
    skip_pending: bool
    promote: bool
    constraint_checks: dict[str, Any]
    confirmation: dict[str, Any] | None
    reason: str
    cited_run_ids: list[str] = field(default_factory=list)


def check_slo(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic SLO. Missing ``error_rate`` is fail-closed (not SLO-ok)."""
    if not summary:
        return {"ok": False, "error_rate": None, "reason": "no_summary"}
    if "error_rate" not in summary or summary.get("error_rate") is None:
        return {"ok": False, "error_rate": None, "reason": "error_rate_missing_fail_closed"}
    err = summary.get("error_rate")
    limit = slo_max_error_rate()
    if err > limit:
        return {"ok": False, "error_rate": err, "reason": "error_rate_exceeds_slo"}
    return {"ok": True, "error_rate": err, "reason": "slo_ok"}


def is_candidate_summary(summary: dict[str, Any] | None) -> bool:
    return bool(summary) and summary.get("param_changed") not in (None, "")


def needs_validity_rollback(summary: dict[str, Any] | None) -> bool:
    """invalid / insufficient_evidence (and unknown-on-candidate) are fail-closed."""
    if not is_candidate_summary(summary):
        return False
    status = str((summary or {}).get("validity_status") or "insufficient_evidence")
    return status in {"invalid", "insufficient_evidence"}


def is_exec_fail(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    status = str(summary.get("validity_status") or "")
    if status == "failed":
        return True
    reason = str(summary.get("failure_reason") or "")
    return any(token in reason.upper() for token in ("OOM", "EXEC FAIL", "BENCHMARK FAILED"))


def is_duplicate_candidate(
    summaries: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    *,
    last_skip_reason: str = "",
) -> bool:
    """Duplicate identity comes from executor skip — not from remasure repeats.

    Confirmation remasures reuse ``(param, value)`` with new ``run_id``s. Counting
    those summaries as duplicates would steal the ⑤ verdict.
    """
    _ = (summaries, latest)
    return last_skip_reason == "duplicate_candidate"


def summarize_config_diff(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    param = (candidate or {}).get("param_changed")
    new_val = (candidate or {}).get("value_changed")
    old_val: Any = None
    if current is not None:
        if current.get("param_changed") == param and param is not None:
            old_val = current.get("value_changed")
        elif current.get("param_changed") is None:
            old_val = "baseline"
    if not param:
        text = "(no candidate change)"
    else:
        text = f"{param}: {old_val} → {new_val}"
    return {"param": param, "from": old_val, "to": new_val, "summary": text}


def hypothesis_record(hyp: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hyp:
        return None
    return {
        "id": hyp.get("id"),
        "param": hyp.get("param"),
        "value": hyp.get("value"),
        "text": hyp.get("rationale") or "",
        "status": hyp.get("status"),
        "experiment_id": hyp.get("experiment_id"),
    }


def cited_run_ids(
    *summaries: dict[str, Any] | None,
    decision: ConfirmationDecision | None = None,
    last_result: Any = None,
) -> list[str]:
    ids: list[str] = []
    for s in summaries:
        rid = (s or {}).get("run_id") if isinstance(s, dict) else None
        if rid:
            ids.append(str(rid))
    if last_result is not None:
        rid = getattr(last_result, "run_id", None)
        if rid:
            ids.append(str(rid))
    if decision is not None:
        for pair in decision.pairs:
            ids.append(pair.baseline_run_id)
            ids.append(pair.candidate_run_id)
    out: list[str] = []
    for rid in ids:
        if rid and rid not in out:
            out.append(rid)
    return out


def serialize_confirmation(decision: ConfirmationDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "phase": decision.phase.value,
        "verdict": decision.verdict.value,
        "numeric_signal": decision.numeric_signal.value,
        "search_winner": bool(decision.search_winner),
        "reason": decision.reason,
        "usable_pairs": decision.usable_pairs,
        "pair_count": decision.pair_count,
        "metric": decision.metric,
        "median_rel_delta": decision.median_rel_delta,
    }


def resolve_confirmation(
    *,
    repeat_ledgers: dict[str, Any] | None,
    confirmation_decision: ConfirmationDecision | None,
    metric: str,
) -> ConfirmationDecision | None:
    """Compute a ⑤ verdict from ledgers when present; never mint confirm here."""
    if repeat_ledgers:
        baseline = list(repeat_ledgers.get("baseline") or [])
        candidate = list(repeat_ledgers.get("candidate") or [])
        if baseline and candidate:
            phase = repeat_ledgers.get("phase") or RepeatPhase.SEARCH
            if isinstance(phase, str):
                phase = RepeatPhase(phase)
            return verdict_from_ledgers(
                baseline,
                candidate,
                metric=repeat_ledgers.get("metric") or metric,
                phase=phase,
                min_pairs=int(repeat_ledgers.get("min_pairs") or DEFAULT_MIN_PAIRS),
                min_rel_delta=float(
                    repeat_ledgers.get("min_rel_delta") or DEFAULT_MIN_REL_DELTA
                ),
            )
    return confirmation_decision


def maybe_promote_best(
    result: Any,
    decision: ConfirmationDecision | None,
    candidate: dict[str, Any] | None,
    current_best: dict[str, Any] | None,
    *,
    bound_run_ids: list[str] | None = None,
    bound_target: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Best updates only through bind + ``is_confirmed_promotable``.

    A ⑤ decision computed for candidate A cannot promote candidate B.
    """
    if candidate is None:
        return current_best, False
    if not decision_binds_to_result(
        decision,
        result,
        candidate=candidate,
        bound_run_ids=bound_run_ids,
        bound_target=bound_target,
    ):
        return current_best, False
    if not is_confirmed_promotable(result, decision):
        return current_best, False
    promoted = dict(candidate)
    promoted["confirmed_promotable"] = True
    return promoted, True


def _confirmation_fields(
    decision: ConfirmationDecision | None,
    *,
    result: Any,
    promoted: bool,
) -> dict[str, Any]:
    return {
        "phase": decision.phase.value if decision else None,
        "verdict": decision.verdict.value if decision else None,
        "search_winner": bool(decision.search_winner) if decision else False,
        "confirmed_promotable": bool(promoted),
        "gate": "is_confirmed_promotable",
        "gate_result": bool(is_confirmed_promotable(result, decision)),
    }


def conclude_experiment(
    *,
    experiments_remaining: int,
    no_improvement_streak: int,
    current_bottleneck: str,
    latest: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    best: dict[str, Any] | None,
    summaries: list[dict[str, Any]],
    last_skip_reason: str = "",
    remeasure_count: int = 0,
    last_result: Any = None,
    confirmation_decision: ConfirmationDecision | None = None,
    repeat_ledgers: dict[str, Any] | None = None,
    primary_metric: str = "throughput_rps",
    bound_run_ids: list[str] | None = None,
    bound_target: dict[str, Any] | None = None,
    last_recovery: dict[str, Any] | None = None,
) -> ReflectConclusion:
    """Priority-ordered Reflect rules. Deterministic; no LLM."""

    decision = resolve_confirmation(
        repeat_ledgers=repeat_ledgers,
        confirmation_decision=confirmation_decision,
        metric=primary_metric,
    )
    if decision is not None and not decision_applies_to_latest(
        decision,
        last_result=last_result,
        latest=latest,
        bound_run_ids=bound_run_ids,
        bound_target=bound_target,
    ):
        decision = None
    slo = check_slo(latest) if is_candidate_summary(latest) else {
        "ok": True,
        "error_rate": (latest or {}).get("error_rate") if latest else None,
        "reason": "baseline_or_no_candidate",
    }
    duplicate = is_duplicate_candidate(
        summaries, latest, last_skip_reason=last_skip_reason
    )
    exec_fail = is_exec_fail(latest)
    validity_rollback = needs_validity_rollback(latest)
    validity = str((latest or {}).get("validity_status") or "unknown")
    budget = int(experiments_remaining)
    _, would_promote = maybe_promote_best(
        last_result,
        decision,
        latest,
        best,
        bound_run_ids=bound_run_ids,
        bound_target=bound_target,
    )
    run_ids = cited_run_ids(baseline, latest, best, decision=decision, last_result=last_result)
    this_attempt_failed = bool(last_recovery and last_recovery.get("this_attempt_failed"))
    if this_attempt_failed:
        for rid in last_recovery.get("cited_run_ids") or []:
            if rid and rid not in run_ids:
                run_ids.append(str(rid))

    checks = {
        "validity_status": validity,
        "slo_ok": bool(slo["ok"]),
        "slo": slo,
        "budget_remaining": budget,
        "is_duplicate": duplicate,
        "exec_fail": exec_fail,
        "validity_rollback": validity_rollback,
        "remeasure_count": remeasure_count,
        "this_attempt_failed": this_attempt_failed,
        "recovery_stage": (last_recovery or {}).get("stage"),
        "recovery_code": (last_recovery or {}).get("code"),
        "recovery_result_persisted": bool((last_recovery or {}).get("result_persisted")),
        **_confirmation_fields(decision, result=last_result, promoted=False),
    }

    def _done(
        action: NextAction,
        *,
        stop: bool,
        stop_reason: str = "",
        streak: int | None = None,
        skip_pending: bool = False,
        promote: bool = False,
        reason: str,
    ) -> ReflectConclusion:
        checks["confirmed_promotable"] = bool(promote)
        checks["gate_result"] = bool(is_confirmed_promotable(last_result, decision))
        return ReflectConclusion(
            next_action=action,
            should_stop=stop,
            stop_reason=stop_reason,
            no_improvement_streak=no_improvement_streak if streak is None else streak,
            skip_pending=skip_pending,
            promote=promote,
            constraint_checks=dict(checks),
            confirmation=serialize_confirmation(decision),
            reason=reason,
            cited_run_ids=run_ids,
        )

    # 1. Duplicate candidate (executor skip or repeated (param, value))
    if duplicate:
        return _done(
            "continue",
            stop=False,
            reason="duplicate_candidate",
        )

    # 2. This-attempt failure — never read a prior success summary as current.
    # Confirmation-slot failures may remasure only while under remasure/budget cap.
    if this_attempt_failed:
        retryable = (
            bool(last_recovery.get("retryable"))
            and remeasure_count < MAX_REMEASURES
            and budget > 0
        )
        if retryable:
            return _done(
                "remeasure",
                stop=False,
                reason="attempt_failed_retryable",
            )
        streak = no_improvement_streak + 1
        if streak >= MAX_STREAK:
            return _done(
                "stop",
                stop=True,
                stop_reason="no_reliable_improvement",
                streak=streak,
                reason="attempt_failed_streak",
            )
        return _done(
            "rollback",
            stop=False,
            streak=streak,
            reason="attempt_failed",
        )

    # 3. OOM / exec fail
    if exec_fail:
        streak = no_improvement_streak + 1
        if streak >= MAX_STREAK:
            return _done(
                "stop",
                stop=True,
                stop_reason="no_reliable_improvement",
                streak=streak,
                reason="exec_fail_streak",
            )
        return _done(
            "rollback",
            stop=False,
            streak=streak,
            reason="oom_or_exec_fail",
        )

    # 3b. invalid / insufficient_evidence — fail-closed rollback
    if validity_rollback:
        streak = no_improvement_streak + 1
        if streak >= MAX_STREAK:
            return _done(
                "stop",
                stop=True,
                stop_reason="no_reliable_improvement",
                streak=streak,
                reason="validity_fail_closed_streak",
            )
        return _done(
            "rollback",
            stop=False,
            streak=streak,
            reason="validity_fail_closed",
        )

    # 4. SLO breach (missing error_rate is fail-closed)
    if not slo["ok"]:
        streak = no_improvement_streak + 1
        if streak >= MAX_STREAK:
            return _done(
                "stop",
                stop=True,
                stop_reason="no_reliable_improvement",
                streak=streak,
                reason="slo_breach_streak",
            )
        return _done(
            "rollback",
            stop=False,
            streak=streak,
            reason="slo_breach",
        )

    # 4b. Budget — only after fail-closed validity / SLO. A last-slot
    # confirmation may still promote; SLO / validity cannot be skipped.
    if budget <= 0:
        if would_promote:
            return _done(
                "stop",
                stop=True,
                stop_reason="budget_exhausted",
                streak=0,
                promote=True,
                reason="confirmed_promotable",
            )
        return _done(
            "stop",
            stop=True,
            stop_reason="budget_exhausted",
            reason="experiments_remaining<=0",
        )

    # 5. Confirmation (⑤) when a decision exists
    if decision is not None:
        verdict = decision.verdict
        if verdict == ConfirmationVerdict.TOO_NOISY:
            if remeasure_count < MAX_REMEASURES:
                return _done(
                    "remeasure",
                    stop=False,
                    reason="too_noisy_queue_remeasure",
                )
            return _done(
                "stop",
                stop=True,
                stop_reason="too_noisy",
                reason="too_noisy_remeasure_cap",
            )
        if verdict in (ConfirmationVerdict.NO_DIFF, ConfirmationVerdict.REGRESSION):
            if decision.search_winner:
                if remeasure_count < MAX_REMEASURES:
                    return _done(
                        "remeasure",
                        stop=False,
                        reason="search_winner_unconfirmed",
                    )
                return _done(
                    "stop",
                    stop=True,
                    stop_reason="no_reliable_improvement",
                    reason="search_winner_unconfirmed_cap",
                )
            return _done(
                "stop",
                stop=True,
                stop_reason="no_reliable_improvement",
                reason=f"confirmation_{verdict.value}",
            )
        if verdict == ConfirmationVerdict.CONFIRMED_IMPROVEMENT:
            if would_promote:
                return _done(
                    "continue",
                    stop=False,
                    streak=0,
                    promote=True,
                    reason="confirmed_promotable",
                )
            # Legal ⑤ confirm numbers but Week-1 gate failed (unevidenced / forged).
            return _done(
                "continue",
                stop=False,
                streak=no_improvement_streak + 1,
                reason="confirm_failed_week1_or_forged",
            )

    # 6. Search-only streak (no ⑤ decision) + bottleneck switch
    improvement = (latest or {}).get("vs_baseline_pct")
    if improvement is not None and improvement >= IMPROVEMENT_THRESHOLD_PCT:
        new_streak = 0
    else:
        new_streak = no_improvement_streak + 1

    if new_streak >= MAX_STREAK:
        return _done(
            "stop",
            stop=True,
            stop_reason="no_reliable_improvement",
            streak=new_streak,
            reason="no_improvement_streak",
        )

    new_bottleneck = (latest or {}).get("bottleneck", "unknown")
    bottleneck_switched = (
        new_bottleneck not in ("unknown", current_bottleneck)
        and current_bottleneck not in ("unknown",)
    )
    return _done(
        "continue",
        stop=False,
        streak=new_streak,
        skip_pending=bottleneck_switched,
        reason=(
            "bottleneck_switched"
            if bottleneck_switched
            else f"search_streak={new_streak}"
        ),
    )


def optional_llm_explanation(conclusion: ReflectConclusion) -> str:
    """Deterministic explanation string. Not an LLM; never authoritative.

    An optional LLM side-channel may paraphrase ``reason`` / hypothesis text
    only. It must not change ``next_action`` or any ``LLM_MUST_NOT_OWN`` field.
    """
    return (
        f"next_action={conclusion.next_action} reason={conclusion.reason} "
        f"checks={{{', '.join(f'{k}={conclusion.constraint_checks.get(k)}' for k in ('validity_status', 'slo_ok', 'is_duplicate', 'verdict', 'budget_remaining'))}}}"
    )
