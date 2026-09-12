"""Attempt recovery / failure event — agent-local, not a metrics schema.

Tune ⑧ contract. Does not invent ④/⑤ fields, GPU numbers, or a parallel
metrics schema. Reflect reads this fact so a this-attempt failure cannot
be mistaken for ``experiment_summaries[-1]`` prior success.
"""

from __future__ import annotations

from typing import Any

# Control-flow signals that must never be treated as a successful tool result.
_HARD_CONTROL_NAMES = frozenset(
    {"KeyboardInterrupt", "SystemExit", "GraphInterrupt", "NodeInterrupt"}
)

RECOVERY_FIELDS = (
    "attempt_id",
    "experiment_id",
    "hypothesis",
    "stage",
    "tool",
    "reason",
    "code",
    "result_persisted",
    "budget_consumed",
    "retryable",
    "next_action",
)

# stage == tool for the current single-tool-per-stage loop.
STAGE_PROPOSE = "propose_config"
STAGE_BENCHMARK = "run_benchmark"
STAGE_ANALYZE = "analyze_bottleneck"
STAGE_COMPARE = "compare_experiments"
STAGE_CONFIRM_SLOT = "confirmation_slot"
STAGE_CONFIRM_CAMPAIGN = "confirmation_campaign"


def is_hard_control_exception(exc: BaseException) -> bool:
    """True for process/graph interrupts that must propagate, never succeed."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return True
    return type(exc).__name__ in _HARD_CONTROL_NAMES


def reraise_hard_control(exc: BaseException) -> None:
    """Re-raise hard signals. Call at the top of every generic except."""
    if is_hard_control_exception(exc):
        raise


def hypothesis_fact(hyp: dict[str, Any] | None) -> dict[str, Any] | None:
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


def recovery_event(
    *,
    experiment_id: str,
    hypothesis: dict[str, Any] | None,
    stage: str,
    reason: str,
    code: str,
    result_persisted: bool,
    budget_consumed: bool,
    retryable: bool,
    next_action: str,
    attempt_id: str | None = None,
    cited_run_ids: list[str] | None = None,
    this_attempt_failed: bool = True,
    tool: str | None = None,
) -> dict[str, Any]:
    """Minimal auditable failure fact for this attempt."""
    return {
        "attempt_id": attempt_id or experiment_id,
        "experiment_id": experiment_id,
        "hypothesis": hypothesis_fact(hypothesis),
        "stage": stage,
        "tool": tool or stage,
        "reason": reason,
        "code": code,
        "result_persisted": bool(result_persisted),
        "budget_consumed": bool(budget_consumed),
        "retryable": bool(retryable),
        "next_action": next_action,
        "this_attempt_failed": bool(this_attempt_failed),
        "cited_run_ids": list(cited_run_ids or []),
    }


def tool_unavailable(exc: BaseException) -> dict[str, Any]:
    """Degraded tool status — never a silent 0 / invented metric."""
    return {
        "status": "unavailable",
        "error": type(exc).__name__,
        "reason": str(exc) or type(exc).__name__,
    }


def latest_is_this_attempt(
    latest: dict[str, Any] | None,
    recovery: dict[str, Any] | None,
) -> bool:
    """True only when summaries[-1] is this attempt's persisted contract row."""
    if not latest or not recovery:
        return False
    if not recovery.get("this_attempt_failed"):
        return True
    if not recovery.get("result_persisted"):
        return False
    rec_eid = str(recovery.get("experiment_id") or "")
    latest_eid = str(latest.get("experiment_id") or "")
    return bool(rec_eid) and rec_eid == latest_eid


def current_attempt_latest(
    summaries: list[dict[str, Any]],
    recovery: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Do not treat a prior success summary as the current failed attempt."""
    latest = summaries[-1] if summaries else None
    if not recovery or not recovery.get("this_attempt_failed"):
        return latest
    if latest_is_this_attempt(latest, recovery):
        return latest
    return None


def clear_stale_attempt_fields() -> dict[str, Any]:
    """Drop bind / ⑤ / last_result so a later candidate cannot inherit them."""
    return {
        "last_result": None,
        "confirmation_decision": None,
        "repeat_ledgers": None,
        "confirmation_target": None,
        "confirmation_bound_run_ids": None,
        "confirmation_blocked": True,
    }
