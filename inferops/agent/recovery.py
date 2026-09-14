"""Attempt recovery / failure event — agent-local, not a metrics schema.

Tune ⑧ contract. Does not invent ④/⑤ fields, GPU numbers, or a parallel
metrics schema. Reflect reads this fact so a this-attempt failure cannot
be mistaken for ``experiment_summaries[-1]`` prior success.

Additive residual fields (Eval freeze surface — do not rewrite goldens):
``validity_status`` (Week-1 contract status or ""), ``retry_count``.
"""

from __future__ import annotations

from typing import Any

from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    config_knobs,
    derive_status,
    empty_latency,
)

# Control-flow signals that must never be treated as a successful tool result.
# ``TaskCancelled`` (bench_runner) is a user/graph cancel: propagate so the
# graph unwinds and ``run_agent`` / UI clean up owned processes.
_HARD_CONTROL_NAMES = frozenset(
    {"KeyboardInterrupt", "SystemExit", "GraphInterrupt", "NodeInterrupt", "TaskCancelled"}
)

# Receipt / ack loss after vLLM startup succeeded. Recover by id fact-check.
CODE_ACK_LOST = "ack_lost"
INCOMPLETE_NOTE_PREFIX = "incomplete:"
ACK_LOST_TOKENS = (
    "ack lost",
    "receipt lost",
    "receipt/ack lost",
    "ack_lost",
    "receipt_lost",
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
    # Additive residual ⑧ — Eval goldens already iterate this tuple.
    "validity_status",
    "retry_count",
)

TRAJECTORY_AUDIT_FIELDS = (
    "retry_count",
    "budget_consumed",
    "next_action",
    "stop_reason",
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


class AckLostError(Exception):
    """vLLM/startup succeeded but the completion receipt/ack was lost."""

    def __init__(
        self,
        message: str = "",
        *,
        experiment_id: str = "",
        result: Any = None,
        result_persisted: bool = False,
    ) -> None:
        super().__init__(message)
        self.experiment_id = experiment_id
        self.result = result
        self.result_persisted = bool(result_persisted)


def is_ack_lost(exc: BaseException) -> bool:
    """True when the tool reports startup-ok + lost receipt/ack."""
    if isinstance(exc, AckLostError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(token in text for token in ACK_LOST_TOKENS)


def is_failed_contract_row(result: Any) -> bool:
    return contract_status_value(result) == ExperimentValidityStatus.FAILED.value


def is_unconfirmable_contract_row(result: Any) -> bool:
    """True for the ① insufficient_evidence row we persist on ack-lost miss."""
    if contract_status_value(result) != ExperimentValidityStatus.INSUFFICIENT_EVIDENCE.value:
        return False
    notes = str(getattr(result, "notes", "") or "")
    return notes.startswith(INCOMPLETE_NOTE_PREFIX)


def keep_failed_recovery_semantics(result: Any, exc: BaseException | None = None) -> bool:
    """Fact-check reuse must not turn a failed/BenchmarkError row into success."""
    if result is not None and (
        is_failed_contract_row(result) or is_unconfirmable_contract_row(result)
    ):
        return True
    if (
        exc is not None
        and type(exc).__name__ == "BenchmarkError"
        and getattr(exc, "result", None) is not None
    ):
        return True
    return False


def contract_status_value(result: Any) -> str:
    """Week-1 validity status from a contract row (empty if none)."""
    if result is None:
        return ""
    status = getattr(result, "status", None)
    if status is None:
        return ""
    return status.value if hasattr(status, "value") else str(status)


def unconfirmable_contract_result(
    *,
    experiment_id: str,
    config: ExperimentConfig,
    session_id: str | None,
    reason: str,
) -> ExperimentResult:
    """Persistable ① row when completion cannot be confirmed.

    Uses ``derive_status`` (no evidence / no actual) → ``insufficient_evidence``.
    Does not invent a parallel ``incomplete`` validity enum. Missing metrics
    stay None. No GPU numbers.
    """
    requested = config_knobs(config)
    notes = (
        reason
        if reason.startswith(INCOMPLETE_NOTE_PREFIX)
        else f"{INCOMPLETE_NOTE_PREFIX} {reason}"
    )
    status = derive_status(
        failed=False,
        evidence=None,
        actual_config=None,
        requested_config=requested,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_requests=0,
        successful_requests=0,
        total_time_s=0.0,
        throughput_rps=None,
        tokens_per_second=None,
        error_rate=None,
        ttft=empty_latency(),
        tpot=empty_latency(),
        e2e_latency=empty_latency(),
        gpu_memory_used_gb=None,
        gpu_utilization_pct=None,
        cost_usd=None,
        session_id=session_id,
        requested_config=requested,
        actual_config=None,
        config_evidence=None,
        status=status,
        notes=notes,
    )


def trajectory_audit_fields(
    state: dict[str, Any] | None,
    *,
    budget_consumed: bool,
    next_action: str = "",
    stop_reason: str = "",
    retry_count: int | None = None,
) -> dict[str, Any]:
    """Auditable retry / budget / stop fields for executor + Reflect steps.

    Planner steps do not carry these fields (planner does not consume budget
    or decide stop / remasure). See ``reports/week3_interrupt_recovery.md``.
    """
    if retry_count is None:
        retry_count = int((state or {}).get("remeasure_count") or 0)
    return {
        "retry_count": int(retry_count),
        "budget_consumed": bool(budget_consumed),
        "next_action": next_action,
        "stop_reason": stop_reason,
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
    validity_status: str = "",
    retry_count: int = 0,
    incomplete: bool = False,
) -> dict[str, Any]:
    """Minimal auditable failure fact for this attempt."""
    event = {
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
        "validity_status": validity_status,
        "retry_count": int(retry_count),
    }
    if incomplete:
        # Additive marker — Week-1 status stays insufficient_evidence.
        event["incomplete"] = True
    return event


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
