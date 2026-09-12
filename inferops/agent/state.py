"""AgentState definition and helpers for the Phase 4 optimizer agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------

class Hypothesis(TypedDict):
    id: str           # "h1", "h2", …
    param: str        # e.g. "max_num_batched_tokens"
    value: Any        # e.g. 4096 or True
    rationale: str    # LLM reasoning — must cite metric evidence
    status: str       # "pending" | "running" | "success" | "failed" | "skipped"
    experiment_id: str | None


class ExperimentSummary(TypedDict):
    experiment_id: str
    param_changed: str | None   # None for baseline
    value_changed: Any
    throughput_rps: float | None
    tokens_per_second: float | None
    ttft_p50_ms: float | None
    ttft_p99_ms: float | None
    e2e_p50_ms: float | None
    bottleneck: str
    vs_baseline_pct: float | None  # None if primary metric missing; never invent 0 gain
    # Week-1 contract fields (required for best-candidate gating)
    run_id: str
    validity_status: str        # valid | invalid | failed | insufficient_evidence
    mlflow_run_id: str | None
    has_config_evidence: bool
    # Single full-gate result — MUST be set via is_promotable(result), never guessed
    promotable: bool
    failure_reason: str  # notes / error from failed contract rows ("" if none)
    error_rate: float | None  # ④ field; Reflect SLO reads this — never invent 0


# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    workload_name: str
    session_prefix: str          # all experiment_ids share this prefix

    # Hypothesis stack
    hypotheses: list[Hypothesis]

    # Experiment tracking
    tried_experiment_ids: list[str]
    experiment_summaries: list[ExperimentSummary]
    baseline_summary: ExperimentSummary | None
    best_summary: ExperimentSummary | None

    # Bottleneck
    current_bottleneck: str      # "compute-bound" | "memory-bound" | "scheduling-bound" | "kv-bound" | "unknown"

    # Budget & control
    experiments_remaining: int
    no_improvement_streak: int
    should_stop: bool
    stop_reason: str
    next_action: str             # continue | remeasure | rollback | stop
    last_skip_reason: str        # executor → reflector (e.g. duplicate_candidate)
    last_skipped_hypothesis_id: str
    remeasure_count: int
    last_result: Any             # latest ExperimentResult (promotion gate input)
    confirmation_decision: Any   # ConfirmationDecision | None — never hand-minted
    repeat_ledgers: dict[str, Any] | None
    confirmation_target: dict[str, Any] | None  # {param, value} the ⑤ campaign is bound to
    confirmation_bound_run_ids: list[str] | None  # candidate run_ids from that campaign
    confirmation_blocked: bool   # True after a confirmation-slot / attempt failure
    last_recovery: dict[str, Any] | None  # ⑧ this-attempt failure fact (not a metrics schema)

    # Trajectory (for eval/judge in Phase 3 eval framework)
    trajectory: list[dict[str, Any]]

    # LangGraph message history (Planner read + write)
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC as _WPM
WORKLOAD_PRIMARY_METRIC: dict[str, str] = {k: v for k, (v, _) in _WPM.items()}

WORKLOAD_DESCRIPTIONS: dict[str, str] = {
    "chat_short": (
        "60 requests · concurrency=16 · 128-token prompts · 128-token outputs. "
        "Tests scheduler throughput on short, uniform sequences."
    ),
    "long_context_qa": (
        "20 requests · concurrency=4 · 1024-token prompts · 256-token outputs. "
        "Stresses prefill efficiency and KV cache memory pressure."
    ),
    "high_concurrency_short_out": (
        "120 requests · concurrency=32 · 64-token prompts · 32-token outputs. "
        "Maximum scheduler concurrency stress with very short sequences."
    ),
    "long_generation": (
        "10 requests · concurrency=2 · 256-token prompts · 512-token outputs. "
        "Stresses decode-phase KV cache and autoregressive throughput."
    ),
    "mixed_traffic": (
        "40 requests · concurrency=8 · 50% short (64-tok) + 50% long (512-tok) prompts. "
        "Tests scheduler fairness between short and long sequences."
    ),
}

# Search space exposed to the agent (keeps it within safe RTX 3060 bounds)
AGENT_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_num_batched_tokens": [2048, 3072, 4096],
    "max_num_seqs":           [64, 128, 256],
    "enable_chunked_prefill": [False, True],
    "enable_prefix_caching":  [False, True],
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def initial_state(
    workload_name: str,
    session_prefix: str,
    max_experiments: int = 8,
) -> AgentState:
    return {
        "workload_name":         workload_name,
        "session_prefix":        session_prefix,
        "hypotheses":            [],
        "tried_experiment_ids":  [],
        "experiment_summaries":  [],
        "baseline_summary":      None,
        "best_summary":          None,
        "current_bottleneck":    "unknown",
        "experiments_remaining": max_experiments,
        "no_improvement_streak": 0,
        "should_stop":           False,
        "stop_reason":           "",
        "next_action":           "continue",
        "last_skip_reason":      "",
        "last_skipped_hypothesis_id": "",
        "remeasure_count":       0,
        "last_result":           None,
        "confirmation_decision": None,
        "repeat_ledgers":        None,
        "confirmation_target":   None,
        "confirmation_bound_run_ids": None,
        "confirmation_blocked":  False,
        "last_recovery":         None,
        "trajectory":            [],
        "messages":              [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summary_from_result(
    result,                    # ExperimentResult
    param_changed: str | None,
    value_changed: Any,
    baseline_primary: float,
    primary_metric: str,
    bottleneck: str = "unknown",
) -> ExperimentSummary:
    from inferops.schemas import (
        ExperimentValidityStatus,
        has_critical_config_evidence,
        is_promotable,
    )

    primary_val = getattr(result, primary_metric, result.throughput_rps)
    if primary_val is None or not baseline_primary:
        vs_baseline = None
    else:
        vs_baseline = (primary_val - baseline_primary) / baseline_primary * 100
    status = result.status
    status_value = status.value if isinstance(status, ExperimentValidityStatus) else str(status)

    def _round(v: float | None, n: int) -> float | None:
        return round(v, n) if v is not None else None

    def _error_rate_from_result(res) -> float | None:
        err = getattr(res, "error_rate", None)
        if err is not None:
            return _round(err, 4)
        total = getattr(res, "total_requests", None)
        ok = getattr(res, "successful_requests", None)
        if total and ok is not None:
            return _round(1.0 - (ok / total), 4)
        return None

    return ExperimentSummary(
        experiment_id=result.experiment_id,
        param_changed=param_changed,
        value_changed=value_changed,
        # Missing latency / throughput stay None — never rewrite to 0.0.
        throughput_rps=_round(result.throughput_rps, 3),
        tokens_per_second=_round(result.tokens_per_second, 1),
        ttft_p50_ms=_round(result.ttft.p50, 1),
        ttft_p99_ms=_round(result.ttft.p99, 1),
        e2e_p50_ms=_round(result.e2e_latency.p50, 1),
        bottleneck=bottleneck,
        vs_baseline_pct=round(vs_baseline, 2) if vs_baseline is not None else None,
        run_id=getattr(result, "run_id", "") or "",
        validity_status=status_value,
        mlflow_run_id=getattr(result, "mlflow_run_id", None),
        has_config_evidence=has_critical_config_evidence(
            getattr(result, "config_evidence", None)
        ),
        # ONE full gate — identical criterion as executor / eval / DB / report
        promotable=is_promotable(result),
        failure_reason=(getattr(result, "notes", None) or "") if (
            status_value == "failed"
        ) else "",
        error_rate=_error_rate_from_result(result),
    )


def is_promotable_summary(summary: ExperimentSummary | dict[str, Any] | None) -> bool:
    """Same full gate as is_promotable(result), via the summary.promotable flag.

    Callers MUST populate `promotable` from is_promotable(result). A summary that
    only claims status=valid / has_config_evidence without promotable=True is
    rejected (prevents the inconsistent-gate bug).
    """
    if not summary:
        return False
    return bool(summary.get("promotable")) is True


def pending_hypotheses(state: AgentState) -> list[Hypothesis]:
    return [h for h in state["hypotheses"] if h["status"] == "pending"]


def is_duplicate(state: AgentState, param: str, value: Any) -> bool:
    """Return True if this (param, value) combo has already been tried."""
    for s in state["experiment_summaries"]:
        if s["param_changed"] == param and str(s["value_changed"]) == str(value):
            return True
    return False
