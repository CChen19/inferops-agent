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
    throughput_rps: float
    tokens_per_second: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    e2e_p50_ms: float
    bottleneck: str
    vs_baseline_pct: float      # % change vs baseline on primary metric (>0 = better)


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

    # Trajectory (for eval/judge in Phase 3 eval framework)
    trajectory: list[dict[str, Any]]

    # LangGraph message history (Planner read + write)
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKLOAD_PRIMARY_METRIC: dict[str, str] = {
    "chat_short":                 "throughput_rps",
    "long_context_qa":            "throughput_rps",
    "high_concurrency_short_out": "throughput_rps",
    "long_generation":            "tokens_per_second",
    "mixed_traffic":              "throughput_rps",
}

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
) -> ExperimentSummary:
    primary_val = getattr(result, primary_metric, result.throughput_rps)
    vs_baseline = (primary_val - baseline_primary) / baseline_primary * 100 if baseline_primary else 0.0
    return ExperimentSummary(
        experiment_id=result.experiment_id,
        param_changed=param_changed,
        value_changed=value_changed,
        throughput_rps=round(result.throughput_rps, 3),
        tokens_per_second=round(result.tokens_per_second, 1),
        ttft_p50_ms=round(result.ttft.p50, 1),
        ttft_p99_ms=round(result.ttft.p99, 1),
        e2e_p50_ms=round(result.e2e_latency.p50, 1),
        bottleneck="unknown",   # filled by executor after analyze_bottleneck
        vs_baseline_pct=round(vs_baseline, 2),
    )


def pending_hypotheses(state: AgentState) -> list[Hypothesis]:
    return [h for h in state["hypotheses"] if h["status"] == "pending"]


def is_duplicate(state: AgentState, param: str, value: Any) -> bool:
    """Return True if this (param, value) combo has already been tried."""
    for s in state["experiment_summaries"]:
        if s["param_changed"] == param and str(s["value_changed"]) == str(value):
            return True
    return False
