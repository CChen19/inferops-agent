"""Planner node — LLM-driven hypothesis generation.

The planner reads the experiment history and current bottleneck, then asks the
LLM for 1-3 (param, value, rationale) hypotheses. The prompt explicitly requires
citing metric evidence; responses without evidence are rejected and the LLM is
asked to try again (once).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from inferops.agent.state import (
    AGENT_SEARCH_SPACE,
    WORKLOAD_DESCRIPTIONS,
    WORKLOAD_PRIMARY_METRIC,
    AgentState,
    Hypothesis,
    is_duplicate,
    pending_hypotheses,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert vLLM inference optimization engineer. Your task is to find the best \
serving configuration for a specific workload on an RTX 3060 Laptop (6 GB VRAM, WSL2).

You will be shown experiment history and asked to generate hypotheses for the next \
parameter change. Each hypothesis MUST:
  1. Cite at least one specific metric value from the history (e.g., "TTFT p99=120ms").
  2. Change exactly ONE parameter.
  3. Not repeat a (param, value) pair that has already been tried.
  4. Be consistent with the identified bottleneck type.

Respond with valid JSON only. No markdown, no prose outside the JSON.\
"""

_USER_TEMPLATE = """\
WORKLOAD: {workload_name}
DESCRIPTION: {workload_description}

PRIMARY METRIC (maximize): {primary_metric}
CURRENT BOTTLENECK: {current_bottleneck}
EXPERIMENTS REMAINING: {budget}

BASELINE:
  {baseline_line}

BEST SO FAR:
  {best_line}

EXPERIMENT HISTORY (most recent first):
{history_table}

ALREADY TRIED — do NOT repeat these (param, value) pairs:
{tried_pairs}

TUNABLE PARAMETERS (safe ranges for RTX 3060):
  max_num_batched_tokens  : {batched_values}  — tokens processed per scheduler step
  max_num_seqs            : {seqs_values}     — max concurrent sequences
  enable_chunked_prefill  : [true, false]     — interleave prefill with decode
  enable_prefix_caching   : [true, false]     — reuse KV for shared prompt prefixes

BOTTLENECK GUIDANCE:
  compute-bound     → try increasing max_num_batched_tokens (more GPU saturation)
  scheduling-bound  → try enable_chunked_prefill=true (reduce TTFT variance)
  memory-bound      → try reducing max_num_seqs (less KV pressure)
  kv-bound          → try enable_prefix_caching=true or reduce max_num_seqs

Generate {n_hypotheses} hypothesis/hypotheses. Respond with:
{{
  "analysis": "<one paragraph citing specific metric values and explaining the bottleneck>",
  "hypotheses": [
    {{"param": "...", "value": ..., "rationale": "... [MUST cite a specific metric value] ..."}},
    ...
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_summary(s: dict | None) -> str:
    if s is None:
        return "(none yet)"
    return (
        f"experiment_id={s['experiment_id']}  "
        f"rps={s['throughput_rps']}  ttft_p99={s['ttft_p99_ms']}ms  "
        f"e2e_p50={s['e2e_p50_ms']}ms  bottleneck={s['bottleneck']}  "
        f"vs_baseline={s['vs_baseline_pct']:+.1f}%"
    )


def _build_history_table(summaries: list[dict]) -> str:
    if not summaries:
        return "  (no experiments yet)"
    lines = ["  param_changed          value   rps     ttft_p99  e2e_p50  bottleneck      vs_baseline"]
    for s in reversed(summaries[-8:]):   # last 8, most recent first
        lines.append(
            f"  {str(s.get('param_changed') or 'baseline'):<22} "
            f"{str(s.get('value_changed', '')):<7} "
            f"{s['throughput_rps']:<7.3f} "
            f"{s['ttft_p99_ms']:<9.1f} "
            f"{s['e2e_p50_ms']:<8.1f} "
            f"{s['bottleneck']:<15} "
            f"{s['vs_baseline_pct']:+.1f}%"
        )
    return "\n".join(lines)


def _tried_pairs(summaries: list[dict]) -> str:
    pairs = [
        f"  {s['param_changed']}={s['value_changed']}"
        for s in summaries
        if s.get("param_changed")
    ]
    return "\n".join(pairs) if pairs else "  (none)"


def _validate_hypotheses(raw_hyps: list[dict], state: AgentState) -> list[dict]:
    """Filter out invalid hypotheses (wrong param, out-of-range, already tried)."""
    valid = []
    for h in raw_hyps:
        param = h.get("param", "")
        value = h.get("value")
        rationale = h.get("rationale", "")

        if param not in AGENT_SEARCH_SPACE:
            continue
        allowed_vals = AGENT_SEARCH_SPACE[param]
        # Coerce bool params
        if isinstance(allowed_vals[0], bool):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    value = True
                elif lowered in ("false", "0", "no"):
                    value = False
                else:
                    continue
            else:
                value = bool(value)
        else:
            try:
                value = type(allowed_vals[0])(value)
            except (TypeError, ValueError):
                continue
        if value not in allowed_vals:
            continue
        if is_duplicate(state, param, value):
            continue
        # Evidence check: rationale must contain a number (metric value)
        if not re.search(r"\d+(\.\d+)?", rationale):
            continue
        h["param"] = param
        h["value"] = value
        valid.append(h)
    return valid


def _parse_llm_response(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response; strip markdown fences if present."""
    content = content.strip()
    # Strip ```json ... ``` fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState, llm) -> dict:
    """
    Generate 1–3 hypotheses using the LLM, enforcing evidence-based reasoning.
    Returns a state patch with updated hypotheses and messages.
    """
    # How many hypotheses to request (fewer when budget is tight)
    n = 1 if state["experiments_remaining"] <= 2 else (2 if state["experiments_remaining"] <= 4 else 3)

    user_msg = _USER_TEMPLATE.format(
        workload_name=state["workload_name"],
        workload_description=WORKLOAD_DESCRIPTIONS.get(state["workload_name"], ""),
        primary_metric=WORKLOAD_PRIMARY_METRIC.get(state["workload_name"], "throughput_rps"),
        current_bottleneck=state["current_bottleneck"],
        budget=state["experiments_remaining"],
        baseline_line=_fmt_summary(state["baseline_summary"]),
        best_line=_fmt_summary(state["best_summary"]),
        history_table=_build_history_table(state["experiment_summaries"]),
        tried_pairs=_tried_pairs(state["experiment_summaries"]),
        batched_values=str(AGENT_SEARCH_SPACE["max_num_batched_tokens"]),
        seqs_values=str(AGENT_SEARCH_SPACE["max_num_seqs"]),
        n_hypotheses=n,
    )

    messages_in = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]
    response = llm.invoke(messages_in)

    # Token tracking
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = (
            response.usage_metadata.get("input_tokens", 0)
            + response.usage_metadata.get("output_tokens", 0)
        )

    # Parse response
    raw_hyps: list[dict] = []
    analysis = ""
    try:
        data = _parse_llm_response(response.content)
        raw_hyps = data.get("hypotheses", [])
        analysis = data.get("analysis", "")
    except (json.JSONDecodeError, ValueError):
        # Retry once with explicit instruction
        retry_msg = HumanMessage(
            content="Your response was not valid JSON. Respond with ONLY the JSON object, no other text."
        )
        retry_response = llm.invoke(messages_in + [response, retry_msg])
        tokens_used += (
            retry_response.usage_metadata.get("input_tokens", 0)
            + retry_response.usage_metadata.get("output_tokens", 0)
        ) if hasattr(retry_response, "usage_metadata") and retry_response.usage_metadata else 0
        try:
            data = _parse_llm_response(retry_response.content)
            raw_hyps = data.get("hypotheses", [])
            analysis = data.get("analysis", "")
            response = retry_response
        except (json.JSONDecodeError, ValueError):
            raw_hyps = []

    # Validate and convert to Hypothesis TypedDicts
    valid = _validate_hypotheses(raw_hyps, state)
    new_hypotheses: list[Hypothesis] = [
        Hypothesis(
            id=f"h{len(state['hypotheses']) + i + 1}",
            param=h["param"],
            value=h["value"],
            rationale=h["rationale"],
            status="pending",
            experiment_id=None,
        )
        for i, h in enumerate(valid)
    ]

    # Add planner step to trajectory
    trajectory_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "planner",
        "workload": state["workload_name"],
        "action": f"generated {len(new_hypotheses)} hypothesis/hypotheses",
        "reasoning": analysis,
        "hypotheses": [{"param": h["param"], "value": h["value"]} for h in new_hypotheses],
        "tokens_used": tokens_used,
    }

    return {
        "hypotheses": state["hypotheses"] + new_hypotheses,
        "trajectory": state["trajectory"] + [trajectory_step],
        "messages": [AIMessage(content=response.content)],
    }
