"""Reflector node — judges each experiment's outcome and controls loop flow.

Decision logic (pure rules, no LLM):
  - Budget exhausted              → stop
  - Latest experiment improved primary metric by >5% vs baseline → reset streak
  - Otherwise                     → increment no_improvement_streak
  - Streak ≥ 3                    → stop (converged or stuck)
  - Bottleneck type changed       → mark all remaining hypotheses "skipped"
                                     (force re-plan with new bottleneck info)

Routing (called by the conditional edge after reflector):
  - should_stop=True              → END
  - pending hypotheses remain     → "executor"
  - no pending hypotheses         → "planner"
"""

from __future__ import annotations

from typing import Literal

from inferops.agent.state import AgentState, Hypothesis, WORKLOAD_PRIMARY_METRIC, pending_hypotheses

# Minimum improvement over baseline to count as "significant"
_IMPROVEMENT_THRESHOLD_PCT = 5.0
# Stop after this many consecutive non-improvements
_MAX_STREAK = 3


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def reflector_node(state: AgentState) -> dict:
    """Evaluate the most recent experiment and update control flags."""

    # --- Budget check ---
    if state["experiments_remaining"] <= 0:
        return {
            "should_stop": True,
            "stop_reason": "budget_exhausted",
        }

    summaries = state["experiment_summaries"]
    if not summaries:
        # No experiments yet — nothing to reflect on
        return {}

    latest = summaries[-1]
    primary_metric = WORKLOAD_PRIMARY_METRIC[state["workload_name"]]
    improvement = latest.get("vs_baseline_pct", 0.0)

    # --- Improvement check ---
    if improvement >= _IMPROVEMENT_THRESHOLD_PCT:
        new_streak = 0
    else:
        new_streak = state["no_improvement_streak"] + 1

    # --- Consecutive failure stop ---
    if new_streak >= _MAX_STREAK:
        return {
            "no_improvement_streak": new_streak,
            "should_stop": True,
            "stop_reason": f"no_improvement_{_MAX_STREAK}_consecutive",
        }

    # --- Bottleneck switch detection ---
    new_bottleneck = latest.get("bottleneck", "unknown")
    old_bottleneck = state["current_bottleneck"]
    bottleneck_switched = (
        new_bottleneck not in ("unknown", old_bottleneck)
        and old_bottleneck not in ("unknown",)
    )
    updated_hyps = state["hypotheses"]
    if bottleneck_switched:
        # Invalidate remaining pending hypotheses — they were generated for the old bottleneck
        updated_hyps = [
            {**h, "status": "skipped"} if h["status"] == "pending" else h
            for h in updated_hyps
        ]

    # --- Trajectory ---
    traj_step = {
        "step": len(state["trajectory"]) + 1,
        "node": "reflector",
        "workload": state["workload_name"],
        "action": "reflect",
        "reasoning": (
            f"improvement={improvement:+.1f}%  streak={new_streak}  "
            f"bottleneck={new_bottleneck}"
            + (f"  [switched from {old_bottleneck}]" if bottleneck_switched else "")
        ),
        "result": {
            "vs_baseline_pct": improvement,
            "streak": new_streak,
            "bottleneck_switched": bottleneck_switched,
        },
    }

    return {
        "no_improvement_streak": new_streak,
        "should_stop": False,
        "hypotheses": updated_hyps,
        "trajectory": state["trajectory"] + [traj_step],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_reflector(
    state: AgentState,
) -> Literal["planner", "executor", "__end__"]:
    """Conditional edge: decides the next node after the reflector."""
    if state["should_stop"]:
        return "__end__"
    if pending_hypotheses(state):
        return "executor"
    return "planner"
