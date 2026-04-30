"""Unit tests for the reflector node and route_after_reflector."""

from __future__ import annotations

from inferops.agent.reflector import _MAX_STREAK, _IMPROVEMENT_THRESHOLD_PCT, reflector_node, route_after_reflector
from inferops.agent.state import AgentState, Hypothesis, initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_state(bottleneck: str = "compute-bound") -> AgentState:
    s = initial_state("chat_short", "test_", max_experiments=6)
    s["current_bottleneck"] = bottleneck
    s["experiments_remaining"] = 4
    s["no_improvement_streak"] = 0
    s["baseline_summary"] = {
        "experiment_id": "test_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 14.96,
        "tokens_per_second": 1916.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 69.0,
        "e2e_p50_ms": 1015.0,
        "bottleneck": bottleneck,
        "vs_baseline_pct": 0.0,
    }
    return s


def _add_summary(state: AgentState, vs_baseline_pct: float, bottleneck: str) -> AgentState:
    state["experiment_summaries"].append({
        "experiment_id": f"test_{len(state['experiment_summaries'])}",
        "param_changed": "max_num_batched_tokens",
        "value_changed": 4096,
        "throughput_rps": 14.96 * (1 + vs_baseline_pct / 100),
        "tokens_per_second": 2000.0,
        "ttft_p50_ms": 50.0,
        "ttft_p99_ms": 70.0,
        "e2e_p50_ms": 900.0,
        "bottleneck": bottleneck,
        "vs_baseline_pct": vs_baseline_pct,
    })
    return state


def _pending_hyp() -> Hypothesis:
    return Hypothesis(id="h1", param="max_num_seqs", value=256,
                      rationale="test", status="pending", experiment_id=None)


# ---------------------------------------------------------------------------
# reflector_node
# ---------------------------------------------------------------------------

def test_stop_on_budget_exhausted():
    s = _base_state()
    s["experiments_remaining"] = 0
    patch = reflector_node(s)
    assert patch["should_stop"] is True
    assert "budget" in patch["stop_reason"]


def test_no_stop_on_significant_improvement():
    s = _base_state()
    s = _add_summary(s, vs_baseline_pct=10.0, bottleneck="compute-bound")
    patch = reflector_node(s)
    assert patch.get("should_stop", False) is False
    assert patch.get("no_improvement_streak", 0) == 0


def test_streak_increments_on_no_improvement():
    s = _base_state()
    s["no_improvement_streak"] = 1
    s = _add_summary(s, vs_baseline_pct=1.0, bottleneck="compute-bound")  # below threshold
    patch = reflector_node(s)
    assert patch["no_improvement_streak"] == 2


def test_stop_on_max_streak():
    s = _base_state()
    s["no_improvement_streak"] = _MAX_STREAK - 1
    s = _add_summary(s, vs_baseline_pct=0.0, bottleneck="compute-bound")
    patch = reflector_node(s)
    assert patch["should_stop"] is True
    assert "no_improvement" in patch["stop_reason"]


def test_streak_resets_on_improvement():
    s = _base_state()
    s["no_improvement_streak"] = 2
    s = _add_summary(s, vs_baseline_pct=_IMPROVEMENT_THRESHOLD_PCT + 1, bottleneck="compute-bound")
    patch = reflector_node(s)
    assert patch["no_improvement_streak"] == 0
    assert patch.get("should_stop", False) is False


def test_bottleneck_switch_invalidates_pending_hypotheses():
    s = _base_state(bottleneck="compute-bound")
    s["hypotheses"] = [_pending_hyp()]
    # New experiment shows scheduling-bound (switched from compute-bound)
    s = _add_summary(s, vs_baseline_pct=2.0, bottleneck="scheduling-bound")
    patch = reflector_node(s)
    # All pending hypotheses should be skipped
    assert all(h["status"] == "skipped" for h in patch["hypotheses"] if h["id"] == "h1")


def test_no_summaries_returns_empty_patch():
    s = _base_state()
    # No experiments yet
    patch = reflector_node(s)
    assert patch == {}


# ---------------------------------------------------------------------------
# route_after_reflector
# ---------------------------------------------------------------------------

def test_route_to_end_when_stopped():
    s = _base_state()
    s["should_stop"] = True
    assert route_after_reflector(s) == "__end__"


def test_route_to_executor_when_pending_hypotheses():
    s = _base_state()
    s["should_stop"] = False
    s["hypotheses"] = [_pending_hyp()]
    assert route_after_reflector(s) == "executor"


def test_route_to_planner_when_no_pending_hypotheses():
    s = _base_state()
    s["should_stop"] = False
    s["hypotheses"] = []
    assert route_after_reflector(s) == "planner"


def test_route_to_planner_when_all_hypotheses_done():
    s = _base_state()
    s["should_stop"] = False
    s["hypotheses"] = [
        Hypothesis(id="h1", param="max_num_batched_tokens", value=4096,
                   rationale="test", status="success", experiment_id="e1"),
        Hypothesis(id="h2", param="enable_chunked_prefill", value=True,
                   rationale="test", status="failed", experiment_id="e2"),
    ]
    assert route_after_reflector(s) == "planner"
