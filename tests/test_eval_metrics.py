"""Unit tests for inferops/eval/metrics.py — pure logic, no DB or network calls."""

from __future__ import annotations

from inferops.eval.metrics import (
    OutcomeMetrics,
    WorkloadScore,
    aggregate_scores,
    composite_score,
    compute_efficiency,
    compute_outcome,
)

# ---------------------------------------------------------------------------
# Fixtures — ground truth and agent result dicts
# ---------------------------------------------------------------------------

_GT = {
    "workload_name":      "chat_short",
    "primary_metric":     "throughput_rps",
    "higher_is_better":   True,
    "best_experiment_id": "grid_chat_short_t4096_c0_p0",
    "best_value":         17.23,
    "experiments": [
        {
            "experiment_id": "grid_chat_short_t4096_c0_p0",
            "throughput_rps": 17.23,
            "ttft_p99_ms":    65.0,
            "e2e_p50_ms":     883.0,
        }
    ],
}

_AGENT_PERFECT = {"throughput_rps": 17.23, "ttft_p99_ms": 65.0, "e2e_p50_ms": 883.0}
_AGENT_WORSE   = {"throughput_rps": 14.00, "ttft_p99_ms": 90.0, "e2e_p50_ms": 1000.0}


# ---------------------------------------------------------------------------
# compute_outcome
# ---------------------------------------------------------------------------

def test_outcome_zero_gap_when_perfect():
    outcome = compute_outcome(_GT, _AGENT_PERFECT)
    assert outcome.gap_pct == 0.0
    assert outcome.agent_value == 17.23
    assert outcome.ground_truth_value == 17.23


def test_outcome_positive_gap_when_agent_worse():
    outcome = compute_outcome(_GT, _AGENT_WORSE)
    assert outcome.gap_pct > 0
    expected = (17.23 - 14.00) / 17.23 * 100
    assert abs(outcome.gap_pct - expected) < 0.1


def test_outcome_secondary_metrics_positive_when_worse():
    outcome = compute_outcome(_GT, _AGENT_WORSE)
    # agent's ttft_p99 (90) > GT (65) → gap > 0
    assert "ttft_p99_ms" in outcome.secondary
    assert outcome.secondary["ttft_p99_ms"] > 0


def test_outcome_secondary_metrics_zero_when_perfect():
    outcome = compute_outcome(_GT, _AGENT_PERFECT)
    for v in outcome.secondary.values():
        assert abs(v) < 0.01


def test_outcome_workload_name_preserved():
    outcome = compute_outcome(_GT, _AGENT_WORSE)
    assert outcome.workload_name == "chat_short"
    assert outcome.primary_metric == "throughput_rps"


# ---------------------------------------------------------------------------
# compute_efficiency
# ---------------------------------------------------------------------------

def test_efficiency_fields():
    eff = compute_efficiency(n_experiments=7, wall_clock_s=420.0, llm_tokens_in=1500)
    assert eff.n_experiments == 7
    assert eff.wall_clock_s == 420.0
    assert eff.llm_tokens_in == 1500
    assert eff.llm_tokens_out == 0


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------

def test_composite_in_unit_range():
    outcome = compute_outcome(_GT, _AGENT_PERFECT)
    eff = compute_efficiency(3, 180.0)
    score = composite_score(outcome, eff, budget_experiments=12)
    assert 0.0 <= score <= 1.0


def test_composite_perfect_beats_worse():
    eff = compute_efficiency(5, 300.0)
    outcome_good = compute_outcome(_GT, _AGENT_PERFECT)
    outcome_bad  = compute_outcome(_GT, _AGENT_WORSE)
    assert composite_score(outcome_good, eff) > composite_score(outcome_bad, eff)


def test_composite_fewer_runs_scores_higher():
    outcome = compute_outcome(_GT, _AGENT_PERFECT)
    eff_few  = compute_efficiency(3, 180.0)
    eff_many = compute_efficiency(12, 720.0)
    assert composite_score(outcome, eff_few) > composite_score(outcome, eff_many)


def test_composite_trajectory_score_used():
    outcome = compute_outcome(_GT, _AGENT_PERFECT)
    eff = compute_efficiency(5, 300.0)
    s_high = composite_score(outcome, eff, trajectory_score=1.0)
    s_low  = composite_score(outcome, eff, trajectory_score=0.0)
    assert s_high >= s_low


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

def _make_score(gap_pct: float, n_exp: int) -> WorkloadScore:
    gt_val = 17.23
    agent_val = gt_val * (1 - gap_pct / 100)
    outcome = OutcomeMetrics("chat_short", "throughput_rps", gt_val, agent_val, gap_pct)
    eff = compute_efficiency(n_exp, 300.0)
    comp = composite_score(outcome, eff)
    return WorkloadScore("chat_short", outcome, eff, composite=comp)


def test_aggregate_mean_gap():
    scores = [_make_score(4.0, 5), _make_score(8.0, 7)]
    agg = aggregate_scores(scores)
    assert agg["mean_gap_pct"] == 6.0


def test_aggregate_mean_runs():
    scores = [_make_score(5.0, 4), _make_score(5.0, 8)]
    agg = aggregate_scores(scores)
    assert agg["mean_n_experiments"] == 6.0


def test_aggregate_empty_returns_empty():
    assert aggregate_scores([]) == {}
