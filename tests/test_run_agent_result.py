"""CPU tests for scripts/run_agent.py RESULT vs_baseline_pct precision."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agent.py"
_spec = importlib.util.spec_from_file_location("run_agent_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_run_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_agent)


def _state(*, vs: float | None) -> dict:
    best = {
        "experiment_id": "sess_max_num_seqs_64",
        "param_changed": "max_num_seqs",
        "value_changed": 64,
        "throughput_rps": 18.931,
        "vs_baseline_pct": vs,
    }
    baseline = {
        "experiment_id": "sess_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 18.245,
        "vs_baseline_pct": 0.0,
    }
    return {
        "workload_name": "chat_short",
        "tried_experiment_ids": ["sess_baseline", "sess_max_num_seqs_64"],
        "stop_reason": "budget_exhausted",
        "baseline_summary": baseline,
        "best_summary": best,
    }


def test_run_agent_result_prints_vs_baseline_at_stored_precision():
    text = _run_agent.format_result_report(_state(vs=3.77), "deepseek")
    assert "+3.77%" in text
    assert "+3.8%" not in text


def test_run_agent_result_missing_vs_is_n_a_not_zero():
    text = _run_agent.format_result_report(_state(vs=None), "deepseek")
    assert "(n/a)" in text
    assert "+0.0%" not in text
