"""CPU tests for Chainlit vs_baseline_pct print precision."""

from __future__ import annotations

from app import _format_all_experiments_table, _format_live_result_message


def _summary(*, experiment_id: str, vs: float | None, rps: float = 18.931) -> dict:
    return {
        "experiment_id": experiment_id,
        "param_changed": "max_num_seqs",
        "value_changed": 64,
        "throughput_rps": rps,
        "ttft_p99_ms": 93.8,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": vs,
    }


def test_live_result_message_prints_vs_baseline_at_stored_precision():
    text = _format_live_result_message(_summary(experiment_id="sess_max_num_seqs_64", vs=3.77))
    assert "+3.77%" in text
    assert "+3.8%" not in text


def test_live_result_message_missing_vs_is_n_a_not_zero():
    text = _format_live_result_message(_summary(experiment_id="sess_x", vs=None))
    assert "n/a vs baseline" in text
    assert "+0.0%" not in text


def test_all_experiments_table_prints_vs_baseline_at_stored_precision():
    rows = _format_all_experiments_table(
        [_summary(experiment_id="sess_max_num_seqs_64", vs=3.77)]
    )
    table = "\n".join(rows)
    assert "+3.77%" in table
    assert "+3.8%" not in table


def test_all_experiments_table_missing_vs_is_n_a_not_zero():
    rows = _format_all_experiments_table([_summary(experiment_id="sess_x", vs=None)])
    table = "\n".join(rows)
    assert "| n/a |" in table
    assert "+0.0%" not in table
