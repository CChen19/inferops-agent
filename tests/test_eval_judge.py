"""Unit tests for heuristic and LLM-backed trajectory judging."""

from __future__ import annotations

from unittest.mock import MagicMock

from inferops.eval.judge import _format_trajectory, judge_trajectory


def test_heuristic_judge_empty_trajectory_is_neutral():
    score = judge_trajectory([])

    assert score.overall == 0.5
    assert "Empty" in score.reasoning


def test_heuristic_judge_penalizes_duplicate_actions_and_detects_replan():
    trajectory = [
        {"step": 1, "action": "run e1", "experiment_id": "e1", "reasoning": "rps=2.0 baseline"},
        {
            "step": 2,
            "action": "run e1",
            "experiment_id": "e1",
            "reasoning": "bottleneck changed, replan",
        },
    ]

    score = judge_trajectory(trajectory)

    assert score.evidence_based == 1.0
    assert score.no_repeat < 1.0
    assert score.replan == 1.0


def test_llm_judge_extracts_json_from_wrapped_response():
    llm = MagicMock()
    resp = MagicMock()
    resp.content = (
        "Here is the score:\n"
        '{"evidence_based": 1, "no_repeat": 0.5, "replan": 0.25, '
        '"efficient": 0.75, "reasoning": "mixed quality"}'
    )
    llm.invoke.return_value = resp

    score = judge_trajectory([{"step": 1, "action": "run", "reasoning": "rps=2.0"}], llm=llm)

    assert score.evidence_based == 1.0
    assert score.no_repeat == 0.5
    assert score.replan == 0.25
    assert score.efficient == 0.75
    assert score.overall == 0.6375


def test_format_trajectory_includes_key_result_fields():
    text = _format_trajectory([
        {
            "step": 1,
            "action": "run",
            "reasoning": "rps=2.0",
            "result": {"throughput_rps": 2.1, "ignored": "x", "bottleneck": "compute-bound"},
        }
    ])

    assert "Step 1: run" in text
    assert "rps=2.0" in text
    assert "throughput_rps" in text
    assert "ignored" not in text
