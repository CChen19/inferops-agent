"""Unit tests for the planner node — LLM is mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.planner import (
    _build_history_table,
    _validate_hypotheses,
    planner_node,
)
from inferops.agent.state import AgentState, initial_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_state() -> AgentState:
    s = initial_state("chat_short", "test_", max_experiments=6)
    s["baseline_summary"] = {
        "experiment_id": "test_baseline",
        "run_id": "run_baseline",
        "param_changed": None,
        "value_changed": None,
        "throughput_rps": 14.96,
        "tokens_per_second": 1916.0,
        "ttft_p50_ms": 48.0,
        "ttft_p99_ms": 69.0,
        "e2e_p50_ms": 1015.0,
        "bottleneck": "compute-bound",
        "vs_baseline_pct": 0.0,
    }
    s["best_summary"] = s["baseline_summary"]
    s["experiment_summaries"] = [s["baseline_summary"]]
    s["current_bottleneck"] = "compute-bound"
    return s


def _mock_llm(content: str):
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    llm.invoke.return_value = response
    return llm


def _cited_hypothesis(param, value, rationale, *, source="vllm_scheduler"):
    return {
        "param": param,
        "value": value,
        "rationale": rationale,
        "citations": {
            "metric": {
                "run_id": "run_baseline",
                "metric": "throughput_rps",
                "value": 14.96,
            },
            "document": {"source": source},
        },
    }


@pytest.fixture(autouse=True)
def _retrieved_context():
    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value=(
            "[source: vllm_scheduler] §Scheduling\nscheduler guidance\n\n"
            "[source: chunked_prefill] §Chunked prefill\nchunked prefill guidance"
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# _validate_hypotheses
# ---------------------------------------------------------------------------


def test_validate_removes_unknown_param():
    state = _base_state()
    raw = [{"param": "unknown_param", "value": 999, "rationale": "because 14.96 rps is low"}]
    assert _validate_hypotheses(raw, state) == []


def test_validate_removes_out_of_range_value():
    state = _base_state()
    raw = [{"param": "max_num_batched_tokens", "value": 9999, "rationale": "ttft p99=69ms is high"}]
    assert _validate_hypotheses(raw, state) == []


def test_validate_removes_already_tried():
    state = _base_state()
    state["experiment_summaries"].append(
        {
            "param_changed": "max_num_batched_tokens",
            "value_changed": 4096,
            "throughput_rps": 17.0,
            "tokens_per_second": 2100.0,
            "ttft_p50_ms": 50.0,
            "ttft_p99_ms": 65.0,
            "e2e_p50_ms": 900.0,
            "bottleneck": "compute-bound",
            "vs_baseline_pct": 13.6,
            "experiment_id": "test_mbt_4096",
        }
    )
    raw = [{"param": "max_num_batched_tokens", "value": 4096, "rationale": "rps=14.96 is low"}]
    assert _validate_hypotheses(raw, state) == []


def test_validate_removes_no_evidence():
    state = _base_state()
    raw = [
        {
            "param": "max_num_batched_tokens",
            "value": 4096,
            "rationale": "seems like a good idea",
        }
    ]
    # No numeric metric citation → rejected
    assert _validate_hypotheses(raw, state) == []


def test_validate_accepts_valid_hypothesis():
    state = _base_state()
    raw = [
        _cited_hypothesis(
            "max_num_batched_tokens",
            4096,
            "rps=14.96 is low; [source: vllm_scheduler] larger batches improve GPU saturation",
        )
    ]
    result = _validate_hypotheses(raw, state, {"vllm_scheduler"})
    assert len(result) == 1
    assert result[0]["param"] == "max_num_batched_tokens"
    assert result[0]["value"] == 4096


def test_validate_coerces_bool():
    state = _base_state()
    raw = [
        _cited_hypothesis(
            "enable_chunked_prefill",
            "true",
            "TTFT p99=69ms is high [source: chunked_prefill]",
            source="chunked_prefill",
        )
    ]
    result = _validate_hypotheses(raw, state, {"chunked_prefill"})
    assert len(result) == 1
    assert result[0]["value"] is True


def test_validate_coerces_false_string_to_false():
    state = _base_state()
    raw = [
        _cited_hypothesis(
            "enable_chunked_prefill",
            "false",
            "TTFT p99=69ms; disabling chunked prefill [source: chunked_prefill]",
            source="chunked_prefill",
        )
    ]
    result = _validate_hypotheses(raw, state, {"chunked_prefill"})
    assert len(result) == 1
    assert result[0]["value"] is False


def test_table_rendered_metric_value_is_accepted_without_precision_loss():
    state = _base_state()
    state["baseline_summary"]["throughput_rps"] = 14.961234567
    raw = [
        _cited_hypothesis(
            "max_num_batched_tokens",
            4096,
            "throughput_rps=14.961234567 [source: vllm_scheduler]",
        )
    ]
    raw[0]["citations"]["metric"]["value"] = 14.961234567

    assert "14.961234567" in _build_history_table(state["experiment_summaries"])
    assert _validate_hypotheses(raw, state, {"vllm_scheduler"})


def test_vs_baseline_pct_rendered_at_stored_precision_and_citable():
    state = _base_state()
    state["baseline_summary"]["vs_baseline_pct"] = 20.05  # production stores 2 decimals

    table = _build_history_table(state["experiment_summaries"])
    assert "+20.05%" in table
    assert "+20.1%" not in table  # the old rounded rendering must not appear

    def _cite(value):
        h = _cited_hypothesis(
            "max_num_batched_tokens",
            4096,
            "vs_baseline_pct=+20.05% [source: vllm_scheduler]",
        )
        h["citations"]["metric"]["metric"] = "vs_baseline_pct"
        h["citations"]["metric"]["value"] = value
        return [h]

    # Citing exactly what the table prints is accepted.
    assert _validate_hypotheses(_cite(20.05), state, {"vllm_scheduler"})
    # Citing the rounded (previously printed) value, or any nearby value, is rejected.
    assert _validate_hypotheses(_cite(20.1), state, {"vllm_scheduler"}) == []
    assert _validate_hypotheses(_cite(20.06), state, {"vllm_scheduler"}) == []


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


def test_planner_adds_hypotheses():
    state = _base_state()
    llm_response = json.dumps(
        {
            "analysis": "throughput_rps=14.96 is suboptimal; compute-bound workload.",
            "hypotheses": [
                _cited_hypothesis(
                    "max_num_batched_tokens",
                    4096,
                    "rps=14.96 suggests overhead; bigger batches increase GPU util "
                    "[source: vllm_scheduler]",
                ),
            ],
        }
    )
    llm = _mock_llm(llm_response)
    patch = planner_node(state, llm)

    assert "hypotheses" in patch
    assert len(patch["hypotheses"]) == 1
    assert patch["hypotheses"][0]["status"] == "pending"
    assert patch["hypotheses"][0]["param"] == "max_num_batched_tokens"


def test_planner_adds_trajectory_step():
    state = _base_state()
    llm_response = json.dumps(
        {
            "analysis": "rps=14.96 is suboptimal.",
            "hypotheses": [
                _cited_hypothesis(
                    "max_num_seqs",
                    256,
                    "rps=14.96 with concurrency=16 suggests low parallelism "
                    "[source: vllm_scheduler]",
                ),
            ],
        }
    )
    patch = planner_node(state, _mock_llm(llm_response))
    assert "trajectory" in patch
    assert patch["trajectory"][-1]["node"] == "planner"


def test_planner_handles_invalid_json_gracefully():
    state = _base_state()
    # LLM returns garbage on first call, then valid JSON on retry
    bad_response = MagicMock()
    bad_response.content = "This is not JSON at all."
    bad_response.usage_metadata = {}

    good_response = MagicMock()
    good_response.content = json.dumps(
        {
            "analysis": "rps=14.96 is low",
            "hypotheses": [
                _cited_hypothesis(
                    "enable_chunked_prefill",
                    True,
                    "TTFT p99=69ms variance suggests scheduling-bound [source: chunked_prefill]",
                    source="chunked_prefill",
                ),
            ],
        }
    )
    good_response.usage_metadata = {"input_tokens": 80, "output_tokens": 40}

    llm = MagicMock()
    llm.invoke.side_effect = [bad_response, good_response]

    patch = planner_node(state, llm)
    # Should have retried and succeeded
    assert len(patch.get("hypotheses", [])) == 1


def test_planner_retries_once_then_rejects_forged_run_id():
    state = _base_state()
    forged = _cited_hypothesis(
        "max_num_batched_tokens",
        4096,
        "rps=14.96 is low [source: vllm_scheduler]",
    )
    forged["citations"]["metric"]["run_id"] = "forged_run"
    response = MagicMock()
    response.content = json.dumps({"analysis": "forged", "hypotheses": [forged]})
    response.usage_metadata = {}
    llm = MagicMock()
    llm.invoke.side_effect = [response, response]

    patch = planner_node(state, llm)

    assert patch["hypotheses"] == []
    assert llm.invoke.call_count == 2


def test_planner_requests_fewer_hypotheses_on_low_budget():
    """With budget=2 the planner should request only 1 hypothesis."""
    state = _base_state()
    state["experiments_remaining"] = 2

    captured = {}

    def capture_invoke(messages):
        # Grab the user message to inspect budget
        captured["user_msg"] = messages[-1].content
        resp = MagicMock()
        resp.content = json.dumps(
            {
                "analysis": "rps=14.96 is low",
                "hypotheses": [
                    _cited_hypothesis(
                        "max_num_batched_tokens",
                        4096,
                        "rps=14.96 is below ceiling [source: vllm_scheduler]",
                    ),
                ],
            }
        )
        resp.usage_metadata = {}
        return resp

    llm = MagicMock()
    llm.invoke.side_effect = capture_invoke
    planner_node(state, llm)
    assert "1 hypothesis" in captured["user_msg"]
