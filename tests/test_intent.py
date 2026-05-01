"""Unit tests for natural language intent extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from inferops.agent.intent import Intent, extract_intent


def _make_llm(json_content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = json_content
    llm.invoke.return_value = resp
    return llm


def test_extract_intent_chat_scenario():
    llm = _make_llm(
        '{"workload_name": "chat_short", "model_hint": "Qwen2.5-1.5B", '
        '"target_qps": 10.0, "gpu_hint": "RTX 3060", "budget": 6, "notes": ""}'
    )
    intent = extract_intent("Qwen2.5-1.5B on RTX 3060, chat QPS=10", llm)

    assert intent.workload_name == "chat_short"
    assert intent.model_hint == "Qwen2.5-1.5B"
    assert intent.target_qps == 10.0
    assert intent.budget == 6


def test_extract_intent_long_generation():
    llm = _make_llm('{"workload_name": "long_generation", "budget": 4}')
    intent = extract_intent("Long creative text generation, 512-token outputs", llm)

    assert intent.workload_name == "long_generation"
    assert intent.budget == 4


def test_extract_intent_defaults_to_chat_short_on_bad_workload():
    llm = _make_llm('{"workload_name": "nonexistent_workload", "budget": 6}')
    intent = extract_intent("Some unknown scenario", llm)

    assert intent.workload_name == "chat_short"


def test_extract_intent_handles_invalid_json_gracefully():
    llm = _make_llm("sorry I cannot parse that")
    intent = extract_intent("anything", llm)

    assert intent.workload_name == "chat_short"
    assert intent.budget == 6


def test_extract_intent_returns_intent_dataclass():
    llm = _make_llm('{"workload_name": "mixed_traffic", "budget": 8, "notes": "burst load"}')
    intent = extract_intent("Mixed burst traffic", llm)

    assert isinstance(intent, Intent)
    assert intent.notes == "burst load"
