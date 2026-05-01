"""Smoke tests for the LangGraph tool registry — schema and name checks."""

from __future__ import annotations

from inferops.tools.registry import ALL_TOOLS

EXPECTED_NAMES = {
    "tool_run_benchmark",
    "tool_propose_config_patch",
    "tool_read_gpu_metrics",
    "tool_profile_with_pyspy",
    "tool_analyze_bottleneck",
    "tool_compare_experiments",
    "tool_query_experiment_memory",
    "tool_write_report_section",
    "tool_knowledge_retriever",
}


def test_all_tools_registered():
    assert len(ALL_TOOLS) == 9


def test_tool_names():
    names = {t.name for t in ALL_TOOLS}
    assert names == EXPECTED_NAMES


def test_tools_have_descriptions():
    for t in ALL_TOOLS:
        assert t.description, f"{t.name} has no description"
        assert len(t.description) > 20, f"{t.name} description too short"


def test_tools_have_args_schema():
    for t in ALL_TOOLS:
        schema = t.args_schema
        assert schema is not None, f"{t.name} has no args_schema"
