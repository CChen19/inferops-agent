"""CPU tests for planner fallback hardware fingerprint collection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from inferops.agent.planner import planner_node
from inferops.agent.state import initial_state
from inferops.schemas import HardwareInfo, InferenceEngine
from inferops.task import default_task_for_workload


def test_planner_fallback_fingerprint_uses_confirmed_task_engine(monkeypatch, tmp_path):
    task = default_task_for_workload("chat_short", 2).model_copy(
        update={"engine": InferenceEngine.OLLAMA}
    )
    state = initial_state(
        "chat_short",
        "resume_",
        max_experiments=2,
        optimization_task=task.model_dump(mode="json"),
    )
    baseline = {
        "experiment_id": "resume_baseline",
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
    state["baseline_summary"] = baseline
    state["best_summary"] = baseline
    state["experiment_summaries"] = [baseline]
    state["current_bottleneck"] = "compute-bound"
    state["memory_db_path"] = str(tmp_path / "memory.db")
    state["hardware_fingerprint"] = None

    collect_kwargs = {}

    def fake_collect_hardware_info(**kwargs):
        collect_kwargs.update(kwargs)
        # Missing GPU fields and version must remain an incomplete fingerprint.
        return HardwareInfo(model_name=kwargs["model_name"], engine=kwargs["engine"])

    query_kwargs = {}

    def fake_query_compatible_history(**kwargs):
        query_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(
        "inferops.memory.hardware.collect_hardware_info",
        fake_collect_hardware_info,
    )
    monkeypatch.setattr(
        "inferops.memory.history.query_compatible_history",
        fake_query_compatible_history,
    )

    response = MagicMock()
    response.content = json.dumps({"analysis": "no history", "hypotheses": []})
    response.usage_metadata = {}
    llm = MagicMock()
    llm.invoke.return_value = response

    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="(knowledge index not built)",
    ):
        patch_out = planner_node(state, llm)

    assert collect_kwargs == {
        "model_name": task.model_name,
        "engine": "ollama",
        "probe_nvidia": True,
    }
    assert query_kwargs["current_fingerprint"] is None
    assert patch_out["compatible_history"] == []
