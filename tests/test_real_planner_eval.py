"""Acceptance tests for Week-1 P0-③ real LangGraph planner eval path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from inferops.agent.state import is_promotable_summary
from inferops.eval.harness import run_mock_eval
from inferops.eval.real_graph import (
    MODE_REAL_GRAPH_LLM,
    MODE_REAL_GRAPH_OFFLINE,
    ScriptedBottleneckLLM,
    StubBenchmarkRecorder,
    require_llm_credentials,
    run_real_graph_eval,
    run_real_planner_on_workload,
)
from inferops.schemas import is_promotable

FIXTURES = Path("tests/fixtures/ground_truth")


# ---------------------------------------------------------------------------
# 1. Mock path is preset simulation — never production planner / build_graph
# ---------------------------------------------------------------------------

def test_mock_eval_does_not_call_build_graph_or_planner():
    with patch("inferops.agent.graph.build_graph") as mock_build, \
         patch("inferops.agent.planner.planner_node") as mock_planner:
        report = run_mock_eval(
            commit_sha="mockproof",
            ground_truth_dir=FIXTURES,
            workloads=["chat_short"],
            budget=2,
            seed=7,
        )

    mock_build.assert_not_called()
    mock_planner.assert_not_called()
    assert report["mode"] == "mock"
    assert report["mode_label"] == "fair_protocol_simulation"


def test_mock_trajectory_nodes_are_baseline_names_not_planner():
    from inferops.eval.baselines import run_greedy_agent, run_random_agent
    from inferops.eval.runner import load_ground_truth

    gt = load_ground_truth("chat_short", FIXTURES)
    for run in (run_random_agent(gt, budget=2, seed=1), run_greedy_agent(gt, budget=2)):
        nodes = {step["node"] for step in run.trajectory}
        assert nodes.isdisjoint({"planner", "executor", "reflector"})
        assert nodes <= {"random_agent", "greedy_agent"}


# ---------------------------------------------------------------------------
# 2. Real node trajectory Plan → Execute → Reflect
# ---------------------------------------------------------------------------

def test_real_graph_trajectory_plan_execute_reflect(tmp_path):
    llm = ScriptedBottleneckLLM(default_bottleneck="compute-bound")
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm,
        budget=3,
        bottleneck="compute-bound",
        db_path=tmp_path / "traj.db",
    )
    nodes = [s["node"] for s in run["trajectory"]]
    assert "planner" in nodes
    assert "executor" in nodes
    assert "reflector" in nodes
    # Ordered subsequence Plan → Execute → Reflect
    i_p = nodes.index("planner")
    i_e = nodes.index("executor")
    i_r = nodes.index("reflector")
    assert i_p < i_e < i_r


# ---------------------------------------------------------------------------
# 3. Observation change → different planner decisions
# ---------------------------------------------------------------------------

def test_observation_change_yields_different_hypotheses(tmp_path):
    llm_a = ScriptedBottleneckLLM()
    llm_b = ScriptedBottleneckLLM()
    run_a = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm_a,
        budget=3,
        bottleneck="compute-bound",
        session_prefix="obs_a_",
        db_path=tmp_path / "obs_a.db",
    )
    run_b = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm_b,
        budget=3,
        bottleneck="scheduling-bound",
        session_prefix="obs_b_",
        db_path=tmp_path / "obs_b.db",
    )
    hyps_a = {(h["param"], h["value"]) for h in run_a["hypotheses"]}
    hyps_b = {(h["param"], h["value"]) for h in run_b["hypotheses"]}
    assert ("max_num_batched_tokens", 4096) in hyps_a
    assert ("enable_chunked_prefill", True) in hyps_b
    assert hyps_a != hyps_b


# ---------------------------------------------------------------------------
# 4. Illegal params never reach benchmark
# ---------------------------------------------------------------------------

def test_illegal_params_never_reach_benchmark(tmp_path):
    llm = ScriptedBottleneckLLM(inject_illegal=True)
    stub = StubBenchmarkRecorder()
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm,
        budget=3,
        bottleneck="compute-bound",
        stub=stub,
        db_path=tmp_path / "illegal.db",
    )
    assert stub.calls == []
    assert run["benchmark_calls"] == []
    # Planner filtered illegal hyp (tensor_parallel_size ∉ AGENT_SEARCH_SPACE)
    assert all(h["param"] != "tensor_parallel_size" for h in run["hypotheses"])


# ---------------------------------------------------------------------------
# 5. No-gain does not promote best candidate
# ---------------------------------------------------------------------------

def test_no_gain_does_not_promote_best(tmp_path):
    llm = ScriptedBottleneckLLM(default_bottleneck="compute-bound")
    stub = StubBenchmarkRecorder(default_gain=False, baseline_rps=15.0)
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm,
        budget=3,
        bottleneck="compute-bound",
        stub=stub,
        baseline_rps=15.0,
        db_path=tmp_path / "nogain.db",
    )
    best = run["best_summary"]
    baseline = run["baseline_summary"]
    assert best is not None and baseline is not None
    assert best["experiment_id"] == baseline["experiment_id"]
    assert stub.calls, "expected at least one stubbed experiment"


# ---------------------------------------------------------------------------
# 6. Budget exhaustion sets clear stop_reason
# ---------------------------------------------------------------------------

def test_budget_exhaustion_sets_stop_reason(tmp_path):
    llm = ScriptedBottleneckLLM(default_bottleneck="compute-bound")
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm,
        budget=2,  # baseline + 1 experiment → remaining hits 0
        bottleneck="compute-bound",
        db_path=tmp_path / "budget.db",
    )
    assert run["stop_reason"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# 7. Regression vs ①: unverified cannot become best (is_promotable helpers)
# ---------------------------------------------------------------------------

def test_unevidenced_high_score_not_promoted_to_best(tmp_path):
    llm = ScriptedBottleneckLLM(default_bottleneck="compute-bound")
    stub = StubBenchmarkRecorder(default_gain=True, force_unevidenced=True, baseline_rps=15.0)
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=llm,
        budget=3,
        bottleneck="compute-bound",
        stub=stub,
        baseline_rps=15.0,
        db_path=tmp_path / "unevid.db",
    )
    best = run["best_summary"]
    baseline = run["baseline_summary"]
    assert is_promotable_summary(baseline) is True
    assert best["experiment_id"] == baseline["experiment_id"]
    assert stub.calls, "expected at least one stubbed experiment so the gate is exercised"
    # Non-baseline summaries in history must not be promotable
    for summary in run["final_state"]["experiment_summaries"]:
        if summary["experiment_id"] != baseline["experiment_id"]:
            assert is_promotable_summary(summary) is False


# ---------------------------------------------------------------------------
# 8. Real LLM entry labeled; missing creds ≠ silent pass
# ---------------------------------------------------------------------------

def test_require_llm_credentials_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        require_llm_credentials("openrouter")


def test_real_llm_mode_fails_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        run_real_graph_eval(
            commit_sha="n creds",
            ground_truth_dir=FIXTURES,
            workloads=["chat_short"],
            budget=2,
            mode=MODE_REAL_GRAPH_LLM,
            db_path=tmp_path / "llm.db",
        )


def test_real_graph_offline_report_labels(tmp_path):
    report = run_real_graph_eval(
        commit_sha="rgsha",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short"],
        budget=3,
        mode=MODE_REAL_GRAPH_OFFLINE,
        db_path=tmp_path / "label.db",
    )
    assert report["mode"] == MODE_REAL_GRAPH_OFFLINE
    assert report["llm_boundary"] == "fake_scripted"
    assert report["tool_boundary"] == "stubbed_benchmark"
    assert "real_planner" in report["strategies"]
    assert report["eval_db_path"].endswith("label.db")


def test_llm_boundary_label_keys_off_actual_llm_not_mode_alone():
    """Unknown inject is never 'live'; only trusted marker / make_llm path is."""
    from inferops.eval.real_graph import _llm_boundary_label, mark_live_llm

    fake = ScriptedBottleneckLLM()
    assert _llm_boundary_label(fake, MODE_REAL_GRAPH_LLM) == "fake_scripted"
    assert _llm_boundary_label(fake, MODE_REAL_GRAPH_OFFLINE) == "fake_scripted"

    class UnknownInjected:
        pass

    # Mode alone must NOT promote unknowns to live
    assert _llm_boundary_label(UnknownInjected(), MODE_REAL_GRAPH_LLM) == "injected"
    assert _llm_boundary_label(UnknownInjected(), MODE_REAL_GRAPH_OFFLINE) == "injected"


def test_llm_boundary_live_requires_trusted_marker_or_make_llm_path():
    from inferops.eval.real_graph import _TrustedLiveLLM, _llm_boundary_label, mark_live_llm

    class Dummy:
        def invoke(self, messages):
            return messages

    marked = mark_live_llm(Dummy())
    assert getattr(marked, "eval_llm_boundary") == "live"
    assert _llm_boundary_label(marked, MODE_REAL_GRAPH_LLM) == "live"
    assert _llm_boundary_label(marked, MODE_REAL_GRAPH_OFFLINE) == "live"
    assert _llm_boundary_label(_TrustedLiveLLM(Dummy()), "") == "live"


def test_real_graph_defaults_to_temp_eval_db_not_production_memory(tmp_path):
    """Forged rows must not land in inferops_memory.db by default."""
    prod = Path("inferops_memory.db")
    before = prod.read_bytes() if prod.exists() else None
    run = run_real_planner_on_workload(
        workload_name="chat_short",
        llm=ScriptedBottleneckLLM(),
        budget=2,
        # db_path omitted → temp eval DB
    )
    assert run["eval_db_path"]
    assert Path(run["eval_db_path"]).name == "eval_memory.db"
    assert "inferops_memory.db" not in run["eval_db_path"]
    assert Path(run["eval_db_path"]).exists()
    after = prod.read_bytes() if prod.exists() else None
    assert after == before


def test_production_reflector_has_no_empty_plan_heuristic():
    """P0-③ must not alter production Reflect heuristics."""
    import inspect

    from inferops.agent import reflector as refl

    src = inspect.getsource(refl.reflector_node)
    assert "empty_plan" not in src
    assert "eval_empty_plan" not in src
    assert "planner produced 0" not in src


# ---------------------------------------------------------------------------
# CLI wiring + separate output dirs
# ---------------------------------------------------------------------------

def test_run_eval_real_graph_writes_separate_dir(tmp_path):
    out = tmp_path / "real_graph_out"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--real-graph",
        "--commit-sha",
        "rgunit",
        "--ground-truth",
        str(FIXTURES),
        "--output-dir",
        str(out),
        "--workloads",
        "chat_short",
        "--budget",
        "3",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out / "rgunit.md").exists()
    text = (out / "rgunit.md").read_text()
    assert "real_graph_offline" in text
    assert "Real-graph offline" in text


def test_run_eval_real_llm_missing_creds_exits_nonzero(tmp_path, monkeypatch):
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()
                     if k not in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")})}
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--real-llm",
        "--commit-sha",
        "llmfail",
        "--ground-truth",
        str(FIXTURES),
        "--output-dir",
        str(tmp_path),
        "--workloads",
        "chat_short",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    assert proc.returncode == 1
    assert "OPENROUTER_API_KEY" in proc.stdout or "credentials" in proc.stdout.lower()


def test_run_eval_without_mode_flag_exits_2(tmp_path):
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--commit-sha",
        "unitsha",
        "--output-dir",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert proc.returncode == 2
    assert "--mock" in proc.stdout
    assert "--real-graph" in proc.stdout


def test_real_graph_uses_production_build_graph(tmp_path):
    """Prove offline path goes through production build_graph (not fake agents)."""
    from inferops.agent import graph as graph_mod

    real_build = graph_mod.build_graph
    calls = {"n": 0}

    def tracking_build(llm):
        calls["n"] += 1
        g = real_build(llm)
        return g

    with patch("inferops.eval.real_graph.build_graph", side_effect=tracking_build):
        run_real_planner_on_workload(
            workload_name="chat_short",
            llm=ScriptedBottleneckLLM(),
            budget=2,
            db_path=tmp_path / "prod.db",
        )
    assert calls["n"] == 1


def test_baseline_contract_gate_helpers_still_apply(tmp_path):
    """① helpers: is_promotable on synthesized baseline must hold for seeding."""
    from inferops.eval.real_graph import make_promotable_baseline_summary
    from inferops.memory.db import get_result_by_id, init_db

    db = tmp_path / "gate.db"
    init_db(db)
    # Use scoped save via make → goes through module save_result (default db).
    # Instead call synthesize path with explicit save:
    import inferops.eval.real_graph as rg

    def _save(r, path=None):
        from inferops.memory.db import save_result
        save_result(r, db_path=db)

    def _get(eid, path=None):
        return get_result_by_id(eid, db_path=db)

    prev_save, prev_get = rg.save_result, rg.get_result_by_id
    rg.save_result = _save  # type: ignore[assignment]
    rg.get_result_by_id = _get  # type: ignore[assignment]
    try:
        summary = make_promotable_baseline_summary(
            "chat_short", "gate_", throughput_rps=15.0
        )
        stored = get_result_by_id("gate_baseline", db_path=db)
        assert stored is not None
        assert is_promotable(stored) is True
        assert is_promotable_summary(summary) is True
    finally:
        rg.save_result = prev_save
        rg.get_result_by_id = prev_get
