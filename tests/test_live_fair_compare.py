"""CPU-only tests for the two-arm live fair-comparison runner."""

from __future__ import annotations

import json
import subprocess
import sys

from inferops.eval.live_fair_compare import (
    CLAIM_LEVEL,
    LiveBenchmarkFixture,
    build_live_compare_report,
    ingest_planner_summary,
    render_live_compare_markdown,
    run_live_search,
)
from inferops.tools.run_benchmark import RunBenchmarkOutput


def _output(inp, *, rps: float, ttft: float = 100.0, status: str = "valid"):
    return RunBenchmarkOutput(
        experiment_id=inp.experiment_id,
        workload_name=inp.workload_name,
        throughput_rps=rps,
        tokens_per_second=1000.0,
        ttft_p50_ms=50.0,
        ttft_p99_ms=ttft,
        e2e_p50_ms=200.0,
        e2e_p99_ms=300.0,
        error_rate=0.0,
        gpu_util_pct=50.0,
        gpu_mem_gb=3.0,
        success_rate="60/60",
        mlflow_run_id=f"mlflow-{inp.experiment_id}",
        run_id=f"run-{inp.experiment_id}",
        status=status,
        ledger_path=f"logs/{inp.experiment_id}.json",
    )


def test_live_search_observes_only_after_pick_and_charges_budget():
    calls: list[dict] = []

    def fake_benchmark(inp):
        calls.append(dict(inp.config_patch))
        return _output(inp, rps=float(len(calls)))

    run, fixture = run_live_search(
        budget=3,
        workload_name="chat_short",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        session_prefix="cpu_",
        max_ttft_ms=250.0,
        benchmark_fn=fake_benchmark,
    )

    assert run.score["n_paid"] == 3
    assert len(calls) == 3
    assert calls[0] == {
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 128,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    # Stable config-key order chooses max_num_seqs=64 before any score exists.
    assert calls[1]["max_num_seqs"] == 64
    assert [row["config"] for row in fixture.observations] == calls


def test_live_observe_fails_closed_on_ttft_slo():
    fixture = LiveBenchmarkFixture(
        workload_name="chat_short",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        session_prefix="cpu_",
        max_ttft_ms=250.0,
        benchmark_fn=lambda inp: _output(inp, rps=99.0, ttft=251.0),
    )

    obs = fixture.observe(fixture.search_space.default_config())

    assert obs.validity_status == "invalid"
    assert fixture.observations[0]["engine_validity"] == "valid"
    assert fixture.observations[0]["slo_ok"] is False


def _minimal_compare_report(*, commit_sha: str, planner_source_sha: str) -> dict:
    return build_live_compare_report(
        commit_sha=commit_sha,
        planner={
            "strategy": "planner",
            "source": "ingested_report",
            "source_sha": planner_source_sha,
            "llm_boundary": "live_openrouter",
            "tool_boundary": "managed_local_vllm",
            "budget_used": 1,
            "observations": [],
            "best": {"experiment_id": "planner_baseline", "validity": "valid", "rps": 18.0},
            "decision_kind": "no_reliable_improvement",
        },
        search_run=None,
        fixture=None,
        budget=10,
        workload_name="chat_short",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_ttft_ms=250.0,
    )


def test_markdown_warns_when_planner_and_live_shas_differ():
    report = _minimal_compare_report(commit_sha="newsha", planner_source_sha="oldsha")
    markdown = render_live_compare_markdown(report)

    assert "Warning:" in markdown
    assert "not on the same code version" in markdown
    assert "`oldsha`" in markdown
    assert "`newsha`" in markdown
    assert "| planner | `oldsha` |" in markdown
    assert "| online_local_search | `newsha` |" in markdown


def test_markdown_no_sha_warning_when_shas_match():
    report = _minimal_compare_report(commit_sha="samesha", planner_source_sha="samesha")
    markdown = render_live_compare_markdown(report)

    assert "not on the same code version" not in markdown
    assert "| planner | `samesha` |" in markdown
    assert "| online_local_search | `samesha` |" in markdown


def test_ingested_planner_and_report_emit_honest_boundaries(tmp_path):
    meta = {
        "git_sha": "oldsha",
        "budget": 10,
        "decision_kind": "no_reliable_improvement",
        "tried_experiment_ids": ["planner_baseline"],
        "llm_boundary": "live_openrouter",
        "tool_boundary": "managed_local_vllm",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "case.md").write_text(
        """# Case
**Best found:** `planner_baseline`  rps=18.135
| constraint | `ttft_p99_ms` | <= | 250 | 93.8 | yes |
## Experiment Log
| # | experiment_id | run_id | mlflow | status | param | value | rps | vs baseline |
|---|---|---|---|---|---|---|---|---|
| 1 | `planner_baseline` | `run1` | `ml1` | `valid` | — | None | 18.135 | +0.0% |
## Recommendation
"""
    )
    planner = ingest_planner_summary(tmp_path)
    report = build_live_compare_report(
        commit_sha="newsha",
        planner=planner,
        search_run=None,
        fixture=None,
        budget=10,
        workload_name="chat_short",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_ttft_ms=250.0,
        blocked_reason="CPU test: live search intentionally not run",
    )

    assert report["claim_level"] == CLAIM_LEVEL
    assert report["arms"]["planner"]["llm_boundary"] == "live_openrouter"
    assert report["arms"]["online_local_search"]["llm_boundary"] == "none"
    assert report["arms"]["online_local_search"]["tool_boundary"] == "managed_local_vllm"
    assert report["arms"]["planner"]["best"]["ttft_p99_ms"] == 93.8
    assert report["arms"]["planner"]["decision_kind"] == "no_reliable_improvement"
    assert report["arms"]["online_local_search"]["protocol_score"] is None

    markdown = render_live_compare_markdown(report)
    assert "not deploy recommendations" in markdown
    assert "confirmed_gain" in markdown
    assert "Status: `blocked`" in markdown


def test_script_missing_live_conditions_is_blocked_not_pass(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_live_fair_compare.py",
            "--no-run-search",
            "--output-dir",
            str(tmp_path),
            "--commit-sha",
            "unitsha",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    data = json.loads((tmp_path / "live_fair_compare.json").read_text())
    assert data["status"] == "blocked"
    assert data["blocked_reason"]
