"""Tests for offline fair-compare reporting and script plumbing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from inferops.eval.fair_compare import render_markdown_report, run_offline_fair_compare

FIXTURES = Path("tests/fixtures/ground_truth")


def test_chat_short_budget2_planner_t4096_visibility_guard():
    report = run_offline_fair_compare(
        commit_sha="unitsha",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short"],
        budget=2,
        seed=7,
    )

    t4096_cfg = {
        "max_num_batched_tokens": 4096,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    for strategy_name in ("planner_rag", "planner_no_rag"):
        row = report["strategies"][strategy_name][0]
        picked = row["picked_configs"]
        chose_t4096 = any(cfg == t4096_cfg for cfg in picked)
        if chose_t4096:
            assert row["agent_value"] == 17.2
        else:
            assert row["agent_value"] != 17.2


def test_report_has_mode_boundaries_and_disclaimer_strings():
    report = run_offline_fair_compare(
        commit_sha="unitsha",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short"],
        budget=2,
        seed=7,
    )
    assert report["mode"] == "fair_compare"
    assert report["mode_label"] == "offline_hidden_result_replay"
    assert report["llm_boundary"] == "fake_scripted"
    assert report["tool_boundary"] == "hidden_fixture_observe"

    md = render_markdown_report(report)
    assert "offline replay of published GT rows" in md
    assert "ScriptedBottleneckLLM" in md
    assert "no live GPU" in md
    assert "no live planner LLM" in md


def test_report_includes_all_five_strategies():
    report = run_offline_fair_compare(
        commit_sha="unitsha",
        ground_truth_dir=FIXTURES,
        workloads=["chat_short", "long_generation"],
        budget=2,
        seed=7,
    )
    assert set(report["strategies"]) == {
        "default",
        "random",
        "online_local_search",
        "planner_rag",
        "planner_no_rag",
    }


def test_run_fair_compare_script_writes_markdown_and_json(tmp_path):
    cmd = [
        sys.executable,
        "scripts/run_fair_compare.py",
        "--commit-sha",
        "unitsha",
        "--ground-truth",
        "tests/fixtures/ground_truth",
        "--output-dir",
        str(tmp_path),
        "--workloads",
        "chat_short",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "unitsha.md").exists()
    assert (tmp_path / "unitsha.json").exists()


def test_run_eval_mock_excludes_planner_strategies_regression(tmp_path):
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--mock",
        "--commit-sha",
        "unitsha",
        "--ground-truth",
        "tests/fixtures/ground_truth",
        "--output-dir",
        str(tmp_path),
        "--workloads",
        "chat_short",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    data = json.loads((tmp_path / "unitsha.json").read_text())
    assert set(data["strategies"]) == {"default", "random", "online_local_search"}
    assert "planner_rag" not in data["strategies"]
    assert "planner_no_rag" not in data["strategies"]
