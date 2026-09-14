"""CPU tests for the offline memory pre-experiment (groups A/B/C)."""

from __future__ import annotations

from pathlib import Path

from inferops.eval.memory_preexperiment import (
    BAD_CONFIG,
    GOAL_CONFIG,
    MODEL,
    OTHER_MODEL,
    WORKLOAD,
    configs_equal,
    eval_fingerprint,
    is_lasting_config_failure,
    load_prior_config_failures,
    meets_business_goal,
    run_memory_preexperiment,
    run_scenario,
    seed_prior_failure,
    should_skip_exact_config_failure,
    should_skip_history_hint,
)
from inferops.eval.protocol import Observation
from inferops.memory.history import is_history_failure, query_compatible_history

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GT = _REPO_ROOT / "tests" / "fixtures" / "ground_truth"


def _assert_no_repo_root_sqlite() -> None:
    for name in (
        "inferops_memory.db",
        "inferops_memory.db-wal",
        "inferops_memory.db-shm",
    ):
        assert not (_REPO_ROOT / name).exists(), f"leaked {name} into repo root"


def test_lasting_config_failure_vs_transient():
    assert is_lasting_config_failure(
        status="failed",
        notes="vLLM OOM during startup — CUDA out of memory",
    )
    assert is_lasting_config_failure(status="invalid", notes="knob mismatch")
    assert not is_lasting_config_failure(
        status="failed",
        notes="vLLM not ready after startup timeout — spawn stalled",
    )
    assert not is_lasting_config_failure(
        status="failed",
        notes="connection refused during health poll",
    )


def test_exact_filter_requires_full_config_match(tmp_path):
    db = tmp_path / "prior.db"
    seed_prior_failure(
        db,
        knobs=BAD_CONFIG,
        notes="CUDA out of memory",
        experiment_id="prior_enable_chunked_prefill_True",
    )
    priors = load_prior_config_failures(
        db,
        model_name=MODEL,
        workload_name=WORKLOAD,
        exclude_session_id="curr_",
    )
    assert should_skip_exact_config_failure(BAD_CONFIG, priors)
    # Same environment but different knobs → do not generalize.
    near = dict(BAD_CONFIG, max_num_batched_tokens=4096)
    assert not should_skip_exact_config_failure(near, priors)
    assert not should_skip_exact_config_failure(GOAL_CONFIG, priors)
    _assert_no_repo_root_sqlite()


def test_compatible_history_excludes_other_model(tmp_path):
    db = tmp_path / "alien.db"
    seed_prior_failure(
        db,
        knobs=BAD_CONFIG,
        notes="CUDA out of memory",
        model_name=OTHER_MODEL,
        experiment_id="prior_alien_enable_chunked_prefill_True",
    )
    rows = query_compatible_history(
        model_name=MODEL,
        workload_name=WORKLOAD,
        exclude_session_id="curr_",
        db_path=db,
        current_fingerprint=eval_fingerprint(MODEL),
    )
    assert rows == []
    priors = load_prior_config_failures(
        db,
        model_name=MODEL,
        workload_name=WORKLOAD,
        exclude_session_id="curr_",
    )
    assert priors == []
    _assert_no_repo_root_sqlite()


def test_history_hint_skips_failed_param_value(tmp_path):
    db = tmp_path / "hint.db"
    seed_prior_failure(
        db,
        knobs=BAD_CONFIG,
        notes="CUDA out of memory",
        experiment_id="prior_enable_chunked_prefill_True",
    )
    rows = query_compatible_history(
        model_name=MODEL,
        workload_name=WORKLOAD,
        exclude_session_id="curr_",
        db_path=db,
        current_fingerprint=eval_fingerprint(MODEL),
    )
    assert rows
    assert is_history_failure(rows[0])
    assert should_skip_history_hint(BAD_CONFIG, rows)
    assert not should_skip_history_hint(GOAL_CONFIG, rows)
    _assert_no_repo_root_sqlite()


def test_reusable_b_and_c_reduce_wasted_executions(tmp_path):
    sc = run_scenario("reusable", ground_truth_dir=_GT, base_tmp=tmp_path / "reusable")
    a, b, c = sc["groups"]["A"], sc["groups"]["B"], sc["groups"]["C"]
    assert a["goal_met"] and b["goal_met"] and c["goal_met"]
    assert a["wasted_executions_before_goal"] == 1
    assert b["wasted_executions_before_goal"] == 0
    assert c["wasted_executions_before_goal"] == 0
    assert b["rejected_without_execute"] >= 1
    assert c["rejected_without_execute"] >= 1
    assert a["execute_failure_count"] >= 1
    assert b["execute_failure_count"] == 0
    _assert_no_repo_root_sqlite()


def test_irrelevant_memory_does_not_beat_a(tmp_path):
    sc = run_scenario("irrelevant", ground_truth_dir=_GT, base_tmp=tmp_path / "irr")
    a, b, c = sc["groups"]["A"], sc["groups"]["B"], sc["groups"]["C"]
    assert a["goal_met"] and b["goal_met"] and c["goal_met"]
    assert b["wasted_executions_before_goal"] == a["wasted_executions_before_goal"]
    assert c["wasted_executions_before_goal"] == a["wasted_executions_before_goal"]
    assert b["rejected_without_execute"] == 0
    assert c["rejected_without_execute"] == 0
    _assert_no_repo_root_sqlite()


def test_transient_timeout_not_lasting_blacklist_for_b(tmp_path):
    sc = run_scenario("transient", ground_truth_dir=_GT, base_tmp=tmp_path / "trans")
    a, b, c = sc["groups"]["A"], sc["groups"]["B"], sc["groups"]["C"]
    assert a["goal_met"] and b["goal_met"]
    assert a["wasted_executions_before_goal"] == 0
    assert b["wasted_executions_before_goal"] == 0
    assert b["rejected_without_execute"] == 0
    # C uses broader is_history_failure(status=failed) — may wrongly filter the goal.
    assert c["feasible_wrongly_filtered"] is True
    assert c["goal_met"] is False
    _assert_no_repo_root_sqlite()


def test_no_history_not_worse_than_a(tmp_path):
    sc = run_scenario("no_history", ground_truth_dir=_GT, base_tmp=tmp_path / "none")
    a, b, c = sc["groups"]["A"], sc["groups"]["B"], sc["groups"]["C"]
    assert a["goal_met"] and b["goal_met"] and c["goal_met"]
    assert b["wasted_executions_before_goal"] == a["wasted_executions_before_goal"]
    assert c["wasted_executions_before_goal"] == a["wasted_executions_before_goal"]
    assert b["execute_count"] == a["execute_count"]
    assert c["execute_count"] == a["execute_count"]
    _assert_no_repo_root_sqlite()


def test_independent_db_copies_no_shared_accumulation(tmp_path):
    sc = run_scenario("reusable", ground_truth_dir=_GT, base_tmp=tmp_path / "iso")
    # Each group must have finished with its own counts; B/C rejects prove
    # they read the seeded prior without needing A's executes.
    assert sc["groups"]["A"]["rejected_without_execute"] == 0
    assert sc["groups"]["B"]["rejected_without_execute"] >= 1
    assert sc["groups"]["C"]["rejected_without_execute"] >= 1
    _assert_no_repo_root_sqlite()


def test_full_report_disclaimer_and_no_gpu_claim(tmp_path):
    report = run_memory_preexperiment(
        commit_sha="mempretest",
        ground_truth_dir=_GT,
        budget_slots=4,
        output_dir=tmp_path / "out",
    )
    assert "scripted" in report["disclaimer"].lower()
    assert "not a live gpu" in report["disclaimer"].lower()
    assert report["llm_boundary"] == "scripted_fixed_proposal_order"
    for name in ("reusable", "irrelevant"):
        sc = next(s for s in report["scenarios"] if s["scenario"] == name)
        assert "scenario override of GT" in (sc.get("scenario_note") or ""), name
    for name in ("transient", "no_history"):
        sc = next(s for s in report["scenarios"] if s["scenario"] == name)
        assert sc.get("scenario_note") is None, name
    assert (tmp_path / "out" / "mempretest.json").exists()
    assert (tmp_path / "out" / "mempretest.md").exists()
    md = (tmp_path / "out" / "mempretest.md").read_text()
    assert "Disclaimer" in md
    assert "scenario override of GT" in md
    assert "nvidia" not in md.lower()
    _assert_no_repo_root_sqlite()


def test_meets_business_goal_requires_threshold():
    good = Observation(
        metrics={"throughput_rps": 17.2},
        validity_status="valid",
        error_rate=0.01,
        config_evidence=True,
        bottleneck="compute-bound",
    )
    mid = Observation(
        metrics={"throughput_rps": 16.0},
        validity_status="valid",
        error_rate=0.01,
        config_evidence=True,
        bottleneck="compute-bound",
    )
    assert meets_business_goal(good)
    assert not meets_business_goal(mid)


def test_configs_equal_is_exact():
    assert configs_equal(BAD_CONFIG, dict(BAD_CONFIG))
    assert not configs_equal(BAD_CONFIG, GOAL_CONFIG)
