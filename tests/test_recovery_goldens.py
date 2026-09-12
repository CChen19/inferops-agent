"""Week-3 P0-⑧: recovery goldens + CI gate.

Fixture / CPU only. Assertions consume Tune ⑧ recovery, Week-1
``is_promotable`` / ``derive_status``, and Week-2 ④/⑤/⑥. GPU-not-run
is not a pass. An empty fixture set is a fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.agent.graph import graph_invoke_config, session_thread_id
from inferops.agent.recovery import RECOVERY_FIELDS
from inferops.eval.recovery_goldens import (
    DEFAULT_FIXTURE_DIR,
    GPU_QUEUE_ENV,
    REQUIRED_GOLDEN_IDS,
    RESUME_EQUIVALENCE_FIELDS,
    TRAJECTORY_IDENTITY_KEYS,
    TUNE_MASTER_SHA,
    TUNE_TIP_SHA,
    _no_false_promote,
    comparable_terminal,
    evaluate_golden,
    load_catalog,
    load_golden_specs,
    recovery_golden_gate,
    required_ids_floor,
)
from inferops.eval.measurement_goldens import promotable_stub_result
from inferops.metrics import is_confirmed_promotable
from inferops.schemas import derive_status, is_promotable

FIXTURE_ROOT = Path(DEFAULT_FIXTURE_DIR)


def _spec(golden_id: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{golden_id}.json").read_text(encoding="utf-8"))


def test_catalog_requires_the_thin_p0_set():
    catalog = load_catalog()
    assert catalog["cpu_only"] is True
    assert catalog["gpu_queued"] is False
    assert tuple(catalog["required_ids"]) == REQUIRED_GOLDEN_IDS
    assert catalog["tune_contract"]["status"] == "frozen"
    assert catalog["tune_contract"]["tip_sha"] == TUNE_TIP_SHA
    assert catalog["tune_contract"]["master_sha"] == TUNE_MASTER_SHA
    for key in RECOVERY_FIELDS:
        assert key


def test_recovery_golden_gate_passes_cpu_fixtures():
    gate = recovery_golden_gate()
    assert gate.gpu_status == "not_run"
    assert gate.passed, gate.report()
    assert {c.golden_id for c in gate.cases} == set(REQUIRED_GOLDEN_IDS)
    assert all(c.ok for c in gate.cases)


def test_gpu_not_run_is_not_a_pass_on_empty_set(tmp_path: Path):
    """Skipping GPU cannot green-light an empty golden set."""
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.recovery_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": list(REQUIRED_GOLDEN_IDS),
                "tune_contract": {
                    "status": "frozen",
                    "tip_sha": TUNE_TIP_SHA,
                },
            }
        ),
        encoding="utf-8",
    )
    gate = recovery_golden_gate(tmp_path)
    assert gate.passed is False
    assert any("no CPU goldens" in f or "required goldens missing" in f for f in gate.failures)


def test_skipped_fixture_set_fails_closed(tmp_path: Path):
    """A present-but-skipped golden set must not green-light the gate."""
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.recovery_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": ["propose_tool_error"],
                "tune_contract": {
                    "status": "frozen",
                    "tip_sha": TUNE_TIP_SHA,
                    "master_sha": TUNE_MASTER_SHA,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "propose_tool_error.json").write_text(
        json.dumps(
            {
                "id": "propose_tool_error",
                "schema": "inferops.recovery_goldens.v1",
                "synthetic": True,
                "gpu_sampled": False,
                "skip": True,
                "driver": "propose_tool_error",
                "expect": {"confirmed_promotable": False},
            }
        ),
        encoding="utf-8",
    )
    gate = recovery_golden_gate(tmp_path)
    assert gate.passed is False
    assert any("skipped" in f for f in gate.failures)


def test_catalog_subset_and_deleted_fixtures_fail_floor(tmp_path: Path):
    """A non-empty catalog subset + deleted fixtures must still FAIL missing ids."""
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.recovery_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": ["propose_tool_error"],
                "tune_contract": {
                    "status": "frozen",
                    "tip_sha": TUNE_TIP_SHA,
                    "master_sha": TUNE_MASTER_SHA,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "propose_tool_error.json").write_text(
        (FIXTURE_ROOT / "propose_tool_error.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    gate = recovery_golden_gate(tmp_path)
    assert gate.passed is False
    blob = " ".join(gate.failures)
    assert "REQUIRED_GOLDEN_IDS floor" in blob
    assert "required goldens missing" in blob
    assert required_ids_floor({"required_ids": ["propose_tool_error"]}) == REQUIRED_GOLDEN_IDS


def test_blocked_catalog_fails_closed(tmp_path: Path):
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.recovery_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": list(REQUIRED_GOLDEN_IDS),
                "tune_contract": {"status": "blocked", "tip_sha": None},
            }
        ),
        encoding="utf-8",
    )
    gate = recovery_golden_gate(tmp_path)
    assert gate.passed is False
    assert any("Blocked" in f or "Tune" in f for f in gate.failures)


def test_invented_gpu_number_fails_the_gate():
    spec = _spec("benchmark_no_result")
    spec["expect"]["gpu_utilization_pct"] = 88.0
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("invented" in f and "gpu_utilization_pct" in f for f in case.failures)


def test_gpu_sampled_without_queue_fails(monkeypatch):
    spec = _spec("propose_tool_error")
    spec["gpu_sampled"] = True
    monkeypatch.delenv(GPU_QUEUE_ENV, raising=False)
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("GPU-not-run" in f for f in case.failures)


def test_resume_equivalence_compares_full_end_state():
    keys = set(comparable_terminal({}).keys())
    assert tuple(sorted(RESUME_EQUIVALENCE_FIELDS)) == tuple(sorted(keys))
    for field in (
        "next_action",
        "best_run_id",
        "best_promotable",
        "summary_run_ids",
        "last_result_run_id",
        "last_result_validity_status",
        "confirmation_verdict",
        "confirmation_phase",
        "confirmation_search_winner",
        "confirmation_bound_run_ids",
        "reflect_cited_run_ids",
        "trajectory_identity",
    ):
        assert field in RESUME_EQUIVALENCE_FIELDS
    assert TRAJECTORY_IDENTITY_KEYS == (
        "node",
        "action_kind",
        "experiment_id",
        "run_id",
        "validity_status",
        "promoted_to_best",
    )


def test_search_winner_best_swap_fails_without_confirmed_flag():
    """⑤: search winner / partial must not become best even if flag stays false."""
    failures = _no_false_promote(
        {
            "baseline_summary": {
                "experiment_id": "sess_baseline",
                "run_id": "base-search-rid",
            },
            "best_summary": {
                "experiment_id": "sess_max_num_batched_tokens_4096",
                "run_id": "search-win-rid",
                "confirmed_promotable": False,
            },
            "last_result": None,
            "confirmation_decision": None,
            "trajectory": [],
        },
        "confirmation_mid_fail",
        starting_best={
            "experiment_id": "sess_baseline",
            "run_id": "base-search-rid",
        },
    )
    assert any("swapped" in f and "search winner" in f for f in failures)


def test_confirmation_mid_fail_keeps_best_identity():
    spec = _spec("confirmation_mid_fail")
    case = evaluate_golden(spec)
    assert case.ok, case.failures
    assert case.terminal is not None
    assert case.terminal["best_experiment_id"] == "sess_baseline"
    assert case.terminal["best_run_id"] == "bb" * 16
    assert case.terminal["best_confirmed_promotable"] is False


def test_single_tool_call_does_not_hide_confirm_calls():
    spec = _spec("pre_tool_interrupt")
    spec["expect"]["single_tool_call"] = True
    case = evaluate_golden(spec)
    assert case.ok, case.failures


def test_loosening_confirmed_promotable_fails_golden():
    spec = _spec("confirmation_mid_fail")
    spec["expect"]["confirmed_promotable"] = True
    case = evaluate_golden(spec)
    assert case.ok is False


def test_week1_and_week2_gates_unchanged():
    stub = promotable_stub_result()
    assert is_promotable(stub) is True
    assert is_confirmed_promotable(stub, None) is False
    assert (
        derive_status(
            evidence=stub.config_evidence,
            actual_config=stub.actual_config,
            requested_config=stub.requested_config,
            successful_requests=stub.successful_requests,
        ).value
        == "valid"
    )


def test_tune_checkpoint_entrypoints_are_consumed():
    assert session_thread_id("sess_") == "sess"
    assert graph_invoke_config("sess_")["configurable"]["thread_id"] == "sess"


def test_every_fixture_is_synthetic_cpu():
    for spec in load_golden_specs():
        assert spec.get("synthetic") is True
        assert spec.get("gpu_sampled") is False
        case = evaluate_golden(spec)
        assert case.ok, (spec["id"], case.failures)


@pytest.mark.parametrize("golden_id", REQUIRED_GOLDEN_IDS)
def test_each_required_golden(golden_id: str):
    case = evaluate_golden(_spec(golden_id))
    assert case.ok, case.failures
