"""Week-3 ⑦ closeout: error-memory goldens + CI gate.

Fixture / CPU only. Consumes Week-1 is_promotable and experiment memory.
GPU-not-run is not a pass. Empty / skipped sets FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.eval.error_memory_goldens import (
    DEFAULT_FIXTURE_DIR,
    GPU_QUEUE_ENV,
    REQUIRED_GOLDEN_IDS,
    evaluate_golden,
    error_memory_golden_gate,
    load_catalog,
    load_golden_specs,
)
from inferops.memory.db import get_promotable_result, query_results, save_result
from inferops.schemas import derive_status, is_promotable

FIXTURE_ROOT = Path(DEFAULT_FIXTURE_DIR)


def _spec(golden_id: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{golden_id}.json").read_text(encoding="utf-8"))


def test_catalog_requires_error_memory_set():
    catalog = load_catalog()
    assert catalog["cpu_only"] is True
    assert catalog["gpu_queued"] is False
    assert tuple(catalog["required_ids"]) == REQUIRED_GOLDEN_IDS


def test_error_memory_gate_passes_cpu_fixtures():
    gate = error_memory_golden_gate()
    assert gate.gpu_status == "not_run"
    assert gate.passed, gate.report()
    assert {c.golden_id for c in gate.cases} == set(REQUIRED_GOLDEN_IDS)
    assert all(c.ok for c in gate.cases)


def test_gpu_not_run_is_not_a_pass_on_empty_set(tmp_path: Path):
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.error_memory_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": list(REQUIRED_GOLDEN_IDS),
            }
        ),
        encoding="utf-8",
    )
    gate = error_memory_golden_gate(tmp_path)
    assert gate.passed is False
    assert any("no CPU goldens" in f or "required goldens missing" in f for f in gate.failures)


def test_skipped_golden_is_not_a_pass():
    spec = _spec("failed_row_remembered_not_promotable")
    spec["skip"] = True
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("skipped" in f for f in case.failures)


def test_catalog_subset_fails_floor(tmp_path: Path):
    shrunk = list(REQUIRED_GOLDEN_IDS[:-1])
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.error_memory_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": shrunk,
            }
        ),
        encoding="utf-8",
    )
    for gid in shrunk:
        (tmp_path / f"{gid}.json").write_text(
            json.dumps(_spec(gid)),
            encoding="utf-8",
        )
    gate = error_memory_golden_gate(tmp_path)
    assert gate.passed is False
    assert any("shrunk below floor" in f or "required goldens missing" in f for f in gate.failures)


def test_invented_gpu_number_fails_the_gate():
    spec = _spec("failed_row_remembered_not_promotable")
    spec["gpu_utilization_pct"] = 88.0
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("invented" in f and "gpu_utilization_pct" in f for f in case.failures)


def test_gpu_sampled_without_queue_fails(monkeypatch):
    spec = _spec("failed_row_remembered_not_promotable")
    spec["gpu_sampled"] = True
    monkeypatch.delenv(GPU_QUEUE_ENV, raising=False)
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("GPU-not-run" in f for f in case.failures)


def test_loosening_is_promotable_true_fails():
    spec = _spec("unevidenced_high_score_not_promotable")
    spec["expect"]["is_promotable"] = True
    case = evaluate_golden(spec)
    assert case.ok is False


@pytest.mark.parametrize("golden_id", REQUIRED_GOLDEN_IDS)
def test_each_required_golden(golden_id: str):
    case = evaluate_golden(_spec(golden_id))
    assert case.ok, (golden_id, case.failures)


def test_week1_promotable_gate_unchanged(result_b, result_b_unevidenced):
    assert is_promotable(result_b) is True
    assert is_promotable(result_b_unevidenced) is False
    assert (
        derive_status(
            evidence=result_b.config_evidence,
            actual_config=result_b.actual_config,
            requested_config=result_b.requested_config,
            successful_requests=result_b.successful_requests,
        ).value
        == "valid"
    )


def test_promotable_only_uses_existing_memory_api(result_b, result_b_unevidenced, tmp_db):
    save_result(result_b, db_path=tmp_db)
    save_result(result_b_unevidenced, db_path=tmp_db)
    promo = query_results(top_k=10, db_path=tmp_db, promotable_only=True)
    assert [r["experiment_id"] for r in promo] == [result_b.experiment_id]
    assert get_promotable_result(result_b_unevidenced.experiment_id, db_path=tmp_db) is None


def test_every_fixture_is_synthetic_cpu():
    for spec in load_golden_specs():
        assert spec.get("synthetic") is True
        assert spec.get("gpu_sampled") is False
        case = evaluate_golden(spec)
        assert case.ok, (spec["id"], case.failures)
