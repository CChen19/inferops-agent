"""Week-3 ⑦ closeout: unified golden manifest + fail-closed CI gate."""

from __future__ import annotations

import json
from pathlib import Path

from inferops.eval.error_memory_goldens import REQUIRED_GOLDEN_IDS as ERROR_MEMORY_IDS
from inferops.eval.measurement_goldens import REQUIRED_GOLDEN_IDS as MEASUREMENT_IDS
from inferops.eval.recovery_goldens import REQUIRED_GOLDEN_IDS as RECOVERY_IDS
from inferops.eval.unified_goldens import (
    DEFAULT_MANIFEST,
    MAX_CASES,
    MIN_CASES,
    REQUIRED_CASE_FIELDS,
    gpu_layer_status,
    load_manifest,
    real_llm_layer_status,
    unified_golden_gate,
)


def test_manifest_has_required_metadata_and_count():
    manifest = load_manifest()
    cases = manifest["cases"]
    assert MIN_CASES <= len(cases) <= MAX_CASES
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    sources = {case["source"] for case in cases}
    assert sources == {
        "measurement_goldens",
        "recovery_goldens",
        "error_memory_goldens",
    }
    for case in cases:
        for key in REQUIRED_CASE_FIELDS:
            assert key in case, (case.get("id"), key)
        assert case["holdout"] in (True, False)
        assert case["reviewer"]
        assert case["expected_behavior"]
        assert case["judge_rule"]
        assert case.get("skip") is not True
    holdout = [case["id"] for case in cases if case["holdout"] is True]
    assert holdout, "at least one holdout case is required"
    by_source = {}
    for case in cases:
        by_source.setdefault(case["source"], set()).add(case["id"])
    assert set(MEASUREMENT_IDS) <= by_source["measurement_goldens"]
    assert set(RECOVERY_IDS) <= by_source["recovery_goldens"]
    assert set(ERROR_MEMORY_IDS) <= by_source["error_memory_goldens"]


def test_unified_gate_passes_cpu_fixtures():
    gate = unified_golden_gate()
    assert gate.passed, gate.report()
    assert gate.offline_fixture.passed is True
    assert gate.gpu.status == "blocked"
    assert gate.gpu.passed is False
    assert "未执行" in gate.gpu.detail or "Blocked" in gate.gpu.detail
    assert gate.real_llm.status == "separate"
    assert "tpot_na" in gate.holdout_ids


def test_empty_manifest_fails(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": "inferops.unified_goldens.v1", "cases": []}),
        encoding="utf-8",
    )
    gate = unified_golden_gate(path)
    assert gate.passed is False
    assert any("empty" in f for f in gate.failures)


def test_skipped_manifest_case_fails(tmp_path: Path):
    manifest = load_manifest()
    manifest["cases"][0]["skip"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = unified_golden_gate(path)
    assert gate.passed is False
    assert any("skipped" in f for f in gate.failures)


def test_shrinking_below_20_fails(tmp_path: Path):
    manifest = load_manifest()
    manifest["cases"] = manifest["cases"][:10]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = unified_golden_gate(path)
    assert gate.passed is False
    assert any("20" in f and "30" in f for f in gate.failures)


def test_gpu_layer_never_marks_pass():
    gpu = gpu_layer_status()
    assert gpu.passed is False
    assert gpu.status in {"blocked", "queued"}
    if gpu.status == "blocked":
        assert "未执行" in gpu.detail


def test_fake_llm_campaign_cannot_pass_as_live():
    layer = real_llm_layer_status(
        {
            "layer": "real_llm",
            "status": "ran",
            "passed": True,
            "llm_boundary": "fake_scripted",
            "summary": "should not count",
        }
    )
    assert layer.status == "mislabeled"
    assert layer.passed is False
    gate = unified_golden_gate(
        real_llm_campaign={
            "layer": "real_llm",
            "status": "ran",
            "passed": True,
            "llm_boundary": "fake_scripted",
        }
    )
    assert gate.passed is False
    assert any("fake" in f or "mislabeled" in f or "live" in f for f in gate.failures)


def test_blocked_real_llm_is_not_a_pass():
    layer = real_llm_layer_status(
        {
            "layer": "real_llm",
            "status": "blocked",
            "passed": False,
            "blocker": "OPENROUTER_API_KEY missing",
        }
    )
    assert layer.status == "blocked"
    assert layer.passed is False


def test_invented_tune8_field_on_a_case_fails(tmp_path: Path):
    manifest = load_manifest()
    manifest["cases"][0]["tune8_fields"] = ["receipt_lost"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    gate = unified_golden_gate(path)
    assert gate.passed is False
    assert any("receipt" in f or "invent" in f for f in gate.failures)


def test_default_manifest_path_exists():
    assert DEFAULT_MANIFEST.is_file()
