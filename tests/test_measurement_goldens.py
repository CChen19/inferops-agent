"""Week-2 P0-⑦: measurement-trust goldens + CI gate.

Fixture / CPU only. Assertions consume ④ ledger, ⑤ confirmation, and ⑥
Reflect conclusions already on master. GPU-not-run is not a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.eval.measurement_goldens import (
    DEFAULT_FIXTURE_DIR,
    REQUIRED_GOLDEN_IDS,
    GPU_QUEUE_ENV,
    evaluate_golden,
    load_catalog,
    load_golden_specs,
    measurement_trust_gate,
    promotable_stub_result,
    rps_ledger,
)
from inferops.metrics import (
    LEDGER_SCHEMA_VERSION,
    TokenCountSource,
    compute_tpot_ms,
    is_confirmed_promotable,
    recalculate_from_ledger,
    verdict_from_ledgers,
)
from inferops.metrics.confirm import RepeatPhase
from inferops.schemas import derive_status, is_promotable


FIXTURE_ROOT = Path(DEFAULT_FIXTURE_DIR)


def _spec(golden_id: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{golden_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Catalog / gate
# ---------------------------------------------------------------------------

def test_catalog_requires_the_thin_p0_set():
    catalog = load_catalog()
    assert catalog["cpu_only"] is True
    assert catalog["gpu_queued"] is False
    assert tuple(catalog["required_ids"]) == REQUIRED_GOLDEN_IDS
    assert LEDGER_SCHEMA_VERSION == "2"


def test_measurement_trust_gate_passes_cpu_fixtures(result_b):
    gate = measurement_trust_gate(result=result_b)
    assert gate.gpu_status == "not_run"
    assert gate.passed, gate.report()
    assert {c.golden_id for c in gate.cases} == set(REQUIRED_GOLDEN_IDS)
    assert all(c.ok for c in gate.cases)


def test_gpu_not_run_is_not_a_pass_on_empty_set(tmp_path: Path):
    """Skipping GPU cannot green-light an empty golden set."""
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "inferops.measurement_goldens.v1",
                "cpu_only": True,
                "gpu_queued": False,
                "required_ids": list(REQUIRED_GOLDEN_IDS),
            }
        ),
        encoding="utf-8",
    )
    gate = measurement_trust_gate(tmp_path)
    assert gate.passed is False
    assert any("no CPU goldens" in f or "required goldens missing" in f for f in gate.failures)


def test_invented_gpu_number_fails_the_gate():
    spec = _spec("tpot_na")
    spec["expect"]["aggregates"]["tpot_na_rows"]["gpu_utilization_pct"] = 88.0
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("invented" in f and "gpu_utilization_pct" in f for f in case.failures)


def test_gpu_sampled_without_queue_fails(monkeypatch):
    spec = _spec("missing_requests")
    spec["gpu_sampled"] = True
    monkeypatch.delenv(GPU_QUEUE_ENV, raising=False)
    case = evaluate_golden(spec)
    assert case.ok is False
    assert any("GPU-not-run" in f for f in case.failures)


# ---------------------------------------------------------------------------
# Per-golden trust rules (anti-loosening)
# ---------------------------------------------------------------------------

def test_missing_requests_none_is_not_zero():
    case = evaluate_golden(_spec("missing_requests"))
    assert case.ok, case.failures
    empty = next(
        raw for raw in _spec("missing_requests")["ledgers"] if raw["role"] == "empty"
    )
    from inferops.eval.measurement_goldens import ledger_from_explicit

    agg = recalculate_from_ledger(ledger_from_explicit(empty))
    assert agg.error_rate is None
    assert agg.throughput_rps is None
    assert agg.tokens_per_second is None
    assert agg.tpot.p50 is None
    assert agg.gpu_utilization_pct is None
    assert agg.error_rate != 0
    assert agg.throughput_rps != 0


def test_missing_token_provenance_is_ignored_for_tok_s():
    from inferops.eval.measurement_goldens import ledger_from_explicit

    lost = next(
        raw
        for raw in _spec("missing_requests")["ledgers"]
        if raw["role"] == "lost_plus_unproven_tokens"
    )
    ledger = ledger_from_explicit(lost)
    unproven = ledger.by_request_id("req-unproven-tokens")
    assert unproven is not None
    assert unproven.token_count_source == TokenCountSource.MISSING
    assert unproven.output_tokens == 99
    assert unproven.tpot_ms is None
    agg = recalculate_from_ledger(ledger)
    assert agg.tokens_per_second is None
    assert agg.total_output_tokens is None
    assert agg.failed_requests == 1
    assert agg.successful_requests == 2


def test_failures_stay_in_error_denominator():
    case = evaluate_golden(_spec("failures_not_dropped"))
    assert case.ok, case.failures
    from inferops.eval.measurement_goldens import ledger_from_explicit

    mixed = next(
        raw
        for raw in _spec("failures_not_dropped")["ledgers"]
        if raw["role"] == "mixed_errors"
    )
    agg = recalculate_from_ledger(ledger_from_explicit(mixed))
    assert agg.total_requests == 5
    assert agg.successful_requests == 1
    assert agg.failed_requests == 4
    assert agg.error_rate == pytest.approx(0.8)
    assert agg.outcome_counts["incomplete"] == 1
    # Loosening: drop incomplete / cancel / timeout from the numerator.
    loosened = agg.failed_requests - 1
    assert loosened != 4


def test_tpot_na_never_writes_zero():
    case = evaluate_golden(_spec("tpot_na"))
    assert case.ok, case.failures
    assert compute_tpot_ms(e2e_ms=200.0, ttft_ms=100.0, output_tokens=0) is None
    assert compute_tpot_ms(e2e_ms=200.0, ttft_ms=100.0, output_tokens=1) is None
    assert (
        compute_tpot_ms(
            e2e_ms=400.0,
            ttft_ms=100.0,
            output_tokens=20,
            token_count_source=TokenCountSource.MISSING,
        )
        is None
    )
    from inferops.eval.measurement_goldens import ledger_from_explicit

    rows = next(raw for raw in _spec("tpot_na")["ledgers"] if raw["role"] == "tpot_na_rows")
    ledger = ledger_from_explicit(rows)
    assert ledger.by_request_id("single-out").tpot_ms is None
    assert ledger.by_request_id("zero-out").tpot_ms is None
    assert ledger.by_request_id("missing-provenance").tpot_ms is None
    agg = recalculate_from_ledger(ledger)
    assert agg.tpot.sample_n == 0
    assert agg.tpot.p50 is None
    assert agg.tpot.p50 != 0


def test_search_win_without_confirm_does_not_promote(result_b):
    spec = _spec("search_win_unconfirmed")
    case = evaluate_golden(spec, result=result_b)
    assert case.ok, case.failures
    decision = verdict_from_ledgers(
        [rps_ledger("swb_00", rps=2.0)],
        [rps_ledger("swc_00", rps=3.0)],
        metric="throughput_rps",
        phase=RepeatPhase.SEARCH,
        min_pairs=1,
    )
    assert decision.search_winner is True
    assert decision.verdict.value == "no_diff"
    assert is_promotable(result_b) is True
    assert is_confirmed_promotable(result_b, decision) is False


def test_loosening_search_win_to_confirmed_fails_golden(result_b):
    spec = _spec("search_win_unconfirmed")
    spec["expect"]["confirmed_promotable"] = True
    spec["expect"]["verdict"] = "confirmed_improvement"
    case = evaluate_golden(spec, result=result_b)
    assert case.ok is False
    assert any("is_confirmed_promotable" in f or "verdict=" in f for f in case.failures)


def test_too_noisy_is_not_confirmed(result_b):
    case = evaluate_golden(_spec("too_noisy"), result=result_b)
    assert case.ok, case.failures


def test_no_reliable_improvement_is_first_class_stop(result_b):
    case = evaluate_golden(_spec("no_reliable_improvement"), result=result_b)
    assert case.ok, case.failures


def test_loosening_no_diff_to_promote_fails_golden(result_b):
    spec = _spec("no_reliable_improvement")
    spec["expect"]["reflect"]["next_action"] = "continue"
    spec["expect"]["reflect"]["promote"] = True
    spec["expect"]["confirmed_promotable"] = True
    case = evaluate_golden(spec, result=result_b)
    assert case.ok is False


def test_week1_promotable_gate_unchanged(result_b, result_b_unevidenced):
    """⑦ must not loosen ①."""
    assert is_promotable(result_b) is True
    assert is_promotable(result_b_unevidenced) is False
    stub = promotable_stub_result()
    assert is_promotable(stub) is True
    assert (
        derive_status(
            evidence=result_b.config_evidence,
            actual_config=result_b.actual_config,
            requested_config=result_b.requested_config,
            successful_requests=result_b.successful_requests,
        ).value
        == "valid"
    )


def test_every_fixture_is_synthetic_cpu_and_schema_v2():
    for spec in load_golden_specs():
        assert spec.get("synthetic") is True
        assert spec.get("gpu_sampled") is False
        for raw in spec.get("ledgers") or []:
            ledger_id = raw["run_id"]
            assert ledger_id
        case = evaluate_golden(spec)
        assert case.ok, (spec["id"], case.failures)
