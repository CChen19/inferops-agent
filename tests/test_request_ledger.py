"""Week-2 P0-④: metric definition + request-ledger regressions.

Deterministic timestamp fixtures prove TTFT / TPOT / throughput / error rate.
Independent recalculation from ledger must match the reported aggregate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.metrics.aggregate import (
    format_aggregate_report,
    recalculate_from_ledger,
)
from inferops.metrics.report import report_from_ledger, report_from_result
from inferops.metrics.definitions import (
    TPOT_MIN_OUTPUT_TOKENS,
    classify_http_outcome,
    compute_tpot_ms,
)
from inferops.metrics.ledger import (
    RequestLedger,
    RequestOutcome,
    RequestRecord,
    RunConditions,
    TerminationReason,
    load_ledger,
    persist_ledger,
)
from inferops.schemas import (
    ExperimentValidityStatus,
    empty_latency,
    is_promotable,
)


RUN_ID = "fixture_run_aaaaaaaaaaaaaaaaaaaa"


def _rec(
    request_id: str,
    *,
    t0: float,
    t_first: float | None,
    t_end: float,
    output_tokens: int | None,
    outcome: RequestOutcome,
    termination: TerminationReason,
    input_tokens: int | None = 10,
    is_warmup: bool = False,
    error: str = "",
) -> RequestRecord:
    ttft = (t_first - t0) * 1000.0 if t_first is not None else None
    e2e = (t_end - t0) * 1000.0
    return RequestRecord(
        run_id=RUN_ID,
        request_id=request_id,
        is_warmup=is_warmup,
        t_start_s=t0,
        t_first_token_s=t_first,
        t_end_s=t_end,
        ttft_ms=ttft,
        e2e_ms=e2e,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        outcome=outcome,
        termination_reason=termination,
        error=error,
    )


def _mixed_ledger() -> RequestLedger:
    """Canonical fixture covering success/fail/timeout/cancel/truncate/zero/single."""
    ledger = RequestLedger(
        run_id=RUN_ID,
        conditions=RunConditions(
            workload_name="chat_short",
            num_requests=7,
            concurrency=2,
            warmup_requests=1,
            stream_response=True,
            arrival_rps=None,
            sampling_temperature=0.0,
            cache_enabled=False,
        ),
        window_start_s=1000.0,
        window_end_s=1010.0,  # 10s window
    )
    ledger.add(
        _rec(
            "warmup-0001",
            t0=999.0,
            t_first=999.05,
            t_end=999.5,
            output_tokens=8,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
            is_warmup=True,
        )
    )
    # success: TTFT=100ms, e2e=1100ms, tokens=11 → TPOT=(1100-100)/10=100
    ledger.add(
        _rec(
            "req-0001",
            t0=1000.0,
            t_first=1000.1,
            t_end=1001.1,
            output_tokens=11,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    # success: TTFT=200ms, e2e=1200ms, tokens=6 → TPOT=200
    ledger.add(
        _rec(
            "req-0002",
            t0=1001.0,
            t_first=1001.2,
            t_end=1002.2,
            output_tokens=6,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    # truncate (still latency-eligible)
    ledger.add(
        _rec(
            "req-0003",
            t0=1002.0,
            t_first=1002.05,
            t_end=1003.05,
            output_tokens=4,
            outcome=RequestOutcome.TRUNCATE,
            termination=TerminationReason.LENGTH,
        )
    )
    ledger.add(
        _rec(
            "req-0004",
            t0=1003.0,
            t_first=None,
            t_end=1003.5,
            output_tokens=0,
            outcome=RequestOutcome.FAIL,
            termination=TerminationReason.ERROR,
            error="HTTP 500",
        )
    )
    ledger.add(
        _rec(
            "req-0005",
            t0=1004.0,
            t_first=None,
            t_end=1006.0,
            output_tokens=0,
            outcome=RequestOutcome.TIMEOUT,
            termination=TerminationReason.TIMEOUT,
            error="timeout",
        )
    )
    ledger.add(
        _rec(
            "req-0006",
            t0=1005.0,
            t_first=None,
            t_end=1005.2,
            output_tokens=0,
            outcome=RequestOutcome.CANCEL,
            termination=TerminationReason.CANCEL,
            error="cancelled",
        )
    )
    ledger.add(
        _rec(
            "req-0007",
            t0=1006.0,
            t_first=None,
            t_end=1006.1,
            output_tokens=0,
            outcome=RequestOutcome.INCOMPLETE,
            termination=TerminationReason.INCOMPLETE,
            error="lost",
        )
    )
    return ledger


def test_tpot_na_for_zero_and_single_output_token():
    assert compute_tpot_ms(e2e_ms=100.0, ttft_ms=10.0, output_tokens=0) is None
    assert compute_tpot_ms(e2e_ms=100.0, ttft_ms=10.0, output_tokens=1) is None
    assert TPOT_MIN_OUTPUT_TOKENS == 2


def test_tpot_computed_for_two_plus_tokens():
    assert compute_tpot_ms(e2e_ms=1100.0, ttft_ms=100.0, output_tokens=11) == 100.0
    assert compute_tpot_ms(e2e_ms=300.0, ttft_ms=100.0, output_tokens=2) == 200.0


def test_request_record_never_stores_tpot_zero_for_single_token():
    rec = RequestRecord(
        run_id=RUN_ID,
        request_id="single",
        t_start_s=0.0,
        t_first_token_s=0.1,
        t_end_s=0.2,
        ttft_ms=100.0,
        e2e_ms=200.0,
        tpot_ms=0.0,
        output_tokens=1,
        outcome=RequestOutcome.SUCCESS,
        termination_reason=TerminationReason.STOP,
    )
    assert rec.tpot_ms is None


def test_recalculate_throughput_ttft_tpot_error_rate():
    ledger = _mixed_ledger()
    agg = recalculate_from_ledger(ledger)

    assert agg.total_requests == 7
    assert agg.successful_requests == 3
    assert agg.failed_requests == 4
    assert agg.error_rate == pytest.approx(4 / 7)

    assert agg.total_time_s == pytest.approx(10.0)
    assert agg.throughput_rps == pytest.approx(0.3)
    assert agg.tokens_per_second == pytest.approx(2.1)

    assert agg.ttft.sample_n == 3
    assert "client_ttft" in agg.ttft.sample_scope
    assert agg.ttft.p50 == pytest.approx(100.0)

    assert agg.tpot.sample_n == 3
    assert agg.tpot.p50 is not None
    assert agg.outcome_counts.get("incomplete") == 1


def test_single_output_token_excluded_from_tpot_aggregate():
    ledger = RequestLedger(run_id=RUN_ID, window_start_s=0.0, window_end_s=1.0)
    ledger.add(
        _rec(
            "one-tok",
            t0=0.0,
            t_first=0.05,
            t_end=0.15,
            output_tokens=1,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    ledger.add(
        _rec(
            "multi-tok",
            t0=0.1,
            t_first=0.15,
            t_end=0.35,
            output_tokens=5,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    agg = recalculate_from_ledger(ledger)
    assert agg.tpot.sample_n == 1
    # e2e=(0.35-0.1)*1000=250, ttft=50, n=5 → (250-50)/4 = 50
    assert agg.tpot.p50 == pytest.approx(50.0)


def test_zero_output_not_latency_success():
    ledger = RequestLedger(run_id=RUN_ID, window_start_s=0.0, window_end_s=1.0)
    ledger.add(
        RequestRecord(
            run_id=RUN_ID,
            request_id="z",
            t_start_s=0.0,
            t_end_s=0.5,
            e2e_ms=500.0,
            output_tokens=0,
            outcome=RequestOutcome.FAIL,
            termination_reason=TerminationReason.ZERO_OUTPUT,
        )
    )
    agg = recalculate_from_ledger(ledger)
    assert agg.successful_requests == 0
    assert agg.error_rate == 1.0
    assert agg.ttft.sample_n == 0
    assert agg.ttft.p50 is None
    assert agg.throughput_rps == pytest.approx(0.0)


def test_incomplete_never_packaged_as_success():
    outcome, term = classify_http_outcome(
        status_code=200,
        timed_out=False,
        cancelled=False,
        finish_reason=None,
        output_tokens=0,
        first_token_seen=False,
    )
    assert outcome == RequestOutcome.INCOMPLETE
    assert term == TerminationReason.INCOMPLETE

    ledger = RequestLedger(run_id=RUN_ID, window_start_s=0.0, window_end_s=2.0)
    ledger.add(
        _rec(
            "lost",
            t0=0.0,
            t_first=None,
            t_end=0.1,
            output_tokens=0,
            outcome=outcome,
            termination=term,
        )
    )
    agg = recalculate_from_ledger(ledger)
    assert agg.successful_requests == 0
    assert agg.failed_requests == 1
    assert agg.throughput_rps == pytest.approx(0.0)


def test_missing_metrics_never_default_to_zero():
    empty = RequestLedger(run_id=RUN_ID)
    agg = recalculate_from_ledger(empty)
    assert agg.total_requests == 0
    assert agg.error_rate is None
    assert agg.throughput_rps is None
    assert agg.tokens_per_second is None
    assert agg.ttft.p50 is None
    assert agg.tpot.p50 is None
    assert agg.e2e.p50 is None
    assert agg.gpu_utilization_pct is None
    assert agg.cost_usd is None

    lat = empty_latency()
    assert lat.p50 is None and lat.sample_n == 0


def test_gpu_and_cost_only_when_provided():
    ledger = _mixed_ledger()
    bare = recalculate_from_ledger(ledger)
    assert bare.gpu_utilization_pct is None
    assert bare.cost_usd is None

    with_gpu = recalculate_from_ledger(
        ledger, gpu_utilization_pct=88.5, gpu_memory_used_gb=3.2
    )
    assert with_gpu.gpu_utilization_pct == 88.5
    priced = recalculate_from_ledger(ledger, cost_usd=0.42)
    assert priced.cost_usd == 0.42


def test_independent_recalculation_matches_persisted_report(tmp_path: Path):
    ledger = _mixed_ledger()
    path = persist_ledger(ledger, tmp_path / "ledger.json")
    reloaded = load_ledger(path)
    a = recalculate_from_ledger(ledger)
    b = recalculate_from_ledger(reloaded)
    assert a.model_dump() == b.model_dump()

    report = format_aggregate_report(a)
    assert f"run_id={RUN_ID}" in report
    assert "error_rate" in report
    assert "n/a (no pricing assumptions)" in report
    assert "scope=`" in report


def test_ledger_json_roundtrip_keys(tmp_path: Path):
    ledger = _mixed_ledger()
    path = persist_ledger(ledger, tmp_path / "l.json")
    raw = json.loads(path.read_text())
    assert raw["run_id"] == RUN_ID
    assert raw["schema_version"] == "1"
    assert all("request_id" in r for r in raw["records"])
    assert any(r["is_warmup"] for r in raw["records"])


def test_validity_gate_preserved_with_ledger_fields(result_b, result):
    """P0-① gates unchanged: unevidenced ≠ best even with rich ledger metrics."""
    ledger = _mixed_ledger()
    unevidenced = result.model_copy(
        update={
            "throughput_rps": 99.0,
            "request_ledger": ledger.model_dump(mode="json"),
            "status": ExperimentValidityStatus.INSUFFICIENT_EVIDENCE,
            "actual_config": None,
            "config_evidence": None,
        }
    )
    assert is_promotable(unevidenced) is False
    assert is_promotable(result_b) is True


def test_report_from_result_matches_ledger(result):
    ledger = _mixed_ledger()
    attached = result.model_copy(
        update={
            "run_id": RUN_ID,
            "request_ledger": ledger.model_dump(mode="json"),
        }
    )
    from_result, md = report_from_result(attached)
    from_ledger, _ = report_from_ledger(
        ledger,
        gpu_utilization_pct=attached.gpu_utilization_pct,
        gpu_memory_used_gb=attached.gpu_memory_used_gb,
        cost_usd=attached.cost_usd,
    )
    assert from_result is not None
    assert from_result.model_dump() == from_ledger.model_dump()
    assert RUN_ID in md


def test_write_minimal_ledger_example():
    """Minimal ledger + matching recalculated report (PR deliverable)."""
    ledger = _mixed_ledger()
    out_dir = Path("reports/week2_request_ledger_examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = persist_ledger(ledger, out_dir / "minimal_ledger.json")
    agg = recalculate_from_ledger(ledger)
    report_path = out_dir / "minimal_report.md"
    report_path.write_text(
        "# Minimal recalculated report (synthetic fixture — not real perf)\n\n"
        + format_aggregate_report(agg)
        + f"\nLedger: `{ledger_path}`\n",
        encoding="utf-8",
    )
    assert ledger_path.is_file()
    assert "throughput_rps" in report_path.read_text()


def test_eof_without_done_or_finish_reason_is_incomplete():
    outcome, term = classify_http_outcome(
        status_code=200,
        timed_out=False,
        cancelled=False,
        finish_reason=None,
        output_tokens=None,
        first_token_seen=True,
        stream_response=True,
        saw_done=False,
    )
    assert outcome == RequestOutcome.INCOMPLETE
    assert term == TerminationReason.INCOMPLETE


def test_missing_output_tokens_excluded_from_tpot_and_tok_s():
    assert compute_tpot_ms(e2e_ms=200.0, ttft_ms=50.0, output_tokens=None) is None
    ledger = RequestLedger(run_id=RUN_ID, window_start_s=0.0, window_end_s=1.0)
    ledger.add(
        _rec(
            "no-usage",
            t0=0.0,
            t_first=0.05,
            t_end=0.25,
            output_tokens=None,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    agg = recalculate_from_ledger(ledger)
    assert agg.tpot.sample_n == 0
    assert agg.tpot.p50 is None
    assert agg.tokens_per_second is None
    assert agg.total_output_tokens is None


def test_duplicate_request_id_rejected():
    ledger = RequestLedger(run_id=RUN_ID)
    ledger.add(
        _rec(
            "dup",
            t0=0.0,
            t_first=0.01,
            t_end=0.02,
            output_tokens=2,
            outcome=RequestOutcome.SUCCESS,
            termination=TerminationReason.STOP,
        )
    )
    with pytest.raises(ValueError, match="duplicate request_id"):
        ledger.add(
            _rec(
                "dup",
                t0=0.1,
                t_first=0.11,
                t_end=0.12,
                output_tokens=2,
                outcome=RequestOutcome.SUCCESS,
                termination=TerminationReason.STOP,
            )
        )
    with pytest.raises(ValueError, match="duplicate request_id"):
        RequestLedger.model_validate(
            {
                "run_id": RUN_ID,
                "records": [
                    _rec(
                        "x",
                        t0=0.0,
                        t_first=0.01,
                        t_end=0.02,
                        output_tokens=2,
                        outcome=RequestOutcome.SUCCESS,
                        termination=TerminationReason.STOP,
                    ).model_dump(mode="json"),
                    _rec(
                        "x",
                        t0=0.1,
                        t_first=0.11,
                        t_end=0.12,
                        output_tokens=3,
                        outcome=RequestOutcome.SUCCESS,
                        termination=TerminationReason.STOP,
                    ).model_dump(mode="json"),
                ],
            }
        )


def test_mismatched_record_run_id_rejected():
    ledger = RequestLedger(run_id=RUN_ID)
    with pytest.raises(ValueError, match="does not match ledger"):
        ledger.add(
            RequestRecord(
                run_id="other_run",
                request_id="r1",
                t_start_s=0.0,
                t_end_s=0.1,
                outcome=RequestOutcome.FAIL,
                termination_reason=TerminationReason.ERROR,
            )
        )
    with pytest.raises(ValueError, match="does not match ledger"):
        RequestLedger.model_validate(
            {
                "run_id": RUN_ID,
                "records": [
                    {
                        "run_id": "other_run",
                        "request_id": "r1",
                        "t_start_s": 0.0,
                        "t_end_s": 0.1,
                        "outcome": "fail",
                        "termination_reason": "error",
                    }
                ],
            }
        )


def test_empty_embedded_ledger_is_recalculable(result):
    empty = RequestLedger(run_id=RUN_ID, records=[])
    attached = result.model_copy(
        update={
            "run_id": RUN_ID,
            "request_ledger": empty.model_dump(mode="json"),
        }
    )
    agg, md = report_from_result(attached)
    assert agg is not None
    assert agg.total_requests == 0
    assert agg.throughput_rps is None
    assert agg.ttft.p50 is None
    assert RUN_ID in md


def test_report_from_result_rejects_run_id_mismatch(result):
    ledger = RequestLedger(run_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", records=[])
    attached = result.model_copy(
        update={
            "run_id": RUN_ID,
            "request_ledger": ledger.model_dump(mode="json"),
        }
    )
    agg, msg = report_from_result(attached)
    assert agg is None
    assert "does not match" in msg


def test_missing_latency_stays_none_in_summary(result):
    from inferops.agent.state import summary_from_result

    missing = result.model_copy(
        update={
            "throughput_rps": None,
            "tokens_per_second": None,
            "ttft": empty_latency(),
            "tpot": empty_latency(),
            "e2e_latency": empty_latency(),
        }
    )
    summary = summary_from_result(
        missing,
        param_changed=None,
        value_changed=None,
        baseline_primary=2.0,
        primary_metric="throughput_rps",
    )
    assert summary["throughput_rps"] is None
    assert summary["tokens_per_second"] is None
    assert summary["ttft_p50_ms"] is None
    assert summary["ttft_p99_ms"] is None
    assert summary["e2e_p50_ms"] is None
    assert summary["vs_baseline_pct"] is None
