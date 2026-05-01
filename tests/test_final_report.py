"""Unit tests for write_final_report tool."""

from __future__ import annotations

from inferops.tools.final_report import FinalReportInput, FinalReportOutput, write_final_report

_BASELINE = {
    "experiment_id": "sess_baseline",
    "param_changed": None,
    "value_changed": None,
    "throughput_rps": 15.0,
    "tokens_per_second": 1900.0,
    "ttft_p50_ms": 48.0,
    "ttft_p99_ms": 70.0,
    "e2e_p50_ms": 900.0,
    "bottleneck": "compute-bound",
    "vs_baseline_pct": 0.0,
}

_BEST = {
    "experiment_id": "sess_chunked_v2",
    "param_changed": "enable_chunked_prefill",
    "value_changed": True,
    "throughput_rps": 17.2,
    "tokens_per_second": 2200.0,
    "ttft_p50_ms": 45.0,
    "ttft_p99_ms": 60.0,
    "e2e_p50_ms": 820.0,
    "bottleneck": "compute-bound",
    "vs_baseline_pct": 14.7,
}


def test_write_final_report_creates_file(tmp_path):
    out_path = tmp_path / "report.md"
    inp = FinalReportInput(
        workload_name="chat_short",
        session_prefix="sess_",
        experiment_summaries=[_BASELINE, _BEST],
        baseline_summary=_BASELINE,
        best_summary=_BEST,
        output_path=str(out_path),
    )

    result = write_final_report(inp)

    assert out_path.exists()
    assert isinstance(result, FinalReportOutput)
    assert result.output_path == str(out_path)
    assert result.improvement_pct == pytest.approx(14.7)


def test_write_final_report_includes_experiment_table(tmp_path):
    out_path = tmp_path / "r.md"
    inp = FinalReportInput(
        workload_name="chat_short",
        session_prefix="s_",
        experiment_summaries=[_BASELINE, _BEST],
        baseline_summary=_BASELINE,
        best_summary=_BEST,
        output_path=str(out_path),
    )
    write_final_report(inp)
    text = out_path.read_text()

    assert "Experiment Log" in text
    assert "sess_baseline" in text
    assert "sess_chunked_v2" in text
    assert "+14.7%" in text


def test_write_final_report_includes_citations(tmp_path):
    out_path = tmp_path / "r.md"
    inp = FinalReportInput(
        workload_name="chat_short",
        session_prefix="s_",
        experiment_summaries=[_BASELINE],
        citations=["[source: vllm_scheduler] chunked prefill reduces TTFT p99 by 29%"],
        output_path=str(out_path),
    )
    write_final_report(inp)
    text = out_path.read_text()

    assert "Knowledge Citations" in text
    assert "vllm_scheduler" in text


def test_write_final_report_counts_sections(tmp_path):
    out_path = tmp_path / "r.md"
    inp = FinalReportInput(
        workload_name="chat_short",
        session_prefix="s_",
        experiment_summaries=[_BASELINE, _BEST],
        baseline_summary=_BASELINE,
        best_summary=_BEST,
        citations=["[source: x] note"],
        output_path=str(out_path),
    )
    result = write_final_report(inp)
    # Executive Summary + Experiment Log + Citations + Recommendation = 4
    assert result.sections_written == 4


import pytest
