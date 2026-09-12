"""Unit tests for traffic load aggregation without a real HTTP server."""

from __future__ import annotations

import pytest

from inferops.schemas import WorkloadSpec
from inferops.tools import traffic
from inferops.tools.traffic import RequestMetrics, extract_percentiles, run_load


def test_extract_percentiles_empty_returns_nones():
    assert extract_percentiles([]) == {
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }


def test_extract_percentiles_nearest_rank_ceil():
    # sorted [10, 50, 90, 100]; p50 = ceil(0.5*4)-1 = 1 → 50
    out = extract_percentiles([100.0, 10.0, 50.0, 90.0])

    assert out["p50"] == 50.0
    assert out["p90"] == 100.0
    assert out["p99"] == 100.0


@pytest.mark.asyncio
async def test_run_load_aggregates_successful_requests(monkeypatch):
    from inferops.metrics.ledger import (
        RequestOutcome,
        RequestRecord,
        TerminationReason,
        TokenCountSource,
    )

    async def fake_send_one(
        client,
        base_url,
        prompt,
        max_tokens,
        *,
        run_id,
        request_id,
        is_warmup=False,
        stream_response=True,
        timeout_s=120.0,
    ):
        t0 = 1000.0
        if prompt == "fail":
            return RequestRecord(
                run_id=run_id,
                request_id=request_id,
                is_warmup=is_warmup,
                t_start_s=t0,
                t_end_s=t0 + 0.005,
                e2e_ms=5.0,
                output_tokens=0,
                outcome=RequestOutcome.FAIL,
                termination_reason=TerminationReason.ERROR,
                error="synthetic",
            )
        if prompt == "warmup":
            return RequestRecord(
                run_id=run_id,
                request_id=request_id,
                is_warmup=True,
                t_start_s=t0,
                t_first_token_s=t0 + 0.001,
                t_end_s=t0 + 0.002,
                ttft_ms=1.0,
                e2e_ms=2.0,
                output_tokens=1,
                outcome=RequestOutcome.SUCCESS,
                termination_reason=TerminationReason.STOP,
            )
        idx = int(prompt.removeprefix("p"))
        return RequestRecord(
            run_id=run_id,
            request_id=request_id,
            is_warmup=is_warmup,
            t_start_s=t0,
            t_first_token_s=t0 + (10.0 + idx) / 1000.0,
            t_end_s=t0 + (100.0 + idx) / 1000.0,
            ttft_ms=10.0 + idx,
            e2e_ms=100.0 + idx,
            output_tokens=2,
            token_count_source=TokenCountSource.USAGE,
            outcome=RequestOutcome.SUCCESS,
            termination_reason=TerminationReason.STOP,
        )

    monkeypatch.setattr(traffic, "_send_one", fake_send_one)
    workload = WorkloadSpec(
        name="unit",
        prompt_template="",
        num_requests=3,
        concurrency=2,
        input_len=1,
        output_len=4,
    )

    out = await run_load(
        "http://example.test",
        workload,
        prompts=["warmup", "p1", "fail", "p3"],
        warmup_requests=1,
        run_id="unitrun",
    )

    assert out.total_requests == 3
    assert out.successful == 2
    assert out.error_rate == pytest.approx(1 / 3)
    assert out.throughput_rps is not None and out.throughput_rps > 0
    assert out.tokens_per_second is not None and out.tokens_per_second > 0
    assert out.ttft_ms == [11.0, 13.0]
    assert out.e2e_ms == [101.0, 103.0]
    assert out.ledger is not None
    assert out.ledger.run_id == "unitrun"
    # fail is in ledger measured rows
    measured = out.ledger.measured()
    assert any(r.outcome == RequestOutcome.FAIL for r in measured)


@pytest.mark.asyncio
async def test_run_load_can_return_without_closing_client(monkeypatch):
    from inferops.metrics.ledger import RequestOutcome, RequestRecord, TerminationReason

    class FakeClient:
        def __init__(self, limits):
            self.limits = limits

        async def aclose(self):
            raise AssertionError("client close should not be awaited")

    async def fake_send_one(
        client,
        base_url,
        prompt,
        max_tokens,
        *,
        run_id,
        request_id,
        is_warmup=False,
        stream_response=True,
        timeout_s=120.0,
    ):
        return RequestRecord(
            run_id=run_id,
            request_id=request_id,
            is_warmup=is_warmup,
            t_start_s=1.0,
            t_first_token_s=1.01,
            t_end_s=1.1,
            ttft_ms=10.0,
            e2e_ms=100.0,
            output_tokens=2,
            outcome=RequestOutcome.SUCCESS,
            termination_reason=TerminationReason.STOP,
        )

    monkeypatch.setattr(traffic.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(traffic, "_send_one", fake_send_one)

    workload = WorkloadSpec(
        name="unit",
        prompt_template="",
        num_requests=1,
        concurrency=1,
        input_len=1,
        output_len=4,
    )

    out = await run_load(
        "http://example.test",
        workload,
        prompts=["warmup", "p1"],
        warmup_requests=1,
        close_client=False,
    )

    assert out.successful == 1


class _FakeStream:
    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


class _FakeStreamClient:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def stream(self, *a, **k):
        return _FakeStream(self._lines)


@pytest.mark.asyncio
async def test_stream_chunks_do_not_invent_tokens():
    from inferops.metrics.ledger import TokenCountSource
    from inferops.tools.traffic import _send_one

    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    rec = await _send_one(
        _FakeStreamClient(lines),
        "http://example.test",
        "hi",
        16,
        run_id="rid",
        request_id="req-1",
        stream_response=True,
    )
    assert rec.output_tokens is None
    assert rec.token_count_source == TokenCountSource.MISSING
    assert rec.tpot_ms is None
    assert rec.ttft_ms is not None
    assert rec.outcome.value == "success"  # stop + [DONE], tokens unknown


@pytest.mark.asyncio
async def test_stream_eof_without_done_is_incomplete():
    from inferops.tools.traffic import _send_one

    lines = [
        'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}',
    ]
    rec = await _send_one(
        _FakeStreamClient(lines),
        "http://example.test",
        "hi",
        16,
        run_id="rid",
        request_id="req-eof",
        stream_response=True,
    )
    assert rec.outcome.value == "incomplete"
    assert rec.termination_reason.value == "incomplete"
    assert rec.output_tokens is None
    assert rec.tpot_ms is None


@pytest.mark.asyncio
async def test_stream_usage_tokens_are_recorded():
    from inferops.metrics.ledger import TokenCountSource
    from inferops.tools.traffic import _send_one

    lines = [
        'data: {"choices":[{"delta":{"content":"ab"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":5}}',
        "data: [DONE]",
    ]
    rec = await _send_one(
        _FakeStreamClient(lines),
        "http://example.test",
        "hi",
        16,
        run_id="rid",
        request_id="req-usage",
        stream_response=True,
    )
    assert rec.output_tokens == 5
    assert rec.input_tokens == 3
    assert rec.token_count_source == TokenCountSource.USAGE
    assert rec.outcome.value == "success"
