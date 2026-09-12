"""Async load generator: per-request ledger + client TTFT / TPOT / throughput.

Week-2 P0-④ rules (encoded here + inferops.metrics):
  - Every measured request becomes a RequestRecord (run_id + request_id).
  - Client TTFT only when streaming observes a first token — never invent
    TTFT from E2E on non-stream / no-token paths.
  - Per-request TPOT = (e2e − ttft) / (output_tokens − 1); output_tokens < 2
    → TPOT None (never 0).
  - Actual input/output tokens only (no max(..., 1) phantom tokens).
  - Failures / timeouts / cancels / incomplete enter error-rate denominator.
  - Aggregates come from recalculate_from_ledger — not ad-hoc averages.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

import httpx

from inferops.metrics.aggregate import AggregateMetrics, recalculate_from_ledger
from inferops.metrics.definitions import classify_http_outcome, compute_tpot_ms
from inferops.metrics.ledger import (
    RequestLedger,
    RequestOutcome,
    RequestRecord,
    RunConditions,
    TerminationReason,
)
from inferops.schemas import WorkloadSpec


@dataclass
class RequestMetrics:
    """Legacy shim kept for older unit tests; prefer RequestRecord."""

    success: bool
    ttft_ms: float | None
    e2e_ms: float | None
    output_tokens: int
    error: str = ""
    request_id: str = ""
    outcome: str = ""
    tpot_ms: float | None = None
    input_tokens: int | None = None


@dataclass
class LoadResult:
    total_requests: int
    successful: int
    total_time_s: float
    throughput_rps: float | None
    tokens_per_second: float | None
    ttft_ms: list[float]
    e2e_ms: list[float]
    error_rate: float | None = None
    ledger: RequestLedger | None = None
    aggregate: AggregateMetrics | None = None
    records: list[RequestRecord] = field(default_factory=list)


def _parse_sse_chunk(chunk: str) -> dict:
    try:
        return json.loads(chunk)
    except Exception:
        return {}


def _content_from_chunk(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content:
        return str(content)
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def _finish_reason_from_chunk(data: dict) -> str | None:
    choices = data.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


async def _send_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    *,
    run_id: str,
    request_id: str,
    is_warmup: bool = False,
    stream_response: bool = True,
    timeout_s: float = 120.0,
) -> RequestRecord:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream_response,
        "temperature": 0.0,
        "stream_options": {"include_usage": True} if stream_response else None,
    }
    # Remove None keys (non-stream must not send stream_options)
    payload = {k: v for k, v in payload.items() if v is not None}

    t_start = time.time()
    t_first: float | None = None
    t_end: float | None = None
    output_tokens = 0
    input_tokens: int | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    timed_out = False
    cancelled = False
    error = ""
    first_token_seen = False

    try:
        if not stream_response:
            resp = await client.post(url, json=payload, timeout=timeout_s)
            t_end = time.time()
            http_status = resp.status_code
            if resp.status_code != 200:
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                body = resp.json()
                usage = body.get("usage") or {}
                try:
                    output_tokens = int(usage.get("completion_tokens") or 0)
                except Exception:
                    output_tokens = 0
                try:
                    if usage.get("prompt_tokens") is not None:
                        input_tokens = int(usage["prompt_tokens"])
                except Exception:
                    input_tokens = None
                choices = body.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                # Non-stream: client TTFT is not observable → leave None
                # (never invent e2e as TTFT).
        else:
            async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
                http_status = resp.status_code
                if resp.status_code != 200:
                    body = await resp.aread()
                    t_end = time.time()
                    error = f"HTTP {resp.status_code}: {body[:200]!r}"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        data = _parse_sse_chunk(chunk)
                        if not data:
                            # Fallback heuristic for non-JSON SSE test doubles
                            if '"content":"' in chunk and '""' not in chunk:
                                if not first_token_seen:
                                    t_first = time.time()
                                    first_token_seen = True
                                output_tokens += 1
                            continue
                        usage = data.get("usage") or {}
                        if usage:
                            try:
                                if usage.get("completion_tokens") is not None:
                                    output_tokens = int(usage["completion_tokens"])
                            except Exception:
                                pass
                            try:
                                if usage.get("prompt_tokens") is not None:
                                    input_tokens = int(usage["prompt_tokens"])
                            except Exception:
                                pass
                        content = _content_from_chunk(data)
                        if content:
                            if not first_token_seen:
                                t_first = time.time()
                                first_token_seen = True
                            # Rough count when usage absent
                            if not usage:
                                output_tokens += 1
                        fr = _finish_reason_from_chunk(data)
                        if fr:
                            finish_reason = fr
                    t_end = time.time()

    except httpx.TimeoutException as exc:
        t_end = time.time()
        timed_out = True
        error = str(exc)[:200]
    except asyncio.CancelledError:
        t_end = time.time()
        cancelled = True
        error = "cancelled"
    except Exception as exc:
        t_end = time.time()
        error = str(exc)[:200]

    if t_end is None:
        t_end = time.time()

    ttft_ms = (t_first - t_start) * 1000.0 if t_first is not None else None
    e2e_ms = (t_end - t_start) * 1000.0

    outcome, term = classify_http_outcome(
        status_code=http_status,
        timed_out=timed_out,
        cancelled=cancelled,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
        first_token_seen=first_token_seen or (not stream_response and output_tokens > 0),
        error=error,
    )
    # Non-stream with tokens: first_token_seen forced true for classify only;
    # TTFT remains None because it was not client-measured.
    if not stream_response and outcome == RequestOutcome.SUCCESS and output_tokens > 0:
        # Still success/truncate, but TTFT missing.
        pass

    tpot_ms = compute_tpot_ms(
        e2e_ms=e2e_ms,
        ttft_ms=ttft_ms,
        output_tokens=output_tokens,
    )

    return RequestRecord(
        run_id=run_id,
        request_id=request_id,
        is_warmup=is_warmup,
        t_start_s=t_start,
        t_first_token_s=t_first,
        t_end_s=t_end,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        outcome=outcome,
        termination_reason=term,
        http_status=http_status,
        finish_reason=finish_reason,
        error=error,
    )


async def run_load(
    base_url: str,
    workload: WorkloadSpec,
    prompts: list[str],
    warmup_requests: int = 5,
    close_client: bool = True,
    stream_response: bool = True,
    run_id: str | None = None,
    timeout_s: float = 120.0,
    cache_enabled: bool | None = None,
) -> LoadResult:
    """Run warmup + measured load; return ledger-backed LoadResult."""
    rid = run_id or uuid.uuid4().hex
    sem = asyncio.Semaphore(workload.concurrency)
    seq = 0
    seq_lock = asyncio.Lock()

    async def next_id(prefix: str) -> str:
        nonlocal seq
        async with seq_lock:
            seq += 1
            return f"{prefix}-{seq:04d}"

    async def bounded(prompt: str, *, is_warmup: bool) -> RequestRecord:
        async with sem:
            req_id = await next_id("warmup" if is_warmup else "req")
            return await _send_one(
                client,
                base_url,
                prompt,
                workload.output_len,
                run_id=rid,
                request_id=req_id,
                is_warmup=is_warmup,
                stream_response=stream_response,
                timeout_s=timeout_s,
            )

    limits = httpx.Limits(
        max_connections=workload.concurrency + 10,
        max_keepalive_connections=workload.concurrency,
    )
    client = httpx.AsyncClient(limits=limits)
    conditions = RunConditions(
        workload_name=workload.name,
        num_requests=workload.num_requests,
        concurrency=workload.concurrency,
        input_len_target=workload.input_len,
        output_len_target=workload.output_len,
        distribution=workload.distribution,
        arrival_rps=workload.rps,
        warmup_requests=warmup_requests,
        stream_response=stream_response,
        sampling_temperature=0.0,
        cache_enabled=cache_enabled,
    )
    ledger = RequestLedger(run_id=rid, conditions=conditions)

    try:
        warmup = prompts[:warmup_requests]
        if warmup:
            warm_records = await asyncio.gather(
                *[bounded(p, is_warmup=True) for p in warmup]
            )
            ledger.extend(warm_records)

        measure = prompts[warmup_requests : warmup_requests + workload.num_requests]
        t0 = time.time()
        results = await asyncio.gather(*[bounded(p, is_warmup=False) for p in measure])
        t1 = time.time()
        ledger.extend(results)
        ledger.window_start_s = t0
        ledger.window_end_s = t1
    finally:
        if close_client:
            await client.aclose()

    agg = recalculate_from_ledger(ledger)
    ttft_ms = [
        r.ttft_ms for r in ledger.measured() if r.ttft_ms is not None and r.outcome
        in (RequestOutcome.SUCCESS, RequestOutcome.TRUNCATE)
    ]
    e2e_ms = [
        r.e2e_ms for r in ledger.measured() if r.e2e_ms is not None and r.outcome
        in (RequestOutcome.SUCCESS, RequestOutcome.TRUNCATE)
    ]

    return LoadResult(
        total_requests=agg.total_requests,
        successful=agg.successful_requests,
        total_time_s=agg.total_time_s if agg.total_time_s is not None else 0.0,
        throughput_rps=agg.throughput_rps,
        tokens_per_second=agg.tokens_per_second,
        ttft_ms=sorted(ttft_ms),
        e2e_ms=sorted(e2e_ms),
        error_rate=agg.error_rate,
        ledger=ledger,
        aggregate=agg,
        records=list(ledger.records),
    )


def _percentile(data: list[float], p: float) -> float | None:
    """Compat wrapper — empty → None (P0-④). Prefer metrics.definitions.percentile."""
    from inferops.metrics.definitions import percentile

    return percentile(data, p)


def extract_percentiles(ms_list: list[float]) -> dict[str, float | None]:
    """Legacy helper. Empty list → all None (never fake 0.0)."""
    from inferops.metrics.definitions import percentiles

    return percentiles(ms_list)


# Re-export for callers that build synthetic RequestMetrics in tests
__all__ = [
    "LoadResult",
    "RequestMetrics",
    "RequestRecord",
    "TerminationReason",
    "extract_percentiles",
    "run_load",
]
