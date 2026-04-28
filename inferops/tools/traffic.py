"""Async load generator: sends concurrent streaming requests, measures TTFT / E2E / throughput."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from inferops.schemas import WorkloadSpec


@dataclass
class RequestMetrics:
    success: bool
    ttft_ms: float        # time to first token
    e2e_ms: float         # total end-to-end
    output_tokens: int
    error: str = ""


@dataclass
class LoadResult:
    total_requests: int
    successful: int
    total_time_s: float
    throughput_rps: float
    tokens_per_second: float
    ttft_ms: list[float]
    e2e_ms: list[float]


async def _send_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> RequestMetrics:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t_start = time.perf_counter()
    t_first: float | None = None
    output_tokens = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=120) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RequestMetrics(False, 0, 0, 0, error=f"HTTP {resp.status_code}: {body[:200]}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                # First non-empty content token → TTFT
                if t_first is None and '"content":"' in chunk and '""' not in chunk:
                    t_first = time.perf_counter()
                # Count output tokens roughly by counting chunks with content
                if '"content"' in chunk:
                    output_tokens += 1

        t_end = time.perf_counter()
        ttft = (t_first - t_start) * 1000 if t_first else (t_end - t_start) * 1000
        e2e = (t_end - t_start) * 1000
        return RequestMetrics(True, ttft, e2e, max(output_tokens, 1))

    except Exception as exc:
        t_end = time.perf_counter()
        return RequestMetrics(False, 0, (t_end - t_start) * 1000, 0, error=str(exc)[:200])


async def run_load(
    base_url: str,
    workload: WorkloadSpec,
    prompts: list[str],
    warmup_requests: int = 5,
) -> LoadResult:
    sem = asyncio.Semaphore(workload.concurrency)

    async def bounded(prompt: str) -> RequestMetrics:
        async with sem:
            return await _send_one(client, base_url, prompt, workload.output_len)

    limits = httpx.Limits(max_connections=workload.concurrency + 10, max_keepalive_connections=workload.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warmup — don't measure these
        warmup = prompts[:warmup_requests]
        await asyncio.gather(*[bounded(p) for p in warmup])

        # Actual benchmark
        measure = prompts[warmup_requests : warmup_requests + workload.num_requests]
        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(p) for p in measure])
        total_time = time.perf_counter() - t0

    successes = [r for r in results if r.success]
    ttft_ms = sorted(r.ttft_ms for r in successes)
    e2e_ms = sorted(r.e2e_ms for r in successes)
    total_tokens = sum(r.output_tokens for r in successes)

    return LoadResult(
        total_requests=len(measure),
        successful=len(successes),
        total_time_s=total_time,
        throughput_rps=len(successes) / total_time if total_time > 0 else 0,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
    )


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def extract_percentiles(ms_list: list[float]) -> dict[str, float]:
    s = sorted(ms_list)
    return {
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
    }
