"""Unit tests for benchmark orchestration helpers."""

from __future__ import annotations

import time

import pytest

from inferops import bench_runner


def test_run_load_workaround_returns_before_asyncio_cleanup(monkeypatch, config):
    sentinel = object()
    captured = {}

    async def fake_run_load(**kwargs):
        captured.update(kwargs)
        return sentinel

    original_run = bench_runner.asyncio.run

    def fake_asyncio_run(coro):
        original_run(coro)
        time.sleep(0.5)

    monkeypatch.setattr(bench_runner, "run_load", fake_run_load)
    monkeypatch.setattr(bench_runner.asyncio, "run", fake_asyncio_run)

    started = time.time()
    out = bench_runner._run_load_with_cleanup_workaround(
        config,
        ["prompt"],
        timeout_s=1,
        poll_interval_s=0.01,
    )

    assert out is sentinel
    assert time.time() - started < 0.25
    assert captured["close_client"] is False
    assert captured["stream_response"] is False


def test_run_load_workaround_propagates_load_errors(monkeypatch, config):
    async def fake_run_load(**kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(bench_runner, "run_load", fake_run_load)

    with pytest.raises(RuntimeError, match="load failed"):
        bench_runner._run_load_with_cleanup_workaround(
            config,
            ["prompt"],
            timeout_s=1,
            poll_interval_s=0.01,
        )
