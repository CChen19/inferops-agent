"""Unit tests for GPU monitor summary logic without touching NVML."""

from __future__ import annotations

from inferops.tools.gpu_monitor import GPUMonitor, GPUSample


def test_gpu_monitor_summarize_empty_samples():
    monitor = GPUMonitor()

    summary = monitor._summarize()

    assert summary.samples == 0
    assert summary.avg_util_pct == 0
    assert summary.max_mem_used_gb == 0


def test_gpu_monitor_summarize_samples():
    monitor = GPUMonitor()
    monitor._samples = [
        GPUSample(timestamp=1.0, util_pct=50.0, mem_used_mb=2048.0, mem_total_mb=6144.0),
        GPUSample(timestamp=2.0, util_pct=90.0, mem_used_mb=4096.0, mem_total_mb=6144.0),
    ]

    summary = monitor._summarize()

    assert summary.samples == 2
    assert summary.avg_util_pct == 70.0
    assert summary.max_util_pct == 90.0
    assert summary.avg_mem_used_gb == 3.0
    assert summary.max_mem_used_gb == 4.0
