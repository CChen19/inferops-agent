"""Tool: read_gpu_metrics — sample GPU utilization and memory with pynvml."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

import pynvml

from inferops.observability import span


class ReadGpuMetricsInput(BaseModel):
    duration_s: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="How many seconds to sample the GPU. Samples taken every 0.5s.",
    )
    device_index: int = Field(default=0, description="GPU device index (0 for single-GPU systems).")


class GpuMetricsOutput(BaseModel):
    device_name: str
    avg_util_pct: float
    max_util_pct: float
    avg_mem_used_gb: float
    max_mem_used_gb: float
    mem_total_gb: float
    samples: int
    duration_s: float


def read_gpu_metrics(inp: ReadGpuMetricsInput) -> GpuMetricsOutput:
    """
    Sample GPU utilization and memory usage over a short window using pynvml.

    Returns average and peak values across the sampling period. Use this before
    or during a benchmark to understand baseline GPU state, or after to check
    whether the GPU was saturated.
    """
    with span("tool.read_gpu_metrics", {"duration_s": str(inp.duration_s)}):
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(inp.device_index)
            device_name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total_gb = mem_info.total / 1024 ** 3

            utils: list[float] = []
            mems: list[float] = []
            interval = 0.5
            steps = max(1, int(inp.duration_s / interval))

            for _ in range(steps):
                u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utils.append(float(u.gpu))
                mems.append(m.used / 1024 ** 3)
                time.sleep(interval)
        finally:
            pynvml.nvmlShutdown()

    return GpuMetricsOutput(
        device_name=device_name,
        avg_util_pct=sum(utils) / len(utils),
        max_util_pct=max(utils),
        avg_mem_used_gb=sum(mems) / len(mems),
        max_mem_used_gb=max(mems),
        mem_total_gb=mem_total_gb,
        samples=len(utils),
        duration_s=inp.duration_s,
    )
