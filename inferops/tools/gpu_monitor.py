"""Background GPU sampler using pynvml. Runs in a daemon thread during benchmarks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pynvml


@dataclass
class GPUSample:
    timestamp: float
    util_pct: float       # GPU compute utilization %
    mem_used_mb: float    # VRAM used in MiB
    mem_total_mb: float


@dataclass
class GPUSummary:
    avg_util_pct: float
    max_util_pct: float
    avg_mem_used_gb: float
    max_mem_used_gb: float
    samples: int


class GPUMonitor:
    """Samples GPU util + memory in a background thread at `interval_s` Hz."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUSummary:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        pynvml.nvmlShutdown()
        return self._summarize()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(GPUSample(
                    timestamp=time.time(),
                    util_pct=float(util.gpu),
                    mem_used_mb=mem.used / 1024 / 1024,
                    mem_total_mb=mem.total / 1024 / 1024,
                ))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _summarize(self) -> GPUSummary:
        if not self._samples:
            return GPUSummary(0, 0, 0, 0, 0)
        utils = [s.util_pct for s in self._samples]
        mems = [s.mem_used_mb / 1024 for s in self._samples]
        return GPUSummary(
            avg_util_pct=sum(utils) / len(utils),
            max_util_pct=max(utils),
            avg_mem_used_gb=sum(mems) / len(mems),
            max_mem_used_gb=max(mems),
            samples=len(self._samples),
        )

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
