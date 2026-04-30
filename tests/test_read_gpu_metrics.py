"""Unit tests for read_gpu_metrics tool — mocks pynvml entirely."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferops.tools.read_gpu_metrics import ReadGpuMetricsInput, read_gpu_metrics


def _make_nvml_mocks(util_pct=75, mem_used_mb=3000, mem_total_mb=6144):
    util = MagicMock()
    util.gpu = util_pct
    mem = MagicMock()
    mem.used = mem_used_mb * 1024 * 1024
    mem.total = mem_total_mb * 1024 * 1024
    return util, mem


def test_read_gpu_metrics_basic():
    util, mem = _make_nvml_mocks(util_pct=80, mem_used_mb=3500)

    with patch("inferops.tools.read_gpu_metrics.pynvml") as mock_nvml, \
         patch("inferops.tools.read_gpu_metrics.time.sleep"):
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3060"
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util

        out = read_gpu_metrics(ReadGpuMetricsInput(duration_s=1.0))

    assert out.device_name == "NVIDIA GeForce RTX 3060"
    assert out.avg_util_pct == 80.0
    assert out.max_util_pct == 80.0
    assert abs(out.avg_mem_used_gb - 3500 / 1024) < 0.01
    assert out.samples >= 1


def test_read_gpu_metrics_multiple_samples():
    util, mem = _make_nvml_mocks(util_pct=60)

    with patch("inferops.tools.read_gpu_metrics.pynvml") as mock_nvml, \
         patch("inferops.tools.read_gpu_metrics.time.sleep"):
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_nvml.nvmlDeviceGetName.return_value = "RTX 3060"
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util

        out = read_gpu_metrics(ReadGpuMetricsInput(duration_s=2.0))

    # 2.0s / 0.5s interval = 4 samples
    assert out.samples == 4
