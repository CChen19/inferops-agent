"""CPU tests for hardware fingerprint helpers. No nvidia-smi required."""

from __future__ import annotations

from inferops.memory.hardware import (
    collect_hardware_info,
    fingerprint_from_hardware,
    fingerprints_compatible,
)
from inferops.schemas import HardwareInfo


def test_fingerprint_requires_complete_fields():
    incomplete = HardwareInfo(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        engine="vllm",
        gpu_name="RTX 3060",
        # missing vllm_version and gpu_memory_total_gb
    )
    assert fingerprint_from_hardware(incomplete) is None

    complete = HardwareInfo(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        engine="vllm",
        vllm_version="0.6.0",
        gpu_name="RTX 3060",
        gpu_memory_total_gb=6.0,
    )
    fp = fingerprint_from_hardware(complete)
    assert fp is not None
    assert fp["gpu_memory_total_gb"] == 6.0


def test_fingerprints_compatible_match_and_mismatch():
    a = fingerprint_from_hardware(
        HardwareInfo(
            model_name="m",
            engine="vllm",
            vllm_version="1",
            gpu_name="RTX 3060",
            gpu_memory_total_gb=6.0,
        )
    )
    b = fingerprint_from_hardware(
        HardwareInfo(
            model_name="m",
            engine="vllm",
            vllm_version="1",
            gpu_name="RTX 3060",
            gpu_memory_total_gb=6.0,
        )
    )
    c = fingerprint_from_hardware(
        HardwareInfo(
            model_name="m",
            engine="vllm",
            vllm_version="1",
            gpu_name="A100",
            gpu_memory_total_gb=40.0,
        )
    )
    assert fingerprints_compatible(a, b) is True
    assert fingerprints_compatible(a, c) is False
    assert fingerprints_compatible(a, None) is False
    assert fingerprints_compatible(None, b) is False


def test_collect_hardware_info_uses_env_without_nvidia(monkeypatch):
    monkeypatch.setenv("INFEROPS_GPU_NAME", "RTX 3060 Laptop")
    monkeypatch.setenv("INFEROPS_GPU_MEM_GB", "6")
    monkeypatch.setenv("VLLM_VERSION", "0.6.1")
    hw = collect_hardware_info(model_name="m", probe_nvidia=False)
    assert hw.gpu_name == "RTX 3060 Laptop"
    assert hw.gpu_memory_total_gb == 6.0
    assert hw.vllm_version == "0.6.1"
    fp = fingerprint_from_hardware(hw)
    assert fp is not None
