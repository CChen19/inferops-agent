"""CPU tests for hardware fingerprint helpers. No nvidia-smi required."""

from __future__ import annotations

from unittest.mock import patch

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


def test_resolve_vllm_version_falls_back_to_package_metadata(monkeypatch):
    from inferops.memory.hardware import resolve_vllm_version

    monkeypatch.delenv("VLLM_VERSION", raising=False)
    with patch("importlib.metadata.version", return_value="0.9.9"):
        assert resolve_vllm_version() == "0.9.9"


def test_resolve_vllm_version_stays_none_when_unavailable(monkeypatch):
    from inferops.memory.hardware import resolve_vllm_version

    monkeypatch.delenv("VLLM_VERSION", raising=False)

    def _missing(name):
        raise Exception("not installed")

    with patch("importlib.metadata.version", side_effect=_missing):
        assert resolve_vllm_version() is None


def test_collect_via_mocked_nvidia_smi_without_env(monkeypatch):
    """Production-like collect: no env; nvidia-smi mocked; package version mocked."""
    for key in ("INFEROPS_GPU_NAME", "INFEROPS_GPU_MEM_GB", "VLLM_VERSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "inferops.memory.hardware._nvidia_smi_probe",
        lambda: ("NVIDIA GeForce RTX 3060 Laptop GPU", 6.0),
    )
    with patch("importlib.metadata.version", return_value="0.6.0"):
        hw = collect_hardware_info(model_name="Qwen/Qwen2.5-0.5B-Instruct", probe_nvidia=True)
    fp = fingerprint_from_hardware(hw)
    assert fp is not None
    assert fp["gpu_name"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert fp["gpu_memory_total_gb"] == 6.0
    assert fp["vllm_version"] == "0.6.0"
