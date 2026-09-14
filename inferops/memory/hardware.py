"""Hardware fingerprint for compatible-history matching.

Capacity (total VRAM), not utilization. Env overrides:
``INFEROPS_GPU_NAME``, ``INFEROPS_GPU_MEM_GB``, ``VLLM_VERSION``.
``nvidia-smi`` is optional and never required in tests.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, TypedDict

from inferops.schemas import HardwareInfo


class HardwareFingerprint(TypedDict):
    gpu_name: str
    gpu_memory_total_gb: float
    model_name: str
    engine: str
    vllm_version: str


def _parse_mem_gb(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _nvidia_smi_probe() -> tuple[str | None, float | None]:
    """Best-effort GPU name + total VRAM. Returns (None, None) if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None, None
    line = (out or "").strip().splitlines()
    if not line:
        return None, None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 2:
        return (parts[0] or None) if parts else None, None
    name = parts[0] or None
    try:
        # nvidia-smi memory.total is MiB
        mem_gb = round(float(parts[1]) / 1024.0, 2)
    except (TypeError, ValueError):
        mem_gb = None
    return name, mem_gb


def resolve_vllm_version() -> str | None:
    """Env ``VLLM_VERSION`` first; else installed package version; else None.

    Never invent a version — incomplete fingerprints stay excluded from ranking.
    """
    env = (os.getenv("VLLM_VERSION") or "").strip()
    if env:
        return env
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("vllm")
    except Exception:
        return None


def collect_hardware_info(
    *,
    model_name: str = "",
    engine: str = "vllm",
    probe_nvidia: bool = True,
) -> HardwareInfo:
    """Fill HardwareInfo from env (and optional nvidia-smi). CPU-safe when probe is off."""
    gpu_name = os.getenv("INFEROPS_GPU_NAME") or None
    mem = _parse_mem_gb(os.getenv("INFEROPS_GPU_MEM_GB"))
    if probe_nvidia and (not gpu_name or mem is None):
        smi_name, smi_mem = _nvidia_smi_probe()
        gpu_name = gpu_name or smi_name
        mem = mem if mem is not None else smi_mem
    return HardwareInfo(
        model_name=model_name,
        engine=engine,
        vllm_version=resolve_vllm_version(),
        gpu_name=gpu_name,
        gpu_memory_total_gb=mem,
        cuda_version=os.getenv("CUDA_VERSION") or None,
    )


def fingerprint_from_hardware(hw: HardwareInfo | dict[str, Any] | None) -> HardwareFingerprint | None:
    """Build a complete fingerprint, or None if any required field is missing."""
    if hw is None:
        return None
    if isinstance(hw, HardwareInfo):
        data = hw.model_dump()
    elif isinstance(hw, dict):
        data = hw
    else:
        return None
    gpu_name = str(data.get("gpu_name") or "").strip()
    model_name = str(data.get("model_name") or "").strip()
    engine = str(data.get("engine") or "").strip() or "vllm"
    vllm_version = str(data.get("vllm_version") or "").strip()
    mem = data.get("gpu_memory_total_gb")
    try:
        mem_f = float(mem) if mem is not None else None
    except (TypeError, ValueError):
        mem_f = None
    if not gpu_name or not model_name or not vllm_version or mem_f is None:
        return None
    return HardwareFingerprint(
        gpu_name=gpu_name,
        gpu_memory_total_gb=mem_f,
        model_name=model_name,
        engine=engine,
        vllm_version=vllm_version,
    )


def fingerprints_compatible(
    current: HardwareFingerprint | None,
    other: HardwareFingerprint | None,
) -> bool:
    """True only when both fingerprints are complete and equal on all fields.

    Unknown (None / incomplete) never matches — do not pretend SKU matching
    on empty fields.
    """
    if current is None or other is None:
        return False
    if current["gpu_name"] != other["gpu_name"]:
        return False
    if current["model_name"] != other["model_name"]:
        return False
    if current["engine"] != other["engine"]:
        return False
    if current["vllm_version"] != other["vllm_version"]:
        return False
    # Capacity equality with small float tolerance
    if abs(float(current["gpu_memory_total_gb"]) - float(other["gpu_memory_total_gb"])) > 1e-6:
        return False
    return True


def normalize_gpu_name(name: str) -> str:
    """Light normalize for display / tests; matching uses exact stored strings."""
    return re.sub(r"\s+", " ", (name or "").strip())
