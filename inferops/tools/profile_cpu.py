"""Tool: profile_with_pyspy — short CPU profile of a running process via py-spy."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, Field

from inferops.observability import span


class ProfileCpuInput(BaseModel):
    pid: int = Field(description="PID of the process to profile (e.g., the vLLM server).")
    duration_s: int = Field(default=5, ge=1, le=60, description="Profiling duration in seconds.")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of top CPU hotspots to return.")


class CpuHotspot(BaseModel):
    rank: int
    own_pct: float
    total_pct: float
    function: str
    location: str


class ProfileCpuOutput(BaseModel):
    pid: int
    duration_s: int
    hotspots: list[CpuHotspot]
    raw_output: str
    error: str = ""


def profile_with_pyspy(inp: ProfileCpuInput) -> ProfileCpuOutput:
    """
    Run a short CPU profile on a live process using py-spy.

    Returns the top-N CPU hotspots as structured data. Useful for diagnosing
    whether a bottleneck is in Python scheduling code, tokenisation, CUDA kernel
    dispatch, or something else. The target process must be accessible to the
    current user (or run with sudo).

    Note: py-spy must be installed (`pip install py-spy`) and the target PID
    must be a running Python process.
    """
    pyspy_bin = shutil.which("py-spy")
    if pyspy_bin is None:
        return ProfileCpuOutput(
            pid=inp.pid,
            duration_s=inp.duration_s,
            hotspots=[],
            raw_output="",
            error="py-spy not found on PATH. Install with: pip install py-spy",
        )

    cmd = [pyspy_bin, "top", "--pid", str(inp.pid), "--duration", str(inp.duration_s), "--noninteractive"]

    with span("tool.profile_with_pyspy", {"pid": str(inp.pid), "duration_s": str(inp.duration_s)}):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=inp.duration_s + 10)
            raw = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return ProfileCpuOutput(pid=inp.pid, duration_s=inp.duration_s, hotspots=[], raw_output="", error="py-spy timed out")
        except Exception as exc:
            return ProfileCpuOutput(pid=inp.pid, duration_s=inp.duration_s, hotspots=[], raw_output="", error=str(exc))

    hotspots = _parse_pyspy_top(raw, inp.top_n)
    return ProfileCpuOutput(
        pid=inp.pid,
        duration_s=inp.duration_s,
        hotspots=hotspots,
        raw_output=raw[:4000],  # cap size for LLM context
    )


def _parse_pyspy_top(output: str, top_n: int) -> list[CpuHotspot]:
    """Parse py-spy top --noninteractive text output into structured hotspots."""
    hotspots = []
    for i, line in enumerate(output.splitlines()):
        line = line.strip()
        # py-spy top lines look like: "  3.00%   3.00% function_name (file.py:123)"
        if not line or line.startswith("Collecting") or line.startswith("%") or "%" not in line:
            continue
        try:
            parts = line.split()
            own = float(parts[0].rstrip("%"))
            total = float(parts[1].rstrip("%"))
            rest = " ".join(parts[2:])
            if "(" in rest:
                func, loc = rest.rsplit("(", 1)
                loc = loc.rstrip(")")
            else:
                func, loc = rest, ""
            hotspots.append(CpuHotspot(rank=len(hotspots) + 1, own_pct=own, total_pct=total, function=func.strip(), location=loc.strip()))
            if len(hotspots) >= top_n:
                break
        except (ValueError, IndexError):
            continue
    return hotspots
