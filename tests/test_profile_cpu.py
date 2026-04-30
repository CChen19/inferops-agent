"""Unit tests for profile_with_pyspy tool — mocks subprocess.run."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inferops.tools.profile_cpu import ProfileCpuInput, _parse_pyspy_top, profile_with_pyspy

_SAMPLE_PYSPY_OUTPUT = """\
Collecting for 5 seconds
  5.00%   5.00% _run_step (/opt/vllm/engine.py:312)
  3.00%   8.00% forward (/opt/torch/nn/modules/module.py:1501)
  2.00%   2.00% schedule (/opt/vllm/scheduler.py:88)
  1.50%   1.50% tokenize (/opt/vllm/tokenizer.py:45)
"""


def test_parse_pyspy_top():
    hotspots = _parse_pyspy_top(_SAMPLE_PYSPY_OUTPUT, top_n=10)
    assert len(hotspots) == 4
    assert hotspots[0].function == "_run_step"
    assert hotspots[0].own_pct == 5.0
    assert "engine.py" in hotspots[0].location


def test_profile_success():
    mock_proc = MagicMock()
    mock_proc.stdout = _SAMPLE_PYSPY_OUTPUT
    mock_proc.stderr = ""

    with patch("inferops.tools.profile_cpu.shutil.which", return_value="/usr/bin/py-spy"), \
         patch("inferops.tools.profile_cpu.subprocess.run", return_value=mock_proc):
        out = profile_with_pyspy(ProfileCpuInput(pid=12345, duration_s=5, top_n=3))

    assert out.error == ""
    assert len(out.hotspots) == 3
    assert out.hotspots[0].function == "_run_step"


def test_profile_pyspy_not_found():
    with patch("inferops.tools.profile_cpu.shutil.which", return_value=None):
        out = profile_with_pyspy(ProfileCpuInput(pid=12345))

    assert "not found" in out.error
    assert out.hotspots == []
