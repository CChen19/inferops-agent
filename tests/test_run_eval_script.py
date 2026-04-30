"""Unit tests for the Phase 5 run_eval script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_eval_mock_writes_report(tmp_path):
    cmd = [
        ".venv/bin/python",
        "scripts/run_eval.py",
        "--mock",
        "--commit-sha",
        "unitsha",
        "--ground-truth",
        "tests/fixtures/ground_truth",
        "--output-dir",
        str(tmp_path),
        "--workloads",
        "chat_short",
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "unitsha.md").exists()
    assert (tmp_path / "unitsha.json").exists()


def test_run_eval_without_mock_exits_with_error(tmp_path):
    cmd = [
        ".venv/bin/python",
        "scripts/run_eval.py",
        "--commit-sha",
        "unitsha",
        "--output-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)

    assert proc.returncode == 2
    assert "Only --mock mode" in proc.stdout
