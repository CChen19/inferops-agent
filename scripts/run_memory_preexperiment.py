#!/usr/bin/env python
"""Run offline memory pre-experiment (CPU A/B/C) and write JSON + Markdown."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from inferops.eval.memory_preexperiment import (
    run_memory_preexperiment,
    write_memory_preexperiment_outputs,
)

console = Console()


def _current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline memory pre-experiment: groups A/B/C over hidden GT fixtures "
            "(scripted proposals; no live LLM/GPU)"
        )
    )
    parser.add_argument("--ground-truth", default="tests/fixtures/ground_truth")
    parser.add_argument("--output-dir", default="eval_reports/memory_preexperiment")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--commit-sha", default=None)
    args = parser.parse_args()

    report = run_memory_preexperiment(
        commit_sha=args.commit_sha or _current_sha(),
        ground_truth_dir=args.ground_truth,
        budget_slots=args.budget,
    )
    md_path, json_path = write_memory_preexperiment_outputs(report, args.output_dir)
    console.print(f"[green]Memory pre-experiment report written:[/] {md_path}")
    console.print(f"[green]Memory pre-experiment JSON written:[/] {json_path}")
    console.print(f"[dim]{report['disclaimer']}[/]")


if __name__ == "__main__":
    main()
