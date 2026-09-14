#!/usr/bin/env python
"""Run offline hidden-result fair comparison and write JSON + Markdown reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from inferops.eval.fair_compare import run_offline_fair_compare, write_fair_compare_outputs

console = Console()


def _current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline fair compare over hidden GT fixtures (no live LLM/GPU)"
    )
    parser.add_argument("--ground-truth", default="tests/fixtures/ground_truth")
    parser.add_argument("--output-dir", default="eval_reports/fair_compare")
    parser.add_argument("--workloads", nargs="+", default=["chat_short", "long_generation"])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--commit-sha", default=None)
    args = parser.parse_args()

    report = run_offline_fair_compare(
        commit_sha=args.commit_sha or _current_sha(),
        ground_truth_dir=args.ground_truth,
        workloads=args.workloads,
        budget=args.budget,
        seed=args.seed,
    )
    md_path, json_path = write_fair_compare_outputs(report, args.output_dir)
    console.print(f"[green]Fair compare report written:[/] {md_path}")
    console.print(f"[green]Fair compare JSON written:[/] {json_path}")


if __name__ == "__main__":
    main()
