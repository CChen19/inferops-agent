#!/usr/bin/env python
"""Run one live online-local-search arm and compare with an ingested planner report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from inferops.eval.live_fair_compare import (
    LIVE_LLM_BOUNDARY,
    TOOL_BOUNDARY,
    LiveCompareBlocked,
    build_live_compare_report,
    current_sha,
    ingest_planner_summary,
    require_managed_live_conditions,
    run_live_search,
    write_live_compare_outputs,
)
from inferops.tools.vllm_process import get_vllm_python

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live fair compare: ingested production planner vs online local search"
    )
    parser.add_argument("--planner-summary", help="Planner meta.json or its report directory")
    parser.add_argument("--run-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="reports/live_fair_compare")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--workload", default="chat_short", choices=["chat_short"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-ttft-ms", type=float, default=250.0)
    parser.add_argument("--prefix", default="livefair_search_")
    parser.add_argument(
        "--vllm-python",
        default=None,
        help="vLLM interpreter path (default: INFEROPS_VLLM_PYTHON or VLLM_PYTHON)",
    )
    parser.add_argument("--commit-sha")
    args = parser.parse_args(argv)

    if args.budget < 1:
        parser.error("--budget must be at least 1")

    planner = {
        "strategy": "planner",
        "source": "missing",
        "llm_boundary": LIVE_LLM_BOUNDARY,
        "tool_boundary": TOOL_BOUNDARY,
        "budget": args.budget,
        "budget_used": 0,
        "experiment_ids": [],
        "observations": [],
        "best": None,
        "decision_kind": None,
        "claim_level": "unavailable",
    }
    search_run = None
    fixture = None
    blocked_reason: str | None = None

    try:
        if not args.planner_summary:
            raise LiveCompareBlocked(
                "--planner-summary is required; this runner ingests the already-executed "
                "production planner arm and never needs OPENROUTER_API_KEY"
            )
        planner = ingest_planner_summary(args.planner_summary)
        if planner.get("budget") != args.budget:
            raise LiveCompareBlocked(
                f"planner budget {planner.get('budget')} does not match search budget {args.budget}"
            )
        for field, expected in (
            ("model_name", args.model),
            ("workload", args.workload),
            ("max_ttft_ms", args.max_ttft_ms),
        ):
            if planner.get(field) != expected:
                raise LiveCompareBlocked(
                    f"planner {field} {planner.get(field)!r} does not match search {expected!r}"
                )

        if args.run_search:
            try:
                vllm_python = args.vllm_python if args.vllm_python is not None else get_vllm_python()
            except RuntimeError as exc:
                raise LiveCompareBlocked(str(exc)) from exc
            require_managed_live_conditions(vllm_python)
            os.environ["INFEROPS_VLLM_PYTHON"] = vllm_python
            search_run, fixture = run_live_search(
                budget=args.budget,
                workload_name=args.workload,
                model_name=args.model,
                session_prefix=args.prefix,
                max_ttft_ms=args.max_ttft_ms,
            )
    except LiveCompareBlocked as exc:
        blocked_reason = str(exc)

    report = build_live_compare_report(
        commit_sha=args.commit_sha or current_sha(),
        planner=planner,
        search_run=search_run,
        fixture=fixture,
        budget=args.budget,
        workload_name=args.workload,
        model_name=args.model,
        max_ttft_ms=args.max_ttft_ms,
        blocked_reason=blocked_reason,
    )
    report["conditions"]["vllm_python"] = args.vllm_python
    md_path, json_path = write_live_compare_outputs(report, args.output_dir)
    console.print(f"Markdown: {md_path}")
    console.print(f"JSON: {json_path}")
    if blocked_reason:
        console.print(f"[red]BLOCKED:[/] {blocked_reason}")
        return 2
    console.print("[green]Live fair comparison complete.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
