#!/usr/bin/env python
"""
Phase 1 baseline sweep: default + 3 variants × 2 workloads = 8 runs.

Usage (from project root, .venv activated):
  python scripts/run_baseline.py [--workload chat_short|long_context_qa|both]
                                  [--variants default,chunked,prefix_cache,big_batch]
                                  [--report-dir reports/]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.rule import Rule

from configs.search_space import make_configs
from inferops.bench_runner import BenchmarkError, OOMError, StartupTimeoutError, print_results_table, run_experiment
from inferops.schemas import ExperimentResult
from workloads.definitions import CHAT_SHORT, LONG_CONTEXT_QA, get_prompts

console = Console()


def generate_report(results: list[ExperimentResult], report_dir: Path) -> Path:
    """Write a Markdown + JSON baseline report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    # JSON dump for programmatic use
    json_path = report_dir / f"baseline_{ts}.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "experiment_id": r.experiment_id,
                    "workload": r.config.workload.name,
                    "variant": r.config.tags.get("variant", ""),
                    "throughput_rps": round(r.throughput_rps, 3),
                    "tokens_per_second": round(r.tokens_per_second, 1),
                    "ttft_p50_ms": round(r.ttft.p50, 1),
                    "ttft_p99_ms": round(r.ttft.p99, 1),
                    "e2e_p50_ms": round(r.e2e_latency.p50, 1),
                    "e2e_p99_ms": round(r.e2e_latency.p99, 1),
                    "gpu_util_pct": round(r.gpu_utilization_pct or 0, 1),
                    "gpu_mem_gb": round(r.gpu_memory_used_gb or 0, 2),
                    "success_rate": f"{r.successful_requests}/{r.total_requests}",
                    "mlflow_run_id": r.mlflow_run_id,
                }
                for r in results
            ],
            indent=2,
        )
    )

    # Markdown report
    md_path = report_dir / f"baseline_{ts}.md"
    lines = [
        "# InferOps Baseline Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Model: Qwen2.5-0.5B-Instruct | Hardware: RTX 3060 Laptop (6 GB)",
        "",
        "## Results",
        "",
    ]

    # Group by workload
    for wl_name in ["chat_short", "long_context_qa"]:
        wl_results = [r for r in results if r.config.workload.name == wl_name]
        if not wl_results:
            continue

        lines += [f"### {wl_name}", ""]
        lines += [
            "| Variant | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU util | GPU mem |",
            "|---------|-----|-------|----------|----------|---------|---------|----------|---------|",
        ]
        for r in wl_results:
            v = r.config.tags.get("variant", r.experiment_id)
            lines.append(
                f"| {v} | {r.throughput_rps:.2f} | {r.tokens_per_second:.0f} "
                f"| {r.ttft.p50:.0f}ms | {r.ttft.p99:.0f}ms "
                f"| {r.e2e_latency.p50:.0f}ms | {r.e2e_latency.p99:.0f}ms "
                f"| {r.gpu_utilization_pct:.0f}% | {r.gpu_memory_used_gb:.2f}GB |"
            )

        # Find best by throughput
        best = max(wl_results, key=lambda r: r.throughput_rps)
        lines += [
            "",
            f"**Best throughput:** `{best.config.tags.get('variant', best.experiment_id)}` "
            f"at {best.throughput_rps:.2f} rps",
            "",
        ]

    lines += [
        "## Config Details",
        "",
        "| Variant | max_num_batched_tokens | enable_chunked_prefill | enable_prefix_caching |",
        "|---------|------------------------|------------------------|----------------------|",
        "| default       | 2048 | False | False |",
        "| chunked       | 2048 | True  | False |",
        "| prefix_cache  | 2048 | False | True  |",
        "| big_batch     | 4096 | False | False |",
        "",
        "## Notes",
        "",
        "- Warmup: 10 requests per run (not counted)",
        "- TTFT = time to first streaming token",
        "- E2E = total request latency",
        "- GPU util / mem sampled at 0.5s intervals during load phase",
        f"- Raw results: `{json_path.name}`",
    ]

    md_path.write_text("\n".join(lines))
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="both", choices=["chat_short", "long_context_qa", "both"])
    parser.add_argument("--variants", default="default,chunked,prefix_cache,big_batch")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    selected_variants = set(args.variants.split(","))

    workloads = []
    if args.workload in ("both", "chat_short"):
        workloads.append(CHAT_SHORT)
    if args.workload in ("both", "long_context_qa"):
        workloads.append(LONG_CONTEXT_QA)

    all_results: list[ExperimentResult] = []
    failed: list[str] = []

    for workload in workloads:
        configs = [c for c in make_configs(workload) if c.tags.get("variant") in selected_variants]
        prompts = get_prompts(workload)

        console.print(Rule(f"[bold]{workload.name}[/bold] — {len(configs)} configs"))

        for cfg in configs:
            console.print(f"\n[bold cyan]▶ {cfg.experiment_id}[/bold cyan]")
            try:
                result = run_experiment(cfg, prompts)
                all_results.append(result)
            except OOMError as e:
                console.print(f"  [red]OOM — skipping:[/red] {e}")
                failed.append(cfg.experiment_id)
            except StartupTimeoutError as e:
                console.print(f"  [red]Startup timeout — skipping:[/red] {e}")
                failed.append(cfg.experiment_id)
            except BenchmarkError as e:
                console.print(f"  [red]Benchmark error — skipping:[/red] {e}")
                failed.append(cfg.experiment_id)

    console.print()
    if all_results:
        print_results_table(all_results)
        report_path = generate_report(all_results, Path(args.report_dir))
        console.print(f"\n[green]Report written:[/green] {report_path}")

    if failed:
        console.print(f"\n[yellow]Failed runs ({len(failed)}):[/yellow] {', '.join(failed)}")

    if not all_results:
        console.print("[red]No successful runs.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
