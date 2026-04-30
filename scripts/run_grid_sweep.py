"""Grid sweep to find ground-truth optimal configs for each golden workload.

Runs 3 × 2 × 2 = 12 configs per workload (fixed: gpu_mem_util=0.80, max_num_seqs=128,
max_model_len=2048). Saves results to the experiment memory DB and writes one
ground-truth JSON per workload under --output-dir.

Usage:
    python scripts/run_grid_sweep.py                          # all 5 workloads
    python scripts/run_grid_sweep.py --workloads chat_short   # one workload
    python scripts/run_grid_sweep.py --dry-run                # print plan only
    python scripts/run_grid_sweep.py --skip-existing          # resume after crash
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from configs.search_space import MODEL
from inferops.bench_runner import BenchmarkError, run_experiment
from inferops.memory.db import get_result_by_id, init_db, save_result
from inferops.schemas import ExperimentConfig, ModelSize, SchedulerPolicy
from workloads.definitions import ALL_WORKLOADS, get_prompts

console = Console()

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

_FIXED = dict(
    model_name=MODEL,
    model_size=ModelSize.HALF_B,
    gpu_memory_utilization=0.80,
    max_num_seqs=128,
    max_model_len=2048,
    enforce_eager=False,
    scheduler_policy=SchedulerPolicy.FCFS,
)

GRID_AXES: dict[str, list] = {
    "max_num_batched_tokens": [1024, 2048, 4096],
    "enable_chunked_prefill": [False, True],
    "enable_prefix_caching":  [False, True],
}

# Primary metric per workload → (field_name, "max"|"min")
PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "chat_short":                 ("throughput_rps",    "max"),
    "long_context_qa":            ("throughput_rps",    "max"),
    "high_concurrency_short_out": ("throughput_rps",    "max"),
    "long_generation":            ("tokens_per_second", "max"),
    "mixed_traffic":              ("throughput_rps",    "max"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _experiment_id(workload_name: str, batched: int, chunked: bool, prefix: bool) -> str:
    return f"grid_{workload_name}_t{batched}_{'c1' if chunked else 'c0'}_{'p1' if prefix else 'p0'}"


def _build_configs(workload) -> list[ExperimentConfig]:
    configs = []
    for combo in itertools.product(*GRID_AXES.values()):
        params = dict(zip(GRID_AXES.keys(), combo))
        eid = _experiment_id(
            workload.name,
            params["max_num_batched_tokens"],
            params["enable_chunked_prefill"],
            params["enable_prefix_caching"],
        )
        configs.append(ExperimentConfig(
            experiment_id=eid,
            workload=workload,
            tags={"variant": "grid", "phase": "3_sweep", **{k: str(v) for k, v in params.items()}},
            **params,
            **_FIXED,
        ))
    return configs


def _result_to_row(r) -> dict:
    return {
        "experiment_id":          r.experiment_id,
        "max_num_batched_tokens": r.config.max_num_batched_tokens,
        "enable_chunked_prefill": r.config.enable_chunked_prefill,
        "enable_prefix_caching":  r.config.enable_prefix_caching,
        "throughput_rps":         round(r.throughput_rps, 4),
        "tokens_per_second":      round(r.tokens_per_second, 2),
        "ttft_p50_ms":            round(r.ttft.p50, 2),
        "ttft_p99_ms":            round(r.ttft.p99, 2),
        "e2e_p50_ms":             round(r.e2e_latency.p50, 2),
        "e2e_p99_ms":             round(r.e2e_latency.p99, 2),
        "gpu_util_pct":           round(r.gpu_utilization_pct, 1) if r.gpu_utilization_pct else None,
        "gpu_mem_gb":             round(r.gpu_memory_used_gb, 3) if r.gpu_memory_used_gb else None,
        "success_rate":           f"{r.successful_requests}/{r.total_requests}",
    }


# ---------------------------------------------------------------------------
# Per-workload sweep
# ---------------------------------------------------------------------------

def sweep_workload(workload, output_dir: Path, skip_existing: bool, dry_run: bool) -> None:
    configs = _build_configs(workload)
    metric_name, direction = PRIMARY_METRIC[workload.name]
    prompts = get_prompts(workload) if not dry_run else []

    console.rule(f"[bold cyan]{workload.name}[/] — {len(configs)} configs")

    rows: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        tag = f"  [{i}/{len(configs)}] {cfg.experiment_id}"

        if dry_run:
            console.print(f"{tag} [dim](dry-run)[/dim]")
            continue

        if skip_existing:
            existing = get_result_by_id(cfg.experiment_id)
            if existing is not None:
                console.print(f"{tag} [dim]skipped (already in DB)[/dim]")
                rows.append(_result_to_row(existing))
                continue

        console.print(f"{tag} …")
        t0 = time.time()
        try:
            result = run_experiment(cfg, prompts)
            save_result(result)
            rows.append(_result_to_row(result))
            console.print(f"{tag} ✓  {result.throughput_rps:.2f} rps  {time.time()-t0:.0f}s")
        except BenchmarkError as exc:
            console.print(f"{tag} [red]FAILED[/]: {exc}")

    if dry_run or not rows:
        return

    # Select best config
    best = (max if direction == "max" else min)(rows, key=lambda r: r[metric_name])

    ground_truth = {
        "workload_name":      workload.name,
        "primary_metric":     metric_name,
        "higher_is_better":   direction == "max",
        "best_experiment_id": best["experiment_id"],
        "best_value":         best[metric_name],
        "sweep_date":         datetime.now(timezone.utc).isoformat(),
        "n_configs":          len(rows),
        "experiments":        rows,
    }

    out_path = output_dir / f"{workload.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ground_truth, indent=2))

    console.print(f"  → [green]{out_path}[/]")
    console.print(f"  → best: {best['experiment_id']}  ({metric_name}={best[metric_name]})")

    # Print mini summary table
    t = Table(show_header=True, show_lines=False, box=None, padding=(0, 1))
    t.add_column("experiment_id", style="dim")
    t.add_column(metric_name, justify="right")
    t.add_column("ttft_p99", justify="right")
    t.add_column("e2e_p50", justify="right")
    for row in sorted(rows, key=lambda r: r[metric_name], reverse=(direction == "max")):
        marker = "★" if row["experiment_id"] == best["experiment_id"] else " "
        t.add_row(
            f"{marker} {row['experiment_id']}",
            str(row[metric_name]),
            f"{row['ttft_p99_ms']}ms",
            f"{row['e2e_p50_ms']}ms",
        )
    console.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep → ground truth JSON files")
    parser.add_argument("--workloads", nargs="+", default=["all"],
                        help="Workload names or 'all' (default: all 5)")
    parser.add_argument("--output-dir", default="data/ground_truth",
                        help="Output directory for ground truth JSON files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Load already-run experiments from DB instead of re-running")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without running anything")
    args = parser.parse_args()

    init_db()

    all_wl_map = {w.name: w for w in ALL_WORKLOADS}
    if args.workloads == ["all"]:
        selected = ALL_WORKLOADS
    else:
        invalid = set(args.workloads) - all_wl_map.keys()
        if invalid:
            console.print(f"[red]Unknown workloads: {invalid}. Valid: {list(all_wl_map)}")
            sys.exit(1)
        selected = [all_wl_map[n] for n in args.workloads]

    n_configs = len(list(itertools.product(*GRID_AXES.values())))
    console.print(
        f"\n[bold]Grid sweep[/]: {len(selected)} workload(s) × {n_configs} configs "
        f"= {len(selected) * n_configs} total experiments"
    )
    if args.dry_run:
        console.print("[yellow]DRY RUN — no experiments will be executed[/]\n")

    t_start = time.time()
    for wl in selected:
        sweep_workload(wl, Path(args.output_dir), args.skip_existing, args.dry_run)

    console.print(f"\n[bold green]Done[/] in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
