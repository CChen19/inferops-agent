"""Small live comparison: production planner report vs honest local search."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC
from inferops.eval.protocol import BudgetPolicy, Observation, SearchSpace, primary_value
from inferops.eval.strategies import StrategyRun, run_online_local_search
from inferops.tools.run_benchmark import (
    RunBenchmarkInput,
    RunBenchmarkOutput,
    run_benchmark,
)
from inferops.tools.vllm_process import get_vllm_python

LIVE_LLM_BOUNDARY = "live_openrouter"
SEARCH_LLM_BOUNDARY = "none"
TOOL_BOUNDARY = "managed_local_vllm"
CLAIM_LEVEL = "observe_after_pick_protocol_score"

BenchmarkFn = Callable[[RunBenchmarkInput], RunBenchmarkOutput]


class LiveCompareBlocked(RuntimeError):
    """A missing or unsafe live precondition; this is not a passing run."""


def current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def require_managed_live_conditions(
    vllm_python: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Fail before benchmarking if mode is external or the dedicated port is occupied."""
    if os.getenv("INFEROPS_EXTERNAL_VLLM"):
        raise LiveCompareBlocked("INFEROPS_EXTERNAL_VLLM must be unset (managed mode required)")
    if vllm_python is None:
        try:
            vllm_python = get_vllm_python()
        except RuntimeError as exc:
            raise LiveCompareBlocked(str(exc)) from exc
    python_path = Path(vllm_python)
    if not python_path.is_absolute() or not python_path.is_file():
        raise LiveCompareBlocked(f"managed vLLM Python is unavailable: {vllm_python}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        if sock.connect_ex((host, port)) == 0:
            raise LiveCompareBlocked(
                f"{host}:{port} is occupied by an unknown process; refusing to stop it"
            )


@dataclass
class LiveBenchmarkFixture:
    """Duck-typed strategy fixture whose observations are real managed benchmarks."""

    workload_name: str
    model_name: str
    session_prefix: str
    max_ttft_ms: float
    benchmark_fn: BenchmarkFn = run_benchmark
    search_space: SearchSpace = field(default_factory=SearchSpace.from_agent_search_space)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def legal_configs(self) -> list[dict[str, Any]]:
        return self.search_space.enumerate_configs()

    def observe(self, config: dict[str, Any]) -> Observation:
        """Run only the already-picked config, then reveal its measured result."""
        if not self.search_space.is_legal(config):
            raise KeyError(f"illegal config: {config}")

        step = len(self.observations) + 1
        experiment_id = f"{self.session_prefix}{step:02d}"
        inp = RunBenchmarkInput(
            experiment_id=experiment_id,
            config_patch=dict(config),
            workload_name=self.workload_name,
            model_name=self.model_name,
            session_id=self.session_prefix,
            persist=True,
        )
        try:
            out = self.benchmark_fn(inp)
            engine_valid = out.status == "valid"
            ttft_ok = out.ttft_p99_ms is not None and out.ttft_p99_ms <= self.max_ttft_ms
            error_ok = out.error_rate is not None and out.error_rate <= 0.05
            protocol_valid = engine_valid and ttft_ok and error_ok
            self.observations.append(
                {
                    "step": step,
                    "experiment_id": out.experiment_id,
                    "run_id": out.run_id or None,
                    "mlflow_run_id": out.mlflow_run_id,
                    "config": dict(config),
                    "validity": "valid" if protocol_valid else "invalid",
                    "engine_validity": out.status,
                    "rps": out.throughput_rps,
                    "ttft_p99_ms": out.ttft_p99_ms,
                    "error_rate": out.error_rate,
                    "slo_ok": ttft_ok and error_ok,
                    "ledger_path": out.ledger_path,
                    "error": out.error or None,
                }
            )
            metrics = {
                key: value
                for key, value in {
                    "throughput_rps": out.throughput_rps,
                    "tokens_per_second": out.tokens_per_second,
                    "ttft_p50_ms": out.ttft_p50_ms,
                    "ttft_p99_ms": out.ttft_p99_ms,
                    "e2e_p50_ms": out.e2e_p50_ms,
                    "e2e_p99_ms": out.e2e_p99_ms,
                }.items()
                if value is not None
            }
            return Observation(
                metrics=metrics,
                validity_status="valid" if protocol_valid else "invalid",
                error_rate=out.error_rate,
                config_evidence=engine_valid,
                bottleneck="unknown",
            )
        except Exception as exc:
            self.observations.append(
                {
                    "step": step,
                    "experiment_id": experiment_id,
                    "run_id": None,
                    "mlflow_run_id": None,
                    "config": dict(config),
                    "validity": "failed",
                    "engine_validity": "failed",
                    "rps": None,
                    "ttft_p99_ms": None,
                    "error_rate": None,
                    "slo_ok": False,
                    "ledger_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return Observation(
                metrics={},
                validity_status="failed",
                error_rate=None,
                config_evidence=False,
                bottleneck="unknown",
            )


def ingest_planner_summary(path: str | Path) -> dict[str, Any]:
    """Ingest planner metadata plus the adjacent Markdown report, without credentials."""
    summary_path = Path(path)
    if summary_path.is_dir():
        summary_path = summary_path / "meta.json"
    try:
        meta = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveCompareBlocked(f"cannot ingest planner summary {summary_path}: {exc}") from exc

    report_path = summary_path.parent / "case.md"
    markdown = report_path.read_text() if report_path.exists() else ""
    observations = _parse_planner_experiment_log(markdown)
    best_id, best_rps = _parse_best_planner_result(markdown)
    best_ttft = _parse_constraint_observed(markdown, "ttft_p99_ms")
    for row in observations:
        if row["experiment_id"] == best_id:
            row["ttft_p99_ms"] = best_ttft

    return {
        "strategy": "planner",
        "source": "ingested_report",
        "source_path": str(summary_path),
        "source_sha": meta.get("git_sha"),
        "model_name": meta.get("model_name"),
        "workload": meta.get("workload"),
        "max_ttft_ms": meta.get("max_ttft_ms"),
        "llm_boundary": meta.get("llm_boundary", LIVE_LLM_BOUNDARY),
        "tool_boundary": meta.get("tool_boundary", TOOL_BOUNDARY),
        "budget": meta.get("budget"),
        "budget_used": len(meta.get("tried_experiment_ids", observations)),
        "experiment_ids": meta.get(
            "tried_experiment_ids", [row["experiment_id"] for row in observations]
        ),
        "observations": observations,
        "best": {
            "experiment_id": best_id,
            "validity": next(
                (row["validity"] for row in observations if row["experiment_id"] == best_id),
                None,
            ),
            "rps": best_rps,
            "ttft_p99_ms": best_ttft,
        },
        "decision_kind": meta.get("decision_kind"),
        "claim_level": "production_decision_report",
    }


def run_live_search(
    *,
    budget: int,
    workload_name: str,
    model_name: str,
    session_prefix: str,
    max_ttft_ms: float,
    benchmark_fn: BenchmarkFn = run_benchmark,
) -> tuple[StrategyRun, LiveBenchmarkFixture]:
    fixture = LiveBenchmarkFixture(
        workload_name=workload_name,
        model_name=model_name,
        session_prefix=session_prefix,
        max_ttft_ms=max_ttft_ms,
        benchmark_fn=benchmark_fn,
    )
    run = run_online_local_search(
        fixture,  # type: ignore[arg-type] -- intentional live observe adapter
        BudgetPolicy(total_slots=budget),
        workload_name=workload_name,
    )
    return run, fixture


def build_live_compare_report(
    *,
    commit_sha: str,
    planner: dict[str, Any],
    search_run: StrategyRun | None,
    fixture: LiveBenchmarkFixture | None,
    budget: int,
    workload_name: str,
    model_name: str,
    max_ttft_ms: float,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    search = _search_arm(search_run, fixture, budget, workload_name)
    return {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "live_fair_compare",
        "status": "blocked" if blocked_reason else "complete",
        "blocked_reason": blocked_reason,
        "conditions": {
            "model_name": model_name,
            "workload": workload_name,
            "budget": budget,
            "max_ttft_ms": max_ttft_ms,
            "search_space": SearchSpace.from_agent_search_space().axes,
            "protocol": "observe_after_pick",
        },
        "claim_level": CLAIM_LEVEL,
        "disclaimer": (
            "Search-path scores are observe-after-pick protocol results, not deploy "
            "recommendations. No confirmed_gain is claimed without Week-2 confirmation."
        ),
        "arms": {"planner": planner, "online_local_search": search},
    }


def write_live_compare_outputs(
    report: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "live_fair_compare.json"
    md_path = out / "live_fair_compare.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_live_compare_markdown(report))
    return md_path, json_path


def render_live_compare_markdown(report: dict[str, Any]) -> str:
    conditions = report["conditions"]
    lines = [
        "# InferOps Live Fair Compare",
        "",
        f"- SHA: `{report['commit_sha']}`",
        f"- Status: `{report['status']}`",
        f"- Model: `{conditions['model_name']}`",
        f"- Workload: `{conditions['workload']}`",
        f"- Budget: `{conditions['budget']}` per arm",
        f"- SLO: `ttft_p99_ms <= {conditions['max_ttft_ms']}` and `error_rate <= 0.05`",
        f"- Claim level: `{report['claim_level']}`",
        "",
        f"> {report['disclaimer']}",
        "",
    ]
    if report.get("blocked_reason"):
        lines += [f"## Blocked\n\n{report['blocked_reason']}", ""]

    planner_sha = _arm_code_sha("planner", report["arms"]["planner"], report)
    search_sha = _arm_code_sha("online_local_search", report["arms"]["online_local_search"], report)

    lines += ["## Arms", ""]
    if planner_sha and search_sha and planner_sha != search_sha:
        lines += [
            "> **Warning:** Arms are not on the same code version. "
            f"Planner was ingested from `{planner_sha}`; "
            f"live search ran at `{search_sha}`.",
            "",
        ]

    lines += [
        "| Arm | Code SHA | LLM boundary | Tool boundary | Budget used | Best ID | "
        "Validity | RPS | TTFT p99 | Decision |",
        "|---|---|---|---|---:|---|---|---:|---:|---|",
    ]
    for name in ("planner", "online_local_search"):
        arm = report["arms"][name]
        best = arm.get("best") or {}
        lines.append(
            f"| {name} | {_fmt_sha(_arm_code_sha(name, arm, report))} | "
            f"{arm.get('llm_boundary')} | {arm.get('tool_boundary')} | "
            f"{arm.get('budget_used', 0)} | {best.get('experiment_id') or '—'} | "
            f"{best.get('validity') or '—'} | {_fmt(best.get('rps'))} | "
            f"{_fmt(best.get('ttft_p99_ms'))} | {arm.get('decision_kind') or '—'} |"
        )

    for name in ("planner", "online_local_search"):
        arm = report["arms"][name]
        lines += [
            "",
            f"### {name} observations",
            "",
            "| # | Experiment ID | Validity | Engine validity | RPS | TTFT p99 |",
            "|---:|---|---|---|---:|---:|",
        ]
        for row in arm.get("observations", []):
            lines.append(
                f"| {row.get('step')} | {row.get('experiment_id')} | "
                f"{row.get('validity')} | {row.get('engine_validity')} | "
                f"{_fmt(row.get('rps'))} | {_fmt(row.get('ttft_p99_ms'))} |"
            )
    return "\n".join(lines) + "\n"


def _search_arm(
    run: StrategyRun | None,
    fixture: LiveBenchmarkFixture | None,
    budget: int,
    workload_name: str,
) -> dict[str, Any]:
    observations = fixture.observations if fixture else []
    best: dict[str, Any] | None = None
    if run is not None:
        metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]
        picked = run.ledger.best_valid(metric, direction)
        if picked:
            config, obs = picked
            matching = next((row for row in observations if row["config"] == config), {})
            best = {
                "experiment_id": matching.get("experiment_id"),
                "validity": matching.get("validity"),
                "rps": matching.get("rps", primary_value(obs, "throughput_rps")),
                "ttft_p99_ms": matching.get("ttft_p99_ms"),
                "config": config,
            }
    return {
        "strategy": "online_local_search",
        "source": "live_managed_benchmark",
        "llm_boundary": SEARCH_LLM_BOUNDARY,
        "tool_boundary": TOOL_BOUNDARY,
        "budget": budget,
        "budget_used": run.score["n_paid"] if run else len(observations),
        "experiment_ids": [row["experiment_id"] for row in observations],
        "observations": observations,
        "best": best,
        "decision_kind": None,
        "claim_level": CLAIM_LEVEL,
        "protocol_score": run.score if run else None,
    }


def _parse_planner_experiment_log(markdown: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_log = False
    for line in markdown.splitlines():
        if line.strip() == "## Experiment Log":
            in_log = True
            continue
        if in_log and line.startswith("## "):
            break
        if not in_log or not line.startswith("| ") or "`" not in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 9 or not cells[0].isdigit():
            continue
        rows.append(
            {
                "step": int(cells[0]),
                "experiment_id": cells[1],
                "run_id": cells[2],
                "mlflow_run_id": cells[3],
                "validity": cells[4],
                "engine_validity": cells[4],
                "rps": _float_or_none(cells[7]),
                "ttft_p99_ms": None,
            }
        )
    return rows


def _parse_best_planner_result(markdown: str) -> tuple[str | None, float | None]:
    match = re.search(r"\*\*Best found:\*\* `([^`]+)`\s+rps=([0-9.]+)", markdown)
    return (match.group(1), float(match.group(2))) if match else (None, None)


def _parse_constraint_observed(markdown: str, metric: str) -> float | None:
    match = re.search(
        rf"\|\s*constraint\s*\|\s*`{re.escape(metric)}`\s*\|[^|]*\|[^|]*\|\s*([0-9.]+)",
        markdown,
    )
    return float(match.group(1)) if match else None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _arm_code_sha(name: str, arm: dict[str, Any], report: dict[str, Any]) -> str | None:
    if name == "planner":
        return arm.get("source_sha") or arm.get("git_sha")
    return report.get("commit_sha")


def _fmt_sha(value: str | None) -> str:
    return "—" if not value else f"`{value}`"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
