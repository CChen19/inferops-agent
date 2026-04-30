"""Regression gate for commit-level eval reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    warnings: list[str]


def load_eval_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def regression_gate(
    current: dict[str, Any],
    baseline: dict[str, Any],
    strategy: str = "greedy_agent",
    max_outcome_regression_pct: float = 5.0,
    min_composite_delta: float = -0.05,
) -> GateResult:
    """Fail when a strategy regresses beyond the configured thresholds."""
    failures: list[str] = []
    warnings: list[str] = []

    current_rows = _rows_by_workload(current, strategy)
    baseline_rows = _rows_by_workload(baseline, strategy)
    if not current_rows:
        failures.append(f"No current rows for strategy '{strategy}'")
    if not baseline_rows:
        failures.append(f"No baseline rows for strategy '{strategy}'")
    if failures:
        return GateResult(False, failures, warnings)

    for workload, cur in current_rows.items():
        prev = baseline_rows.get(workload)
        if prev is None:
            warnings.append(f"No baseline row for workload '{workload}'")
            continue

        gap_delta = cur["gap_pct"] - prev["gap_pct"]
        if gap_delta > max_outcome_regression_pct:
            failures.append(
                f"{strategy}/{workload} gap regressed by {gap_delta:.2f}pp "
                f"({prev['gap_pct']:.2f}% -> {cur['gap_pct']:.2f}%)"
            )

        composite_delta = cur["composite"] - prev["composite"]
        if composite_delta < min_composite_delta:
            failures.append(
                f"{strategy}/{workload} composite dropped by {composite_delta:.3f} "
                f"({prev['composite']:.3f} -> {cur['composite']:.3f})"
            )

    return GateResult(not failures, failures, warnings)


def _rows_by_workload(report: dict[str, Any], strategy: str) -> dict[str, dict[str, Any]]:
    rows = report.get("strategies", {}).get(strategy, [])
    return {row["workload_name"]: row for row in rows}
