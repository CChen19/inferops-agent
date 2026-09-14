"""Regression gate for commit-level eval reports."""

from __future__ import annotations

import json
import math
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
    strategy: str = "online_local_search",
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

    all_workloads = sorted(set(current_rows) | set(baseline_rows))
    for workload in all_workloads:
        cur = current_rows.get(workload)
        prev = baseline_rows.get(workload)
        if cur is None:
            failures.append(f"No current row for workload '{workload}'")
            continue
        if prev is None:
            failures.append(f"No baseline row for workload '{workload}'")
            continue

        metric_failures = _metric_failures(strategy, workload, cur, prev)
        if metric_failures:
            failures.extend(metric_failures)
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


def _metric_failures(
    strategy: str,
    workload: str,
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for side, row in (("current", current), ("baseline", baseline)):
        for field in ("gap_pct", "composite"):
            value = row.get(field)
            prefix = f"{strategy}/{workload} {side} {field}"
            if value is None:
                failures.append(f"{prefix} is None")
            elif not isinstance(value, (int, float)):
                failures.append(f"{prefix} is non-numeric")
            elif isinstance(value, float) and math.isnan(value):
                failures.append(f"{prefix} is NaN")
    return failures


def _rows_by_workload(report: dict[str, Any], strategy: str) -> dict[str, dict[str, Any]]:
    rows = report.get("strategies", {}).get(strategy, [])
    return {row["workload_name"]: row for row in rows}
