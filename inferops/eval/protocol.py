"""Fair eval protocol — shared search space, observations, budget, SLO, and scoring.

Strategies may only differ in how they pick the next config; everything else is
identical across runs.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from inferops.agent.reflect_constraints import check_slo
from inferops.agent.state import AGENT_SEARCH_SPACE
from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC

# Ground-truth grid sweeps vary only these three axes (see scripts/run_grid_sweep.py).
GT_THREE_AXIS: tuple[str, ...] = (
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
)

METRIC_FIELDS: tuple[str, ...] = (
    "throughput_rps",
    "tokens_per_second",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
)


@dataclass(frozen=True)
class SearchSpace:
    """Legal config combinations for a fair eval run."""

    axes: dict[str, list[Any]]

    @classmethod
    def from_agent_search_space(cls) -> SearchSpace:
        return cls(axes={k: list(v) for k, v in AGENT_SEARCH_SPACE.items()})

    @classmethod
    def gt_three_axis_projection(
        cls,
        source: dict[str, list[Any]] | None = None,
    ) -> SearchSpace:
        """Project the full agent search space to the 3-axis GT grid axes."""
        src = source or AGENT_SEARCH_SPACE
        return cls(axes={axis: list(src[axis]) for axis in GT_THREE_AXIS})

    @classmethod
    def from_gt_rows(cls, rows: list[dict[str, Any]]) -> SearchSpace:
        """Build a 3-axis search space from observed GT sweep rows (sparse grids)."""
        axes: dict[str, list[Any]] = {}
        for axis in GT_THREE_AXIS:
            values = sorted({row[axis] for row in rows if axis in row}, key=str)
            if values:
                axes[axis] = values
        return cls(axes=axes)

    def knob_names(self) -> tuple[str, ...]:
        return tuple(self.axes)

    def enumerate_configs(self) -> list[dict[str, Any]]:
        names = self.knob_names()
        values = [self.axes[name] for name in names]
        return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)]

    def config_key(self, config: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(config.get(name) for name in self.knob_names())

    def diff_count(self, left: dict[str, Any], right: dict[str, Any]) -> int:
        return sum(left.get(name) != right.get(name) for name in self.knob_names())

    def neighbors(
        self,
        config: dict[str, Any],
        pool: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = pool if pool is not None else self.enumerate_configs()
        return [row for row in candidates if self.diff_count(row, config) == 1]

    def default_config(self) -> dict[str, Any]:
        """Production default projected into this search space."""
        cfg: dict[str, Any] = {}
        for name, values in self.axes.items():
            if name == "max_num_batched_tokens":
                cfg[name] = min(values)
            elif name == "max_num_seqs" and 128 in values:
                cfg[name] = 128
            elif name in ("enable_chunked_prefill", "enable_prefix_caching"):
                cfg[name] = False if False in values else values[0]
            else:
                cfg[name] = values[0]
        return cfg

    def is_legal(self, config: dict[str, Any]) -> bool:
        return all(config.get(name) in self.axes[name] for name in self.knob_names())


@dataclass
class BudgetPolicy:
    """Slot accounting shared by every strategy.

    ``total_slots`` includes the baseline run. Baseline always charges 1 slot.
    Confirmation runs charge 1 slot when used. Duplicate re-runs are free.
    """

    total_slots: int
    n_paid: int = 0
    baseline_charged: bool = False
    confirmation_charged: bool = False

    def slots_remaining(self) -> int:
        return max(0, self.total_slots - self.n_paid)

    def charge_baseline(self) -> bool:
        if self.baseline_charged:
            return True
        if self.n_paid >= self.total_slots:
            return False
        self.baseline_charged = True
        self.n_paid += 1
        return True

    def charge_trial(self, *, duplicate: bool = False, confirmation: bool = False) -> bool:
        if duplicate:
            return True
        cost = 1 + (1 if confirmation else 0)
        if self.n_paid + cost > self.total_slots:
            return False
        self.n_paid += 1
        if confirmation:
            self.confirmation_charged = True
            self.n_paid += 1
        return True


@dataclass(frozen=True)
class Observation:
    """What a strategy may learn from running one config — no hidden GT score."""

    metrics: dict[str, Any]
    validity_status: str
    error_rate: float | None
    config_evidence: bool
    bottleneck: str

    def to_summary(self) -> dict[str, Any]:
        return {
            **self.metrics,
            "validity_status": self.validity_status,
            "error_rate": self.error_rate,
            "has_config_evidence": self.config_evidence,
            "bottleneck": self.bottleneck,
        }


class SLOPolicy:
    """Thin wrapper over ``check_slo`` — missing ``error_rate`` fails closed."""

    @staticmethod
    def check(observation: Observation | dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(observation, Observation):
            return check_slo(observation.to_summary())
        return check_slo(observation)

    @staticmethod
    def is_ok(observation: Observation | dict[str, Any] | None) -> bool:
        return bool(SLOPolicy.check(observation)["ok"])


def is_valid_observation(observation: Observation, metric: str | None = None) -> bool:
    """Usable observation: valid status, SLO ok, config evidence, usable primary.

    When ``metric`` is provided, the primary must pass ``primary_value``
    (missing / non-finite / non-numeric / bool fail closed). Rows that fail
    remain in the ledger — callers must not delete them to pretty the score.
    """
    if observation.validity_status != "valid":
        return False
    if not observation.config_evidence:
        return False
    if not SLOPolicy.is_ok(observation):
        return False
    if metric is not None:
        try:
            primary_value(observation, metric)
        except ValueError:
            return False
    return True


def primary_value(observation: Observation, metric: str) -> float:
    """Return the primary metric as a finite float. Missing or non-numeric fails closed."""
    if metric not in observation.metrics:
        raise ValueError(f"missing primary metric {metric!r}")
    value = observation.metrics[metric]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"primary metric {metric!r} is not a finite number: {value!r}")
    return float(value)


def is_better(
    candidate: Observation,
    current: Observation,
    metric: str,
    direction: str,
) -> bool:
    cand = primary_value(candidate, metric)
    cur = primary_value(current, metric)
    return cand > cur if direction == "max" else cand < cur


@dataclass
class TrialRecord:
    step: int
    config: dict[str, Any]
    observation: Observation | None
    paid: bool
    duplicate: bool
    confirmation: bool
    kind: str  # baseline | trial | confirmation


@dataclass
class TrialLedger:
    budget: BudgetPolicy
    records: list[TrialRecord] = field(default_factory=list)

    def add(
        self,
        *,
        config: dict[str, Any],
        observation: Observation | None,
        paid: bool,
        duplicate: bool = False,
        confirmation: bool = False,
        kind: str = "trial",
    ) -> TrialRecord:
        record = TrialRecord(
            step=len(self.records) + 1,
            config=dict(config),
            observation=observation,
            paid=paid,
            duplicate=duplicate,
            confirmation=confirmation,
            kind=kind,
        )
        self.records.append(record)
        return record

    @property
    def n_paid(self) -> int:
        return self.budget.n_paid

    def tried_keys(self, space: SearchSpace) -> set[tuple[Any, ...]]:
        return {space.config_key(record.config) for record in self.records}

    def first_valid_n(
        self,
        metric: str | None = None,
        direction: str | None = None,
    ) -> int | None:
        del direction  # ranking direction is strategy-local; usability needs metric
        for record in self.records:
            obs = record.observation
            if obs is not None and is_valid_observation(obs, metric):
                return record.step
        return None

    def wasted_trials(self, metric: str, direction: str) -> int:
        """Paid non-baseline trials that did not become the best valid observation."""
        best: Observation | None = None
        wasted = 0
        for record in self.records:
            if record.kind == "baseline" or not record.paid or record.duplicate:
                continue
            obs = record.observation
            if obs is None or not is_valid_observation(obs, metric):
                wasted += 1
                continue
            if best is None:
                best = obs
                continue
            if is_better(obs, best, metric, direction):
                best = obs
            else:
                wasted += 1
        return wasted

    def best_valid(
        self,
        metric: str,
        direction: str,
    ) -> tuple[dict[str, Any], Observation] | None:
        best_cfg: dict[str, Any] | None = None
        best_obs: Observation | None = None
        for record in self.records:
            obs = record.observation
            if obs is None or not is_valid_observation(obs, metric):
                continue
            if best_obs is None or is_better(obs, best_obs, metric, direction):
                best_cfg = record.config
                best_obs = obs
        if best_cfg is None or best_obs is None:
            return None
        return best_cfg, best_obs


@dataclass(frozen=True)
class RunScore:
    """Decomposed protocol scores. ``valid_result_in_budget`` is not a business win.

    ``valid_result_in_budget`` is True when the ledger has at least one SLO-valid
    observation and paid slots stayed within the budget. A baseline-only run
    qualifies. It does **not** mean goals were met or a gain was confirmed.
    ``confirmed_gain`` stays None unless a caller supplies a confirmation result.
    """

    valid_result_in_budget: bool
    first_valid_n: int | None
    confirmed_gain: float | None
    wasted_trials: int
    n_paid: int
    gap_pct: float | None


def score_run(
    ledger: TrialLedger,
    *,
    workload_name: str,
    gt_optimum: dict[str, Any] | None = None,
    confirmed_gain: float | None = None,
) -> RunScore:
    """Emit decomposed scores — never a single composite-only result."""
    metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]
    best = ledger.best_valid(metric, direction)
    first_valid = ledger.first_valid_n(metric, direction)
    valid_in_budget = best is not None and ledger.n_paid <= ledger.budget.total_slots

    gap_pct: float | None = None
    if gt_optimum is not None and best is not None:
        _cfg, best_obs = best
        gt_val = float(gt_optimum.get("best_value", gt_optimum.get(metric, 0.0)))
        agent_val = primary_value(best_obs, metric)
        if direction == "max":
            gap_pct = (gt_val - agent_val) / gt_val * 100 if gt_val else 0.0
        else:
            gap_pct = (agent_val - gt_val) / gt_val * 100 if gt_val else 0.0
        gap_pct = round(gap_pct, 2)

    return RunScore(
        valid_result_in_budget=valid_in_budget,
        first_valid_n=first_valid,
        confirmed_gain=confirmed_gain,
        wasted_trials=ledger.wasted_trials(metric, direction),
        n_paid=ledger.n_paid,
        gap_pct=gap_pct,
    )


ObserveFn = Callable[[dict[str, Any]], Observation]


@dataclass
class HiddenResultFixture:
    """Hidden config→metrics table revealed only through ``observe``."""

    search_space: SearchSpace
    _rows: dict[tuple[Any, ...], dict[str, Any]]

    def observe(self, config: dict[str, Any]) -> Observation:
        if not self.search_space.is_legal(config):
            raise KeyError(f"illegal config: {config}")
        key = self.search_space.config_key(config)
        if key not in self._rows:
            raise KeyError(f"no fixture row for config key {key!r}")
        return row_to_observation(self._rows[key])

    def legal_configs(self) -> list[dict[str, Any]]:
        names = self.search_space.knob_names()
        return [dict(zip(names, key, strict=True)) for key in self._rows]

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        search_space: SearchSpace | None = None,
    ) -> HiddenResultFixture:
        space = search_space or SearchSpace.from_gt_rows(rows)
        indexed = {space.config_key(row): row for row in rows}
        return cls(search_space=space, _rows=indexed)

    @classmethod
    def from_ground_truth(
        cls,
        ground_truth: dict[str, Any],
        search_space: SearchSpace | None = None,
    ) -> HiddenResultFixture:
        space = search_space or SearchSpace.gt_three_axis_projection()
        return cls.from_rows(list(ground_truth.get("experiments", [])), space)


def row_to_observation(row: dict[str, Any]) -> Observation:
    metrics = {name: row[name] for name in METRIC_FIELDS if name in row}
    return Observation(
        metrics=metrics,
        validity_status=str(row.get("validity_status", "valid")),
        error_rate=row.get("error_rate"),
        config_evidence=bool(row.get("has_config_evidence", True)),
        bottleneck=str(row.get("bottleneck", "unknown")),
    )
