"""Fair-eval search strategies — differ only in next-config selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC
from inferops.eval.protocol import (
    BudgetPolicy,
    HiddenResultFixture,
    SearchSpace,
    TrialLedger,
    is_better,
    is_valid_observation,
    score_run,
)


@dataclass
class StrategyRun:
    strategy_name: str
    workload_name: str
    ledger: TrialLedger
    score: dict[str, Any]


def _metric_for(workload_name: str) -> tuple[str, str]:
    return WORKLOAD_PRIMARY_METRIC[workload_name]


def _run_baseline(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    ledger: TrialLedger,
) -> tuple[dict[str, Any], Any] | None:
    default = fixture.search_space.default_config()
    if not budget.charge_baseline():
        return None
    obs = fixture.observe(default)
    ledger.add(
        config=default,
        observation=obs,
        paid=True,
        kind="baseline",
    )
    return default, obs


def run_default_strategy(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    gt_optimum: dict[str, Any] | None = None,
) -> StrategyRun:
    """Evaluate the default config only (baseline slot)."""
    ledger = TrialLedger(budget=budget)
    _run_baseline(fixture, budget, ledger)
    run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
    return StrategyRun(
        strategy_name="default",
        workload_name=workload_name,
        ledger=ledger,
        score=_score_dict(run_score),
    )


def run_random_strategy(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    seed: int = 42,
    gt_optimum: dict[str, Any] | None = None,
) -> StrategyRun:
    """Sample unseen legal configs; observe only after choosing."""
    space = fixture.search_space
    ledger = TrialLedger(budget=budget)
    baseline = _run_baseline(fixture, budget, ledger)
    current_cfg, current_obs = baseline if baseline else (None, None)

    legal = fixture.legal_configs()
    rng = random.Random(seed)
    rng.shuffle(legal)

    for config in legal:
        key = space.config_key(config)
        if key in ledger.tried_keys(space):
            continue
        if not budget.charge_trial(duplicate=False):
            break
        obs = fixture.observe(config)
        ledger.add(config=config, observation=obs, paid=True, kind="trial")
        metric, direction = _metric_for(workload_name)
        if (
            current_obs is None
            or (is_valid_observation(obs, metric) and is_better(obs, current_obs, metric, direction))
        ):
            current_cfg, current_obs = config, obs

    del current_cfg, current_obs  # best tracked in ledger for scoring
    run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
    return StrategyRun(
        strategy_name="random",
        workload_name=workload_name,
        ledger=ledger,
        score=_score_dict(run_score),
    )


def run_online_local_search(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    gt_optimum: dict[str, Any] | None = None,
) -> StrategyRun:
    """One-knob local search without peeking at unread neighbor scores."""
    space = fixture.search_space
    ledger = TrialLedger(budget=budget)
    metric, direction = _metric_for(workload_name)

    baseline = _run_baseline(fixture, budget, ledger)
    if baseline is None:
        run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
        return StrategyRun("online_local_search", workload_name, ledger, _score_dict(run_score))

    current_cfg, current_obs = baseline
    pool = fixture.legal_configs()

    while budget.slots_remaining() > 0:
        tried = ledger.tried_keys(space)
        neighbors = [
            row for row in space.neighbors(current_cfg, pool)
            if space.config_key(row) not in tried
        ]
        if not neighbors:
            neighbors = [row for row in pool if space.config_key(row) not in tried]
        if not neighbors:
            break

        # Pick WITHOUT observing — stable key order, not best hidden metric.
        neighbors.sort(key=lambda row: space.config_key(row))
        chosen = neighbors[0]

        if not budget.charge_trial(duplicate=False):
            break
        obs = fixture.observe(chosen)
        ledger.add(config=chosen, observation=obs, paid=True, kind="trial")
        if is_valid_observation(obs, metric) and is_better(obs, current_obs, metric, direction):
            current_cfg, current_obs = chosen, obs

    run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
    return StrategyRun(
        strategy_name="online_local_search",
        workload_name=workload_name,
        ledger=ledger,
        score=_score_dict(run_score),
    )


def run_all_strategies(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    seed: int = 42,
    gt_optimum: dict[str, Any] | None = None,
) -> dict[str, StrategyRun]:
    """Thin wrapper: run default, random, and online_local_search on one fixture."""
    total = budget.total_slots
    return {
        "default": run_default_strategy(
            fixture,
            BudgetPolicy(total_slots=total),
            workload_name=workload_name,
            gt_optimum=gt_optimum,
        ),
        "random": run_random_strategy(
            fixture,
            BudgetPolicy(total_slots=total),
            workload_name=workload_name,
            seed=seed,
            gt_optimum=gt_optimum,
        ),
        "online_local_search": run_online_local_search(
            fixture,
            BudgetPolicy(total_slots=total),
            workload_name=workload_name,
            gt_optimum=gt_optimum,
        ),
    }


def _score_dict(run_score: Any) -> dict[str, Any]:
    return {
        "valid_result_in_budget": run_score.valid_result_in_budget,
        "first_valid_n": run_score.first_valid_n,
        "confirmed_gain": run_score.confirmed_gain,
        "wasted_trials": run_score.wasted_trials,
        "n_paid": run_score.n_paid,
        "gap_pct": run_score.gap_pct,
    }
