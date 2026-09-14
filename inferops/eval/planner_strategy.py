"""Fair-protocol planner strategies — production planner_node, hidden GT observe.

Differs from ``real_graph`` eval: no ``build_graph``, no stub benchmark. The planner
proposes a (param, value); we map it to a legal ``SearchSpace`` config, charge a
trial via ``BudgetPolicy``, then ``fixture.observe(config)``. Strategies never read
unobserved GT rows to choose the next config.

Wasted-trial policy (illegal / already-tried proposals):
  - Param outside the fixture search-space axes, or value out of range: skip the
    hypothesis without charging (planner may emit knobs like ``max_num_seqs`` that
    are not in the 3-axis GT grid).
  - Config key already in the ledger: skip without charging.
  - Mapped config legal but missing from the fixture table: charge one wasted trial
    slot with ``observation=None`` — never substitute a different unread row.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

from inferops.agent import planner as planner_module
from inferops.agent.planner import planner_node
from inferops.agent.state import AgentState, ExperimentSummary, initial_state
from inferops.eval.metrics import WORKLOAD_PRIMARY_METRIC
from inferops.eval.protocol import (
    BudgetPolicy,
    HiddenResultFixture,
    Observation,
    TrialLedger,
    is_better,
    is_valid_observation,
    primary_value,
    score_run,
)
from inferops.eval.real_graph import ScriptedBottleneckLLM
from inferops.eval.strategies import StrategyRun, _run_baseline, _score_dict


def _observation_to_summary(
    *,
    config: dict[str, Any],
    observation: Observation,
    workload_name: str,
    experiment_id: str,
    param_changed: str | None = None,
    value_changed: Any = None,
    baseline_primary: float,
) -> ExperimentSummary:
    """Build planner-visible history from a fair-protocol observation."""
    metric, _ = WORKLOAD_PRIMARY_METRIC[workload_name]
    primary_val = primary_value(observation, metric)
    vs_baseline: float | None = None
    if baseline_primary:
        vs_baseline = (primary_val - baseline_primary) / baseline_primary * 100

    promotable = (
        observation.validity_status == "valid"
        and observation.config_evidence
        and observation.error_rate is not None
    )

    return ExperimentSummary(
        experiment_id=experiment_id,
        param_changed=param_changed,
        value_changed=value_changed,
        throughput_rps=observation.metrics.get("throughput_rps"),
        tokens_per_second=observation.metrics.get("tokens_per_second"),
        ttft_p50_ms=observation.metrics.get("ttft_p50_ms"),
        ttft_p99_ms=observation.metrics.get("ttft_p99_ms"),
        e2e_p50_ms=observation.metrics.get("e2e_p50_ms"),
        bottleneck=observation.bottleneck,
        vs_baseline_pct=round(vs_baseline, 2) if vs_baseline is not None else None,
        baseline_primary=baseline_primary,
        run_id=uuid.uuid4().hex,
        validity_status=observation.validity_status,
        mlflow_run_id=None,
        has_config_evidence=observation.config_evidence,
        promotable=promotable,
        failure_reason="",
        error_rate=observation.error_rate,
    )


def _apply_hypothesis(
    base_config: dict[str, Any],
    param: str,
    value: Any,
    space,
) -> dict[str, Any] | None:
    """Map planner (param, value) onto a legal search-space config, or None."""
    if param not in space.knob_names():
        return None
    allowed = space.axes[param]
    if isinstance(allowed[0], bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                value = True
            elif lowered in ("false", "0", "no"):
                value = False
            else:
                return None
        else:
            value = bool(value)
    else:
        try:
            value = type(allowed[0])(value)
        except (TypeError, ValueError):
            return None
    if value not in allowed:
        return None
    cfg = dict(base_config)
    cfg[param] = value
    if not space.is_legal(cfg):
        return None
    return cfg


def _build_planner_state(
    *,
    workload_name: str,
    bottleneck: str,
    baseline_summary: ExperimentSummary,
    best_summary: ExperimentSummary | None,
    summaries: list[ExperimentSummary],
    experiments_remaining: int,
) -> AgentState:
    prefix = f"fair_planner_{uuid.uuid4().hex[:8]}_"
    state = initial_state(workload_name, prefix, max_experiments=experiments_remaining)
    state["baseline_summary"] = baseline_summary
    state["best_summary"] = best_summary
    state["experiment_summaries"] = list(summaries)
    state["current_bottleneck"] = bottleneck
    state["experiments_remaining"] = experiments_remaining
    return state


def _pick_config_from_planner(
    state: AgentState,
    llm: ScriptedBottleneckLLM,
    *,
    space,
    tried_keys: set[tuple[Any, ...]],
    base_config: dict[str, Any],
    use_rag: bool,
) -> tuple[dict[str, Any], str, Any] | None:
    """Invoke planner_node and return the first mappable unseen config."""
    if use_rag:
        knowledge_context = planner_module._retrieve_knowledge(
            bottleneck=state["current_bottleneck"],
            workload=state["workload_name"],
        )
        retrieve_ctx = patch(
            "inferops.agent.planner._retrieve_knowledge",
            return_value=knowledge_context,
        )
    else:
        retrieve_ctx = patch(
            "inferops.agent.planner._retrieve_knowledge",
            return_value="(no RAG context — fair compare)",
        )
    with retrieve_ctx:
        patch_out = planner_node(state, llm)

    candidates = [h for h in patch_out.get("hypotheses", []) if h.get("status") == "pending"]

    for hyp in candidates:
        param = hyp["param"]
        value = hyp["value"]
        cfg = _apply_hypothesis(base_config, param, value, space)
        if cfg is None:
            continue
        key = space.config_key(cfg)
        if key in tried_keys:
            continue
        return cfg, param, value
    return None


def run_planner_strategy(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    strategy_name: str,
    use_rag: bool = True,
    gt_optimum: dict[str, Any] | None = None,
    bottleneck: str = "scheduling-bound",
    llm: ScriptedBottleneckLLM | None = None,
) -> StrategyRun:
    """Run one fair-protocol planner trial loop (baseline + planner picks)."""
    space = fixture.search_space
    ledger = TrialLedger(budget=budget)
    metric, direction = WORKLOAD_PRIMARY_METRIC[workload_name]
    model = llm or ScriptedBottleneckLLM(default_bottleneck=bottleneck)

    baseline = _run_baseline(fixture, budget, ledger)
    if baseline is None:
        run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
        return StrategyRun(strategy_name, workload_name, ledger, _score_dict(run_score))

    default_cfg, default_obs = baseline
    baseline_primary = primary_value(default_obs, metric)
    baseline_summary = _observation_to_summary(
        config=default_cfg,
        observation=default_obs,
        workload_name=workload_name,
        experiment_id="fair_baseline",
        baseline_primary=baseline_primary,
    )
    best_cfg, best_obs = default_cfg, default_obs
    if not is_valid_observation(default_obs):
        best_obs = None  # type: ignore[assignment]
    summaries: list[ExperimentSummary] = [baseline_summary]
    current_cfg = default_cfg
    current_obs = default_obs

    while budget.slots_remaining() > 0:
        best_summary = (
            _observation_to_summary(
                config=best_cfg,
                observation=best_obs,
                workload_name=workload_name,
                experiment_id="fair_best",
                baseline_primary=baseline_primary,
            )
            if best_obs is not None
            else None
        )
        state = _build_planner_state(
            workload_name=workload_name,
            bottleneck=current_obs.bottleneck or bottleneck,
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            summaries=summaries,
            experiments_remaining=budget.slots_remaining(),
        )
        state["hypotheses"] = []

        pick = _pick_config_from_planner(
            state,
            model,
            space=space,
            tried_keys=ledger.tried_keys(space),
            base_config=current_cfg,
            use_rag=use_rag,
        )
        if pick is None:
            break

        chosen, param, value = pick
        key = space.config_key(chosen)

        if not budget.charge_trial(duplicate=False):
            break

        try:
            obs = fixture.observe(chosen)
        except KeyError:
            ledger.add(config=chosen, observation=None, paid=True, kind="trial")
            continue

        ledger.add(config=chosen, observation=obs, paid=True, kind="trial")
        trial_summary = _observation_to_summary(
            config=chosen,
            observation=obs,
            workload_name=workload_name,
            experiment_id=f"fair_trial_{key}",
            param_changed=param,
            value_changed=value,
            baseline_primary=baseline_primary,
        )
        summaries.append(trial_summary)
        if is_valid_observation(obs) and is_better(obs, current_obs, metric, direction):
            current_cfg, current_obs = chosen, obs
        if is_valid_observation(obs) and (
            best_obs is None or is_better(obs, best_obs, metric, direction)
        ):
            best_cfg, best_obs = chosen, obs

    run_score = score_run(ledger, workload_name=workload_name, gt_optimum=gt_optimum)
    return StrategyRun(strategy_name, workload_name, ledger, _score_dict(run_score))


def run_planner_rag_strategy(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    gt_optimum: dict[str, Any] | None = None,
    bottleneck: str = "scheduling-bound",
    llm: ScriptedBottleneckLLM | None = None,
) -> StrategyRun:
    return run_planner_strategy(
        fixture,
        budget,
        workload_name=workload_name,
        strategy_name="planner_rag",
        use_rag=True,
        gt_optimum=gt_optimum,
        bottleneck=bottleneck,
        llm=llm,
    )


def run_planner_no_rag_strategy(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    gt_optimum: dict[str, Any] | None = None,
    bottleneck: str = "scheduling-bound",
    llm: ScriptedBottleneckLLM | None = None,
) -> StrategyRun:
    return run_planner_strategy(
        fixture,
        budget,
        workload_name=workload_name,
        strategy_name="planner_no_rag",
        use_rag=False,
        gt_optimum=gt_optimum,
        bottleneck=bottleneck,
        llm=llm,
    )


def run_fair_comparison(
    fixture: HiddenResultFixture,
    budget: BudgetPolicy,
    *,
    workload_name: str,
    seed: int = 42,
    gt_optimum: dict[str, Any] | None = None,
    planner_bottleneck: str = "scheduling-bound",
) -> dict[str, StrategyRun]:
    """Run default, random, online_local_search, planner_rag, planner_no_rag.

    Each strategy receives a fresh ``BudgetPolicy`` with the same ``total_slots``.
    Not wired into ``run_mock_eval`` — use this entry for honest Agent? comparisons
    without changing the CI mock baseline.
    """
    from inferops.eval.strategies import run_all_strategies

    total = budget.total_slots
    runs = run_all_strategies(
        fixture,
        budget,
        workload_name=workload_name,
        seed=seed,
        gt_optimum=gt_optimum,
    )
    runs["planner_rag"] = run_planner_rag_strategy(
        fixture,
        BudgetPolicy(total_slots=total),
        workload_name=workload_name,
        gt_optimum=gt_optimum,
        bottleneck=planner_bottleneck,
    )
    runs["planner_no_rag"] = run_planner_no_rag_strategy(
        fixture,
        BudgetPolicy(total_slots=total),
        workload_name=workload_name,
        gt_optimum=gt_optimum,
        bottleneck=planner_bottleneck,
    )
    return runs
