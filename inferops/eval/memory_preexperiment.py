"""Offline memory pre-experiment (CPU step 1) — groups A / B / C.

Question: can cross-session history reduce repeated wasted executions? If yes,
is exact failed-config filtering already enough vs planner history hints?

This module uses a *scripted* proposal order and HiddenResultFixture.observe.
It does **not** attribute lookup-table gains to a paid LLM. Mechanism proof
only. Offline GT replay; no GPU; no live OpenRouter.

Groups (independent temp SQLite copies per run):
  A — no cross-session memory (current-task history only)
  B — deterministic exact failed-config filter (lasting config failures only)
  C — existing ``query_compatible_history`` + ``is_history_failure`` hints

Compatibility for B/C environment matching: model_name + workload_name.
Hardware SKU is not matched by ``query_compatible_history`` on this SHA.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from inferops.eval.protocol import (
    BudgetPolicy,
    HiddenResultFixture,
    Observation,
    TrialLedger,
    is_valid_observation,
    primary_value,
)
from inferops.eval.runner import load_ground_truth
from inferops.memory.db import init_db, save_result
from inferops.memory.history import is_history_failure, query_compatible_history
from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    HardwareInfo,
    InferenceEngine,
    LatencyPercentiles,
    ModelSize,
    SchedulerPolicy,
    WorkloadSpec,
    config_knobs,
)

DISCLAIMER = (
    "scripted LLM / fixed proposal order; offline GT replay via "
    "HiddenResultFixture.observe; not a live GPU result; not proof a paid "
    "planner uses memory; do not attribute lookup-table gains to the LLM"
)

GroupName = Literal["A", "B", "C"]
ScenarioName = Literal["reusable", "irrelevant", "transient", "no_history"]

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OTHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WORKLOAD = "chat_short"
CURRENT_SESSION = "curr_"
PRIOR_SESSION = "prior_"

# Explicit business goal: named GT best config (primary >= best_value).
GOAL_CONFIG = {
    "max_num_batched_tokens": 4096,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
}
GOAL_PRIMARY_MIN = 17.2

# Known-bad config used in reusable / irrelevant scenarios (OOM if retried).
BAD_CONFIG = {
    "max_num_batched_tokens": 2048,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": False,
}

# Mid config that is valid but does not meet GOAL_PRIMARY_MIN.
MID_CONFIG = {
    "max_num_batched_tokens": 2048,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
}

# Fixed scripted proposal orders (mechanism only — not an LLM).
# reusable / irrelevant: hit the known-bad config before the GT best.
# transient: propose the goal first so a wrongful blacklist is visible.
# no_history: same order as reusable (empty memory → A=B=C).
SCRIPTED_ORDERS: dict[ScenarioName, list[dict[str, Any]]] = {
    "reusable": [BAD_CONFIG, GOAL_CONFIG],
    "irrelevant": [BAD_CONFIG, GOAL_CONFIG],
    "transient": [GOAL_CONFIG, MID_CONFIG, BAD_CONFIG],
    "no_history": [BAD_CONFIG, GOAL_CONFIG],
}

_CONFIG_KNOBS = (
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
)
_TRANSIENT_NOTE_MARKERS = (
    "timeout",
    "not ready",
    "spawn error",
    "spawn failed",
    "connection refused",
)
_OOM_MARKERS = ("oom", "out of memory", "cuda out of memory")


def is_lasting_config_failure(*, status: str, notes: str) -> bool:
    """True only for lasting *config* failures — not transient timeout/spawn.

    OOM (including during startup) counts as a lasting bad config.
    Generic ``failed`` without OOM/invalid does **not** permanently blacklist.
    """
    s = (status or "").lower()
    n = (notes or "").lower()
    if any(m in n for m in _OOM_MARKERS) or s == "oom":
        return True
    if s == "invalid":
        return True
    if any(m in n for m in _TRANSIENT_NOTE_MARKERS):
        return False
    return False


def configs_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(k) == right.get(k) for k in _CONFIG_KNOBS)


def meets_business_goal(obs: Observation, *, primary_metric: str = "throughput_rps") -> bool:
    """Valid+SLO and primary metric reaches the documented GT threshold."""
    if not is_valid_observation(obs):
        return False
    return primary_value(obs, primary_metric) >= GOAL_PRIMARY_MIN


def _missing_lat() -> LatencyPercentiles:
    return LatencyPercentiles()


def _workload_spec(name: str = WORKLOAD) -> WorkloadSpec:
    return WorkloadSpec(
        name=name,
        prompt_template="",
        num_requests=10,
        concurrency=4,
        input_len=64,
        output_len=64,
    )


def _config_for(
    experiment_id: str,
    *,
    model_name: str,
    workload_name: str,
    knobs: dict[str, Any],
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        model_name=model_name,
        model_size=ModelSize.HALF_B,
        engine=InferenceEngine.VLLM,
        max_num_seqs=128,
        max_num_batched_tokens=int(knobs["max_num_batched_tokens"]),
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_chunked_prefill=bool(knobs["enable_chunked_prefill"]),
        enable_prefix_caching=bool(knobs["enable_prefix_caching"]),
        scheduler_policy=SchedulerPolicy.FCFS,
        workload=_workload_spec(workload_name),
    )


def seed_prior_failure(
    db_path: Path,
    *,
    knobs: dict[str, Any],
    notes: str,
    model_name: str = MODEL,
    workload_name: str = WORKLOAD,
    session_id: str = PRIOR_SESSION,
    experiment_id: str | None = None,
    status: ExperimentValidityStatus = ExperimentValidityStatus.FAILED,
    run_id: str = "priorfail00000000000000000000001",
) -> ExperimentResult:
    """Persist a prior-session failure row (never the current-test answer)."""
    eid = experiment_id or f"{session_id}enable_chunked_prefill_{knobs['enable_chunked_prefill']}"
    cfg = _config_for(eid, model_name=model_name, workload_name=workload_name, knobs=knobs)
    knobs_snap = config_knobs(cfg)
    result = ExperimentResult(
        experiment_id=eid,
        config=cfg,
        total_requests=0,
        successful_requests=0,
        total_time_s=0.0,
        throughput_rps=None,
        tokens_per_second=None,
        error_rate=None,
        ttft=_missing_lat(),
        tpot=_missing_lat(),
        e2e_latency=_missing_lat(),
        run_id=run_id,
        session_id=session_id,
        mlflow_run_id="mlf-memory-pre",
        requested_config=knobs_snap,
        actual_config=None,
        config_evidence=None,
        status=status,
        notes=notes,
        hardware=HardwareInfo(model_name=model_name, engine="vllm", gpu_name=None),
    )
    init_db(db_path)
    save_result(result, db_path=db_path)
    return result


def load_prior_config_failures(
    db_path: Path,
    *,
    model_name: str,
    workload_name: str,
    exclude_session_id: str,
) -> list[dict[str, Any]]:
    """Load prior rows that are lasting config failures under the same environment."""
    init_db(db_path)
    from inferops.memory.db import _connect

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, workload_name, config_json, result_json,
                   status, session_id, run_id
            FROM experiments
            WHERE workload_name = ?
              AND json_extract(config_json, '$.model_name') = ?
              AND IFNULL(session_id, '') != ?
            """,
            (workload_name, model_name, exclude_session_id),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            config = json.loads(row["config_json"] or "{}")
            result = json.loads(row["result_json"] or "{}")
        except json.JSONDecodeError:
            continue
        notes = str(result.get("notes") or "") if isinstance(result, dict) else ""
        status = str(row["status"] or "")
        if not is_lasting_config_failure(status=status, notes=notes):
            continue
        if not isinstance(config, dict) or config.get("model_name") != model_name:
            continue
        out.append(
            {
                "experiment_id": row["experiment_id"],
                "config": {k: config.get(k) for k in _CONFIG_KNOBS},
                "model_name": config.get("model_name"),
                "workload_name": row["workload_name"],
                "status": status,
                "notes": notes,
                "session_id": row["session_id"],
                "run_id": row["run_id"] or "",
            }
        )
    return out


def should_skip_exact_config_failure(
    proposed: dict[str, Any],
    prior_failures: list[dict[str, Any]],
) -> bool:
    """Group B: skip execute iff full config identical to a prior lasting failure."""
    return any(configs_equal(proposed, row["config"]) for row in prior_failures)


def should_skip_history_hint(
    proposed: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> bool:
    """Group C: mirror production ``is_history_failure`` on recovered (param, value)."""
    for row in history_rows:
        if not is_history_failure(row):
            continue
        param = row.get("param")
        if param is None:
            continue
        if proposed.get(param) == row.get("value"):
            return True
    return False


@dataclass
class GroupRunResult:
    group: GroupName
    scenario: ScenarioName
    wasted_executions_before_goal: int | None
    rejected_without_execute: int
    execute_count: int
    execute_failure_count: int
    first_qualified_n: int | None
    feasible_wrongly_filtered: bool
    goal_met: bool
    n_paid: int
    proposal_log: list[dict[str, Any]] = field(default_factory=list)
    overhead: str = "cpu_only"


def _enrich_fixture_for_scenario(
    gt: dict[str, Any],
    scenario: ScenarioName,
) -> HiddenResultFixture:
    """Build observe table. Bad config observes as failed (wasted if executed)."""
    rows = []
    for row in gt.get("experiments", []):
        enriched = {
            **row,
            "validity_status": row.get("validity_status", "valid"),
            "error_rate": row.get("error_rate", 0.01),
            "has_config_evidence": row.get("has_config_evidence", True),
            "bottleneck": row.get("bottleneck", "compute-bound"),
        }
        cfg = {k: enriched[k] for k in _CONFIG_KNOBS}
        if scenario in ("reusable", "irrelevant") and configs_equal(cfg, BAD_CONFIG):
            enriched["validity_status"] = "failed"
            enriched["error_rate"] = 1.0
            enriched["throughput_rps"] = None
            enriched["tokens_per_second"] = None
        rows.append(enriched)
    return HiddenResultFixture.from_rows(rows)


def _seed_db_for_scenario(db_path: Path, scenario: ScenarioName) -> None:
    init_db(db_path)
    if scenario == "no_history":
        return
    if scenario == "reusable":
        seed_prior_failure(
            db_path,
            knobs=BAD_CONFIG,
            notes="vLLM OOM during startup — CUDA out of memory",
            model_name=MODEL,
            workload_name=WORKLOAD,
            experiment_id=f"{PRIOR_SESSION}enable_chunked_prefill_True",
        )
        return
    if scenario == "irrelevant":
        # Same knobs, different model — must be excluded by environment match.
        seed_prior_failure(
            db_path,
            knobs=BAD_CONFIG,
            notes="vLLM OOM during startup — CUDA out of memory",
            model_name=OTHER_MODEL,
            workload_name=WORKLOAD,
            experiment_id=f"{PRIOR_SESSION}alien_enable_chunked_prefill_True",
            run_id="priorfail00000000000000000000002",
        )
        return
    if scenario == "transient":
        # Transient timeout on the *goal* config — must not permanently blacklist.
        seed_prior_failure(
            db_path,
            knobs=GOAL_CONFIG,
            notes="vLLM not ready after startup timeout — spawn stalled",
            model_name=MODEL,
            workload_name=WORKLOAD,
            experiment_id=f"{PRIOR_SESSION}max_num_batched_tokens_4096",
            run_id="priorfail00000000000000000000003",
        )
        return


def run_group(
    group: GroupName,
    scenario: ScenarioName,
    *,
    fixture: HiddenResultFixture,
    db_path: Path,
    budget_slots: int = 4,
    proposal_order: list[dict[str, Any]] | None = None,
) -> GroupRunResult:
    """Run one group with a scripted proposal order over an independent DB copy."""
    order = proposal_order or [dict(c) for c in SCRIPTED_ORDERS[scenario]]
    budget = BudgetPolicy(total_slots=budget_slots)
    ledger = TrialLedger(budget=budget)
    space = fixture.search_space

    prior_exact = (
        load_prior_config_failures(
            db_path,
            model_name=MODEL,
            workload_name=WORKLOAD,
            exclude_session_id=CURRENT_SESSION,
        )
        if group == "B"
        else []
    )
    history_rows = (
        query_compatible_history(
            model_name=MODEL,
            workload_name=WORKLOAD,
            exclude_session_id=CURRENT_SESSION,
            db_path=db_path,
            top_k=32,
        )
        if group == "C"
        else []
    )

    rejected = 0
    execute_count = 0
    execute_failure_count = 0
    first_qualified_n: int | None = None
    wasted_before_goal: int | None = None
    goal_met = False
    feasible_wrongly_filtered = False
    proposal_log: list[dict[str, Any]] = []
    tried = set()

    for proposed in order:
        if not space.is_legal(proposed):
            continue
        key = space.config_key(proposed)
        if key in tried:
            continue

        skip = False
        skip_reason = ""
        if group == "B" and should_skip_exact_config_failure(proposed, prior_exact):
            skip = True
            skip_reason = "exact_config_failure"
        elif group == "C" and should_skip_history_hint(proposed, history_rows):
            skip = True
            skip_reason = "history_failure_hint"

        if skip:
            rejected += 1
            # Detect wrongful filter of a feasible goal config.
            if configs_equal(proposed, GOAL_CONFIG):
                feasible_wrongly_filtered = True
            proposal_log.append(
                {
                    "config": dict(proposed),
                    "action": "reject_without_execute",
                    "reason": skip_reason,
                }
            )
            tried.add(key)
            continue

        if not budget.charge_trial(duplicate=False):
            break
        obs = fixture.observe(proposed)
        execute_count += 1
        paid = True
        ledger.add(config=proposed, observation=obs, paid=paid, kind="trial")
        tried.add(key)
        failed_obs = not is_valid_observation(obs)
        if failed_obs:
            execute_failure_count += 1
        proposal_log.append(
            {
                "config": dict(proposed),
                "action": "execute",
                "validity_status": obs.validity_status,
                "primary": obs.metrics.get("throughput_rps"),
            }
        )
        if meets_business_goal(obs) and first_qualified_n is None:
            first_qualified_n = execute_count
            # Wasted = failed or non-goal executes before the qualifying one.
            wasted = 0
            for entry in proposal_log:
                if entry["action"] != "execute":
                    continue
                if entry is proposal_log[-1]:
                    break
                wasted += 1
            wasted_before_goal = wasted
            goal_met = True
            break

    return GroupRunResult(
        group=group,
        scenario=scenario,
        wasted_executions_before_goal=wasted_before_goal,
        rejected_without_execute=rejected,
        execute_count=execute_count,
        execute_failure_count=execute_failure_count,
        first_qualified_n=first_qualified_n,
        feasible_wrongly_filtered=feasible_wrongly_filtered,
        goal_met=goal_met,
        n_paid=budget.n_paid,
        proposal_log=proposal_log,
        overhead="cpu_only",
    )


def run_scenario(
    scenario: ScenarioName,
    *,
    ground_truth_dir: str | Path = "tests/fixtures/ground_truth",
    budget_slots: int = 4,
    base_tmp: Path | None = None,
) -> dict[str, Any]:
    """Run A/B/C on independent DB copies for one scenario."""
    gt = load_ground_truth(WORKLOAD, ground_truth_dir)
    fixture = _enrich_fixture_for_scenario(gt, scenario)

    root = Path(base_tmp) if base_tmp is not None else Path(tempfile.mkdtemp(prefix="mempre_"))
    root.mkdir(parents=True, exist_ok=True)

    # Seed a template DB, then copy per group so groups never share accumulation.
    template = root / f"{scenario}_template.db"
    _seed_db_for_scenario(template, scenario)

    group_results: dict[str, Any] = {}
    for group in ("A", "B", "C"):
        db_path = root / f"{scenario}_{group}.db"
        shutil.copy2(template, db_path)
        result = run_group(
            group,  # type: ignore[arg-type]
            scenario,
            fixture=fixture,
            db_path=db_path,
            budget_slots=budget_slots,
        )
        group_results[group] = {
            "wasted_executions_before_goal": result.wasted_executions_before_goal,
            "rejected_without_execute": result.rejected_without_execute,
            "execute_count": result.execute_count,
            "execute_failure_count": result.execute_failure_count,
            "first_qualified_n": result.first_qualified_n,
            "feasible_wrongly_filtered": result.feasible_wrongly_filtered,
            "goal_met": result.goal_met,
            "n_paid": result.n_paid,
            "overhead": result.overhead,
            "proposal_log": result.proposal_log,
        }

    return {
        "scenario": scenario,
        "business_goal": {
            "definition": (
                "first observe that is valid+SLO and throughput_rps >= "
                f"{GOAL_PRIMARY_MIN} (GT best config "
                "grid_chat_short_t4096_c0_p0)"
            ),
            "goal_config": dict(GOAL_CONFIG),
            "primary_metric": "throughput_rps",
            "primary_min": GOAL_PRIMARY_MIN,
        },
        "compatibility": "model_name + workload_name (no GPU SKU match on this SHA)",
        "groups": group_results,
    }


def run_memory_preexperiment(
    commit_sha: str,
    *,
    ground_truth_dir: str | Path = "tests/fixtures/ground_truth",
    budget_slots: int = 4,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all four scenarios and optionally write JSON/Markdown reports."""
    scenarios: list[ScenarioName] = ["reusable", "irrelevant", "transient", "no_history"]
    with tempfile.TemporaryDirectory(prefix="memory_preexperiment_") as tmp:
        tmp_path = Path(tmp)
        scenario_rows = [
            run_scenario(
                name,
                ground_truth_dir=ground_truth_dir,
                budget_slots=budget_slots,
                base_tmp=tmp_path / name,
            )
            for name in scenarios
        ]

    report = {
        "commit_sha": commit_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "memory_preexperiment",
        "mode_label": "offline_scripted_memory_abc",
        "llm_boundary": "scripted_fixed_proposal_order",
        "tool_boundary": "hidden_fixture_observe",
        "budget_slots": budget_slots,
        "disclaimer": DISCLAIMER,
        "question": (
            "Can cross-session history reduce repeated wasted executions? "
            "If yes, is exact failed-config filtering already enough vs "
            "planner history hints?"
        ),
        "groups": {
            "A": "no cross-session memory",
            "B": "exact lasting config-failure filter (not transient)",
            "C": "query_compatible_history + is_history_failure hints",
        },
        "scenarios": scenario_rows,
    }
    if output_dir is not None:
        write_memory_preexperiment_outputs(report, output_dir)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Memory Pre-Experiment Report: `{report['commit_sha']}`",
        "",
        f"- Mode: `{report.get('mode', '')}`",
        f"- Mode label: `{report.get('mode_label', '')}`",
        f"- LLM boundary: `{report.get('llm_boundary', '')}`",
        f"- Tool boundary: `{report.get('tool_boundary', '')}`",
        f"- Budget slots: `{report.get('budget_slots', '')}`",
        "",
        f"> **Disclaimer:** {report.get('disclaimer', DISCLAIMER)}",
        "",
        f"**Question:** {report.get('question', '')}",
        "",
        "## Groups",
        "",
    ]
    for name, desc in (report.get("groups") or {}).items():
        lines.append(f"- **{name}:** {desc}")
    lines += ["", "## Scenarios", ""]

    for sc in report.get("scenarios", []):
        lines += [
            f"### {sc['scenario']}",
            "",
            f"- Business goal: {sc['business_goal']['definition']}",
            f"- Compatibility: {sc['compatibility']}",
            "",
            "| Group | Goal met | Wasted before goal | Rejected w/o exec | "
            "Executes | Exec failures | 1st qualified N | Wrongly filtered | Paid |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for gname, grow in sc["groups"].items():
            wasted = grow.get("wasted_executions_before_goal")
            wasted_s = "—" if wasted is None else str(wasted)
            first = grow.get("first_qualified_n")
            first_s = "—" if first is None else str(first)
            lines.append(
                f"| {gname} | {'yes' if grow.get('goal_met') else 'no'} | {wasted_s} | "
                f"{grow.get('rejected_without_execute', 0)} | {grow.get('execute_count', 0)} | "
                f"{grow.get('execute_failure_count', 0)} | {first_s} | "
                f"{'yes' if grow.get('feasible_wrongly_filtered') else 'no'} | "
                f"{grow.get('n_paid', 0)} |"
            )
        lines.append("")

    lines += [
        "## Reading the table",
        "",
        "- Primary metric is **wasted executions before the business goal**, "
        "not \"a valid baseline exists\".",
        "- Group B should help on **reusable** lasting OOM failures and must "
        "not blacklist on **transient** timeout/spawn notes.",
        "- Group C uses existing history-failure hints (broader than B); "
        "wrongful filters of a feasible config are recorded.",
        "- **no_history** / **irrelevant** should not make B/C worse than A.",
        "",
    ]
    return "\n".join(lines)


def write_memory_preexperiment_outputs(
    report: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = report["commit_sha"]
    json_path = out / f"{sha}.json"
    md_path = out / f"{sha}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown_report(report))
    return md_path, json_path
