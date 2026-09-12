"""Separate real-LLM multi-run evidence for ⑦ closeout.

Never labels fake / offline / injected LLM as live. Missing credentials
are an explicit blocker — not a pass, not a silent skip, and not a
fake-LLM substitute.

N ≥ 3 when credentials exist. GPU is out of scope here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferops.eval.real_graph import (
    MODE_REAL_GRAPH_LLM,
    MODE_REAL_GRAPH_OFFLINE,
    ScriptedBottleneckLLM,
    _llm_boundary_label,
    require_llm_credentials,
    run_real_graph_eval,
)

LAYER = "real_llm"
N_RUNS = 3
DEFAULT_BACKEND = "openrouter"

# First-class terminal stops from ⑥ Reflect / ③ budget. Empty, unknown, or
# harness-only ``eval_empty_plan`` is a wrong-stop — not an accepted live run.
ACCEPTED_LIVE_STOP_REASONS = frozenset(
    {
        "budget_exhausted",
        "no_reliable_improvement",
    }
)
ILLEGAL_BENCHMARK_KEYS = frozenset({"tensor_parallel_size"})
CRED_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
DEFAULT_REPORT = Path("reports/week3_real_llm.md")
DEFAULT_JSON = Path("eval_reports/real_llm/week3_real_llm_campaign.json")


@dataclass
class LiveRunJudgement:
    """Fail-closed per-run acceptance. ``accepted`` is never 'call didn't throw'."""

    accepted: bool
    failures: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    n_experiments: int = 0
    trajectory_score: float | None = None
    composite: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealLlmRun:
    run_index: int
    status: str  # passed | failed | blocked
    llm_boundary: str | None = None
    stop_reason: str | None = None
    mode: str | None = None
    error: str | None = None
    accepted: bool = False
    acceptance_failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealLlmCampaign:
    layer: str
    status: str  # blocked | ran
    passed: bool
    blocker: str | None
    n_requested: int
    n_completed: int
    n_accepted: int
    pass_rate: float | None
    llm_boundary: str | None
    backend: str
    generated_at: str
    runs: list[RealLlmRun] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runs"] = [run.as_dict() for run in self.runs]
        return payload

    def report_markdown(self) -> str:
        lines = [
            "# Week-3 ⑦ real-LLM multi-run evidence",
            "",
            "This report is **separate** from offline / fake-LLM eval.",
            "Fake-scripted or `--real-graph` output must not be labeled live.",
            "",
            "`pass_rate` is **accepted / n_requested** from a fail-closed",
            "per-run judge (`judge_live_run`): live boundary, ordered",
            "planner→executor→reflector, first-class stop",
            "(`budget_exhausted` / `no_reliable_improvement`), and non-zero",
            "quality. Empty rows, missing scores, or `eval_empty_plan` do",
            "not count. A call that merely did not throw is not a pass.",
            "",
            f"- **layer**: `{self.layer}`",
            f"- **status**: `{self.status}`",
            f"- **passed**: `{self.passed}`",
            f"- **backend**: `{self.backend}`",
            f"- **n_requested**: `{self.n_requested}`",
            f"- **n_completed**: `{self.n_completed}`",
            f"- **n_accepted**: `{self.n_accepted}`",
            f"- **pass_rate**: `{self.pass_rate}` "
            "(accepted / n_requested; never 'call didn't throw')",
            f"- **llm_boundary**: `{self.llm_boundary}`",
            f"- **generated_at**: `{self.generated_at}`",
            "",
        ]
        if self.blocker:
            lines.extend(
                [
                    "## Blocker",
                    "",
                    self.blocker,
                    "",
                    "This is **not** a real-LLM pass. Offline/fake-LLM evidence",
                    "lives under `eval_reports/real_graph/` and the unified",
                    "offline golden gate.",
                    "",
                ]
            )
        lines.extend(
            [
                "## Per-run outcomes",
                "",
                "| run | status | accepted | llm_boundary | stop_reason | error |",
                "|---:|---|---|---|---|---|",
            ]
        )
        if not self.runs:
            lines.append("| — | blocked | no | — | — | no live run |")
        for run in self.runs:
            lines.append(
                f"| {run.run_index} | `{run.status}` | `{run.accepted}` | "
                f"`{run.llm_boundary}` | `{run.stop_reason or '—'}` | "
                f"{run.error or '—'} |"
            )
        lines.extend(["", f"## Summary", "", self.summary, ""])
        return "\n".join(lines)


def credential_env_name(backend: str = DEFAULT_BACKEND) -> str:
    return CRED_ENV.get(backend, "OPENROUTER_API_KEY")


def missing_credential(backend: str = DEFAULT_BACKEND) -> str | None:
    env_key = credential_env_name(backend)
    if os.environ.get(env_key):
        return None
    return (
        f"real-LLM campaign requires {env_key} for backend={backend!r}. "
        "No live key in this environment. Refusing to substitute fake/offline LLM."
    )


def _ordered_plan_execute_reflect(nodes: list[Any]) -> bool:
    try:
        i_p = list(nodes).index("planner")
        i_e = list(nodes).index("executor")
        i_r = list(nodes).index("reflector")
    except ValueError:
        return False
    return i_p < i_e < i_r


def judge_live_run(report: dict[str, Any] | None) -> LiveRunJudgement:
    """Fail-closed acceptance for one live real-LLM eval report.

    A run is accepted only when the trusted live boundary holds **and** the
    ③/⑥ trajectory is non-empty, ordered Plan→Execute→Reflect, has a
    first-class stop, and is not zero-quality. Empty rows, missing scores,
    ``eval_empty_plan``, or 'the call did not throw' do not count.
    """
    failures: list[str] = []
    if not report:
        return LiveRunJudgement(accepted=False, failures=["missing live eval report"])
    if report.get("llm_boundary") != "live":
        failures.append(
            f"llm_boundary={report.get('llm_boundary')!r} is not live"
        )
    if report.get("mode") != MODE_REAL_GRAPH_LLM:
        failures.append(f"mode={report.get('mode')!r} is not {MODE_REAL_GRAPH_LLM}")
    rows = (report.get("strategies") or {}).get("real_planner") or []
    if not rows:
        failures.append("empty real_planner rows — not a pass")
        return LiveRunJudgement(accepted=False, failures=failures)

    row = rows[0] if isinstance(rows[0], dict) else {}
    stop = str(row.get("stop_reason") or "")
    n_exp = int(row.get("n_experiments") or 0)
    try:
        traj_score = float(row["trajectory_score"]) if row.get("trajectory_score") is not None else None
    except (TypeError, ValueError):
        traj_score = None
        failures.append("trajectory_score is not numeric")
    try:
        composite = float(row["composite"]) if row.get("composite") is not None else None
    except (TypeError, ValueError):
        composite = None
        failures.append("composite is not numeric")

    if not stop:
        failures.append("missing stop_reason — not a pass")
    elif stop not in ACCEPTED_LIVE_STOP_REASONS:
        failures.append(
            f"wrong-stop {stop!r} (accepted: {sorted(ACCEPTED_LIVE_STOP_REASONS)})"
        )
    nodes = list(row.get("trajectory_nodes") or [])
    if not _ordered_plan_execute_reflect(nodes):
        failures.append("trajectory missing ordered planner→executor→reflector")
    if n_exp < 1:
        failures.append("zero experiments — zero-quality")
    if traj_score is None or traj_score <= 0:
        failures.append("zero/missing trajectory_score — zero-quality")
    if composite is None:
        failures.append("missing composite")
    if not (row.get("hypotheses") or []):
        failures.append("empty hypotheses — zero-quality")
    for call in row.get("benchmark_calls") or []:
        patch = (call or {}).get("config_patch") or {}
        illegal = set(patch) & ILLEGAL_BENCHMARK_KEYS
        if illegal:
            failures.append(f"illegal param reached benchmark: {sorted(illegal)}")

    return LiveRunJudgement(
        accepted=not failures,
        failures=failures,
        stop_reason=stop or None,
        n_experiments=n_exp,
        trajectory_score=traj_score,
        composite=composite,
    )


def _strict_nonneg_int(value: Any) -> int | None:
    """Accept only a real non-negative ``int``.

    ``int(3.9) == 3`` and ``int(True) == 1`` must not count. Bools, floats,
    strings, and negatives are rejected.
    """
    if type(value) is not int:
        return None
    if value < 0:
        return None
    return value


def _finite_number(value: Any) -> float | None:
    """Accept a finite real number. Reject bool, NaN, and ±Inf."""
    if type(value) not in (int, float):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _accepted_runs(runs: list[Any]) -> tuple[int, list[str]]:
    """Count runs that are accepted on both ``accepted`` and ``status``.

    Either field missing, non-bool ``accepted``, or a contradiction
    (``accepted=False`` + ``status=passed``, or the reverse) is a problem.
    """
    failures: list[str] = []
    consistent = 0
    for i, row in enumerate(runs):
        if not isinstance(row, dict):
            failures.append(f"run[{i}] is not an object")
            continue
        if "accepted" not in row:
            failures.append(f"run[{i}] missing accepted")
            continue
        flag = row.get("accepted")
        if flag not in (True, False):
            failures.append(f"run[{i}] accepted must be bool, got {flag!r}")
            continue
        run_status = row.get("status")
        if flag is True and run_status != "passed":
            failures.append(
                f"run[{i}] accepted=True inconsistent with status={run_status!r}"
            )
            continue
        if flag is False and run_status == "passed":
            failures.append(
                f"run[{i}] accepted=False inconsistent with status='passed'"
            )
            continue
        if flag is True and run_status == "passed":
            consistent += 1
    return consistent, failures


def validate_real_llm_campaign(campaign: dict[str, Any] | None) -> list[str]:
    """Fail-closed shape + pass-claim checks for external real-LLM JSON."""
    if campaign is None:
        return []
    failures: list[str] = []
    if campaign.get("layer") != LAYER:
        failures.append("real-LLM campaign layer must be 'real_llm'")
    status = campaign.get("status")
    if status not in {"blocked", "ran"}:
        failures.append(f"real-LLM campaign status {status!r} is not blocked|ran")
        return failures

    n_req = campaign.get("n_requested")
    n_comp = campaign.get("n_completed")
    n_acc = campaign.get("n_accepted")
    runs = campaign.get("runs")
    rate = campaign.get("pass_rate")
    passed = campaign.get("passed")
    boundary = campaign.get("llm_boundary")

    if status == "blocked":
        if passed is not False:
            failures.append("blocked real-LLM campaign must have passed=False")
        if "pass_rate" not in campaign or rate is not None:
            failures.append("blocked real-LLM campaign must have pass_rate=None")
        if "n_requested" in campaign and _strict_nonneg_int(n_req) is None:
            failures.append("blocked real-LLM campaign n_requested must be a non-negative int")
        if "n_completed" not in campaign:
            failures.append("blocked real-LLM campaign must present n_completed=0")
        elif _strict_nonneg_int(n_comp) != 0:
            failures.append("blocked real-LLM campaign must have n_completed=0")
        if "n_accepted" not in campaign:
            failures.append("blocked real-LLM campaign must present n_accepted=0")
        elif _strict_nonneg_int(n_acc) != 0:
            failures.append("blocked real-LLM campaign must have n_accepted=0")
        if "runs" not in campaign:
            failures.append("blocked real-LLM campaign must present empty runs")
        elif not isinstance(runs, list) or runs:
            failures.append("blocked real-LLM campaign must have empty runs")
        return failures

    if "runs" not in campaign or not isinstance(runs, list):
        failures.append("ran campaign missing runs list")
        runs = []
    n_req_i = _strict_nonneg_int(n_req) if "n_requested" in campaign else None
    if n_req_i is None:
        n_req_i = 0
        failures.append(
            "ran campaign n_requested must be a non-negative int"
            if "n_requested" in campaign
            else "ran campaign missing n_requested"
        )
    n_comp_i = _strict_nonneg_int(n_comp) if "n_completed" in campaign else None
    if "n_completed" not in campaign:
        failures.append("ran campaign missing n_completed")
    elif n_comp_i is None:
        failures.append("ran campaign n_completed must be a non-negative int")
    elif n_comp_i != len(runs):
        failures.append("n_completed inconsistent with runs")

    accepted, accept_failures = _accepted_runs(runs)
    failures.extend(accept_failures)
    n_acc_i = _strict_nonneg_int(n_acc) if "n_accepted" in campaign else None
    if "n_accepted" not in campaign:
        failures.append("ran campaign missing n_accepted")
    elif n_acc_i is None:
        failures.append("ran campaign n_accepted must be a non-negative int")
    elif n_acc_i != accepted:
        failures.append(
            f"n_accepted={n_acc!r} inconsistent with accepted runs ({accepted})"
        )

    expected_rate = (accepted / n_req_i) if n_req_i else None
    if "pass_rate" not in campaign:
        failures.append("ran campaign missing pass_rate")
    elif expected_rate is None:
        if rate is not None:
            failures.append("pass_rate must be None when n_requested is 0")
    elif rate is None:
        failures.append("ran campaign missing pass_rate")
    else:
        parsed_rate = _finite_number(rate)
        if parsed_rate is None:
            failures.append(f"pass_rate {rate!r} is not a finite number")
        elif abs(parsed_rate - expected_rate) > 1e-9:
            failures.append(
                f"pass_rate {rate!r} != accepted/n_requested "
                f"({accepted}/{n_req_i})"
            )
    expected_passed = bool(n_req_i >= N_RUNS and accepted == n_req_i)
    if passed is True and not expected_passed:
        failures.append(
            f"passed={passed!r} inconsistent with accepted={accepted} n={n_req_i}"
        )
    elif passed is not True and expected_passed:
        failures.append(
            f"passed={passed!r} inconsistent with accepted={accepted} n={n_req_i}"
        )
    if passed is True:
        if boundary != "live":
            failures.append(
                f"real-LLM pass requires llm_boundary='live', got {boundary!r}"
            )
        if n_req_i < N_RUNS:
            failures.append("N<3 cannot pass")
        if any(
            not isinstance(row, dict) or row.get("llm_boundary") != "live"
            for row in runs
        ):
            failures.append("every run in a passing campaign must have llm_boundary=live")
        if any(
            not isinstance(row, dict)
            or row.get("status") != "passed"
            or row.get("accepted") is not True
            for row in runs
        ):
            failures.append("every run in a passing campaign must be accepted")
        if n_acc_i is None or n_acc_i != n_req_i:
            failures.append(
                "claimed pass requires n_accepted consistent with n_requested"
            )
    return failures


def refuse_fake_labeled_live(llm: Any, mode: str) -> str | None:
    """Return an error if fake/offline is presented as a live real-LLM run."""
    label = _llm_boundary_label(llm, mode)
    if mode == MODE_REAL_GRAPH_LLM and label != "live":
        return (
            f"real-LLM mode labeled {label!r}, not live. "
            "Fake/offline/injected must not be reported as real LLM."
        )
    if label == "fake_scripted" and mode == MODE_REAL_GRAPH_LLM:
        return "ScriptedBottleneckLLM cannot be labeled as real LLM"
    return None


def blocked_campaign(
    *,
    backend: str = DEFAULT_BACKEND,
    n: int = N_RUNS,
    blocker: str | None = None,
) -> RealLlmCampaign:
    reason = blocker or missing_credential(backend) or "blocked"
    return RealLlmCampaign(
        layer=LAYER,
        status="blocked",
        passed=False,
        blocker=reason,
        n_requested=n,
        n_completed=0,
        n_accepted=0,
        pass_rate=None,
        llm_boundary=None,
        backend=backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
        runs=[],
        summary=(
            f"BLOCKED — not a pass. Requested N={n} live runs; completed 0. "
            "Pass rate is undefined (None), not 100% and not 0/0."
        ),
    )


def run_real_llm_campaign(
    *,
    n: int = N_RUNS,
    backend: str = DEFAULT_BACKEND,
    ground_truth_dir: str | Path = "tests/fixtures/ground_truth",
    commit_sha: str = "w3-7-closeout",
    llm: Any | None = None,
) -> RealLlmCampaign:
    """Run N live-LLM evaluations or return an explicit blocker.

    Passing a scripted/fake ``llm`` under this entry is a failed campaign,
    not a live pass.
    """
    if n < 3:
        return blocked_campaign(
            backend=backend,
            n=n,
            blocker=f"N={n} is below the ⑦ requirement N≥3",
        )
    if llm is not None:
        misuse = refuse_fake_labeled_live(llm, MODE_REAL_GRAPH_LLM)
        if misuse:
            campaign = blocked_campaign(backend=backend, n=n, blocker=misuse)
            campaign.status = "blocked"
            campaign.llm_boundary = _llm_boundary_label(llm, MODE_REAL_GRAPH_LLM)
            return campaign
    blocker = missing_credential(backend)
    if blocker:
        return blocked_campaign(backend=backend, n=n, blocker=blocker)

    require_llm_credentials(backend)
    runs: list[RealLlmRun] = []
    accepted = 0
    boundary: str | None = None
    for i in range(1, n + 1):
        try:
            report = run_real_graph_eval(
                commit_sha=f"{commit_sha}-r{i}",
                ground_truth_dir=ground_truth_dir,
                workloads=["chat_short"],
                budget=2,
                mode=MODE_REAL_GRAPH_LLM,
                llm=llm,
                llm_backend=backend,
            )
            boundary = str(report.get("llm_boundary"))
            verdict = judge_live_run(report)
            if verdict.accepted:
                accepted += 1
                runs.append(
                    RealLlmRun(
                        run_index=i,
                        status="passed",
                        llm_boundary=boundary,
                        stop_reason=verdict.stop_reason,
                        mode=str(report.get("mode")),
                        accepted=True,
                    )
                )
            else:
                runs.append(
                    RealLlmRun(
                        run_index=i,
                        status="failed",
                        llm_boundary=boundary,
                        stop_reason=verdict.stop_reason,
                        mode=str(report.get("mode")),
                        error="; ".join(verdict.failures),
                        accepted=False,
                        acceptance_failures=list(verdict.failures),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — per-run evidence
            runs.append(
                RealLlmRun(
                    run_index=i,
                    status="failed",
                    llm_boundary=boundary,
                    error=f"{type(exc).__name__}: {exc}",
                    accepted=False,
                    acceptance_failures=[f"{type(exc).__name__}: {exc}"],
                )
            )
    rate = accepted / n if n else None
    return RealLlmCampaign(
        layer=LAYER,
        status="ran",
        passed=accepted == n and n >= N_RUNS,
        blocker=None,
        n_requested=n,
        n_completed=len(runs),
        n_accepted=accepted,
        pass_rate=rate,
        llm_boundary=boundary,
        backend=backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
        runs=runs,
        summary=(
            f"Live LLM campaign accepted {accepted}/{n} "
            f"(pass_rate={rate} = accepted/n_requested). "
            f"llm_boundary={boundary}. pass_rate is not 'call didn't throw'."
        ),
    )


def write_campaign_outputs(
    campaign: RealLlmCampaign,
    *,
    markdown_path: str | Path = DEFAULT_REPORT,
    json_path: str | Path = DEFAULT_JSON,
) -> tuple[Path, Path]:
    md = Path(markdown_path)
    js = Path(json_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(campaign.report_markdown(), encoding="utf-8")
    js.write_text(json.dumps(campaign.as_dict(), indent=2) + "\n", encoding="utf-8")
    return md, js


def assert_offline_not_labeled_live(report: dict[str, Any]) -> None:
    """Invariant: --real-graph / fake LLM reports stay off the live label."""
    if report.get("mode") == MODE_REAL_GRAPH_OFFLINE:
        if report.get("llm_boundary") == "live":
            raise AssertionError("offline real-graph labeled llm_boundary=live")
    if report.get("layer") == LAYER and report.get("llm_boundary") == "fake_scripted":
        if report.get("status") == "ran" and report.get("passed"):
            raise AssertionError("fake_scripted campaign marked as a real-LLM pass")


def scripted_llm_is_not_live() -> str:
    return _llm_boundary_label(ScriptedBottleneckLLM(), MODE_REAL_GRAPH_OFFLINE)
