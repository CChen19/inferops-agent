"""Separate real-LLM multi-run evidence for ⑦ closeout.

Never labels fake / offline / injected LLM as live. Missing credentials
are an explicit blocker — not a pass, not a silent skip, and not a
fake-LLM substitute.

N ≥ 3 when credentials exist. GPU is out of scope here.
"""

from __future__ import annotations

import json
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
CRED_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
DEFAULT_REPORT = Path("reports/week3_real_llm.md")
DEFAULT_JSON = Path("eval_reports/real_llm/week3_real_llm_campaign.json")


@dataclass
class RealLlmRun:
    run_index: int
    status: str  # passed | failed | blocked
    llm_boundary: str | None = None
    stop_reason: str | None = None
    mode: str | None = None
    error: str | None = None

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
            f"- **layer**: `{self.layer}`",
            f"- **status**: `{self.status}`",
            f"- **passed**: `{self.passed}`",
            f"- **backend**: `{self.backend}`",
            f"- **n_requested**: `{self.n_requested}`",
            f"- **n_completed**: `{self.n_completed}`",
            f"- **pass_rate**: `{self.pass_rate}`",
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
                "| run | status | llm_boundary | stop_reason | error |",
                "|---:|---|---|---|---|",
            ]
        )
        if not self.runs:
            lines.append("| — | blocked | — | — | no live run |")
        for run in self.runs:
            lines.append(
                f"| {run.run_index} | `{run.status}` | `{run.llm_boundary}` | "
                f"`{run.stop_reason or '—'}` | {run.error or '—'} |"
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
    passed = 0
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
            if boundary != "live":
                runs.append(
                    RealLlmRun(
                        run_index=i,
                        status="failed",
                        llm_boundary=boundary,
                        mode=str(report.get("mode")),
                        error=f"llm_boundary={boundary!r} is not live",
                    )
                )
                continue
            rows = (report.get("strategies") or {}).get("real_planner") or []
            stop = rows[0].get("stop_reason") if rows else None
            runs.append(
                RealLlmRun(
                    run_index=i,
                    status="passed",
                    llm_boundary=boundary,
                    stop_reason=stop,
                    mode=str(report.get("mode")),
                )
            )
            passed += 1
        except Exception as exc:  # noqa: BLE001 — per-run evidence
            runs.append(
                RealLlmRun(
                    run_index=i,
                    status="failed",
                    llm_boundary=boundary,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    rate = passed / n if n else None
    return RealLlmCampaign(
        layer=LAYER,
        status="ran",
        passed=passed == n,
        blocker=None,
        n_requested=n,
        n_completed=len(runs),
        pass_rate=rate,
        llm_boundary=boundary,
        backend=backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
        runs=runs,
        summary=(
            f"Live LLM campaign completed {passed}/{n} "
            f"(pass_rate={rate}). llm_boundary={boundary}."
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
