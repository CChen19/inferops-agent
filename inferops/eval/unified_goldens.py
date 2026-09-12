"""Week-3 ⑦ closeout: unified golden manifest + fail-closed CI gate.

Spans existing measurement-trust goldens, recovery goldens, and
error-memory goldens. Each case records source, expected behavior,
judge rule, reviewer, and a holdout flag (unused for prompt tuning).

Does not redefine ④ ledger, ⑤ confirmation, ⑥ Reflect, Tune ⑧
recovery fields, or Week-1 ``is_promotable``. Empty / skipped sets FAIL.
GPU-not-run is never a pass. Fake/offline LLM is never labeled live.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferops.agent.recovery import RECOVERY_FIELDS
from inferops.eval.error_memory_goldens import (
    REQUIRED_GOLDEN_IDS as ERROR_MEMORY_IDS,
)
from inferops.eval.error_memory_goldens import error_memory_golden_gate
from inferops.eval.judge import FEW_SHOT_EXAMPLES, RUBRIC
from inferops.eval.measurement_goldens import (
    REQUIRED_GOLDEN_IDS as MEASUREMENT_IDS,
)
from inferops.eval.measurement_goldens import measurement_trust_gate
from inferops.eval.real_llm_goldens import validate_real_llm_campaign
from inferops.eval.recovery_goldens import REQUIRED_GOLDEN_IDS as RECOVERY_IDS
from inferops.eval.recovery_goldens import recovery_golden_gate

MANIFEST_SCHEMA = "inferops.unified_goldens.v1"
DEFAULT_MANIFEST = Path("tests/fixtures/unified_goldens/manifest.json")
GPU_QUEUE_ENV = "INFEROPS_GPU_GOLDENS"
MIN_CASES = 20
MAX_CASES = 30

REQUIRED_CASE_FIELDS: tuple[str, ...] = (
    "id",
    "source",
    "expected_behavior",
    "judge_rule",
    "reviewer",
    "holdout",
)

ALLOWED_SOURCES = frozenset(
    {"measurement_goldens", "recovery_goldens", "error_memory_goldens"}
)

# Tune ⑧ has not frozen incomplete / receipt-lost fields. Claiming them
# here would invent schema.
UNFROZEN_TUNE8_FIELDS = (
    "receipt_lost",
    "receipt_lost_at",
    "incomplete_receipt",
    "receipt_incomplete",
)

SOURCE_REQUIRED_IDS = {
    "measurement_goldens": MEASUREMENT_IDS,
    "recovery_goldens": RECOVERY_IDS,
    "error_memory_goldens": ERROR_MEMORY_IDS,
}


@dataclass
class LayerStatus:
    name: str
    status: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class UnifiedGateResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    case_ids: list[str]
    holdout_ids: list[str]
    gpu: LayerStatus
    real_llm: LayerStatus
    offline_fixture: LayerStatus

    def report(self) -> str:
        lines = [
            "### Unified ⑦ golden gate",
            "",
            f"- **passed**: `{self.passed}` (deterministic offline invariants only)",
            f"- **cases**: {len(self.case_ids)}",
            f"- **holdout**: {len(self.holdout_ids)} (unused for prompt tuning)",
            f"- **offline_fixture**: `{self.offline_fixture.status}` "
            f"passed=`{self.offline_fixture.passed}`",
            f"- **real_llm**: `{self.real_llm.status}` "
            f"— {self.real_llm.detail}",
            f"- **gpu**: `{self.gpu.status}` — {self.gpu.detail}",
            "",
        ]
        if self.failures:
            lines.append("Gate failures:")
            for fail in self.failures:
                lines.append(f"- {fail}")
            lines.append("")
        if self.warnings:
            for warn in self.warnings:
                lines.append(f"- warning: {warn}")
            lines.append("")
        return "\n".join(lines)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unified manifest schema {data.get('schema')!r} is not {MANIFEST_SCHEMA!r}"
        )
    return data


def gpu_layer_status() -> LayerStatus:
    """Never mark GPU as pass when no worker / no queued goldens."""
    queued = os.environ.get(GPU_QUEUE_ENV) == "1"
    has_nvidia = shutil.which("nvidia-smi") is not None
    if queued and has_nvidia:
        return LayerStatus(
            name="gpu",
            status="queued",
            passed=False,
            detail="INFEROPS_GPU_GOLDENS=1 and nvidia-smi present; archive real logs only",
        )
    reason = "no 3060 / no GPU worker in this environment"
    if queued and not has_nvidia:
        reason = "INFEROPS_GPU_GOLDENS=1 but nvidia-smi missing — still Blocked"
    return LayerStatus(
        name="gpu",
        status="blocked",
        passed=False,
        detail=f"GPU: 未执行 / Blocked — {reason}. GPU-not-run ≠ pass.",
    )


def real_llm_layer_status(campaign: dict[str, Any] | None = None) -> LayerStatus:
    """Separate from offline/fake-LLM. A claimed pass is fail-closed.

    ``llm_boundary`` must be exactly ``live`` to pass. Missing / unknown /
    fake / injected boundaries cannot pass. Blocked shape must stay
    ``passed=False``, ``pass_rate=None``, ``n_completed=0``.
    """
    if campaign is None:
        return LayerStatus(
            name="real_llm",
            status="separate",
            passed=False,
            detail=(
                "real-LLM evidence is produced by scripts/run_real_llm_goldens.py; "
                "fake/offline must not be labeled live"
            ),
        )
    issues = validate_real_llm_campaign(campaign)
    status = str(campaign.get("status") or "unknown")
    claiming_pass = campaign.get("passed") is True
    if issues:
        kind = "invalid_blocked" if status == "blocked" else "mislabeled"
        return LayerStatus(
            name="real_llm",
            status=kind,
            passed=False,
            detail="; ".join(issues),
        )
    if status == "blocked":
        return LayerStatus(
            name="real_llm",
            status="blocked",
            passed=False,
            detail=str(campaign.get("blocker") or "blocked — not a pass"),
        )
    return LayerStatus(
        name="real_llm",
        status=status,
        passed=claiming_pass,
        detail=str(campaign.get("summary") or status),
    )


def _holdout_leaked_into_prompt_tuning(holdout_ids: list[str]) -> list[str]:
    blob = json.dumps({"few_shot": FEW_SHOT_EXAMPLES, "rubric": RUBRIC})
    leaked = [gid for gid in holdout_ids if gid and gid in blob]
    tuning_notes = Path("data/corpus/tuning_notes.md")
    if tuning_notes.is_file():
        text = tuning_notes.read_text(encoding="utf-8")
        leaked.extend(gid for gid in holdout_ids if gid and gid in text and gid not in leaked)
    return leaked


def _invented_tune8_fields(manifest: dict[str, Any]) -> list[str]:
    frozen = set(RECOVERY_FIELDS)
    invented_in_tune = [name for name in UNFROZEN_TUNE8_FIELDS if name in frozen]
    hits: list[str] = [f"tune_field:{n}" for n in invented_in_tune]
    for case in manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        claimed = case.get("tune8_fields") or []
        for name in claimed:
            if name in UNFROZEN_TUNE8_FIELDS:
                hits.append(f"{case.get('id')}:{name}")
    return hits


def _validate_cases(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    cases = manifest.get("cases") or []
    if not cases:
        failures.append("empty unified golden set is not a pass")
        return failures, warnings, []
    if any(case.get("skip") or case.get("skipped") for case in cases):
        failures.append("skipped golden is not a pass")
    n = len(cases)
    if n < MIN_CASES or n > MAX_CASES:
        failures.append(f"unified manifest has {n} cases; need {MIN_CASES}–{MAX_CASES}")

    ids: list[str] = []
    seen: set[str] = set()
    by_source: dict[str, set[str]] = {src: set() for src in ALLOWED_SOURCES}
    for case in cases:
        if not isinstance(case, dict):
            failures.append("manifest case is not an object")
            continue
        missing = [key for key in REQUIRED_CASE_FIELDS if key not in case]
        if missing:
            failures.append(f"{case.get('id')}: missing fields {missing}")
        gid = str(case.get("id") or "")
        if not gid:
            failures.append("manifest case missing id")
            continue
        if gid in seen:
            failures.append(f"duplicate unified case id {gid!r}")
        seen.add(gid)
        ids.append(gid)
        source = str(case.get("source") or "")
        if source not in ALLOWED_SOURCES:
            failures.append(f"{gid}: unknown source {source!r}")
        else:
            by_source[source].add(gid)
        if case.get("holdout") not in (True, False):
            failures.append(f"{gid}: holdout must be a boolean")
        if not str(case.get("expected_behavior") or "").strip():
            failures.append(f"{gid}: expected_behavior is empty")
        if not str(case.get("judge_rule") or "").strip():
            failures.append(f"{gid}: judge_rule is empty")
        if not str(case.get("reviewer") or "").strip():
            failures.append(f"{gid}: reviewer is empty")
        if case.get("layer") == "real_llm":
            failures.append(f"{gid}: offline manifest must not claim real_llm layer")
        if case.get("gpu_sampled") is True:
            failures.append(f"{gid}: GPU-sampled case in CPU unified manifest")

    for source, required in SOURCE_REQUIRED_IDS.items():
        missing_src = [gid for gid in required if gid not in by_source.get(source, set())]
        if missing_src:
            failures.append(f"{source} ids missing from unified manifest: {missing_src}")

    holdout_ids = [
        str(case["id"])
        for case in cases
        if isinstance(case, dict) and case.get("holdout") is True
    ]
    leaked = _holdout_leaked_into_prompt_tuning(holdout_ids)
    if leaked:
        failures.append(f"holdout cases leaked into prompt-tuning corpus: {leaked}")
    invented = _invented_tune8_fields(manifest)
    if invented:
        failures.append(
            "Tune ⑧ incomplete/receipt-lost fields are not frozen; "
            f"do not invent schema: {invented}"
        )
    follow = manifest.get("follow_ups") or []
    if not any(
        isinstance(item, dict) and item.get("id") == "tune_8_incomplete_receipt_lost"
        for item in follow
    ):
        warnings.append(
            "note follow-up: Tune ⑧ incomplete/receipt-lost fields are not frozen"
        )
    return failures, warnings, ids


def unified_golden_gate(
    manifest_path: str | Path | None = None,
    *,
    real_llm_campaign: dict[str, Any] | None = None,
) -> UnifiedGateResult:
    """Fail-closed deterministic gate over the unified ⑦ manifest."""
    failures: list[str] = []
    warnings: list[str] = []
    gpu = gpu_layer_status()
    real_llm = real_llm_layer_status(real_llm_campaign)
    if gpu.status in {"pass", "passed", "ok"} or gpu.passed:
        failures.append("GPU-not-run ≠ pass")
    if real_llm.status in {"mislabeled", "invalid_blocked"}:
        failures.append(real_llm.detail)
    if real_llm_campaign and real_llm_campaign.get("passed") and not real_llm.passed:
        failures.append("claimed real-LLM pass rejected (fail-closed)")

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"unified manifest unreadable: {exc}")
        return UnifiedGateResult(
            passed=False,
            failures=failures,
            warnings=warnings,
            case_ids=[],
            holdout_ids=[],
            gpu=gpu,
            real_llm=real_llm,
            offline_fixture=LayerStatus(
                name="offline_fixture",
                status="failed",
                passed=False,
                detail="manifest missing",
            ),
        )

    case_failures, case_warnings, case_ids = _validate_cases(manifest)
    failures.extend(case_failures)
    warnings.extend(case_warnings)
    holdout_ids = [
        str(case["id"])
        for case in (manifest.get("cases") or [])
        if isinstance(case, dict) and case.get("holdout") is True
    ]

    measurement = measurement_trust_gate()
    recovery = recovery_golden_gate()
    error_memory = error_memory_golden_gate()
    for name, gate in (
        ("measurement", measurement),
        ("recovery", recovery),
        ("error_memory", error_memory),
    ):
        if not gate.passed:
            failures.append(f"{name} golden gate failed")
            failures.extend(gate.failures)
        if getattr(gate, "gpu_status", "not_run") == "not_run" and not gate.cases:
            failures.append(f"{name}: empty/skipped set FAIL")

    offline_ok = (
        measurement.passed
        and recovery.passed
        and error_memory.passed
        and not case_failures
    )
    offline = LayerStatus(
        name="offline_fixture",
        status="passed" if offline_ok else "failed",
        passed=offline_ok,
        detail=(
            f"measurement={measurement.passed} recovery={recovery.passed} "
            f"error_memory={error_memory.passed}"
        ),
    )
    return UnifiedGateResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        case_ids=case_ids,
        holdout_ids=holdout_ids,
        gpu=gpu,
        real_llm=real_llm,
        offline_fixture=offline,
    )
