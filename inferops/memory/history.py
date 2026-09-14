"""Compatible prior-session experiment hints for the planner.

Compatibility is model + workload + hardware fingerprint. Rows with a
mismatched or unknown fingerprint stay in SQLite for inspection but are
excluded from planner ranking and failure-memory duplicate suppression.

Returned rows are ``claim_level=prior_session_hint``. They must never be
treated as this-run metric evidence and must not skip confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferops.memory.db import _connect, init_db
from inferops.memory.hardware import (
    HardwareFingerprint,
    fingerprint_from_hardware,
    fingerprints_compatible,
)
from inferops.schemas import ExperimentConfig, HardwareInfo

CLAIM_LEVEL = "prior_session_hint"
_SEARCH_KNOBS = (
    "max_num_batched_tokens",
    "max_num_seqs",
    "enable_chunked_prefill",
    "enable_prefix_caching",
)
_FAILURE_STATUSES = frozenset({"failed", "invalid", "oom"})


def is_history_failure(row: dict[str, Any]) -> bool:
    """True for failed / invalid / OOM prior rows (failure memory)."""
    status = str(row.get("status") or "").lower()
    notes = str(row.get("notes") or "").lower()
    if status in _FAILURE_STATUSES:
        return True
    return "oom" in notes or "out of memory" in notes


def _knob_default(key: str) -> Any:
    return ExperimentConfig.model_fields[key].default


def _recover_param_value(_experiment_id: str, config: dict[str, Any]) -> tuple[str | None, Any]:
    """Recover the single search-knob that differs from ExperimentConfig defaults.

    Exactly one differing search knob → that (param, value).
    Zero or more than one → (None, None). Never guess from experiment_id tokens
    or pick an arbitrary diffs[0], so multi-knob rows cannot suppress an untried pair.
    """
    diffs: list[tuple[str, Any]] = []
    for key in _SEARCH_KNOBS:
        if key not in config:
            continue
        if config[key] != _knob_default(key):
            diffs.append((key, config[key]))
    if len(diffs) == 1:
        return diffs[0]
    return (None, None)


def _hardware_from_result_json(result_json: str) -> HardwareInfo | None:
    try:
        result = json.loads(result_json or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    raw = result.get("hardware")
    if not isinstance(raw, dict):
        return None
    try:
        return HardwareInfo.model_validate(raw)
    except Exception:
        return None


def query_compatible_history(
    *,
    model_name: str,
    workload_name: str,
    exclude_session_id: str,
    db_path: Path | str,
    top_k: int = 8,
    current_fingerprint: HardwareFingerprint | None = None,
) -> list[dict[str, Any]]:
    """Return prior-session rows that match model + workload + hardware.

    Rows from ``exclude_session_id`` (the current session) are omitted.
    Different ``config_json.model_name`` values are incompatible and excluded.
    Mismatch or unknown hardware fingerprints are excluded from ranking /
    failure memory (they remain in SQLite for inspection via ``query_results``).
    """
    if not model_name or not workload_name:
        return []
    path = Path(db_path)
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, workload_name, config_json, result_json,
                   throughput_rps, run_id, status, session_id
            FROM experiments
            WHERE workload_name = ?
              AND json_extract(config_json, '$.model_name') = ?
              AND IFNULL(session_id, '') != ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (workload_name, model_name, exclude_session_id, max(top_k * 4, 32)),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            config = json.loads(row["config_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(config, dict):
            continue
        if config.get("model_name") != model_name:
            continue
        notes = ""
        try:
            result = json.loads(row["result_json"] or "{}")
            if isinstance(result, dict):
                notes = str(result.get("notes") or "")
        except json.JSONDecodeError:
            notes = ""
        row_fp = fingerprint_from_hardware(_hardware_from_result_json(row["result_json"]))
        if not fingerprints_compatible(current_fingerprint, row_fp):
            continue
        param, value = _recover_param_value(str(row["experiment_id"] or ""), config)
        out.append(
            {
                "experiment_id": row["experiment_id"],
                "workload_name": row["workload_name"],
                "model_name": config.get("model_name"),
                "param": param,
                "value": value,
                "status": row["status"],
                "run_id": row["run_id"] or "",
                "session_id": row["session_id"],
                "throughput_rps": row["throughput_rps"],
                "notes": notes,
                "claim_level": CLAIM_LEVEL,
                "hardware_fingerprint": dict(row_fp) if row_fp is not None else None,
            }
        )
        if len(out) >= top_k:
            break
    return out
