"""Compatible prior-session experiment hints for the planner.

Compatibility is ``model_name`` + ``workload_name`` only. Experiment rows do
not store a hardware SKU, so this module does not match GPU SKU.

Returned rows are ``claim_level=prior_session_hint``. They must never be
treated as this-run metric evidence and must not skip confirmation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from inferops.memory.db import _connect, init_db
from inferops.schemas import ExperimentConfig

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


def _coerce_knob(key: str, raw: str) -> Any:
    default = _knob_default(key)
    text = re.sub(r"_r\d+$", "", raw)
    if isinstance(default, bool):
        lowered = text.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return default
    try:
        return type(default)(text)
    except (TypeError, ValueError):
        return text


def _recover_param_value(experiment_id: str, config: dict[str, Any]) -> tuple[str | None, Any]:
    diffs: list[tuple[str, Any]] = []
    for key in _SEARCH_KNOBS:
        if key not in config:
            continue
        if config[key] != _knob_default(key):
            diffs.append((key, config[key]))
    if len(diffs) == 1:
        return diffs[0]
    for key in _SEARCH_KNOBS:
        token = f"_{key}_"
        if token in experiment_id:
            return key, _coerce_knob(key, experiment_id.split(token, 1)[1])
    return (diffs[0] if diffs else (None, None))


def query_compatible_history(
    *,
    model_name: str,
    workload_name: str,
    exclude_session_id: str,
    db_path: Path | str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Return prior-session rows that match model + workload.

    Rows from ``exclude_session_id`` (the current session) are omitted.
    Different ``config_json.model_name`` values are incompatible and excluded.
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
            (workload_name, model_name, exclude_session_id, top_k),
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
            }
        )
    return out
