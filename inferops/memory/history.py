"""Compatible prior-session experiment hints for the planner.

Compatibility is model + workload_hash + hardware fingerprint. Rows with a
mismatched or unknown fingerprint, or a missing/mismatched workload_hash, stay
in SQLite for inspection but are excluded from planner ranking and
failure-memory duplicate suppression.

Returned rows are ``claim_level=prior_session_hint``. They must never be
treated as this-run metric evidence and must not skip confirmation.

Cross-session hard filter: only *lasting config failures* suppress retries, and
only for the identical full search-knob config — not a shared (param, value)
knob, and not transient timeout/cancel/startup/generic failed rows.
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
_OOM_MARKERS = ("oom", "out of memory", "cuda out of memory")
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "cancelled",
    "canceled",
    "spawn stalled",
    "startup fault",
    "connection refused",
    "not ready after startup",
)
_EVIDENCE_OR_MISMATCH_STATUSES = frozenset({"insufficient_evidence", "invalid"})
_SCAN_PAGE_SIZE = 32
_MAX_SCAN_ROWS = 512


def is_lasting_config_failure(*, status: str, notes: str) -> bool:
    """True only for lasting *config* failures — not transient or evidence gaps.

    OOM (including during startup) counts as a lasting bad config.
    Explicit config-validation language in notes also counts.
    ``invalid`` / ``insufficient_evidence`` (missing evidence or actual-config
    mismatch via derive_status) do **not** prove the candidate is infeasible.
    Generic ``failed`` without OOM/validation (timeout/cancel/spawn) does
    **not** permanently blacklist.
    """
    s = (status or "").lower()
    n = (notes or "").lower()

    if s in _EVIDENCE_OR_MISMATCH_STATUSES:
        return False
    if s in {"cancelled", "canceled"}:
        return False
    if any(m in n for m in _TRANSIENT_MARKERS):
        return False

    if s == "oom" or any(m in n for m in _OOM_MARKERS):
        return True

    if (
        "config validation" in n
        or "validation error" in n
        or "invalid argument" in n
        or "invalid config" in n
    ):
        return True

    return False


def is_history_failure(row: dict[str, Any]) -> bool:
    """True when a prior row is a lasting config failure (cross-session filter)."""
    return is_lasting_config_failure(
        status=str(row.get("status") or ""),
        notes=str(row.get("notes") or ""),
    )


def _knob_default(key: str) -> Any:
    return ExperimentConfig.model_fields[key].default


def normalize_search_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Fill search knobs from ``config``, defaulting missing keys from schema."""
    src = config if isinstance(config, dict) else {}
    return {k: src[k] if k in src else _knob_default(k) for k in _SEARCH_KNOBS}


def search_configs_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    """Exact equality on the agent search-knob set (missing → schema default)."""
    a = normalize_search_config(left)
    b = normalize_search_config(right)
    return all(a[k] == b[k] for k in _SEARCH_KNOBS)


def history_row_suppresses_config(row: dict[str, Any], proposed: dict[str, Any]) -> bool:
    """True iff ``row`` is a lasting failure of the identical full search config."""
    if not is_history_failure(row):
        return False
    row_cfg = row.get("config")
    if not isinstance(row_cfg, dict):
        return False
    return search_configs_equal(proposed, row_cfg)


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
    workload_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Return prior-session rows that match model + workload_hash + hardware.

    Rows from ``exclude_session_id`` (the current session) are omitted.
    Different ``config_json.model_name`` values are incompatible and excluded.
    Missing or mismatched ``workload_hash`` (same workload_name is not enough)
    is excluded from ranking / failure memory (rows remain in SQLite).
    Mismatch or unknown hardware fingerprints are excluded from ranking /
    failure memory (they remain in SQLite for inspection via ``query_results``).
    Candidate rows are paged newest-first until ``top_k`` matches are found,
    the table is exhausted, or ``_MAX_SCAN_ROWS`` have been inspected.
    """
    if not model_name or not workload_name or top_k <= 0:
        return []
    if not workload_hash:
        return []
    current_fingerprint = fingerprint_from_hardware(current_fingerprint)
    if current_fingerprint is None:
        return []

    path = Path(db_path)
    init_db(path)
    out: list[dict[str, Any]] = []
    scanned = 0
    offset = 0
    with _connect(path) as conn:
        while len(out) < top_k and scanned < _MAX_SCAN_ROWS:
            page_size = min(_SCAN_PAGE_SIZE, _MAX_SCAN_ROWS - scanned)
            rows = conn.execute(
                """
                SELECT experiment_id, workload_name, workload_hash, config_json,
                       result_json, throughput_rps, run_id, status, session_id
                FROM experiments
                WHERE workload_name = ?
                  AND json_extract(config_json, '$.model_name') = ?
                  AND IFNULL(session_id, '') != ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (workload_name, model_name, exclude_session_id, page_size, offset),
            ).fetchall()
            if not rows:
                break
            scanned += len(rows)
            offset += len(rows)

            for row in rows:
                stored_hash = row["workload_hash"]
                if not stored_hash or stored_hash != workload_hash:
                    continue
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
                row_fp = fingerprint_from_hardware(
                    _hardware_from_result_json(row["result_json"])
                )
                if not fingerprints_compatible(current_fingerprint, row_fp):
                    continue
                param, value = _recover_param_value(
                    str(row["experiment_id"] or ""), config
                )
                out.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "workload_name": row["workload_name"],
                        "workload_hash": stored_hash,
                        "model_name": config.get("model_name"),
                        "config": normalize_search_config(config),
                        "param": param,
                        "value": value,
                        "status": row["status"],
                        "run_id": row["run_id"] or "",
                        "session_id": row["session_id"],
                        "throughput_rps": row["throughput_rps"],
                        "notes": notes,
                        "claim_level": CLAIM_LEVEL,
                        "hardware_fingerprint": dict(row_fp),
                    }
                )
                if len(out) >= top_k:
                    break

            if len(rows) < page_size:
                break
    return out
