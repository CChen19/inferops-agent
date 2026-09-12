"""SQLite experiment memory — stores (workload, config) → result for agent lookups."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from inferops.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentValidityStatus,
    is_promotable,
)

_DEFAULT_DB = Path("inferops_memory.db")

_CONTRACT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "TEXT"),
    ("status", "TEXT"),
    ("session_id", "TEXT"),
    ("mlflow_run_id", "TEXT"),
    ("schema_version", "TEXT"),
    ("workload_hash", "TEXT"),
    ("promotable", "INTEGER"),
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_contract_columns(conn: sqlite3.Connection) -> None:
    """Add Week-1 contract columns to older DBs without wiping data."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    for col, typedef in _CONTRACT_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {typedef}")


def init_db(db_path: Path = _DEFAULT_DB) -> None:
    """Create tables if they don't exist and migrate contract columns."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id    TEXT    UNIQUE NOT NULL,
                workload_name    TEXT    NOT NULL,
                config_hash      TEXT    NOT NULL,
                config_json      TEXT    NOT NULL,
                result_json      TEXT    NOT NULL,
                throughput_rps   REAL,
                ttft_p50_ms      REAL,
                ttft_p99_ms      REAL,
                e2e_p50_ms       REAL,
                e2e_p99_ms       REAL,
                gpu_util_pct     REAL,
                gpu_mem_gb       REAL,
                created_at       TEXT    DEFAULT (datetime('now')),
                run_id           TEXT,
                status           TEXT,
                session_id       TEXT,
                mlflow_run_id    TEXT,
                schema_version   TEXT,
                workload_hash    TEXT,
                promotable       INTEGER DEFAULT 0
            )
        """)
        _migrate_contract_columns(conn)
        conn.commit()


def _config_hash(cfg: ExperimentConfig) -> str:
    """Stable hash of the tuneable vLLM knobs (excludes experiment_id / tags)."""
    knobs = {
        "model_name": cfg.model_name,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enforce_eager": cfg.enforce_eager,
        "enable_chunked_prefill": cfg.enable_chunked_prefill,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "scheduler_policy": cfg.scheduler_policy.value,
    }
    return hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[:16]


def save_result(result: ExperimentResult, db_path: Path = _DEFAULT_DB) -> None:
    """Upsert an ExperimentResult into the memory DB (including contract fields)."""
    init_db(db_path)
    cfg = result.config
    status_value = (
        result.status.value
        if isinstance(result.status, ExperimentValidityStatus)
        else str(result.status)
    )
    promotable_flag = 1 if is_promotable(result) else 0
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, workload_name, config_hash, config_json, result_json,
                 throughput_rps, ttft_p50_ms, ttft_p99_ms, e2e_p50_ms, e2e_p99_ms,
                 gpu_util_pct, gpu_mem_gb,
                 run_id, status, session_id, mlflow_run_id, schema_version,
                 workload_hash, promotable)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                result_json    = excluded.result_json,
                throughput_rps = excluded.throughput_rps,
                ttft_p50_ms    = excluded.ttft_p50_ms,
                ttft_p99_ms    = excluded.ttft_p99_ms,
                e2e_p50_ms     = excluded.e2e_p50_ms,
                e2e_p99_ms     = excluded.e2e_p99_ms,
                gpu_util_pct   = excluded.gpu_util_pct,
                gpu_mem_gb     = excluded.gpu_mem_gb,
                run_id         = excluded.run_id,
                status         = excluded.status,
                session_id     = excluded.session_id,
                mlflow_run_id  = excluded.mlflow_run_id,
                schema_version = excluded.schema_version,
                workload_hash  = excluded.workload_hash,
                promotable     = excluded.promotable,
                config_hash    = excluded.config_hash,
                config_json    = excluded.config_json
            """,
            (
                result.experiment_id,
                cfg.workload.name,
                _config_hash(cfg),
                cfg.model_dump_json(),
                result.model_dump_json(),
                result.throughput_rps,
                result.ttft.p50,
                result.ttft.p99,
                result.e2e_latency.p50,
                result.e2e_latency.p99,
                result.gpu_utilization_pct,
                result.gpu_memory_used_gb,
                result.run_id,
                status_value,
                result.session_id,
                result.mlflow_run_id,
                result.schema_version,
                result.workload_hash,
                promotable_flag,
            ),
        )
        conn.commit()


def query_results(
    workload_name: str | None = None,
    sort_by: str = "throughput_rps",
    top_k: int = 5,
    db_path: Path = _DEFAULT_DB,
    *,
    promotable_only: bool = False,
    status: str | ExperimentValidityStatus | None = None,
) -> list[dict[str, Any]]:
    """Return top-k experiment summaries, optionally filtered by workload/status.

    When promotable_only=True, filter on the denormalized `promotable` flag set
    by the full is_promotable() gate at save time (not status='valid' alone).
    """
    init_db(db_path)
    allowed_sort = {"throughput_rps", "ttft_p50_ms", "e2e_p50_ms", "ttft_p99_ms", "e2e_p99_ms"}
    if sort_by not in allowed_sort:
        sort_by = "throughput_rps"
    # latency metrics: lower is better
    order = "ASC" if "ms" in sort_by else "DESC"

    clauses: list[str] = []
    params: list[Any] = []
    if workload_name:
        clauses.append("workload_name = ?")
        params.append(workload_name)
    if promotable_only:
        clauses.append("promotable = 1")
    elif status is not None:
        clauses.append("status = ?")
        params.append(status.value if isinstance(status, ExperimentValidityStatus) else status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(top_k)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT experiment_id, workload_name, config_hash,
                   throughput_rps, ttft_p50_ms, ttft_p99_ms,
                   e2e_p50_ms, e2e_p99_ms, gpu_util_pct, gpu_mem_gb,
                   created_at, run_id, status, session_id, mlflow_run_id,
                   schema_version, workload_hash, promotable
            FROM experiments
            {where}
            ORDER BY {sort_by} {order}
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_result_by_id(experiment_id: str, db_path: Path = _DEFAULT_DB) -> ExperimentResult | None:
    """Fetch the full ExperimentResult for a given experiment_id.

    Legacy JSON without contract fields deserializes with
    status=insufficient_evidence and a stable legacy run_id. Missing run_id is
    backfilled into the DB so re-reads stay identical (P2-6).
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        return None
    raw = json.loads(row["result_json"])
    had_run_id = bool(raw.get("run_id"))
    result = ExperimentResult.model_validate(raw)
    if not had_run_id:
        # Persist stable backfill so subsequent reads never mint a new id.
        save_result(result, db_path=db_path)
    return result


def get_promotable_result(
    experiment_id: str,
    db_path: Path = _DEFAULT_DB,
) -> ExperimentResult | None:
    """Return the result only if it passes the promotable gate; else None."""
    result = get_result_by_id(experiment_id, db_path=db_path)
    if result is None or not is_promotable(result):
        return None
    return result
