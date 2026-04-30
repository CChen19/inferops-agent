"""SQLite experiment memory — stores (workload, config) → result for agent lookups."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from inferops.schemas import ExperimentConfig, ExperimentResult

_DEFAULT_DB = Path("inferops_memory.db")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = _DEFAULT_DB) -> None:
    """Create tables if they don't exist."""
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
                created_at       TEXT    DEFAULT (datetime('now'))
            )
        """)
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
    """Upsert an ExperimentResult into the memory DB."""
    init_db(db_path)
    cfg = result.config
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (experiment_id, workload_name, config_hash, config_json, result_json,
                 throughput_rps, ttft_p50_ms, ttft_p99_ms, e2e_p50_ms, e2e_p99_ms,
                 gpu_util_pct, gpu_mem_gb)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                result_json    = excluded.result_json,
                throughput_rps = excluded.throughput_rps,
                ttft_p50_ms    = excluded.ttft_p50_ms,
                ttft_p99_ms    = excluded.ttft_p99_ms,
                e2e_p50_ms     = excluded.e2e_p50_ms,
                e2e_p99_ms     = excluded.e2e_p99_ms,
                gpu_util_pct   = excluded.gpu_util_pct,
                gpu_mem_gb     = excluded.gpu_mem_gb
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
            ),
        )
        conn.commit()


def query_results(
    workload_name: str | None = None,
    sort_by: str = "throughput_rps",
    top_k: int = 5,
    db_path: Path = _DEFAULT_DB,
) -> list[dict[str, Any]]:
    """Return top-k experiment summaries, optionally filtered by workload."""
    init_db(db_path)
    allowed_sort = {"throughput_rps", "ttft_p50_ms", "e2e_p50_ms", "ttft_p99_ms", "e2e_p99_ms"}
    if sort_by not in allowed_sort:
        sort_by = "throughput_rps"
    # latency metrics: lower is better
    order = "ASC" if "ms" in sort_by else "DESC"

    where = "WHERE workload_name = ?" if workload_name else ""
    params: list[Any] = [workload_name] if workload_name else []
    params.append(top_k)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT experiment_id, workload_name, config_hash,
                   throughput_rps, ttft_p50_ms, ttft_p99_ms,
                   e2e_p50_ms, e2e_p99_ms, gpu_util_pct, gpu_mem_gb,
                   created_at
            FROM experiments
            {where}
            ORDER BY {sort_by} {order}
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_result_by_id(experiment_id: str, db_path: Path = _DEFAULT_DB) -> ExperimentResult | None:
    """Fetch the full ExperimentResult for a given experiment_id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        return None
    return ExperimentResult.model_validate_json(row["result_json"])
