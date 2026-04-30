"""Unit tests for experiment_memory tool and the underlying SQLite DB."""

from __future__ import annotations

import pytest

from inferops.memory.db import get_result_by_id, init_db, query_results, save_result
from inferops.tools.experiment_memory import QueryMemoryInput, query_experiment_memory


def test_save_and_retrieve(result, tmp_db):
    save_result(result, db_path=tmp_db)
    fetched = get_result_by_id(result.experiment_id, db_path=tmp_db)

    assert fetched is not None
    assert fetched.experiment_id == result.experiment_id
    assert fetched.throughput_rps == result.throughput_rps


def test_upsert_overwrites(result, tmp_db):
    save_result(result, db_path=tmp_db)
    result.throughput_rps = 99.9
    save_result(result, db_path=tmp_db)

    fetched = get_result_by_id(result.experiment_id, db_path=tmp_db)
    assert fetched.throughput_rps == 99.9


def test_query_results_sorted(result, result_b, tmp_db):
    save_result(result, db_path=tmp_db)
    save_result(result_b, db_path=tmp_db)

    rows = query_results(sort_by="throughput_rps", top_k=10, db_path=tmp_db)
    assert len(rows) == 2
    assert rows[0]["throughput_rps"] >= rows[1]["throughput_rps"]


def test_query_results_workload_filter(result, result_b, tmp_db):
    save_result(result, db_path=tmp_db)
    save_result(result_b, db_path=tmp_db)

    rows = query_results(workload_name="chat_short", db_path=tmp_db)
    assert all(r["workload_name"] == "chat_short" for r in rows)


def test_get_nonexistent_returns_none(tmp_db):
    result = get_result_by_id("does_not_exist", db_path=tmp_db)
    assert result is None


def test_query_memory_tool(result, result_b, tmp_db):
    save_result(result, db_path=tmp_db)
    save_result(result_b, db_path=tmp_db)

    from unittest.mock import patch
    with patch("inferops.tools.experiment_memory.query_results",
               wraps=lambda **kw: query_results(**{**kw, "db_path": tmp_db})):
        out = query_experiment_memory(QueryMemoryInput(workload_name="chat_short", top_k=5))

    assert out.total_found >= 1
