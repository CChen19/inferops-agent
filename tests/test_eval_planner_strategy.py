"""Tests for fair-protocol planner strategies (Stage B slice 3)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inferops.citations import sources_from_context, valid_structured_citations
from inferops.eval.planner_strategy import (
    run_fair_comparison,
    run_planner_no_rag_strategy,
    run_planner_rag_strategy,
)
from inferops.eval.protocol import BudgetPolicy, HiddenResultFixture
from inferops.eval.real_graph import ScriptedBottleneckLLM


def _contract_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "validity_status": "valid",
            "error_rate": 0.01,
            "has_config_evidence": True,
            "bottleneck": row.get("bottleneck", "scheduling-bound"),
        }
        for row in rows
    ]


def _tiny_fixture() -> HiddenResultFixture:
    """Default + unread high-score row + planner-target row (chunked prefill)."""
    rows = _contract_rows(
        [
            {
                "max_num_batched_tokens": 2048,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "throughput_rps": 10.0,
                "tokens_per_second": 1000.0,
                "ttft_p99_ms": 210.0,
                "e2e_p50_ms": 1000.0,
                "bottleneck": "scheduling-bound",
            },
            {
                "max_num_batched_tokens": 4096,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "throughput_rps": 99.0,
                "tokens_per_second": 9000.0,
                "ttft_p99_ms": 50.0,
                "e2e_p50_ms": 500.0,
                "bottleneck": "compute-bound",
            },
            {
                "max_num_batched_tokens": 2048,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "throughput_rps": 12.0,
                "tokens_per_second": 1200.0,
                "ttft_p99_ms": 180.0,
                "e2e_p50_ms": 950.0,
                "bottleneck": "scheduling-bound",
            },
        ]
    )
    return HiddenResultFixture.from_rows(rows)


def test_planner_observes_only_after_pick_not_unread_high_score():
    """Planner cannot adopt an unread high-score row it never chose."""
    fixture = _tiny_fixture()
    llm = ScriptedBottleneckLLM(default_bottleneck="scheduling-bound")
    run = run_planner_rag_strategy(
        fixture,
        BudgetPolicy(total_slots=2),
        workload_name="chat_short",
        bottleneck="scheduling-bound",
        llm=llm,
    )

    assert run.ledger.records[0].kind == "baseline"
    assert len(run.ledger.records) == 2

    trial = run.ledger.records[1]
    assert trial.config["enable_chunked_prefill"] is True
    assert trial.config["max_num_batched_tokens"] == 2048
    assert trial.config["max_num_batched_tokens"] != 4096

    tried = {fixture.search_space.config_key(r.config) for r in run.ledger.records}
    high_key = fixture.search_space.config_key(
        {
            "max_num_batched_tokens": 4096,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
        }
    )
    assert high_key not in tried


def test_planner_no_rag_skips_retrieval():
    """planner_no_rag patches retrieval to empty — never hits the corpus."""
    fixture = _tiny_fixture()
    llm = ScriptedBottleneckLLM(default_bottleneck="scheduling-bound")

    with patch(
        "inferops.agent.planner._retrieve_knowledge",
        return_value="SHOULD_NOT_BE_USED",
    ) as mock_retrieve:
        run = run_planner_no_rag_strategy(
            fixture,
            BudgetPolicy(total_slots=2),
            workload_name="chat_short",
            bottleneck="scheduling-bound",
            llm=llm,
        )

    mock_retrieve.assert_not_called()
    assert run.strategy_name == "planner_no_rag"
    assert len(run.ledger.records) == 2
    assert [record.kind for record in run.ledger.records] == ["baseline", "trial"]


def test_planner_rag_may_call_retrieval():
    fixture = _tiny_fixture()
    llm = ScriptedBottleneckLLM(default_bottleneck="scheduling-bound")
    retrieval_context = (
        "[source: test_doc] §Test\n"
        "chunk_id=chunk_0 version=inferops-corpus-1\n"
        "stub chunk"
    )
    assert sources_from_context(retrieval_context)

    with (
        patch(
            "inferops.agent.planner._retrieve_knowledge",
            return_value=retrieval_context,
        ) as mock_retrieve,
        patch(
            "inferops.agent.planner.valid_structured_citations",
            wraps=valid_structured_citations,
        ) as mock_citation_gate,
    ):
        run = run_planner_rag_strategy(
            fixture,
            BudgetPolicy(total_slots=2),
            workload_name="chat_short",
            bottleneck="scheduling-bound",
            llm=llm,
        )

    mock_retrieve.assert_called()
    assert mock_citation_gate.call_args.args[2] == {"test_doc"}
    assert mock_citation_gate.call_args.args[3] == {
        ("chunk_0", "test_doc", "inferops-corpus-1")
    }
    assert [record.kind for record in run.ledger.records] == ["baseline", "trial"]


def test_planner_budget_policy_matches_other_strategies():
    """baseline=1 slot; budget=2 => at most one planner trial."""
    fixture = _tiny_fixture()
    budget = BudgetPolicy(total_slots=2)

    run = run_planner_rag_strategy(
        fixture,
        budget,
        workload_name="chat_short",
        bottleneck="scheduling-bound",
    )

    assert run.score["n_paid"] == 2
    assert len(run.ledger.records) == 2
    assert run.ledger.records[0].kind == "baseline"
    assert run.ledger.records[1].kind == "trial"


def test_planner_score_run_decomposed_fields():
    fixture = _tiny_fixture()
    run = run_planner_rag_strategy(
        fixture,
        BudgetPolicy(total_slots=2),
        workload_name="chat_short",
        bottleneck="scheduling-bound",
    )

    assert set(run.score.keys()) == {
        "valid_result_in_budget",
        "first_valid_n",
        "confirmed_gain",
        "wasted_trials",
        "n_paid",
        "gap_pct",
    }
    assert "composite" not in run.score


def test_run_fair_comparison_includes_planner_strategies():
    fixture = _tiny_fixture()
    runs = run_fair_comparison(
        fixture,
        BudgetPolicy(total_slots=2),
        workload_name="chat_short",
        seed=1,
    )
    assert set(runs) == {
        "default",
        "random",
        "online_local_search",
        "planner_rag",
        "planner_no_rag",
    }


def test_planner_no_build_graph_or_benchmark():
    fixture = _tiny_fixture()
    with (
        patch("inferops.agent.graph.build_graph") as mock_build,
        patch("inferops.eval.real_graph.StubBenchmarkRecorder") as mock_stub,
    ):
        run_planner_rag_strategy(
            fixture,
            BudgetPolicy(total_slots=2),
            workload_name="chat_short",
            bottleneck="scheduling-bound",
        )

    mock_build.assert_not_called()
    mock_stub.assert_not_called()
