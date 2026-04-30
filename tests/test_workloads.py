"""Unit tests for workload definitions and prompt dispatch."""

from __future__ import annotations

import pytest

from inferops.schemas import WorkloadSpec
from workloads.definitions import ALL_WORKLOADS, get_prompts


def test_all_workloads_have_unique_names():
    names = [w.name for w in ALL_WORKLOADS]

    assert len(names) == len(set(names))


@pytest.mark.parametrize("workload", ALL_WORKLOADS, ids=lambda w: w.name)
def test_get_prompts_returns_warmup_plus_measure_prompts(workload):
    prompts = get_prompts(workload)

    assert len(prompts) == workload.num_requests + 10
    assert all(isinstance(p, str) and p for p in prompts)


def test_get_prompts_unknown_workload_raises():
    workload = WorkloadSpec(
        name="unknown",
        prompt_template="",
        num_requests=1,
        concurrency=1,
        input_len=1,
        output_len=1,
    )

    with pytest.raises(ValueError, match="Unknown workload"):
        get_prompts(workload)
