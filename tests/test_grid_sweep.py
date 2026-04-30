"""Unit tests for grid sweep config generation."""

from __future__ import annotations

import scripts.run_grid_sweep as grid
from workloads.definitions import CHAT_SHORT


def test_grid_configs_respect_vllm_batched_tokens_constraint():
    configs = grid._build_configs(CHAT_SHORT)

    assert len(configs) == 12
    assert all(c.max_num_batched_tokens >= c.max_model_len for c in configs)
    assert {c.max_num_batched_tokens for c in configs} == {2048, 3072, 4096}
