"""Unit tests for propose_config_patch tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inferops.tools.propose_config import ProposeConfigInput, propose_config_patch


def test_propose_valid_numeric_param(result, tmp_db):
    with patch("inferops.tools.propose_config.get_result_by_id", return_value=result):
        out = propose_config_patch(ProposeConfigInput(
            base_experiment_id="test_default",
            param="max_num_batched_tokens",
            value=4096,
            rationale="Larger batch improves GPU utilisation on short sequences.",
            new_experiment_id="big_batch_v2",
        ))

    assert out.param == "max_num_batched_tokens"
    assert out.new_value == 4096.0
    assert out.patch == {"max_num_batched_tokens": 4096.0}
    assert out.old_value == 2048  # from fixture config
    assert out.warning == ""


def test_propose_valid_bool_param(result):
    with patch("inferops.tools.propose_config.get_result_by_id", return_value=result):
        out = propose_config_patch(ProposeConfigInput(
            base_experiment_id="test_default",
            param="enable_chunked_prefill",
            value=True,
            rationale="Chunked prefill reduces head-of-line blocking for long inputs.",
            new_experiment_id="chunked_v2",
        ))

    assert out.new_value is True
    assert out.patch == {"enable_chunked_prefill": True}


def test_propose_out_of_range():
    with pytest.raises(ValueError, match="outside safe range"):
        propose_config_patch(ProposeConfigInput(
            base_experiment_id="x",
            param="gpu_memory_utilization",
            value=0.99,
            rationale="trying to use all VRAM",
            new_experiment_id="risky",
        ))


def test_propose_unknown_param():
    with pytest.raises(ValueError, match="not tunable"):
        propose_config_patch(ProposeConfigInput(
            base_experiment_id="x",
            param="nonexistent_knob",
            value=42,
            rationale="...",
            new_experiment_id="bad",
        ))


def test_propose_missing_base_experiment():
    with patch("inferops.tools.propose_config.get_result_by_id", return_value=None):
        out = propose_config_patch(ProposeConfigInput(
            base_experiment_id="ghost",
            param="max_num_seqs",
            value=64,
            rationale="reduce concurrency",
            new_experiment_id="new_run",
        ))

    assert "not found" in out.warning
    assert out.old_value is None
