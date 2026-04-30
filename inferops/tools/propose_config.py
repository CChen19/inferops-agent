"""Tool: propose_config_patch — validate and materialise a single-parameter config change."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from inferops.memory.db import get_result_by_id
from inferops.observability import span

# The knobs the LLM is allowed to propose changes for
TUNABLE_PARAMS = {
    "max_num_batched_tokens": {"type": float, "min": 2048, "max": 8192, "step": 512},
    "max_num_seqs":           {"type": float, "min": 16,   "max": 256,  "step": 16},
    "max_model_len":          {"type": float, "min": 512,  "max": 4096, "step": 512},
    "gpu_memory_utilization": {"type": float, "min": 0.50, "max": 0.85, "step": 0.05},
    "enable_chunked_prefill": {"type": bool},
    "enable_prefix_caching":  {"type": bool},
}


class ProposeConfigInput(BaseModel):
    """Input for propose_config_patch."""
    base_experiment_id: str = Field(
        description="experiment_id of the run to use as the starting config."
    )
    param: str = Field(
        description=f"The single vLLM parameter to change. One of: {list(TUNABLE_PARAMS)}"
    )
    value: Any = Field(
        description="New value for the parameter. Must be within the safe range for RTX 3060."
    )
    rationale: str = Field(
        description="One sentence explaining why this change is expected to improve performance."
    )
    new_experiment_id: str = Field(
        description="Unique name for the resulting experiment, e.g. 'chunked_v2'."
    )


class ProposeConfigOutput(BaseModel):
    """Output from propose_config_patch."""
    new_experiment_id: str
    base_experiment_id: str
    param: str
    old_value: Any
    new_value: Any
    rationale: str
    patch: dict[str, Any]
    warning: str = ""


def propose_config_patch(inp: ProposeConfigInput) -> ProposeConfigOutput:
    """
    Validate a single-parameter config change and return the patch dict.

    Accepts a diff (one param + value) rather than a full config to reduce
    LLM error surface. Validates the proposed value against hardware-safe
    bounds for the RTX 3060. The returned patch can be passed directly to
    run_benchmark's config_patch field.
    """
    if inp.param not in TUNABLE_PARAMS:
        raise ValueError(
            f"'{inp.param}' is not tunable. Choose from: {list(TUNABLE_PARAMS)}"
        )

    spec = TUNABLE_PARAMS[inp.param]

    # Type-coerce boolean params
    value = inp.value
    if spec["type"] is bool:
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")
        else:
            value = bool(value)
    else:
        value = float(value)
        lo, hi = spec["min"], spec["max"]
        if not (lo <= value <= hi):
            raise ValueError(f"{inp.param}={value} outside safe range [{lo}, {hi}]")

    with span("tool.propose_config_patch", {"param": inp.param, "value": str(value)}):
        result = get_result_by_id(inp.base_experiment_id)

    warning = ""
    old_value: Any = None
    if result is None:
        warning = f"base_experiment_id '{inp.base_experiment_id}' not found in memory DB — patch is relative to default config"
    else:
        old_value = getattr(result.config, inp.param, None)

    return ProposeConfigOutput(
        new_experiment_id=inp.new_experiment_id,
        base_experiment_id=inp.base_experiment_id,
        param=inp.param,
        old_value=old_value,
        new_value=value,
        rationale=inp.rationale,
        patch={inp.param: value},
        warning=warning,
    )
