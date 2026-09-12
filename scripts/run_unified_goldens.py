#!/usr/bin/env python
"""Deterministic unified ⑦ golden gate (CPU / fixture only).

Consumes measurement + recovery + error-memory goldens. Empty / skipped
sets FAIL. GPU-not-run is not a pass. Fake/offline is not real-LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.eval.unified_goldens import (
    DEFAULT_MANIFEST,
    unified_golden_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-3 ⑦ unified golden gate")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Unified golden manifest path",
    )
    parser.add_argument(
        "--real-llm-json",
        default="",
        help="Optional separate real-LLM campaign JSON (never required to pass)",
    )
    args = parser.parse_args()
    campaign = None
    if args.real_llm_json:
        campaign = json.loads(Path(args.real_llm_json).read_text(encoding="utf-8"))
    gate = unified_golden_gate(args.manifest, real_llm_campaign=campaign)
    print(gate.report(), end="")
    print(
        f"GPU: {gate.gpu.status} — {gate.gpu.detail}\n"
        f"real-LLM: {gate.real_llm.status} — {gate.real_llm.detail}"
    )
    if not gate.passed:
        print("unified golden gate FAILED", file=sys.stderr)
        return 1
    print(
        "unified golden gate passed "
        "(offline fixtures; GPU-not-run ≠ pass; real-LLM is separate)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
