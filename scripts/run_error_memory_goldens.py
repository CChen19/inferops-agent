#!/usr/bin/env python
"""Deterministic error-memory golden gate (CPU / fixture only).

GPU-not-run is not a pass. Empty / skipped fixture set is a fail.
Consumes Week-1 is_promotable + experiment memory — does not invent schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.eval.error_memory_goldens import (
    DEFAULT_FIXTURE_DIR,
    error_memory_golden_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-3 ⑦ error-memory golden gate")
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURE_DIR),
        help="Golden fixture directory",
    )
    args = parser.parse_args()
    gate = error_memory_golden_gate(args.fixtures)
    print(gate.report(), end="")
    if not gate.passed:
        print("error-memory golden gate FAILED", file=sys.stderr)
        return 1
    print("error-memory golden gate passed (CPU/fixture; GPU-not-run ≠ pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
