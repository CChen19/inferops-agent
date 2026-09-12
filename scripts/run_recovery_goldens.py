#!/usr/bin/env python
"""Deterministic recovery golden gate (CPU / fixture only).

GPU-not-run is not a pass. Empty / skipped fixture set is a fail.
Consumes Tune ⑧ recovery — does not invent a second schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.eval.recovery_goldens import (
    DEFAULT_FIXTURE_DIR,
    recovery_golden_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-3 P0-⑧ recovery golden gate")
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURE_DIR),
        help="Golden fixture directory",
    )
    args = parser.parse_args()
    gate = recovery_golden_gate(args.fixtures)
    print(gate.report(), end="")
    if not gate.passed:
        print("recovery golden gate FAILED", file=sys.stderr)
        return 1
    print("recovery golden gate passed (CPU/fixture; GPU-not-run ≠ pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
