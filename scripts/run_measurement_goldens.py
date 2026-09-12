#!/usr/bin/env python
"""Deterministic measurement-trust golden gate (CPU / fixture only).

GPU-not-run is not a pass. Never invents GPU/perf numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.eval.measurement_goldens import (
    DEFAULT_FIXTURE_DIR,
    measurement_trust_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-2 P0-⑦ measurement-trust gate")
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURE_DIR),
        help="Golden fixture directory",
    )
    args = parser.parse_args()
    gate = measurement_trust_gate(args.fixtures)
    print(gate.report(), end="")
    if not gate.passed:
        print("measurement-trust golden gate FAILED", file=sys.stderr)
        return 1
    print("measurement-trust golden gate passed (CPU/fixture; GPU-not-run ≠ pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
