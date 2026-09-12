#!/usr/bin/env python
"""Separate real-LLM multi-run evidence (N≥3) for ⑦ closeout.

Missing API key → explicit blocker report (not a pass, not fake-LLM).
Never labels offline / scripted LLM as live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.eval.real_llm_goldens import (
    DEFAULT_JSON,
    DEFAULT_REPORT,
    N_RUNS,
    run_real_llm_campaign,
    write_campaign_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-3 ⑦ real-LLM multi-run evidence")
    parser.add_argument("--n", type=int, default=N_RUNS, help="Live runs (N≥3)")
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--ground-truth", default="tests/fixtures/ground_truth")
    parser.add_argument("--markdown", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    campaign = run_real_llm_campaign(
        n=args.n,
        backend=args.backend,
        ground_truth_dir=args.ground_truth,
    )
    md, js = write_campaign_outputs(
        campaign, markdown_path=args.markdown, json_path=args.json_out
    )
    print(campaign.report_markdown())
    print(f"wrote {md}")
    print(f"wrote {js}")
    if campaign.status == "blocked":
        print("real-LLM: BLOCKED (not a pass; not fake-LLM)")
        return 0
    if not campaign.passed:
        print("real-LLM campaign FAILED", file=sys.stderr)
        return 1
    print(f"real-LLM campaign passed ({campaign.n_completed}/{campaign.n_requested})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
