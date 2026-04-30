"""Run the inferops optimizer agent on one workload.

Usage:
    python scripts/run_agent.py --workload chat_short --llm deepseek --budget 8
    python scripts/run_agent.py --workload long_context_qa --llm claude --budget 6
    python scripts/run_agent.py --workload chat_short --llm deepseek --prefix myrun1_

Requires:
    DEEPSEEK_API_KEY env var  (for --llm deepseek)
    ANTHROPIC_API_KEY env var (for --llm claude)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inferops.agent.graph import make_llm, run_agent
from inferops.agent.state import WORKLOAD_PRIMARY_METRIC

VALID_WORKLOADS = list(WORKLOAD_PRIMARY_METRIC.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inferops optimizer agent")
    parser.add_argument(
        "--workload", required=True, choices=VALID_WORKLOADS,
        help="Workload to optimize",
    )
    parser.add_argument(
        "--llm", default="deepseek", choices=["deepseek", "claude"],
        help="LLM backend for the planner (default: deepseek)",
    )
    parser.add_argument(
        "--budget", type=int, default=8,
        help="Max number of vLLM experiments (including baseline). Default: 8",
    )
    parser.add_argument(
        "--prefix", default=None,
        help="Experiment ID prefix for this session (auto-generated if omitted)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    args = parser.parse_args()

    llm = make_llm(backend=args.llm, temperature=args.temperature)
    final_state = run_agent(
        workload_name=args.workload,
        llm=llm,
        max_experiments=args.budget,
        session_prefix=args.prefix,
    )

    primary = WORKLOAD_PRIMARY_METRIC[args.workload]
    best = final_state.get("best_summary")
    baseline = final_state.get("baseline_summary")

    print("\n=== RESULT ===")
    print(f"Workload:    {args.workload}")
    print(f"LLM:         {args.llm}")
    print(f"Experiments: {len(final_state['tried_experiment_ids'])}")
    print(f"Stop reason: {final_state['stop_reason']}")
    if baseline and best:
        print(f"Baseline {primary}: {baseline[primary]:.3f}")
        print(f"Best     {primary}: {best[primary]:.3f}  ({best['vs_baseline_pct']:+.1f}%)")
        if best["param_changed"]:
            print(f"Best config:  {best['param_changed']}={best['value_changed']}")


if __name__ == "__main__":
    main()
