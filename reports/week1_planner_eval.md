# Week-1 P0-③: Real LangGraph Planner Eval Path

Evidence for the real production-graph eval entry (planner → executor →
reflector). Reuses the Week-1 ① experiment contract / `is_promotable` helpers
unchanged — no second contract.

## Modes & flags

| Flag | Mode | What runs | Output dir (default) |
|---|---|---|---|
| `--mock` | `mock` / `preset_strategy_simulation` | `run_random_agent` / `run_greedy_agent` over GT rows. **Not** `build_graph`. | `eval_reports/` |
| `--real-graph` | `real_graph_offline` | Production `build_graph(llm)` + `planner_node`; **fake/scripted LLM**; **stubbed** `run_benchmark` via `tool_boundary_overrides`. | `eval_reports/real_graph/` |
| `--real-llm` | `real_graph_llm` | Same graph; **live** ChatModel. Missing API key → **hard fail** (exit 1). ≤1 workload, small budget. | `eval_reports/real_llm/` |
| `--prefix` | `session` | Score persisted session (unchanged). | `eval_reports/` |

Exactly one mode flag is required.

## CI wiring

`.github/workflows/eval-mock.yml`:

1. `pytest -q` — includes deterministic real-graph / fake-LLM tests (no live OpenRouter).
2. `scripts/run_eval.py --mock …` — preset simulation + regression gate.
3. `scripts/run_eval.py --real-graph …` — offline production planner path, separate `eval_reports/real_graph/` artifact.

Live `--real-llm` is **not** required in CI.

## Acceptance evidence (mapped to tests)

| # | Requirement | Proof |
|---|---|---|
| 1 | `--mock` does not call production planner / `build_graph` | `test_mock_eval_does_not_call_build_graph_or_planner`, `test_mock_trajectory_nodes_are_baseline_names_not_planner`; report `mode_label=preset_strategy_simulation` |
| 2 | Real trajectory Plan→Execute→Reflect | `test_real_graph_trajectory_plan_execute_reflect` — trajectory `node` fields include ordered `planner` → `executor` → `reflector` |
| 3 | Observation → different decisions | `test_observation_change_yields_different_hypotheses` — compute-bound → `max_num_batched_tokens=4096`; scheduling-bound → `enable_chunked_prefill=True` |
| 4 | Illegal params never reach benchmark | `test_illegal_params_never_reach_benchmark` — scripted illegal `tensor_parallel_size` filtered; stub call log empty |
| 5 | No-gain does not promote best | `test_no_gain_does_not_promote_best` — best stays baseline experiment_id |
| 6 | Budget exhaustion `stop_reason` | `test_budget_exhaustion_sets_stop_reason` → `budget_exhausted` |
| 7 | ① regression: unverified ≠ best | `test_unevidenced_high_score_not_promoted_to_best` — uses `is_promotable` / `is_promotable_summary`; high unevidenced score stays off `best_summary` |
| 8 | Real LLM labeled; missing creds ≠ silent pass | `test_require_llm_credentials_fails_loudly`, `test_real_llm_mode_fails_without_credentials`, `test_run_eval_real_llm_missing_creds_exits_nonzero` |
| 9 | `pytest -q` green; mock still works; separate dirs | Full suite green; `test_run_eval_real_graph_writes_separate_dir`; mock help/disclaimer updated |

## Implementation notes

- **Stub surface:** `inferops.agent.executor.tool_boundary_overrides` — only `run_benchmark` / `propose_config` callables. Nodes remain production `planner_node` / `executor_node` / `reflector_node`.
- **Fake LLM:** `ScriptedBottleneckLLM` implements `.invoke()` only; bound through real `build_graph(llm)`.
- **Reflector:** empty-plan detection increments no-improvement streak so scripted / exhausted hypothesis spaces cannot spin past LangGraph recursion limits.
- **Contract:** best selection still gated by ① `is_promotable` / `is_promotable_summary` (executor + eval scoring).

## Commands

```bash
# Preset simulation (CI regression)
python scripts/run_eval.py --mock --ground-truth tests/fixtures/ground_truth \
  --output-dir eval_reports --workloads chat_short --budget 2

# Offline real graph (fake LLM)
python scripts/run_eval.py --real-graph --ground-truth tests/fixtures/ground_truth \
  --output-dir eval_reports/real_graph --workloads chat_short --budget 3

# Live LLM (fails without OPENROUTER_API_KEY)
python scripts/run_eval.py --real-llm --ground-truth tests/fixtures/ground_truth \
  --output-dir eval_reports/real_llm --workloads chat_short --budget 3

pytest -q
```
