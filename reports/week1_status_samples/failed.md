# Status sample: `failed`

**Synthetic fixture — not a real performance claim.**

| Field | Value |
|---|---|
| experiment_id | `w1_sample_failed` |
| run_id | `cccccccccccccccccccccccccccccccc` |
| session_id | `w1samp_` |
| mlflow_run_id | `mlflow-failed-001` |
| schema_version | `1` |
| status | `failed` |
| notes | `vLLM OOM during startup` |
| actual_config | `null` |
| config_evidence | `null` |

## Gate

Startup / crash / OOM → `failed`. `is_promotable` → **false**.
