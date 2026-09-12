# Status sample: `insufficient_evidence`

**Synthetic fixture — not a real performance claim.**

| Field | Value |
|---|---|
| experiment_id | `w1_sample_insufficient` |
| run_id | `dddddddddddddddddddddddddddddddd` |
| session_id | `w1samp_` |
| mlflow_run_id | `mlflow-insuff-001` |
| schema_version | `1` |
| status | `insufficient_evidence` |
| throughput_rps (synthetic) | `99.9` (intentionally high — still not promotable) |
| actual_config | `null` |
| config_evidence.kind | `external_unverified` |
| config_evidence.verified | `false` |
| config_evidence notes | `HTTP health OK only; healthy ≠ config applied` |

## Gate

High score **without** critical actual-config evidence → still
`insufficient_evidence`. `is_promotable` → **false**. This is the exact class of
candidate that previously could win `best_summary` and is now blocked.
