# Status sample: `invalid`

**Synthetic fixture — not a real performance claim.**

| Field | Value |
|---|---|
| experiment_id | `w1_sample_invalid` |
| run_id | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| session_id | `w1samp_` |
| mlflow_run_id | `mlflow-invalid-001` |
| schema_version | `1` |
| status | `invalid` |
| requested_config.max_num_batched_tokens | `4096` |
| actual_config.max_num_batched_tokens | `2048` |
| config_evidence.kind | `instance_identity` |
| config_evidence.verified | `true` |

## Gate

Critical evidence present, but actual knobs disagree with requested → `invalid`.
`is_promotable` → **false**. Must not become best / deploy.
