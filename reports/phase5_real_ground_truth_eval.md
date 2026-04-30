# InferOps Eval Report: `real-gt-eaa0495`

- Mode: `mock`
- Generated: `2026-04-30T10:04:52.451544+00:00`
- Budget: `2` experiments per strategy

## Summary

| Strategy | Mean gap % | Mean runs | Mean composite |
|---|---:|---:|---:|
| greedy_agent | +2.51 | 2.0 | 0.8574 |
| random_agent | +4.49 | 2.0 | 0.8176 |

## Workloads

### greedy_agent

| Workload | Metric | GT | Agent | Gap % | Runs | Traj | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| chat_short | throughput_rps | 17.881 | 17.796 | +0.47 | 2 | 0.60 | 0.8676 |
| long_context_qa | throughput_rps | 2.783 | 2.783 | +0.00 | 2 | 0.60 | 0.8700 |
| high_concurrency_short_out | throughput_rps | 84.724 | 83.127 | +1.88 | 2 | 0.60 | 0.8606 |
| long_generation | tokens_per_second | 398.300 | 398.300 | +0.00 | 2 | 0.60 | 0.8700 |
| mixed_traffic | throughput_rps | 10.661 | 9.571 | +10.22 | 2 | 0.60 | 0.8189 |

### random_agent

| Workload | Metric | GT | Agent | Gap % | Runs | Traj | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| chat_short | throughput_rps | 17.881 | 17.831 | +0.28 | 2 | 0.45 | 0.8386 |
| long_context_qa | throughput_rps | 2.783 | 2.755 | +0.98 | 2 | 0.45 | 0.8351 |
| high_concurrency_short_out | throughput_rps | 84.724 | 74.815 | +11.70 | 2 | 0.45 | 0.7815 |
| long_generation | tokens_per_second | 398.300 | 396.540 | +0.44 | 2 | 0.45 | 0.8378 |
| mixed_traffic | throughput_rps | 10.661 | 9.698 | +9.03 | 2 | 0.45 | 0.7948 |
