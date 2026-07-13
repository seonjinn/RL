# vLLM SpecDec Live Log Metrics

CSV: `docs/oci_qmath_pard_pard2_k16_focus_live_log_metrics_20260620.csv`

These rows are live log telemetry. Final `breakdown.json` rows remain authoritative when available.

## State Counts

- `COMPLETED`: 5
- `FAILED`: 12

## Snapshot

| domain | model | k | temp | batch | state | elapsed | live tok/s/GPU | live acceptance | mean accept len | final tok/s/GPU | final acceptance |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | qwen32 | 16 | 0.0 | 1 | FAILED | 00:01:59 |  |  |  |  |  |
| math | qwen32 | 16 | 0.0 | 2 | FAILED | 00:01:56 |  |  |  |  |  |
| math | qwen32 | 16 | 0.0 | 4 | FAILED | 00:01:47 |  |  |  |  |  |
| math | qwen32 | 16 | 0.0 | 8 | FAILED | 00:01:46 |  |  |  |  |  |
| math | qwen32 | 16 | 0.0 | 16 | FAILED | 00:02:03 |  |  |  |  |  |
| math | qwen32 | 16 | 0.0 | 32 | FAILED | 00:01:59 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 1 | FAILED | 00:01:42 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 2 | COMPLETED | 01:19:36 | 1.7000 | 0.0 | 1.00 | 18.7918 | 11.4813 |
| math | qwen32 | 16 | 1.0 | 2 | FAILED | 00:01:52 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 4 | COMPLETED | 01:17:20 | 2.4500 | 2.8 | 1.46 | 23.2253 | 4.8716 |
| math | qwen32 | 16 | 1.0 | 4 | FAILED | 00:01:47 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 8 | COMPLETED | 01:38:24 | 2.5250 | 3.1 | 1.50 | 49.1491 | 9.4058 |
| math | qwen32 | 16 | 1.0 | 8 | FAILED | 00:01:39 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 16 | COMPLETED | 01:57:02 | 1.9250 | 0.9 | 1.15 | 72.1916 | 10.8988 |
| math | qwen32 | 16 | 1.0 | 16 | FAILED | 00:01:49 |  |  |  |  |  |
| math | qwen32 | 16 | 1.0 | 32 | COMPLETED | 02:32:09 | 1.6750 | 0.0 | 1.00 | 109.4968 | 8.4613 |
| math | qwen32 | 16 | 1.0 | 32 | FAILED | 00:01:41 |  |  |  |  |  |
