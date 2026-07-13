# vLLM SpecDec Live Log Metrics

CSV: `docs/lyris_qwen235b_swe_pard2_k_sweep_live_log_metrics_20260620.csv`

These rows are live log telemetry. Final `breakdown.json` rows remain authoritative when available.

## State Counts

- `COMPLETED`: 35

## Snapshot

| domain | model | k | temp | batch | state | elapsed | live tok/s/GPU | live acceptance | mean accept len | final tok/s/GPU | final acceptance |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| swe | qwen235b | 1 | 0.0 | 1 | COMPLETED | 02:29:13 | 1.5500 | 8.8 | 1.09 | 1.8625 | 12.5202 |
| swe | qwen235b | 1 | 1.0 | 1 | COMPLETED | 02:35:04 | 1.4500 | 3.5 | 1.04 | 1.7766 | 8.0741 |
| swe | qwen235b | 1 | 1.0 | 2 | COMPLETED | 02:33:12 | 1.4000 | 0.0 | 1.00 | 3.5950 | 7.6799 |
| swe | qwen235b | 1 | 1.0 | 4 | COMPLETED | 02:39:49 | 1.4000 | 0.0 | 1.00 | 6.9856 | 9.6106 |
| swe | qwen235b | 1 | 1.0 | 8 | COMPLETED | 02:47:00 | 1.4000 | 0.0 | 1.00 | 13.3138 | 20.4293 |
| swe | qwen235b | 1 | 1.0 | 16 | COMPLETED | 02:51:06 | 1.3750 | 0.0 | 1.00 | 26.0030 | 13.1989 |
| swe | qwen235b | 1 | 1.0 | 32 | COMPLETED | 02:54:47 | 1.3500 | 1.9 | 1.02 | 50.2250 | 14.4944 |
| swe | qwen235b | 3 | 0.0 | 1 | COMPLETED | 02:25:40 | 1.4500 | 1.2 | 1.04 | 1.9083 | 4.9933 |
| swe | qwen235b | 3 | 0.0 | 2 | COMPLETED | 02:30:54 | 2.8250 | 0.6 | 1.02 | 3.6966 | 4.5441 |
| swe | qwen235b | 3 | 0.0 | 4 | COMPLETED | 02:32:58 | 2.2500 | 0.4 | 1.01 | 7.3051 | 5.2513 |
| swe | qwen235b | 3 | 0.0 | 8 | COMPLETED | 02:37:13 | 1.3750 | 0.0 | 1.00 | 14.4123 | 4.8336 |
| swe | qwen235b | 3 | 0.0 | 16 | COMPLETED | 02:39:36 | 2.8250 | 0.6 | 1.02 | 28.0011 | 5.1835 |
| swe | qwen235b | 3 | 0.0 | 32 | COMPLETED | 02:51:32 | 1.4000 | 1.2 | 1.04 | 52.5270 | 6.7100 |
| swe | qwen235b | 3 | 1.0 | 1 | COMPLETED | 02:32:29 | 1.4750 | 1.2 | 1.04 | 1.8340 | 3.6790 |
| swe | qwen235b | 3 | 1.0 | 2 | COMPLETED | 02:36:20 | 1.3750 | 0.0 | 1.00 | 3.5587 | 3.6157 |
| swe | qwen235b | 3 | 1.0 | 4 | COMPLETED | 02:36:50 | 1.4000 | 0.6 | 1.02 | 7.0393 | 4.1792 |
| swe | qwen235b | 3 | 1.0 | 8 | COMPLETED | 02:44:50 | 1.3750 | 0.0 | 1.00 | 13.4002 | 4.2757 |
| swe | qwen235b | 3 | 1.0 | 16 | COMPLETED | 02:49:53 | 1.3500 | 0.0 | 1.00 | 26.3989 | 4.1337 |
| swe | qwen235b | 3 | 1.0 | 32 | COMPLETED | 03:01:38 | 1.4250 | 0.0 | 1.00 | 48.5545 | 5.1070 |
| swe | qwen235b | 5 | 0.0 | 1 | COMPLETED | 02:23:59 | 1.4500 | 0.7 | 1.04 | 1.9268 | 3.0672 |
| swe | qwen235b | 5 | 0.0 | 2 | COMPLETED | 02:28:59 | 1.5250 | 1.8 | 1.09 | 3.7698 | 3.0093 |
| swe | qwen235b | 5 | 0.0 | 4 | COMPLETED | 02:31:23 | 1.4500 | 0.7 | 1.04 | 7.3930 | 3.1207 |
| swe | qwen235b | 5 | 0.0 | 8 | COMPLETED | 02:37:24 | 1.4000 | 0.0 | 1.00 | 14.0920 | 4.1922 |
| swe | qwen235b | 5 | 0.0 | 16 | COMPLETED | 02:39:45 | 1.4000 | 0.0 | 1.00 | 27.4034 | 3.3788 |
| swe | qwen235b | 5 | 0.0 | 32 | COMPLETED | 02:46:16 | 1.3750 | 1.1 | 1.06 | 52.8727 | 3.5747 |
| swe | qwen235b | 5 | 1.0 | 1 | COMPLETED | 02:28:15 | 1.4000 | 0.0 | 1.00 | 1.9243 | 3.4410 |
| swe | qwen235b | 5 | 1.0 | 2 | COMPLETED | 02:37:00 | 1.4000 | 0.4 | 1.02 | 3.5128 | 2.2315 |
| swe | qwen235b | 5 | 1.0 | 4 | COMPLETED | 02:40:00 | 1.4000 | 0.0 | 1.00 | 6.8748 | 3.5121 |
| swe | qwen235b | 5 | 1.0 | 8 | COMPLETED | 02:45:46 | 1.3750 | 0.0 | 1.00 | 13.3737 | 1.8703 |
| swe | qwen235b | 5 | 1.0 | 16 | COMPLETED | 02:49:07 | 2.1750 | 0.0 | 1.00 | 25.8771 | 3.6397 |
| swe | qwen235b | 5 | 1.0 | 32 | COMPLETED | 02:57:09 | 1.3250 | 0.0 | 1.00 | 49.9330 | 2.9528 |
| swe | qwen235b | 9 | 0.0 | 1 | COMPLETED | 02:25:21 | 1.4250 | 0.2 | 1.02 | 1.9089 | 1.7194 |
| swe | qwen235b | 9 | 1.0 | 1 | COMPLETED | 02:29:19 | 1.4250 | 0.0 | 1.00 | 1.8681 | 1.3173 |
| swe | qwen235b | 11 | 0.0 | 1 | COMPLETED | 02:24:14 | 1.4750 | 0.5 | 1.05 | 1.9275 | 1.4075 |
| swe | qwen235b | 11 | 1.0 | 1 | COMPLETED | 02:28:36 | 1.4500 | 0.2 | 1.02 | 1.8575 | 1.0778 |
