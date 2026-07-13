# Lyris SWE OSL32K Temp0/Temp1 Matrix - 2026-06-16

Refreshed at `2026-06-16 01:36 PDT` from `login-lyris`.

Scope: standalone vLLM SWE-Bench Verified OSL32K rows for `temperature=0.0` and `temperature=1.0`, using `top_p=1.0`, `top_k=-1`, `PROMPT_COUNT=4`, `ISL=4096`, `OSL=32768`, and batch sizes `1 2`.

## Current State

| State | Count | Notes |
|---|---:|---|
| `RUNNING` | 13 | Includes all current baseline retries and most Eagle-3 / suffix retries. |
| `COMPLETED` | 4 | Completed jobs are suffix rows from the home retry path. |
| `FAILED` | 14 | Mostly first-attempt `/lustre` submissions and obsolete temp1 rows; home retry rows replace the useful suffix failures. |

Baseline rows are still running, so baseline-relative speedups are still waiting. The table below reports completed or already-written `breakdown.json` rows only; a few rows are from jobs that are still running but have emitted partial breakdowns.

## Breakdown Rows Available

| Temp | Model | Method | Batch | tok/s/GPU | Acceptance | Mean accept len | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 0.0 | Qwen3-235B-A22B | suffix K32 | 1 | 15.18 | 88.49% | 7.57 | completed home retry |
| 0.0 | Qwen3-235B-A22B | suffix K32 | 2 | 22.28 | 79.08% | 6.55 | completed home retry |
| 0.0 | Qwen3-32B | suffix K32 | 1 | 53.76 | 82.94% | 7.68 | completed home retry |
| 0.0 | Qwen3-32B | suffix K32 | 2 | 106.70 | 86.77% | 8.44 | completed home retry |
| 0.0 | Qwen3-30B-A3B | suffix K32 | 1 | 176.72 | 93.92% | 10.01 | completed home retry |
| 0.0 | Qwen3-30B-A3B | suffix K32 | 2 | 309.86 | 90.62% | 9.88 | completed home retry |
| 1.0 | Qwen3-32B | Eagle-3 K3 | 1 | 15.93 | 49.26% | 2.48 | partial breakdown, job still running |
| 1.0 | Qwen3-32B | suffix K32 | 1 | 16.16 | 42.12% | 2.37 | partial breakdown, job still running |
| 1.0 | Qwen3-30B-A3B | suffix K32 | 1 | 168.53 | 91.15% | 9.46 | completed home retry |
| 1.0 | Qwen3-30B-A3B | suffix K32 | 2 | 306.91 | 90.42% | 9.29 | completed home retry |

## Live Telemetry

These are recent Slurm-log readings, not final benchmark rows.

| Temp | Model | Method | Job | Live gen tok/s | Acceptance | Mean accept len |
|---:|---|---|---:|---:|---:|---:|
| 0.0 | Qwen3-235B-A22B | baseline | `2133873` | 8.2 |  |  |
| 0.0 | Qwen3-32B | baseline | `2133876` | 16.2 |  |  |
| 0.0 | Qwen3-30B-A3B | baseline | `2133880` | 20.0 |  |  |
| 1.0 | Qwen3-235B-A22B | baseline | `2133935` | 8.3 |  |  |
| 1.0 | Qwen3-32B | baseline | `2133938` | 17.0 |  |  |
| 1.0 | Qwen3-30B-A3B | baseline | `2133941` | 19.4 |  |  |
| 0.0 | Qwen3-235B-A22B | Eagle-3 K3 | `2133875` | 22.4 | 61.2% | 2.84 |
| 0.0 | Qwen3-30B-A3B | Eagle-3 K3 | `2133934` | 21.6 | 6.9% | 1.21 |
| 1.0 | Qwen3-235B-A22B | Eagle-3 K3 | `2133937` | 15.7 | 32.5% | 1.98 |
| 1.0 | Qwen3-32B | Eagle-3 K3 | `2133940` | 73.4 | 51.0% | 2.53 |
| 1.0 | Qwen3-30B-A3B | Eagle-3 K3 | `2133943` | 19.9 | 5.7% | 1.17 |
| 0.0 | Qwen3-235B-A22B | suffix K32 | `2133931` | 82.9 | 97.7% | 11.30 |
| 0.0 | Qwen3-32B | suffix K32 | `2133932` | 157.1 | 98.2% | 12.70 |
| 0.0 | Qwen3-30B-A3B | suffix K32 | `2133933` | 155.2 | 88.6% | 11.52 |
| 1.0 | Qwen3-235B-A22B | suffix K32 | `2133936` | 14.4 | 31.4% | 1.73 |
| 1.0 | Qwen3-32B | suffix K32 | `2133939` | 107.5 | 67.1% | 5.24 |
| 1.0 | Qwen3-30B-A3B | suffix K32 | `2133942` | 326.3 | 97.0% | 12.56 |

## Artifacts

- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_status.csv`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_live_progress.csv`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_completed_runs.txt`
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics.csv`
