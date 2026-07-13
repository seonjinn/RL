# Lyris Qwen3-235B Standalone Refresh - lyris_swebench_osl32k_temp01_core_matrix_20260616_partial

Refreshed at `2026-06-16T15:55:32-07:00`.

## Queue State

| State | Count |
| --- | ---: |
| `COMPLETED` | 26 |
| `FAILED` | 21 |
| `TIMEOUT` | 5 |

## Suite State

| Suite | State | Count |
| --- | --- | ---: |
| swe | `COMPLETED` | 26 |
| swe | `FAILED` | 21 |
| swe | `TIMEOUT` | 5 |

## Live Telemetry Averages

Live values are parsed from recent Slurm logs and are not final benchmark rows.

| Dataset | Method | Live rows | Avg gen tok/s | Avg acceptance | Avg accept len |
| --- | --- | ---: | ---: | ---: | ---: |
| unknown | `baseline` | 15 | 20.3 |  |  |
| unknown | `eagle3_k3` | 12 | 17.7 | 38.5% | 2.15 |
| unknown | `suffix_k32` | 8 | 119.1 | 78.8% | 9.69 |

## Completed Breakdown Rows

Completed `breakdown.json` files found: `30`.

| Label | Batch | tok/s/GPU | Speedup | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: | ---: |
| `unknown_osl32k_qwen235b_t00_baseline` | 1 | 2.06 | 1.000 | % |  |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 1 | 5.35 |  | 57.24% | 2.72 |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 1 | 5.35 |  | 57.24% | 2.72 |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 1 | 5.35 |  | 57.24% | 2.72 |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 1 | 5.47 | 2.662 | 59.71% | 2.79 |
| `unknown_osl32k_qwen235b_t00_suffix_k32` | 1 | 15.18 |  | 88.49% | 7.57 |
| `unknown_osl32k_qwen235b_t00_suffix_k32` | 1 | 12.25 | 5.957 | 79.30% | 5.97 |
| `unknown_osl32k_qwen235b_t10_baseline` | 1 | 2.13 | 1.038 | % |  |
| `unknown_osl32k_qwen235b_t10_eagle3_k3` | 1 | 3.22 |  | 21.08% | 1.63 |
| `unknown_osl32k_qwen235b_t10_eagle3_k3` | 1 | 3.22 |  | 21.08% | 1.63 |
| `unknown_osl32k_qwen235b_t10_eagle3_k3` | 1 | 3.44 | 1.672 | 25.06% | 1.75 |
| `unknown_osl32k_qwen235b_t10_suffix_k32` | 1 | 4.30 |  | 46.75% | 2.28 |
| `unknown_osl32k_qwen235b_t10_suffix_k32` | 1 | 6.10 | 2.966 | 55.12% | 3.12 |
| `unknown_osl32k_qwen30ba3b_t00_baseline` | 1 | 19.88 | 1.015 | % |  |
| `unknown_osl32k_qwen30ba3b_t00_eagle3_k3` | 1 | 22.41 | 1.145 | 12.26% | 1.37 |
| `unknown_osl32k_qwen30ba3b_t00_suffix_k32` | 1 | 176.72 | 9.026 | 93.92% | 10.01 |
| `unknown_osl32k_qwen30ba3b_t10_baseline` | 1 | 19.58 | 1.000 | % |  |
| `unknown_osl32k_qwen30ba3b_t10_eagle3_k3` | 1 | 21.40 | 1.093 | 11.21% | 1.34 |
| `unknown_osl32k_qwen30ba3b_t10_suffix_k32` | 1 | 168.53 | 8.608 | 91.15% | 9.46 |
| `unknown_osl32k_qwen32_t00_baseline` | 1 | 8.24 | 0.985 | % |  |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 1 | 19.44 | 2.326 | 66.86% | 3.01 |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 1 | 19.44 | 2.326 | 66.86% | 3.01 |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 1 | 19.44 | 2.326 | 66.86% | 3.01 |
| `unknown_osl32k_qwen32_t00_suffix_k32` | 1 | 53.76 | 6.433 | 82.94% | 7.68 |
| `unknown_osl32k_qwen32_t10_baseline` | 1 | 8.36 | 1.000 | % |  |
| `unknown_osl32k_qwen32_t10_eagle3_k3` | 1 | 15.93 | 1.906 | 49.26% | 2.48 |
| `unknown_osl32k_qwen32_t10_suffix_k32` | 1 | 16.16 | 1.934 | 42.12% | 2.37 |
| `unknown_osl32k_qwen235b_t00_baseline` | 2 | 4.18 | 1.000 | % |  |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 2 | 8.33 | 1.991 | 49.45% | 2.48 |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 2 | 8.33 | 1.991 | 49.45% | 2.48 |
| `unknown_osl32k_qwen235b_t00_eagle3_k3` | 2 | 8.33 | 1.991 | 49.45% | 2.48 |
| `unknown_osl32k_qwen235b_t00_suffix_k32` | 2 | 22.28 | 5.328 | 79.08% | 6.55 |
| `unknown_osl32k_qwen235b_t10_baseline` | 2 | 4.12 | 0.985 | % |  |
| `unknown_osl32k_qwen235b_t10_eagle3_k3` | 2 | 6.14 | 1.468 | 23.06% | 1.69 |
| `unknown_osl32k_qwen235b_t10_suffix_k32` | 2 | 7.80 | 1.865 | 50.56% | 2.66 |
| `unknown_osl32k_qwen30ba3b_t00_baseline` | 2 | 39.69 | 1.023 | % |  |
| `unknown_osl32k_qwen30ba3b_t00_eagle3_k3` | 2 | 43.50 | 1.122 | 12.36% | 1.37 |
| `unknown_osl32k_qwen30ba3b_t00_suffix_k32` | 2 | 309.86 | 7.990 | 90.62% | 9.88 |
| `unknown_osl32k_qwen30ba3b_t10_baseline` | 2 | 38.78 | 1.000 | % |  |
| `unknown_osl32k_qwen30ba3b_t10_eagle3_k3` | 2 | 40.30 | 1.039 | 10.77% | 1.32 |
| `unknown_osl32k_qwen30ba3b_t10_suffix_k32` | 2 | 306.91 | 7.913 | 90.42% | 9.29 |
| `unknown_osl32k_qwen32_t00_baseline` | 2 | 16.35 | 0.997 | % |  |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 2 | 39.24 | 2.393 | 71.33% | 3.14 |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 2 | 39.24 | 2.393 | 71.33% | 3.14 |
| `unknown_osl32k_qwen32_t00_eagle3_k3` | 2 | 39.24 | 2.393 | 71.33% | 3.14 |
| `unknown_osl32k_qwen32_t00_suffix_k32` | 2 | 106.70 | 6.508 | 86.77% | 8.44 |
| `unknown_osl32k_qwen32_t10_baseline` | 2 | 16.40 | 1.000 | % |  |
| `unknown_osl32k_qwen32_t10_eagle3_k3` | 2 | 29.73 | 1.813 | 48.80% | 2.46 |
| `unknown_osl32k_qwen32_t10_suffix_k32` | 2 | 39.60 | 2.415 | 62.33% | 4.06 |
