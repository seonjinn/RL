# Lyris Qwen3-235B Standalone Fast Refresh - 2026-06-13

Refreshed at `2026-06-14T20:37:03-07:00`.

## Queue State

| State | Count |
| --- | ---: |
| `COMPLETED` | 83 |
| `TIMEOUT` | 2 |

## Suite State

| Suite | State | Count |
| --- | --- | ---: |
| math500 | `COMPLETED` | 3 |
| math500 | `TIMEOUT` | 2 |
| swe | `COMPLETED` | 80 |

## Live Telemetry Averages

Live values are parsed from recent Slurm logs and are not final benchmark rows.

| Dataset | Method | Live rows | Avg gen tok/s | Avg acceptance | Avg accept len |
| --- | --- | ---: | ---: | ---: | ---: |
| full | `eagle3_k11` | 5 | 20.4 | 13.9% | 2.53 |
| full | `eagle3_k9` | 5 | 18.5 | 15.2% | 2.37 |
| full | `pard2_k11` | 5 | 5.5 | 0.4% | 1.05 |
| full | `pard2_k9` | 5 | 5.7 | 0.5% | 1.05 |
| full | `pard_k11` | 5 | 8.6 | 5.2% | 1.57 |
| full | `pard_k9` | 5 | 7.5 | 4.0% | 1.36 |
| full | `suffix_k16` | 5 | 95.6 | 96.0% | 12.47 |
| full | `suffix_k8` | 5 | 43.7 | 67.0% | 6.03 |
| math500 | `baseline` | 1 | 8.3 |  |  |
| math500 | `eagle3_k3` | 1 | 20.4 | 61.1% | 2.83 |
| math500 | `official_pard2_k1` | 1 | 8.1 | 39.0% | 1.39 |
| math500 | `pard_k5` | 1 | 16.5 | 37.9% | 2.89 |
| math500 | `suffix_k32` | 1 | 103.0 | 100.0% | 13.00 |
| verified | `eagle3_k11` | 5 | 16.7 | 13.1% | 2.44 |
| verified | `eagle3_k9` | 5 | 21.8 | 18.0% | 2.62 |
| verified | `pard2_k11` | 5 | 5.6 | 0.3% | 1.03 |
| verified | `pard2_k9` | 5 | 5.6 | 0.2% | 1.02 |
| verified | `pard_k11` | 5 | 7.2 | 2.6% | 1.29 |
| verified | `pard_k9` | 5 | 8.0 | 3.1% | 1.28 |
| verified | `suffix_k16` | 5 | 120.0 | 98.4% | 12.77 |
| verified | `suffix_k8` | 5 | 76.9 | 98.0% | 8.84 |

## Completed Breakdown Rows

Completed `breakdown.json` files found: `83`.

| Label | Batch | tok/s/GPU | Speedup | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: | ---: |
| `math500_osl32k_qwen235b_eagle3_k3` | 1 | 6.19 |  | 69.42% | 3.08 |
| `math500_osl32k_qwen235b_pard_k5` | 1 | 6.31 |  | 56.10% | 3.80 |
| `math500_osl32k_qwen235b_suffix_k32` | 1 | 15.98 |  | 83.79% | 7.82 |
| `math500_osl32k_qwen235b_eagle3_k3` | 2 | 12.00 |  | 69.74% | 3.09 |
| `math500_osl32k_qwen235b_pard_k5` | 2 | 10.59 |  | 50.08% | 3.50 |
| `math500_osl32k_qwen235b_suffix_k32` | 2 | 32.18 |  | 88.71% | 8.71 |
| `full_osl32k_qwen235b_eagle3_k11` | 2 | 9.85 |  | 17.28% | 2.90 |
| `full_osl32k_qwen235b_eagle3_k9` | 2 | 8.54 |  | 19.15% | 2.72 |
| `full_osl32k_qwen235b_pard2_k11` | 2 | 3.68 |  | 1.85% | 1.20 |
| `full_osl32k_qwen235b_pard2_k9` | 2 | 3.76 |  | 5.09% | 1.46 |
| `full_osl32k_qwen235b_pard_k11` | 2 | 7.81 |  | 17.94% | 2.97 |
| `full_osl32k_qwen235b_pard_k9` | 2 | 6.18 |  | 16.25% | 2.46 |
| `full_osl32k_qwen235b_suffix_k16` | 2 | 25.44 |  | 87.55% | 7.43 |
| `full_osl32k_qwen235b_suffix_k8` | 2 | 23.39 |  | 86.50% | 6.44 |
| `full_osl32k_qwen235b_eagle3_k11` | 4 | 14.25 |  | 12.41% | 2.37 |
| `full_osl32k_qwen235b_eagle3_k9` | 4 | 16.38 |  | 17.04% | 2.53 |
| `full_osl32k_qwen235b_pard2_k11` | 4 | 7.30 |  | 1.68% | 1.19 |
| `full_osl32k_qwen235b_pard2_k9` | 4 | 7.10 |  | 1.83% | 1.16 |
| `full_osl32k_qwen235b_pard_k11` | 4 | 11.80 |  | 12.88% | 2.42 |
| `full_osl32k_qwen235b_pard_k9` | 4 | 9.59 |  | 14.46% | 2.30 |
| `full_osl32k_qwen235b_suffix_k16` | 4 | 28.27 |  | 76.24% | 5.42 |
| `full_osl32k_qwen235b_suffix_k8` | 4 | 30.55 |  | 77.12% | 5.03 |
| `full_osl32k_qwen235b_eagle3_k11` | 8 | 27.98 |  | 15.10% | 2.66 |
| `full_osl32k_qwen235b_eagle3_k9` | 8 | 30.44 |  | 18.86% | 2.70 |
| `full_osl32k_qwen235b_pard2_k11` | 8 | 13.59 |  | 2.77% | 1.30 |
| `full_osl32k_qwen235b_pard2_k9` | 8 | 13.56 |  | 3.31% | 1.30 |
| `full_osl32k_qwen235b_pard_k11` | 8 | 19.74 |  | 13.91% | 2.53 |
| `full_osl32k_qwen235b_pard_k9` | 8 | 19.98 |  | 18.12% | 2.63 |
| `full_osl32k_qwen235b_suffix_k16` | 8 | 62.96 |  | 81.19% | 6.84 |
| `full_osl32k_qwen235b_suffix_k8` | 8 | 56.81 |  | 81.41% | 5.55 |
| `full_osl32k_qwen235b_eagle3_k11` | 16 | 60.85 |  | 16.57% | 2.82 |
| `full_osl32k_qwen235b_eagle3_k9` | 16 | 66.08 |  | 20.03% | 2.80 |
| `full_osl32k_qwen235b_pard2_k11` | 16 | 25.51 |  | 1.78% | 1.20 |
| `full_osl32k_qwen235b_pard2_k9` | 16 | 25.38 |  | 2.06% | 1.19 |
| `full_osl32k_qwen235b_pard_k11` | 16 | 35.14 |  | 10.83% | 2.19 |
| `full_osl32k_qwen235b_pard_k9` | 16 | 29.51 |  | 10.43% | 1.94 |
| `full_osl32k_qwen235b_suffix_k16` | 16 | 109.24 |  | 81.08% | 6.87 |
| `full_osl32k_qwen235b_suffix_k8` | 16 | 90.80 |  | 75.59% | 5.09 |
| `full_osl32k_qwen235b_eagle3_k11` | 32 | 115.76 |  | 16.32% | 2.80 |
| `full_osl32k_qwen235b_eagle3_k9` | 32 | 108.23 |  | 18.25% | 2.64 |
| `full_osl32k_qwen235b_pard2_k11` | 32 | 49.98 |  | 1.78% | 1.20 |
| `full_osl32k_qwen235b_pard2_k9` | 32 | 49.45 |  | 2.45% | 1.22 |
| `full_osl32k_qwen235b_pard_k11` | 32 | 69.25 |  | 12.06% | 2.33 |
| `full_osl32k_qwen235b_pard_k9` | 32 | 71.81 |  | 17.21% | 2.55 |
| `full_osl32k_qwen235b_suffix_k16` | 32 | 220.88 |  | 78.86% | 6.46 |
| `full_osl32k_qwen235b_suffix_k8` | 32 | 101.19 |  | 72.68% | 4.90 |
| `verified_osl32k_qwen235b_eagle3_k11` | 2 | 7.62 |  | 12.49% | 2.37 |
| `verified_osl32k_qwen235b_eagle3_k9` | 2 | 10.23 |  | 21.98% | 2.98 |
| `verified_osl32k_qwen235b_pard2_k11` | 2 | 3.73 |  | 1.37% | 1.15 |
| `verified_osl32k_qwen235b_pard2_k9` | 2 | 3.77 |  | 1.68% | 1.15 |
| `verified_osl32k_qwen235b_pard_k11` | 2 | 5.69 |  | 7.18% | 1.79 |
| `verified_osl32k_qwen235b_pard_k9` | 2 | 5.60 |  | 8.77% | 1.79 |
| `verified_osl32k_qwen235b_suffix_k16` | 2 | 18.95 |  | 71.53% | 5.63 |
| `verified_osl32k_qwen235b_suffix_k8` | 2 | 19.95 |  | 83.58% | 5.76 |
| `verified_osl32k_qwen235b_eagle3_k11` | 4 | 18.10 |  | 16.07% | 2.77 |
| `verified_osl32k_qwen235b_eagle3_k9` | 4 | 19.45 |  | 22.15% | 2.99 |
| `verified_osl32k_qwen235b_pard2_k11` | 4 | 7.07 |  | 1.42% | 1.16 |
| `verified_osl32k_qwen235b_pard2_k9` | 4 | 7.08 |  | 1.50% | 1.13 |
| `verified_osl32k_qwen235b_pard_k11` | 4 | 9.11 |  | 9.95% | 2.09 |
| `verified_osl32k_qwen235b_pard_k9` | 4 | 9.80 |  | 10.52% | 1.95 |
| `verified_osl32k_qwen235b_suffix_k16` | 4 | 48.99 |  | 86.20% | 7.62 |
| `verified_osl32k_qwen235b_suffix_k8` | 4 | 33.75 |  | 84.78% | 5.75 |
| `verified_osl32k_qwen235b_eagle3_k11` | 8 | 29.32 |  | 15.69% | 2.73 |
| `verified_osl32k_qwen235b_eagle3_k9` | 8 | 30.97 |  | 18.93% | 2.70 |
| `verified_osl32k_qwen235b_pard2_k11` | 8 | 13.41 |  | 2.19% | 1.24 |
| `verified_osl32k_qwen235b_pard2_k9` | 8 | 13.85 |  | 1.89% | 1.17 |
| `verified_osl32k_qwen235b_pard_k11` | 8 | 17.17 |  | 12.48% | 2.37 |
| `verified_osl32k_qwen235b_pard_k9` | 8 | 16.38 |  | 11.67% | 2.05 |
| `verified_osl32k_qwen235b_suffix_k16` | 8 | 65.70 |  | 80.05% | 6.35 |
| `verified_osl32k_qwen235b_suffix_k8` | 8 | 66.12 |  | 81.39% | 5.43 |
| `verified_osl32k_qwen235b_eagle3_k11` | 16 | 55.13 |  | 15.43% | 2.70 |
| `verified_osl32k_qwen235b_eagle3_k9` | 16 | 59.01 |  | 18.10% | 2.63 |
| `verified_osl32k_qwen235b_pard2_k11` | 16 | 26.54 |  | 1.80% | 1.20 |
| `verified_osl32k_qwen235b_pard2_k9` | 16 | 27.50 |  | 1.78% | 1.16 |
| `verified_osl32k_qwen235b_pard_k11` | 16 | 32.50 |  | 10.99% | 2.21 |
| `verified_osl32k_qwen235b_pard_k9` | 16 | 33.72 |  | 13.76% | 2.24 |
| `verified_osl32k_qwen235b_suffix_k16` | 16 | 124.25 |  | 79.92% | 6.52 |
| `verified_osl32k_qwen235b_suffix_k8` | 16 | 105.89 |  | 79.69% | 5.38 |
| `verified_osl32k_qwen235b_eagle3_k11` | 32 | 114.66 |  | 15.95% | 2.75 |
| `verified_osl32k_qwen235b_eagle3_k9` | 32 | 116.16 |  | 19.23% | 2.73 |
| `verified_osl32k_qwen235b_pard2_k11` | 32 | 46.85 |  | 2.04% | 1.22 |
| `verified_osl32k_qwen235b_pard2_k9` | 32 | 48.05 |  | 3.18% | 1.29 |
| `verified_osl32k_qwen235b_pard_k11` | 32 | 66.37 |  | 12.63% | 2.39 |
| `verified_osl32k_qwen235b_pard_k9` | 32 | 64.59 |  | 18.18% | 2.64 |
| `verified_osl32k_qwen235b_suffix_k16` | 32 | 163.80 |  | 81.30% | 6.93 |
| `verified_osl32k_qwen235b_suffix_k8` | 32 | 170.08 |  | 82.01% | 5.77 |
