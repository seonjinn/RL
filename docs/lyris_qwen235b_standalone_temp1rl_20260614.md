# Lyris Qwen3-235B Standalone Refresh - lyris_qwen235b_standalone_temp1rl_20260614

Refreshed at `2026-06-15T23:17:17-07:00`.

## Queue State

| State | Count |
| --- | ---: |
| `COMPLETED` | 93 |
| `TIMEOUT` | 2 |

## Suite State

| Suite | State | Count |
| --- | --- | ---: |
| math500 | `COMPLETED` | 3 |
| math500 | `TIMEOUT` | 2 |
| swe | `COMPLETED` | 90 |

## Live Telemetry Averages

Live values are parsed from recent Slurm logs and are not final benchmark rows.

| Dataset | Method | Live rows | Avg gen tok/s | Avg acceptance | Avg accept len |
| --- | --- | ---: | ---: | ---: | ---: |
| full | `baseline` | 5 | 99.4 |  |  |
| full | `eagle3_k11` | 5 | 12.3 | 7.6% | 1.83 |
| full | `eagle3_k9` | 5 | 10.5 | 6.2% | 1.56 |
| full | `pard2_k11` | 5 | 5.4 | 0.0% | 1.00 |
| full | `pard2_k9` | 5 | 6.1 | 0.0% | 1.00 |
| full | `pard_k11` | 5 | 7.0 | 2.3% | 1.26 |
| full | `pard_k9` | 5 | 12.8 | 12.3% | 2.11 |
| full | `suffix_k16` | 5 | 12.6 | 29.0% | 1.81 |
| full | `suffix_k8` | 5 | 27.1 | 41.4% | 3.29 |
| math500 | `baseline` | 1 | 8.3 |  |  |
| math500 | `eagle3_k3` | 1 | 12.5 | 25.4% | 1.76 |
| math500 | `official_pard2_k1` | 1 | 8.6 | 24.3% | 1.24 |
| math500 | `pard_k5` | 1 | 9.3 | 12.4% | 1.62 |
| math500 | `suffix_k32` | 1 | 15.3 | 29.2% | 2.23 |
| verified | `baseline` | 5 | 101.3 |  |  |
| verified | `eagle3_k11` | 5 | 26.7 | 26.4% | 3.91 |
| verified | `eagle3_k9` | 5 | 9.5 | 4.2% | 1.38 |
| verified | `pard2_k11` | 5 | 5.6 | 0.1% | 1.02 |
| verified | `pard2_k9` | 5 | 6.3 | 0.1% | 1.01 |
| verified | `pard_k11` | 5 | 6.2 | 1.0% | 1.11 |
| verified | `pard_k9` | 5 | 6.0 | 1.0% | 1.09 |
| verified | `suffix_k16` | 5 | 13.4 | 24.0% | 1.64 |
| verified | `suffix_k8` | 5 | 14.2 | 26.0% | 2.00 |

## Completed Breakdown Rows

Completed `breakdown.json` files found: `93`.

| Label | Batch | tok/s/GPU | Speedup | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: | ---: |
| `math500_osl32k_qwen235b_eagle3_k3` | 1 | 4.48 |  | 43.08% | 2.29 |
| `math500_osl32k_qwen235b_pard_k5` | 1 | 4.54 |  | 34.70% | 2.74 |
| `math500_osl32k_qwen235b_suffix_k32` | 1 | 7.55 |  | 59.25% | 3.73 |
| `math500_osl32k_qwen235b_eagle3_k3` | 2 | 7.99 |  | 42.04% | 2.26 |
| `math500_osl32k_qwen235b_pard_k5` | 2 | 7.20 |  | 28.84% | 2.44 |
| `math500_osl32k_qwen235b_suffix_k32` | 2 | 23.18 |  | 73.36% | 6.52 |
| `full_osl32k_qwen235b_baseline` | 2 | 4.13 | 1.000 | % |  |
| `full_osl32k_qwen235b_eagle3_k11` | 2 | 5.09 | 1.233 | 6.06% | 1.67 |
| `full_osl32k_qwen235b_eagle3_k9` | 2 | 5.16 | 1.251 | 9.49% | 1.85 |
| `full_osl32k_qwen235b_pard2_k11` | 2 | 3.32 | 0.803 | 0.74% | 1.08 |
| `full_osl32k_qwen235b_pard2_k9` | 2 | 3.61 | 0.874 | 1.67% | 1.15 |
| `full_osl32k_qwen235b_pard_k11` | 2 | 5.88 | 1.424 | 8.41% | 1.92 |
| `full_osl32k_qwen235b_pard_k9` | 2 | 5.74 | 1.392 | 11.60% | 2.04 |
| `full_osl32k_qwen235b_suffix_k16` | 2 | 14.04 | 3.402 | 73.90% | 4.71 |
| `full_osl32k_qwen235b_suffix_k8` | 2 | 15.64 | 3.790 | 72.63% | 4.14 |
| `full_osl32k_qwen235b_baseline` | 4 | 8.35 | 1.000 | % |  |
| `full_osl32k_qwen235b_eagle3_k11` | 4 | 11.05 | 1.324 | 8.03% | 1.88 |
| `full_osl32k_qwen235b_eagle3_k9` | 4 | 11.65 | 1.395 | 10.29% | 1.93 |
| `full_osl32k_qwen235b_pard2_k11` | 4 | 7.03 | 0.843 | 1.79% | 1.20 |
| `full_osl32k_qwen235b_pard2_k9` | 4 | 7.22 | 0.865 | 1.68% | 1.15 |
| `full_osl32k_qwen235b_pard_k11` | 4 | 9.26 | 1.110 | 8.38% | 1.92 |
| `full_osl32k_qwen235b_pard_k9` | 4 | 7.95 | 0.952 | 5.87% | 1.53 |
| `full_osl32k_qwen235b_suffix_k16` | 4 | 14.04 | 1.682 | 50.72% | 2.66 |
| `full_osl32k_qwen235b_suffix_k8` | 4 | 12.29 | 1.473 | 46.32% | 2.36 |
| `full_osl32k_qwen235b_baseline` | 8 | 16.07 | 1.000 | % |  |
| `full_osl32k_qwen235b_eagle3_k11` | 8 | 17.28 | 1.076 | 6.11% | 1.67 |
| `full_osl32k_qwen235b_eagle3_k9` | 8 | 21.57 | 1.343 | 9.62% | 1.87 |
| `full_osl32k_qwen235b_pard2_k11` | 8 | 13.65 | 0.849 | 1.40% | 1.15 |
| `full_osl32k_qwen235b_pard2_k9` | 8 | 14.09 | 0.877 | 1.79% | 1.16 |
| `full_osl32k_qwen235b_pard_k11` | 8 | 19.37 | 1.206 | 9.11% | 2.00 |
| `full_osl32k_qwen235b_pard_k9` | 8 | 15.92 | 0.991 | 9.77% | 1.88 |
| `full_osl32k_qwen235b_suffix_k16` | 8 | 23.70 | 1.475 | 54.53% | 2.95 |
| `full_osl32k_qwen235b_suffix_k8` | 8 | 22.78 | 1.418 | 54.18% | 2.72 |
| `full_osl32k_qwen235b_baseline` | 16 | 31.77 | 1.000 | % |  |
| `full_osl32k_qwen235b_eagle3_k11` | 16 | 37.73 | 1.188 | 8.84% | 1.97 |
| `full_osl32k_qwen235b_eagle3_k9` | 16 | 32.37 | 1.019 | 9.03% | 1.81 |
| `full_osl32k_qwen235b_pard2_k11` | 16 | 26.44 | 0.832 | 1.04% | 1.11 |
| `full_osl32k_qwen235b_pard2_k9` | 16 | 26.12 | 0.822 | 1.66% | 1.15 |
| `full_osl32k_qwen235b_pard_k11` | 16 | 28.96 | 0.912 | 5.96% | 1.66 |
| `full_osl32k_qwen235b_pard_k9` | 16 | 28.21 | 0.888 | 6.31% | 1.57 |
| `full_osl32k_qwen235b_suffix_k16` | 16 | 47.27 | 1.488 | 52.01% | 2.81 |
| `full_osl32k_qwen235b_suffix_k8` | 16 | 47.98 | 1.510 | 42.72% | 2.20 |
| `full_osl32k_qwen235b_baseline` | 32 | 64.92 | 1.000 | % |  |
| `full_osl32k_qwen235b_eagle3_k11` | 32 | 64.47 | 0.993 | 6.63% | 1.73 |
| `full_osl32k_qwen235b_eagle3_k9` | 32 | 77.20 | 1.189 | 9.44% | 1.85 |
| `full_osl32k_qwen235b_pard2_k11` | 32 | 47.36 | 0.730 | 1.34% | 1.15 |
| `full_osl32k_qwen235b_pard2_k9` | 32 | 47.37 | 0.730 | 2.01% | 1.18 |
| `full_osl32k_qwen235b_pard_k11` | 32 | 49.46 | 0.762 | 5.81% | 1.64 |
| `full_osl32k_qwen235b_pard_k9` | 32 | 49.81 | 0.767 | 7.03% | 1.63 |
| `full_osl32k_qwen235b_suffix_k16` | 32 | 90.18 | 1.389 | 48.92% | 2.70 |
| `full_osl32k_qwen235b_suffix_k8` | 32 | 94.29 | 1.453 | 47.24% | 2.50 |
| `verified_osl32k_qwen235b_baseline` | 2 | 4.12 | 1.000 | % |  |
| `verified_osl32k_qwen235b_eagle3_k11` | 2 | 5.97 | 1.449 | 7.40% | 1.81 |
| `verified_osl32k_qwen235b_eagle3_k9` | 2 | 5.99 | 1.453 | 10.87% | 1.98 |
| `verified_osl32k_qwen235b_pard2_k11` | 2 | 3.54 | 0.859 | 0.86% | 1.10 |
| `verified_osl32k_qwen235b_pard2_k9` | 2 | 3.56 | 0.864 | 1.01% | 1.09 |
| `verified_osl32k_qwen235b_pard_k11` | 2 | 4.32 | 1.049 | 6.15% | 1.68 |
| `verified_osl32k_qwen235b_pard_k9` | 2 | 4.51 | 1.095 | 5.32% | 1.48 |
| `verified_osl32k_qwen235b_suffix_k16` | 2 | 7.38 | 1.791 | 38.89% | 2.05 |
| `verified_osl32k_qwen235b_suffix_k8` | 2 | 9.65 | 2.341 | 61.05% | 3.12 |
| `verified_osl32k_qwen235b_baseline` | 4 | 8.40 | 1.000 | % |  |
| `verified_osl32k_qwen235b_eagle3_k11` | 4 | 11.08 | 1.319 | 9.21% | 2.01 |
| `verified_osl32k_qwen235b_eagle3_k9` | 4 | 9.24 | 1.099 | 7.54% | 1.68 |
| `verified_osl32k_qwen235b_pard2_k11` | 4 | 6.85 | 0.815 | 1.40% | 1.15 |
| `verified_osl32k_qwen235b_pard2_k9` | 4 | 7.04 | 0.838 | 1.20% | 1.11 |
| `verified_osl32k_qwen235b_pard_k11` | 4 | 7.95 | 0.946 | 4.58% | 1.50 |
| `verified_osl32k_qwen235b_pard_k9` | 4 | 8.28 | 0.986 | 5.91% | 1.53 |
| `verified_osl32k_qwen235b_suffix_k16` | 4 | 16.15 | 1.922 | 46.82% | 2.36 |
| `verified_osl32k_qwen235b_suffix_k8` | 4 | 14.32 | 1.704 | 52.51% | 2.63 |
| `verified_osl32k_qwen235b_baseline` | 8 | 16.03 | 1.000 | % |  |
| `verified_osl32k_qwen235b_eagle3_k11` | 8 | 22.70 | 1.416 | 8.25% | 1.91 |
| `verified_osl32k_qwen235b_eagle3_k9` | 8 | 19.44 | 1.212 | 8.98% | 1.81 |
| `verified_osl32k_qwen235b_pard2_k11` | 8 | 13.44 | 0.838 | 1.32% | 1.14 |
| `verified_osl32k_qwen235b_pard2_k9` | 8 | 13.64 | 0.851 | 3.30% | 1.30 |
| `verified_osl32k_qwen235b_pard_k11` | 8 | 15.15 | 0.945 | 4.91% | 1.54 |
| `verified_osl32k_qwen235b_pard_k9` | 8 | 15.89 | 0.991 | 6.83% | 1.61 |
| `verified_osl32k_qwen235b_suffix_k16` | 8 | 26.15 | 1.631 | 48.78% | 2.58 |
| `verified_osl32k_qwen235b_suffix_k8` | 8 | 25.54 | 1.593 | 46.25% | 2.35 |
| `verified_osl32k_qwen235b_baseline` | 16 | 32.74 | 1.000 | % |  |
| `verified_osl32k_qwen235b_eagle3_k11` | 16 | 38.21 | 1.167 | 8.03% | 1.88 |
| `verified_osl32k_qwen235b_eagle3_k9` | 16 | 34.35 | 1.049 | 7.29% | 1.66 |
| `verified_osl32k_qwen235b_pard2_k11` | 16 | 26.40 | 0.806 | 1.39% | 1.15 |
| `verified_osl32k_qwen235b_pard2_k9` | 16 | 25.96 | 0.793 | 1.23% | 1.11 |
| `verified_osl32k_qwen235b_pard_k11` | 16 | 27.38 | 0.836 | 5.36% | 1.59 |
| `verified_osl32k_qwen235b_pard_k9` | 16 | 30.53 | 0.932 | 6.43% | 1.58 |
| `verified_osl32k_qwen235b_suffix_k16` | 16 | 41.98 | 1.282 | 49.12% | 2.63 |
| `verified_osl32k_qwen235b_suffix_k8` | 16 | 41.48 | 1.267 | 44.40% | 2.26 |
| `verified_osl32k_qwen235b_baseline` | 32 | 64.73 | 1.000 | % |  |
| `verified_osl32k_qwen235b_eagle3_k11` | 32 | 73.26 | 1.132 | 6.92% | 1.76 |
| `verified_osl32k_qwen235b_eagle3_k9` | 32 | 69.60 | 1.075 | 8.57% | 1.77 |
| `verified_osl32k_qwen235b_pard2_k11` | 32 | 49.20 | 0.760 | 1.66% | 1.18 |
| `verified_osl32k_qwen235b_pard2_k9` | 32 | 48.53 | 0.750 | 2.24% | 1.20 |
| `verified_osl32k_qwen235b_pard_k11` | 32 | 53.65 | 0.829 | 5.66% | 1.62 |
| `verified_osl32k_qwen235b_pard_k9` | 32 | 51.34 | 0.793 | 6.47% | 1.58 |
| `verified_osl32k_qwen235b_suffix_k16` | 32 | 85.77 | 1.325 | 45.34% | 2.45 |
| `verified_osl32k_qwen235b_suffix_k8` | 32 | 82.46 | 1.274 | 49.96% | 2.59 |
