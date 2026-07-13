# OCI-HSG Qwen3-235B MATH500 Live History Summary

Source: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_qwen235b_math500_live_progress_history_20260613.csv`
CSV: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_qwen235b_math500_live_history_summary_20260613.csv`

This summarizes live vLLM logger snapshots only; it is not a replacement for final `breakdown.json` rows.

| method | job | snapshots | latest state | latest gen tok/s | mean gen tok/s | speedup latest/mean/min/max | draft accept latest/mean/min/max |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline` | `3288484` | 22 | `TIMEOUT` | 15.24 | 8.28 | 1.000/1.000/1.000/1.000 | /// |
| `eagle3_k11` | `3288927` | 22 | `COMPLETED` | 17.70 | 22.32 | 1.161/2.766/1.161/5.165 | 17.20/18.30/10.28/24.28 |
| `eagle3_k3` | `3288491` | 22 | `COMPLETED` | 25.62 | 28.75 | 1.681/3.564/1.681/6.598 | 69.96/70.05/57.86/80.30 |
| `eagle3_k9` | `3288926` | 22 | `COMPLETED` | 24.84 | 24.53 | 1.630/3.031/1.460/5.702 | 28.90/22.42/9.14/28.90 |
| `official_pard2_k1` | `3288490` | 22 | `TIMEOUT` | 11.02 | 6.55 | 0.723/0.794/0.633/0.964 | 1.44/3.20/0.00/13.30 |
| `official_pard2_k11` | `3288921` | 22 | `TIMEOUT` | 13.26 | 7.01 | 0.870/0.847/0.756/1.145 | 0.10/0.28/0.00/3.58 |
| `official_pard2_k9` | `3288922` | 22 | `TIMEOUT` | 13.18 | 6.95 | 0.865/0.840/0.723/1.129 | 0.22/0.34/0.00/3.68 |
| `pard_k11` | `3288919` | 22 | `COMPLETED` | 35.78 | 34.67 | 2.348/4.283/2.063/11.877 | 19.72/27.07/15.56/57.98 |
| `pard_k5` | `3288488` | 22 | `COMPLETED` | 16.40 | 25.48 | 1.076/3.184/1.076/6.684 | 36.76/49.25/35.42/92.24 |
| `pard_k9` | `3288918` | 22 | `COMPLETED` | 16.72 | 32.25 | 1.097/4.037/1.097/8.879 | 21.04/30.01/15.90/67.66 |
| `suffix_k32` | `3288487` | 22 | `FAILED` |  |  | /// | /// |
| `suffix_k32` | `3288594` | 22 | `COMPLETED` | 88.70 | 87.25 | 5.820/10.706/3.318/22.785 | 87.22/81.04/26.50/100.00 |
