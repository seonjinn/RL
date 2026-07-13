# OCI-HSG Qwen3-235B MATH500 Live Progress

CSV: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/oci_qwen235b_math500_live_progress_20260613.csv`

Live numbers are parsed from recent vLLM logger lines, not final `breakdown.json` rows. Sample columns show how many matching lines contributed to each average.

| job | method | state | gen tok/s | gen samples | speedup vs baseline | accept len | accepted tok/s | drafted tok/s | draft accept | spec samples |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3288484 | `baseline` | `TIMEOUT` | 15.24 | 5 | 1.000 |  |  |  |  | 0 |
| 3288487 | `suffix_k32` | `FAILED` |  | 0 |  |  |  |  |  | 0 |
| 3288488 | `pard_k5` | `COMPLETED` | 16.40 | 5 | 1.076 | 2.84 | 10.61 | 28.85 | 36.76% | 5 |
| 3288490 | `official_pard2_k1` | `TIMEOUT` | 11.02 | 5 | 0.723 | 1.02 | 0.16 | 10.87 | 1.44% | 5 |
| 3288491 | `eagle3_k3` | `COMPLETED` | 25.62 | 5 | 1.681 | 3.10 | 17.41 | 24.81 | 69.96% | 5 |
| 3288594 | `suffix_k32` | `COMPLETED` | 88.70 | 5 | 5.820 | 11.25 | 80.82 | 92.64 | 87.22% | 5 |
| 3288918 | `pard_k9` | `COMPLETED` | 16.72 | 5 | 1.097 | 2.89 | 10.94 | 51.99 | 21.04% | 5 |
| 3288922 | `official_pard2_k9` | `TIMEOUT` | 13.18 | 5 | 0.865 | 1.02 | 0.26 | 116.33 | 0.22% | 5 |
| 3288926 | `eagle3_k9` | `COMPLETED` | 24.84 | 5 | 1.630 | 3.60 | 17.92 | 62.00 | 28.90% | 5 |
| 3288919 | `pard_k11` | `COMPLETED` | 35.78 | 5 | 2.348 | 3.17 | 24.51 | 124.20 | 19.72% | 5 |
| 3288921 | `official_pard2_k11` | `TIMEOUT` | 13.26 | 5 | 0.870 | 1.01 | 0.14 | 144.50 | 0.10% | 5 |
| 3288927 | `eagle3_k11` | `COMPLETED` | 17.70 | 5 | 1.161 | 2.89 | 11.59 | 67.34 | 17.20% | 5 |
