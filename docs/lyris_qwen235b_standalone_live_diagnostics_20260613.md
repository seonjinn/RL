# Lyris Qwen3-235B Standalone Live Diagnostics - 2026-06-13

Refreshed at `2026-06-13T09:51:16+02:00`.

This report uses live vLLM logger telemetry from the current Lyris standalone sweep. Treat it as a health/performance signal only; final claims still require completed `breakdown.json` rows.

## Current Read

- SWE-Bench full Suffix: 10 live rows, 10 final rows, mean live gen 69.62 tok/s, mean acceptance 81.5%.
- SWE-Bench full PARD: 10 live rows, 3 final rows, mean live gen 100.35 tok/s, mean acceptance 8.0%.
- SWE-Bench full PARD-2: 10 live rows, 0 final rows, mean live gen 60.67 tok/s, mean acceptance 4.3%.
- SWE-Bench full Eagle-3: 10 live rows, 8 final rows, mean live gen 21.55 tok/s, mean acceptance 14.6%.
- SWE-Bench-Verified Suffix: 10 live rows, 10 final rows, mean live gen 98.47 tok/s, mean acceptance 98.2%.
- SWE-Bench-Verified PARD: 10 live rows, 0 final rows, mean live gen 115.56 tok/s, mean acceptance 7.6%.
- SWE-Bench-Verified PARD-2: 10 live rows, 0 final rows, mean live gen 111.49 tok/s, mean acceptance 4.7%.
- SWE-Bench-Verified Eagle-3: 10 live rows, 9 final rows, mean live gen 19.23 tok/s, mean acceptance 15.5%.

## Aggregate

| Dataset | Method | Jobs | Final rows | Running | Mean gen tok/s | Mean acceptance | Mean accept len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench full | `Suffix K8` | 5 | 5 | 0 | 43.66 | 67.0% | 6.03 |
| SWE-Bench full | `Suffix K16` | 5 | 5 | 0 | 95.58 | 96.0% | 12.47 |
| SWE-Bench full | `PARD K9` | 5 | 2 | 4 | 98.80 | 7.8% | 1.69 |
| SWE-Bench full | `PARD K11` | 5 | 1 | 4 | 101.90 | 8.2% | 1.90 |
| SWE-Bench full | `PARD-2 K9` | 5 | 0 | 5 | 59.22 | 4.6% | 1.42 |
| SWE-Bench full | `PARD-2 K11` | 5 | 0 | 5 | 62.12 | 4.1% | 1.45 |
| SWE-Bench full | `Eagle-3 K9` | 5 | 4 | 1 | 18.58 | 15.0% | 2.35 |
| SWE-Bench full | `Eagle-3 K11` | 5 | 4 | 1 | 24.52 | 14.2% | 2.56 |
| SWE-Bench-Verified | `Suffix K8` | 5 | 5 | 0 | 76.92 | 98.0% | 8.84 |
| SWE-Bench-Verified | `Suffix K16` | 5 | 5 | 0 | 120.02 | 98.4% | 12.77 |
| SWE-Bench-Verified | `PARD K9` | 5 | 0 | 5 | 101.74 | 6.9% | 1.62 |
| SWE-Bench-Verified | `PARD K11` | 5 | 0 | 5 | 129.38 | 8.3% | 1.91 |
| SWE-Bench-Verified | `PARD-2 K9` | 5 | 0 | 5 | 160.42 | 5.1% | 1.46 |
| SWE-Bench-Verified | `PARD-2 K11` | 5 | 0 | 5 | 62.56 | 4.3% | 1.48 |
| SWE-Bench-Verified | `Eagle-3 K9` | 5 | 5 | 0 | 21.76 | 18.0% | 2.62 |
| SWE-Bench-Verified | `Eagle-3 K11` | 5 | 4 | 1 | 16.70 | 13.1% | 2.44 |

## Per-Job Rows

| Dataset | Batch | Method | Job | Queue | Final | Live gen tok/s | Acceptance | Mean accept len | Tail |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| SWE-Bench full | 2 | `Suffix K8` | `2113100` | `COMPLETED` | 1 | 56.80 | 89.7% | 7.84 | `ok` |
| SWE-Bench full | 2 | `Suffix K16` | `2113112` | `COMPLETED` | 1 | 99.10 | 99.9% | 12.85 | `ok` |
| SWE-Bench full | 2 | `PARD K9` | `2113131` | `COMPLETED` | 1 | 8.30 | 5.3% | 1.47 | `ok` |
| SWE-Bench full | 2 | `PARD K11` | `2113146` | `COMPLETED` | 1 | 10.60 | 9.8% | 2.08 | `ok` |
| SWE-Bench full | 2 | `PARD-2 K9` | `2113164` | `RUNNING` | 0 | 27.60 | 11.0% | 1.99 | `ok` |
| SWE-Bench full | 2 | `PARD-2 K11` | `2113186` | `RUNNING` | 0 | 5.20 | 0.0% | 1.00 | `ok` |
| SWE-Bench full | 2 | `Eagle-3 K9` | `2113197` | `COMPLETED` | 1 | 12.90 | 11.1% | 2.00 | `ok` |
| SWE-Bench full | 2 | `Eagle-3 K11` | `2113211` | `COMPLETED` | 1 | 21.20 | 19.1% | 3.10 | `ok` |
| SWE-Bench full | 4 | `Suffix K8` | `2113101` | `COMPLETED` | 1 | 63.50 | 96.6% | 8.73 | `ok` |
| SWE-Bench full | 4 | `Suffix K16` | `2113116` | `COMPLETED` | 1 | 96.40 | 95.2% | 12.40 | `ok` |
| SWE-Bench full | 4 | `PARD K9` | `2113132` | `RUNNING` | 0 | 19.90 | 7.0% | 1.63 | `ok` |
| SWE-Bench full | 4 | `PARD K11` | `2113148` | `RUNNING` | 0 | 9.70 | 5.9% | 1.64 | `ok` |
| SWE-Bench full | 4 | `PARD-2 K9` | `2113167` | `RUNNING` | 0 | 32.10 | 2.3% | 1.21 | `ok` |
| SWE-Bench full | 4 | `PARD-2 K11` | `2113187` | `RUNNING` | 0 | 55.90 | 9.2% | 2.01 | `ok` |
| SWE-Bench full | 4 | `Eagle-3 K9` | `2113199` | `COMPLETED` | 1 | 18.40 | 18.4% | 2.66 | `ok` |
| SWE-Bench full | 4 | `Eagle-3 K11` | `2113214` | `COMPLETED` | 1 | 13.90 | 9.1% | 2.00 | `ok` |
| SWE-Bench full | 8 | `Suffix K8` | `2113103` | `COMPLETED` | 1 | 65.70 | 100.0% | 9.00 | `ok` |
| SWE-Bench full | 8 | `Suffix K16` | `2113117` | `COMPLETED` | 1 | 97.00 | 97.8% | 12.69 | `ok` |
| SWE-Bench full | 8 | `PARD K9` | `2113136` | `RUNNING` | 1 | 8.50 | 5.7% | 1.51 | `ok` |
| SWE-Bench full | 8 | `PARD K11` | `2113152` | `RUNNING` | 0 | 48.50 | 8.0% | 1.88 | `ok` |
| SWE-Bench full | 8 | `PARD-2 K9` | `2113170` | `RUNNING` | 0 | 85.30 | 6.4% | 1.57 | `ok` |
| SWE-Bench full | 8 | `PARD-2 K11` | `2113188` | `RUNNING` | 0 | 116.70 | 10.5% | 2.15 | `ok` |
| SWE-Bench full | 8 | `Eagle-3 K9` | `2113200` | `RUNNING` | 0 | 18.40 | 17.6% | 2.58 | `ok` |
| SWE-Bench full | 8 | `Eagle-3 K11` | `2113219` | `COMPLETED` | 1 | 17.50 | 14.2% | 2.57 | `ok` |
| SWE-Bench full | 16 | `Suffix K8` | `2113104` | `COMPLETED` | 1 | 20.40 | 32.4% | 2.85 | `ok` |
| SWE-Bench full | 16 | `Suffix K16` | `2113118` | `COMPLETED` | 1 | 92.30 | 93.0% | 12.16 | `ok` |
| SWE-Bench full | 16 | `PARD K9` | `2113138` | `RUNNING` | 0 | 124.30 | 7.3% | 1.65 | `ok` |
| SWE-Bench full | 16 | `PARD K11` | `2113154` | `RUNNING` | 0 | 143.80 | 8.1% | 1.89 | `ok` |
| SWE-Bench full | 16 | `PARD-2 K9` | `2113171` | `RUNNING` | 0 | 140.70 | 3.5% | 1.32 | `ok` |
| SWE-Bench full | 16 | `PARD-2 K11` | `2113189` | `RUNNING` | 0 | 13.20 | 0.3% | 1.03 | `ok` |
| SWE-Bench full | 16 | `Eagle-3 K9` | `2113201` | `COMPLETED` | 1 | 29.30 | 17.0% | 2.53 | `ok` |
| SWE-Bench full | 16 | `Eagle-3 K11` | `2113221` | `COMPLETED` | 1 | 31.50 | 12.2% | 2.34 | `ok` |
| SWE-Bench full | 32 | `Suffix K8` | `2113105` | `COMPLETED` | 1 | 11.90 | 16.5% | 1.75 | `ok` |
| SWE-Bench full | 32 | `Suffix K16` | `2113120` | `COMPLETED` | 1 | 93.10 | 94.1% | 12.23 | `ok` |
| SWE-Bench full | 32 | `PARD K9` | `2113139` | `RUNNING` | 0 | 333.00 | 13.5% | 2.21 | `ok` |
| SWE-Bench full | 32 | `PARD K11` | `2113156` | `RUNNING` | 0 | 296.90 | 9.1% | 2.01 | `ok` |
| SWE-Bench full | 32 | `PARD-2 K9` | `2113172` | `RUNNING` | 0 | 10.40 | 0.0% | 1.00 | `ok` |
| SWE-Bench full | 32 | `PARD-2 K11` | `2113190` | `RUNNING` | 0 | 119.60 | 0.3% | 1.04 | `ok` |
| SWE-Bench full | 32 | `Eagle-3 K9` | `2113202` | `COMPLETING` | 1 | 13.90 | 11.1% | 2.00 | `ok` |
| SWE-Bench full | 32 | `Eagle-3 K11` | `2113222` | `RUNNING` | 0 | 38.50 | 16.3% | 2.80 | `ok` |
| SWE-Bench-Verified | 2 | `Suffix K8` | `2113088` | `COMPLETED` | 1 | 65.30 | 100.0% | 9.00 | `ok` |
| SWE-Bench-Verified | 2 | `Suffix K16` | `2113106` | `COMPLETED` | 1 | 97.10 | 98.6% | 12.68 | `ok` |
| SWE-Bench-Verified | 2 | `PARD K9` | `2113121` | `RUNNING` | 0 | 18.30 | 5.2% | 1.47 | `ok` |
| SWE-Bench-Verified | 2 | `PARD K11` | `2113140` | `RUNNING` | 0 | 18.20 | 4.3% | 1.47 | `ok` |
| SWE-Bench-Verified | 2 | `PARD-2 K9` | `2113158` | `RUNNING` | 0 | 15.00 | 1.1% | 1.10 | `ok` |
| SWE-Bench-Verified | 2 | `PARD-2 K11` | `2113175` | `RUNNING` | 0 | 16.70 | 2.1% | 1.24 | `ok` |
| SWE-Bench-Verified | 2 | `Eagle-3 K9` | `2113191` | `COMPLETED` | 1 | 18.70 | 18.3% | 2.65 | `ok` |
| SWE-Bench-Verified | 2 | `Eagle-3 K11` | `2113204` | `COMPLETED` | 1 | 16.60 | 12.9% | 2.42 | `ok` |
| SWE-Bench-Verified | 4 | `Suffix K8` | `2113089` | `COMPLETED` | 1 | 66.00 | 100.0% | 9.00 | `ok` |
| SWE-Bench-Verified | 4 | `Suffix K16` | `2113107` | `COMPLETED` | 1 | 99.10 | 100.0% | 13.00 | `ok` |
| SWE-Bench-Verified | 4 | `PARD K9` | `2113124` | `RUNNING` | 0 | 14.50 | 2.5% | 1.23 | `ok` |
| SWE-Bench-Verified | 4 | `PARD K11` | `2113141` | `RUNNING` | 0 | 29.70 | 6.9% | 1.75 | `ok` |
| SWE-Bench-Verified | 4 | `PARD-2 K9` | `2113159` | `RUNNING` | 0 | 29.00 | 1.0% | 1.09 | `ok` |
| SWE-Bench-Verified | 4 | `PARD-2 K11` | `2113179` | `RUNNING` | 0 | 38.70 | 4.2% | 1.46 | `ok` |
| SWE-Bench-Verified | 4 | `Eagle-3 K9` | `2113192` | `COMPLETED` | 1 | 35.80 | 18.3% | 2.65 | `ok` |
| SWE-Bench-Verified | 4 | `Eagle-3 K11` | `2113205` | `COMPLETED` | 1 | 19.40 | 16.7% | 2.84 | `ok` |
| SWE-Bench-Verified | 8 | `Suffix K8` | `2113092` | `COMPLETED` | 1 | 128.60 | 99.6% | 8.97 | `ok` |
| SWE-Bench-Verified | 8 | `Suffix K16` | `2113108` | `COMPLETED` | 1 | 99.20 | 99.0% | 12.88 | `ok` |
| SWE-Bench-Verified | 8 | `PARD K9` | `2113126` | `RUNNING` | 0 | 39.30 | 5.0% | 1.45 | `ok` |
| SWE-Bench-Verified | 8 | `PARD K11` | `2113142` | `RUNNING` | 0 | 58.00 | 8.0% | 1.88 | `ok` |
| SWE-Bench-Verified | 8 | `PARD-2 K9` | `2113160` | `RUNNING` | 0 | 64.20 | 2.3% | 1.21 | `ok` |
| SWE-Bench-Verified | 8 | `PARD-2 K11` | `2113181` | `RUNNING` | 0 | 136.60 | 14.5% | 2.60 | `ok` |
| SWE-Bench-Verified | 8 | `Eagle-3 K9` | `2113193` | `COMPLETED` | 1 | 17.50 | 16.8% | 2.51 | `ok` |
| SWE-Bench-Verified | 8 | `Eagle-3 K11` | `2113207` | `COMPLETED` | 1 | 17.10 | 13.2% | 2.46 | `ok` |
| SWE-Bench-Verified | 16 | `Suffix K8` | `2113093` | `COMPLETED` | 1 | 59.80 | 90.2% | 8.22 | `ok` |
| SWE-Bench-Verified | 16 | `Suffix K16` | `2113109` | `COMPLETED` | 1 | 210.20 | 97.0% | 12.64 | `ok` |
| SWE-Bench-Verified | 16 | `PARD K9` | `2113127` | `RUNNING` | 0 | 90.90 | 7.4% | 1.67 | `ok` |
| SWE-Bench-Verified | 16 | `PARD K11` | `2113144` | `RUNNING` | 0 | 130.30 | 7.5% | 1.82 | `ok` |
| SWE-Bench-Verified | 16 | `PARD-2 K9` | `2113162` | `RUNNING` | 0 | 130.00 | 2.3% | 1.21 | `ok` |
| SWE-Bench-Verified | 16 | `PARD-2 K11` | `2113182` | `RUNNING` | 0 | 16.70 | 0.3% | 1.04 | `ok` |
| SWE-Bench-Verified | 16 | `Eagle-3 K9` | `2113194` | `COMPLETED` | 1 | 18.00 | 17.4% | 2.56 | `ok` |
| SWE-Bench-Verified | 16 | `Eagle-3 K11` | `2113209` | `RUNNING` | 0 | 13.80 | 9.1% | 2.00 | `ok` |
| SWE-Bench-Verified | 32 | `Suffix K8` | `2113095` | `COMPLETED` | 1 | 64.90 | 100.0% | 9.00 | `ok` |
| SWE-Bench-Verified | 32 | `Suffix K16` | `2113110` | `COMPLETED` | 1 | 94.50 | 97.2% | 12.65 | `ok` |
| SWE-Bench-Verified | 32 | `PARD K9` | `2113128` | `RUNNING` | 0 | 345.70 | 14.3% | 2.28 | `ok` |
| SWE-Bench-Verified | 32 | `PARD K11` | `2113145` | `RUNNING` | 0 | 410.70 | 15.0% | 2.65 | `ok` |
| SWE-Bench-Verified | 32 | `PARD-2 K9` | `2113163` | `RUNNING` | 0 | 563.90 | 18.6% | 2.68 | `ok` |
| SWE-Bench-Verified | 32 | `PARD-2 K11` | `2113183` | `RUNNING` | 0 | 104.10 | 0.4% | 1.05 | `ok` |
| SWE-Bench-Verified | 32 | `Eagle-3 K9` | `2113196` | `COMPLETED` | 1 | 18.80 | 19.0% | 2.71 | `ok` |
| SWE-Bench-Verified | 32 | `Eagle-3 K11` | `2113210` | `COMPLETED` | 1 | 16.60 | 13.4% | 2.47 | `ok` |

Source CSV: `/Users/sna/Nemo-RL_Qwen3_Roadmap/docs/lyris_qwen235b_standalone_fast_20260613_live_progress.csv`
