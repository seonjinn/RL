# Qwen3-235B SWE-Bench OSL32K PARD Live Snapshot - 2026-06-13

These PARD/PARD-2 rows are live telemetry only. They are not final benchmark results because every PARD/PARD-2 cell still had `completed_batch_rows=0` in the latest local refresh, so no final `breakdown.json` rows were available.

| Dataset | Batch | Method | Job | State | Live tok/s/GPU | Live baseline tok/s/GPU | Live speedup | Live acceptance | Mean accept len | Final rows |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-Bench full | 2 | PARD-2 K1 | `2109552` | `RUNNING` | 5.08 | 4.12 | 1.23x | 48.6% | 1.49 | 0 |
| SWE-Bench full | 2 | PARD K5 | `2109551` | `RUNNING` | 1.68 | 4.12 | 0.41x | 5.8% | 1.29 | 0 |
| SWE-Bench full | 4 | PARD-2 K1 | `2109557` | `RUNNING` | 8.32 | 8.15 | 1.02x | 25.4% | 1.25 | 0 |
| SWE-Bench full | 4 | PARD K5 | `2109556` | `RUNNING` | 8.57 | 8.15 | 1.05x | 14.1% | 1.71 | 0 |
| SWE-Bench full | 8 | PARD-2 K1 | `2109564` | `RUNNING` | 21.82 | 16.40 | 1.33x | 63.2% | 1.63 | 0 |
| SWE-Bench full | 8 | PARD K5 | `2109563` | `RUNNING` | 18.32 | 16.40 | 1.12x | 21.6% | 2.08 | 0 |
| SWE-Bench full | 16 | PARD-2 K1 | `2109574` | `RUNNING` | 42.20 | 32.20 | 1.31x | 59.9% | 1.60 | 0 |
| SWE-Bench full | 16 | PARD K5 | `2109572` | `RUNNING` | 20.27 | 32.20 | 0.63x | 8.2% | 1.41 | 0 |
| SWE-Bench full | 32 | PARD-2 K1 | `2109579` | `RUNNING` | 15.55 | 64.53 | 0.24x | 4.8% | 1.05 | 0 |
| SWE-Bench full | 32 | PARD K5 | `2109578` | `RUNNING` | 118.15 | 64.53 | 1.83x | 26.2% | 2.31 | 0 |
| SWE-Bench-Verified | 2 | PARD-2 K1 | `2109521` | `RUNNING` | 3.60 | 4.17 | 0.86x | 5.8% | 1.06 | 0 |
| SWE-Bench-Verified | 2 | PARD K5 | `2109519` | `RUNNING` | 4.17 | 4.17 | 1.00x | 10.2% | 1.51 | 0 |
| SWE-Bench-Verified | 4 | PARD-2 K1 | `2109530` | `RUNNING` | 8.03 | 8.25 | 0.97x | 17.8% | 1.18 | 0 |
| SWE-Bench-Verified | 4 | PARD K5 | `2109528` | `RUNNING` | 6.75 | 8.25 | 0.82x | 10.6% | 1.53 | 0 |
| SWE-Bench-Verified | 8 | PARD-2 K1 | `2109536` | `RUNNING` | 17.65 | 16.43 | 1.07x | 33.4% | 1.33 | 0 |
| SWE-Bench-Verified | 8 | PARD K5 | `2109535` | `RUNNING` | 9.95 | 16.43 | 0.61x | 13.0% | 1.65 | 0 |
| SWE-Bench-Verified | 16 | PARD-2 K1 | `2109541` | `RUNNING` | 37.42 | 33.23 | 1.13x | 39.5% | 1.40 | 0 |
| SWE-Bench-Verified | 16 | PARD K5 | `2109540` | `RUNNING` | 38.95 | 33.23 | 1.17x | 17.6% | 1.88 | 0 |
| SWE-Bench-Verified | 32 | PARD-2 K1 | `2109546` | `RUNNING` | 91.20 | 64.38 | 1.42x | 75.5% | 1.76 | 0 |
| SWE-Bench-Verified | 32 | PARD K5 | `2109545` | `RUNNING` | 91.42 | 64.38 | 1.42x | 22.6% | 2.13 | 0 |

Interpretation: use this only as a running-job health signal. The final table should be populated from completed `breakdown.json` files once these jobs finish.
