# Lyris Qwen3 MATH500 OSL32K SpecDec Refresh - 2026-06-12

Refreshed at `2026-06-12T22:02:42+02:00`.

## Queue State

| State | Count |
| --- | ---: |
| `COMPLETED` | 7 |
| `TIMEOUT` | 2 |

## Jobs

| Job | Model | Method | Queue | Accounting | Elapsed | Node/Reason |
| ---: | --- | --- | --- | --- | --- | --- |
| 2107295 | `qwen8` | `baseline` | `` | `COMPLETED` | 01:59:58 | lyris0079 |
| 2107332 | `qwen30` | `baseline` | `` | `COMPLETED` | 03:38:48 | lyris0178 |
| 2107297 | `qwen8` | `eagle3_k3` | `` | `COMPLETED` | 01:06:00 | lyris0219 |
| 2107298 | `qwen8` | `official_pard2_k3` | `` | `TIMEOUT` | 03:00:20 | lyris0255 |
| 2107302 | `qwen8` | `official_pard2_k5` | `` | `TIMEOUT` | 03:00:05 | lyris0055 |
| 2107335 | `qwen30` | `pard_k3` | `` | `COMPLETED` | 02:02:57 | lyris0108 |
| 2107334 | `qwen30` | `pard_k5` | `` | `COMPLETED` | 02:02:54 | lyris0085 |
| 2107296 | `qwen8` | `suffix_k32` | `` | `COMPLETED` | 00:21:15 | lyris0038 |
| 2107333 | `qwen30` | `suffix_k32` | `` | `COMPLETED` | 00:31:36 | lyris0029 |

## Completed Metrics

| Model | Method | Batch | tok/s/GPU | Speedup | Acceptance | Mean accept len |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen30` | `baseline` | 1 | 20.11 | 1.000x |  |  |
| `qwen30` | `pard_k3` | 1 | 37.20 | 1.849x | 70.67% | 3.12 |
| `qwen30` | `pard_k5` | 1 | 42.33 | 2.105x | 52.15% | 3.61 |
| `qwen30` | `suffix_k32` | 1 | 146.74 | 7.296x | 88.43% | 7.85 |
| `qwen8` | `baseline` | 1 | 36.79 | 1.000x |  |  |
| `qwen8` | `eagle3_k3` | 1 | 74.38 | 2.021x | 62.94% | 2.89 |
| `qwen8` | `official_pard2_k3` | 1 | 15.99 | 0.435x | 0.07% | 1.00 |
| `qwen8` | `official_pard2_k5` | 1 | 16.10 | 0.438x | 0.04% | 1.00 |
| `qwen8` | `suffix_k32` | 1 | 216.02 | 5.871x | 84.51% | 6.88 |
| `qwen30` | `baseline` | 2 | 40.12 | 1.000x |  |  |
| `qwen30` | `pard_k3` | 2 | 68.87 | 1.716x | 71.95% | 3.16 |
| `qwen30` | `pard_k5` | 2 | 61.57 | 1.535x | 43.45% | 3.17 |
| `qwen30` | `suffix_k32` | 2 | 303.81 | 7.572x | 92.52% | 9.81 |
| `qwen8` | `baseline` | 2 | 73.79 | 1.000x |  |  |
| `qwen8` | `eagle3_k3` | 2 | 118.53 | 1.606x | 63.62% | 2.91 |
| `qwen8` | `suffix_k32` | 2 | 469.83 | 6.367x | 91.24% | 9.48 |
