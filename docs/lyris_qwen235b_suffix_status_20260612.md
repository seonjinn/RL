# Lyris Qwen3-235B SpecDec Status - 2026-06-12

Refreshed at `2026-06-12T22:04:00+02:00`.

Notes:

- `pard2_k*` jobs currently run the official PARD-2 checkpoint through vLLM's `draft_model` proposer path.
- `pard2_native_k*` jobs run `method=pard2` through the Lyris `sitecustomize.py` compatibility hook; stock vLLM still needs the equivalent source patch.

## Jobs

| Job | Label | Method | Shape | Queue | Accounting | Elapsed | Node |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `2101420` | `qwen235b_n16_isl4096_osl1024_baseline` | `baseline` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:57:24` | `lyris0239` |
| `2101421` | `qwen235b_n16_isl4096_osl1024_suffix_k32` | `suffix` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:39:27` | `lyris0214` |
| `2101504` | `qwen235b_n16_isl4096_osl1024_suffix_k1` | `suffix` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:42:09` | `lyris0027` |
| `2101505` | `qwen235b_n16_isl4096_osl1024_suffix_k2` | `suffix` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:39:34` | `lyris0009` |
| `2101506` | `qwen235b_n16_isl4096_osl1024_suffix_k4` | `suffix` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:38:51` | `lyris0241` |
| `2101507` | `qwen235b_n16_isl4096_osl1024_suffix_k8` | `suffix` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:39:23` | `lyris0249` |
| `2102005` | `qwen235b_n2_isl4096_osl32768_suffix_k1` | `suffix` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `01:53:23` | `lyris0180` |
| `2102006` | `qwen235b_n2_isl4096_osl32768_suffix_k2` | `suffix` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `01:26:11` | `lyris0087` |
| `2102008` | `qwen235b_n2_isl4096_osl32768_suffix_k4` | `suffix` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `01:06:09` | `lyris0090` |
| `2102009` | `qwen235b_n2_isl4096_osl32768_suffix_k8` | `suffix` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `00:46:18` | `lyris0239` |
| `2101422` | `qwen235b_n2_isl4096_osl32768_baseline` | `baseline` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:17:58` | `lyris0150` |
| `2101423` | `qwen235b_n2_isl4096_osl32768_suffix_k32` | `suffix` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `00:40:11` | `lyris0066` |
| `2101569` | `qwen235b_n16_isl4096_osl1024_pard_k3` | `pard` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:32:10` | `lyris0008` |
| `2101570` | `qwen235b_n16_isl4096_osl1024_pard_k5` | `pard` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:31:02` | `lyris0174` |
| `2101572` | `qwen235b_n16_isl4096_osl1024_pard2_k3` | `pard2` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:33:14` | `lyris0175` |
| `2101573` | `qwen235b_n16_isl4096_osl1024_pard2_k5` | `pard2` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:32:00` | `lyris0117` |
| `2101574` | `qwen235b_n16_isl4096_osl1024_pard2_native_k3` | `pard2` | `n16 ISL4096/OSL1024 BS1 2` | `` | `FAILED` | `00:00:36` | `lyris0126` |
| `2101575` | `qwen235b_n16_isl4096_osl1024_eagle3_k1` | `eagle3` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:36:04` | `lyris0243` |
| `2101576` | `qwen235b_n16_isl4096_osl1024_eagle3_k3` | `eagle3` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:29:40` | `lyris0126` |
| `2101577` | `qwen235b_n2_isl4096_osl32768_pard_k3` | `pard` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `02:24:50` | `lyris0088` |
| `2101578` | `qwen235b_n2_isl4096_osl32768_pard_k5` | `pard` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `02:21:23` | `lyris0271` |
| `2101579` | `qwen235b_n2_isl4096_osl32768_pard2_k3` | `pard2` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:41:13` | `lyris0272` |
| `2101580` | `qwen235b_n2_isl4096_osl32768_pard2_k5` | `pard2` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:38:38` | `lyris0273` |
| `2101581` | `qwen235b_n2_isl4096_osl32768_eagle3_k1` | `eagle3` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `01:50:16` | `lyris0275` |
| `2101582` | `qwen235b_n2_isl4096_osl32768_eagle3_k3` | `eagle3` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `01:19:28` | `lyris0276` |
| `2101640` | `qwen235b_n1_isl4096_osl256_pard2_native_k3` | `pard2` | `n1 ISL4096/OSL256 BS1` | `` | `FAILED` | `00:02:03` | `lyris0283` |
| `2101655` | `qwen235b_n1_isl4096_osl256_pard2_native_k3` | `pard2` | `n1 ISL4096/OSL256 BS1` | `` | `COMPLETED` | `00:02:53` | `lyris0221` |
| `2101759` | `qwen235b_n16_isl4096_osl1024_pard2_native_k3` | `pard2` | `n16 ISL4096/OSL1024 BS1 2` | `` | `COMPLETED` | `00:33:36` | `lyris0214` |
| `2101818` | `qwen235b_n2_isl4096_osl32768_pard2_native_k1` | `pard2` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:40:10` | `lyris0243` |
| `2101819` | `qwen235b_n2_isl4096_osl32768_pard2_native_k2` | `pard2` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:37:39` | `lyris0248` |
| `2101761` | `qwen235b_n2_isl4096_osl32768_pard2_native_k3` | `pard2` | `n2 ISL4096/OSL32768 BS1` | `` | `COMPLETED` | `03:35:36` | `lyris0216` |
| `2104334` | `qwen235b_n1_isl4096_osl65536_baseline` | `baseline` | `n1 ISL4096/OSL65536 BS1` | `` | `COMPLETED` | `04:25:01` | `lyris0162` |
| `2104335` | `qwen235b_n1_isl4096_osl65536_suffix_k32` | `suffix` | `n1 ISL4096/OSL65536 BS1` | `` | `COMPLETED` | `02:37:32` | `lyris0179` |
| `2104336` | `qwen235b_n1_isl4096_osl65536_suffix_k8` | `suffix` | `n1 ISL4096/OSL65536 BS1` | `` | `COMPLETED` | `02:23:03` | `lyris0006` |
| `2104337` | `qwen235b_n1_isl4096_osl65536_pard_k5` | `pard` | `n1 ISL4096/OSL65536 BS1` | `` | `FAILED` | `02:28:12` | `lyris0007` |
| `2104995` | `qwen235b_n1_isl4096_osl65536_pard_k3` | `pard` | `n1 ISL4096/OSL65536 BS1` | `` | `FAILED` | `02:26:29` | `lyris0220` |
| `2104338` | `qwen235b_n1_isl4096_osl65536_eagle3_k3` | `eagle3` | `n1 ISL4096/OSL65536 BS1` | `` | `COMPLETED` | `04:02:14` | `lyris0009` |
| `2104340` | `qwen235b_n1_isl4096_osl65536_pard2_native_k1` | `pard2` | `n1 ISL4096/OSL65536 BS1` | `` | `FAILED` | `02:48:50` | `lyris0013` |

## Live Progress

| Job | Label | Gen tok/s | Acceptance | Mean accept len | Completed rows | Tail |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `2101420` | `qwen235b_n16_isl4096_osl1024_baseline` | 16.2 |  |  | 2 | `ok` |
| `2101421` | `qwen235b_n16_isl4096_osl1024_suffix_k32` | 14.4 | 22.4 | 1.44 | 2 | `ok` |
| `2101504` | `qwen235b_n16_isl4096_osl1024_suffix_k1` | 22.3 | 28.4 | 1.28 | 2 | `ok` |
| `2101505` | `qwen235b_n16_isl4096_osl1024_suffix_k2` | 20.2 | 23.0 | 1.40 | 2 | `ok` |
| `2101506` | `qwen235b_n16_isl4096_osl1024_suffix_k4` | 24.4 | 19.5 | 1.45 | 2 | `ok` |
| `2101507` | `qwen235b_n16_isl4096_osl1024_suffix_k8` | 25.5 | 20.8 | 1.51 | 2 | `ok` |
| `2102005` | `qwen235b_n2_isl4096_osl32768_suffix_k1` | 14.4 | 100.0 | 2.00 | 1 | `ok` |
| `2102006` | `qwen235b_n2_isl4096_osl32768_suffix_k2` | 21.6 | 100.0 | 3.00 | 1 | `ok` |
| `2102008` | `qwen235b_n2_isl4096_osl32768_suffix_k4` | 35.5 | 99.0 | 4.96 | 1 | `ok` |
| `2102009` | `qwen235b_n2_isl4096_osl32768_suffix_k8` | 65.4 | 100.0 | 9.00 | 1 | `ok` |
| `2101422` | `qwen235b_n2_isl4096_osl32768_baseline` | 8.3 |  |  | 1 | `ok` |
| `2101423` | `qwen235b_n2_isl4096_osl32768_suffix_k32` | 101.4 | 100.0 | 13.00 | 1 | `ok` |
| `2101569` | `qwen235b_n16_isl4096_osl1024_pard_k3` | 24.5 | 30.8 | 1.92 | 2 | `ok` |
| `2101570` | `qwen235b_n16_isl4096_osl1024_pard_k5` | 31.0 | 25.9 | 2.29 | 2 | `ok` |
| `2101572` | `qwen235b_n16_isl4096_osl1024_pard2_k3` | 27.8 | 35.5 | 2.07 | 2 | `ok` |
| `2101573` | `qwen235b_n16_isl4096_osl1024_pard2_k5` | 34.2 | 30.9 | 2.54 | 2 | `ok` |
| `2101574` | `qwen235b_n16_isl4096_osl1024_pard2_native_k3` |  |  |  | 0 | `ok` |
| `2101575` | `qwen235b_n16_isl4096_osl1024_eagle3_k1` | 26.8 | 62.0 | 1.62 | 2 | `ok` |
| `2101576` | `qwen235b_n16_isl4096_osl1024_eagle3_k3` | 21.8 | 35.5 | 2.07 | 2 | `ok` |
| `2101577` | `qwen235b_n2_isl4096_osl32768_pard_k3` | 7.6 | 11.7 | 1.35 | 1 | `ok` |
| `2101578` | `qwen235b_n2_isl4096_osl32768_pard_k5` | 8.1 | 8.9 | 1.45 | 1 | `ok` |
| `2101579` | `qwen235b_n2_isl4096_osl32768_pard2_k3` | 5.8 | 1.2 | 1.04 | 1 | `ok` |
| `2101580` | `qwen235b_n2_isl4096_osl32768_pard2_k5` | 5.9 | 1.1 | 1.05 | 1 | `ok` |
| `2101581` | `qwen235b_n2_isl4096_osl32768_eagle3_k1` | 13.4 | 87.5 | 1.88 | 1 | `ok` |
| `2101582` | `qwen235b_n2_isl4096_osl32768_eagle3_k3` | 17.5 | 50.5 | 2.51 | 1 | `ok` |
| `2101640` | `qwen235b_n1_isl4096_osl256_pard2_native_k3` |  |  |  | 0 | `ok` |
| `2101655` | `qwen235b_n1_isl4096_osl256_pard2_native_k3` | 14.2 | 36.8 | 2.10 | 1 | `ok` |
| `2101759` | `qwen235b_n16_isl4096_osl1024_pard2_native_k3` | 27.9 | 35.5 | 2.07 | 2 | `ok` |
| `2101818` | `qwen235b_n2_isl4096_osl32768_pard2_native_k1` | 5.9 | 5.4 | 1.05 | 1 | `ok` |
| `2101819` | `qwen235b_n2_isl4096_osl32768_pard2_native_k2` | 6.0 | 3.6 | 1.07 | 1 | `ok` |
| `2101761` | `qwen235b_n2_isl4096_osl32768_pard2_native_k3` | 5.7 | 1.2 | 1.04 | 1 | `ok` |
| `2104334` | `qwen235b_n1_isl4096_osl65536_baseline` | 8.3 |  |  | 1 | `ok` |
| `2104335` | `qwen235b_n1_isl4096_osl65536_suffix_k32` | 7.1 | 16.0 | 1.80 | 1 | `ok` |
| `2104336` | `qwen235b_n1_isl4096_osl65536_suffix_k8` | 15.5 | 41.8 | 3.92 | 1 | `ok` |
| `2104337` | `qwen235b_n1_isl4096_osl65536_pard_k5` |  |  |  | 0 | `ok` |
| `2104995` | `qwen235b_n1_isl4096_osl65536_pard_k3` |  |  |  | 0 | `ok` |
| `2104338` | `qwen235b_n1_isl4096_osl65536_eagle3_k3` | 3.9 | 0.0 | 1.00 | 1 | `ok` |
| `2104340` | `qwen235b_n1_isl4096_osl65536_pard2_native_k1` |  |  |  | 0 | `ok` |

## Final Metrics

| Label | Batch | tok/s/GPU | Acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| `qwen235b_n16_isl4096_osl1024_baseline` | 1 | 2.04 |  |  |
| `qwen235b_n16_isl4096_osl1024_eagle3_k1` | 1 | 3.38 | 64.57% | 1.65 |
| `qwen235b_n16_isl4096_osl1024_eagle3_k3` | 1 | 4.23 | 38.98% | 2.17 |
| `qwen235b_n16_isl4096_osl1024_pard2_k3` | 1 | 3.70 | 38.86% | 2.17 |
| `qwen235b_n16_isl4096_osl1024_pard2_k5` | 1 | 3.86 | 26.22% | 2.31 |
| `qwen235b_n16_isl4096_osl1024_pard2_native_k3` | 1 | 3.68 | 38.86% | 2.17 |
| `qwen235b_n16_isl4096_osl1024_pard_k3` | 1 | 3.82 | 40.77% | 2.22 |
| `qwen235b_n16_isl4096_osl1024_pard_k5` | 1 | 4.00 | 27.96% | 2.40 |
| `qwen235b_n16_isl4096_osl1024_suffix_k1` | 1 | 2.83 | 34.90% | 1.35 |
| `qwen235b_n16_isl4096_osl1024_suffix_k2` | 1 | 2.98 | 29.10% | 1.45 |
| `qwen235b_n16_isl4096_osl1024_suffix_k32` | 1 | 3.01 | 23.82% | 1.46 |
| `qwen235b_n16_isl4096_osl1024_suffix_k4` | 1 | 3.06 | 26.51% | 1.49 |
| `qwen235b_n16_isl4096_osl1024_suffix_k8` | 1 | 3.01 | 23.96% | 1.46 |
| `qwen235b_n1_isl4096_osl256_pard2_native_k3` | 1 | 3.30 | 34.13% | 2.02 |
| `qwen235b_n1_isl4096_osl65536_baseline` | 1 | 2.09 |  |  |
| `qwen235b_n1_isl4096_osl65536_eagle3_k3` | 1 | 2.29 | 21.84% | 1.66 |
| `qwen235b_n1_isl4096_osl65536_suffix_k32` | 1 | 3.82 | 38.93% | 2.88 |
| `qwen235b_n1_isl4096_osl65536_suffix_k8` | 1 | 4.40 | 46.38% | 3.13 |
| `qwen235b_n2_isl4096_osl32768_baseline` | 1 | 2.09 |  |  |
| `qwen235b_n2_isl4096_osl32768_eagle3_k1` | 1 | 3.75 | 86.93% | 1.87 |
| `qwen235b_n2_isl4096_osl32768_eagle3_k3` | 1 | 5.19 | 54.28% | 2.63 |
| `qwen235b_n2_isl4096_osl32768_pard2_k3` | 1 | 1.87 | 4.64% | 1.14 |
| `qwen235b_n2_isl4096_osl32768_pard2_k5` | 1 | 1.89 | 2.86% | 1.14 |
| `qwen235b_n2_isl4096_osl32768_pard2_native_k1` | 1 | 1.88 | 11.53% | 1.12 |
| `qwen235b_n2_isl4096_osl32768_pard2_native_k2` | 1 | 1.90 | 6.65% | 1.13 |
| `qwen235b_n2_isl4096_osl32768_pard2_native_k3` | 1 | 1.91 | 4.64% | 1.14 |
| `qwen235b_n2_isl4096_osl32768_pard_k3` | 1 | 3.10 | 29.30% | 1.88 |
| `qwen235b_n2_isl4096_osl32768_pard_k5` | 1 | 3.19 | 18.39% | 1.92 |
| `qwen235b_n2_isl4096_osl32768_suffix_k1` | 1 | 3.78 | 82.10% | 1.82 |
| `qwen235b_n2_isl4096_osl32768_suffix_k2` | 1 | 5.13 | 77.92% | 2.48 |
| `qwen235b_n2_isl4096_osl32768_suffix_k32` | 1 | 12.60 | 82.63% | 6.28 |
| `qwen235b_n2_isl4096_osl32768_suffix_k4` | 1 | 7.01 | 75.43% | 3.42 |
| `qwen235b_n2_isl4096_osl32768_suffix_k8` | 1 | 11.65 | 86.35% | 5.89 |
| `qwen235b_n16_isl4096_osl1024_baseline` | 2 | 4.08 |  |  |
| `qwen235b_n16_isl4096_osl1024_eagle3_k1` | 2 | 6.68 | 64.61% | 1.65 |
| `qwen235b_n16_isl4096_osl1024_eagle3_k3` | 2 | 8.04 | 39.45% | 2.18 |
| `qwen235b_n16_isl4096_osl1024_pard2_k3` | 2 | 7.12 | 39.14% | 2.17 |
| `qwen235b_n16_isl4096_osl1024_pard2_k5` | 2 | 7.36 | 25.76% | 2.29 |
| `qwen235b_n16_isl4096_osl1024_pard2_native_k3` | 2 | 6.92 | 38.95% | 2.17 |
| `qwen235b_n16_isl4096_osl1024_pard_k3` | 2 | 7.44 | 41.62% | 2.25 |
| `qwen235b_n16_isl4096_osl1024_pard_k5` | 2 | 7.61 | 27.28% | 2.36 |
| `qwen235b_n16_isl4096_osl1024_suffix_k1` | 2 | 5.86 | 40.11% | 1.40 |
| `qwen235b_n16_isl4096_osl1024_suffix_k2` | 2 | 6.52 | 35.76% | 1.61 |
| `qwen235b_n16_isl4096_osl1024_suffix_k32` | 2 | 6.47 | 27.47% | 1.68 |
| `qwen235b_n16_isl4096_osl1024_suffix_k4` | 2 | 6.53 | 28.48% | 1.63 |
| `qwen235b_n16_isl4096_osl1024_suffix_k8` | 2 | 6.58 | 27.07% | 1.65 |
