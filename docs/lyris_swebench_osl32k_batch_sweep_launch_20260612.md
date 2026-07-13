# Lyris SWE-Bench OSL32K Batch Sweep Launch - 2026-06-12

## Current measured coverage before this sweep

The existing OSL32K results did not cover the full batch-size matrix `2,4,8,16,32`.

| Dataset | Model group | Already measured OSL32K batches | Notes |
|---|---:|---:|---|
| SWE-Bench full | Qwen3-30B-A3B | 1, 2 | baseline, suffix K32, PARD K5 |
| SWE-Bench full | Qwen3-8B | 1, 2 | baseline, suffix K32, PARD-2 K5, Eagle-3 K3 |
| SWE-Bench-Verified | Qwen3-30B-A3B | 1, 2 | baseline, suffix K32/K5, PARD K5/K9/K11 |
| SWE-Bench-Verified | Qwen3-8B | 1, 2, partial 4 | batch 4 had baseline, suffix K5, Eagle-3; PARD-2 batch 4 was not in the extracted metrics |
| SWE-Bench-Verified | Qwen3-235B-A22B | 1 | suffix/PARD/PARD-2/Eagle pilot rows only |

## New Lyris sweep

Submitted 56 jobs through `experiments/eagle3_qwen3_235b/submit_lyris_swebench_osl32k_batch_sweep_20260612.sh`.

Tracker: `latest_lyris_swebench_osl32k_batch_sweep_20260612_jobs.csv`

Refresh path: `scripts/refresh_lyris_swebench_longosl_results.sh` now includes the new tracker by default.

| Dataset | Model group | Batches | Methods |
|---|---:|---:|---|
| SWE-Bench full | Qwen3-30B-A3B | 4, 8, 16, 32 | baseline, suffix K32, PARD K5 |
| SWE-Bench full | Qwen3-8B | 4, 8, 16, 32 | baseline, suffix K32, PARD-2 K5, Eagle-3 K3 |
| SWE-Bench-Verified | Qwen3-30B-A3B | 4, 8, 16, 32 | baseline, suffix K32, PARD K5 |
| SWE-Bench-Verified | Qwen3-8B | 4, 8, 16, 32 | baseline, suffix K32, PARD-2 K5, Eagle-3 K3 |

Each batch size is submitted as a separate vLLM job with `prompt_count=batch_size`, `max_num_seqs=batch_size`, `max_model_len=40960`, and `max_num_batched_tokens=40960*batch_size`.

Job ID range: `2109238` to `2109325`.

Initial queue snapshot after submission:

| Group | Count | State at snapshot |
|---|---:|---|
| SWE-Bench full batch sweep | 28 | running |
| SWE-Bench-Verified batch sweep | 28 | pending |

## Immediate retry note

The initial Qwen3-30B-A3B `bs32` rows used `max_num_batched_tokens=1310720` and failed during vLLM engine profiling with CUDA OOM:

| Dataset | Method | Failed job IDs |
|---|---|---|
| SWE-Bench full | baseline, suffix K32, PARD K5 | `2109288`, `2109290`, `2109291` |
| SWE-Bench-Verified | baseline, suffix K32, PARD K5 | `2109318`, `2109319`, `2109320` |

The wrapper was patched to use `max_num_batched_tokens=max(65536, ISL*batch_size)`, which is `131072` for `bs32`.

Low-cap retry tracker: `latest_lyris_swebench_osl32k_batch_sweep_qwen30_bs32_lowmbt_retry_20260612_jobs.csv`

Intended Qwen3-30B-A3B retry job IDs:

| Dataset | Method | Retry job ID |
|---|---|---:|
| SWE-Bench full | baseline | `2109375` |
| SWE-Bench full | suffix K32 | `2109376` |
| SWE-Bench full | PARD K5 | `2109380` |
| SWE-Bench-Verified | baseline | `2109389` |
| SWE-Bench-Verified | suffix K32 | `2109390` |
| SWE-Bench-Verified | PARD K5 | `2109391` |

The retry command also submitted duplicate Qwen3-8B `bs32` low-cap rows because the wrapper previously treated an empty `QWEN8_METHODS` override as unset. That override bug is patched. The duplicate Qwen3-8B low-cap rows are retained as useful fallback rows for `bs32`, since the original Qwen3-8B Eagle-3 `bs32` jobs failed quickly under the high-cap configuration.

Additional transient retry:

| Dataset | Model | Batch | Method | Failed job ID | Cause | Retry job ID |
|---|---|---:|---|---:|---|---:|
| SWE-Bench-Verified | Qwen3-30B-A3B | 16 | suffix K32 | `2109311` | `spank_sybil` credential/plugin failure before container launch | `2109447` |

Latest refresh artifacts:

- `docs/lyris_swebench_osl32k_batch_sweep_status_20260612.md`
- `docs/lyris_swebench_osl32k_batch_sweep_metrics_20260612.csv`

As of the refresh at `2026-06-12T23:31:29+02:00`, two new rows had completed:

| Dataset | Model | Batch | Method | tok/s/GPU | Acceptance |
|---|---|---:|---|---:|---:|
| SWE-Bench full | Qwen3-30B-A3B | 4 | suffix K32 | 595.65 | 89.94% |
| SWE-Bench full | Qwen3-8B | 4 | suffix K32 | 1089.84 | 94.04% |

## Refresh at 2026-06-12T23:35:54+02:00

Tracked rows: 71 total (`56` running, `6` completed metrics, `9` failed original attempts).

The failed original attempts are all covered by active replacement jobs:

| Failed cell | Failed job | Replacement |
|---|---:|---:|
| full Qwen3-30B-A3B bs32 baseline | `2109288` | `2109375` |
| full Qwen3-30B-A3B bs32 suffix K32 | `2109290` | `2109376` |
| full Qwen3-30B-A3B bs32 PARD K5 | `2109291` | `2109380` |
| full Qwen3-8B bs32 Eagle-3 K3 | `2109295` | `2109387` |
| verified Qwen3-30B-A3B bs32 baseline | `2109318` | `2109389` |
| verified Qwen3-30B-A3B bs32 suffix K32 | `2109319` | `2109390` |
| verified Qwen3-30B-A3B bs32 PARD K5 | `2109320` | `2109391` |
| verified Qwen3-30B-A3B bs16 suffix K32 | `2109311` | `2109447` |
| verified Qwen3-8B bs32 Eagle-3 K3 | `2109325` | `2109397` |

Completed metrics at this refresh:

| Dataset | Model | Batch | Method | tok/s/GPU | Acceptance |
|---|---|---:|---|---:|---:|
| SWE-Bench full | Qwen3-30B-A3B | 4 | suffix K32 | 595.65 | 89.94% |
| SWE-Bench full | Qwen3-30B-A3B | 8 | suffix K32 | 1010.10 | 89.97% |
| SWE-Bench full | Qwen3-8B | 4 | suffix K32 | 1089.84 | 94.04% |
| SWE-Bench-Verified | Qwen3-8B | 4 | suffix K32 | 846.22 | 92.46% |
| SWE-Bench-Verified | Qwen3-8B | 8 | suffix K32 | 1397.79 | 91.85% |
| SWE-Bench-Verified | Qwen3-8B | 16 | suffix K32 | 2597.43 | 92.13% |

## Refresh at 2026-06-12T23:59:08+02:00

The broader long-OSL refresh completed after making the collector robust to
non-UTF8 log bytes. Current artifacts:

- `docs/lyris_swebench_longosl_status_20260612.md`
- `docs/lyris_swebench_longosl_metrics_20260612.csv`
- `docs/lyris_swebench_longosl_live_progress_20260612.csv`

For OSL32K rows, the metrics CSV now contains 44 completed batch rows:

| Dataset | Model | Completed OSL32K metric rows | Current note |
|---|---|---:|---|
| SWE-Bench full | Qwen3-30B-A3B | 11 | Suffix K32 complete through bs32; baseline/PARD replacements are still running for several batches. |
| SWE-Bench full | Qwen3-8B | 19 | Baseline/eagle/suffix are complete through bs16, with suffix/eagle bs32 also complete. |
| SWE-Bench-Verified | Qwen3-30B-A3B | 3 | Suffix K32 complete for bs4/8/16; baselines and PARD rows still running/retrying. |
| SWE-Bench-Verified | Qwen3-8B | 11 | Baseline/eagle/suffix complete through bs16, with suffix/eagle bs32 complete. |
| SWE-Bench full | Qwen3-235B-A22B | 0 | All 25 Qwen3-235B full jobs are running. |
| SWE-Bench-Verified | Qwen3-235B-A22B | 0 | All 25 Qwen3-235B Verified jobs are running. |

Selected completed OSL32K batch results:

| Dataset | Model | Batch | Method | tok/s/GPU | Speedup | Acceptance |
|---|---|---:|---|---:|---:|---:|
| full | Qwen3-8B | 4 | suffix K32 | 1089.84 | 7.319x | 94.04% |
| full | Qwen3-8B | 8 | suffix K32 | 344.42 | 1.184x | 69.46% |
| full | Qwen3-8B | 16 | suffix K32 | 1512.85 | 2.500x | 87.48% |
| full | Qwen3-8B | 32 | suffix K32 | 2412.84 |  | 89.88% |
| full | Qwen3-8B | 32 | Eagle-3 K3 | 1350.12 |  | 64.58% |
| verified | Qwen3-8B | 4 | suffix K32 | 846.22 | 5.685x | 92.46% |
| verified | Qwen3-8B | 8 | suffix K32 | 1397.79 | 4.737x | 91.85% |
| verified | Qwen3-8B | 16 | suffix K32 | 2597.43 | 4.320x | 92.13% |
| verified | Qwen3-8B | 32 | suffix K32 | 2405.20 |  | 88.14% |
| full | Qwen3-30B-A3B | 32 | suffix K32 | 2280.54 |  | 88.53% |
| verified | Qwen3-30B-A3B | 16 | suffix K32 | 1801.36 |  | 89.47% |

Blank speedup cells mean the matching baseline for that same batch/model/dataset
has not completed yet.
