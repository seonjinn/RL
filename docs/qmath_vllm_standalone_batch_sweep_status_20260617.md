# qmath vLLM standalone batch sweep status

Updated: 2026-06-18 15:02 

## Scope

- Cluster: OCI-HSG (`oci-hsg-cs-001-vscode-02`)
- Account/partition: `nemotron_n3_post` / `batch_long`
- Sweep: qmath MATH-500, batch sizes 4/8/16/32, temperatures 0.0 and 1.0
- Models: Qwen3-235B-A22B, Qwen3-30B-A3B, Qwen3-8B
- Methods: baseline, suffix K=32, PARD K=5, PARD-2 where configured, Eagle-3 K=3
- Shape: ISL=4096, OSL=32768, max_model_len=40960, max_num_batched_tokens=131072
- Batch 1/2 rows are from earlier Lyris/OCI runs and are not part of this 112-job OCI-HSG sweep.

## Slurm status

- Total jobs: 112
- COMPLETED: 112
- RUNNING: 0
- PENDING: 0
- Breakdown JSONs parsed into metrics CSV: 112
- Failed jobs: 0

| model | temp | completed | running | parsed breakdowns | total |
|---|---:|---:|---:|---:|---:|
| qwen235b | 0.0 | 20 | 0 | 20 | 20 |
| qwen235b | 1.0 | 20 | 0 | 20 | 20 |
| qwen30ba3b | 0.0 | 16 | 0 | 16 | 16 |
| qwen30ba3b | 1.0 | 16 | 0 | 16 | 16 |
| qwen8 | 0.0 | 20 | 0 | 20 | 20 |
| qwen8 | 1.0 | 20 | 0 | 20 | 20 |

| batch | completed | running | parsed breakdowns | total |
|---:|---:|---:|---:|---:|
| 4 | 28 | 0 | 28 | 28 |
| 8 | 28 | 0 | 28 | 28 |
| 16 | 28 | 0 | 28 | 28 |
| 32 | 28 | 0 | 28 | 28 |

## Best completed speedups

Baseline-relative speedups are shown only when the matching baseline row has completed and been parsed.

| rank | model | temp | batch | method | tok/s/GPU | speedup | acceptance | mean accept len |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | qwen30ba3b | 0.0 | 4 | suffix | 483.19 | 6.85x | 91.7% | 9.32 |
| 2 | qwen235b | 0.0 | 4 | suffix | 51.63 | 6.57x | 89.8% | 8.80 |
| 3 | qwen30ba3b | 1.0 | 8 | suffix | 664.00 | 5.37x | 81.9% | 7.22 |
| 4 | qwen30ba3b | 1.0 | 4 | suffix | 367.50 | 5.21x | 86.8% | 7.78 |
| 5 | qwen235b | 0.0 | 16 | suffix | 121.46 | 5.16x | 80.4% | 6.78 |
| 6 | qwen30ba3b | 0.0 | 8 | suffix | 637.60 | 5.16x | 85.5% | 8.19 |
| 7 | qwen8 | 0.0 | 8 | suffix | 1476.47 | 4.84x | 87.6% | 8.92 |
| 8 | qwen235b | 1.0 | 4 | suffix | 35.39 | 4.50x | 74.3% | 6.16 |
| 9 | qwen8 | 0.0 | 4 | suffix | 663.50 | 4.37x | 89.6% | 9.05 |
| 10 | qwen30ba3b | 0.0 | 16 | suffix | 1139.92 | 4.12x | 84.6% | 8.05 |
| 11 | qwen235b | 0.0 | 8 | suffix | 61.08 | 3.94x | 86.7% | 7.79 |
| 12 | qwen235b | 1.0 | 8 | suffix | 54.63 | 3.52x | 70.0% | 5.52 |
| 13 | qwen235b | 1.0 | 16 | suffix | 76.25 | 3.24x | 63.2% | 4.45 |
| 14 | qwen30ba3b | 0.0 | 32 | suffix | 1591.20 | 3.14x | 82.0% | 7.53 |
| 15 | qwen8 | 0.0 | 16 | suffix | 1617.77 | 2.95x | 83.7% | 8.07 |
| 16 | qwen8 | 1.0 | 4 | suffix | 442.19 | 2.91x | 78.7% | 6.55 |

## Files

- Tracker: `latest_oci_qmath_math500_osl32k_batch_sweep_20260617_jobs.csv`
- Completed metrics: `docs/qmath_vllm_standalone_batch_sweep_completed_metrics_live_20260617.csv`
- Annotated status CSV: `docs/qmath_vllm_standalone_batch_sweep_status_20260617.csv`
