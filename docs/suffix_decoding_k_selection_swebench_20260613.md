# Suffix Decoding K Selection for SWE-Bench - 2026-06-13

Current conclusion: K32 is a strong default for Qwen3-235B SWE-Bench OSL32K, but it is not proven to be globally best. The completed large batch sweep only ran Suffix at K32; the smaller pilot K sweeps show that the best K depends on output length.

## Completed Qwen3-235B OSL32K Batch Sweep

This sweep covers SWE-Bench full and SWE-Bench-Verified at batch sizes 2, 4, 8, 16, and 32, but Suffix was run only at K32.

| Dataset | Batches | K | tok/s/GPU range | Speedup range | Acceptance range |
| --- | --- | ---: | ---: | ---: | ---: |
| SWE-Bench full | 2, 4, 8, 16, 32 | 32 | 25.17-223.15 | 3.42x-6.10x | 76.24%-87.55% |
| SWE-Bench-Verified | 2, 4, 8, 16, 32 | 32 | 25.80-223.25 | 3.47x-6.18x | 79.92%-86.20% |

Because no K8/K16 rows exist in this same full batch sweep yet, K32 should be reported as "best completed setting" rather than "best possible setting."

## Pilot K Sweep Evidence

Qwen3-235B SWE-Bench-Verified pilots with TP4/FP8 KV cache:

| Scenario | Batch | K values completed | Best throughput | Best acceptance | Interpretation |
| --- | ---: | --- | --- | --- | --- |
| OSL1K, n16 | 1 | 1, 2, 4, 8, 32 | K4: 3.06 tok/s/GPU, 1.50x | K1: 34.90% | K32 is not best for short OSL. |
| OSL1K, n16 | 2 | 1, 2, 4, 8, 32 | K8: 6.58 tok/s/GPU, 1.61x | K1: 40.11% | K8 is slightly better than K32. |
| OSL32K, n2 | 1 | 1, 2, 4, 8, 32 | K32: 12.60 tok/s/GPU, 6.04x | K8: 86.35% | K32 has best speed; K8 has best acceptance. |
| OSL64K, n1 | 1 | 8, 32 | K8: 4.40 tok/s/GPU, 2.10x | K8: 46.38% | K8 beats K32 on this pilot. |

## Next Sweep

For SWE-Bench and SWE-Bench-Verified, the next Suffix sweep should compare K8, K16, and K32 under the same OSL32K batch sweep conditions. If the 64K/128K cases remain important, include K8 and K16 there as well instead of assuming K32 remains optimal.

The helper below submits the missing Qwen3-235B OSL32K K8/K16 rows for both SWE-Bench full and SWE-Bench-Verified across batches 2, 4, 8, 16, and 32. Existing K32 rows can be reused from the completed batch sweep; set `K_SWEEP='8 16 32'` only if a fresh K32 rerun is needed.

```bash
K_SWEEP='8 16' \
DATASETS='verified full' \
BATCH_SWEEP='2 4 8 16 32' \
OUT=latest_lyris_qwen235b_swebench_osl32k_suffix_k8_k16_20260613_jobs.csv \
bash experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_suffix_k_sweep_20260613.sh
```

Evidence files:

- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv`
- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.md`
- `docs/lyris_qwen235b_suffix_metrics_20260612.csv`
- `docs/lyris_qwen235b_suffix_status_20260612.md`
- `experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_suffix_k_sweep_20260613.sh`
