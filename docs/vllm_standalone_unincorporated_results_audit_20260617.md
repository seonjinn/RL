# vLLM Standalone Unincorporated Results Audit

Updated: `2026-06-17`

Scope: audit of local standalone vLLM metrics/status-style artifacts against `docs/vllm_standalone_clean_status_20260617.md` and `docs/vllm_standalone_clean_status_20260617.csv`. I did not edit the clean status files.

Detailed ledger: `docs/vllm_standalone_unincorporated_results_audit_20260617.csv`.

## Findings

The clean status is not a complete row-level ledger. It correctly captures the latest Lyris temp0/temp1 core matrix and some archived Qwen235B Math and batch-sweep evidence, but several local standalone result groups are either absent or only mentioned at file level.

Highest-priority unincorporated candidates:

| Source | Rows | What is missing from clean status | Action |
| --- | ---: | --- | --- |
| `docs/oci_qwen235b_math500_osl32k_metrics_20260613.csv` plus `docs/oci_qwen235b_math500_suffix_py312_retry1_metrics_20260613.csv` | 8 | OCI Qwen235B Math baseline/eagle3/pard/pard2/suffix at ISL/OSL `4096/32768`, batches `1/2`; this fills a matched-baseline gap that Lyris Qwen235B Math did not fill. | include in clean report |
| `docs/oci_qwen235b_math500_drafter_k9_metrics_20260613.csv` and `docs/oci_qwen235b_math500_drafter_k11_metrics_20260613.csv` | 10 | OCI Qwen235B Math high-k drafter rows for EAGLE/PARD/PARD2 at k9/k11. | include in clean report |
| `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r4_noprof_metrics.csv` | 24 | OCI Qwen8 PARD1 temp0/temp1 Math and SWE rows. They are in `clean_supplemental`/`clean_results`, but not in `clean_status`. | include or cross-reference in clean status |
| `docs/lyris_qwen235b_standalone_fast_20260613_metrics.csv` | 80 SWE rows | Qwen235B SWE temp0/high-k fast sweep: EAGLE/PARD/PARD2 k9/k11 and suffix k8/k16 over batches `2/4/8/16/32`. | include as archived supplemental |
| `docs/lyris_qwen235b_standalone_temp1rl_20260614_metrics.csv` | 90 SWE rows | Qwen235B SWE temp1 baseline and high-k rows over batches `2/4/8/16/32`; clean status cites the file but omits these row groups. | include as archived supplemental |
| `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_speedups_20260612.csv` | 20 | Provisional Qwen235B larger-batch speedups from final spec rows plus live baseline telemetry. | include only with caveat |
| `docs/lyris_swebench_osl32k_batch_sweep_metrics_20260612.csv` | 6 | Older Qwen8/Qwen30 suffix batch-sweep rows at OSL32k, batches `4/8/16`. | include as partial batch evidence |
| `docs/lyris_swebench_longosl_metrics_20260612.csv` | 95 | Qwen8/Qwen30 LongOSL sweep, including OSL32k larger batches and OSL16k/64k/96k/128k rows. | include as archived LongOSL supplemental |
| `docs/lyris_specdec_32k_metrics_20260612.csv` | 64 | Older mixed-model OSL32k sweep for Qwen8/Qwen14/Qwen30/Qwen235B with EAGLE/PARD/PARD2/suffix variants. | include as archived 32k supplemental |

Already represented or intentionally superseded:

- `docs/lyris_qwen235b_swebench_osl32k_batch_sweep_metrics_20260612.csv` is represented in clean status as aggregate Qwen235B larger-batch evidence for EAGLE3 and suffix. The raw file also has batch2 rows that clean status does not list individually.
- `docs/vllm_standalone_temp0_temp1_trends_20260616.csv` is a derived rollup, not independent raw evidence. Its important underlying source rows are audited separately.
- `docs/lyris_math500_osl32k_metrics_20260612.csv` is older Math temp0 Qwen8/Qwen30 evidence and is superseded by the 20260616 Math temp matrix.
- `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics.csv`, `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_partial_metrics.csv`, and `docs/lyris_swebench_osl32k_temp01_home_retry_metrics_live_20260616.csv` are intermediate shards superseded by `docs/lyris_swebench_osl32k_temp01_core_matrix_20260616_metrics_live.csv`.

Ignored or archived:

- Empty metrics files: `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_metrics.csv`, `docs/oci_hsg_qwen8_pard1_standalone_temp01_20260616_r2_metrics.csv`, and `docs/lyris_qwen235b_math500_osl32k_specdec_metrics_20260613.csv`.
- Synthetic or OSL1024/OSL16k standalone files are valid historical artifacts but outside the clean status scope: examples include `docs/qwen32_qwen30ba3b_vllm_standalone_eagle3_metrics_20260606.csv`, `docs/eagle3_focus_vllm_standalone_metrics.csv`, `docs/public_pard2_vllm_standalone_20260611.csv`, and `docs/swebench_specdec_standalone_comparison_20260611.csv`.

## Recommendation

If the clean status should remain a current core-matrix status, leave it as-is and archive the unincorporated rows under a supplemental/historical appendix. If it should be a complete standalone evidence ledger, add the OCI Qwen235B Math rows, OCI Qwen8 PARD1 rows, Qwen235B fast/temp1 SWE high-k sweeps, and the older 32k/LongOSL batch evidence with clear labels that they are archived or partial.
