# SpecDec Missing Result Log Triage - 2026-06-19

## What Was Checked

- Rechecked Lyris and OCI-HSG access.
- Revalidated old missing/failed vLLM standalone rows against `sacct` and available run logs.
- Rechecked current `/lustre/fsw` quota on Lyris and confirmed a new run directory can be created.
- Submitted missing standalone coverage that was still needed for clean Math/SWE comparisons.

## Failure Causes From Logs

| Area | Jobs | Cause | OOM? | Action |
|---|---:|---|---|---|
| Math Qwen3-235B baseline temp0 | 2113223 | Cancelled by 5h time limit while processing first prompt. | No | Resubmitted split baseline batch1/2 on OCI-HSG. |
| Math Qwen3-235B baseline temp1 | 2124147 | Cancelled by 5h time limit while processing first prompt. | No | Resubmitted split baseline batch1/2 on OCI-HSG. |
| SWE Qwen3-235B temp0 baseline | 2133873 | Cancelled by 5h time limit. | No | Already has home retry evidence for batch1/2; large batch needs OCI long split if we want temp-matched rows. |
| SWE Qwen3-235B temp0 suffix/eagle3 | 2133874, 2133875 | `Disk quota exceeded` while writing `profile` / `breakdown.json`. | No | Quota is now clear; previous failure was filesystem write failure. |
| SWE Qwen3-235B temp1 baseline/suffix/eagle3 | 2133883-2133885 | Failed in 17s with no `slurm-*.out` created under the requested `/lustre` output path; consistent with quota/output-path failure at startup. | No evidence of OOM | Quota is now clear. |
| Older SWE high batch Qwen3-30B | prior batch32 high-cap | `max_num_batched_tokens=1310720` caused CUDA OOM. | Yes | Later retries use `131072`. |

Current Lyris quota probe: `/lustre/fsw` mkdir succeeds; user quota showed about `45.11T / 250T` and `6.39M / 26.2M` files used.

## New Jobs Submitted

| Cluster | Domain | Model | Coverage | Methods | Jobs | Status snapshot |
|---|---|---|---|---|---:|---|
| OCI-HSG | Math | Qwen3-32B | temp0/1, batch 4/8/16/32, ISL 4096, OSL 32768 | baseline, suffix k32, PARD k5, Eagle3 k3 | 32 | COMPLETED; 32 metric rows parsed |
| OCI-HSG | Math | Qwen3-235B | temp0/1, batch 1/2 baseline retry | baseline | 4 | COMPLETED; 4 metric rows parsed |
| OCI-HSG | Math | Qwen3-32B | temp0/1, batch 1/2 | baseline, suffix k32, PARD k5, Eagle3 k3 | 16 | COMPLETED; 16 metric rows parsed |
| OCI-HSG | SWE | Qwen3-235B-A22B | temp0/1, batch 4/8/16/32, ISL 4096, OSL 32768 | baseline, suffix k32, Eagle3 k3 | 24 | COMPLETED; 24 metric rows parsed |
| Lyris | SWE | Qwen3-32B | temp0/1, batch 4/8/16/32, ISL 4096, OSL 32768 | baseline, suffix k32, Eagle3 k3 | 24 | COMPLETED; 24 metric rows parsed |
| Lyris | SWE | Qwen3-30B-A3B | temp0/1, batch 4/8/16/32, ISL 4096, OSL 32768 | baseline, suffix k32, Eagle3 k3 | 24 | COMPLETED; 24 metric rows parsed |

## Local Ledgers

- `docs/oci_qmath_qwen32_math500_osl32k_batch_sweep_20260619_jobs.csv`
- `docs/oci_qmath_qwen32_math500_osl32k_batch_sweep_20260619_status_live.csv`
- `docs/oci_qmath_qwen32_math500_osl32k_batch_sweep_20260619_metrics_live.csv`
- `docs/oci_qmath_qwen235b_baseline_qwen32_bs12_20260619_jobs.csv`
- `docs/oci_qmath_qwen235b_baseline_qwen32_bs12_20260619_status_live.csv`
- `docs/oci_swebench_qwen235b_osl32k_temp01_bsweep_20260619_jobs.csv`
- `docs/oci_swebench_qwen235b_osl32k_temp01_bsweep_20260619_status_live.csv`
- `docs/lyris_swebench_qwen32_osl32k_temp01_bsweep_20260619_jobs.csv`
- `docs/lyris_swebench_qwen32_osl32k_temp01_bsweep_20260619_status_live.csv`
- `docs/lyris_swebench_qwen30ba3b_osl32k_temp01_bsweep_20260619_jobs.csv`
- `docs/lyris_swebench_qwen30ba3b_osl32k_temp01_bsweep_20260619_status_live.csv`

## Still Open

- All newly submitted vLLM standalone jobs completed successfully and wrote `breakdown.json`.
- Live metrics CSVs now contain 124 new metric rows across Math/SWE.
