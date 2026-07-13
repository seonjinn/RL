# Qwen3-235B SWE/Math SpecDec Launch Plan - 2026-06-13

This is the executable plan for the next Lyris pass once MFA/ControlMaster is active again. Current SSH status on 2026-06-13 at `06:00+02:00` was still `Permission denied (keyboard-interactive)` in batch mode, with no active ControlMaster socket, so no new Lyris jobs were submitted in the latest pass.

## Guarded Next Pass

Use the guarded wrapper first. It runs local validators, verifies Lyris SSH, refreshes existing SWE/Math/NeMo-RL trackers before submission, refuses `SUBMIT=true` unless SSH is verified, and skips duplicate tracker outputs unless `FORCE_RESUBMIT=true` is set.

Plan-only mode:

```bash
SUBMIT=false bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

Actual next-pass submission after MFA/ControlMaster is active:

```bash
SUBMIT=true bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

The default integrated NeMo-RL retry submits only `qwen30ba3b qwen32` and now includes matched `baseline suffix pard eagle3` rows. Refresh `2110001`, `2110002`, and `2110003` before including Qwen235B integrated cells.

For the official PARD-2 online training impact specifically, the Qwen8 matched comparison has already completed on OCI-HSG as jobs `3288181`, `3288182`, and `3288183`. The Lyris command below is optional and should only be used if we intentionally want a second-cluster rerun:

```bash
SUBMIT=true RUN_SWE_SUFFIX=false RUN_SWE_DRAFTER=false RUN_MATH500=false \
RUN_NEMORL_INTEGRATED=false RUN_QWEN8_PARD2_COMPARISON=true \
bash scripts/run_lyris_specdec_next_pass_20260613.sh
```

Runbook: `docs/qwen8_pard2_official_comparison_plan_20260613.md`.

Focused report after logs are fetched:

```bash
python3 scripts/build_qwen8_pard2_official_comparison_report.py
```

## SWE-Bench OSL32K Suffix K Sweep

Existing completed rows already cover Suffix K32 for SWE-Bench full and SWE-Bench-Verified at batches 2, 4, 8, 16, and 32. Submit K8/K16 companion rows to decide whether K32 is actually best under the same conditions.

```bash
K_SWEEP='8 16' \
DATASETS='verified full' \
BATCH_SWEEP='2 4 8 16 32' \
OUT=latest_lyris_qwen235b_swebench_osl32k_suffix_k8_k16_20260613_jobs.csv \
bash experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_suffix_k_sweep_20260613.sh
```

## SWE-Bench OSL32K Drafter K Sweep

Existing rows/liveness cover PARD K5, PARD-2 K1, and Eagle-3 K3. Submit high-K companion rows for PARD, PARD-2, and Eagle-3.

```bash
METHODS='pard pard2 eagle3' \
PARD_K_SWEEP='9 11' \
PARD2_K_SWEEP='9 11' \
EAGLE3_K_SWEEP='9 11' \
DATASETS='verified full' \
BATCH_SWEEP='2 4 8 16 32' \
OUT=latest_lyris_qwen235b_swebench_osl32k_drafter_k9_k11_20260613_jobs.csv \
bash experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_drafter_k_sweep_20260613.sh
```

If allocation pressure is high, run the drafter K sweep only for `BATCH_SWEEP='8 16 32'` first.

Refresh SWE/SWE-Verified after either sweep completes:

```bash
bash scripts/refresh_lyris_swebench_longosl_results.sh
```

The refresh script's default tracker list includes the existing Qwen235B OSL32K batch sweep plus the K8/K16 Suffix and K9/K11 PARD/PARD-2/Eagle-3 companion trackers.

## MATH500 OSL32K Qwen235B

Qwen8/Qwen30 MATH500 OSL32K results exist. This adds Qwen235B with baseline, Suffix K32, PARD K5, PARD-2 K1, and Eagle-3 K3.

```bash
OUT=latest_lyris_qwen235b_math500_osl32k_specdec_20260613_jobs.txt \
bash experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_math500_osl32k_specdec_20260613.sh
```

Because Lyris SSH is still gated by MFA, I also staged the MATH500 prompt file on OCI-HSG and submitted the same Qwen235B MATH500 OSL32K standalone matrix there after dry-run/preflight:

```bash
SUBMIT=true bash experiments/eagle3_qwen3_235b/submit_oci_qwen235b_math500_osl32k_specdec_20260613.sh
```

OCI-HSG jobs:

- baseline `3288484`
- Suffix K32 `3288487` failed at startup because the original arctic site had only a Python 3.13 `_C` extension while the OCI vLLM container imports Python 3.12.
- PARD K5 `3288488`
- PARD-2 K1 `3288490`
- Eagle-3 K3 `3288491`

I patched the OCI arctic default to the py312 compiled site and strengthened preflight to require `_C*.so` under `arctic_inference/suffix_decoding`. Suffix K32 was then resubmitted as retry job `3288594`.

Refresh OCI-HSG MATH500 status and completed metrics:

```bash
bash scripts/refresh_oci_qwen235b_math500_osl32k_results.sh
```

Refresh OCI-HSG live MATH500 progress from recent vLLM logger lines while the jobs are still running:

```bash
python3 scripts/refresh_oci_qwen235b_math500_live_progress.py
```

Current OCI-HSG outputs:

- `latest_oci_qwen235b_math500_osl32k_specdec_20260613_jobs.txt`
- `latest_oci_qwen235b_math500_suffix_py312_retry1_20260613_jobs.txt`
- `docs/oci_qwen235b_math500_osl32k_status_20260613.md`
- `docs/oci_qwen235b_math500_suffix_py312_retry1_status_20260613.md`
- `docs/oci_qwen235b_math500_live_progress_20260613.md`
- `docs/oci_qwen235b_math500_osl32k_metrics_20260613.csv`

Refresh after completion:

```bash
TRACKER_FILES='latest_lyris_qwen8_math500_osl32k_baseline_suffix_eagle3_20260612_jobs.txt latest_lyris_qwen8_math500_osl32k_pard2_official_k3_20260612_jobs.txt latest_lyris_qwen8_math500_osl32k_pard2_official_k5_20260612_jobs.txt latest_lyris_qwen30ba3b_math500_osl32k_baseline_suffix_pardk5_20260612_jobs.txt latest_lyris_qwen30ba3b_math500_osl32k_pardk3_20260612_jobs.txt latest_lyris_qwen235b_math500_osl32k_specdec_20260613_jobs.txt' \
bash scripts/refresh_lyris_math500_osl32k_results.sh
```

Regenerate the combined SWE/Math report after the SWE or Math refresh:

```bash
python3 scripts/build_qwen235b_specdec_swe_math_status.py
```

Regenerate the integrated NeMo-RL performance table after log collection:

```bash
bash scripts/fetch_lyris_nemorl_integrated_logs.sh
python3 scripts/summarize_lyris_nemorl_integrated_specdec.py
```

Current local report outputs:

- `docs/qwen235b_specdec_swe_math_status_20260613.md`
- `docs/qwen235b_specdec_swe_math_status_20260613.html`
- `docs/qwen235b_specdec_swe_math_status_20260613.csv`
- `docs/qwen235b_specdec_swe_math_status_20260613.png`
- `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.md`
- `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.csv`

## NeMo-RL Integrated SpecDec

Use the guarded wrapper above or the log-safe retry commands in `docs/lyris_nemorl_integrated_specdec_maxsteps10_launch_20260613.md`. The old integrated trackers were spec-only, so they cannot produce no-spec speedups by themselves. The patched launcher includes matched `baseline` cells in the next run, and `scripts/summarize_lyris_nemorl_integrated_specdec.py` computes speedups once logs are fetched.

Refresh Slurm state with `scripts/refresh_lyris_nemorl_integrated_specdec_results.sh` before duplicating Qwen235B integrated cells.

Fetch integrated logs with `scripts/fetch_lyris_nemorl_integrated_logs.sh`; it copies only `ray-driver.log` and `slurm-*.out` into `tmp/lyris_nemorl_integrated_logs/` and then rebuilds the metrics table.

## Validation

Local validation passed for all launchers and validators on 2026-06-13. An OCI-HSG remote dry-run/preflight for the SWE-RL Full-GRPO Qwen235B launcher also passed again at `06:00+02:00` using `CLUSTER_PROFILE=oci-hsg`, `SUBMIT=false`, `nemotron_n3_post`, `batch`, and `04:00:00`; output was written to `tmp/oci_hsg_swerl_fullgrpo_specdec_preflight_dryrun_20260613.csv`.

- `experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_suffix_k_sweep_20260613.sh`
- `experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_swebench_osl32k_drafter_k_sweep_20260613.sh`
- `experiments/eagle3_qwen3_235b/submit_lyris_qwen235b_math500_osl32k_specdec_20260613.sh`
- `scripts/refresh_lyris_math500_osl32k_results.sh`
- `experiments/eagle3_qwen3_235b/submit_oci_qwen235b_math500_osl32k_specdec_20260613.sh`
- `scripts/refresh_oci_qwen235b_math500_osl32k_results.sh`
- `scripts/refresh_lyris_nemorl_integrated_specdec_results.sh`
- `scripts/refresh_lyris_swebench_longosl_results.sh`
- `scripts/run_lyris_specdec_next_pass_20260613.sh`
- `scripts/fetch_lyris_nemorl_integrated_logs.sh`
- `scripts/build_qwen235b_specdec_swe_math_status.py`
- `scripts/summarize_lyris_nemorl_integrated_specdec.py`
- `scripts/build_qwen8_pard2_official_comparison_report.py`
- `scripts/validate_qwen8_pard2_comparison_contract.py`
- `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_comparison_20260613.sh`
- `scripts/validate_nemorl_online_specdec_contract.py`
- `scripts/validate_nemorl_pard_source_bundle.py`

Validation outputs:

- `docs/nemorl_online_specdec_contract_validation_20260613.md`
- `docs/nemorl_pard_source_bundle_validation_20260613.md`
- `docs/qwen8_pard2_official_comparison_contract_validation_20260613.md`
- `docs/lyris_nemorl_integrated_specdec_maxsteps10_metrics_20260613.md`
- `docs/qwen8_pard2_official_comparison_plan_20260613.md`
- `docs/qwen8_pard2_official_online_impact_20260613.md`
