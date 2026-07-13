# OCI-HSG Qwen8 Official PARD-2 Online Comparison - 2026-06-13

Status at `2026-06-13T04:26+02:00`: the reviewed short-QoS retry `20260613_oci_hsg_qwen8_pard2_official_comparison_gafix2_shortqos` reached generation/reward processing, but baseline `3287708` and static PARD-2 `3287710` failed before GRPO step metrics with `TypeError: calculate_baseline_and_std_per_prompt() got an unexpected keyword argument 'std_rewards'`. Online PARD-2 `3287712` was canceled before useful work to avoid repeating the same known failure. The older normal-QoS `gafix2` jobs `3287662`, `3287663`, and `3287665` also ended failed/canceled after cleanup.

Fix status: `nemo_rl/algorithms/utils.py` now accepts the `std_rewards` keyword used by the staged OCI `grpo.py`, and uses it only for prompt-group std while preserving shaped rewards for the baseline. The launcher remote preflight now also compiles `nemo_rl/algorithms/utils.py`, and the local contract validator checks both the helper signature and the preflight coverage before resubmission.

Corrected resubmission at `2026-06-13T04:30+02:00`: remote staging/preflight passed again against the corrected tree, including the official PARD-2 vLLM checks and `nemo_rl/algorithms/utils.py` compile. New short-QoS jobs are baseline `3287931` and static PARD-2 `3287932` running, with online PARD-2 `3287933` pending only on `QOSMaxJobsPerUserLimit`.

Second correction at `2026-06-13T05:07+02:00`: jobs `3287931`, `3287932`, and `3287933` all passed the earlier `std_rewards` point, completed step-1 generation/reward/logprob/advantage/policy-training work, then failed in metric printing with `TypeError: 'MasterConfig' object is not subscriptable`. `print_performance_metrics()` now normalizes Pydantic-style configs with `master_config.model_dump()`. I restored the interrupted remote preflight helper files, reran the remote dry-run/preflight, and submitted replacement short-QoS jobs: baseline `3288181`, static PARD-2 `3288182`, and online PARD-2 `3288183`.

Final Qwen8 result at `2026-06-13T05:36+02:00`: all three replacement jobs completed successfully and parsed 9 step-2+ metrics rows. Baseline measured `225.469` gen-worker tok/s/GPU. Static PARD-2 K1 measured `136.891` tok/s/GPU, `0.6071x` baseline gen-worker speed, `0.8291x` E2E speed, and `1.836` acceptance. Online PARD-2 K1 measured `132.736` tok/s/GPU, `0.5887x` baseline gen-worker speed, `0.6705x` E2E speed, `2.553` acceptance, and 9 draft refits. Online improved acceptance by `+0.717` over static but was slightly slower than static (`0.9696x` gen-worker and `0.8087x` E2E online/static), so this Qwen8 K1 official PARD-2 setting is a functional online-training pass, not a speedup win.

Purpose: matched NeMo-RL comparison for online drafter training impact.

Launcher:

- Local: `experiments/eagle3_online/submit_lyris_qwen8_pard2_official_comparison_20260613.sh`
- Tracker: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv`
- Failed first-run tracker: `docs/oci_hsg_qwen8_pard2_official_comparison_failed_gafusion_20260613_jobs.csv`
- Preserved normal-QoS retry tracker: `docs/oci_hsg_qwen8_pard2_official_comparison_gafix2_normal_jobs_20260613.csv`
- Dry-run tracker: `latest_oci_hsg_qwen8_pard2_official_comparison_20260613_dryrun.csv`
- Status: `docs/oci_hsg_qwen8_pard2_official_comparison_status_20260613.md`
- Remote stage repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-comparison-oci-20260613`
- Base repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606`

Preflight evidence:

- Local launcher syntax and contract validation passed.
- OCI-HSG profile uses `oci-hsg-cs-001-vscode-02`, `coreai_dlalgo_llm`, `batch`, and `04:00:00`.
- Remote `ray.sub`, Qwen8 GRPO recipe, container, and official PARD-2 vLLM site exist.
- Remote PARD-2 checks passed:
  - `vLLM draft refit target_proj tests passed`
  - `PARD2_OFFICIAL_PATCH_CHECKS` showed all required patch checks true.
- The first submit attempt was blocked before any job was created because the inherited `04:30:00` walltime exceeded the OCI-HSG `batch` limit. The launcher now uses profile-specific `DEFAULT_WALLTIME="04:00:00"` for OCI-HSG.
- A dry-run after the fix generated all three sbatch commands with `--time=04:00:00`, `--gres=gpu:4`, and `--segment 1`.
- First submitted jobs `3286953`, `3286955`, and `3286956` reached Megatron policy worker creation and failed before step metrics because Megatron used `gradient_accumulation_fusion=True` without the APEX fused extension.
- First fusion-overlay retry jobs `3287330`, `3287331`, and `3287332` failed the same way during HF checkpoint import. The printed NeMo-RL config had `gradient_accumulation_fusion=False`, but `community_import.py` finalized the Megatron-Bridge provider and called `provide_distributed_model()` before the flag was forced onto that provider.
- The current patch applies the configured fusion flag to provider attrs and nested `transformer`/`transformer_config`/`_model_config` holders in both `setup.py` and `community_import.py`, then reapplies it after `model_provider.finalize()`.
- After that patch, local compile, launcher syntax, `git diff --check`, Qwen8 contract validation, remote stage sync, remote py_compile for `setup.py`/`community_import.py`, remote PARD-2 checks, source/container/path checks, account/partition checks, and a three-variant dry-run all passed.
- Before the short-QoS retry, I extended the launcher to forward `SBATCH_EXTRA_ARGS`, updated the validator to check that remote handoff, reran shell syntax, Python compile, Qwen8 contract validation, and `git diff --check`, then ran a non-submit remote dry-run with `SBATCH_EXTRA_ARGS='--qos=short'` and `WALLTIME=02:00:00`. The dry-run emitted the expected `--qos=short --time=02:00:00` sbatch lines.

Latest corrected short-QoS jobs:

| job_id | variant | vLLM SpecDec | online draft training | K | max_steps | status |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 3288181 | `baseline` | disabled | disabled | 0 | 10 | `COMPLETED` |
| 3288182 | `static_pard2` | `method=pard2` | disabled | 1 | 10 | `COMPLETED` |
| 3288183 | `online_pard2` | `method=pard2` | enabled | 1 | 10 | `COMPLETED` |

Previous corrected short-QoS jobs from the metric-printing mismatch:

| job_id | variant | vLLM SpecDec | online draft training | K | max_steps | status |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 3287931 | `baseline` | disabled | disabled | 0 | 10 | `FAILED` after step-1 metric printing |
| 3287932 | `static_pard2` | `method=pard2` | disabled | 1 | 10 | `FAILED` after step-1 metric printing |
| 3287933 | `online_pard2` | `method=pard2` | enabled | 1 | 10 | `FAILED` after step-1 metric printing |

Failed short-QoS jobs from the previous staged helper mismatch:

| job_id | variant | vLLM SpecDec | online draft training | K | max_steps | status |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 3287708 | `baseline` | disabled | disabled | 0 | 10 | `FAILED` before step metrics |
| 3287710 | `static_pard2` | `method=pard2` | disabled | 1 | 10 | `FAILED` before step metrics |
| 3287712 | `online_pard2` | `method=pard2` | enabled | 1 | 10 | `CANCELLED` before useful work |

Preserved normal-QoS retry jobs:

| job_id | variant | K | max_steps | status before short-QoS retry |
| ---: | --- | ---: | ---: | --- |
| 3287662 | `baseline` | 0 | 10 | `FAILED` during the same bad staged helper window |
| 3287663 | `static_pard2` | 1 | 10 | `FAILED` during the same bad staged helper window |
| 3287665 | `online_pard2` | 1 | 10 | `FAILED`/cleanup-canceled during the same bad staged helper window |

Common controls:

- Target: `Qwen/Qwen3-8B`
- Drafter: `amd/PARD2-Qwen3-8B`
- `num_prompts=4`, `num_generations=4`, `train_global_batch_size=16`
- `max_new_tokens=256`, `min_tokens=128`, `max_model_len=2048`
- `policy.generation.temperature=1.0`, `top_p=1.0`, `top_k=-1`
- W&B disabled for this submission batch.

Report outputs:

- `docs/oci_hsg_qwen8_pard2_official_comparison_metrics_20260613.csv`
- `docs/oci_hsg_qwen8_pard2_official_comparison_metrics_20260613.md`
- `docs/qwen8_pard2_official_comparison_metrics_20260613.csv`
- `docs/qwen8_pard2_official_comparison_metrics_20260613.md`
- `docs/qwen8_pard2_official_online_impact_20260613.csv`
- `docs/qwen8_pard2_official_online_impact_20260613.md`

Current report status is fully parsed for baseline/static/online. Baseline/static/online all completed the 10-step run; the report filters to step>=2, so each row shows `9/9` parsed steps. Static and online PARD-2 are slower than no-spec in this Qwen8 K1 setup, mainly because acceptance is very low. Online training/refit is functional and ran on every parsed step, but it did not recover enough acceptance or throughput to beat static or baseline.

Refresh commands:

```bash
FETCH_LOGS=true REBUILD_REPORTS=true bash scripts/refresh_oci_hsg_specdec_results_20260613.sh

# Or run the pieces manually:
python3 scripts/refresh_oci_hsg_qwen8_pard2_comparison_status.py

REMOTE_HOST=oci-hsg-cs-001-vscode-02 \
TRACKER_FILES=latest_oci_hsg_qwen8_pard2_official_comparison_20260613_jobs.csv \
bash scripts/fetch_lyris_nemorl_integrated_logs.sh

python3 scripts/build_qwen8_pard2_official_comparison_report.py
```
