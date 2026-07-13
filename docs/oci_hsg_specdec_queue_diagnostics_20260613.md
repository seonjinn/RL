# OCI-HSG SpecDec Queue Diagnostics - 2026-06-13

Checked at `2026-06-13T04:26+02:00`.

## Connectivity

- OCI-HSG batch SSH works via existing ControlMaster: `oci-hsg-cs-001-vscode-02`.
- Lyris batch SSH is still blocked: `login-lyris` returns `Permission denied (keyboard-interactive)` without an active ControlMaster.

## Qwen8 PARD-2 Online Comparison

Updated at `2026-06-13T05:36+02:00`: corrected `masterconfigfix_shortqos` jobs completed after remote preflight passed with `nemo_rl/algorithms/utils.py` including both the `std_rewards` helper fix and the `MasterConfig.model_dump()` metric-printing compatibility fix.

Current corrected short-QoS jobs:

| job_id | variant | state | reason | account | nodes | GPUs | start estimate |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 3288181 | baseline | COMPLETED | exit 0 | coreai_dlalgo_llm | 1 | 4 | done |
| 3288182 | static_pard2 | COMPLETED | exit 0 | coreai_dlalgo_llm | 1 | 4 | done |
| 3288183 | online_pard2 | COMPLETED | exit 0 | coreai_dlalgo_llm | 1 | 4 | done |

Final parsed Qwen8 metrics:

| variant | gen worker tok/s/GPU | speedup vs baseline | E2E speedup | acceptance | draft refits |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 225.469 | 1.0000x | 1.0000x |  | 0 |
| static_pard2 | 136.891 | 0.6071x | 0.8291x | 1.836 | 0 |
| online_pard2 | 132.736 | 0.5887x | 0.6705x | 2.553 | 9 |

Metric-printing failure retry jobs:

| job_id | variant | state | reason | account | nodes | GPUs | tracker |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 3287931 | baseline | FAILED | `MasterConfig` object was not subscriptable in `print_performance_metrics()` | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_failed_masterconfig_20260613_jobs.csv` |
| 3287932 | static_pard2 | FAILED | same metric-printing mismatch | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_failed_masterconfig_20260613_jobs.csv` |
| 3287933 | online_pard2 | FAILED | same metric-printing mismatch | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_failed_masterconfig_20260613_jobs.csv` |

Previous short-QoS retry jobs:

| job_id | variant | state | reason | account | nodes | GPUs | start estimate |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 3287708 | baseline | FAILED | `std_rewards` helper signature mismatch | coreai_dlalgo_llm | 1 | 4 | ended |
| 3287710 | static_pard2 | FAILED | `std_rewards` helper signature mismatch | coreai_dlalgo_llm | 1 | 4 | ended |
| 3287712 | online_pard2 | CANCELLED | canceled before repeating known failure | coreai_dlalgo_llm | 1 | 4 | ended |

Preserved normal-QoS jobs:

| job_id | variant | state | reason | account | nodes | GPUs | tracker |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 3287662 | baseline | FAILED | same bad staged helper window | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_gafix2_normal_jobs_20260613.csv` |
| 3287663 | static_pard2 | FAILED | same bad staged helper window | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_gafix2_normal_jobs_20260613.csv` |
| 3287665 | online_pard2 | FAILED | cleanup/cancel in same bad staged helper window | coreai_dlalgo_llm | 1 | 4 | `docs/oci_hsg_qwen8_pard2_official_comparison_gafix2_normal_jobs_20260613.csv` |

`scontrol show job` confirms:

- `Dependency=(null)`
- short-QoS retry `TimeLimit=02:00:00`
- `Partition=batch`
- `ReqTRES=cpu=64,mem=920G,node=1,billing=1,gres/gpu=4`
- `WorkDir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-pard2-official-comparison-oci-20260613`

Interpretation: no dependency/account/path issue was visible. The allocated jobs failed because staged `grpo.py` passed `std_rewards=` into `calculate_baseline_and_std_per_prompt()`, but the staged `utils.py` did not yet accept that keyword. This has been patched locally and added to remote preflight coverage before resubmission.

Runtime log evidence:

- `3287708` baseline and `3287710` static PARD-2 both completed math environment setup, initialized `4/4` vLLM workers, loaded Megatron policy/reference workers, generated, and reached reward/advantage computation.
- Static PARD-2 passed `PARD2_OFFICIAL_PATCH_CHECKS` and emitted early vLLM acceptance logs around `2.0%-3.4%`.
- Both failed at the GRPO helper signature mismatch before any completed step metrics, so performance remains unavailable.

## Qwen235B SWE-RL Full-GRPO

Current jobs:

| jobs | state | reason | account | nodes/job | GPUs/job |
| --- | --- | --- | --- | ---: | ---: |
| 3286445, 3286458, 3286519, 3286521, 3286522 | PENDING | Priority | nemotron_n3_post | 16 | 64 |
| 3286523, 3286524, 3286525, 3286526, 3286527 | PENDING | Priority | nemotron_n3_post | 16 | 64 |

`scontrol show job` confirms:

- `Dependency=(null)`
- `TimeLimit=04:00:00`
- `Partition=batch`
- `ReqTRES=cpu=16,mem=14720G,node=16,billing=16,gres/gpu=64`
- `WorkDir=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_post/users/ruit/evolution_rl`

`sprio` shows Qwen235B SWE-RL jobs have higher priority than the Qwen8 jobs, but all remain pending. `squeue --start` still reports `PD (Priority)`, so the printed start-time candidates should not be treated as actual allocation.
