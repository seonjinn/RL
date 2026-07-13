# OCI-HSG SWE-RL Full-GRPO SpecDec After-Prewarm Launch - 2026-06-13

Status captured at `2026-06-13 12:22 CEST`.

## Prewarm State

- Prewarm/reconvert job: `3291097`
- Run ID: `20260613_oci_hsg_swerl_qwen235b_megatron_reconvert_r1`
- Last verified state before SSH timeout: RUNNING on `nvl72097-T[01-16]`
- Megatron cache readiness: `run_config.yaml` was present at `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/nemo_rl/Qwen/Qwen3-235B-A22B-Thinking-2507/iter_0000000/run_config.yaml`

## Matrix Submission State

The intended after-prewarm matrix is:

| method | max_steps | draft model | K |
| --- | ---: | --- | ---: |
| baseline | 10 |  | 0 |
| suffix | 10 |  | 32 |
| PARD | 10 | `amd/PARD-Qwen3-0.6B` | 5 |
| PARD-2 | 10 | `amd/PARD2-Qwen3-8B` | 1 |
| Eagle-3 | 10 | `nvidia/Qwen3-235B-A22B-Eagle3` | 3 |

Submission command was attempted with `SBATCH_DEPENDENCY=afterok:3291097`, but local SSH timed out before job IDs or a local tracker CSV were produced. The expected tracker file `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_r1_20260613_jobs.csv` did not exist after the failed SSH session, so there is no local evidence of a completed duplicate submission.

Prepared retry wrapper:

- `experiments/eagle3_online/submit_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_20260613.sh`

The wrapper uses run ID `20260613_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_r1`, checks active `squeue` entries for that run ID before submission, and submits the five-job matrix with dependency `afterok:3291097`.

## Launch Inputs

- Remote repo: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL-SWE_bench-20260613`
- Branch/head: `ruit/SWE_bench`, `4cc7d70c89e83876b06805bf439e866b0fcbe708`
- Account/partition: `nemotron_n3_post`, `batch`
- Container: `/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs`
- Target model: `Qwen/Qwen3-235B-A22B-Thinking-2507`
- Train data: `/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/blends/balanced_language.jsonl`
- Val data: `/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/swe_public_datasets_val_swebench.jsonl`

Last pre-submit checks passed for the remote repo branch/head, Rui launcher syntax, container path, dataset paths, and Megatron converted cache file. Live OCI submit is pending SSH/DNS recovery from the local machine.
