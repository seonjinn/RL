# OCI-HSG SWE-RL Full-GRPO SpecDec After-Prewarm N3Post Launch - 2026-06-14

Submitted Qwen3-235B SWE-RL Full-GRPO SpecDec matrix on OCI-HSG with account `nemotron_n3_post`.

Run ID: `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_r1`

Tracker: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_r1_20260614_jobs.csv`

Status: `docs/oci_hsg_swerl_fullgrpo_specdec_after_prewarm_n3post_r1_status_20260614.md`

Launch settings:

- Host: `oci-hsg-cs-001-vscode-02`
- Account: `nemotron_n3_post`
- Partition: `batch`
- Walltime: `04:00:00`
- Shape: 16 nodes, 4 GPUs per node, `--segment=16`
- Max steps: `10`
- Methods: baseline, suffix K32, PARD K5, PARD-2 K1, Eagle-3 K3
- Target model: `Qwen/Qwen3-235B-A22B-Thinking-2507`
- Prewarm cache checked: `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/nemo_rl/Qwen/Qwen3-235B-A22B-Thinking-2507/iter_0000000/run_config.yaml`

Jobs:

| job_id | method | K | state at first refresh |
| --- | --- | ---: | --- |
| 3299465 | baseline | 0 | RUNNING |
| 3299466 | suffix | 32 | RUNNING |
| 3299467 | PARD | 5 | RUNNING |
| 3299468 | PARD-2 | 1 | RUNNING |
| 3299469 | Eagle-3 | 3 | RUNNING |

Final status:

- All five jobs `3299465`-`3299469` failed after about `00:03:40`.
- Fetched driver logs show the common root cause was `wandb.errors.errors.UsageError: No API key configured. Use wandb login to log in.`
- This run is superseded by W&B-key retry run `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`.

Notes:

- Prewarm job `3291097` completed with `COMPLETED 0:0`, and the converted Megatron cache was present.
- The first submit attempt with `--dependency=afterok:3291097` was rejected by Slurm with `Job dependency problem`. Since the cache was already present, the final submission was made without the prewarm dependency.
- The launcher patcher now removes Rui's original `--dependency=singleton` line when `SBATCH_DEPENDENCY` is explicitly empty, so the five matrix jobs can queue independently.
- Dry-run diagnostics are redacted in `tmp/oci_hsg_swerl_after_prewarm_n3post_r1_nodep_dryrun_20260614.redacted.log`.
