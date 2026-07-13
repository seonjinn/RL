# Lyris SWE-RL Full-GRPO SpecDec W&B Retry - 2026-06-14

Checked at `2026-06-14T08:18-07:00` local time (`2026-06-14T15:18Z`).

## Context

The previous Lyris SWE-RL prewarm/reconvert job `2114333` failed before step 1. The fetched driver log shows `wandb.errors.errors.UsageError: No API key configured. Use wandb login to log in.` The dependent matrix jobs `2114342`-`2114346` were cancelled by the failed dependency.

For this retry, `WANDB_API_KEY` was passed only as a transient submission environment variable. The key is not persisted in tracker files, docs, shell scripts, or status artifacts.

## Submitted Jobs

Prewarm/reconvert tracker:

- `latest_lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_20260614_jobs.csv`

Matrix tracker:

- `latest_lyris_swerl_qwen235b_fullgrpo_specdec_after_wandb_prewarm_r1_20260614_jobs.csv`

Slurm walltime was set to `05:00:00` because the Lyris `gb200` partition limit is 5 hours.

| Job | Method | Steps | K | State | Dependency |
| ---: | --- | ---: | ---: | --- | --- |
| `2118715` | baseline prewarm/reconvert | 1 | 0 | `CANCELLED` after `03:28:15` | none |
| `2118719` | baseline | 10 | 0 | `CANCELLED` | cancelled with broken prewarm |
| `2118720` | suffix | 10 | 32 | `CANCELLED` | cancelled with broken prewarm |
| `2118721` | PARD | 10 | 5 | `CANCELLED` | cancelled with broken prewarm |
| `2118724` | PARD-2 | 10 | 1 | `CANCELLED` | cancelled with broken prewarm |
| `2118725` | Eagle-3 | 10 | 3 | `CANCELLED` | cancelled with broken prewarm |

## Status Artifacts

- `docs/lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_status_20260614.md`
- `docs/lyris_swerl_fullgrpo_specdec_after_wandb_prewarm_r1_status_20260614.md`
- `docs/lyris_swerl_fullgrpo_log_error_summary_20260613.md`

## Latest Log Read

`2118715` passed the earlier W&B failure point, started rollout collection, and then repeatedly failed inside SWE Gym/OpenHands because no SWE-rebench SIF images were visible on Lyris. The driver log reached about `1.5 GiB` with repeated `No SIF found for SWE-rebench instance ...` exceptions. The job and dependent matrix were cancelled to free the allocation.

The needed `1,689` missing SIFs exist on OCI-HSG and total about `753.26 GiB`. The first smoke SIF transfers to `/lustre/fsw/coreai_dlalgo_llm/users/sna/images/swerebench` completed successfully, and full staging is now split across four local shard streams. Latest Lyris-side check shows `266` complete `.sif` files, `4` active temporary files at the instant checked, and about `141G` in the target directory; `1,423` ids remain unstaged. Progress logs are `tmp/lyris_swerebench_sif_stage_s0_20260614.log` through `tmp/lyris_swerebench_sif_stage_s3_20260614.log`.

Filtered smoke preparation: `scripts/refresh_lyris_staged_sif_smoke_dataset_20260614.sh` generated `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/filtered/staged_sif_20260614_1516/train.jsonl` with `266` train rows matching currently staged SIFs. The staged validation subset has `0` rows, so `experiments/eagle3_online/submit_lyris_swerl_qwen235b_stagedsif_smoke_20260614.sh` points `VAL_DATA_PATH` at the train subset. The wrapper reads `tmp/latest_lyris_staged_sif_smoke_dataset_20260614.env` by default and dry-runs baseline, PARD K5, and PARD-2 K1 step-1 jobs with the Lyris-visible image root, local container, writable cache path, and smoke SWE-agent concurrency `64`; actual submit still requires a transient `WANDB_API_KEY`, and the current non-interactive Lyris login environment reports it as unset.

## Next Check

Refresh:

```bash
python3 scripts/refresh_oci_hsg_swerl_fullgrpo_specdec_status.py \
  --host login-lyris \
  --tracker latest_lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_20260614_jobs.csv \
  --csv-out docs/lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_status_20260614.csv \
  --markdown-out docs/lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_status_20260614.md \
  --raw-prefix lyris_swerl_qwen235b_megatron_reconvert_wandb_r1 \
  --title "Lyris SWE-RL Qwen235B Megatron Reconvert W&B Retry Status"

python3 scripts/refresh_oci_hsg_swerl_fullgrpo_specdec_status.py \
  --host login-lyris \
  --tracker latest_lyris_swerl_qwen235b_fullgrpo_specdec_after_wandb_prewarm_r1_20260614_jobs.csv \
  --csv-out docs/lyris_swerl_fullgrpo_specdec_after_wandb_prewarm_r1_status_20260614.csv \
  --markdown-out docs/lyris_swerl_fullgrpo_specdec_after_wandb_prewarm_r1_status_20260614.md \
  --raw-prefix lyris_swerl_fullgrpo_specdec_after_wandb_prewarm_r1 \
  --title "Lyris SWE-RL Full-GRPO SpecDec After W&B Prewarm Status"
```

If `2118715` fails or the matrix becomes terminal, fetch logs:

```bash
REMOTE_HOST=login-lyris \
TRACKER_FILES="latest_lyris_swerl_qwen235b_megatron_reconvert_wandb_r1_20260614_jobs.csv latest_lyris_swerl_qwen235b_fullgrpo_specdec_after_wandb_prewarm_r1_20260614_jobs.csv" \
bash scripts/fetch_swerl_fullgrpo_logs_20260613.sh
```
