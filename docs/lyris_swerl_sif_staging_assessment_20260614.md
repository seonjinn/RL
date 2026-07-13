# Lyris SWE-RL SIF Staging Assessment - 2026-06-14

Last checked: `2026-06-14T13:10-07:00` (`2026-06-14T20:10Z`).

## Current State

- Lyris W&B retry prewarm job `2118715` was cancelled after `03:28:15`, because rollout was not useful: SWE Gym/OpenHands could not find SWE-rebench SIF images and the log had grown to about `1.5 GiB` of repeated missing-SIF exceptions.
- Dependent matrix jobs `2118719` baseline, `2118720` suffix, `2118721` PARD, `2118724` PARD-2, and `2118725` Eagle-3 were also cancelled. They should be relaunched only after image staging or dataset filtering.
- Lyris login does not expose the config's original `/lustre/fsw/portfolios/llmservice/users/sdevare/...` dataset/image paths.
- Lyris-visible target image root `/lustre/fsw/coreai_dlalgo_llm/users/sna/images/swerebench` exists. Full staging of the `1,689` required subset is still running as four local shard streams; the latest Lyris-side dataset refresh found `607` staged `.sif` files, `4` active temporary files at the instant checked, and about `285G` in the target directory.
- A filtered staged-SIF smoke dataset was refreshed at `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/filtered/staged_sif_20260614_2010/train.jsonl` with `607` train rows. The active r7/r8 jobs use the earlier `504`-row snapshot; this `607`-row snapshot is for the next relaunch.
- The r3/r4 step-1 smoke jobs are no longer the active proof point: r3 baseline `2120957` failed on an empty vLLM compiled-DAG channel env, r3 PARD `2120958` failed/cancelled from the same bad-env submission, and r4 PARD-2 `2120970` was cancelled. The r5 baseline `2121065` and PARD `2121066` then proved the staged-SIF setup could progress through model/Ray/vLLM setup into SWE rollout, but exposed a new launch-layer blocker: the inner SWEBench agent SIF mount failed inside Pyxis because `/dev/fuse` was not mounted. PARD-2 r6 `2121079` was also cancelled because its running allocation had the same missing `/dev/fuse` launch shape. The current proof point is r7 baseline `2121104`, submitted after patching the launcher to mount `/dev/fuse:/dev/fuse`; worker logs confirm the mount, the job reached `SETUP COMPLETE`, async GRPO startup, policy-generation refit, and continued live rollout collection with no FUSE/no-completion/ValueError/EngineDead marker. Latest fetched log shows eight completed `32/32` rollout progress bars, up from six in the prior fetch, so the run is still moving. It has not yet emitted `global_step` or checkpoint/step-complete metrics. The matching r8 PARD `2121199` and PARD-2 `2121201` step-1 smoke jobs are also running with the same mount; their Ray clusters are up and their driver logs are still at TransformerEngine build.

## Missing SIF Set

- Parsed current running driver log:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/swerl_fullgrpo_logs/20260614_lyris_swerl_qwen235b_megatron_reconvert_wandb_r1/baseline_steps1/2118715-logs/ray-driver.log`
- Real unique missing SWE-rebench instance ids: `1,689`
- OCI-HSG source root checked:
  `/lustre/fsw/portfolios/llmservice/users/sdevare/images/swerebench`
- Source coverage: `1,689 / 1,689` SIF files found on OCI-HSG
- Combined source size: `808,810,123,264` bytes (`753.26 GiB`)
- Largest observed SIFs are about `1.5-1.75 GiB` each.

The local generated id list is under `tmp/lyris_swerl_missing_sif_ids_real_20260614.txt`.

## Implication

Staging a few smoke SIFs will not fix the current full rollout. A useful Lyris relaunch needs one of:

- Stage the current `1,689`-file subset, then relaunch with `SWE_REBENCH_IMAGE_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/images/swerebench`.
- Stage the full SWE-rebench image set if future runs should not depend on whichever instances the first rollout samples.
- Filter the training dataset to instances whose SIFs have been staged on Lyris.

The current running job cannot be fixed only by copying into the Lyris-visible root, because its config still points at the missing `/lustre/fsw/portfolios/llmservice/users/sdevare/...` roots. It needs a relaunch after staging or filtering.

## Transfer Path

Direct cross-cluster SSH is not currently usable:

- OCI-HSG to Lyris using `login-lyris.nvidia.com` resolves but times out on port `22`.
- Lyris to OCI-HSG using the full OCI hostnames resolves and reaches SSH, but publickey authentication is not available from Lyris.
- The local machine can SSH to both clusters, so staging needs to run through the local client or through another shared transfer system.

Helper script: `scripts/stage_lyris_swerebench_sifs_from_oci_20260614.sh`.

The helper streams one SIF at a time through the local client, skips complete destination files by size, and writes each destination through a temporary file before rename. Default mode is `DRY_RUN=true`; use `DRY_RUN=false` only when ready to start the approximately `753.26 GiB` transfer.

Staging progress:

- Foreground smoke transfer of the first `2` SIFs completed successfully in `3:32.42`.
- The initial single-stream session was replaced with four shard streams to improve throughput.
- Local progress logs: `tmp/lyris_swerebench_sif_stage_s0_20260614.log`, `tmp/lyris_swerebench_sif_stage_s1_20260614.log`, `tmp/lyris_swerebench_sif_stage_s2_20260614.log`, and `tmp/lyris_swerebench_sif_stage_s3_20260614.log`.
- At the latest dataset refresh, `607` staged ids matched the Lyris-visible image root. The local shard processes are still active and were writing four `.tmp.*` files in the target directory at that check.
- The latest target directory size is now about `285G`; the full source subset remains about `753.26 GiB`, so staging is still partial. Treat throughput estimates as moving observations, not scheduler guarantees.
- `scripts/refresh_lyris_staged_sif_smoke_dataset_20260614.sh` refreshes staged ids, remaining ids, and the filtered smoke JSONLs in one step. Using the latest `607` staged ids, it wrote `/lustre/fsw/coreai_dlalgo_llm/users/sna/datasets/swe/filtered/staged_sif_20260614_2010/train.jsonl` with `607` train rows and an empty `val.jsonl`; the latest full missing set still has `1,082` unstaged ids.
- `experiments/eagle3_online/submit_lyris_swerl_qwen235b_stagedsif_smoke_20260614.sh` wraps the filtered smoke launch and dry-runs baseline/PARD/PARD-2 step-1 cells by default against the latest `tmp/latest_lyris_staged_sif_smoke_dataset_20260614.env` snapshot (`staged_sif_20260614_1841/train.jsonl` at this check), with smoke SWE-agent train/val concurrency overridden down to `64` and `EXTRA_CONTAINER_MOUNTS` defaulting to `/dev/fuse:/dev/fuse`. The cancelled r5/r6 FUSE-rooted jobs are tracked in `latest_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_r5_envfix_20260614_jobs.csv`, `latest_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_r6_pard2_pyoverlay_envfix_20260614_jobs.csv`, and their matching status/log-summary docs. The r7 baseline proof is tracked in `latest_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_r7_fusemount_20260614_jobs.csv`, and the r8 SpecDec follow-up is tracked in `latest_lyris_swerl_qwen235b_fullgrpo_stagedsif_smoke_r8_fusemount_specdec_20260614_jobs.csv`; status output is generated at `docs/lyris_swerl_fullgrpo_stagedsif_smoke_r7_fusemount_status_20260614.md` and `docs/lyris_swerl_fullgrpo_stagedsif_smoke_r8_fusemount_specdec_status_20260614.md`.
