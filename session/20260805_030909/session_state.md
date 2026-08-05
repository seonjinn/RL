# Session State

- Session: 20260805_030909
- Repo: `/Users/sna/MXFP8_generation/.worktrees/nemorl-bf16-nvfp4-nccl`
- Branch: `sna/bf16-nvfp4-nccl-reshard`
- Started: 2026-08-05 03:09 PDT
- Updated: 2026-08-05 03:19 PDT

## Goal

Validate plain BF16 Megatron training with real NVFP4 W4A16 and W4A4 vLLM
rollout on GCP-NRT B200, including legacy and NCCL-Reshard refit paths.

## Current Subtask

Run committed two-step W4A16 legacy and NCCL-Reshard smoke tests, then prepare
the provenance-checked W4A4 calibration artifact and run the W4A4 pair.

## Loaded Skills

- `nemo-rl-auto-research` - reproducible GPU experiment lifecycle.
- `nemo-rl-session-memory` - durable state for long-running jobs.
- `e2etrain:ssh-slurm` - GCP-NRT SLURM submission and monitoring.
- `testing` - NeMo-RL recipe and test conventions.

## Current Status

- The branch is clean and matches `fork/sna/bf16-nvfp4-nccl-reshard` at
  `46b7621afe28b773fd0c3fded048a28d20d967cb`.
- W4A16 and W4A4 rollout-only recipes and smoke contracts exist.
- No GPU job has been submitted for this exact BF16-training/NVFP4-rollout
  combination yet.
- GCP-NRT FairShare is 0.951794 for `coreai_chef_posttrain`.
- No existing provenance-valid W4A4 calibration artifact was found.
- Both smoke scripts pass `bash -n` and their static `TEST_DRYRUN` contracts.
- Local pytest cannot resolve the uninitialized Gym workspace in this worktree;
  the focused test must run in the recursive-submodule GCP clone/container.
- Added a GCP-NRT launcher with unique snapshots and caches,
  `--gpus-per-node=8`, scheduler preflight, and secure W&B credential loading.
- The recursive-submodule GCP clone is ready at commit `441dc40df`.

## Plan

- [x] Run local static smoke-script tests.
- [ ] Run focused recipe tests in the target container.
- [x] Create or refresh the committed branch on GCP-NRT.
- [ ] Run scheduler preflight for W4A16 legacy and NCCL-Reshard.
- [ ] Submit and monitor both W4A16 jobs for at least five minutes.
- [ ] Generate and verify the W4A4 calibration artifact.
- [ ] Submit and monitor W4A4 legacy and NCCL-Reshard.
- [ ] Record job IDs, W&B links, metrics, and failures in a report.

## Assumptions

- The vLLM 0.25.1 nightly container used by the PR 3477 GCP measurements is
  compatible with this branch; runtime validation must confirm this.

## Blockers

- W4A4 requires a real calibration artifact before submission.
