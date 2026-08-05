# Session State

- Session: 20260804_131500
- Repo: `/Users/sna/MXFP8_generation/.worktrees/nemorl-pr3477-perf-ab`
- Branch: `sna/pr3477-perf-ab`
- Started: 2026-08-04 13:15 PDT
- Updated: 2026-08-04 20:10 PDT

## Goal

Measure and summarize the refit-time effects of NeMo-RL PR 3477 and PR 3478.

## Current Subtask

Run a matched current-head PR 3477 legacy versus NCCL-Reshard A/B on GCP-NRT.

## Loaded Skills

- `nemo-rl-auto-research` - reproducible experiment lifecycle.
- `nemo-rl-session-memory` - durable state for the long-running jobs.
- `e2etrain:ssh-slurm` - GCP-NRT SLURM operations.
- `nemo-rl-wandb-reporting` - matched step-window reporting.

## Current Status

Jobs `496508` and `496509` recorded all 20 steps. Legacy completed with exit 0.
The NCCL arm completed the measured workload and W&B finalization, then exited
1 in Ray interpreter shutdown because the core worker was already initialized.
All requested W&B metrics are present for steps 3-20.

## Plan

- [x] Validate the two-arm launcher with `sbatch --test-only`.
- [x] Submit both 20-step arms.
- [x] Monitor both arms through step 20.
- [x] Extract steps 3-20 and write the team summary.

## Assumptions

- Measurements are labeled PR 3477 plus the required vLLM 0.25 runtime fixes.

## Blockers

- None.
