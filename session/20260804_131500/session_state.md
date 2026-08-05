# Session State

- Session: 20260804_131500
- Repo: `/Users/sna/MXFP8_generation/.worktrees/nemorl-pr3477-perf-ab`
- Branch: `sna/pr3477-perf-ab`
- Started: 2026-08-04 13:15 PDT
- Updated: 2026-08-04 13:15 PDT

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

The historical trainer-prequant A/B and PR 3478 result are available. The
current PR 3477 receiver-quant head has no reportable performance run yet.

## Plan

- [ ] Validate the two-arm launcher with `sbatch --test-only`.
- [ ] Submit and monitor both 20-step arms.
- [ ] Extract steps 3-20 and write the team summary.

## Assumptions

- The staged vLLM 0.25.1 nightly and shared GCP-NRT caches remain valid.

## Blockers

- None known.
