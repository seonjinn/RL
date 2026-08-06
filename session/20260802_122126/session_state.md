# Session State

- Session: 20260802_122126
- Repo: /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-latest-main-20260804
- Branch: experiment/thd-cg-hybrid-nemotron-latest-main-20260804
- Started: 2026-08-02 12:21:26 PDT
- Updated: 2026-08-05 23:54:00 PDT

## Goal

Establish correctness and performance of Transformer Engine partial CUDA Graph
scopes for packed Nemotron 3 policies. For Nano, measure every valid subset of
`attn`, `mamba`, `moe_router`, and `moe_preprocess`, then determine whether the
all-enabled scope is the fastest graph-safe configuration.

## Current Subtask

Launch a matched 20-step Nano matrix on OCI-HSG using the exact persistent graph
bank implementation that completed the 100-step baseline and attention runs.

## Loaded Skills

- `e2etrain:ssh-slurm` - cluster preflight, submission, monitoring, and failure triage.
- `nemo-rl-auto-research` - experiment lifecycle and reproducibility ledger.
- `nemo-rl-session-memory` - durable campaign state and handoff.
- `superpowers:using-git-worktrees` - preserve the existing isolated worktree.
- `superpowers:brainstorming` - consulted for possible launcher changes; no new launcher behavior is needed because persistent scope leaves already cover the approved matrix.

## Current Status

Campaign source `e95e40325` is pushed as
`experiment/nano-cg-4axis-matrix-20260805`. A clean recursive OCI-HSG worktree
uses Bridge `0142aebf` and MCore `281200606`. The first 12 jobs (`5912632` to
`5912655`) failed before NeMo-RL startup because the direct launcher's stale
2026-08-01 container could not resolve project Python 3.13.14. The successful
100-step runs used the newer 2026-08-05 nightly ending in runtime job `5884993`.
A corrected matrix now uses that exact image. All 12 corrected jobs (`5913139`,
`5913180`, `5913182`, `5913184`, `5913186`, `5913188`, `5913190`, `5913192`,
`5913194`, `5913196`, `5913198`, and `5913200`) are running without
dependencies. Every row loaded the NeMo-RL config with Python 3.13.14 and
initialized all eight vLLM workers; no fatal runtime marker was present during
the first five minutes. Generation-side vLLM graph capture is in progress.

## Plan

- [x] Merge the persistent graph-bank experiment with latest NeMo-RL main.
- [x] Define the valid four-axis Nano scope matrix.
- [x] Commit and push the exact campaign source SHA.
- [x] Check OCI-HSG scheduling with `sbatch --test-only`.
- [x] Submit one matched 20-step run for baseline and each valid scope.
- [x] Monitor the corrected matrix for at least five minutes and verify Python/config entry.
- [ ] Collect phase timing, throughput, CUDA Graph coverage/cache telemetry, memory, and correctness metrics.
- [ ] Repeat only the stable leading configurations after the first complete matrix.

## Assumptions

- All rows use the same Nano performance recipe, seed, 24 GPUs, all-to-all dispatcher, sequence packing, fused attention, warmup 3, no checkpoints, container, and source SHA.
- `moe_preprocess` requires `moe_router`; its isolated incremental effect is the comparison between router and router-plus-preprocess.
- The all-enabled scope is a performance hypothesis, not a guaranteed result. It wins only if extra graph coverage outweighs graph-bank misses, recaptures, padding, and replay overhead.

## Blockers

- None known. The next gate is successful policy-worker initialization and first optimizer step for every scope.
