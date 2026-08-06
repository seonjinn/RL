# Session State

- Session: 20260802_122126
- Repo: /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-latest-main-20260804
- Branch: experiment/thd-cg-hybrid-nemotron-latest-main-20260804
- Started: 2026-08-02 12:21:26 PDT
- Updated: 2026-08-05 23:21:25 PDT

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

Local `main` at `4ed047b48` merges current `origin/main` (`ae07eafe8`) with the
persistent graph-bank experiment and interactive explainer. The active isolated
worktree was fast-forwarded to that same commit. Focused merged-tree validation
passed 100 CUDA Graph lifecycle/storage tests and 8 explainer tests. OCI-HSG
100-step baseline job `5908997` and attention job `5909007` both completed with
exit code zero. The existing persistent scope leaves cover all 11 non-baseline
valid subsets, so no launcher implementation change is required.

## Plan

- [x] Merge the persistent graph-bank experiment with latest NeMo-RL main.
- [x] Define the valid four-axis Nano scope matrix.
- [ ] Commit and push the exact campaign source SHA.
- [ ] Check OCI-HSG scheduling with `sbatch --test-only`.
- [ ] Submit one matched 20-step run for baseline and each valid scope.
- [ ] Monitor every submitted job for at least five minutes.
- [ ] Collect phase timing, throughput, CUDA Graph coverage/cache telemetry, memory, and correctness metrics.
- [ ] Repeat only the stable leading configurations after the first complete matrix.

## Assumptions

- All rows use the same Nano performance recipe, seed, 24 GPUs, all-to-all dispatcher, sequence packing, fused attention, warmup 3, no checkpoints, container, and source SHA.
- `moe_preprocess` requires `moe_router`; its isolated incremental effect is the comparison between router and router-plus-preprocess.
- The all-enabled scope is a performance hypothesis, not a guaranteed result. It wins only if extra graph coverage outweighs graph-bank misses, recaptures, padding, and replay overhead.

## Blockers

- None known. OCI-HSG SSH and the last successful runtime checkout are reachable.
