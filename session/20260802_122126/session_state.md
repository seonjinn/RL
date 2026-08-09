# Session State

- Session: 20260802_122126
- Repo: /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-main-20260806
- Branch: experiment/thd-cg-hybrid-nemotron-main-20260806
- Started: 2026-08-02 12:21:26 PDT
- Updated: 2026-08-09 08:40:00 PDT

## Goal

Establish correctness and performance of Transformer Engine partial CUDA Graph
scopes for packed Nemotron 3 policies. For Nano, measure valid individual and
combined `attn`, `mamba`, `moe_router`, and `moe_preprocess` scopes and select
the fastest graph-safe configuration.

## Current Subtask

Promote the corrected HybridEP autograd-lifetime implementation into the nested
Bridge/NeMo-RL gitlinks, re-attest it on ptyche, and run a matched Nano 20-step
baseline versus partial-CUDA-Graph comparison.

## Loaded Skills

- `e2etrain:ssh-slurm` - cluster preflight, submission, monitoring, and failure triage.
- `nemo-rl-auto-research` - experiment lifecycle and reproducibility ledger.
- `nemo-rl-session-memory` - durable campaign state and handoff.
- `superpowers:systematic-debugging` - root-cause isolation before product changes.
- `superpowers:test-driven-development` - regression-first product fix.
- `superpowers:brainstorming` - explicit candidate-integration design choice.
- `mcore-testing` - targeted distributed test and exact result requirements.

## Current Status

NeMo-RL `c538a5442` is fully merged with current NeMo-RL main. MCore candidate
`2dbad0a2d` is pushed, includes current MCore main `d12f6c8c9`, and fixes two stale `_HybridEPManager` autograd references
by detaching manager-held `token_probs` after dispatch and
`dispatched_probs` after combine. ptyche job `2551742` passed the exact
4-node/16-GPU Nano HybridEP gate: three backward warmups, TE capture, 20
changing-route replays, output/loss/route/input-gradient/parameter-gradient
parity, structural THD padding, and optimizer-update parity. Its attestation is
under `experiment-logs/attestations/mcore/fc718cf4c.../`.

Bridge `2f6338610` is pushed, includes current Bridge main `355ef3ea`, and pins
MCore `2dbad0a2d`. The candidate snapshot path in `run_scope.sh` is not consumed
by `run_nemorl_scope.sub`, so NeMo-RL performance uses explicit nested gitlink
promotion and a refreshed runtime attestation rather than silently testing old
integration MCore `4013232a9`. NeMo-RL integration commit `4e5f9bac7` now pins
Bridge `2f6338610`.

## Plan

- [x] Isolate the HybridEP capture contamination to stale manager-held autograd references.
- [x] Implement and push the value-preserving detach fix plus focused regression coverage.
- [x] Pass the exact 16-GPU Nano HybridEP correctness gate on ptyche.
- [x] Merge current MCore and Bridge main into the candidate stack.
- [x] Promote Bridge `2f6338610` through the NeMo-RL gitlink.
- [ ] Re-run the exact MCore gate and refresh the ptyche source/runtime attestation.
- [ ] Submit matched 20-step Nano baseline and graph rows with warmup 3 and checkpoints disabled.
- [ ] Collect phase timing, throughput, CUDA Graph coverage/cache telemetry, memory, and correctness metrics.

## Assumptions

- All comparison rows use identical source, runtime, seed, recipe, topology,
  dispatcher, sequence packing, fused attention, and model snapshot.
- `moe_preprocess` requires `moe_router`; its isolated effect is router versus
  router-plus-preprocess.
- Twenty steps validate function and short-window metrics, not convergence.

## Blockers

- None. The next gate is the exact 16-GPU rerun on the merged candidate.
