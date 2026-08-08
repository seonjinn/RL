# HybridEP + THD Partial CUDA Graph Correctness Plan

**Goal:** Establish a source-bound, reproducible correctness gate for NeMo-RL using
dropless HybridEP with packed THD and Transformer Engine partial CUDA Graphs, then
measure the proven scope without silently claiming unsupported RouterReplay or
`moe_preprocess` coverage.

**Approved design:** This is the execution delta for
`2026-07-31-nemotron-thd-te-cuda-graph-correctness-design.md` and
`2026-08-04-dropless-partial-moe-capacity-policy.md`. The production scope starts
with `attn,mamba,moe_router`; dispatch, expert compute, combine, postprocess, and
`moe_preprocess` remain eager until their own distributed parity gates pass.

## Fixed source state

- NeMo-RL worktree: `RL-thd-cg-hybrid-nemotron-main-20260806`
- NeMo-RL branch: `experiment/thd-cg-hybrid-nemotron-main-20260806`
- Megatron-Bridge branch: `sna/thd-cg-hybrid-nemotron-main-20260806`
- Megatron-Core branch: `sj/thd-cg-hybrid-nemotron-main-20260806`
- Current NeMo-RL SHA: `4fa21f4e744801f9e74bfaf878eaf44aa764a253`
- Current Bridge SHA: `ce248fdd4da0de710d7f6968c9fb82843f41d7f1`
- Current MCore SHA: `eba4308af850562c2fdb677cfac1cdb4d0eb9f20`
- Proven pre-merge MCore candidate: `c3a5fc29a`
- Current upstream MCore main merged by `eba4308af`: `59b72fa57`
- Latest fetched upstream MCore main: `07a23196d`

## Safety constraints

- Use dropless HybridEP: both expert and rank capacity factors unset; no
  drop-and-pad mode.
- Keep router probabilities in FP32.
- Use exactly three successful optimizer steps for graph warmup.
- Do not graph `moe_preprocess` with uneven runtime padding or EP overlap.
- Keep RouterReplay fail-closed for `moe_router` and `moe_preprocess` until its
  active-bank input ownership and 1F1B lifetime tests exist in the pinned source.
- Unknown THD physical signatures must use a pre-forward recapture or eager
  fallback at a globally safe boundary; never reset the graph bank per step.
- Correctness gates precede performance submission.
- SLURM jobs use batch, full four-GPU GB200 nodes, no exclusive subset request,
  checkpointing disabled, and W&B project `sna-cg-study`.

## Task 1: Repair the source/manifest mismatch

- [ ] Preserve the RED evidence by running candidate-wide literal-node collection
  against MCore `eba4308af`; it must report the absent graph-bank, RouterReplay,
  and distributed HybridEP nodes.
- [ ] Merge the proven MCore candidate `c3a5fc29a` into the latest-main MCore
  branch, resolving upstream conflicts without dropping either implementation.
- [ ] Merge fetched upstream main `07a23196d` after the candidate integration and
  repeat the focused regression gates against that exact result.
- [ ] Remove manifest rows that still name RouterReplay tests not present in the
  merged candidate. Keep the corresponding NeMo-RL scope gate fail-closed.
- [ ] Re-run candidate-wide literal-node collection and require every remaining
  manifest node to resolve exactly.

## Task 2: Validate the real distributed HybridEP contract

- [ ] Confirm that `dropless_hybridep_nano16` initializes TP2/PP2/CP2/EP8 and the
  real Flex/HybridEP backend, not a mocked dispatcher.
- [ ] Require 3 warmups and 20 A/B changed-route replays with identical physical
  THD geometry.
- [ ] Compare valid-token outputs, loss, selected route IDs/probabilities/counts,
  every local gradient, and simulated optimizer deltas between eager and graph.
- [ ] Require nonzero graph calls for the requested partial scope and eager calls
  for dispatch/expert/combine/postprocess.
- [ ] Require zero dropped assignments, zero valid-token drops, and zero HybridEP
  rank-overflow events.

## Task 3: Preserve fail-closed NeMo-RL configuration

- [ ] Add or update focused tests for dropless HybridEP
  `attn,mamba,moe_router` acceptance.
- [ ] Reject packed HybridEP router scopes when expert capacity, rank capacity,
  or drop-and-pad is enabled; these modes currently budget physical padding rows.
- [ ] Reject fixed-THD quantile and sinkhorn routing until their padding-aware
  semantics are proven.
- [ ] Verify RouterReplay plus graphed router/preprocess is rejected.
- [ ] Verify `moe_preprocess` plus overlap or uneven runtime padding is rejected.
- [ ] Explicitly declare and forward `overlap_moe_expert_parallel_comm`; do not
  rely on a YAML value that the provider never receives.
- [ ] Verify model-specific Mamba scope and `cuda_graph_warmup_steps: 3` reach the
  MCore model configuration unchanged.

## Task 4: Pin, submit, and measure

- [ ] Run focused local/static tests and `git diff --check` in all three repos.
- [ ] Commit and push MCore, then update/push Bridge, then update/push NeMo-RL.
- [ ] On the selected GB200 cluster, pull the immutable candidate, run scheduler
  `--test-only`, then submit the 16-rank MCore correctness gate.
- [ ] Monitor for at least five minutes and cancel any allocation stalled before
  rank initialization.
- [ ] Only after the gate passes, submit matched NeMo-RL baseline and
  `attn,mamba,moe_router` rows: 5-step smoke, 20-step performance, then 100-step
  accuracy.
- [ ] Report E2E/logprob/policy-training/generation step time and
  tokens/s/GPU, graph-call coverage, reward, KL/error metrics, losses, and route
  parity.

## Task 5: Keep the report reproducible

- [ ] Update the experiment context and self-contained explain-diff HTML with
  exact SHAs, commands, job IDs, pass/fail evidence, and measured results.
- [ ] Clearly separate proven, failed, blocked, and not-yet-tested scopes.
