# Session State

- Session: 20260802_122126
- Repo: /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731
- Branch: experiment/thd-cg-hybrid-nemotron-20260731
- Started: 2026-08-02 12:21:26 PDT
- Updated: 2026-08-02 15:54:34 PDT

## Goal

Establish correctness and performance of Transformer Engine partial CUDA Graph scopes for packed MoE policies, then extend validated settings from Qwen3-30B-A3B to Qwen3-235B-A22B.

## Current Subtask

Document and locally verify the reproducible Qwen3-30B-A3B and
Qwen3-235B-A22B campaign that separates CUDA Graph effects from
vLLM-to-Megatron router mismatch.

## Loaded Skills

- `e2etrain:ssh-slurm` - cluster preflight, submission, monitoring, and failure triage.
- `nemo-rl-auto-research` - experiment lifecycle and reproducibility ledger.
- `nemo-rl-session-memory` - durable campaign state and handoff.
- `superpowers:systematic-debugging` - evidence-first root-cause isolation from the preceding diagnostic turn.
- `superpowers:brainstorming` - design gate before adding the Qwen3-235B model selector or launch behavior.
- `.agents/contributor-skills/testing` - recipe and model-support test conventions.

## Current Status

The source worktree was clean at `e1c24cd9d` before these session notes were generated. The existing harness supports Qwen3-30B-A3B but not Qwen3-235B-A22B. Qwen3-30B-A3B has both a 4-node performance recipe and an 8-node Router Replay recipe. Latest main is one commit ahead and already contains a validated Qwen3-235B-A22B 16n4g recipe. OCI-HSG job `5794372` completed that 64-GPU recipe for 20 steps, proving the model snapshot and topology are available; it is not a matched CG baseline because it used a separate source branch and W&B project. Current evidence does not prove an eager-versus-CG router defect; the strongest hypothesis is a rare vLLM Triton versus eager Megatron route mismatch. Router Replay plus reusable router CUDA graphs has a separate stale-route-input risk and must not be treated as a safe production comparison until routed expert IDs become explicit graph inputs or persistent buffers.

## Plan

- [x] Inventory prior Qwen runs, recipes, cluster state, and reusable reporting paths.
- [x] Present staged alternatives and obtain user approval before adding selectors or launching GPU jobs.
- [x] Obtain user review of the committed written design before implementation.
- [x] Merge the current NeMo-RL main into the experiment branch while preserving
  the Bridge and MCore gitlinks.
- [x] Add the Qwen3-235B-A22B 16n4g selector and the persistent A/B/C/E
  validation condition launchers.
- [x] Make R3 plus `moe_router`/`moe_preprocess` fail before any scheduler call.
- [x] Add R3 identity and trace fields to normalized result collection/reporting.
- [ ] Run the dedicated final source review, then push the reviewed branch.
- [ ] Create the remote OCI campaign checkout and fresh four-GPU runtime
  attestation.
- [ ] Dry-run and submit the approved Qwen3-30B-A3B comparison on OCI-HSG.
- [ ] Gate Qwen3-235B-A22B submission on the 30B smoke/correctness result.
- [ ] Collect E2E, generation, logprob, policy-training, CUDA Graph coverage, and correctness metrics into the HTML report.

## Assumptions

- Performance runs use 20 optimizer steps, three CUDA Graph warmup steps, sequence packing, no checkpoints, and W&B project `sna-cg-study` unless the user changes these constraints.
- `Qwen3-235B` means `Qwen/Qwen3-235B-A22B`, matching the checked-in performance recipe.

## Blockers

- The concrete OCI runtime profile must be regenerated for this branch before
  an attested matrix launch.
- R3 plus `moe_router`/`moe_preprocess` CUDA Graph reuse remains intentionally
  unsupported because route IDs are not explicit graph replay inputs.
- A final source review, remote checkout, and fresh runtime attestation are
  required before any non-TEST_ONLY submission.
