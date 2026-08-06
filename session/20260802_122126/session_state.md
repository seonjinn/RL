# Session State

- Session: 20260802_122126
- Repo: /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731
- Branch: experiment/thd-cg-hybrid-nemotron-20260731
- Started: 2026-08-02 12:21:26 PDT
- Updated: 2026-08-02 19:13:37 PDT

## Goal

Establish correctness and performance of Transformer Engine partial CUDA Graph scopes for packed MoE policies, then extend validated settings from Qwen3-30B-A3B to Qwen3-235B-A22B.

## Current Subtask

Finish the source review and local verification for the reproducible
Qwen3-30B-A3B and Qwen3-235B-A22B campaign, then push it and create the exact
OCI-HSG runtime attestation required before the first GPU smoke.

## Loaded Skills

- `e2etrain:ssh-slurm` - cluster preflight, submission, monitoring, and failure triage.
- `nemo-rl-auto-research` - experiment lifecycle and reproducibility ledger.
- `nemo-rl-session-memory` - durable campaign state and handoff.
- `superpowers:systematic-debugging` - evidence-first root-cause isolation from the preceding diagnostic turn.
- `superpowers:brainstorming` - design gate before adding the Qwen3-235B model selector or launch behavior.
- `.agents/contributor-skills/testing` - recipe and model-support test conventions.

## Current Status

Implementation through `75ddbef3d` contains the safe Qwen A/B/C/E launch
matrix, immutable gate/profile/runtime bindings, self-validating Router Replay
execution, exact submission metadata, identity-safe TensorBoard/W&B exporters,
and first-class `cache_miss_count`: warming and capture are misses, hit is not,
and runtime
validation enforces `capture_count <= cache_miss_count` plus
`eviction_count <= capture_count`. Local focused verification is green:
89 lifecycle, 70 policy-worker/packing, and 59 algorithm telemetry tests.
Qwen30 and Qwen235 `TEST_ONLY=1` smoke renders pass without scheduler contact.
No campaign GPU job has been submitted from this branch. OCI-HSG job `5794372`
is useful prior Qwen235 readiness evidence, but is not a matched CG baseline.
The final documentation audit also found that the former Qwen235 C/E R3 gate
was self-attested. R3 gate validation is now disabled until a content-bound
Slurm diagnostic producer exists; Qwen235 A/B remain runnable.
The reviewed branch was pushed through `f9673c5a0`. OCI setup is currently
paused before any remote mutation because GlobalProtect is disconnected and
the internal cluster host does not resolve. The agents were reloaded and the
GlobalProtect UI opened; fresh Connect/SAML/MFA completion is required.

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
- [x] Add identity-safe TensorBoard/W&B export and exact graph telemetry.
- [x] Render Qwen30 and Qwen235 smoke matrices in offline `TEST_ONLY=1` mode.
- [x] Complete the dedicated final source review and fail-closed probes; the
  review verdict is `ADDRESSED` with 0 critical findings, warnings, or nits.
- [x] Reject Qwen use through legacy generic performance/accuracy wrappers and
  disable self-attested Qwen235 R3 evidence before scheduler contact.
- [x] Commit fail-closed unattested-route remediation as `75ddbef3d`.
- [x] Commit the final runbook/session ledger and push the reviewed branch
  through `f9673c5a0`.
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
  fail-closed because route IDs are not explicit graph replay inputs. The safe
  R3 graph comparison is arm E (`attn` only, router eager).
- Qwen235 C/E additionally require a new producer that binds raw diagnostics,
  exact argv, exit status, Slurm identity, and runtime provenance. Until then,
  only Qwen235 A/B may run.
- A remote checkout and fresh runtime attestation are required before any
  non-TEST_ONLY submission.
- OCI internal DNS is unavailable until the user completes a fresh
  GlobalProtect Connect/SAML/MFA flow. No GPU is allocated while blocked.
