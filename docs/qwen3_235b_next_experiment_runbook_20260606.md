# Qwen3-235B Next Experiment Runbook

Date: 2026-06-06 PDT

Current E2E claim-gate state: **1 positive E2E rows, 5 pending metric rows**.

This runbook is the operational companion to the team report. It defines
the next actions, success criteria, and failure interpretations so the
same negative paths are not repeated.

## Current Poll Command

```bash
squeue -j 3197584,3197585,3197586
```

These are the latest-main/nightly Qwen3-235B public PARD Full-GRPO jobs.
Once they start producing logs, parse E2E step-time/throughput before
making any performance claim.

## Experiment Matrix

| Priority | Experiment | Jobs | Current status | Success criterion | Failure interpretation | Next action |
|---|---|---|---|---|---|---|
| P0 | Qwen3-235B latest-main/nightly public PARD no-stop Full-GRPO step5 | baseline 3197584; public PARD K5 3197585; public PARD K3 3197586 | submitted_pending; NeMo-RL latest main nightly with VllmGenerationWorker vLLM 0.20.0 | Parsed no-stop Full-GRPO E2E throughput or E2E step-time speedup > 1.0 for K3 or K5 against job 3197584. | If launch fails, fix latest-main/nightly wrapper/submodule/env first; if generation is positive but E2E is not, treat RL tail/MoE/logprob/orchestration cost as the likely bottleneck. | Poll jobs 3197584/3197585/3197586; parse E2E and generation metrics once logs are available. |
| P0 | Qwen3-235B latest-main/nightly public PARD no-stop Full-GRPO step20 | baseline 3197620 afterok:3197584; public PARD K5 3197621 afterok:3197585; public PARD K3 3197622 afterok:3197586 | submitted_dependency; runs only if the corresponding step5 job succeeds | 20-step E2E throughput or E2E step-time speedup > 1.0 for K3 or K5 against baseline 3197620. | If step5 succeeds but step20 times out, use step5 for functional E2E and resubmit step20 with a longer allowed partition or fewer metrics/log overhead. | Wait for step5 launcher validation, then use step20 for more stable E2E averages. |
| P0 | Qwen3-235B no-stop Full-GRPO local CAT/TPP-mask PARD K5 | 3195285 | pending_last_seen; older worktree/local CAT checkpoint path | Parsed no-stop Full-GRPO row with E2E throughput or E2E step-time speedup > 1.0 against matched baseline. | If generation speedup exists but E2E <= 1.0, treat as RL tail / MoE verification / orchestration overhead, not drafter-only failure. | Keep as secondary local-checkpoint validation; primary current path is latest-main public PARD jobs 3197584/3197585/3197586. |
| P0 | Qwen3-235B no-stop Full-GRPO sampling-step4 dynamic D-PACE K3/K5 | baseline 3186510; CAT K5 3186511; D-PACE K5 3192180; D-PACE K3 3192438 | missing_log in local status CSVs | K3 or K5 has numeric E2E speedup > 1.0; prefer the setting with better E2E step-time and stable acceptance. | If K3 wins generation but not E2E, optimize RL tail/scheduling before more drafter training. | Remote refresh, then compare qwen3_235b_fullgrpo_e2e_claim_gate_20260606.csv rows. |
| P1 | Qwen3-30B-A3B post-shallow-copy MoE validation | baseline 3195815; local CAT/PARD-2-style K5 3195816 | submitted_post_patch; secondary MoE validation path | MoE policy construction passes SequentialMLP ProcessGroup shallow-copy failure and reaches Full-GRPO metrics. | If ProcessGroup error repeats, patch did not apply to active checkout or another deepcopy path exists. | Run scripts/apply_remote_megatron_moe_pg_shallowcopy_patch.sh before polling. |
| P1 | Qwen3-235B public PARD K5 fixed256 no-stop diagnostic | baseline 3186342; local CAT K5 3186343; public PARD K5 3186344 | missing_log in local status CSVs | Public PARD K5 has numeric E2E speedup > 1.0 or clearly isolates local checkpoint quality from runtime overhead. | If public PARD and local CAT both lose E2E, focus on systems/runtime rather than CAT objective. | Refresh remote fixed256 status and inspect claim gate. |
| P1 | vLLM version-skew control for Qwen3-235B PARD | generation control 3197507/3197508/3197509; Full-GRPO step5 3197584/3197585/3197586; step20 3197620/3197621/3197622 | generation-only control passed on NeMo-RL latest-main/nightly vLLM 0.20.0: K3 1.498x, K5 1.591x | Use latest-main/nightly Full-GRPO jobs to decide E2E; version skew is no longer the generation-path blocker. | If Full-GRPO loses despite vLLM0.20 generation speedup, focus on E2E cost buckets rather than vLLM version first. | Poll latest-main Full-GRPO jobs and update the claim gate. |
| P2 | DFlash Qwen3-235B follow-up | historical retry28 metrics only | do_not_promote; OpenMath acceptance only ~0.25-1.22% | Before NeMo-RL, a held-out OpenMath standalone gate should show materially improved acceptance and >1.0 speedup. | Near-zero acceptance means checkpoint alignment/training failure, not a NeMo-RL integration issue. | Do not spend NeMo-RL nodes until a better DFlash checkpoint is trained and gated. |
| P2 | SpecKV/adaptive-K controller | not submitted | design follow-up | Use only after no-stop Full-GRPO K3/K5 rows exist; controller should improve E2E, not just acceptance. | Controller without E2E telemetry risks optimizing the wrong K. | Wait for K3/K5 Full-GRPO metrics, then implement confidence/accepted-length gating if useful. |
| P2 | SpecDecode-Bench-style Qwen3-235B cost breakdown | not submitted | design follow-up; current report has speedup/acceptance but not full draft-vs-verify overlap timing inside Full-GRPO | For a parsed Full-GRPO pair, split generation time into draft, target verification, scheduler/orchestration, and RL tail components. | If acceptance is high but E2E <= 1.0, missing cost buckets make it impossible to choose training versus systems fixes. | Add instrumentation only after the pending Full-GRPO rows parse, then prioritize the largest non-overlapped bucket. |
| P2 | SPECTRE-style draft/verify overlap feasibility check | not submitted | tracked systems fallback; public implementation path is SGLang-side, not NeMo-RL/vLLM drop-in | Show that Qwen3-235B Full-GRPO loses mainly to draft/verify serialization or target verification overhead despite generation-path speedup. | If the loss is RL tail/logprob/training dominated, SPECTRE-style overlap will not fix the bottleneck by itself. | Use only after cost breakdown proves verification/serialization is the dominant E2E gap. |
| P2 | SpecForge/EAGLE-3 retraining fallback for Qwen3-235B-A22B | not submitted | tracked training fallback; current local path is PARD/PARD-2-style, not SpecForge/SGLang | Train or obtain a Qwen3-235B-A22B draft model with held-out OpenMath acceptance and standalone speedup clearly above current public EAGLE3/PARD baselines. | Do not start this if Full-GRPO loss is systems overhead rather than drafter acceptance/accepted-length quality. | Use as next training branch only if PARD-2-style K3/K5 has weak acceptance/accepted length on matched OpenMath prompts. |

## Promotion Rules

- Promote a Qwen3-235B method only when the E2E claim gate reports
  `claim_allowed_positive_e2e` for a matched no-stop Full-GRPO pair.
- Treat worker32 stop-after-generation as generation-segment proof only.
- Treat sync `VllmGeneration` as runtime-path proof, not training-loop E2E proof.
- Do not call local CAT/D-PACE checkpoints official PARD-2 until upstream
  PARD-2 code/checkpoints are public and matched.
- Do not spend NeMo-RL nodes on DFlash until held-out OpenMath acceptance
  and standalone throughput are both positive.
- If Qwen3-235B generation speedup is positive but Full-GRPO E2E is not,
  split the next step by evidence: systems overlap/instrumentation for
  draft/verify overhead, training fallback for low accepted length.
- vLLM version skew has a positive generation-only control now:
  latest-main/nightly NeMo-RL worker vLLM 0.20.0 gives K5 1.591x
  generation throughput on OpenMath bs32/o256.

## Contingency Logic

| Observation | Interpretation | Next method family |
|---|---|---|
| High acceptance / accepted length but E2E <= 1.0 | Runtime/system cost dominates | SpecDecode-Bench-style breakdown, then SPECTRE-style overlap if verify serialization is dominant |
| Positive latest-main vLLM0.20 generation speedup but no Full-GRPO E2E speedup | RL tail/logprob/training/orchestration cost dominates | E2E cost breakdown and targeted systems fix |
| Low OpenMath acceptance or mean accepted length | Draft objective/domain mismatch | PARD-2/CAT/D-PACE refinement, SpecForge/EAGLE-3 retraining fallback |
| Dense Qwen3-32B works but MoE Qwen3-30B/235B fails before rollout | MoE construction/runtime issue | Megatron MoE shallow-copy patch, MoE-specific verification follow-up |
| DFlash acceptance remains near zero | Checkpoint alignment failure | Do not promote DFlash to NeMo-RL until standalone gate is positive |

## Related Entry Points

```text
docs/qwen3_235b_team_report_20260606.html
docs/qwen3_235b_fullgrpo_e2e_claim_gate_20260606.md
docs/qwen3_235b_goal_completion_audit_20260606.md
docs/qwen3_235b_specdec_cost_model_20260606.md
```
