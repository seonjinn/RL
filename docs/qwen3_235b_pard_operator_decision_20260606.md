# Qwen3-235B PARD / PARD-2 Operator Decision

Date: 2026-06-06 PDT

This is the current execution-facing summary. It separates measured
generation benefits from still-unverified Full-GRPO E2E benefit.

## Decision Table

| Area | Best / Status | Evidence | Decision |
|---|---|---|---|
| vLLM standalone short synthetic | PARD K=12 bs32 | 3.290x, 92.95% acceptance | Useful sanity proof only; not representative of OpenMath or RL. |
| vLLM standalone OpenMath current local | local_pard_k5_dpace_draft_ce_2048_gate | job 3190567; 1.296x, 47.01% acceptance | Promote 2K dynamic D-PACE K5 as the current local checkpoint gate. |
| vLLM OpenMath public recalibration | public_pard_k5_current_harness_recal | job 3171868; 1.001x, 45.99% acceptance | Use as current-harness public-PARD reference; avoid mixing with older historical baseline. |
| vLLM OpenMath high batch | local_cat_k5 bs64 | 1.315x, 45.07% acceptance | bs64 still helps; bs128 drops to about 1.19x, so batching does not solve the issue alone. |
| NeMo-RL sync VllmGeneration | public PARD K5 job 3186417 | 1.522x, 42.29% acceptance | PARD path gives real in-launcher generation benefit. |
| NeMo-RL dynamic D-PACE K sweep | vllmgeneration_sync_dynamic_dpace_k3 job 3192349 | 1.454x, 57.50% acceptance | K3 is the best systems tradeoff in sync generation despite K1/K2 higher acceptance. |
| NeMo-RL latest-main vLLM0.20 generation-only | public_pard_k5 job 3197509 | 1.591x generation throughput only, 42.17% acceptance, mean accepted length 3.108 | Version-skew control is positive; use this runtime for the pending Full-GRPO E2E gate. |
| NeMo-RL worker32 stop-after-generation | K5 local PARD/CAT TPP-mask job 3175808 | 1.718x generation TPS, 53.53% acceptance | Generation-segment proof only; do not use as E2E Full-GRPO claim. |
| Qwen3-235B no-stop Full-GRPO | active 3207001 public PARD K3 mem70/bt8k skip-reference fuse_loss_off temp1 pending; recent failed 3198185 public PARD K3 mem70/bt8k failed; 3198324 baseline mem70/bt8k skip-reference failed_startup; 3198325 public PARD K3 mem70/bt8k skip-reference failed_startup; 3198380 public PARD K3 mem70/bt8k skip-reference failed; 3198436 public PARD K3 mem70/bt8k logchunk2048 failed; 3198648 public PARD K3 mem70/bt8k skip-reference fuse_loss_off failed | No parsed E2E throughput/step-time metric yet for Qwen3-235B no-stop Full-GRPO. | Main unresolved gate; default K3 still fails in reference logprobs, while skip-reference with fuse_loss=false gets past the packed-loss copy issue. The remote latest-main worktree now guards temperature-zero training scaling; poll temp1 job 3207001 for the next evidence point. |
| DFlash | DFlash K3 bs1 | max observed acceptance 1.22% | Do not promote current checkpoint; training/alignment issue, not runtime priority. |

## Do Not Repeat

| Pattern | Why not |
|---|---|
| Treat short synthetic PARD numbers as OpenMath evidence | Short `ISL=1000/OSL=512` shows high acceptance and up to `3.29x`, but OpenMath/current-harness results are much weaker. |
| Claim Qwen3-235B Full-GRPO E2E speedup from generation-only runs | Worker32 stop-after-generation proves generation segment benefit only; no no-stop Full-GRPO E2E metric has completed. |
| Call local CAT/D-PACE runs official PARD-2 | AMD PARD repo still has no public PARD-2 code/checkpoint; these are PARD-2-style approximations. |
| Scale teacher rows with the same objective blindly | 4K D-PACE regressed versus 2K D-PACE despite similar acceptance. Objective/runtime quality is the limiter, not only row count. |
| Promote the current DFlash checkpoint to NeMo-RL | Runtime works, but held-out OpenMath acceptance is only about `0.25-1.22%`. |
| Re-run greedy `temperature=0.0` diagnostics without the training guard | The old training path divided logits by zero for temperature zero. The remote latest-main worktree now guards non-positive temperatures; use the patched worktree only. |
| Interpret MoE policy-construction failures as PARD-quality failures | Qwen3-30B-A3B failed on Megatron `SequentialMLP deepcopy(config)` touching `ProcessGroup`; the shallow-copy patch is needed before interpreting MoE results. |

## Immediate Resume

Current priority:

```bash
ssh oci-hsg-cs-001-vscode-02 'squeue -j 3207001; sacct -j 3207001 -o JobID,JobName%80,State,Elapsed,ExitCode -P'
```

The priority evidence gate is Qwen3-235B no-stop Full-GRPO E2E
throughput/step-time. Until that lands, the correct claim is:
PARD/PARD-2-style improves Qwen3-235B generation in NeMo-RL gates,
but Qwen3-235B Full-GRPO E2E benefit remains unverified.
