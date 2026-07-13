# Qwen3-235B Operability Status

Date: 2026-06-06 PDT, latest update 2026-06-08 CEST / 2026-06-07 PDT

This separates runtime operability from the stricter Full-GRPO E2E
performance claim.

| Area | Status | Evidence | Implication |
|---|---|---|---|
| Remote access | ssh_working_recently | SSH/Slurm verified on 2026-06-08 CEST: fixed1024 Full-GRPO jobs 3209170/3209171 emitted all 3 timing steps. | Remote latest-main/nightly validation can be submitted and polled again. |
| Qwen3-235B vLLM standalone runtime | works_generation_path | short synthetic best 3.290x | PARD runtime can load and accelerate favorable Qwen3-235B standalone cases. |
| Qwen3-235B vLLM OpenMath current local | works_but_weaker | local_pard_k5_dpace_draft_ce_2048_gate job 3190567 1.296x acceptance 47.01% | OpenMath/domain and runtime cost reduce benefit; still positive for K5 local D-PACE. |
| Qwen3-235B NeMo-RL sync generation | works_generation_path | vllmgeneration_sync_dynamic_dpace_k3 job 3192349 1.454x acceptance 57.50% | NeMo-RL launcher/runtime can exercise Qwen3-235B PARD-style generation benefit. |
| Qwen3-235B NeMo-RL latest-main vLLM0.20 generation-only | works_generation_path | public_pard_k5 job 3197509 1.591x generation throughput only acceptance 42.17% | Version skew is no longer the blocker for generation-path benefit; this led to the positive non-colocated TP4 Full-GRPO branch. |
| Full-GRPO control path | works_on_qwen32_dense_control | Qwen3-32B PARD K5 E2E 1.045x, gen-time 1.172x, acceptance 49.4% | Full-GRPO integration can complete with PARD; Qwen3-235B now also has a direct non-colocated TP4 positive result. |
| Qwen3-235B no-stop Full-GRPO E2E | proven_positive_noncolocated_tp4 | Fixed256 3209048 vs 3209047, Step 2-5: E2E throughput 1.420x, total step-time 1.421x, generation-time 1.887x, acceptance 57.58%. Fixed1024 3209171 vs 3209170, Step 2-3: E2E throughput 1.591x, total step-time 1.580x, generation-time 1.797x, acceptance 50.66%. | Claim Qwen3-235B fixed256 and fixed1024 Full-GRPO E2E speedup for the non-colocated TP4 branch; keep 8K/16K Qwen3-235B open. |
| Historical Qwen3-235B colocated TP4 Full-GRPO jobs | superseded_by_noncolocated_tp4_result | completed non-colocated current branch: baseline_noncolocated_tp4_fixed256 3209047 completed_metrics; public_pard_k3_noncolocated_tp4_fixed256 3209048 completed_metrics; baseline_noncolocated_tp4_fixed1024_step3 3209170 completed_with_shutdown_recvbytes_noise; public_pard_k3_noncolocated_tp4_fixed1024_step3 3209171 completed_metrics / older colocated/scheduler snapshot: local CAT/TPP-mask PARD K5 3195285 failed; baseline mem80/bt16k 3198040 failed; public PARD K5 mem80/bt16k 3198041 failed; public PARD K3 mem80/bt16k 3198042 failed; baseline mem70/bt8k 3198183 failed; public PARD K5 mem70/bt8k 3198184 failed_startup; public PARD K3 mem70/bt8k 3198185 failed; baseline mem70/bt8k skip-reference 3198324 failed_startup; public PARD K3 mem70/bt8k skip-reference 3198325 failed_startup; public PARD K3 mem70/bt8k skip-reference 3198380 failed; public PARD K3 mem70/bt8k logchunk2048 3198436 failed; public PARD K3 mem70/bt8k skip-reference fuse_loss_off 3198648 failed; public PARD K3 mem70/bt8k skip-reference fuse_loss_off temp1 3207001 pending | Earlier colocated TP4 attempts exposed reference-logprob, packed-loss, vLLM wake-up, and host-memory failures. The current positive branch separates generation workers onto 4 generation-only nodes. |
| MoE policy construction risk | partially_mitigated_new_refit_failure | SequentialMLP ProcessGroup shallow-copy patch moved Qwen3-30B-A3B past the original construction failure, but jobs 3195815/3195816 then failed in MegatronPolicyWorker.prepare_refit_info when Megatron-Bridge MoE gate mapping parsed an empty expert suffix. | If Qwen3-235B fails before rollout/refit, check both the shallow-copy patch and MoE gate mapping/refit metadata conversion. |

## Current Answer

Qwen3-235B PARD/PARD-style paths are operational for standalone,
NeMo-RL generation tests, and NeMo-RL no-stop Full-GRPO in the
non-colocated TP4 branch. Public PARD K3 fixed256 `3209048` vs
baseline `3209047`, Step 2-5, gives E2E throughput `1.420x`,
total step-time `1.421x`, generation-time `1.887x`, and
acceptance `57.58%`. Public PARD K3 fixed1024 `3209171` vs
baseline `3209170`, Step 2-3, gives E2E throughput `1.591x`,
total step-time `1.580x`, generation-time `1.797x`, and
acceptance `50.66%`. 8K/16K Qwen3-235B Full-GRPO is still open
because long-OSL diagnostics hit the vLLM sleep/wake CuMem path.
