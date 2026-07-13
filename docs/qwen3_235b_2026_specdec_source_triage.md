# Qwen3-235B 2026 Speculative-Decoding Source Triage

Date: 2026-06-08 CEST / 2026-06-07 PDT

This table is the source-facing companion to the operator decision and
cost-model notes. It records which 2025-2026 methods are actionable for
Qwen3-235B now and which should remain tracked but deprioritized.

Related local evidence:

```text
docs/qwen3_235b_pard_operator_decision_20260606.md
docs/qwen3_235b_specdec_cost_model_20260606.md
docs/qwen3_235b_current_pard_jobs_20260607.csv
```

| Method | Primary source | Implementation source | Public status | Local Qwen3-235B evidence | Decision |
|---|---|---|---|---|---|
| PARD | https://arxiv.org/abs/2504.18583 | https://github.com/AMD-AGI/PARD ; https://docs.vllm.ai/en/v0.20.0/features/speculative_decoding/parallel_draft_model/ | Actionable now: vLLM supports draft_model + parallel_drafting with amd/PARD-Qwen3-0.6B. AMD-AGI/PARD HEAD is still `77eee0a12a729aaa4cc38b2a30fd544e11a8173b` when rechecked on 2026-06-08. | NeMo-RL Full-GRPO public PARD is positive in the clean non-colocated TP4 branch. The realistic GBS512 pair `3212012` vs `3212209` gives K5 Step2-5 total `1.810x`, generation `2.285x`, E2E `1.815x`, and generation-worker `2.287x` with `43.08%` acceptance and no OOM. GBS512 K3 `3212919` also completed but is slower: total `1.597x`, generation `1.934x`. Standalone OpenMath batch64/128 favors K3 over K5, so standalone and NeMo-RL select different static K. | Keep as the validated runtime substrate. Use K5 as the current NeMo-RL GBS512 default, K3 as the standalone high-batch OpenMath winner, and do not promote K7/K8/K9 as static defaults. |
| PARD-2 / CAT | https://arxiv.org/abs/2605.08632 | https://github.com/AMD-AGI/PARD | Paper is public; official code/checkpoints are not exposed in AMD-AGI/PARD at HEAD 77eee0a12a729aaa4cc38b2a30fd544e11a8173b, rechecked 2026-06-08. | Local CAT/D-PACE approximations remain useful for objective work: best vLLM OpenMath local D-PACE K5 1.296x; NeMo-RL sync D-PACE K3 1.454x. These are not official PARD-2 checkpoints. | Continue local PARD-2-style objective work, but label it as an approximation. |
| DFlash | https://arxiv.org/abs/2602.06036 | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/ | vLLM Speculators documents DFlash as a block-parallel draft model with active support. | Runtime support was made to work, but current Qwen3-235B OpenMath checkpoint has only about 0.25-1.22% acceptance. | Do not spend NeMo-RL nodes until a better Qwen3-235B DFlash checkpoint exists. |
| P-EAGLE | https://arxiv.org/abs/2602.01469 | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/peagle/ | vLLM Speculators documents the method, but reports no pretrained P-EAGLE models. | No Qwen3-235B P-EAGLE checkpoint or NeMo-RL result in the current stack. | Secondary training branch after PARD-2-style full-GRPO evidence lands. |
| Speculators Eagle-3 | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/ | https://docs.vllm.ai/projects/speculators/en/latest/ | Most mature Speculators algorithm per the decision guide. | Public EAGLE3 did not solve Qwen3-235B benefit; earlier Qwen3-32B/30BA3B were more favorable. | Keep as baseline context, not the primary Qwen3-235B fix. |
| SpecForge / SpecBundle | https://arxiv.org/abs/2603.18567 | https://github.com/sgl-project/specforge | Open-source training framework and draft-model bundle direction; paper explicitly targets scalable EAGLE-3 training including Qwen3-235B-A22B. | Useful as a training-infra reference, but current local Qwen3-235B benefit work is PARD/PARD-2-style in vLLM/NeMo-RL, not SGLang EAGLE-3. | Track as the strongest EAGLE-3 retraining reference if PARD-style Full-GRPO E2E remains weak. |
| Nightjar | https://arxiv.org/abs/2512.22420 | No local drop-in implementation; the deployed NeMo-RL runtime batch gate and dynamic-K cap are systems approximations. | Public arXiv method for load-aware adaptive speculative length and disabling speculation when load makes it unprofitable. | Static gate `3212702` was baseline-like because it disabled speculation in dense `requests=32`. Dynamic-K medium16 `3213606` proved runtime drafter-depth control at GBS512 and completed 5/5 without OOM, but it was static-K3-like and slower than K5. | Keep as the systems-policy direction, but the first useful controller should preserve K5 for dense NeMo-RL GBS512 and only lower K under measured tail or pressure conditions. Do not claim Nightjar itself is implemented. |
| TETRIS | https://arxiv.org/abs/2502.15197 | No local vLLM/NeMo-RL implementation. | Public 2025 batch speculative-decoding method that selects promising draft tokens across requests instead of assigning the same fixed draft depth to every request. | Direct fit for GBS512: Qwen3-235B batches have heterogeneous acceptance, and fixed K can waste verifier capacity on low-probability later draft positions. | Use as the conceptual target for an offline simulator over logged PARD draft/acceptance traces before changing vLLM token packing. |
| FASER | https://arxiv.org/abs/2604.20503 | Paper reports a vLLM prototype, but no local drop-in implementation is present. | Public 2026 serving-system method with per-request speculative length, early pruning during verification, and draft/verify overlap. | Strong systems fit for high-batch Qwen3-235B if static K5 helps generation but E2E is still limited by wasted verification/draft serialization. | Track as the deeper systems target after the current static gate and simple dynamic-K controller; do not claim implemented. |
| BanditSpec | https://arxiv.org/abs/2505.15141 | No local implementation. | Training-free online bandit controller for speculative-decoding hyperparameters. | Practical controller layer for choosing among K values once K3/K5/K8 standalone and NeMo-RL evidence is complete. | Start with a simpler bucketed controller: reward = accepted tokens/sec minus draft/verify waste, segmented by active batch and prompt/domain bucket. |
| Speculative Speculative Decoding / Saguaro | https://arxiv.org/abs/2603.03251 | No local implementation. | Public 2026 method that overlaps drafting with verification by preparing likely next speculations before the verifier returns. | Relevant only if profiling shows draft latency is still exposed after dynamic-K; it is more invasive than request-level K control. | Later systems branch; not a near-term Qwen3-235B NeMo-RL patch. |
| Calibrated Speculative Decoding (CSD) | https://arxiv.org/abs/2604.13634 | No local vLLM/NeMo-RL implementation. | Training-free 2026 method aimed at reducing false rejections by using online correction memory and probability-guarded acceptance. | Relevant to the user's domain-mismatch/PARD-2 question: if PARD acceptance drops because the draft is lexically different but semantically close, CSD-style rescue could raise effective acceptance without retraining. No Qwen3-235B local implementation yet. | Track as an acceptance-rescue research branch. It requires careful distribution-correctness review before NeMo-RL use, so it is behind PARD/PARD-2 and runtime gating. |
| SPECTRE | https://arxiv.org/abs/2605.08151 | https://github.com/sgl-project/sglang/pull/22272 | Serving-system method for overlapping draft generation and target verification; public path is SGLang-side, not a NeMo-RL drop-in. | Relevant to our Qwen3-235B failure mode if generation speedup exists but Full-GRPO E2E loses to orchestration/verification cost. | Systems follow-up after no-stop Full-GRPO metrics identify whether verify/draft overlap is the actual bottleneck. |
| SpecDecode-Bench | https://specdecode-bench.github.io/ | https://github.com/orgs/SpecDecode-Bench/repositories | Benchmark/profiling work focused on when speculative decoding speedup is real versus illusory. | Matches our Qwen3-235B observation: high acceptance K1 and long-context cases can still be slower due draft/verify/system cost. | Use as diagnostic framing; keep acceptance, accepted length, draft cost, verify cost, and E2E step time in the same report. |
| DFlash-family follow-ups: DFlare/DDTree/D2SD/TreeFlash | arXiv 2606.02091 / 2604.12989 / 2606.04446 / 2606.03819 | No drop-in Qwen3-235B NeMo-RL path found in the current stack. | Relevant block/diffusion/tree drafting research, but not actionable before DFlash checkpoint quality improves. | Current DFlash acceptance is near zero on held-out OpenMath. | Track only; do not prioritize over PARD/PARD-2-style. |
| SpecKV / adaptive-K controllers | https://arxiv.org/abs/2605.02888 | No local drop-in controller integrated with NeMo-RL yet. | Relevant for choosing K from confidence/entropy rather than fixing K. | Our K sweep already shows K3 is better than K1/K2/K5 in NeMo-RL sync generation for local D-PACE. | Controller follow-up once no-stop full-GRPO K3/K5 metrics exist. |
| MoE-Spec / MoE-specific verification | https://arxiv.org/abs/2602.16052 | No local vLLM/NeMo-RL implementation in the current stack. | Relevant to Qwen3-235B-A22B because MoE verification/expert bandwidth can erase speculative gains. | Qwen3-30B-A3B and Qwen3-235B need MoE construction/runtime care; SequentialMLP shallow-copy patch was required. | Systems follow-up if no-stop full-GRPO still loses despite generation speedup. |

## Current Priority

1. Treat public PARD non-colocated TP4 as the validated Qwen3-235B NeMo-RL
   runtime path. K5 is the current GBS512 NeMo-RL static-K winner because
   matched job `3212209` completed with `2.285x` generation speedup, `1.810x`
   total-step speedup, and `43.08%` acceptance.
2. Treat K3 as the standalone OpenMath high-batch winner, not automatically as
   the NeMo-RL winner. K3 jobs `3212856` and `3212919` both completed, but the
   NeMo-RL GBS512 K3 job is slower than K5.
3. Treat runtime gate `3212702` and dynamic-K medium16 `3213606` as negative or
   neutral controller evidence: the infrastructure works, but policies that
   disable speculation or force K3 during the dense GBS512 region lose to
   static K5.
4. Continue local PARD-2-style CAT/D-PACE objective work, but label it as
   an approximation until official PARD-2 code/checkpoints are public.
5. Track CSD as a domain-mismatch acceptance-rescue idea, but keep it behind
   PARD/PARD-2 and runtime gating because it needs a correctness review and a
   vLLM implementation.
6. Do not promote DFlash/P-EAGLE until checkpoint availability and held-out
   acceptance justify a NeMo-RL run.
