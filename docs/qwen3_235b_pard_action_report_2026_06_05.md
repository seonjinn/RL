# Qwen3-235B PARD / PARD-2 Action Report

Date: 2026-06-05 PDT

## Executive Summary

Qwen3-235B speculative decoding is not blocked by the runtime path anymore.
Public `amd/PARD-Qwen3-0.6B` with `draft_model`, `parallel_drafting=true`,
K=5, target `TP=4`, and draft `TP=4` works in both vLLM standalone and a
lightweight NeMo-RL `VllmGeneration` direct gate.

Latest sync-engine smoke update: both public PARD K5 and local CAT/TPP-mask
PARD-2-style K5 work through the NeMo-RL sync `VllmGeneration` path
(`async_engine=false`, fixed 256-token decode, 1 node x 4 GB200, target TP=4,
draft TP=4). Baseline is `257.30 tok/s`. Local PARD-2-style K5 reaches
`386.50 tok/s` (`1.502x`) with `43.77%` aggregate acceptance and `3.19` mean
acceptance length. Public PARD K5 reaches `391.49 tok/s` (`1.522x`) with
`42.29%` aggregate acceptance and `3.11` mean acceptance length. This proves
the PARD runtime path can produce material Qwen3-235B generation speedup inside
NeMo-RL, but the current local PARD-2-style checkpoint is not yet better than
public PARD on this sync fixed-256 smoke.

The remaining problem is drafter quality and runtime overhead under real math
prompts. Synthetic short prompts can show large PARD speedups, but OpenMath
acceptance stays around `45-47%` for K=5 and collapses for larger K. The local
dynamic D-PACE draft-probability 2K checkpoint is now the best current-harness
local standalone OpenMath checkpoint at `1.296x` and `47.01%` K5 acceptance.
The same 2K checkpoint with K3 completed at `1.207x` despite `61.55%`
aggregate K3 acceptance, so standalone still prefers K5. The 4K expansion
completed, but it regressed to `1.212x` for K5 and `1.191x` for K3, so simply
adding more teacher rows to this objective is not enough. A direct accept-rate
objective over the same 2K teacher set completed at `1.267x` and `46.80%`
acceptance, better than 4K D-PACE but still below the 2K D-PACE K5 best. The
hybrid D-PACE + accepted-prefix reward variant also completed and regressed to
`1.241x` with `46.86%` acceptance. D-PACE smoothing alpha `0.2` and `0.8`
also completed at `1.243x` and `1.221x` respectively. None of these are
promoted; the best remains 2K D-PACE K5 with alpha `0.5`.

NeMo-RL sync `VllmGeneration` tells a different systems story: using the same
2K D-PACE checkpoint, the measured K sweep is K1 `1.250x`, K2 `1.303x`, K3
`1.454x`, and K5 `1.379x` over the matched sync baseline. K3 is the current
best NeMo-RL generation setting when runtime overhead is included, even though
standalone OpenMath K5 is better. K1/K2 have higher aggregate acceptance, but
they issue too few speculative tokens to beat K3 throughput.

The active follow-up is still the true no-stop Full-GRPO PARD-2-style matrix,
which is queued and has not emitted E2E metrics yet.
Preflight on 2026-06-06 10:12 PDT passed for the active remote checkout:
the PackedSeqParams compatibility guard, omitted-logprob repair, greedy
finite-loss guard, source vLLM `PYTHONPATH` override, PARD `draft_model`
runtime, `parallel_drafting=true`, and draft TP wiring are all present.

## Best Current Numbers

| Gate | Drafter | Shape | Result |
|---|---|---|---|
| vLLM standalone synthetic short | public PARD K12 | `ISL=1000`, `OSL=512`, bs32 | `3.29x`, about `93%` acceptance |
| vLLM standalone OpenMath historical | public PARD K5 | `ISL=1024`, `OSL=1024`, bs32 | `1.31x`, `45.5%` acceptance |
| vLLM standalone OpenMath current harness recal | public PARD K5 | same shape | `1.00x`, `46.0%` acceptance |
| vLLM standalone OpenMath CAT smoke | local CAT K5, 128 rows | same shape | `0.996x`, `47.0%` acceptance |
| vLLM standalone OpenMath CAT prefix-product 1K | local CAT K5, 1024 rows | same shape | `0.901x`, `47.7%` acceptance |
| vLLM standalone OpenMath CAT TPP-mask 1K | local CAT TPP-mask K5, 1024 rows | same shape | `1.122x`, `46.7%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 2K | local D-PACE K5, 2048 rows | same shape | `1.296x`, `47.0%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 2K alpha `0.2` | local D-PACE K5, 2048 rows | same shape | `1.243x`, `47.8%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 2K alpha `0.8` | local D-PACE K5, 2048 rows | same shape | `1.221x`, `47.4%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 2K | local D-PACE K3, 2048 rows | same shape | `1.207x`, `61.6%` K3-position acceptance |
| vLLM standalone OpenMath accept-rate 2K | local accept-rate K5, 2048 rows | same shape | `1.267x`, `46.8%` acceptance |
| vLLM standalone OpenMath D-PACE + accept reward 2K | local hybrid K5, 2048 rows | same shape | `1.241x`, `46.9%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 4K | local D-PACE K5, 4096 rows | same shape | `1.212x`, `46.9%` acceptance |
| vLLM standalone OpenMath D-PACE draft CE 4K | local D-PACE K3, 4096 rows | same shape | `1.191x`, `61.7%` K3-position acceptance |
| NeMo-RL direct generation gate, graph-on | public PARD K5 | 1 node, 4 GB200, `TP=4`, 32 async prompts | `1.20x`, `46.8%` acceptance |
| NeMo-RL direct generation gate, graph-on | local CAT TPP-mask K5, 1024 rows | same shape | `1.282x`, `46.6%` acceptance |
| NeMo-RL sync generation smoke, eager fixed256 | local CAT TPP-mask PARD-2-style K5, 1024 rows | 1 node, 4 GB200, `TP=4`, batch 32, fixed `256` decode | `1.502x`, `43.8%` acceptance, mean acceptance length `3.19` |
| NeMo-RL sync generation smoke, eager fixed256 | public PARD K5 | same shape | `1.522x`, `42.3%` acceptance, mean acceptance length `3.11` |
| NeMo-RL sync generation smoke, eager fixed256 | local D-PACE K5, 2048 rows | same shape | `1.379x`, `44.25%` acceptance, mean acceptance length `3.21` |
| NeMo-RL sync generation smoke, eager fixed256 | local D-PACE K3, 2048 rows | same shape | `1.454x`, `57.50%` acceptance, mean acceptance length `2.73` |
| NeMo-RL sync generation smoke, eager fixed256 | local D-PACE K2, 2048 rows | same shape | `1.303x`, `67.61%` acceptance, mean acceptance length `2.35` |
| NeMo-RL sync generation smoke, eager fixed256 | local D-PACE K1, 2048 rows | same shape | `1.250x`, `76.42%` acceptance, mean acceptance length `1.76` |
| NeMo-RL direct generation gate, TP16 fixed256 eager | local CAT TPP-mask K5, 1024 rows | 32 prompts, fixed `256` decode | `1.621x`, `47.4%` acceptance |
| NeMo-RL direct generation gate, eager sanity | public PARD K5 | same shape, eager | `2.00x`, `46.4%` acceptance |
| NeMo-RL full-GRPO stop-after-generation | local CAT TPP-mask K5, 1024 rows | 32 nodes, 128 GB200, GBS `256`, per-engine requests `32`, generation `TP=4` | `1.718x` generation throughput, `53.5%` acceptance |
| NeMo-RL true no-stop Full-GRPO | local CAT TPP-mask PARD-2-style K5, 1024 rows | real sampling, worker32 shape, `MAX_STEPS=4` | pending: baseline `3186510`, local `3186511`; no E2E metric yet |
| NeMo-RL true no-stop Full-GRPO | public/local K5 comparison | fixed-256 diagnostic, worker32 shape, `MAX_STEPS=20` | pending: baseline `3186342`, local `3186343`, public `3186344`; no E2E metric yet |

Generated plots:

- `docs/qwen3_235b_nemorl_direct_vllmgeneration_speedup_acceptance.png`
- `docs/qwen3_235b_nemorl_sync_vllmgeneration_speedup_acceptance.png`
- `docs/qwen3_235b_nemorl_sync_dpace_comparison.png`
- `docs/qwen3_235b_nemorl_sync_efficiency.png`
- `docs/qwen3_235b_pard_math_local_checkpoint_gates.png`
- `docs/qwen3_235b_fullgrpo_validation_status_20260606.png`
- `docs/qwen3_235b_pard_operator_decision_20260606.png`
- `docs/qwen3_235b_specdec_cost_model_20260606.png`

Operator decision summary:

- `docs/qwen3_235b_team_report_20260606.html`
- `docs/qwen3_235b_team_report_20260606.md`
- `docs/qwen3_235b_pard_operator_decision_20260606.md`
- `docs/qwen3_235b_pard_operator_decision_20260606.csv`
- `docs/qwen3_235b_specdec_cost_model_20260606.md`
- `docs/qwen3_235b_specdec_cost_model_20260606.csv`
- `docs/qwen3_235b_2026_specdec_source_triage.md`
- `docs/qwen3_235b_2026_specdec_source_triage.csv`
- `docs/qwen3_235b_fullgrpo_e2e_claim_gate_20260606.md`
- `docs/qwen3_235b_fullgrpo_e2e_claim_gate_20260606.csv`
- `docs/qwen3_235b_goal_completion_audit_20260606.md`
- `docs/qwen3_235b_goal_completion_audit_20260606.csv`

Bundle refresh:

```bash
scripts/refresh_qwen235b_report_bundle.sh
TRY_REMOTE=true scripts/refresh_qwen235b_report_bundle.sh
```

Raw metrics:

- `docs/qwen3_235b_nemorl_direct_vllmgeneration_metrics_20260605.csv`
- `docs/qwen3_235b_nemorl_sync_vllmgeneration_smoke_20260605.csv`
- `docs/qwen3_235b_nemorl_dpace_validation_20260606.csv`
- `docs/qwen3_235b_nemorl_sync_efficiency_20260606.csv`
- `docs/qwen3_235b_nemorl_worker32_generation_only_metrics_20260605.csv`
- `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`
- `docs/qwen3_235b_fullgrpo_sampling_step4_status_20260606.csv`
- `docs/qwen3_235b_fullgrpo_sampling_step4_status_20260606.md`
- `docs/qwen3_235b_fullgrpo_fixed256_step20_status_20260606.csv`
- `docs/qwen3_235b_fullgrpo_fixed256_step20_status_20260606.md`
- `docs/qwen3_235b_fullgrpo_scheduler_status_20260606.csv`
- `docs/qwen3_235b_pard2style_training_cost_20260606.csv`

## 2026-06-06 PDT Update

High-batch standalone OpenMath has now been measured for Qwen3-235B PARD K5
with `ISL=1024`, `OSL=1024`, target `TP=4`, draft `TP=4`, and 4 GB200 GPUs.
This is the closest standalone comparison to a larger NeMo-RL per-worker
generation batch.

| Case | Batch | Throughput / GPU | Speedup | Acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|---:|
| baseline | 64 | `806.66 tok/s` | `1.000x` | n/a | n/a |
| public PARD K5 | 64 | `1058.61 tok/s` | `1.312x` | `44.59%` | `3.230` |
| local CAT/TPP-mask K5 | 64 | `1060.88 tok/s` | `1.315x` | `45.07%` | `3.253` |
| baseline | 128 | `1409.65 tok/s` | `1.000x` | n/a | n/a |
| public PARD K5 | 128 | `1681.08 tok/s` | `1.193x` | `44.69%` | `3.234` |
| local CAT/TPP-mask K5 | 128 | `1673.85 tok/s` | `1.187x` | `44.24%` | `3.212` |

Interpretation: `bs=64` still shows useful standalone speedup, but `bs=128`
already drops from about `1.31x` to about `1.19x`. This supports a saturation /
system-overhead explanation: larger batches do not automatically make PARD K5
faster on Qwen3-235B OpenMath, even when acceptance remains around `44-45%`.

DFlash retry28 is now past the dependency/runtime issues and is executing
Qwen3-235B OpenMath generation with `vllm-0.19.1rc1.dev315+g0b790a250`,
`cutlass.cute`, `quack-kernels`, and `tvm_ffi` all importable. The support
probe `3187927` completed with `dflash_ready=true`, `SUPPORTED_SPECULATORS_TYPES`
including `dflash`, and both `vllm.model_executor.models.qwen3_dflash` and
`vllm.v1.spec_decode.dflash` importable. The checkpoint is not useful for this
domain yet:

| Method | Batch | Throughput / GPU | Acceptance | Mean acceptance length | Accepted / Drafted |
|---|---:|---:|---:|---:|---:|
| DFlash K3 | 1 | `1.92 tok/s` | `1.22%` | `1.036` | `36 / 2961` |
| DFlash K3 | 2 | `3.90 tok/s` | `0.45%` | `1.013` | `27 / 6057` |
| DFlash K3 | 4 | `7.67 tok/s` | `0.32%` | `1.010` | `39 / 12159` |
| DFlash K3 | 8 | `15.37 tok/s` | `0.73%` | `1.022` | `176 / 24024` |
| DFlash K3 | 16 | `30.78 tok/s` | `0.61%` | `1.018` | `295 / 48219` |
| DFlash K3 | 32 | `61.41 tok/s` | `0.93%` | `1.028` | `890 / 95538` |
| DFlash K5 | 1 | `2.09 tok/s` | `0.44%` | `1.022` | `22 / 5005` |
| DFlash K5 | 2 | `3.88 tok/s` | `0.25%` | `1.012` | `25 / 10105` |
| DFlash K5 | 4 | `8.00 tok/s` | `0.28%` | `1.014` | `56 / 20180` |
| DFlash K5 | 8 | `9.05 tok/s` | `0.43%` | `1.021` | `172 / 40065` |
| DFlash K5 | 16 | `31.02 tok/s` | `0.35%` | `1.017` | `281 / 80435` |
| DFlash K5 | 32 | `64.67 tok/s` | `0.54%` | `1.027` | `866 / 159360` |

Interpretation: DFlash runtime is now a solved issue, but the available
`dflash_openmath_reasoning_cot_smoke512_k5_aligned` checkpoint is severely
misaligned for Qwen3-235B OpenMath. Do not promote this checkpoint to NeMo-RL.
The next DFlash action would need better hidden-state extraction/training, not
another runtime retry.

Raw DFlash retry28 metrics:

- `docs/qwen3_235b_dflash_retry28_openmath_metrics.csv`

NeMo-RL Full-GRPO 5-step status for Qwen3-32B and Qwen3-30B-A3B:

| Run set | Jobs | Result | Root cause |
|---|---|---|---|
| r1 | `3186660-3186664` | all failed before useful work | Ray/Python mismatch: cluster started with Ray `2.49.2` / Python `3.12.13`, driver venv used Ray `2.54.0` / Python `3.13.13`. |
| r2 `raymatch` | `3186983-3186987` | version mismatch fixed, still failed | Qwen32 baseline hit `TypeError: cannot pickle code objects` during Megatron worker creation; Qwen30 hit Qwen3-MoE weight-streaming tensor shard handling. |
| r3 `cgraphfix` | `3187428`, `3187431-3187433` | reached Step 1 generation, failed after generation/update | Qwen32 public PARD K5 produced Step 1 generation acceptance around `49-61%` and mean acceptance length around `3.4-4.1`, then Megatron policy forward failed with TransformerEngine RMSNorm CUDA invalid argument. Qwen30 baseline/public/local failed in vLLM weight update with `shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1 or 2)` in Qwen3-MoE fused-MoE weight loading. |

These failures are not evidence that PARD has no NeMo-RL benefit. The Qwen32
PARD generation path did run and emitted healthy acceptance, but Full-GRPO still
needs policy forward / weight streaming fixes before E2E numbers are valid.

Active training follow-up: Qwen3-235B CAT/TPP-mask expansion from 1024 rows to
2048 rows is running. Teacher-logprob jobs `3189381`, `3189384`, `3189387`,
`3189395`, `3189401`, `3189409`, `3189412`, and `3189437` are running; train
job `3189439` and OpenMath gate job `3189440` are waiting on dependency.

## Current PARD-Math Training Evidence

External non-NeMo-RL math teacher data currently available:

| Teacher chunk | Rows |
|---|---:|
| offset 0, c4 | `571` |
| offset 1000, c16 | `1000` |
| offset 2000, c16 | `1000` |
| offset 3000, c16 | `1000` |

Local checkpoint gates on OpenMath bs32:

| Checkpoint | Train rows | Job | Speedup vs baseline | Acceptance | Conclusion |
|---|---:|---:|---:|---:|---|
| public PARD K5 current-harness recal | 0 | `3171868` | `1.001x` | `45.99%` | current paired reference |
| local PARD smoke | 64 | `3170614` | `1.084x` | `46.32%` | compatibility only |
| local PARD partial | 180 | `3170854` | `1.076x` | `46.34%` | no material improvement |
| local PARD plain CE | 450 | `3171063` | `0.866x` | `44.75%` | regressed |
| local prefix-reward | 1024 | `3171517` | `1.045x` | `45.45%` | stable, not better |
| public/local interpolation alpha 0.10 | 1024 | `3171764` | `1.091x` | `46.55%` | best local checkpoint, still weak |
| local CAT prefix-product smoke | 128 | `3172639` | `0.996x` | `46.97%` | CAT path works, not a performance checkpoint |
| local CAT prefix-product 1K | 1024 | `3173688` | `0.901x` | `47.68%` | acceptance rose, throughput regressed |
| local weighted CE token-prefix-product 1K | 1024 | `3173861`/`3173874` | `0.869x` | `46.66%` | PARD-2/D-PACE-style loss ablation; regressed |

Interpretation: simple target-token distillation and prefix-position masking are
not enough. They validate the integration path but do not solve acceptance or
throughput. The next training objective should use target confidence/logprobs
or a closer PARD-2/CAT objective rather than just scaling the same mask. The
128-row CAT smoke validated the PARD-2-style logprob path, but it is too small
to promote: acceptance rose only about `+0.98pp` versus current public-PARD
recal while throughput stayed flat.

## Training Sample and Time Guidance

The current local PARD-2-style path does not need many samples to prove that the
plumbing works, but it likely needs substantially more target-aligned data to
beat public PARD robustly. The observed cost split is:

| Scale | What it is good for | Time signal |
|---:|---|---|
| `128-1K` rows | smoke, tokenizer/template/logprob plumbing, loss sanity | drafter training completes in about `3 min` on 4 GB200 GPUs |
| `2K` rows | first objective ranking | teacher logprob chunks complete in about `40 min` wall-clock when parallelized; training job `3190562` took `00:04:50` |
| `4K` rows | small scale-up check | training job `3192611` took `00:07:17`; result regressed versus 2K |
| `50K` rows | first serious domain-adaptation candidate | teacher generation/logprob collection dominates and is expected to be half-day/day scale depending on parallelism |
| `500K` rows | production-scale drafter exploration | not justified until the objective improves; teacher collection would be multi-day scale |

Decision: do not launch 50K/500K just to add rows. The 4K D-PACE regression is
evidence that sample count alone is not the limiter. The next useful work is an
objective/controller change that improves held-out OpenMath K5 acceptance and
keeps throughput above the 2K D-PACE K5 reference.

## PARD-2 Direction

Primary sources:

- PARD: https://arxiv.org/abs/2504.18583
- PARD-2: https://arxiv.org/abs/2605.08632
- AMD PARD repo: https://github.com/AMD-AGI/PARD
- vLLM parallel draft model docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/parallel_draft_model/

PARD-2 is the closest match to the observed failure. It explicitly moves the
objective from independent next-token accuracy toward inference-time consecutive
acceptance length and Confidence-Adaptive Token optimization. AMD's public repo
currently says PARD-2 code/checkpoints are still pending. A fresh clone of
`https://github.com/AMD-AGI/PARD.git` on 2026-06-06 resolved to
`77eee0a12a729aaa4cc38b2a30fd544e11a8173b`; it contains PARD train/inference
files but no dedicated `pard2` implementation or PARD-2 checkpoint assets. The
immediate runnable path is therefore:

1. Keep using public PARD/vLLM `draft_model` runtime.
2. Collect Qwen3-235B teacher generated-token logprobs on external math
   prompts.
3. Build a CAT-like trainer that weights target tokens by target confidence and
   accepted-prefix utility.
4. Gate every checkpoint on held-out OpenMath before NeMo-RL full GRPO.

This can be done without using the NeMo-RL training dataset. The current data
path uses external math/reasoning prompts and keeps OpenMath as the held-out
acceptance/speed gate. That separation is important because OpenMath is where
the public PARD checkpoint drops to about `46%` acceptance.

Scope note: the current runnable implementation approximates the PARD-2 CAT
objective. It does not yet implement target-hidden-state injection or stochastic
target-feature gating from full PARD-2. That would require an additional runtime
path to expose target hidden states to the draft model.

Practical answer for math-domain PARD training: yes, we can train using external
math datasets rather than the NeMo-RL rollout dataset. The current pipeline uses
external math/reasoning prompts to generate Qwen3-235B teacher continuations,
stores target-token logprobs, converts them into PARD JSONL, trains the PARD
0.6B drafter, and then gates on held-out OpenMath. This keeps training data and
evaluation data separated.

## External Math Dataset Training Recipe

Use OpenMath as the held-out gate, not as the only training source. Good
external prompt pools for drafter adaptation are:

| Dataset | Why useful | How to use |
|---|---|---|
| [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Large competition-style math CoT corpus, about `860K+` problem-solution pairs. | Use prompts/questions as teacher-generation seeds; optionally keep solutions only for filtering. |
| [`nvidia/OpenMathReasoning`](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | Large AoPS-style reasoning corpus with millions of CoT/TIR traces. | Use a non-overlapping train split for prompts; keep our current OpenMath gate as held-out. |
| `Tyrion279/deepscaler` or `PALM-Lab/math-deepscaler` | Smaller, harder math prompt pools. | Good for a high-difficulty 5K-40K ablation after NuminaMath/OpenMathReasoning. |
| MATH/GSM8K-derived instruction data | Stable baseline math distribution. | Use as low-risk seed data, but do not expect it alone to fix long OpenMath reasoning acceptance. |

The runnable procedure:

1. Sample external math prompts with no overlap against the OpenMath gate.
2. Generate Qwen3-235B teacher continuations with target-token logprobs enabled.
3. Convert each `{prompt, teacher_answer, teacher_logprobs}` row into PARD
   conversation JSONL.
4. Train `amd/PARD-Qwen3-0.6B` with `para_num=5`, Qwen PARD token `151670`,
   target confidence weights, and a PARD-2-style accepted-prefix objective.
5. Gate each checkpoint on held-out OpenMath `ISL=1024`, `OSL=1024`, `bs=32`,
   target `TP=4`, draft `TP=4`, K5 before any NeMo-RL run.

Current implementation status: steps 1-5 are runnable for a 1K-row external
math sample. The open question is objective quality, not pipeline feasibility.

## Adjacent 2025-2026 Methods To Track

These are relevant, but not all are immediately lower risk than the current
PARD/CAT path:

| Method | Source | Relevance to Qwen3-235B | Current action |
|---|---|---|---|
| Domain draft distillation | https://arxiv.org/abs/2503.07807 | Directly matches our observed domain shift: generic draft acceptance drops when target/domain changes. Supports using external math data and teacher distillation before runtime changes. | Already reflected in the non-NeMo math teacher pipeline. |
| PARD-2 CAT | https://arxiv.org/abs/2605.08632 | Optimizes consecutive acceptance length instead of uniform token accuracy. This is the closest match to our K5 OpenMath acceptance failure. | Implemented runnable approximation in `pard_train_cat_weighted.py`; 128-row smoke passed integration but not performance. |
| P-EAGLE | https://arxiv.org/abs/2602.01469 and https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/peagle/ | Parallelizes EAGLE-style multi-token drafting and is implemented in the vLLM speculators project, but no pretrained P-EAGLE models are currently available in the docs. | Candidate only after PARD/CAT gate; would require training a Qwen3-235B-specific speculator. |
| DFlash | https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/ | Block-diffusion parallel drafter, conditioned on target hidden states. It targets exactly the AR-drafter latency issue but needs hidden-state/runtime integration and available compatible model weights. | Track as a higher-effort follow-up, not the first Qwen3-235B fix. |
| LK losses | https://arxiv.org/abs/2602.23881 | Direct acceptance-rate training objective; easy to integrate conceptually if the current CAT label-mask improves but remains acceptance-limited. | Candidate loss replacement/ablation for `pard_train_cat_weighted.py`. |
| D-PACE-style dynamic CE | https://arxiv.org/abs/2605.18810 | Reweights draft-position training by dynamic acceptance utility instead of treating every target token equally. This is close to our failure mode when K5 has high target confidence but low realized throughput. | Add as the next trainer ablation if 1K CAT prefix-product remains flat: replace stochastic label masking with explicit per-position weighted CE. |
| Test-Time Speculation | https://arxiv.org/abs/2605.09329 | Addresses long-response degradation by online distillation during inference; relevant to our decode-heavy/OpenMath long-output runs. | Interesting for NeMo-RL online adaptation, but more invasive than offline PARD/CAT. |

Fresh scan on 2026-06-05:

| Source | What changed or matters | Action for this project |
|---|---|---|
| AMD PARD repo, https://github.com/AMD-AGI/PARD | The repo now documents PARD-2 and says the 2026-05-09 PARD-2 paper is released, but code and model checkpoints are still "released soon". The repo also lists Qwen3 PARD weights and a basic `pard.train` example. | Keep using the public PARD runtime/checkpoint path and our local CAT approximation; do not wait idle for unreleased PARD-2 artifacts. |
| PARD-2, https://arxiv.org/abs/2605.08632 | The core claim matches our failure mode: optimize consecutive acceptance length via Confidence-Adaptive Token weighting, not just next-token prediction accuracy. | Our `CAT_LOSS_MODE=mask`/`token_prefix_product` trainer is the current runnable approximation. If NeMo-RL direct gate is weak, implement a closer accepted-prefix surrogate rather than scaling rows. |
| LK Losses, https://arxiv.org/abs/2602.23881 | v2 was posted 2026-06-01. The paper claims direct acceptance-rate objectives improve average acceptance length by about `8-10%` across general, coding, and math domains with no inference overhead. | Strong next loss candidate because it is training-only and can fit our current PARD trainer without changing vLLM/NeMo-RL runtime. |
| D-PACE, https://arxiv.org/abs/2605.18810 | Submitted 2026-05-12. It derives dynamic per-position CE weights from expected accepted draft length and reports wall-clock and emitted-length improvements without inference changes. | Replace the failed manual weighted CE with a correctly scaled, normalized D-PACE-style position weighting if the current mask-mode gate is insufficient. |
| vLLM Speculators decision guide, https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/ | Official docs say Speculators currently supports Eagle-3, P-EAGLE, and DFlash; P-EAGLE/DFlash are newer but available, with DFlash using single-forward-pass block prediction. | DFlash/P-EAGLE are viable follow-ups only if we can train Qwen3-235B-compatible speculators; they are higher effort than PARD/CAT because they need hidden-state extraction/training integration. |
| vLLM Speculators DFlash tutorial, https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/train_dflash_online/ | DFlash online training requires `--target-layer-ids`, `--speculator-type dflash`, `--block-size`, `--max-anchors`, and typically more draft layers. The example uses Qwen3-8B but says the process is the same for other models. | Feasibility path: first run a tiny Qwen3-235B DFlash hidden-state extraction smoke on one node before attempting full training or NeMo-RL integration. |

New code added for step 2:

- `experiments/eagle3_qwen3_235b/generate_training_conversations_openai.py`
  now supports `--include-generation-logprobs` and `--top-logprobs`.
- `experiments/eagle3_qwen3_235b/prepare_training_conversations.sh` passes
  `GENERATION_LOGPROBS` and `GENERATION_TOP_LOGPROBS`.
- `experiments/pard_qwen3_235b_math/generate_qwen235b_teacher_math_continuations.sh`
  and `submit_teacher_math_continuations.sh` pass the same options.
- `experiments/pard_qwen3_235b_math/inspect_teacher_logprobs.py` summarizes
  logprob coverage, generated-token counts, and target-confidence statistics.
- `experiments/pard_qwen3_235b_math/pard_train_cat_weighted.py` adds a
  runnable CAT-style PARD trainer. It combines draft-position priority with
  teacher generated-token confidence in the label mask while keeping the
  default causal-LM loss. Its default `cat_importance_mode=prefix_product`
  approximates PARD-2 CAT by weighting each draft-position target with the
  cumulative product of the previous generated-token confidences inside that
  draft window.
- The same trainer now also supports `cat_loss_mode=weighted_ce`, which keeps
  all target labels and applies explicit per-token CE weights. The active
  setting is `cat_importance_mode=token_prefix_product`, combining each target
  token's confidence with accepted-prefix utility. This is the current
  D-PACE/PARD-2-style ablation after the 1K stochastic CAT run regressed
  throughput.
- `experiments/pard_qwen3_235b_math/make_pard_train_config.py`,
  `train_pard_math_k5.sh`, and `submit_pard_math_k5_train.sh` now pass CAT
  knobs: `CAT_CONFIDENCE_FLOOR`, `CAT_CONFIDENCE_GAMMA`,
  `CAT_PROMPT_KEEP_PROB`, `CAT_MISSING_CONFIDENCE_KEEP_PROB`, and
  `CAT_REQUIRE_LOGPROBS`, plus optional `CAT_IMPORTANCE_MODE` and
  `CAT_LOSS_MODE`.

Use this shape for the next CAT-data smoke:

```bash
GENERATION_LOGPROBS=true \
GENERATION_TOP_LOGPROBS=1 \
LIMIT=128 \
SAMPLE_OFFSET=4000 \
OUTPUT_SUFFIX=128_offset4000_logprobs \
GENERATION_CONCURRENCY=8 \
VLLM_MAX_NUM_SEQS=8 \
VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
  bash experiments/pard_qwen3_235b_math/submit_teacher_math_continuations.sh
```

Submitted smoke:

| Job | Status at submit | Output |
|---:|---|---|
| `3172443` | cancelled; wrong prompt path was 0-byte `mixed_math_nonopenmath_prompts_500k.jsonl` | not used |
| `3172456` | completed in `36:02` on 2026-06-05 PDT with `mixed_math_nonopenmath_prompts_10000.jsonl` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/data/qwen235b_teacher_math_continuations_128_offset4000_logprobs.jsonl` |

Early validation for `3172456`: at 8/128 generated rows, all 8 rows had
`generation.teacher_logprobs`, with `8192` generated-token logprobs and mean
target confidence about `0.893`. This proves the logprob collection path works;
the job is still running to finish the 128-row smoke.

Later validation: at 16/128 rows, all rows had logprobs, with `16384`
generated-token logprobs and mean target confidence about `0.876`.
At 48/128 rows, all rows still had logprobs, with `49152` generated-token
logprobs and mean target confidence about `0.866`.
At 80/128 rows, all rows still had logprobs, with `81920` generated-token
logprobs, mean target confidence about `0.863`, and median target confidence
about `0.995`.

Final validation for `3172456`: all `128/128` rows passed schema/token
validation with no warnings or failures. Logprob inspection reported
`128/128` rows with logprobs, `130685` generated-token logprobs, mean target
confidence about `0.862`, and median target confidence about `0.995`.

Dependent CAT train smoke:

| Job | Dependency | Purpose |
|---:|---|---|
| `3172523` | `afterok:3172456`; completed in `4:11` | K5 CAT-style trainer smoke on the 128-row logprob teacher set, `lr=3e-6`, `CAT_CONFIDENCE_FLOOR=0.2`, `CAT_REQUIRE_LOGPROBS=true`, default `CAT_IMPORTANCE_MODE=prefix_product` |

`3172523` produced:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_logprob_128_lr3e6/checkpoint-2
```

Training completed 2 optimizer steps over 128 records with `train_loss` about
`1.256`. The log confirms `importance_mode=prefix_product`.

OpenMath vLLM standalone gate:

| Job | Shape | Draft model | Status |
|---:|---|---|---|
| `3172639` | OpenMath `ISL=1024`, `OSL=1024`, bs32, K5, target/draft `TP=4` | CAT 128-row `checkpoint-2` | completed in `18:42`; `0.996x`, `46.97%` acceptance |

Gate output path:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard_math_trained_k5_openmath_isl1024_osl1024_bs1-32_20260605_074822/breakdown.json
```

Validate the finished output with:

```bash
python3 experiments/pard_qwen3_235b_math/inspect_teacher_logprobs.py \
  --input /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/data/qwen235b_teacher_math_continuations_128_offset4000_logprobs.jsonl \
  --require-logprobs
```

Parsed result for `3172639`: `482.08` output tok/s/GPU, `46.97%`
acceptance, `2.35` accepted tokens per draft, mean acceptance length `3.35`.
That is `0.996x` versus the OpenMath bs32 baseline and `0.995x` versus current
public-PARD recal `3171868`. The smoke therefore proves the data/objective path
loads and runs, but it should not be used for NeMo-RL full GRPO.

CAT 1K scale-up after the 128-row smoke:

| Job | Status | Purpose |
|---:|---|---|
| `3173152` | cancelled | 512-row teacher-logprob chunk, offset `5000`; too slow for 4h partition |
| `3173162` | cancelled | 512-row teacher-logprob chunk, offset `5512`; too slow for 4h partition |
| `3173170` | cancelled | dependent CAT train for the cancelled 512x2 data path |
| `3173184` | cancelled | dependent OpenMath gate for the cancelled 512x2 data path |
| `3173299` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5000` |
| `3173300` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5128` |
| `3173304` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5256` |
| `3173307` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5384` |
| `3173311` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5512` |
| `3173312` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5640` |
| `3173315` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5768` |
| `3173317` | running as of 2026-06-05 08:39 PDT | 128-row teacher-logprob chunk, offset `5896` |
| `3173318` | pending dependency on all eight 128-row chunks | CAT prefix-product K5 train, `1024` rows, `lr=3e-6` |
| `3173321` | pending dependency `afterok:3173318` | OpenMath `ISL=1024`, `OSL=1024`, bs32 K5 gate for expected `checkpoint-16` |

The 1K teacher collection first failed the scheduling shape: the batch
partition rejected an 8-hour single job, and two 512-row chunks were progressing
too slowly for the 4-hour partition limit. The active replacement is eight
128-row chunks. The gate uses the expected checkpoint:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_logprob_1024_8x128_lr3e6/checkpoint-16
```

Use the pipeline poll helper to avoid retyping the same remote checks:

```bash
scripts/poll_qwen235b_pard_cat_pipeline_status.sh
```

At 2026-06-05 08:26 PDT, this helper showed both 512-row teacher jobs still
running, both dependency jobs pending, and no teacher JSONL output yet. The
slurm logs were still at direct vLLM server startup.

At 2026-06-05 08:33 PDT, both teacher jobs had moved into generation. Partial
outputs already contained valid logprobs:

| Job | Rows written | Logprob coverage | Mean target confidence |
|---:|---:|---:|---:|
| `3173152` | `9` | `9/9` | `0.865` |
| `3173162` | `16` | `16/16` | `0.885` |

At 2026-06-05 08:35 PDT, the 512-row jobs were cancelled because they were
unlikely to complete within the 4-hour limit. The replacement 128x8 pipeline was
submitted, and the poll helper now tracks the new job files by default.

At 2026-06-05 08:43 PDT, all eight 128-row teacher jobs were running, the train
and gate jobs were dependency-pending, and no 128-row teacher JSONL output had
been written yet. This is still within the expected Qwen3-235B server startup
window; the earlier 128-row smoke took about 36 minutes end to end.

At 2026-06-05 08:47 PDT, several 128-row chunks had started writing valid
teacher rows. Observed partial coverage:

| Job | Rows written | Logprob coverage | Mean target confidence |
|---:|---:|---:|---:|
| `3173299` | `8` | `8/8` | `0.866` |
| `3173304` | `8` | `8/8` | `0.898` |
| `3173307` | `8` | `8/8` | `0.857` |
| `3173311` | `16` | `16/16` | `0.884` |

Other 128-row chunks had reached server-ready/generation state but had not yet
written rows at that poll point.

At 2026-06-05 08:51 PDT, all eight 128-row chunks had started writing rows.
Total observed partial rows: `95`, with `100%` logprob coverage in every chunk.

| Job | Rows written | Logprob coverage | Mean target confidence |
|---:|---:|---:|---:|
| `3173299` | `16` | `16/16` | `0.881` |
| `3173300` | `8` | `8/8` | `0.857` |
| `3173304` | `9` | `9/9` | `0.894` |
| `3173307` | `16` | `16/16` | `0.865` |
| `3173311` | `16` | `16/16` | `0.884` |
| `3173312` | `8` | `8/8` | `0.857` |
| `3173315` | `8` | `8/8` | `0.860` |
| `3173317` | `8` | `8/8` | `0.869` |

At 2026-06-05 09:02 PDT, the 128x8 teacher collection had reached `538/1024`
rows, with `538/538` rows carrying logprobs and weighted mean target confidence
`0.870`. Per-chunk progress:

| Job | Rows written | Logprob coverage | Mean target confidence |
|---:|---:|---:|---:|
| `3173299` | `72` | `72/72` | `0.871` |
| `3173300` | `64` | `64/64` | `0.861` |
| `3173304` | `65` | `65/65` | `0.872` |
| `3173307` | `72` | `72/72` | `0.868` |
| `3173311` | `72` | `72/72` | `0.881` |
| `3173312` | `64` | `64/64` | `0.864` |
| `3173315` | `65` | `65/65` | `0.872` |
| `3173317` | `64` | `64/64` | `0.869` |

At 2026-06-05 09:04 PDT, the same pipeline had reached `595/1024` rows, again
with `100%` logprob coverage. The train job `3173318` and gate `3173321`
remained dependency-pending.

At 2026-06-05 09:18 PDT, all eight teacher chunks had completed. Final teacher
coverage was `1024/1024` rows with `100%` logprob coverage and weighted mean
target confidence `0.867`. The dependent CAT train `3173318` started
successfully. Its log confirmed:

```text
written: 1024
skipped: 0
position_loss_weighting: cat_prefix_reward
cat_importance_mode: prefix_product
learning_rate: 3e-6
```

Final per-chunk confidence:

| Job | Rows written | Logprob coverage | Mean target confidence |
|---:|---:|---:|---:|
| `3173299` | `128` | `128/128` | `0.869` |
| `3173300` | `128` | `128/128` | `0.862` |
| `3173304` | `128` | `128/128` | `0.871` |
| `3173307` | `128` | `128/128` | `0.864` |
| `3173311` | `128` | `128/128` | `0.874` |
| `3173312` | `128` | `128/128` | `0.867` |
| `3173315` | `128` | `128/128` | `0.870` |
| `3173317` | `128` | `128/128` | `0.863` |

At 2026-06-05 09:38 PDT, train job `3173318` was still running and gate job
`3173321` was still dependency-pending. The expected `checkpoint-16` and gate
`breakdown.json` were not present yet. The train log was still updating in the
Python dependency-install phase; the run venv had been created under the fs1
run directory.

At 2026-06-05 09:46 PDT, `3173318` was still in `pip install -r
PARD/requirement.txt --no-build-isolation`, about `22` minutes into that process
inside the allocated container. An overlap import check showed the partial venv
was not yet usable: `datasets` and `trl` were still missing, and `deepspeed`
was partially initialized. The local wrapper was updated so future submissions
can use `INSTALL_PARD_REQUIREMENTS=auto` and explicit pip arguments to avoid
repeating an unnecessary fresh install when the container or a supplied venv
already has the trainer imports.

At 2026-06-05 09:49 PDT, the known partial180 venv was validated inside the
same container image:

```text
accelerate 1.13.0
datasets 4.8.5
deepspeed 0.16.3
torch 2.10.0+cu130
transformers 4.51.3
trl 0.14.0
```

The stuck train `3173318` and dependent gate `3173321` were cancelled. The same
1K CAT train was resubmitted with the validated venv and
`INSTALL_PARD_REQUIREMENTS=false`:

| Job | Status | Purpose |
|---:|---|---|
| `3173685` | submitted 2026-06-05 09:49 PDT | CAT prefix-product K5 train, `1024` rows, `lr=3e-6`, partial180 venv reuse |
| `3173688` | submitted 2026-06-05 09:50 PDT, dependency `afterok:3173685` | OpenMath `ISL=1024`, `OSL=1024`, bs32 K5 gate for the venreuse `checkpoint-16` |

Expected venreuse checkpoint:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_logprob_1024_8x128_lr3e6_venvreuse/checkpoint-16
```

At 2026-06-05 09:53 PDT, `3173685` completed in `3:14`; the trainer itself
reported `train_runtime=90.27s`, `global_step=16`, and
`train_loss=1.2977116107940674`. The expected `checkpoint-16` is present and
gate job `3173688` started.

At 2026-06-05 10:08 PDT, gate job `3173688` completed. Result on held-out
OpenMath, `ISL=1024`, `OSL=1024`, `bs=32`, target `TP=4`, draft `TP=4`,
vLLM `v0.17.0`:

| Checkpoint | Output tok/s/GPU | Speedup vs baseline | Acceptance | Mean acceptance length | Notes |
|---|---:|---:|---:|---:|---|
| 1K CAT prefix-product venreuse | `436.34` | `0.901x` | `47.68%` | `3.384` | Acceptance improved over the 128-row CAT smoke, but throughput regressed below baseline and current public PARD recal. |

Accepted tokens by PARD K5 position:

| Position | Acceptance rate |
|---:|---:|
| 1 | `77.20%` |
| 2 | `57.54%` |
| 3 | `43.53%` |
| 4 | `33.69%` |
| 5 | `26.42%` |

Conclusion: scaling the CAT prefix-product approximation from `128` to `1024`
rows moved acceptance from `46.97%` to `47.68%`, but it did not recover
throughput. This is not good enough to promote to a NeMo-RL full GRPO run. The
next training-side step should be a real weighted CE / D-PACE-style dynamic
accepted-length loss rather than stochastic label masking only. A cheaper
NeMo-RL direct `VllmGeneration` gate can still be run if we need runtime
compatibility evidence for this checkpoint, but it should not be expected to
solve the Qwen3-235B speedup gap.

At 2026-06-05 10:19 PDT, the weighted CE / token-prefix-product ablation was
submitted using the same 1K teacher-logprob data:

| Job | Status | Purpose |
|---:|---|---|
| `3173861` | completed in `3:32` | K5 weighted CE train, `1024` rows, `lr=3e-6`, partial180 venv reuse, `CAT_LOSS_MODE=weighted_ce`, `CAT_IMPORTANCE_MODE=token_prefix_product` |
| `3173874` | completed | OpenMath `ISL=1024`, `OSL=1024`, bs32 K5 gate for expected `checkpoint-16` |

Expected weighted CE checkpoint:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_weightedce_tpp_1024_8x128_lr3e6/checkpoint-16
```

Training produced the checkpoint, but the reported `train_loss=19.09375` is a
warning sign. It may simply reflect a different weighted-CE loss scale, but it
resembles the older failed manual weighted-CE loss range. The gate result should
decide whether this objective is viable; if it fails, the safer next ablation is
to keep the original PARD/SFTTrainer loss path and express CAT via deterministic
weights/masks rather than popping `labels` and computing CE outside the model.

At 2026-06-05 10:35 PDT, gate job `3173874` completed. Result on held-out
OpenMath, `ISL=1024`, `OSL=1024`, `bs=32`, target `TP=4`, draft `TP=4`,
vLLM `v0.17.0`:

| Checkpoint | Output tok/s/GPU | Speedup vs baseline | Acceptance | Mean acceptance length | Notes |
|---|---:|---:|---:|---:|---|
| 1K weighted CE token-prefix-product | `420.62` | `0.869x` | `46.66%` | `3.333` | Regressed below both 1K CAT and baseline; do not promote. |

Accepted tokens by PARD K5 position:

| Position | Acceptance rate |
|---:|---:|
| 1 | `76.42%` |
| 2 | `56.26%` |
| 3 | `42.05%` |
| 4 | `32.62%` |
| 5 | `25.97%` |

Conclusion: this weighted-CE implementation should not be scaled. It reduced
acceptance versus the 1K stochastic CAT run and reduced throughput to `0.869x`
of baseline. The checkpoint can be treated as a cleanup candidate after the
result is retained in CSV/report form.

Cleanup note: obsolete PARD local checkpoints, including this weighted-CE
checkpoint, were removed after their results were captured. See
`docs/qwen3_235b_drafter_cleanup_2026_06_05.md`. The PARD-only cleanup reclaimed
about `86.2GB`. The latest 1K CAT venreuse checkpoint remains available for
comparison.

At 2026-06-05 11:11 PDT, two safer CAT ablations were submitted after the
weighted-CE failure. Both keep the original PARD/SFTTrainer model-loss path and
use `CAT_LOSS_MODE=mask`, so they avoid the custom CE loss-scale issue:

| Job | Status | Purpose |
|---:|---|---|
| `3174060` | completed in `3:27`, `train_loss=1.1969220638275146` | K5 mask-mode CAT train with `CAT_IMPORTANCE_MODE=token_prefix_product` |
| `3174175` | completed | OpenMath `ISL=1024`, `OSL=1024`, bs32 K5 gate for `3174060` checkpoint |
| `3174061` | completed in `3:26`, `train_loss=1.3213860988616943` | K5 mask-mode CAT train with `CAT_IMPORTANCE_MODE=token` |
| `3174176` | completed | OpenMath `ISL=1024`, `OSL=1024`, bs32 K5 gate for `3174061` checkpoint |

Both mask-mode runs have normal loss scale, unlike weighted CE
(`train_loss=19.09375`). Gate results:

| Checkpoint | Output tok/s/GPU | Speedup vs baseline | Acceptance | Mean acceptance length | Notes |
|---|---:|---:|---:|---:|---|
| 1K CAT `token_prefix_product` mask | `543.08` | `1.122x` | `46.71%` | `3.336` | Best local current-harness PARD/CAT result so far; still below historical public PARD K5 `1.31x`. |
| 1K CAT `token` mask | `495.88` | `1.024x` | `46.73%` | `3.337` | Near baseline/current public recal. |

Accepted-token rate by PARD K5 position:

| Checkpoint | Pos1 | Pos2 | Pos3 | Pos4 | Pos5 |
|---|---:|---:|---:|---:|---:|
| `token_prefix_product` mask | `75.62%` | `56.44%` | `42.58%` | `33.26%` | `25.65%` |
| `token` mask | `76.26%` | `56.99%` | `42.71%` | `32.57%` | `25.13%` |

Interpretation: custom weighted CE was the wrong implementation path; keeping
the PARD/SFTTrainer model loss and applying the PARD-2/CAT signal through
masking is materially better. The acceptance rate did not increase versus the
1K CAT venreuse run, so the `1.122x` speedup is likely runtime-shape/noise plus
better retained token distribution rather than a solved drafter-quality issue.
This checkpoint is worth a cheap NeMo-RL direct `VllmGeneration` gate, but not
yet a full GRPO run.

At 2026-06-05 11:46 PDT, that cheap NeMo-RL direct gate was submitted with
graph/CUDA-graph mode enabled, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, and
`VLLM_MOE_BACKEND=triton`. Both runs passed:

| Job | Status | Draft model | Output JSON |
|---:|---|---|---|
| `3174485` | pass | none, matched baseline | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/reports/nemorl_vllmgeneration/qwen235b_pard_nemorl_vllmgeneration_baseline_20260605_114643.json` |
| `3174486` | pass | `PARD-Qwen3-0.6B_qwen235b_math_k5_cat_tpp_mask_1024_8x128_lr3e6/checkpoint-16` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/reports/nemorl_vllmgeneration/qwen235b_pard_nemorl_vllmgeneration_local_cat_tpp_mask_k5_20260605_114646.json` |

Run shape: target `Qwen/Qwen3-235B-A22B`, OpenMath prompt file, prompt limit
`32`, generation batch size `32`, max new tokens `1024`, one node with four
GB200 GPUs, target `TP=4`, draft `TP=4`, `parallel_drafting=true`, and
`NUM_SPECULATIVE_TOKENS=5`.

Direct-gate result:

| Comparison | Generation tok/s | Speedup | Acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|
| matched baseline `3174485` | `1248.02` | `1.000x` | N/A | N/A |
| local CAT TPP-mask K5 `3174486` | `1599.75` | `1.282x` | `46.63%` | `3.332` |

Interpretation: the local `token_prefix_product` mask checkpoint now has a
positive NeMo-RL generation-backend signal and beats the previous public PARD
graph-on direct gate (`1.20x`) under the same 32-prompt direct shape. It is now
reasonable to promote this checkpoint to a GRPO-stack worker32 generation-only
smoke, while keeping the direct gate as the matched generation-only proof.

At 2026-06-05 15:20 PDT, the corrected full-GRPO worker32 generation-only pair
completed the first generation after fixing the invalid TP4
`gpu_memory_utilization=0.6` condition by rerunning with
`gpu_memory_utilization=0.90`, `max_model_len=8192`, fixed `256` decode tokens,
`temperature=0`, `top_p=1`, `ignore_eos=true`, and `enforce_eager=true`.

Run shape: target `Qwen/Qwen3-235B-A22B`, 32 nodes, 4 GB200 per node, total 128
GPUs, generation `TP=4`, training Megatron `TP=2`, `PP=8`, `CP=2`,
`EP=16`, GBS `256`, and about `32` requests per generation engine.

Full-GRPO stop-after-generation result:

| Job | Mode | Generation time | Gen tok/s/GPU | Policy-generate time | Policy-generate tok/s/GPU | Acceptance | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| `3175807` | baseline | `32.871s` | `15.576` | `31.582s` | `16.212` | N/A | `1.000x` |
| `3175808` | local CAT TPP-mask K5 | `19.136s` | `26.756` | `17.807s` | `28.753` | `53.53%` | `1.718x` generation, `1.774x` policy-generate |

This is the first controlled evidence in this run that Qwen3-235B SpecDec can
produce a real speedup inside the full NeMo-RL GRPO launcher path, not only in
standalone vLLM or direct `VllmGeneration`. Generation time decreased by
`41.8%`. The diagnostic still stops before reward/logprob/training, so it does
not yet prove E2E training-step speedup. Both jobs completed with Slurm exit
code `0:0`: `3175807` elapsed `35:14`, and `3175808` elapsed `31:38`.

Follow-up submitted at 15:22 PDT: `3176284` baseline and `3176285` K5 use the
same fixed-256, generation `TP=4`, `gpu_memory_utilization=0.90` shape, but set
`MAX_STEPS=2`, `NRL_STOP_AFTER_GENERATION=false`, and
`NRL_STOP_AFTER_GENERATION_AFTER_STEP=2`. This should exercise reward/logprob
and training after step 1, then stop after step 2 generation. It is the next
test for whether the generation-only win translates into useful E2E step-time
behavior. At 15:34 PDT both jobs were running in initial bootstrap with driver
logs created but not yet at Ray connect or `MasterConfig`; no failure signatures
had appeared. By 15:52 PDT, baseline `3176284` had passed vLLM model load/KV
allocation, entered vLLM sleep, and initialized `128/128` `lm_policy` workers.
K5 `3176285` failed before setup/generation during policy actor venv creation:
`megatron-core` build returned non-zero after `could not delete
megatron/core/datasets/helpers_cpp...so`. This is a launcher/actor-venv build
race, not a SpecDec runtime or PARD result. A K5-only retry `3176682` was
submitted with the same tail2 shape but with the previously successful
generation-only actor venv suffix reused to avoid rebuilding that policy venv.
Baseline `3176284` then reached `SETUP COMPLETE`, completed step 1 generation
and reward processing, and failed in `Computing logprobs` while calling
`get_reference_policy_logprobs`: Megatron DDP parameter sync raised
`RuntimeError: No backend type associated with device type cpu`. This is also
not a SpecDec failure; it is a reference-logprob/training-tail issue. The K5
reuse retry was cancelled before spending more time on the same unpatched tail
path. A corrected diagnostic pair, baseline `3176851` and K5 `3176852`, was
submitted with the same generation shape, reused actor venv suffixes,
`grpo.skip_reference_policy_logprobs_calculation=true`, and
`loss_fn.reference_policy_kl_penalty=0.0`. This changes the training objective,
so it should be interpreted only as an E2E performance-tail diagnostic, not as a
valid GRPO training-quality run.

At 16:38 PDT, both skip-reference jobs reached the same next failure point:
baseline `3176851` and K5 `3176852` both completed setup, step 1 generation,
reward processing, and logprob computation, then failed in `Training policy`.
The shared error was:

```text
TypeError: PackedSeqParams.__init__() got an unexpected keyword argument 'total_tokens'
```

This is not a SpecDec/PARD failure. The baseline had `speculative_config=None`
and hit the same error as the K5 run. Root cause: NeMo-RL's
`nemo_rl/models/megatron/data.py` was passing `total_tokens` into
Megatron-LM's `PackedSeqParams`, while the Megatron-LM version in this worktree
defines `PackedSeqParams` without that field. K5 did emit vLLM internal
SpecDecoding metrics before the training failure, with representative average
draft acceptance lines around `50-60%`, but no completed aggregate generation
time metric was emitted for this tail2 run.

Patch applied at 16:39 PDT: `nemo_rl/models/megatron/data.py` now builds
`PackedSeqParams` kwargs and passes `total_tokens` only when
`dataclasses.fields(PackedSeqParams)` contains that field. Remote
`python -m py_compile nemo_rl/models/megatron/data.py` passed. A fresh
actor-venv-label retry was submitted as baseline `3177357` and K5 `3177358`
with the same fixed-256/eager/TP4/`gpu_memory_utilization=0.90` tail2
skip-reference shape and `RUN_LABEL=fixed256-eagertrue-tp4-gpuutil090-packedparams-r1`.
Both were pending for priority at submission time.

At 16:46 PDT, `scontrol show job` still reported both `3177357` and `3177358`
as pending for `Priority`, not for a missing dependency or invalid resource
shape. Each job requests 32 nodes / 128 GPUs. The scheduler's current estimated
start for both jobs is `2026-06-05 18:54:39 PDT`, with candidate node lists
already assigned in `SchedNodeList`.

At 16:50 PDT, the scheduler estimate moved slightly to
`2026-06-05 19:01:50 PDT`. The jobs were still pending only for `Priority`.
Candidate node lists remained assigned by `scontrol`, so the retry is queued
normally rather than blocked by an invalid request.

At 16:57 PDT, both jobs were still pending only for `Priority`, with no driver
logs yet. The scheduler estimate moved slightly again to
`2026-06-05 19:05:07 PDT`. The next expected artifact is the pair of
`ray-driver.log` files under `experiments/eagle3_online/3177357-logs/` and
`experiments/eagle3_online/3177358-logs/`; once present, run
`scripts/extract_qwen235b_tail2_metrics.py --root experiments/eagle3_online --jobs 3177357 3177358`
from the remote worktree.

At 16:59 PDT, the status remained unchanged: both jobs were pending for
`Priority`, still with estimated start `2026-06-05 19:05:07 PDT` and no driver
logs.

Operational helper added at 17:00 PDT:
`scripts/poll_qwen235b_tail2_retry.sh` polls `3177357` / `3177358`, prints the
same stage counters used in manual polling, and then invokes
`scripts/extract_qwen235b_tail2_metrics.py` when logs exist. The helper was
copied to the remote worktree, `bash -n` passed, and a remote test run correctly
reported both jobs pending with no readable logs yet.

At 17:02 PDT, both jobs were still pending for `Priority` with no driver logs.
The scheduler estimate moved to `2026-06-05 18:29:22 PDT` for both jobs.

Additional patch validation at 16:55 PDT: a one-node / four-GPU container smoke
job `3177454` ran the same patched
`nemo_rl.models.megatron.data._pack_sequences_for_megatron` path with tiny CPU
tensors inside the NeMo-RL container. The container reported torch
`2.10.0+cu129`, and `PackedSeqParams` fields were:

```text
['qkv_format', 'cu_seqlens_q', 'cu_seqlens_kv', 'cu_seqlens_q_padded',
 'cu_seqlens_kv_padded', 'max_seqlen_q', 'max_seqlen_kv']
```

That confirms the local Megatron-LM dataclass has no `total_tokens` field. The
patched packing function returned a valid `PackedSeqParams` object and printed
`packedparams_smoke=PASS`; Slurm reported job `3177454` as `COMPLETED`, exit
`0:0`, elapsed `00:02:47`. This does not prove the full training tail is fixed,
but it proves the exact constructor mismatch that killed `3176851` / `3176852`
is resolved in the container environment.

Extraction tooling added: `scripts/extract_qwen235b_tail2_metrics.py` parses
`[SpecDec diag metrics]`, vLLM internal `SpecDecoding metrics`, stage markers,
and failure signatures from `ray-driver.log`. It was copied to the remote
worktree and verified against jobs `3175807`, `3175808`, `3176851`, and
`3176852`. The local snapshot CSV is
`docs/qwen3_235b_nemorl_tail_diagnostics_extracted_20260605.csv`.

Recent-method triage was refreshed in
`docs/qwen3_235b_recent_specdec_method_triage_20260605.md`. Current decision:
continue with PARD/PARD-2-style work because PARD is already runnable in vLLM
and PARD-2/CAT directly targets the observed acceptance-length objective. The
official AMD PARD repository still marks PARD-2 code/checkpoints as upcoming,
so the local CAT/TPP-mask trainer remains a practical approximation rather than
a complete official PARD-2 implementation.

At 2026-06-05 12:15 PDT, that worker32 promotion was submitted. This uses the
same Qwen3-235B recipe path and 32-node GRPO launcher, but still sets
`NRL_STOP_AFTER_GENERATION=true` so the first question is whether the speedup
survives the full NeMo-RL generation stack before spending time on optimizer
and training overhead:

| Job | Status | Mode | Shape |
|---:|---|---|---|
| `3174762` | running; Ray cluster up, 128/128 worker units connected; actor runtime env still building, no model init yet | baseline, SpecDec off | 32 nodes, 4 GPUs/node, generation `TP=16`, generation DP about `8`, GBS `256`, about `32` requests per engine |
| `3174763` | pending, priority | local CAT TPP-mask K5, always-on SpecDec | same matched shape |

This run intentionally does not submit K12 because the local checkpoint was
trained and gated as a K5 PARD checkpoint.

Early baseline progress at 2026-06-05 12:24 PDT: the 32-node Ray cluster came
up and all `128/128` worker units connected, then the driver entered Ray actor
runtime environment setup. No Megatron or vLLM model initialization log had
appeared yet. This is the same broad infrastructure area that made the prior
public-PARD worker32 run fail before useful generation metrics, so interpret a
failure here as GRPO-launcher/Ray stability unless the logs show a drafter or
SpecDec-specific error.

At about 12:30 PDT, `3174762` was still building actor runtime environments.
The logs showed the driver venv being ignored for actor envs and repeated
per-node uv downloads/builds for vLLM worker dependencies. The local worker32
wrapper now exposes `NEMO_RL_PY_EXECUTABLES_SYSTEM`; if this baseline fails
before model init/generation, the next controlled retry should set
`NEMO_RL_PY_EXECUTABLES_SYSTEM=1` to test whether bypassing actor uv env
creation gets the GRPO launcher path to generation.

At about 12:33 PDT, `3174762` was still running in the same actor-env build
phase after about `21` minutes. The latest log still showed packages such as
FlashInfer, fastsafetensors, and DeepGEMM being installed/built, with no
Megatron or vLLM model initialization yet. `3174763` remained pending.

At about 12:42 PDT, `3174762` was still running after about `27` minutes. The
driver had reached `MasterConfig` printing and W&B initialization, but there
were still no Megatron/vLLM model-load lines and no generation throughput or
acceptance metrics. Slurm/Ray still reported all `128/128` worker units
connected, while Ray resource usage for worker units was still `0.0/128.0`.
That means the run has not yet entered the actual generation workload. `3174763`
remained pending for priority.

At about 12:45 PDT, the same baseline run made forward progress: all vLLM actor
venvs finished and `vllm_policy` initialization completed for `128/128` workers
in `11.67s`. The next evidence needed is model-load completion and the
generation-only metric emitted by `NRL_STOP_AFTER_GENERATION=true`.

At about 12:47 PDT, the baseline entered vLLM engine/model initialization. The
log confirmed vLLM `v0.20.0`, `Qwen3MoeForCausalLM`, `bf16`, `TP=16`,
`max_model_len=8192`, `max_num_seqs=32`, `max_num_batched_tokens=32768`, and
loading of `118` safetensor checkpoint shards. This is now past the earlier
actor-venv bottleneck; the next pass/fail point is whether the 32-node
cross-node TP16 vLLM engine completes model load and emits generation timing.

At about 12:49 PDT, `3174762` was still running and model loading was still in
progress. The latest observed safetensor progress was `35%` (`41/118` shards).
No generation throughput or acceptance metric had been emitted yet.

At about 12:51 PDT, model loading was still progressing with no error markers.
The latest observed safetensor progress was `59%` (`70/118` shards). `3174763`
was still pending for priority.

At about 12:53 PDT, the baseline was still running and the latest observed
loading progress was `81%` (`96/118` shards). The main job was still healthy;
no generation metric had appeared yet.

At about 12:55 PDT, the latest observed loading progress was `99%`
(`117/118` shards). The run still had not emitted generation throughput yet,
but model load was effectively at the final shard.

At about 12:58 PDT, the baseline had progressed further. The log showed vLLM
weight loading reaching `100%` (`118/118` shards) on some workers, followed by
CUDA graph capture for mixed prefill/decode and decode-only graphs. The most
recent driver tail then moved into Megatron policy-worker runtime environment
builds (`transformer-engine`, `modelopt`, Megatron packages). There was still
no `NRL_STOP_AFTER_GENERATION` generation throughput metric and no acceptance
metric yet. This is still forward progress, not a failed run; the remaining
question is whether the full GRPO stack reaches the generation-only stop point
within the 4-hour allocation.

At about 13:00 PDT, `3174762` was still running after about `47` minutes and
`3174763` remained pending for priority. The driver log had grown and showed
policy-worker venv creation finishing across many nodes, followed by
`IsolatedWorkerInitializer` import-time warnings from the policy-worker venv.
No generation elapsed/throughput/tokens metric had been emitted yet.

Code-path note: `NRL_STOP_AFTER_GENERATION=true` does not skip worker setup. In
the patched GRPO path, vLLM generation workers and Megatron policy workers are
initialized before the training loop. This worker32 run is configured as
colocated inference (`policy.generation.colocated.enabled=true`), so it uses
sequential worker initialization on the same `32x4` GPUs rather than a separate
generation allocation. With `vLLM TP=16`, the effective generation DP is about
`8`, and `generation_batch_size=32` gives `256` responses per step. The stop
flag is checked only after the first rollout generation finishes and
`policy_generation.finish_generation()` returns. Therefore this worker32 smoke
is a generation-only measurement once it reaches rollout, but it still pays the
full GRPO stack initialization cost before the first metric.

At about 13:03 PDT, `3174762` was still running after about `50` minutes. The
`lm_policy` worker initialization progress reached `128/128`, with Megatron
policy-worker import warnings but no fatal error marker. There was still no
early generation throughput metric and no SpecDec counter metric; the run had
not reached rollout generation yet.

At about 13:04 PDT, baseline `3174762` reached the rollout generation stage:
`Generating responses for batch of size 256`. The setup timing printed by GRPO
was `vLLM init: 2068.5s`, `Policy init: 291.6s`, `Other setup: 162.9s`, and
`Total setup: 2527.5s`. This confirms the first metric is delayed mostly by
vLLM initialization and full GRPO worker setup, not by a failure before
generation. At the same time, K5 job `3174763` moved from pending to running on
`nvl72040-T[01-16],nvl72101-T[01-16]`, but its driver log had not been created
yet.

At about 13:06 PDT, baseline `3174762` was still in generation with no early
generation metric yet; the driver log had not advanced past the generation
start marker. K5 `3174763` was still running, but still had no ray-driver log.

At about 13:09 PDT, baseline `3174762` was still running in the same generation
stage with no metric emitted yet. This is not necessarily abnormal: the step is
`256` responses with fixed `1024` generated tokens, or about `262K` generated
tokens before accounting for prompt tokens. K5 `3174763` had all `128/128` Ray
worker units connected and the driver `srun` launched, but its ray-driver log
was still zero bytes at that poll.

At about 13:13 PDT, baseline `3174762` was still running in generation with no
metric yet, but a sampled baseline node showed active GPU work (`100%, 100%,
0%, 100%` utilization with about `123GB` used per GPU). That supports "long
decode still running" rather than a driver-log stall. K5 `3174763` had printed
`MasterConfig` with `speculative_config.method=draft_model`,
`num_speculative_tokens=5`, and `parallel_drafting=true`; it had not reached
vLLM load or generation metrics yet. A sampled K5 node still showed near-zero
GPU memory/use at that point.

At about 13:21 PDT, baseline `3174762` had left the queue and `sacct` reported
`FAILED` after `01:02:04` with exit code `1:0`. It did not emit the early
generation metric. The failure happened inside baseline vLLM generation:
the dumped vLLM engine config had `speculative_config=None`, so this is not a
PARD/SpecDec-specific failure. The run had entered `Generating responses for
batch of size 256`; sampled scheduler dumps showed partial decode progress
with `num_output_tokens` roughly `102` to `309` per request and very low
`kv_cache_usage` around `0.0025` to `0.0061`. The fatal signature was Ray
`SYSTEM_ERROR` / `ActorDiedError`, `RayWorkerWrapper` `Aborted`, and connection
error code `2` / end-of-file. This means the full colocated GRPO worker32 path
is currently unstable for the fixed `256 x 1024` decode even without SpecDec.
K5 `3174763` was still running at this point; it had initialized all
`128/128` vLLM policy workers and was loading the 118 Qwen3-235B safetensor
shards, but had not emitted timing or acceptance metrics yet.

After that failure, the worker32 launcher was updated to expose
`MAX_NEW_TOKENS`, `MIN_TOKENS`, `VLLM_ENFORCE_EAGER`, and `RUN_LABEL`
overrides. A controlled stability retry was submitted at about 13:25 PDT:
baseline `3175119` and K5 `3175120`. This retry keeps the same `32` nodes x
`4` GPUs, generation `TP=16`, effective generation `DP~8`, total responses
`256`, and per-engine request batch `32`, but changes the fixed decode from
`1024` tokens to `256` tokens and sets `policy.generation.vllm_cfg.enforce_eager=true`.
It also uses `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` to avoid repeating avoidable
actor-venv setup pressure. At the first poll both retry jobs were pending due
to priority, while the original fixed-1024 K5 job `3174763` continued running.

At about 13:28 PDT, K5 `3174763` had completed target Qwen3-235B weight loading
on sampled workers (`118/118` safetensor shards) and moved on to draft-model
loading (`0/1` shard marker). It still had not reached rollout generation or
emitted timing/acceptance metrics.

At about 13:32 PDT, K5 `3174763` had moved into CUDA graph capture. The log
showed mixed prefill-decode capture over `41` sizes and decode-full capture
over `25` sizes on sampled workers. It still had not printed the rollout
`Generating responses` marker or any generation throughput / acceptance metric.

At about 13:37 PDT, K5 `3174763` had completed vLLM engine initialization and
put vLLM workers into sleep mode while the Megatron `lm_policy` worker group
initialized. A sampled node showed `0%` GPU utilization and about `10.7GB`
memory per GPU, matching the expected sleep-mode state before rollout
generation. No timing or acceptance metric had emitted yet.

At about 13:41 PDT, K5 `3174763` printed `SETUP COMPLETE` and entered rollout
generation with `Generating responses for batch of size 256`. This reaches the
same phase where baseline `3174762` failed, so the next observation determines
whether the remaining issue is baseline-path instability shared by K5 or a
K5-specific runtime/performance result.

At about 13:52 PDT, K5 `3174763` failed after `00:46:32` elapsed, before the
early generation aggregate metric could be emitted. This failure is more
specific than the baseline actor-death signature: vLLM workers reported
`ProcessGroupNCCL` watchdog timeouts on `_ALLGATHER_BASE`
(`NumelIn=1823232`, `NumelOut=29171712`, `Timeout(ms)=600000`), then
`ProcessGroupNCCL` took the process down and Ray reported
`ActorDiedError` / `SYSTEM_ERROR` connection end-of-file. This means fixed
`256 x 1024` full-GRPO worker32 generation is still not a usable performance
comparison path. The baseline fails without SpecDec; K5 fails with a clearer
NCCL all-gather timeout inside the speculative vLLM generation path.

K5 did emit partial worker-level SpecDec metrics immediately before the
failure, but no completed generation throughput. The observed worker-level
metrics included mean acceptance length around `3.31` to `3.99`, accepted
tokens around `11923` to `15305`, drafted tokens around `25600` to `25760`,
and average draft acceptance rate around `46.3%` to `59.8%`. These values are
useful evidence that the K5 drafter was active, but they are not a completed
NeMo-RL throughput result because the generation task failed before finishing.

Scheduler dumps at failure showed `32` running requests per generation engine,
scheduled speculative decode tokens present, total scheduled tokens of `192`
(`32` requests times one target token plus five draft tokens), and partial
outputs around `113` to `269` tokens per request on sampled engines. KV usage
was low (`~0.0042` to `~0.0067`), so the failure is not explained by KV cache
capacity. The most plausible current root cause is the full-GRPO colocated
vLLM generation shape: Qwen3-235B uses generation `TP=16`, which spans multiple
4-GPU GB200 nodes per generation engine. The direct `VllmGeneration` gate that
passed used one node / four GPUs (`TP=4`), while the full worker32 path uses
cross-node TP collectives and failed in a vLLM `_ALLGATHER_BASE` collective.

To test that root-cause hypothesis, the worker32 wrapper was updated to expose
`GENERATION_TP`. A second stability pair was submitted at about 13:58 PDT:
baseline `3175394` and K5 `3175395`, using fixed `256` decode,
`enforce_eager=true`, `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`, and generation
`TP=4`. This keeps the full GRPO/colocated launcher path but avoids cross-node
TP inside each vLLM generation engine. The tradeoff is that, at `GBS=256`, the
per-engine request batch becomes smaller than the prior worker32 comparison;
this run is a stability/root-cause test first, not the final throughput
comparison.

At about 14:01 PDT, a cheaper direct `VllmGeneration` isolation pair was also
submitted to test the cross-node TP hypothesis without Megatron/GRPO setup:
baseline `3175487` and local CAT TPP-mask K5 `3175488`. This uses
`NUM_NODES=4`, `GPUS_PER_NODE=4`, target `TP=16`, draft `TP=16`,
`max_new_tokens=256`, `generation_batch_size=32`, `max_model_len=4096`,
`enforce_eager=true`, and the same local checkpoint
`PARD-Qwen3-0.6B_qwen235b_math_k5_cat_tpp_mask_1024_8x128_lr3e6/checkpoint-16`.
If this direct TP16 gate fails with the same `_ALLGATHER_BASE` / actor-death
signature, the failure is likely a cross-node vLLM TP runtime issue rather
than a full-GRPO-only problem. If it passes, the next focus is the colocated
GRPO launcher/sleep/resume path and per-engine scheduling pressure.

At about 14:09 PDT, the fixed-256/eager full-GRPO TP16 pair `3175119` /
`3175120` failed before model load with
`ModuleNotFoundError: No module named 'vllm'` inside
`VllmGenerationWorker.__init__`. This is not a drafter or SpecDec result. It
is an environment regression caused by using
`NEMO_RL_PY_EXECUTABLES_SYSTEM=1` without also putting the source vLLM site on
the actor `PYTHONPATH`. The TP4 pair `3175394` / `3175395` used the same
system-env setting, so it was cancelled before spending more setup time.

The worker32 wrapper now exposes `SOURCE_VLLM_SITE` and propagates it through
`PYTHONPATH` before `submit_nemorl_online_draft_specdec.sh` builds the driver
command. This matches the direct `VllmGeneration` gate, which already imports
vLLM from the same source site. The fixed-256/eager full-GRPO TP4 stability
pair was resubmitted with this fix at about 14:11 PDT: baseline `3175527` and
K5 `3175528`.

That TP4 system-env retry failed before the GRPO driver started. The root
cause was not missing vLLM anymore; it was global `PYTHONPATH` poisoning of
Ray startup. The Ray CLI imported `jsonschema` from the source vLLM site, then
failed with `ModuleNotFoundError: No module named 'rpds.rpds'` because the
compiled extension was not available in that site path. Jobs `3175527` and
`3175528` therefore are also not generation/performance results. The wrapper
was changed again to remove the global source-vLLM `PYTHONPATH` injection. For
full GRPO, use the actor venv path for now (`NEMO_RL_PY_EXECUTABLES_SYSTEM=0`)
unless we implement actor-only vLLM path propagation.

At about 14:19 PDT, the fixed-256/eager full-GRPO TP4 stability pair was
resubmitted using the actor venv path instead of system actors: baseline
`3175668` and K5 `3175669`. This is the next valid full-GRPO stability test.

At about 14:25 PDT, K5 job `3175669` failed during Ray worker bootstrap before
the GRPO driver was created. The Slurm launcher reported a background srun
death for `ray-worker-21`, followed by multiple late worker steps failing or
being cancelled. No model load, generation metric, or SpecDec acceptance metric
was produced, so this is a startup failure rather than a PARD/runtime result.
Baseline `3175668` was still running at the same poll, with the Ray cluster
starting and no driver log yet.

To avoid spending another full baseline allocation, a K5-only wrapper was added:
`experiments/eagle3_online/submit_qwen235b_pard_local_tpp_mask_gbs256_worker32_k5_only.sh`.
It reuses the same fixed-256/eager/TP4 actor-venv settings and submits only the
SpecDec side. K5-only retry `3175744` was submitted at about 14:32 PDT and was
pending on priority at the first poll.

At about 14:36 PDT, baseline `3175668` had all `128/128` Ray worker units
connected and had printed the MasterConfig. It had not yet reached model init
or generation metrics. K5-only retry `3175744` was running and had passed the
initial two-minute startup failure window, but no driver log had appeared yet.

At about 14:40 PDT, baseline `3175668` was still building the vLLM generation
worker actor environment, including DeepGEMM. K5-only `3175744` had created an
empty driver log but had not printed MasterConfig yet. Neither job had emitted
model init, generation throughput, or acceptance metrics.

At about 14:43 PDT, K5-only `3175744` had also printed MasterConfig and was
building the same vLLM generation worker actor environment. This means it
passed the prior `3175669` Ray-worker startup failure point. Both full-GRPO
jobs were still pre-model-init, so there were still no throughput or acceptance
metrics.

At about 14:48 PDT, baseline `3175668` failed before generation after vLLM
loaded Qwen3-235B at `TP=4`. The loaded model used about `109.55GiB` per GPU,
and the recipe default `gpu_memory_utilization=0.6` left no memory for KV cache
blocks. The failure was `ValueError: No available memory for the cache blocks`,
raised inside vLLM's KV-cache initialization. This is a configuration-capacity
failure, not a PARD or generation-speed result. K5-only `3175744` was cancelled
because it used the same invalid `TP=4` / `gpu_memory_utilization=0.6` setting.

The full-GRPO wrapper now supports explicit `GPU_MEMORY_UTILIZATION` and
`MAX_MODEL_LEN` overrides. A corrected pair was submitted at about 14:50 PDT:
baseline `3175807` and K5 `3175808`, with fixed-256/eager/TP4,
`gpu_memory_utilization=0.90`, and `max_model_len=8192`.

At about 14:17 PDT, the direct TP16 `VllmGeneration` pair completed and changed
the root-cause picture. Baseline `3175487` passed with `8192` generated tokens,
`37.362s` generation elapsed, and `219.26 tok/s`. K5 `3175488` passed with
`8192` generated tokens, `23.045s` generation elapsed, and `355.47 tok/s`.
This is `1.621x` generation throughput speedup at `47.42%` acceptance,
`5773` accepted tokens, `12175` draft tokens, `2435` drafts, and mean
acceptance length `3.371`.

Conclusion from the direct TP16 isolation: cross-node target/draft `TP=16`
itself is not the primary blocker. It can load and generate in NeMo-RL's direct
`VllmGeneration` path, and it shows a clear K5 speedup. The remaining full-GRPO
problem is more likely in the colocated GRPO launcher path, Ray actor
environment setup, vLLM sleep/resume, or per-engine scheduling pressure, not
the PARD drafter or TP16 vLLM runtime in isolation.

## Do Not Repeat

- Do not claim the historical public PARD K5 `1.31x` OpenMath row as current
  harness behavior without a paired current rerun. Job `3171868` showed only
  `1.00x` under the current harness/node-era condition.
- Do not scale the abandoned manual weighted-CE trainer. Jobs `3171157` and
  `3171202` produced abnormal loss around `22.95`.
- Do not keep new training artifacts on the fsw artifact root when scaling.
  Several jobs failed before stdout with `RaisedSignal:53`; moving cache,
  checkpoints, logs, and run dirs to fs1 fixed job `3171476`.
- Do not use `draft_tensor_parallel_size=1` for Qwen3-235B PARD gates. vLLM
  requires draft TP to match target TP in this path; use `draft_tp=4` for
  target `TP=4`.
- Do not interpret a failed full 32-node GRPO worker32 baseline as a drafter
  result. Job `3174762` failed with `speculative_config=None`, so it proves the
  colocated full-GRPO generation path needs a stability fix before a K5
  comparison is meaningful.
- Do not interpret K5 fixed-1024 worker32 job `3174763` as a completed speedup
  run. It emitted partial SpecDec acceptance metrics, then failed in NCCL
  `_ALLGATHER_BASE` watchdog timeout before generation completed.
- Do not spend full 32-node GRPO runs before a generation-only gate passes and
  the full baseline path is stable. The direct `VllmGeneration` gate is the
  cheaper proof that NeMo-RL can load and exercise the runtime.
- Do not set `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` in full GRPO generation runs
  without an actor-only vLLM path plan. Jobs `3175119` and `3175120` failed in
  `VllmGenerationWorker.__init__` with `ModuleNotFoundError: No module named
  'vllm'`. Adding the source vLLM site to global `PYTHONPATH` fixed that import
  but broke Ray startup itself via `jsonschema`/`rpds.rpds` shadowing in jobs
  `3175527` and `3175528`.
- Do not treat K5 job `3175669` as evidence about PARD speedup or acceptance.
  It died during Ray worker bootstrap before the driver, model load, or
  generation phase.
- Do not rerun Qwen3-235B full-GRPO `generation_tp4` with the recipe default
  `gpu_memory_utilization=0.6`. Job `3175668` loaded the model at about
  `109.55GiB` per GPU and then failed with no available memory for KV cache
  blocks. Use an explicit higher utilization such as `0.90` or return to
  `generation_tp16`.
- Do not promote the 128-row CAT checkpoint. It is an integration smoke for the
  PARD-2-style logprob path, not a speedup result.
- Do not submit a single 1K teacher-logprob job with an 8-hour limit on the
  current batch partition; the partition rejected that limit. Also avoid 512-row
  chunks for this Qwen3-235B logprob path under the 4-hour batch limit. Use
  128-row chunks and depend the trainer on all chunks.
- Do not burn a full allocation in dependency installation if the same trainer
  imports are already present. Prefer `INSTALL_PARD_REQUIREMENTS=auto`, or pass
  a validated reusable `VENV_DIR`/`PYTHON_RUNNER`. Job `3173318` showed the
  cost of a fresh venv on the current container.
- Do not delete final internally trained 500K EAGLE3 checkpoints. Only
  intermediate openmath chunk directories were removed; the final
  `eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel` and
  `eagle3_qwen3_30ba3b_mixed_math_nonopenmath_500k_parallel` checkpoints were
  verified present after cleanup.
- Do not interpret tail2 skip-reference jobs `3176851` / `3176852` as failed
  SpecDec performance runs. Both baseline and K5 completed generation,
  rewards, and logprobs, then failed in training because the local Megatron-LM
  `PackedSeqParams` dataclass does not accept `total_tokens`. Use the
  patched retry `3177357` / `3177358`, which passes `total_tokens` only when
  that field exists.

## Next Experiments

1. Poll the fixed-256/eager retry pairs:
   - `3175119` / `3175120`: generation `TP=16`; failed before model load due
     to missing vLLM in system actor env, not a performance result.
   - `3175394` / `3175395`: generation `TP=4`; cancelled after the same
     environment risk was confirmed.
   - `3175527` / `3175528`: generation `TP=4`, fixed source-vLLM `PYTHONPATH`;
     failed before driver because global source-vLLM `PYTHONPATH` poisoned Ray
     startup via `jsonschema` / `rpds.rpds`.
   - `3175668` / `3175669`: generation `TP=4`, actor venv path, fixed-256
     decode. `3175669` failed during Ray worker bootstrap before driver/model
     load; `3175668` later failed before generation because
     `gpu_memory_utilization=0.6` left no room for KV cache blocks.
   - `3175744`: K5-only retry with the same fixed-256/eager/TP4 actor-venv
     settings; cancelled because it shared the same invalid TP4/0.6 memory
     condition as `3175668`.
   - `3175807` / `3175808`: corrected fixed-256/eager/TP4 pair with
     `gpu_memory_utilization=0.90` and `max_model_len=8192`; both reached
     `MasterConfig` by the 14:58 PDT poll. `3175807` is the no-spec baseline and
     `3175808` is K5 with `parallel_drafting=true`. At 15:03 PDT, `3175808`
     reached Qwen3-235B safetensors loading with sampled workers around
     `38-46/118` shards loaded. At 15:05 PDT, `3175808` printed vLLM engine
     initialization as vLLM `v0.20.0` with K5 `draft_model` speculative config,
     target `TP=4`, `max_seq_len=8192`, and `enforce_eager=True`; sampled model
     loads reported about `109.84GiB` memory and no KV-cache memory failure yet.
     At 15:07 PDT, `3175807` also reached vLLM `v0.20.0` engine initialization
     with `speculative_config=None`, target `TP=4`, and sampled model-load memory
     around `109.55GiB`. Neither job had hit `No available memory`, `SETUP
     COMPLETE`, or generation metrics yet. By 15:10 PDT, sampled workers in both
     jobs had passed KV-cache allocation: baseline showed about `50.3GiB`
     available KV memory, `1,122,160` GPU KV-cache tokens, and `136.98x`
     max concurrency for `8192` tokens/request; K5 showed about `50.0GiB`
     available KV memory, `698,992` GPU KV-cache tokens, and `85.33x` max
     concurrency. This confirms the TP4/`gpu_memory_utilization=0.90` retry
     clears the earlier `gpu_memory_utilization=0.6` KV-cache allocation failure,
     at least on sampled workers. By 15:17 PDT, both jobs had moved past vLLM
     generation-worker initialization into full-GRPO policy setup: vLLM workers
     had entered sleep mode, all `128/128` `lm_policy` workers initialized, and
     both jobs were loading the Qwen3-235B Megatron checkpoint at iteration `0`.
     No `SETUP COMPLETE`, generation throughput, or SpecDec acceptance metric had
     appeared yet, and neither job had a failure signature.
     Code inspection confirms this wait is expected for the current diagnostic:
     `NRL_STOP_AFTER_GENERATION=true` is checked only after
     `policy_generation.finish_generation()`, so it does not skip setup or
     Megatron policy checkpoint loading. It only stops before reward, logprob,
     and training after the first generation is complete. At 15:20 PDT, the pair
     produced matched generation-only metrics: baseline `3175807` generated in
     `32.871s` at `15.576` tok/s/GPU; local CAT TPP-mask K5 `3175808`
     generated in `19.136s` at `26.756` tok/s/GPU with `53.53%` acceptance and
     mean acceptance length `3.677`. This is `1.718x` generation throughput
     speedup and `41.8%` generation-time reduction. The policy-generate timer
     speedup was `1.774x`. Both jobs completed with exit code `0:0`.
2. Poll the direct TP16 `VllmGeneration` isolation pair:
   - `3175487` / `3175488`: completed direct TP16 fixed-256 gate. K5 was
     `1.621x` faster than baseline with `47.42%` acceptance.
3. The full colocated GRPO generation-only baseline is now stable enough for
   matched K5 comparison at fixed `256` decode, generation `TP=4`,
   `gpu_memory_utilization=0.90`, and `enforce_eager=true`. The next controlled
   run should test whether this survives beyond generation-only by stopping
   after step 2 or by enabling the normal reward/logprob/training tail. The
   active patched retry for this is `3177357` baseline and `3177358` K5.
   Important: `3175807` / `3175808` are generation-only diagnostics and must not
   be presented as full-GRPO E2E speedup. A true no-stop full-GRPO pair was
   submitted at 2026-06-05 17:18 PDT: baseline `3177848` and K5 `3177849`.
   This pair uses `MAX_STEPS=2`, `STOP_AFTER_GENERATION=false`,
   empty `STOP_AFTER_GENERATION_AFTER_STEP`, and
   `SKIP_REFERENCE_LOGPROBS=false`, so reward, reference logprobs, and policy
   training remain in the step path.
   A longer full-GRPO pair was also submitted at 2026-06-05 17:22 PDT for a
   less noisy E2E measurement: baseline `3177855` and K5 `3177856`, same fixed
   decode/runtime shape but `MAX_STEPS=20`. It is also no-stop:
   `STOP_AFTER_GENERATION=false`, empty `STOP_AFTER_GENERATION_AFTER_STEP`,
   and `SKIP_REFERENCE_LOGPROBS=false`.
4. Keep the direct `VllmGeneration` gate as the cheap isolation gate, but the
   worker32 pair is now the best NeMo-RL launcher-path evidence. The same-node
   TP4 local TPP-mask gate gave baseline `1248.02 tok/s`, K5
   `1599.75 tok/s`, `1.282x` speedup, and `46.63%` acceptance. The cross-node
   TP16 fixed-256 direct gate gave baseline `219.26 tok/s`, K5 `355.47 tok/s`,
   `1.621x` speedup, and `47.42%` acceptance.
5. Keep the original PARD/SFTTrainer model-loss path unless a custom loss can
   reproduce the baseline loss scale on a tiny checkpoint first.
6. If the full NeMo-RL/GRPO smoke loses the direct-gate speedup, diagnose the
   non-generation overhead split before changing the drafter again.
7. Scale beyond 1K rows only after the 1K objective shows a throughput trend;
   the 1K stochastic CAT run already showed that more rows alone are not enough.

## Qwen3-30B-A3B PARD-2-Style Follow-Up

The official AMD PARD repository still does not expose a usable PARD-2 training
implementation or public PARD-2 checkpoints, so the runnable test uses the
local CAT/PARD-2-style approximation: AMD PARD parallel drafting plus
target-token confidence masking with `CAT_IMPORTANCE_MODE=token_prefix_product`
and `CAT_LOSS_MODE=mask`.

Submitted at 2026-06-05 17:14-17:18 PDT:

| Job | Purpose | Status shape |
| --- | --- | --- |
| `3177753` | Qwen3-30B-A3B-Thinking-2507 teacher generation with generated-token logprobs | cancelled after a context-length retry stall at `178/1024` rows |
| `3177808` | CAT/PARD-2-style K5 train | cancelled with the original stalled teacher dependency |
| `3177811` | vLLM standalone OpenMath baseline | completed; `ISL=1024`, `OSL=1024`, bs `1 2 4 8 16 32`, target `TP=1` |
| `3177818` | vLLM standalone public PARD K5 gate | completed; same held-out OpenMath shape |
| `3177823` | vLLM standalone local CAT/PARD-2-style K5 gate | cancelled with the original stalled train dependency |

Current Qwen3-30B-A3B standalone public-PARD result:

| Batch size | Baseline tok/s | Public PARD K5 tok/s | Speedup | Acceptance | Mean acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `305.49` | `151.53` | `0.496x` | `53.07%` | `3.654` |
| 2 | `515.89` | `177.39` | `0.344x` | `40.53%` | `3.027` |
| 4 | `884.75` | `381.57` | `0.431x` | `37.15%` | `2.858` |
| 8 | `1451.49` | `739.54` | `0.510x` | `38.54%` | `2.927` |
| 16 | `2323.43` | `1397.68` | `0.602x` | `38.22%` | `2.911` |
| 32 | `3770.51` | `2663.93` | `0.707x` | `37.58%` | `2.879` |

This is a clear slowdown for generic public PARD on Qwen3-30B-A3B Thinking
OpenMath at this shape. It should not be confused with the earlier RedHatAI
EAGLE3 high-batch result, which used a different drafter and K.

Raw CSV:

- `docs/qwen3_30ba3b_pard_public_k5_vllm_metrics_20260605.csv`

Early local CAT/PARD-2-style result:

While the 1024-row teacher job was still running, an early 256-row local CAT
checkpoint was trained and gated to avoid waiting for the full chain before
checking directionality. This is not the final 1024-row result, but it is useful
evidence about the objective.

| Batch size | Public PARD K5 speedup | Local CAT 256 K5 speedup | Public acceptance | Local acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.496x` | `0.520x` | `53.07%` | `53.33%` |
| 2 | `0.344x` | `0.456x` | `40.53%` | `41.29%` |
| 4 | `0.431x` | `0.469x` | `37.15%` | `38.03%` |
| 8 | `0.510x` | `0.533x` | `38.54%` | `39.07%` |
| 16 | `0.602x` | `0.634x` | `38.22%` | `38.57%` |
| 32 | `0.707x` | `0.759x` | `37.58%` | `38.25%` |

Interpretation: the local CAT objective improves over public PARD slightly at
the same OpenMath shape, especially at bs32 (`0.759x` versus `0.707x`), but it
is still a slowdown against baseline. The 1024-row checkpoint still needs to be
gated before deciding whether to scale this objective.

Artifacts:

- `docs/qwen3_30ba3b_pard2cat_early256_vllm_metrics_20260605.csv`
- `docs/qwen3_30ba3b_pard2cat_early256_vllm_speedup_acceptance.png`

Follow-up scale check:

| Job | Purpose | Status |
| ---: | --- | --- |
| `3178294` | early 512-row local CAT/PARD-2-style train | completed, exit `0:0` |
| `3178295` | early 512-row local CAT/PARD-2-style vLLM gate | completed, exit `0:0` |

512-row result:

| Batch size | Public PARD K5 speedup | Local CAT 256 K5 speedup | Local CAT 512 K5 speedup | Public acceptance | CAT 256 acceptance | CAT 512 acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.496x` | `0.520x` | `0.536x` | `53.07%` | `53.33%` | `53.86%` |
| 2 | `0.344x` | `0.456x` | `0.481x` | `40.53%` | `41.29%` | `42.37%` |
| 4 | `0.431x` | `0.469x` | `0.488x` | `37.15%` | `38.03%` | `39.02%` |
| 8 | `0.510x` | `0.533x` | `0.651x` | `38.54%` | `39.07%` | `42.32%` |
| 16 | `0.602x` | `0.634x` | `0.670x` | `38.22%` | `38.57%` | `40.59%` |
| 32 | `0.707x` | `0.759x` | `0.717x` | `37.58%` | `38.25%` | `40.02%` |

Interpretation after 256/512 rows: CAT training improves acceptance over public
PARD, and 512 rows improves acceptance more than 256 rows. However, throughput
does not monotonically follow acceptance: bs32 regressed from `0.759x` at 256
rows to `0.717x` at 512 rows despite acceptance rising from `38.25%` to
`40.02%`. That suggests the current objective/runtime combination is still
overhead-limited at this shape; acceptance needs to rise much further, or the
parallel-draft overhead must be reduced, before this becomes a baseline speedup.

Additional artifacts:

- `docs/qwen3_30ba3b_pard2cat_early512_vllm_metrics_20260605.csv`
- `docs/qwen3_30ba3b_pard2cat_early512_vllm_speedup_acceptance.png`

1024-row local CAT/PARD-2-style result:

| Batch size | Public PARD K5 speedup | CAT 256 speedup | CAT 512 speedup | CAT 1024 speedup | CAT 1024 acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.496x` | `0.520x` | `0.536x` | `0.542x` | `54.95%` |
| 2 | `0.344x` | `0.456x` | `0.481x` | `0.411x` | `42.57%` |
| 4 | `0.431x` | `0.469x` | `0.488x` | `0.422x` | `39.36%` |
| 8 | `0.510x` | `0.533x` | `0.651x` | `0.466x` | `42.74%` |
| 16 | `0.602x` | `0.634x` | `0.670x` | `0.605x` | `40.71%` |
| 32 | `0.707x` | `0.759x` | `0.717x` | `0.870x` | `40.22%` |

Interpretation after 1024 rows: scaling the current CAT objective helps the
large-batch bs32 case substantially (`0.870x` versus public `0.707x`), but it
still does not cross baseline. It also regresses smaller batches relative to
the 512-row checkpoint. This points to a mixed problem: acceptance is improving,
but the public PARD runtime overhead and/or draft quality is still not good
enough for a real OpenMath vLLM speedup at `TP=1`, `ISL=1024`, `OSL=1024`.

Additional final artifacts:

- `docs/qwen3_30ba3b_pard2cat_1024_vllm_metrics_20260605.csv`
- `docs/qwen3_30ba3b_pard2cat_1024_vllm_speedup_acceptance.png`

Fix applied after the original teacher stall:

- `generate_training_conversations_openai.py` now parses the newer vLLM
  context-limit error form: `passed X input tokens and requested Y output
  tokens ... context length is only Z tokens`, then reduces `max_tokens`
  instead of blindly retrying the same invalid request.
- `submit_teacher_math_continuations.sh` now passes
  `GENERATION_SKIP_FAILED` through to the remote sbatch payload.
- `submit_qwen30ba3b_pard2_cat_pipeline.sh` defaults
  `TEACHER_SKIP_FAILED=true` for this PARD-2-style teacher-logprob path.
- A further safety margin of `8` tokens is now applied when reducing
  `max_tokens` after context-limit responses. This avoids repeated one-token
  boundary failures on prompts close to `max_model_len`.

Resume chain submitted at 2026-06-05 17:34-17:35 PDT:

| Job | Purpose | Status at submit |
| --- | --- | --- |
| `3178010` | teacher-logprob resume, appending to the original 178-row output | completed, `1023` rows written, `1` long prompt skipped |
| `3178032` | CAT/PARD-2-style K5 train | completed, exit `0:0`, train loss `1.2494` |
| `3178033` | vLLM standalone local CAT/PARD-2-style K5 gate | completed, exit `0:0`; best row bs32 `0.870x`, `40.22%` acceptance |

The active status file is
`latest_qwen30ba3b_pard2_cat_resume_jobs.txt`. Do not treat `3178033` as a
completed local PARD-2-style result until the dependent teacher/train jobs
finish and its `breakdown.json` is present.

Queue hygiene update: after the user requested the `MAX_STEPS=20` full-GRPO
measurement, the older pending 32-node Qwen3-235B diagnostics were cancelled at
17:38 PDT to keep the queue focused on the real no-stop run:

| Job | Reason for cancellation |
| ---: | --- |
| `3177357` / `3177358` | tail2 skip-reference diagnostic, not final full-GRPO |
| `3177848` / `3177849` | no-stop `MAX_STEPS=2` sanity pair, superseded by `MAX_STEPS=20` |

The active Qwen3-235B full-GRPO no-stop pair remains `3177855` baseline and
`3177856` K5, both `MAX_STEPS=20`. At the latest check they are still pending
for scheduler priority. At 20:03 PDT, both jobs were still `PENDING`, elapsed
time `0:00`, with a `4:00:00` time limit. The remote login node could stat
`/lustre/fsw/portfolios/coreai/users/sna`, but `/lustre/fs1/.../users/sna`
was not accessible, so the next risk is filesystem availability when these jobs
actually start.

## DFlash Follow-up

PARD/CAT is still the nearest drop-in NeMo-RL path, but the Qwen3-30B-A3B and
Qwen3-235B evidence now shows that acceptance improvements alone are not enough
when draft overhead remains high. The next lower-overhead algorithmic branch is
vLLM Speculators DFlash: it predicts a draft block in one forward pass, instead
of autoregressively stepping through each draft token. This directly targets the
current failure mode where the verifier can accept some tokens but the draft
path and MoE/system overhead erase the gain.

Primary references:

- vLLM Speculators DFlash docs:
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/
- vLLM Speculators decision guide:
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/
- DFlash paper: https://arxiv.org/abs/2602.06036
- P-EAGLE docs:
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/peagle/

Submitted compatibility smoke:

| Job | Purpose | Shape | Status |
| ---: | --- | --- | --- |
| `3178708` | Qwen3-235B DFlash train-only smoke | newly prepared 512-row data, existing 50K OpenMath hidden-state pool, `MAX_SAMPLES=512`, `TARGET_LAYER_IDS=1 46 90`, `NUM_LAYERS=2`, `BLOCK_SIZE=6`, `MAX_ANCHORS=128`, `NOISE_STD=0`, `MASK_TOKEN_ID=151670`, explicit DFlash forward dtype cast, 1 node / 4 GB200 | completed, exit `0:0`; checkpoint created; metrics all zero |
| `3178746` | Qwen3-235B DFlash aligned train-only smoke | 512-row subset selected from the original 50K prepared Arrow data, same hidden-state pool, same DFlash config | completed, exit `0:0`; aligned checkpoint created |
| `3178867` | DFlash checkpoint vLLM container config probe | no target model load; checks current container's Speculators converter against checkpoint `3178746` | completed, exit `0:0`; vLLM `0.17.0` supports `eagle3` only, so DFlash inference is not yet runnable in this container |

The first DFlash smoke attempt, `3178648`, was cancelled after the logs showed
that `MAX_SAMPLES` does not limit `scripts/train.py` when `RUN_PREPARE=false`.
It started training against the full 50K Arrow dataset, which was not the
intended smoke. The next attempt, `3178667`, ran `RUN_CONVERT=true` and
`RUN_PREPARE=true` into a separate
`dflash_openmath_reasoning_cot_smoke512_k5/prepared` directory, proving the
512-row data path, but failed at DFlash forward with a dtype mismatch:
hidden states reached `self.fc` as `float32` while DFlash weights were
`bfloat16`.

Retry `3178693` passed `--noise-std 0`, but it was cancelled after confirming
that the zero-noise change alone did not remove the dtype mismatch. The current
retry, `3178708`, applies both
`speculators_noise_zero_preserve_dtype_compat.patch` and
`speculators_dflash_forward_dtype_compat.patch`. The DFlash forward patch casts
`hidden_states` and `verifier_last_hidden_states` to the DFlash module weight
dtype before the projection path.

`3178708` result:

- Runtime: `00:02:54`, exit `0:0`.
- Checkpoint:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/dflash_openmath_reasoning_cot_smoke512_k5/checkpoints/0`
- Config confirms `speculators_model_type=dflash`, `block_size=6`, and
  `speculative_tokens=5`.
- `val_metrics.json` is all zero:
  `loss_epoch=0.0`, `full_acc_epoch=0.0`, and all position accuracies `0.0`.
- Logs show loaded-token-id versus input-id mismatch warnings. This likely means
  the 512-row prepared data and the reused hidden-state files are not perfectly
  aligned, even though the hidden-state filenames exist.

`3178746` result:

- Runtime: `00:02:51`, exit `0:0`.
- Checkpoint:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/dflash_openmath_reasoning_cot_smoke512_k5_aligned/checkpoints/0`
- Config confirms `speculators_model_type=dflash`, `block_size=6`,
  `speculative_tokens=5`, `num_hidden_layers=2`, `max_anchors=128`,
  `mask_token_id=151670`, and `dtype=bfloat16`.
- The aligned subset was created from the original 50K prepared Arrow data, so
  it matches existing `hs_0..hs_511` hidden-state files instead of mixing a
  newly tokenized 512-row JSONL with old hidden states.
- Validation metrics are nonzero but weak: `loss_epoch=4.457`,
  `full_acc_epoch=0.0326`, and position validation accuracies are about
  `3.0-3.6%`.

Interpretation: DFlash is now runnable for Qwen3-235B at the training-compatibility
level, and the prepared-data / hidden-state alignment issue is fixed for this
smoke. The checkpoint should still be treated as a compatibility checkpoint
rather than a performance checkpoint because validation accuracy is low.

Why this is train-only first: the existing artifact root already has
Speculators `49,996` hidden-state files under
`/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_openmath_reasoning_cot_50k`.
The smoke intentionally sets `RUN_DATAGEN=false` so it does not re-launch the
Qwen3-235B verifier for hidden-state extraction. Because `3178746` now has a
loadable aligned checkpoint with nonzero metrics, the next intended gate was a
vLLM standalone DFlash benchmark on held-out OpenMath `ISL=1024`, `OSL=1024`,
bs32. Probe `3178867` shows this is blocked in the current container before
model load: the vLLM module is `/opt/vllm-build/vllm/vllm/__init__.py`,
version `0.17.0`, and
`SUPPORTED_SPECULATORS_TYPES=['eagle3']`. The checkpoint advertises
`speculators_model_type=dflash` and `DFlashDraftModel`, but vLLM raises:
`Expected one of: {'eagle3': ...}`. Therefore the next step is not a throughput
benchmark yet; it is either a newer vLLM container with DFlash Speculators
runtime support or a local vLLM patch that registers DFlash config/model
loading. Only after that load path works should we attempt NeMo-RL integration.

Backport/runtime audit:

- Official upstream evidence is vLLM PR `#38300`, "add DFlash speculators
  support." That patch adds DFlash Speculators config conversion and tests for
  DFlash inference via the speculators auto-detect path. The PR was merged into
  vLLM `main` as commit `0b790a2`.
- Local `.tmp_vllm_v020` has the DFlash runtime pieces:
  `vllm/v1/spec_decode/dflash.py`,
  `vllm/model_executor/models/qwen3_dflash.py`,
  `vllm/model_executor/models/registry.py` entry `DFlashDraftModel`, and
  `register_speculator("dflash")`.
- The current remote vLLM `0.17.0` site is missing `qwen3_dflash.py` and its
  Speculators `algos.py` registers `eagle3` only. Applying PR `#38300` directly
  is insufficient because that PR assumes a tree where `qwen3_dflash.py` already
  exists.
- Reusable probe scripts were added:
  `experiments/eagle3_qwen3_235b/probe_vllm_dflash_support.py` and
  `experiments/eagle3_qwen3_235b/submit_qwen235b_dflash_vllm_support_probe.sh`.
  The wrapper creates an explicit remote sbatch file to avoid nested heredoc
  quoting failures.
- A DFlash-capable vLLM source-build wrapper was added:
  `experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_dflash.sh`.
  It pins `VLLM_SOURCE_SPEC` to
  `git+https://github.com/vllm-project/vllm.git@0b790a2`, writes to
  `python_site/vllm_dflash_pr38300_0b790a2_cu129_torch28nv_source_py312`, and
  requires the post-build probe to pass `register_speculator("dflash")`,
  `qwen3_dflash`, and `vllm.v1.spec_decode.dflash` imports. Local dry-run
  validation passed. The first submit attempt with `SBATCH_TIME=08:00:00`
  exceeded the partition limit; the corrected `04:00:00` source-build job
  `3179079` is now running.
- A dependent DFlash runtime support probe, `3179087`, was submitted with
  `SBATCH_DEPENDENCY=afterok:3179079`,
  `SOURCE_VLLM_SITE=.../vllm_dflash_pr38300_0b790a2_cu129_torch28nv_source_py312`,
  and `REQUIRE_DFLASH=true`. This will verify the built source site against the
  aligned Qwen3-235B DFlash checkpoint immediately after the source build
  succeeds.
- A dependent vLLM standalone OpenMath gate was also queued behind that runtime
  probe: baseline `3179107` and DFlash K5 `3179108`, both using
  `ISL=1024`, `OSL=1024`, batch sizes `1 2 4 8 16 32`, target `TP=4`, draft
  `TP=4`, and the same DFlash-capable source site. These jobs remain pending
  on `afterok:3179087` and will not run unless the source build and runtime
  support probe both pass.
- A poll helper was added:
  `scripts/poll_qwen235b_dflash_chain_status.sh`. At 20:18 PDT, it showed
  `3179079` still running in the vLLM wheel build step and the support/benchmark
  outputs missing as expected because their jobs are still dependency-gated.
- A result collector/plotter was added:
  `scripts/plot_qwen235b_dflash_openmath_gate.py`. It reads the baseline and
  DFlash `breakdown.json` files, writes
  `docs/qwen3_235b_dflash_openmath_metrics_20260605.csv`, and renders
  `docs/qwen3_235b_dflash_openmath_speedup_acceptance.png` once the dependency
  jobs complete. It currently reports missing result JSONs, which is expected
  until `3179107` / `3179108` run.
- A wrapper retry was attempted, but at 2026-06-05 19:56 PDT the login node
  reported `/lustre/fs1` as `Cannot send after transport endpoint shutdown`.
  `/lustre/fsw/portfolios/coreai/users/sna` is a symlink to `/lustre/fs1`, and
  both the DFlash checkpoint and the current container image were inaccessible.
  The support probe should be retried after Lustre recovers or with a container
  and checkpoint staged outside the broken mount.

The DFlash train submit wrapper is
`experiments/eagle3_qwen3_235b/submit_qwen235b_dflash_smoke_train_only.sh`.
The status files are `latest_qwen235b_dflash_smoke_train_only_jobs.txt`,
`latest_qwen235b_dflash_vllm_config_probe_jobs.txt`, and
`latest_qwen235b_dflash_vllm_support_probe_jobs.txt`. The source-build status
file is `latest_vllm_native_source_build_dflash_job.txt`, currently tracking
build job `3179079`, dependent support probe `3179087`, and dependent OpenMath
gate jobs `3179107` / `3179108`.

Latest runtime update at 20:36 PDT:

- Source-build job `3179079` failed in native vLLM compilation because upstream
  vLLM commit `0b790a2` expects
  `torch/headeronly/util/Float8_e4m3fnuz.h`, while the GB200 NeMo container's
  Torch `2.8.0a0+nv25.05` exposes the FP8 FNUZ type without that header.
- The build wrapper now injects a narrow compatibility include shim for
  `Float8_e4m3fnuz.h` and limits `TORCH_CUDA_ARCH_LIST` to `10.0` for the
  GB200 target. Retry source-build job `3179221` is running and has passed the
  previous header failure point so far; logs show the shim in `CMAKE_ARGS` and
  the wheel build is still running.
- The active DFlash dependency chain is now:
  source-build `3179221` -> runtime probe `3179225` ->
  OpenMath baseline `3179227` and DFlash K5 `3179228`.
- `scripts/poll_qwen235b_dflash_chain_status.sh` now defaults to these new job
  IDs and output paths.
- `scripts/plot_qwen235b_dflash_openmath_gate.py` now also defaults to the new
  `3179227` / `3179228` output paths. Until those dependency jobs run, it
  correctly exits with missing-result JSONs.
- The active PARD/PARD-2-style full-GRPO no-stop validation remains baseline
  `3177855` and local CAT/TPP-mask K5 `3177856`, both `MAX_STEPS=20` and still
  `PENDING (Priority)` at 20:35 PDT. The K5 path is the parallel-drafting
  PARD runtime path; prior K5 full-GRPO launcher logs explicitly showed
  `speculative_config_k5_parallel_drafting_true`.
- At 20:39 PDT, `scontrol` showed both full-GRPO jobs still pending for
  priority, with scheduler candidate `StartTime=2026-06-05T22:46:23`.
  Their stdout logs do not exist yet, as expected while pending. A dedicated
  poll helper was added: `scripts/poll_qwen235b_fullgrpo20_status.sh`.
- At 20:41 PDT, the scheduler candidate times shifted to baseline `3177855`
  at `2026-06-05T21:46:10` and K5 `3177856` at
  `2026-06-05T22:50:28`. Treat these as backfill estimates, not guaranteed
  start times.
- At 20:48 PDT, DFlash source-build retry `3179221` was still running at
  `17:22` elapsed with no failure markers. This passed the previous
  `3179079` failure time (`16:50`), so the FP8 header shim appears to have
  cleared the immediate `Float8_e4m3fnuz.h` compile blocker. The build is not
  yet complete; support probe `3179225` and OpenMath jobs `3179227` /
  `3179228` remain dependency-pending.
- The best completed full-GRPO-related evidence is still the
  stop-after-generation pair `3175807` / `3175808`: `1.718x` generation
  throughput speedup, `1.774x` policy-generate timer speedup, and `53.5%`
  acceptance. This is strong generation-segment evidence, but not yet no-stop
  E2E full-GRPO step-time proof.

Latest runtime update at 20:54 PDT:

- DFlash source-build retry `3179221` failed at `18:29`. The first shim fixed
  `Float8_e4m3fnuz.h`, but the upstream vLLM DFlash tree also includes
  `torch/headeronly/util/Float8_e4m3fn.h`. The build wrapper now injects both
  FP8 header wrappers under `torch/headeronly/util` and forwards to the
  container's `c10/util` headers.
- New DFlash dependency chain:
  source-build `3179391` -> runtime probe `3179395` -> OpenMath baseline
  `3179397` and DFlash K5 `3179398`. At 20:54 PDT, `3179391` was running at
  `1:16` elapsed on `nvl72120-T18`; downstream jobs were dependency-pending.
- `scripts/poll_qwen235b_dflash_chain_status.sh` and
  `scripts/plot_qwen235b_dflash_openmath_gate.py` now default to the new
  `3179391` / `3179395` / `3179397` / `3179398` chain and output paths.
- The no-stop full-GRPO validation pair remains baseline `3177855` and K5
  `3177856`, both still `PENDING (Priority)` at 20:54 PDT with no stdout logs.
  Therefore the only completed strong number is still the stop-after-generation
  generation-segment result, not E2E full-GRPO.
- At 20:58 PDT, `3179391` was still running and the build log confirmed the
  two-header FP8 shim in `CMAKE_ARGS`. The dependent probe and OpenMath jobs
  had not started.
- A fallback audit of the existing vLLM `0.17.0` extracted site found that a
  simple DFlash Python overlay is not enough: the runtime is missing
  `use_dflash`, `DFlashProposer`, the DFlash Triton input-copy kernel,
  `llm_base_proposer.py` DFlash behavior, `gpu_model_runner.py` DFlash dispatch,
  and the `DFlashDraftModel` model/registry entry. A source build or a larger
  backport is required.
- At 21:09 PDT, source-build retry `3179391` failed at `16:16` on
  `torch/headeronly/util/BFloat16.h`, showing that the container is missing the
  broader `torch/headeronly` stable-ABI wrapper tree rather than only FP8
  wrappers.
- The build wrapper now creates minimal compatibility headers for `BFloat16`,
  `Half`, both FP8 types, `Exception`, `shim_utils`, `core/ScalarType`, and
  `core/Dispatch`. New chain: source-build `3179564` -> runtime probe
  `3179565` -> OpenMath baseline `3179567` and DFlash K5 `3179568`.
- At 21:14 PDT, `3179564` was running on `nvl72010-T14` and its log confirmed
  the broader `torch/headeronly` shim in `CMAKE_ARGS`. The no-stop full-GRPO
  pair was still pending; Slurm candidate starts were baseline `3177855` at
  `2026-06-05T23:22:40` PDT and K5 `3177856` at `2026-06-06T01:09:25` PDT.

Latest update at 21:32 PDT:

- The true no-stop full-GRPO pair still has no completed E2E result. Baseline
  `3177855` and K5 local CAT/TPP-mask `3177856` are both `PENDING (Priority)`,
  elapsed `0:00`, with no stdout logs yet. A follow-up 21:33 PDT `scontrol`
  poll showed scheduler candidate starts of `2026-06-05T23:20:00` for
  `3177855` and `2026-06-05T23:40:19` for `3177856`; treat those as backfill
  estimates, not guaranteed starts. The only good completed number remains the
  stop-after-generation generation-segment result, not full-loop GRPO proof.
- Completed generation-segment evidence remains: baseline `3175807` versus K5
  `3175808`, `1.718x` generated-token throughput speedup, `1.774x`
  policy-generate timer speedup, and `53.5%` acceptance.
- DFlash source-build retry `3179564` is still running at `19:08` elapsed on
  `nvl72010-T14`. The log remains in the vLLM wheel-build phase with the
  broader `torch/headeronly` shim enabled. Runtime probe `3179565` and
  OpenMath gate jobs `3179567` / `3179568` remain dependency-pending, so their
  JSON outputs are not expected to exist yet.
- At 21:41 PDT, an overlap process check confirmed that `3179564` is not hung:
  `cmake --build`, `ninja -j16`, `nvcc`, and `cicc` processes are actively
  compiling. It also showed the current build is compiling multiple CUDA arches
  (`sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120`) because the arch env did not
  reach the container `srun env` command. The source-build wrapper has been
  patched for future retries to forward `TORCH_CUDA_ARCH_LIST`, `CMAKE_ARGS`,
  `MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL`, `NVCC_THREADS`, and related build
  variables into the container; the DFlash submit wrapper now defaults
  `CMAKE_ARGS` to include `-DCMAKE_CUDA_ARCHITECTURES=100`. `bash -n` and a
  local dry-run passed.

Latest update at 21:56 PDT:

- DFlash source-build retry4 is active as job `3180764`. This retry fixes two
  issues seen in `3179564`: it forwards the build environment into the
  container and interposes a `torch/all.h` compatibility shim for the unstable
  vLLM `_C` target that uses `torch::headeronly::*` symbols.
- The build log confirms `CMAKE_CUDA_ARCHITECTURES=100` and the compatibility
  include path in `CMAKE_ARGS`. The job is currently in `Building wheel for
  vllm`.
- The queued DFlash standalone chain is: support probe `3180920`
  (`afterok:3180764`), OpenMath baseline `3180951`, OpenMath DFlash K=3
  `3181077`, and OpenMath DFlash K=5 `3180953` (`afterok:3180920`). The sweep
  uses `ISL=1024`, `OSL=1024`, batch sizes `1 2 4 8 16 32`, `TP=4`, and
  `draft_TP=4`.
- The no-stop Full-GRPO validation pair is still pending: baseline `3177855`
  and local CAT/TPP-mask K5 `3177856`. No E2E Full-GRPO metric has completed
  yet.

Latest update at 22:16 PDT:

- The no-stop Full-GRPO validation pair remains queued, not failed. Baseline
  `3177855` is `PENDING (Priority)` with candidate start
  `2026-06-06T00:25:56` PDT; local CAT/TPP-mask K5 `3177856` is
  `PENDING (Priority)` with candidate start `2026-06-06T01:28:49` PDT. These
  are scheduler estimates, and no stdout/ray-driver logs exist yet.
- A same-shape public PARD K5 no-stop Full-GRPO job was added to isolate
  runtime/checkpoint effects from the local PARD-2-style CAT checkpoint:
  `3182758`, `PENDING (Priority)`, candidate start `2026-06-06T01:38:05` PDT.
  It uses the shared baseline `3177855`, `amd/PARD-Qwen3-0.6B`,
  `parallel_drafting=true`, K5, `MAX_STEPS=20`, `stop_after_generation=false`,
  generation `TP=4`, train `TP=2/PP=8/CP=2/EP=16`, GBS `256`, fixed decode
  `256`, `temperature=0`, `top_p=1`, `ignore_eos=true`,
  `gpu_memory_utilization=0.90`, and `enforce_eager=true`.
- Current root-cause split:
  - The prior hard Full-GRPO failure is not a PARD-2 drafter failure. Baseline
    and K5 both hit `PackedSeqParams.__init__() got an unexpected keyword
    argument 'total_tokens'` after generation/logprob. The remote code now
    guards that kwarg with `dataclasses.fields(PackedSeqParams)`, and the
    container smoke passed.
  - The remaining PARD-2-style concern is quality/performance, not a new hard
    error: our runnable CAT/TPP-mask approximation is not official PARD-2. It
    lacks target-hidden-state injection / target-feature gating and has so far
    only lifted current-harness local Qwen3-235B OpenMath to `1.122x` at
    `46.7%` acceptance, while the weighted-CE ablation regressed to `0.869x`.
    The queued public-vs-local no-stop pair is the clean test for whether the
    local CAT checkpoint itself causes a Full-GRPO issue or whether the full
    loop now passes after the packing fix.

Latest update at 22:19 PDT:

- Follow-up poll still shows baseline `3177855`, local CAT/TPP-mask K5
  `3177856`, and public PARD K5 `3182758` all `PENDING (Priority)` with no
  stdout logs. There is still no no-stop Full-GRPO E2E result to report.
- `scripts/extract_qwen235b_tail2_metrics.py` was extended to parse E2E fields
  from completed Full-GRPO logs: latest/mean `E2E (Tokens/sec/gpu)`,
  latest/mean `Generation Worker Group (Tokens/sec/gpu)`, and latest/mean
  `Total step time`. A local 235B log sample parsed correctly, and the updated
  script was copied to the remote SpecDec-RL worktree where
  `python3 -m py_compile` passed. Once the queued jobs start, this parser is
  ready to compute both generation and E2E speedups for throughput and step
  time.
- The DFlash source-build chain is now retry5: `3181912` is running in vLLM
  wheel build; support probe `3181930` and OpenMath baseline / DFlash K3 /
  DFlash K5 jobs `3181932` / `3181956` / `3181937` are dependency-pending.
  No DFlash standalone performance JSON exists yet.

Latest update at 22:25 PDT:

- The fixed no-stop Full-GRPO jobs are still queued, not failed:
  `3177855` baseline, `3177856` local CAT/TPP-mask PARD-2-style K5, and
  `3182758` public PARD K5 are all `PENDING (Priority)` with no stdout logs.
- Added `scripts/summarize_qwen235b_fullgrpo_pard.py`. It reads the three
  matched `ray-driver.log` files and computes:
  generation-worker throughput speedup, diag generation throughput speedup,
  generation step-time speedup, policy-generate step-time speedup, E2E
  throughput speedup, and E2E step-time speedup against baseline `3177855`.
  It is log-tolerant and prints `missing_log` rows while jobs are still
  pending.
- `scripts/poll_qwen235b_fullgrpo20_status.sh` now prints this parsed speedup
  summary at the end of every poll. The updated extractor, summarizer, and poll
  helper were copied to the remote worktree; `python3 -m py_compile` and
  `bash -n` passed.
- DFlash retry5 `3181912` is still running in the vLLM wheel-build phase at
  about `14:32` elapsed; downstream support probe and OpenMath jobs remain
  dependency-pending.

Latest update at 22:46 PDT:

- Code review split the PARD/PARD-2 Full-GRPO risk into runtime-tail issues,
  not a drafter-specific failure. The prior hard crash was shared by baseline
  and K5 and came from Megatron `PackedSeqParams.total_tokens`; the remote
  `data.py` guard is present.
- Claude Code review found two more Full-GRPO greedy safeguards that were
  missing from the Qwen3-235B PARD wrapper but already used in the mature
  Qwen3-8B greedy recipe: allow the omitted-logprob repair path for the
  non-identity greedy sampler, and disable Megatron DDP overlap during
  reference logprob computation. The remote Megatron train path already has
  the `temperature > 0.0` finite-loss guard for `temperature=0`.
- The remote runtime checkout now has:
  - `grpo.py::_repair_specdec_generation_logprobs_if_safe()`
  - `NRL_ALLOW_SPECDEC_LOGPROB_REPAIR_WITH_SAMPLER_MISMATCH=true`
  - `policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false`
  - `policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false`
- The pre-patch pending jobs were cancelled:
  `3177855`, `3177856`, `3182758`, `3185571`, `3185572`, `3185573`.
- Corrected r3 Full-GRPO no-stop jobs were submitted:
  baseline `3185585`, local CAT/TPP-mask PARD-2-style K5 `3185586`, and
  public PARD K5 `3185587`. All are currently `PENDING (Priority)` with no
  stdout/ray-driver logs yet. Scheduler estimates were baseline start
  `2026-06-06T01:40:00` PDT and K5/public start `2026-06-06T02:28:00` PDT.
- These runs are fixed-decode greedy throughput/step-time diagnostics
  (`temperature=0`, `top_p=1`, `ignore_eos=true`, OSL `256`, `MAX_STEPS=20`).
  They should not be interpreted as learning-quality GRPO runs because greedy
  fixed decode can make per-prompt generations identical and collapse GRPO
  advantages toward zero.

Latest update at 22:55 PDT:

- PARD/PARD-2 Full-GRPO r3 is still waiting for allocation:
  `3185585` baseline, `3185586` local CAT/TPP-mask PARD-2-style K5, and
  `3185587` public PARD K5 are all `PENDING (Priority)` with no logs. No new
  no-stop E2E speedup can be claimed yet.
- DFlash runtime validation advanced from a build failure to a runtime import
  compatibility failure. Retry5 `3181912` built the vLLM wheel but failed the
  support probe because the NeMo container/vLLM source build lacked
  `_C.cutlass_scaled_mm_supports_fp8` and logged a Triton kernel import
  mismatch. This is a runtime compatibility blocker, not a DFlash training
  failure.
- DFlash retry6 is now active with a source patch that makes missing CUTLASS
  support-query ops return `False` rather than aborting Qwen3-MoE import:
  build `3185614`, support probe `3185615`, OpenMath baseline `3185616`,
  DFlash K3 `3185618`, and DFlash K5 `3185621`.

Latest update at 23:00 PDT:

- DFlash retry6 `3185614` failed before wheel build because the source patch
  tried to patch a group-GEMM support function that already had an
  `AttributeError` fallback in this vLLM commit. This was a bad patch anchor,
  not a new DFlash model/runtime failure.
- DFlash retry7 was submitted with only the needed FP8/block-FP8 missing-op
  fallbacks: build `3185715`, support probe `3185716`, OpenMath baseline
  `3185717`, DFlash K3 `3185718`, and DFlash K5 `3185724`.

Latest update at 23:02 PDT:

- No new PARD/PARD-2 Full-GRPO E2E result exists yet. Jobs `3185585`,
  `3185586`, and `3185587` are still `PENDING (Priority)` and have no logs.

Latest update at 23:10 PDT:

- Independent code-review agent agreed that the current r3 jobs are not failing;
  they are still blocked only by Slurm `Priority` on a 32-node / 128-GPU
  request. The current scheduler candidate start is
  `2026-06-06T03:07:28` PDT for all three jobs.
- Terminology correction: the local arm should be reported as
  `local CAT/TPP-mask PARD-2-style K5`, not official PARD-2. The public PARD
  control remains `3185587` with `amd/PARD-Qwen3-0.6B`.
- The local and remote submit wrapper
  `experiments/eagle3_online/submit_qwen235b_pard_local_tpp_mask_gbs256_worker32_step1.sh`
  was tightened so its defaults match the corrected r3 diagnostic shape:
  fixed decode `256`, `GENERATION_TP=4`, `VLLM_ENFORCE_EAGER=true`,
  `gpu_memory_utilization=0.90`, `MAX_STEPS=20`,
  `STOP_AFTER_GENERATION=false`, DDP overlap disabled, and greedy SpecDec
  logprob repair opt-in enabled.
- Added and ran
  `experiments/eagle3_online/preflight_qwen235b_pard_fullgrpo.sh` on the remote
  runtime checkout. It passed all checks: `PackedSeqParams.total_tokens` guard,
  SpecDec omitted-logprob repair, greedy sampler-mismatch opt-in, Megatron
  `temperature > 0.0` guard, fixed256/TP4 wrapper defaults, DDP overlap off,
  `draft_model`, and `parallel_drafting=true`. `py_compile` also passed for the
  patched runtime files.
- Manifest cleanup: the public PARD job key is now `PUBLIC_PARD_K5=3185587`
  instead of overwriting `K5_LOCAL_TPP_MASK`.

Latest update at 23:14 PDT:

- Claude Code read-only review agreed with the main risk split and added one
  important semantic point: the r3 no-stop jobs are still `temperature=0`,
  fixed-256-token diagnostics. They can measure stability and fixed-work
  throughput/E2E step time, but they are not a learning-quality GRPO claim
  because greedy generations can collapse per-prompt reward variance.
- The submit wrapper was extended to support a separate sampling path:
  `FIXED_DECODE=false`, `GENERATION_TEMPERATURE=1.0`, natural EOS
  (`MIN_TOKENS=` and no forced `ignore_eos`), while preserving the fixed-decode
  r3 defaults for the throughput diagnostic. K5 now explicitly sets
  `DRAFT_TP=${GENERATION_TP}` instead of relying on vLLM defaults.
- A real-sampling no-stop smoke was submitted with `MAX_STEPS=2`,
  `max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`,
  generation `TP=4`, and `gpu_memory_utilization=0.90`:
  - baseline `3186018`
  - local CAT/TPP-mask PARD-2-style K5 `3186020`
  - public PARD K5 `3186021`
- These sampling smoke jobs are also currently `PENDING (Priority)`. They are
  the right jobs to answer "does PARD/PARD-2-style run through actual GRPO
  reward/logprob/training without fixed-decode artifacts"; the earlier r3 jobs
  remain useful for fixed-work timing.
- DFlash retry7 build `3185715` is running and has reached the dependency
  install/source-clone phase without a new failure marker. Probe `3185716` and
  OpenMath standalone jobs `3185717` / `3185718` / `3185724` are waiting on the
  build/probe dependency chain.

Latest update at 23:16 PDT:

- Fixed-decode r3 jobs are still queued, not failed. `3185585`, `3185586`, and
  `3185587` are all `PENDING (Priority)` with no stdout/ray-driver logs.
  Scheduler candidate start moved to `2026-06-06T01:54:51` PDT for all three.
- Real-sampling smoke jobs are also queued with no logs yet:
  baseline `3186018` candidate start `2026-06-06T01:54:51` PDT,
  local CAT/TPP-mask PARD-2-style K5 `3186020` candidate start
  `2026-06-06T02:00:00` PDT, and public PARD K5 `3186021` candidate start
  `2026-06-06T02:40:00` PDT.
- DFlash retry7 build `3185715` is still running at `00:16:29` elapsed in
  vLLM wheel build. The support probe and OpenMath standalone DFlash jobs remain
  dependency-pending, so there is no DFlash speedup number yet.

Latest update at 23:59 PDT:

- Corrected sync `VllmGeneration` smoke results are now available for baseline
  and local CAT/TPP-mask PARD-2-style K5. The stale-JSON poll issue was fixed,
  so summaries now read the current job's `--output-json` path only.
- Baseline `3186339` passed: `8192` generated tokens in `31.8386s`,
  `257.30 tok/s`, no active SpecDec counters.
- Local CAT/TPP-mask PARD-2-style K5 `3186340` passed:
  `8192` generated tokens in `21.1956s`, `386.50 tok/s`, `1.502x`
  generation throughput speedup, `43.77%` aggregate acceptance,
  `3.19` mean acceptance length. Per-position acceptance decays from
  `73.49%` at position 1 to `22.54%` at position 5, so K=5 works but the
  later draft positions still limit the ceiling.
- Public PARD K5 retry `3186417` later completed successfully. It generated
  `8192` tokens in `20.9250s`, `391.49 tok/s`, `1.522x` generation throughput
  speedup, `42.29%` aggregate acceptance, and `3.11` mean acceptance length.
  This is slightly faster than the local CAT/TPP-mask PARD-2-style checkpoint
  on the same sync fixed-256 smoke (`391.49` vs `386.50 tok/s`, about `1.013x`
  public-over-local). The earlier public job `3186341` failed before the
  driver due to a Slurm `srun` socket timeout, not a PARD runtime failure.
- True no-stop Full-GRPO fixed-decode jobs remain pending:
  baseline `3186342`, local `3186343`, public `3186344`. Real-sampling
  Full-GRPO smoke also remains pending: baseline `3186345`, local `3186354`,
  public `3186355`.

Latest update at 00:10 PDT on 2026-06-06:

- Focus shifted back to the PARD-2-style path first. DFlash remains a secondary
  branch and is not needed to answer the immediate Full-GRPO question.
- Submitted a PARD-2-style-only real-sampling Full-GRPO Step-4 pair, excluding
  public PARD to keep the comparison focused:
  - baseline `3186510`
  - local CAT/TPP-mask PARD-2-style K5 `3186511`
- Shape: Qwen3-235B-A22B, 32 nodes x 4 GB200, `TRAIN_GLOBAL_BATCH_SIZE=256`,
  generation `TP=4`, training Megatron `TP=2`, `PP=8`, `CP=2`, `EP=16`,
  `MAX_STEPS=4`, `STOP_AFTER_GENERATION=false`, `max_new_tokens=256`,
  natural EOS, `temperature=1.0`, `top_p=1.0`, `top_k=-1`,
  `VLLM_ENFORCE_EAGER=true`, `gpu_memory_utilization=0.90`,
  `policy.draft.enabled=false`, `draft_model`,
  `parallel_drafting=true`, and `DRAFT_TP=4` for the K5 arm.
- Current state: both jobs are `PENDING (Priority)` with no logs yet. Slurm
  shows candidate start `2026-06-06T03:57:56` PDT.
- Local manifest:
  `latest_qwen235b_pard2_nemorl_fullgrpo_sampling_step4_jobs.txt`.
- Poll helper:
  `scripts/poll_qwen235b_pard2_step4_status.sh`.

Latest update at 00:12 PDT on 2026-06-06:

- Step-4 real-sampling Full-GRPO PARD-2-style pair remains queued:
  baseline `3186510`, local CAT/TPP-mask K5 `3186511`, both
  `PENDING (Priority)` with no stdout logs yet. Slurm still reports candidate
  start `2026-06-06T03:57:56` PDT for both jobs.
- Fixed-256 no-stop Full-GRPO comparison remains queued as well:
  baseline `3186342`, local `3186343`, public `3186344`. Baseline candidate
  start moved earlier to `2026-06-06T02:23:57` PDT; local/public still show
  `2026-06-06T03:57:56` PDT.
- The completed sync smoke now has all three rows in
  `docs/qwen3_235b_nemorl_sync_vllmgeneration_smoke_20260605.csv`.

Latest update at 00:16 PDT on 2026-06-06:

- Step-4 real-sampling Full-GRPO PARD-2-style pair is still queued with no
  logs:
  - baseline `3186510`: `PENDING (Priority)`, candidate start
    `2026-06-06T04:07:58` PDT
  - local CAT/TPP-mask PARD-2-style K5 `3186511`: `PENDING (Priority)`,
    candidate start `2026-06-06T04:07:58` PDT
- Fixed-256 20-step diagnostic is also still queued:
  - baseline `3186342`: `PENDING (Priority)`, candidate start
    `2026-06-06T02:25:18` PDT
  - local CAT/TPP-mask PARD-2-style K5 `3186343`: `PENDING (Priority)`,
    candidate start `2026-06-06T04:07:58` PDT
  - public PARD K5 `3186344`: `PENDING (Priority)`, candidate start
    `2026-06-06T04:07:58` PDT
- The real-sampling step-2 smoke jobs `3186345` / `3186354` / `3186355` are
  also still `PENDING (Priority)`.
- No smaller fallback has been submitted. For Qwen3-235B Full-GRPO, preserving
  the 32-node training shape is more important than getting a weaker
  non-comparable run quickly; the 1-node sync generation smoke already proves
  the PARD runtime/generation path.

Latest update at 00:17 PDT on 2026-06-06:

- No Full-GRPO logs exist yet; all relevant jobs remain `PENDING (Priority)`.
- Step-4 real-sampling PARD-2-style pair:
  - baseline `3186510`: candidate start moved to `2026-06-06T03:36:18` PDT
  - local CAT/TPP-mask PARD-2-style K5 `3186511`: candidate start moved to
    `2026-06-06T03:36:18` PDT
- Fixed-256 20-step diagnostic:
  - baseline `3186342`: candidate start `2026-06-06T02:26:43` PDT
  - local CAT/TPP-mask PARD-2-style K5 `3186343`: candidate start
    `2026-06-06T03:36:18` PDT
  - public PARD K5 `3186344`: candidate start `2026-06-06T03:36:18` PDT
- Real-sampling step-2 smoke `3186345` / `3186354` / `3186355` is also
  `PENDING (Priority)` with candidate start `2026-06-06T03:36:18` PDT.

Latest update at 00:31 PDT on 2026-06-06:

- The "no smaller fallback has been submitted" note above is now superseded.
  To get a faster Full-GRPO signal while the 32-node Qwen3-235B jobs wait for
  allocation, a smaller-model PARD/PARD-2-style Full-GRPO 5-step matrix was
  submitted on 4 nodes x 4 GB200:
  - Qwen3-32B baseline `3186660`
  - Qwen3-32B public PARD K5 `3186661`
  - Qwen3-30B-A3B baseline `3186662`
  - Qwen3-30B-A3B public PARD K5 `3186663`
  - Qwen3-30B-A3B local CAT/PARD-2-style K5 `3186664`
- Shape for all five jobs: `MAX_STEPS=5`, `STOP_AFTER_GENERATION=false`,
  `max_new_tokens=256`, natural EOS, `temperature=1.0`, `top_p=1.0`,
  `top_k=-1`, always-on SpecDec for the PARD arms, `policy.draft.enabled=false`,
  `method=draft_model`, `parallel_drafting=true`, `num_speculative_tokens=5`.
- Qwen3-32B uses `amd/PARD-Qwen3-0.6B` as a runnable PARD-style control because
  there is no local Qwen3-32B CAT/PARD-2-style checkpoint in the current
  artifacts. Qwen3-30B-A3B uses both public PARD and the local checkpoint:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_30ba3b_pard2_cat_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen30ba3b_math_k5_cat_tpp_mask_1024_resume_20260605_173358/checkpoint-32`.
- Current state: all five jobs are `RUNNING` and have reached
  `All workers connected`. They are still in driver venv/dependency setup, so
  no rollout, generation throughput, E2E throughput, or acceptance metric has
  been emitted yet. The transient `Traceback` marker observed in `3186663` came
  from a Ray readiness probe and was followed by `All workers connected`; it is
  not currently evidence of a failed PARD run.
- Local manifest:
  `latest_qwen32_qwen30ba3b_pard2style_fullgrpo5_jobs.txt`.
- Poll helper:
  `scripts/poll_qwen32_qwen30ba3b_pard2style_fullgrpo5_status.sh`.
- Qwen3-235B state at the same poll:
  - fixed-256 20-step jobs `3186342` / `3186343` / `3186344` are still
    `PENDING (Priority)` with candidate start around `2026-06-06T02:40:20` PDT.
  - real-sampling step-4 pair `3186510` / `3186511` is still
    `PENDING (Priority)` with candidate start `2026-06-06T04:19:57` PDT.
  - real-sampling step-2 smoke `3186345` / `3186354` / `3186355` is still
    `PENDING (Priority)` with candidate start `2026-06-06T04:19:57` PDT.

Latest update at 00:35 PDT on 2026-06-06:

- The smaller-model matrix remains alive and running; no rollout metric has
  appeared yet.
- Driver logs show this is still dependency preparation, not a model/runtime
  failure. `3186661` and `3186662` have reached `Installed 272 packages`;
  the other three are still in the same download/build/install path.
- Therefore the current blocker for small-model evidence is startup latency in
  fresh driver venv creation, not PARD/PARD-2 correctness. Do not interpret the
  lack of acceptance/throughput metrics yet as a failed PARD result.

Latest update at 00:57 PDT on 2026-06-06:

- The first small-model Full-GRPO matrix `3186660`-`3186664` is now confirmed
  failed before any rollout metric. Root cause was not PARD/PARD-2 runtime:
  `ray.init()` failed because the Ray cluster started with Ray `2.49.2` /
  Python `3.12.13`, while the driver venv used Ray `2.54.0` / Python
  `3.13.13`.
- The small-model submit wrapper was patched to set
  `RAY_VERSION=2.54.0`, `RAY_PYTHON_VERSION=3.13.13`, and
  `RAY_PYTHON_SPEC=3.13.13` for both the Ray cluster and driver path.
- The corrected raymatch matrix was submitted:
  - Qwen3-32B baseline `3186983`
  - Qwen3-32B public PARD K5 `3186984`
  - Qwen3-30B-A3B baseline `3186985`
  - Qwen3-30B-A3B public PARD K5 `3186986`
  - Qwen3-30B-A3B local CAT/PARD-2-style K5 `3186987`
- All five raymatch jobs are `RUNNING` on 4 nodes x 4 GB200. At the latest
  poll, no `Version mismatch` marker is present and no driver log has been
  created yet; the jobs are still in Ray/container worker startup. No rollout,
  generation throughput, E2E throughput, or acceptance metric has been emitted.
- Qwen3-235B Full-GRPO PARD/PARD-2 jobs are still waiting for 32-node
  allocation:
  - fixed-256 20-step jobs `3186342` / `3186343` / `3186344` are
    `PENDING (Priority)`.
  - real-sampling step-2 smoke `3186345` / `3186354` / `3186355` is
    `PENDING (Priority)`.
  - real-sampling step-4 pair `3186510` / `3186511` is
    `PENDING (Priority)`.
- Qwen3-32B and Qwen3-30B-A3B vLLM standalone EAGLE3 data has been consolidated
  in `docs/qwen32_qwen30ba3b_vllm_standalone_eagle3_metrics_20260606.csv` and
  plotted in
  `docs/qwen32_qwen30ba3b_vllm_standalone_eagle3_speedup_acceptance.png`.
  Standalone ceilings:
  - Qwen3-32B: K1 bs32 `1.625x` at `79.9%` acceptance; K2 bs32 `2.043x` at
    `72.9%`; K3 bs32 `2.288x` at `67.1%` and best K3 `2.744x` at bs1.
  - Qwen3-30B-A3B Thinking-2507: K1 bs32 `1.450x` at `90.3%`; K2 bs32
    `1.834x` at `86.6%`; K3 bs32 `1.861x` at `81.2%` and best K3 `1.924x`
    at bs16.

Latest update at 01:01 PDT on 2026-06-06:

- The raymatch small-model Full-GRPO matrix remains `RUNNING`.
- All five jobs reached `All workers connected`, and driver logs have been
  created:
  - `3186983-logs/ray-driver.log`
  - `3186984-logs/ray-driver.log`
  - `3186985-logs/ray-driver.log`
  - `3186986-logs/ray-driver.log`
  - `3186987-logs/ray-driver.log`
- None of the five driver logs contains `Version mismatch`. The Ray/Python
  correction is therefore working so far.
- The current stage is driver venv dependency installation/build, visible as
  package downloads plus local builds for `megatron-bridge`, `nemo-rl`,
  `megatron-core`, and `transferqueue`. No `SETUP COMPLETE`, rollout metric,
  generation throughput, E2E throughput, or acceptance metric has appeared yet.
- Added a dedicated poll helper:
  `scripts/poll_qwen32_qwen30ba3b_pard2style_fullgrpo5_raymatch_status.sh`.

Latest update at 01:09 PDT on 2026-06-06:

- The raymatch small-model Full-GRPO matrix is still `RUNNING`:
  - Qwen3-32B baseline `3186983`
  - Qwen3-32B public PARD K5 `3186984`
  - Qwen3-30B-A3B baseline `3186985`
  - Qwen3-30B-A3B public PARD K5 `3186986`
  - Qwen3-30B-A3B local CAT/PARD-2-style K5 `3186987`
- The prior Ray/Python version mismatch has not reappeared. All jobs have
  created driver logs, loaded the NeMo-RL config, initialized W&B/TensorBoard,
  and installed the vLLM worker environment packages.
- The active configuration for the smaller-model run is full GRPO, not
  stop-after-generation: 4 nodes x 4 GB200, `max_num_steps=5`,
  `stop_after_generation=false`, `GBS=2048`, `num_prompts_per_step=64`,
  `num_generations_per_prompt=32`, `generation_batch_size=32`,
  `max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`, natural EOS,
  always-on `draft_model` speculative decoding for the PARD/PARD-2 arms,
  `parallel_drafting=true`, and `num_speculative_tokens=5`.
- No `SETUP COMPLETE`, rollout metric, generation throughput, E2E throughput,
  or acceptance metric has appeared yet. The jobs are therefore alive, but not
  far enough to claim Full-GRPO performance.
- Qwen3-235B Full-GRPO PARD/PARD-2 jobs remain `PENDING (Priority)` for the
  32-node allocation: `3186342` / `3186343` / `3186344`, `3186345` /
  `3186354` / `3186355`, and `3186510` / `3186511`.

Latest update at 01:17 PDT on 2026-06-06:

- Naming clarification: the official public artifact currently in use is PARD,
  specifically `amd/PARD-Qwen3-0.6B`.
- The upstream AMD-AGI/PARD repository has the PARD-2 paper and README section,
  but its update note says the PARD-2 code and model checkpoints will be
  released soon. Therefore, there is no official public PARD-2 checkpoint in
  our current runs.
- What we have been labeling `PARD-2-style` is our local experimental drafter
  trained with CAT/TPP-mask-style target-alignment ideas, not an upstream
  released PARD-2 artifact. In result tables and plots this should be labeled
  as `local CAT/PARD-2-style`, not simply `PARD-2`.
- Current smaller-model Full-GRPO coverage:
  - Qwen3-32B: baseline plus public PARD K5.
  - Qwen3-30B-A3B: baseline, public PARD K5, plus local CAT/PARD-2-style K5.
  - All five jobs are still `RUNNING`; vLLM policy workers have initialized,
    and the PARD/PARD-2-style arms have entered safetensors checkpoint loading.
    No rollout throughput or acceptance metrics have appeared yet.
- Current Qwen3-235B Full-GRPO coverage remains pending: baseline, public PARD
  K5, and local CAT/PARD-2-style K5 jobs are queued for the 32-node allocation.

Latest update at 01:28 PDT on 2026-06-06:

- Qwen3-235B OpenMath standalone high-batch PARD jobs were submitted for
  `ISL=1024`, `OSL=1024`, batch sizes `64` and `128`, target `TP=4`, draft
  `TP=4`, K5:
  - baseline `3187309`
  - public PARD K5 `3187310`
  - local CAT/PARD-2-style K5 `3187312`
- Current Qwen3-235B standalone state:
  - baseline `3187309` is `RUNNING` and still loading the 235B checkpoint.
  - public PARD `3187310` is `RUNNING` and has initialized the vLLM engine,
    with `speculative_config=SpeculativeConfig(method='draft_model',
    model='amd/PARD-Qwen3-0.6B', num_spec_tokens=5)`.
  - local CAT/PARD-2-style `3187312` was requeued once with `ExitCode=0:54`
    and is currently `PENDING`. Slurm reports `Requeue=1`, `Restarts=1`; no
    benchmark log was emitted before the requeue. Treat this as queued/retry,
    not as a performance result.
- Smaller-model Full-GRPO matrix state:
  - Qwen3-32B baseline `3186983` failed before rollout metrics during
    Megatron policy worker initialization. The primary error is
    `TypeError: cannot pickle code objects` from
    `torch.distributed.distributed_c10d._object_to_tensor()` during
    distributed checkpoint loading. This is not a PARD runtime result.
  - Qwen3-32B public PARD K5 `3186984` failed at Step 1 before throughput or
    acceptance metrics. The visible worker error is
    `ValueError: invalid literal for int() with base 10: ''` while Ray imports
    `ray.dag.context` and parses `RAY_CGRAPH_get_timeout`. vLLM also dumped
    the scheduler input around the same failure. This is currently classified
    as a Ray env/config failure, not as evidence that PARD K5 is slow.
  - Qwen3-30B-A3B baseline `3186985`, public PARD `3186986`, and local
    CAT/PARD-2-style `3186987` remain `RUNNING` after about 33 minutes. No
    rollout throughput, E2E throughput, or acceptance metric has appeared yet.
- Fix applied for future NeMo-RL submissions:
  - `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now
    defaults `RAY_CGRAPH_GET_TIMEOUT` to `7200` instead of the empty string.
  - `experiments/eagle3_online/submit_qwen32_qwen30ba3b_pard2style_fullgrpo5.sh`
    now explicitly passes `RAY_CGRAPH_GET_TIMEOUT=7200` for both Qwen3-32B and
    Qwen3-30B-A3B arms.
  - The patched scripts were copied to the remote NeMo-RL checkout. Existing
    running jobs keep their old environment; any retry submitted from this
    point uses the non-empty Ray timeout.

Latest update at 01:33 PDT on 2026-06-06:

- The old-env Qwen3-30B-A3B public PARD K5 job `3186986` is now confirmed
  `FAILED` after `00:35:49`. The observed error is the same Ray env parse
  failure: `ValueError: invalid literal for int() with base 10: ''` from
  `RAY_CGRAPH_get_timeout`. This job emitted no usable rollout throughput,
  E2E throughput, or acceptance metrics and must not be interpreted as a
  PARD/PARD-2 performance result.
- The old-env Qwen3-30B-A3B baseline `3186985` and local CAT/PARD-2-style
  `3186987` are still `RUNNING`, but they also started before the Ray timeout
  patch. Keep them as opportunistic runs only.
- A patched clean retry set was submitted with
  `RAY_CGRAPH_GET_TIMEOUT=7200` and run label
  `sampling-temp1-eos-k5-fullgrpo5-r3-cgraphfix`:
  - Qwen3-32B public PARD K5 `3187428`
  - Qwen3-30B-A3B baseline `3187431`
  - Qwen3-30B-A3B public PARD K5 `3187432`
  - Qwen3-30B-A3B local CAT/PARD-2-style K5 `3187433`
- Qwen3-32B baseline was intentionally not included in this retry because the
  previous Qwen3-32B baseline failure was a separate Megatron checkpoint-load
  issue: `TypeError: cannot pickle code objects`.
- Local manifest for the clean retry:
  `latest_qwen32_qwen30ba3b_pard2style_fullgrpo5_cgraphfix_jobs.txt`.

Latest update at 02:05 PDT on 2026-06-06:

- Qwen3-235B OpenMath standalone high-batch K5 completed for baseline,
  public PARD, and local CAT/PARD-2-style. Shape: `ISL=1024`, `OSL=1024`,
  batch sizes `64` and `128`, target `TP=4`, draft `TP=4`,
  `max_num_seqs=128`, `max_num_batched_tokens=393216`, vLLM standalone
  `draft_model`, `parallel_drafting=true`.
- Raw summary was captured in
  `docs/qwen3_235b_pard_openmath_bs64_128_k5_metrics_20260606.csv`.

| Case | bs | Output tok/s | Tok/s/GPU | Throughput speedup | Acceptance | Mean acceptance length |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 64 | `3226.65` | `806.66` | `1.000x` | n/a | n/a |
| public PARD K5 | 64 | `4234.42` | `1058.60` | `1.312x` | `44.59%` | `3.23` |
| local CAT/PARD-2-style K5 | 64 | `4243.53` | `1060.88` | `1.315x` | `45.07%` | `3.25` |
| baseline | 128 | `5638.61` | `1409.65` | `1.000x` | n/a | n/a |
| public PARD K5 | 128 | `6724.30` | `1681.08` | `1.193x` | `44.69%` | `3.23` |
| local CAT/PARD-2-style K5 | 128 | `6695.38` | `1673.85` | `1.187x` | `44.24%` | `3.21` |

Interpretation:

- The current high-batch OpenMath result is no longer "no standalone benefit":
  K5 public PARD/local CAT both produce a measurable standalone throughput
  gain. The gain is strongest at bs64, about `1.31x`, and falls to about
  `1.19x` at bs128.
- Public PARD and local CAT/PARD-2-style are nearly tied here. The local CAT
  checkpoint is not yet a quality improvement over public PARD; it only proves
  the local PARD-2-style training/runtime path remains compatible at high batch.
- Acceptance remains low for a K5 drafter, about `44-45%`, with mean acceptance
  length about `3.2`. This explains why the result is far below the synthetic
  short-prompt PARD ceiling and why simply increasing batch size is not enough
  to get a 2x+ OpenMath result.

Latest update at 02:10 PDT on 2026-06-06:

- The old-env Qwen3-30B-A3B baseline `3186985` and local CAT/PARD-2-style
  `3186987` are now confirmed failed at first rollout weight update. Both
  reached `SETUP COMPLETE`, `Step 1/5`, and `Generating responses for batch
  of size 2048`, then vLLM failed inside
  `VllmInternalWorkerExtension.update_weights_via_ipc_zmq`.
- The primary error in both jobs is:
  `ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor
  (expected 1 or 2)`, raised by vLLM's Qwen3-MoE fused-MoE loader while
  loading `_load_w13`.
- This is not a PARD/PARD-2 performance failure. It happened in the baseline
  job as well as the local CAT job, before any usable throughput/acceptance
  metric could be emitted.
- Root-cause direction from vLLM `v0.20.2` source: for 3D expert tensors,
  vLLM assumes dim 0 is the expert axis and shifts `shard_dim += 1` before
  loading. Seeing `shard_dim=0` at `_load_w13` means the incoming NeMo/Megatron
  export is not being interpreted as the expected per-expert HF/vLLM weight
  mapping. Most likely the Qwen3-MoE grouped expert tensor is being streamed
  without the Megatron-Bridge EP gather/per-expert split patch.
- Local evidence: `remote_nemo_main_patch/megatron_bridge/models/conversion/param_mapping.online.py`
  already contains a `gather_from_ep_ranks` branch that detects grouped expert
  tensors with leading local-expert dimension and returns per-expert 2D HF
  names. That is exactly the missing behavior implied by the vLLM loader
  error. The next actionable step is to make sure this Megatron-Bridge patch
  is actually applied to the remote NeMo-RL runtime/worker venv before
  resubmitting Qwen3-30B-A3B and Qwen3-235B no-stop Full-GRPO.
- Clean retry jobs with the Ray cgraph timeout fix are still running, but at
  the latest poll they were still in dependency preparation. If they reach the
  same MoE refit point before the bridge patch is applied, they are expected
  to fail in the same way.

Latest update at 02:18 PDT on 2026-06-06:

- The remote NeMo-RL `vllm_backend.py` was patched with a diagnostic-only
  MoE refit dump. On policy weight load failure it now prints:
  incoming policy MoE weight names/shapes/dtypes and vLLM MoE parameter
  names/shapes plus loader attributes. This does not change the normal
  load path and only emits on exception.
- Remote syntax check passed:
  `python3 -m py_compile nemo_rl/models/generation/vllm/vllm_backend.py`.
- The diagnostic patch is also saved locally at
  `experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_backend.py`.
- A fresh Qwen3-30B-A3B baseline-only diagnostic job was submitted so the
  patch is definitely loaded from a fresh actor suffix:
  - `3187658`: Qwen3-30B-A3B baseline, no speculative decoding,
    `MAX_STEPS=1`, full GRPO no-stop, `GBS=2048`, generation `TP=1`,
    training `TP=1`, expert parallel `16`.
- Local manifest:
  `latest_qwen30ba3b_moe_refit_diag_jobs.txt`.

Latest update at 02:28 PDT on 2026-06-06:

- A concrete candidate fix for the Qwen3-MoE refit failure was identified and
  applied to the remote Megatron-Bridge source. In
  `GroupedGatedMLPMapping.megatron_to_hf`, grouped MoE tensors should split
  gate/up on the expert-data dimension, not the leading expert dimension.
- The old code split all `linear_fc1` tensors with `torch.chunk(..., dim=0)`.
  That is correct for dense or per-expert 2D tensors, but not for grouped MoE
  tensors shaped like `[num_local_experts, 2 * ffn_hidden, hidden]`. For those,
  splitting on dim 0 corrupts the local expert stack and can leave vLLM seeing
  a 3D w13 tensor with `shard_dim=0`.
- The patched code uses:
  `gate_up_dim = 1 if self.is_expert and megatron_weights.ndim >= 3 else 0`,
  then splits/concats gate/up along `gate_up_dim`.
- Local patch source:
  `remote_nemo_main_patch/megatron_bridge/models/conversion/param_mapping.online.py`.
- Remote patched file:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py`.
- Remote syntax check passed:
  `python3 -m py_compile 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py`.
- Validation is pending. Job `3187658` was still `PENDING (Priority)` at the
  latest poll, so it should start with both the vLLM diagnostic patch and this
  Megatron-Bridge split-dimension fix.

Latest update at 02:35 PDT on 2026-06-06:

- Independent code-review sidecar confirmed the same root cause: this is a
  Megatron-Bridge Qwen3-MoE export/layout bug, not a PARD runtime bug and not
  IPC corruption. The trigger pattern is a grouped expert `linear_fc1`/`w13`
  tensor with leading local-expert axis, e.g.
  `[num_local_experts, 2 * moe_ffn_hidden, hidden]`, being streamed under a
  per-expert HF/vLLM key such as
  `model.layers.<L>.mlp.experts.<E>.gate_proj.weight`.
- The sidecar also noted that the local old-stack compatibility plugin imported
  mapping classes from `megatron.bridge.models.param_mapping`, while the
  current remote Megatron-Bridge source uses
  `megatron.bridge.models.conversion.param_mapping`.
- The local compatibility plugin was patched to support both import paths:
  `experiments/eagle3_qwen3_235b/megatron_bridge_qwen3moe/qwen3_moe_bridge_plugin.py`.
  In the new conversion path, `TPAwareMapping` is absent, so the legacy grouped
  linear registration is skipped. Syntax check passed.

Latest continuation update after the 02:35 PDT section on 2026-06-06:

- Qwen3-32B public PARD K5 clean retry `3187428` reached `SETUP COMPLETE`,
  entered `Step 1/5`, and completed the first generation segment for a batch
  of size `2048`. This confirms the Ray cgraph timeout fix and the PARD
  runtime path are usable through NeMo-RL generation for the dense 32B model.
- The same job emitted vLLM SpecDecoding metrics during generation. The
  repeated per-engine lines show mean acceptance length roughly `3.4-4.1` and
  average draft acceptance rates roughly `49-61%` for K5. Treat this as
  generation telemetry, not an E2E full-GRPO result.
- `3187428` then failed during `policy.get_logprobs()` before optimizer update:
  TransformerEngine RMSNorm raised
  `RuntimeError: ... rmsnorm_fwd_cuda_kernel.cu:97 ... CUDA Error: invalid argument`.
  This is a Megatron/TransformerEngine logprob-forward issue, not a vLLM
  speculative decoding failure. Do not use `3187428` as E2E speedup evidence.
- Qwen3-30B-A3B jobs `3187431` baseline, `3187432` public PARD K5, and
  `3187433` local CAT/PARD-2-style K5 are still running and remain in
  vLLM/MoE initialization at the latest poll. They have not yet reached the
  previous failing weight-refit point, so the grouped-MoE split-dimension fix
  is still unverified.
- Qwen3-30B-A3B diagnostic job `3187658` is also running with the fresh
  `vllm_backend.py` MoE refit dump and Megatron-Bridge split-dim patch. The
  success gate is still: reach first rollout weight update without
  `shard_dim=0`; if it fails, inspect `moe-refit-debug` output before any
  further Qwen3-MoE full-GRPO resubmission.

PARD-2-style training sample/time guidance:

- Public PARD-2 code/checkpoints are still not part of the current runnable
  stack, so local "PARD-2-style" means CAT/acceptance-length-oriented training
  over the PARD parallel drafting runtime.
- Smoke/integration checks only need `128-1K` samples. A meaningful domain
  trend needs at least `10K`; an actionable OpenMath/GRPO-style comparison
  should use about `50K`; broader domain adaptation should be `100K-200K`.
- Teacher/sample generation is the wall-clock bottleneck, not drafter training.
  From the current 1K experience, 50K teacher data is roughly `28-30h` at
  8-node concurrency or `7-8h` at 32-node concurrency. Drafter training for
  that scale should be around `1-2h`.
- Do not scale to 500K until a 10K/50K objective shows higher held-out
  acceptance than public PARD. The current 50K candidate is compatible but not
  yet clearly better than public PARD on Qwen3-235B OpenMath.

DFlash continuation update after the 02:35 PDT section on 2026-06-06:

- The DFlash-capable vLLM source build finally passed in retry7. Job `3185715`
  completed successfully in `00:51:35`.
- Build artifact:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_dflash_pr38300_0b790a2_cu129_torch28nv_source_py312`.
- Build report:
  `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/vllm_native_source_build_dflash_pr38300.json`.
  It reports vLLM `0.19.1rc1.dev315+g0b790a250.d20260606`,
  `supported_speculators_types=["dflash","eagle3"]`,
  `qwen3_dflash_import_ok=true`, and `dflash_proposer_import_ok=true`.
- The first external support probe after that build, `3185716`, failed before
  writing `probe.json`: importing source-built `vllm._C` in the standalone
  probe container raised `ImportError: libcudart.so.12: cannot open shared
  object file`.
- A newer probe retry, `3187827`, exposed a submit-script quoting bug:
  `pybin: unbound variable`. The support-probe submitter was patched to create
  a small `probe_runner.sh` and run that under `srun`, avoiding nested
  `bash -lc` variable expansion. The base standalone benchmark submitter was
  also patched to inject CUDA runtime library paths when a source-built vLLM
  site is prepended through `SOURCE_VLLM_SITE`.
- New DFlash retry9 jobs:
  - `3187842`: DFlash support probe, no dependency.
  - `3187845`: Qwen3-235B OpenMath baseline, `afterok:3187842`.
  - `3187848`: Qwen3-235B OpenMath DFlash K3, `afterok:3187842`.
  - `3187849`: Qwen3-235B OpenMath DFlash K5, `afterok:3187842`.
- The retry9 gate uses `ISL=1024`, `OSL=1024`, batch sizes
  `1 2 4 8 16 32`, target `TP=4`, draft `TP=4`, and the aligned smoke512
  DFlash checkpoint. It is still a runtime feasibility/performance gate, not
  NeMo-RL E2E evidence.
- Retry9 result: `3187842` wrote a structured `probe.json` but failed because
  `vllm.model_executor.models.qwen3_dflash` still triggers
  `ImportError: libcudart.so.12: cannot open shared object file`. The source
  build itself is valid; the remaining blocker is runtime container linkage
  for `vllm._C` in the external benchmark/probe container. Dependent jobs
  `3187845`, `3187848`, and `3187849` were cancelled because the `afterok`
  dependency could not be satisfied.
- Retry10 was submitted with the same container that produced the successful
  source build:
  `/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`.
  Jobs are:
  - `3187887`: DFlash support probe.
  - `3187909`: Qwen3-235B OpenMath baseline, `afterok:3187887`.
  - `3187911`: Qwen3-235B OpenMath DFlash K3, `afterok:3187887`.
  - `3187912`: Qwen3-235B OpenMath DFlash K5, `afterok:3187887`.
  This retry tests whether matching the build/runtime container resolves the
  `libcudart.so.12` linkage problem.
- Retry10 result: `3187887` switched to the build container but the generated
  probe runner did not set `PYTHONPATH` inside the container, so it failed with
  `No module named 'vllm'`. This was a submitter bug, not a DFlash runtime
  result. Dependent jobs `3187909`, `3187911`, and `3187912` were cancelled.
- Retry11 was submitted after exporting `SOURCE_VLLM_SITE` into the runner and
  setting `PYTHONPATH` inside the container:
  - `3187927`: DFlash support probe.
  - `3187932`: Qwen3-235B OpenMath baseline, `afterok:3187927`.
  - `3187934`: Qwen3-235B OpenMath DFlash K3, `afterok:3187927`.
  - `3187946`: Qwen3-235B OpenMath DFlash K5, `afterok:3187927`.
- Retry11 support result: `3187927` completed successfully in `00:01:41`.
  The probe reports `dflash_ready=true`, no errors,
  `vllm.model_executor.models.qwen3_dflash=true`,
  `vllm.v1.spec_decode.dflash=true`, and
  `supported_speculators_types=["dflash","eagle3"]` under vLLM
  `0.19.1rc1.dev315+g0b790a250.d20260606`.
- This means DFlash is now a runnable Qwen3-235B vLLM runtime candidate in the
  matching build container. The OpenMath gate jobs `3187932`, `3187934`, and
  `3187946` were dependency-unblocked.
- Follow-up poll: all three OpenMath gate jobs started at
  `2026-06-06T02:21:43`:
  - `3187932`: baseline K0 on node `nvl72139-T14`.
  - `3187934`: DFlash K3 on node `nvl72041-T18`.
  - `3187946`: DFlash K5 on node `nvl72081-T08`.
- Those three jobs failed quickly at `vllm.LLM` import, before model load or
  benchmark data collection. Error:
  `ImportError: cannot import name 'ChatCompletionFunctionToolParam' from
  openai.types.chat`. This is a stale `openai` package in the build container,
  not a DFlash runtime failure. The source-built site's `transformers`
  metadata includes `openai>=1.98.0` for serving, so the next retry should
  install `openai>=1.98.0,<2` into the job-local `pydeps` path before importing
  vLLM.
- Retry12 was submitted with `openai>=1.98.0,<2` installed into per-job
  `pydeps` and no stale dependency:
  - `3188028`: Qwen3-235B OpenMath baseline K0.
  - `3188031`: Qwen3-235B OpenMath DFlash K3.
  - `3188033`: Qwen3-235B OpenMath DFlash K5.

Qwen3-30B-A3B NeMo-RL continuation update after the 02:35 PDT section:

- Clean retry jobs `3187431` baseline, `3187432` public PARD K5, and `3187433`
  local CAT/PARD-2-style K5 all reached `SETUP COMPLETE`, `Step 1/5`, and
  first generation, then failed during `update_weights_via_ipc_zmq` with the
  same `shard_dim=0 is not a valid data dimension for a 3D tensor` error.
- Because these jobs had already started before the final fresh diagnostic
  gate, they should be treated as stale/runtime-mixed evidence. They still
  reinforce that baseline/public/local all fail at the same Qwen3-MoE refit
  boundary, so this is not caused by PARD.
- Fresh diagnostic job `3187658` remains the authoritative validation for the
  Megatron-Bridge split-dimension patch. At the latest poll it was running and
  initializing vLLM workers; it had not yet reached the refit failure/success
  gate.

DFlash retry12 and Qwen3-30B-A3B diagnostic result after the 02:35 PDT section:

- Retry12 DFlash OpenMath jobs all failed before any benchmark datapoint:
  - `3188028`: baseline K0, failed in `00:02:33`.
  - `3188031`: DFlash K3, failed in `00:02:49`.
  - `3188033`: DFlash K5, failed in `00:02:22`.
- The stale `openai` package issue is fixed. The new blocker is source-built
  vLLM's compile/fusion import path:
  `torch.ops._C.per_token_group_fp8_quant.default` is missing, raising
  `AttributeError: '_OpNamespace' '_C' object has no attribute
  'per_token_group_fp8_quant'`.
- This is not a DFlash performance/acceptance result. It happens while
  constructing `VllmConfig`, before model load. The next DFlash retry should
  either disable the fusion path or guard the missing FP8 dynamic quant op in
  the source-built vLLM site before rerunning K0/K3/K5.
- Fresh Qwen3-30B-A3B diagnostic job `3187658` failed after `SETUP COMPLETE`,
  `Step 1/1`, and first generation request batch `2048`.
- The debug dump shows incoming NeMo weights are already per-expert 2D tensors,
  for example `model.layers.0.mlp.experts.0.gate_proj.weight` has shape
  `(768, 2048)` and `down_proj.weight` has shape `(2048, 768)`.
- vLLM's loaded Qwen3-MoE parameters are grouped/packed tensors:
  `model.layers.0.mlp.experts.w13_weight` has shape
  `(128, 32, 1536, 64)` and `w2_weight` has shape `(128, 12, 2048, 64)`.
- The remaining failure is still
  `ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor
  (expected 1 or 2)` inside `update_weights_via_ipc_zmq`.
- Updated interpretation: the earlier Megatron-Bridge split-dim patch got the
  incoming side into per-expert 2D form, but the NeMo-RL refit path still calls
  vLLM's grouped-MoE weight loader through names/shapes that do not match the
  packed `w13_weight`/`w2_weight` representation. This remains a Qwen3-MoE
  refit bug and is independent of public PARD/local CAT quality.
- DFlash retry13 was submitted with the same source-built vLLM site, build
  container, `openai>=1.98.0,<2`, and an explicit compilation config disabling
  the problematic allreduce/RMS/quant fusion pass path:
  - `3188109`: Qwen3-235B OpenMath baseline K0.
  - `3188110`: Qwen3-235B OpenMath DFlash K3.
  - `3188112`: Qwen3-235B OpenMath DFlash K5.
- Retry13 was still `PENDING (Priority)` immediately after submission. The
  success gate is reaching `breakdown.json`; the failure gate is whether any
  new error replaces the previous `per_token_group_fp8_quant` import failure.

DFlash retry13, retry14, retry15 update after the 02:46 PDT section:

- Retry13 jobs `3188109`, `3188110`, and `3188112` failed before model load
  because the submitter passed raw JSON through `bash -lc "..."`. The JSON
  double quotes were stripped by shell parsing and Python received an invalid
  value for `--compilation-config-json`. This was a submitter quoting bug, not
  DFlash evidence.
- The benchmark script now accepts `--compilation-config-json @path`, and the
  submitter writes `${LOGS_DIR}/compilation_config.json` before sbatch. Local
  checks passed:
  `bash -n submit_vllm_standalone_specdec_breakdown.sh`,
  `bash -n submit_qwen235b_dflash_openmath_gate.sh`, and
  `python3 -m py_compile standalone_vllm_specdec_breakdown.py`.
- Retry14 jobs `3188178`, `3188179`, and `3188180` then reached source-built
  vLLM worker initialization but failed in TP/NCCL setup:
  `AttributeError: module 'torch.accelerator' has no attribute 'device_index'`.
  This is a source-vLLM/container torch API mismatch, not a DFlash acceptance
  or throughput result.
- `specdec_breakdown_instrumentation/sitecustomize.py` now installs a
  compatibility shim that maps missing `torch.accelerator.device_index` to a
  `torch.cuda.device(index)` context manager. The shim runs regardless of the
  profiler instrumentation flag because the vLLM startup path needs it even in
  wall-clock-only benchmark mode.
- Retry15 was submitted with the JSON file path fix plus the torch accelerator
  compatibility shim:
  - `3188231`: Qwen3-235B OpenMath baseline K0.
  - `3188232`: Qwen3-235B OpenMath DFlash K3.
  - `3188233`: Qwen3-235B OpenMath DFlash K5.
- Retry15 was `PENDING (Priority)` immediately after submission. Its success
  gate is still `breakdown.json` for K0/K3/K5; its failure gate is whether a
  new source-vLLM compatibility issue appears after the TP/NCCL setup shim.

Qwen3-30B-A3B Triton-MoE NeMo-RL retry after the 02:46 PDT section:

- The Qwen3-30B-A3B Full-GRPO step1 validation was resubmitted with
  `++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton` to avoid
  refitting into FlashInfer/TRTLLM packed grouped-MoE weights.
- Submitted jobs:
  - `3188183`: baseline, no specdec.
  - `3188184`: public PARD K5, `amd/PARD-Qwen3-0.6B`.
  - `3188185`: local CAT/PARD-2-style K5 checkpoint
    `PARD-Qwen3-0.6B_qwen30ba3b_math_k5_cat_tpp_mask_1024_resume_20260605_173358/checkpoint-32`.
- All three use `4n4g`, GBS `2048`, `max_steps=1`, natural EOS,
  `max_new_tokens=256`, sampling `temperature=1.0`, `top_p=1.0`, `top_k=-1`,
  generation TP `1`, training TP `1`, expert parallel `16`, and always-on
  parallel drafting for the two K5 runs.
- The command lines in `slurm-3188183.out`, `slurm-3188184.out`, and
  `slurm-3188185.out` confirm the Triton MoE override is present. At latest
  poll all workers had connected (`16/16`) and the driver was still installing
  or entering the NeMo-RL runtime. No repeat of the earlier
  `shard_dim=0 is not a valid data dimension for a 3D tensor` refit failure had
  been observed yet.

DFlash retry16 and retry17 update after the 02:55 PDT section:

- Retry16 was submitted after expanding the torch accelerator shim to cover
  `device_index`, `current_device_index`, `device_count`, `set_device_index`,
  `empty_cache`, `synchronize`, and common memory-stat APIs via `torch.cuda`.
  Jobs were:
  - `3188270`: Qwen3-235B OpenMath baseline K0.
  - `3188273`: Qwen3-235B OpenMath DFlash K3.
  - `3188288`: Qwen3-235B OpenMath DFlash K5.
- Retry16 advanced beyond the previous `torch.accelerator.device_index` and
  `torch.accelerator.empty_cache` failures. The new blocker is source-vLLM's
  FlashInfer sampler path:
  `ImportError: cannot import name 'fast_decode_plan' from 'flashinfer.decode'`.
- The root cause is another source-vLLM/container library mismatch. The
  submitter had been forcing `VLLM_USE_FLASHINFER_SAMPLER=1`; the container's
  FlashInfer package does not expose the `fast_decode_plan` API expected by the
  source-built vLLM site.
- `submit_vllm_standalone_specdec_breakdown.sh` now exposes
  `VLLM_USE_FLASHINFER_SAMPLER` as an override instead of hardcoding it to `1`.
  `submit_qwen235b_dflash_openmath_gate.sh` defaults it to `0` for this DFlash
  source-build gate.
- Retry17 was submitted with FlashInfer sampler disabled:
  - `3188317`: Qwen3-235B OpenMath baseline K0.
  - `3188318`: Qwen3-235B OpenMath DFlash K3.
  - `3188319`: Qwen3-235B OpenMath DFlash K5.
- Retry17 still uses the same model, DFlash checkpoint, source vLLM site,
  build container, `openai>=1.98.0,<2`, compilation-fusion-off config, torch
  accelerator compatibility shim, `ISL=1024`, `OSL=1024`, batch sizes
  `1 2 4 8 16 32`, target `TP=4`, draft `TP=4`, and wall-clock
  `LLM.generate` benchmark mode.

DFlash retry17 and retry18 update after the 03:00 PDT section:

- Retry17 failed before benchmark data collection:
  - `3188317` baseline reached model worker construction but still hit the
    source-vLLM compile backend import path and failed on missing
    `torch.ops._C.per_token_group_fp8_quant.default`.
  - `3188318` DFlash K3 and `3188319` DFlash K5 avoided the FlashInfer sampler
    import failure but failed earlier in DFlash/non-causal attention setup:
    `Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for this
    configuration. Reason: ['non-causal attention not supported']`.
- The next runtime requirements are therefore:
  - Use `enforce_eager=true` for this source-built DFlash benchmark to avoid
    the compile/fusion matcher import path.
  - Do not use `TRITON_ATTN` for DFlash; use Qwen3-235B's usual
    `FLASH_ATTN` path.
  - Keep `VLLM_USE_FLASHINFER_SAMPLER=0` because the container FlashInfer API
    is older than the source-built vLLM expects.
- The standalone submitter now exposes `ATTENTION_BACKEND` instead of
  hardcoding `TRITON_ATTN`. The DFlash OpenMath gate wrapper now defaults to
  `ATTENTION_BACKEND=FLASH_ATTN`, `ENFORCE_EAGER=true`, and
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Retry18 was submitted with these settings:
  - `3188367`: Qwen3-235B OpenMath baseline K0.
  - `3188369`: Qwen3-235B OpenMath DFlash K3.
  - `3188370`: Qwen3-235B OpenMath DFlash K5.
- Retry18 was `PENDING` immediately after submission. The success gate remains
  producing `breakdown.json`; any new failure should be treated as the next
  source-built vLLM/container compatibility issue rather than algorithmic
  DFlash performance evidence.

DFlash retry18, retry19, and Qwen3-30B-A3B Triton-MoE update after the
03:06 PDT section:

- Retry18 failed before benchmark data collection:
  - `3188367` baseline K0 failed after model execution reached the
    `FLASH_ATTN` backend because source-built vLLM imported
    `vllm.vllm_flash_attn.cute`, which requires `cutlass.cute`; the container
    does not provide the `cutlass` Python package.
  - `3188369` DFlash K3 and `3188370` DFlash K5 used a correct
    `{"method": "dflash", ...}` speculative config, but worker startup still
    imported the common spec-decode drafter path and failed in the compile
    fusion matcher on missing
    `torch.ops._C.per_token_group_fp8_quant.default`.
- Interpretation: retry18 is still not DFlash algorithm evidence. It exposes
  two separate source-built-vLLM/container mismatches:
  missing CUTE/CUTLASS support for `FLASH_ATTN`, and missing FP8 dynamic group
  quant op for compile/fusion module import.
- `specdec_breakdown_instrumentation/sitecustomize.py` now also installs
  import-time placeholders for `torch.ops._C.per_token_group_fp8_quant` and
  `per_token_group_fp8_quant_packed`. The placeholders only allow disabled
  fusion modules to import; if the op is actually executed, they raise a clear
  runtime error.
- Retry19 was submitted for DFlash only, skipping the source-build baseline,
  with `ATTENTION_BACKEND=FLASHINFER`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `ENFORCE_EAGER=true`, the torch accelerator shim, and the missing-FP8-op
  import shim:
  - `3188432`: Qwen3-235B OpenMath DFlash K3.
  - `3188434`: Qwen3-235B OpenMath DFlash K5.
- Retry19 was `PENDING (Priority)` immediately after submission. The success
  gate remains `breakdown.json`; the likely next failure gate is whether the
  container's FlashInfer attention backend is compatible with this
  source-built vLLM site.
- Qwen3-30B-A3B Triton-MoE NeMo-RL step1 jobs `3188183`, `3188184`, and
  `3188185` are all running at the latest poll. Their driver logs confirm that
  vLLM workers are using `TRITON Unquantized MoE` and
  `MoEPrepareAndFinalizeNoDPEPModular`, so the intended MoE backend override is
  active.
- No repeat of the earlier FlashInfer/TRTLLM grouped-MoE refit error has been
  observed yet in these jobs. They have not emitted generation throughput,
  acceptance, or E2E step metrics yet; they are still in worker startup/model
  load/runtime initialization.

DFlash retry19/retry20 and Qwen3-30B-A3B local-transformer retry after the
03:15 PDT section:

- Retry19 DFlash-only jobs `3188432` K3 and `3188434` K5 failed before vLLM
  import completed. The immediate blocker was
  `ImportError: libcudart.so.12: cannot open shared object file`, caused by the
  source-built DFlash vLLM site requiring CUDA runtime libraries that were not
  visible inside the selected benchmark container.
- Retry20 attempted to install `openai>=1.98.0,<2` and
  `nvidia-cuda-runtime-cu12>=12.8,<13` into job-local `pydeps`, but both jobs
  failed in the generated shell command before Python startup:
  - `3188480`: DFlash K3, failed with bash syntax error around the pip spec.
  - `3188481`: DFlash K5, same syntax error.
- Root cause of retry20 was submitter quoting, not algorithm behavior. The
  generated `[[ ... != '' ]]` condition embedded shell-quoted pip specs such
  as `openai>=1.98.0,<2`; the `<2` token was parsed as part of a malformed
  conditional expression.
- `submit_vllm_standalone_specdec_breakdown.sh` was fixed to write the pip
  requirements into `${LOGS_DIR}/pip_requirements.txt` and install with
  `python3 -m pip install ... -r ...`, avoiding shell interpolation of version
  constraints. The job-local CUDA runtime path
  `${LOGS_DIR}/pydeps/nvidia/cuda_runtime/lib` is now also added to
  `LD_LIBRARY_PATH` before importing source-built vLLM.
- Retry21 was submitted with the corrected submitter, still using
  `ATTENTION_BACKEND=FLASHINFER`, FlashInfer sampler disabled, eager mode,
  `TP=4`, draft `TP=4`, OpenMath prompts, `ISL=1024`, `OSL=1024`, and batch
  sizes `1 2 4 8 16 32`:
  - `3188536`: Qwen3-235B DFlash K3.
  - `3188547`: Qwen3-235B DFlash K5.
- Latest poll immediately after submission showed `3188536` running and
  `3188547` pending; neither had emitted `breakdown.json` yet.

Qwen3-30B-A3B NeMo-RL follow-up:

- The Triton-MoE step1 jobs `3188183` baseline, `3188184` public PARD K5, and
  `3188185` local CAT/PARD-2-style K5 all failed in the Megatron policy forward
  path after worker startup. Because the baseline failed as well, this was not
  PARD-specific. The failure class is TransformerEngine RMSNorm CUDA invalid
  argument during policy/logprob forward.
- The old grouped-MoE refit blocker did not recur; logs confirmed the intended
  `++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton` override.
- A new Qwen3-30B-A3B Full-GRPO retry was submitted with
  `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`,
  `policy.megatron_cfg.force_reconvert_from_hf=true`, and a fresh Megatron
  checkpoint cache
  `nrl_megatron_ckpts_online_pard_qwen30ba3b_localtransformer`. This mirrors
  the Qwen3-8B TE RMSNorm workaround while preserving the Triton-MoE generation
  override.
- Submitted Qwen3-30B-A3B local-transformer jobs:
  - `3188532`: baseline, no specdec.
  - `3188533`: public PARD K5, `amd/PARD-Qwen3-0.6B`.
  - `3188534`: local CAT/PARD-2-style K5 checkpoint
    `PARD-Qwen3-0.6B_qwen30ba3b_math_k5_cat_tpp_mask_1024_resume_20260605_173358/checkpoint-32`.
- These jobs use `max_steps=5`, `4n4g`, GBS `2048`, generation TP `1`,
  training TP `1`, expert parallel `16`, `max_new_tokens=256`, sampling
  `temperature=1.0`, `top_p=1.0`, `top_k=-1`, and natural EOS. Latest poll
  immediately after submission showed all three pending.

DFlash retry21/retry22/retry23 update after the 03:23 PDT section:

- Retry21 used the original standalone benchmark container with the fixed
  requirements-file submitter:
  - `3188536`: DFlash K3.
  - `3188547`: DFlash K5.
- Both retry21 jobs failed before model load with
  `ImportError: /usr/lib/aarch64-linux-gnu/libc.so.6: version GLIBC_2.38 not
  found`, required by the source-built `vllm/_C.abi3.so`. This confirms the
  original standalone container is ABI-incompatible with the DFlash source site.
- Retry22 switched back to the matching build container
  `/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`
  and installed only `openai>=1.98.0,<2` into job-local `pydeps`:
  - `3188601`: DFlash K3.
  - `3188604`: DFlash K5.
- Retry22 passed the glibc/openai gates and reached vLLM worker model
  construction, but failed because `ATTENTION_BACKEND=FLASHINFER` was not valid
  in that container/source-site combination. The concrete import symptom in the
  log is `No module named 'triton.language.target_info'`, followed by
  `ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not valid for
  this configuration. Reason: ['ImportError']`.
- Retry23 was submitted using the same build container and requirements-file
  submitter, but with `ATTENTION_BACKEND=FLASH_ATTN`:
  - `3188637`: DFlash K3.
  - `3188638`: DFlash K5.
- Latest poll immediately after submission showed retry23 pending. The next
  success gate is `breakdown.json`; the likely next failure gate is whether the
  build container has the `cutlass.cute` support needed by source-built
  `FLASH_ATTN`.

Qwen3-30B-A3B local-transformer retry status after the 03:23 PDT section:

- Local-transformer retry jobs `3188532`, `3188533`, and `3188534` all started
  and reached Ray worker connection. Their driver commands confirm
  `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`,
  `policy.megatron_cfg.force_reconvert_from_hf=true`, the fresh
  `nrl_megatron_ckpts_online_pard_qwen30ba3b_localtransformer` checkpoint
  cache, and the Triton MoE generation override.
- At latest poll all three jobs were still building/downloading the fresh driver
  environment and had not reached `SETUP COMPLETE`, generation, or the
  TransformerEngine RMSNorm gate yet.

DFlash retry23/retry24/retry25 update after the 03:35 PDT section:

- Retry23 with the matching build container and `ATTENTION_BACKEND=FLASH_ATTN`
  progressed further than all previous DFlash attempts. It selected the
  source-built DFlash-capable vLLM runtime
  `0.19.1rc1.dev315+g0b790a250.d20260606`, resolved
  `Qwen3MoeForCausalLM` plus `DFlashDraftModel`, and loaded the Qwen3-235B
  checkpoint weights.
- Retry23 then failed at first forward because the build container did not have
  the CUTLASS CuTe DSL Python module required by the source-built
  FlashAttention path:
  - `3188637`: DFlash K3, failed.
  - `3188638`: DFlash K5, failed.
  - Error class: `ModuleNotFoundError: No module named 'cutlass'` while
    importing `vllm.vllm_flash_attn.cute.interface` and
    `cutlass.cute as cute`.
- Retry24 added `nvidia-cutlass-dsl>=4.3,<5` and confirmed that PyPI resolves
  `nvidia-cutlass-dsl==4.5.2`. This solved the missing package directionally,
  but the job-local pip resolver also installed `numpy==2.4.6` into `pydeps`.
  That shadowed the container's NumPy-1.x ABI and broke `cv2`/PyTorch/vLLM
  startup:
  - `3188704`: DFlash K3, cancelled after the NumPy ABI blocker was confirmed.
  - `3188705`: DFlash K5, cancelled after the same blocker was confirmed.
  - Error class: `A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.4.6`, followed by vLLM worker `RuntimeError: Numpy is not
    available`.
- Metadata inspection showed `nvidia-cutlass-dsl==4.5.2` itself only requires
  `nvidia-cutlass-dsl-libs-base==4.5.2` and optional `cu13` libs; it does not
  require NumPy 2. The next retry therefore pins NumPy below 2 to avoid
  polluting the container ABI.
- Retry25 was submitted with the matching build container, `FLASH_ATTN`,
  `TP=4`, draft `TP=4`, OpenMath prompts, `ISL=1024`, `OSL=1024`, batch sizes
  `1 2 4 8 16 32`, and job-local requirements
  `numpy<2 openai>=1.98.0,<2 nvidia-cutlass-dsl>=4.3,<5`:
  - `3188770`: Qwen3-235B DFlash K3, pending at latest poll.
  - `3188778`: Qwen3-235B DFlash K5, pending at latest poll.
- The next hard gate remains `breakdown.json`. If retry25 passes Python import
  and first forward, it will finally produce measured DFlash/PARD-style
  standalone throughput and acceptance evidence for Qwen3-235B.

Qwen3-30B-A3B local-transformer retry status after the 03:35 PDT section:

- Jobs `3188532` baseline, `3188533` public PARD K5, and `3188534` local
  CAT/PARD-2-style K5 are still running at the latest poll.
- They are still in driver/Ray worker virtual-environment setup and have not
  emitted `SETUP COMPLETE`, GRPO step metrics, generation throughput,
  acceptance, or E2E throughput yet.
- The job logs confirm the intended experiment shape:
  - baseline: `policy.draft.enabled=false` and no speculative config.
  - public PARD: `amd/PARD-Qwen3-0.6B`, `num_speculative_tokens=5`,
    `parallel_drafting=true`.
  - local CAT/PARD-2-style: checkpoint
    `PARD-Qwen3-0.6B_qwen30ba3b_math_k5_cat_tpp_mask_1024_resume_20260605_173358/checkpoint-32`,
    `num_speculative_tokens=5`, `parallel_drafting=true`.
- Shared settings are still `4n4g`, GBS `2048`, generation batch size `32`,
  generation TP `1`, training TP `1`, expert parallel `16`,
  `max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`, `top_k=-1`, natural
  EOS, Triton MoE generation backend, and the local-transformer Megatron
  reconversion path.

DFlash retry25/retry26 update after the 03:46 PDT section:

- Retry25 advanced past the retry24 NumPy ABI blocker. Both K3 and K5 installed
  `numpy==1.26.4`, `nvidia-cutlass-dsl==4.5.2`, and
  `nvidia-cutlass-dsl-libs-base==4.5.2`, selected `FLASH_ATTN`, loaded the
  235B target weights, initialized KV cache, and entered the first scheduler
  step.
- Retry25 still failed before benchmark metrics because `import cutlass.cute`
  was not visible inside the worker processes:
  - `3188770`: DFlash K3, failed after engine init.
  - `3188778`: DFlash K5, failed after engine init.
  - Error class: `ModuleNotFoundError: No module named 'cutlass'`.
- The root cause is now path handling, not package availability. The
  `nvidia-cutlass-dsl` wheel installs the importable module under
  `pydeps/nvidia_cutlass_dsl/python_packages/cutlass` and exposes it through
  `nvidia_cutlass_dsl.pth`. The job submitter was adding `pydeps` directly to
  `PYTHONPATH`, which does not reliably process that `.pth` file.
- Probe result with the explicit nested path succeeded:
  `PYTHONPATH=$pydeps/nvidia_cutlass_dsl/python_packages:$pydeps` and
  `LD_LIBRARY_PATH=$pydeps/nvidia_cutlass_dsl/lib` can import both `cutlass` and
  `cutlass.cute`.
- `submit_vllm_standalone_specdec_breakdown.sh` was updated to add
  `${LOGS_DIR}/pydeps/nvidia_cutlass_dsl/python_packages` to `PYTHONPATH` and
  `${LOGS_DIR}/pydeps/nvidia_cutlass_dsl/lib` to `LD_LIBRARY_PATH` whenever
  job-local `pydeps` are used.
- Retry26 was submitted with the same DFlash setup and the CUTLASS path fix:
  - `3188925`: Qwen3-235B DFlash K3.
  - `3188937`: Qwen3-235B DFlash K5.
- Success gate remains `breakdown.json`. The next likely blocker, if any, is no
  longer package discovery but CUTLASS runtime/ABI compatibility during the
  actual FlashAttention/CuTe call.

Qwen3-30B-A3B local-transformer retry update after the 03:46 PDT section:

- Jobs `3188532` baseline and `3188533` public PARD K5 failed before GRPO
  metrics, but the failure class changed from the previous TransformerEngine
  RMSNorm runtime error. The new blocker is policy-side Megatron model
  construction:
  `AssertionError: Grouped GEMM is not available. Please run pip install
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`.
- This is not PARD-specific because the baseline failed the same way. It is
  caused by the forced local-transformer Megatron import path preserving
  `moe_grouped_gemm=true` while the actor venv lacks the grouped_gemm extension.
- The submit wrapper now supports `QWEN30_MOE_GROUPED_GEMM`, defaulting to the
  previous behavior `true`, and passes
  `++policy.megatron_cfg.moe_grouped_gemm=${QWEN30_MOE_GROUPED_GEMM}`.
- The old local CAT/PARD-style job `3188534` was cancelled because it had the
  same grouped-GEMM config. A fresh retry was submitted with
  `QWEN30_MOE_GROUPED_GEMM=false`, fresh actor venv suffix, and fresh Megatron
  checkpoint cache:
  - `3188907`: Qwen3-30B-A3B baseline, no specdec.
  - `3188908`: Qwen3-30B-A3B public PARD K5.
  - `3188909`: Qwen3-30B-A3B local CAT/PARD-2-style K5.
- These retry jobs are pending at latest poll. The next gate is whether
  disabling grouped GEMM lets Megatron policy creation reach the previous
  RMSNorm/forward gate and then actual GRPO step metrics.

DFlash retry26/retry27 update after the 04:00 PDT section:

- Retry26 passed the previous `cutlass.cute` path blocker. Both jobs loaded the
  Qwen3-235B target, loaded the DFlash checkpoint, initialized KV cache, and
  reached the first scheduler/forward step.
- Retry26 then failed before any benchmark row because the source-built vLLM
  FlashAttention CUTE path imports `quack`:
  - `3188925`: Qwen3-235B DFlash K3, failed after engine init.
  - `3188937`: Qwen3-235B DFlash K5, failed after engine init.
  - Error class: `ModuleNotFoundError: No module named 'quack'` from
    `vllm/vllm_flash_attn/cute/*`.
- This is a missing runtime dependency, not a DFlash quality result. The
  source-built vLLM metadata explicitly requires `quack-kernels>=0.3.3`.
  Installing PyPI `quack` is the wrong fix: it lacks required submodules such as
  `quack.activation`.
- Probe result: `quack-kernels==0.5.0` provides the expected `quack`
  submodules. However, installing it with normal dependencies can pull a
  job-local `torch`, which risks shadowing the container's validated torch. The
  safer installation pattern is:
  - normal deps: `numpy<2 openai>=1.98.0,<2 nvidia-cutlass-dsl==4.5.2`.
  - no-deps deps: `quack-kernels==0.5.0`.
- `submit_vllm_standalone_specdec_breakdown.sh` now supports
  `PIP_INSTALL_NODEPS_SPECS` and installs those requirements with
  `pip install --no-deps --target $LOGS_DIR/pydeps`.
- Retry27 was submitted with the build container, `FLASH_ATTN`, explicit
  CUTLASS path, and no-deps `quack-kernels`:
  - `3189070`: Qwen3-235B DFlash K3, running at latest poll.
  - `3189071`: Qwen3-235B DFlash K5, running at latest poll.
- Success gate remains `breakdown.json`. The next immediate failure to watch for
  is CUTE/Quack runtime compatibility after import succeeds.

Qwen3-30B-A3B local-transformer no-grouped-GEMM retry correction:

- Retry jobs `3188907`, `3188908`, and `3188909` failed before actor setup or
  rollout metrics. This was a wrapper/Hydra syntax issue, not a PARD or
  Megatron execution result.
- Failure class:
  `ConfigCompositionException: Could not append to config. An item is already at
  'policy.megatron_cfg.moe_grouped_gemm'`.
- Root cause: the retry submitted `+policy.megatron_cfg.moe_grouped_gemm=false`
  while the key already exists in the recipe. The correct override is
  `++policy.megatron_cfg.moe_grouped_gemm=false`.
- `submit_qwen32_qwen30ba3b_pard2style_fullgrpo5.sh` was corrected to use the
  double-plus override.
- Retry2 was submitted with the same experiment shape, `moe_grouped_gemm=false`,
  fresh actor venv suffix, and the local-transformer Megatron cache:
  - `3189111`: Qwen3-30B-A3B baseline, no specdec.
  - `3189112`: Qwen3-30B-A3B public PARD K5.
  - `3189113`: Qwen3-30B-A3B local CAT/PARD-2-style K5.
- The next gate is now the real one again: whether no-grouped-GEMM local
  transformer construction gets past Megatron policy creation and into rollout
  metrics.

Latest poll after retry submissions:

- DFlash retry27:
  - `3189070` K3 is running; no `breakdown.json` yet. It is still in startup/pip
    or early model-init logging.
  - `3189071` K5 is running; no `breakdown.json` yet.
- Qwen3-30B-A3B no-grouped-GEMM retry2:
  - `3189111` baseline is running.
  - `3189112` public PARD K5 is pending.
  - `3189113` local CAT/PARD-2-style K5 is pending.

DFlash retry27/retry28 update after the 04:12 PDT section:

- Retry27 passed both previous dependency blockers:
  - `cutlass.cute` was visible through the explicit
    `pydeps/nvidia_cutlass_dsl/python_packages` path.
  - `quack` was visible through `quack-kernels==0.5.0`.
- Both retry27 jobs then reached Qwen3-235B weight load, DFlash checkpoint load,
  KV cache allocation, engine warmup, and first forward. That is further than
  retry26.
- Retry27 still failed before benchmark metrics because the CUTE FlashAttention
  path imports `tvm_ffi`:
  - `3189070`: Qwen3-235B DFlash K3, failed after first forward setup.
  - `3189071`: Qwen3-235B DFlash K5, failed after first forward setup.
  - Error class: `ModuleNotFoundError: No module named 'tvm_ffi'` from
    `vllm/vllm_flash_attn/cute/cache_utils.py`.
- Root cause: `quack-kernels` metadata requires
  `apache-tvm-ffi<0.2,>=0.1.6`, `torch-c-dlpack-ext`, and `einops`. We installed
  `quack-kernels` with `--no-deps` to avoid pulling a job-local `torch`, so the
  non-torch runtime deps must be supplied explicitly.
- Probe results:
  - `apache-tvm-ffi==0.1.11` provides importable `tvm_ffi`.
  - `torch-c-dlpack-ext==0.1.5` declares `Requires-Dist: torch`, so it should be
    installed with `--no-deps` to use the container torch.
- Retry28 was submitted with:
  - normal deps:
    `numpy<2 openai>=1.98.0,<2 nvidia-cutlass-dsl==4.5.2 apache-tvm-ffi==0.1.11 einops==0.8.2`.
  - no-deps deps:
    `quack-kernels==0.5.0 torch-c-dlpack-ext==0.1.5`.
  - `3189242`: Qwen3-235B DFlash K3, running at latest poll.
  - `3189243`: Qwen3-235B DFlash K5, running at latest poll.
- This is still a runtime feasibility gate, not a performance result. The
  success gate remains `breakdown.json`.

Latest poll after retry28 runtime entry, 2026-06-06 04:22 PDT:

- DFlash retry28 is the first Qwen3-235B DFlash run to pass the dependency
  stack through actual live generation:
  - `cutlass.cute`: passed via the explicit CUTLASS python package path.
  - `quack`: passed via `quack-kernels==0.5.0`.
  - `tvm_ffi`: passed via `apache-tvm-ffi==0.1.11`.
  - `torch-c-dlpack-ext`: installed with `--no-deps`, avoiding a local torch
    shadow of the container torch.
- Jobs are still running and have not emitted `breakdown.json` yet:
  - `3189242`: DFlash K3, running.
  - `3189243`: DFlash K5, running.
- The new evidence is a quality/performance warning, not another dependency
  blocker. Both K3 and K5 have entered generation and vLLM emits SpecDec
  metrics, but the DFlash drafter acceptance is near zero on OpenMath:
  - K3 observed lines: drafted `210-246` tokens per 10s window, accepted `0`;
    per-position acceptance `0.000, 0.000, 0.000`.
  - K5 observed lines: mostly accepted `0-6` tokens out of about `390-425`
    drafted tokens per 10s window; per-position acceptance only appears in the
    first draft position and peaks around `0.077, 0, 0, 0, 0` in the sampled
    windows.
  - K5 generation progress example: one prompt completed in `211.34s`, output
    speed about `4.85 toks/s`, while live engine output throughput windows were
    about `7.8-8.7 toks/s`.
- Interpretation: retry28 proves the DFlash runtime path can now execute, but
  the current DFlash checkpoint is not target/domain aligned enough for
  Qwen3-235B OpenMath. Unless the final `breakdown.json` contradicts the live
  metrics, this should not be promoted to NeMo-RL. The likely next Qwen3-235B
  path remains PARD public runtime plus larger target-confidence/CAT
  adaptation, or a true PARD-2 implementation when released.

Qwen3-30B-A3B PARD/PARD-2-style NeMo-RL retry2 poll, 2026-06-06 04:22 PDT:

- `3189111` baseline is still running and has passed the previous real blockers:
  - Hydra override for `moe_grouped_gemm=false` is accepted.
  - The driver config shows `policy.megatron_cfg.moe_grouped_gemm=False`.
  - vLLM generation worker venv creation completed on all 16 workers.
  - vLLM policy worker initialization reached `16/16`.
  - FlashInfer autotuning started and ended.
- No GRPO step metrics yet. The job is currently a setup-progress signal, not a
  performance result.
- `3189112` public PARD K5 and `3189113` local CAT/PARD-2-style K5 are still
  pending on priority. They should be allowed to run only after the baseline
  confirms the no-grouped-GEMM/local-transformer setup reaches rollout metrics.

PARD/PARD-2 public implementation status, 2026-06-06 check:

- The public AMD PARD repo exposes PARD training/inference and vLLM PARD
  integration, including Qwen3 PARD weights.
- The PARD-2 paper is public, but the repo still presents PARD-2 code/checkpoint
  release as pending rather than a directly usable training/runtime package.
- Current runnable approximation is therefore intentionally limited to:
  public PARD parallel-drafting runtime plus local CAT/token-prefix-product
  target-confidence weighting in the PARD trainer. Treat it as PARD-2-style
  adaptation, not as a claim that full PARD-2 has been reproduced.

Qwen3-235B CAT/PARD-2-style 2048-row expansion submitted, 2026-06-06 04:31 PDT:

- Added a reusable submitter:
  `experiments/pard_qwen3_235b_math/submit_qwen235b_pard2_cat_expand_2048.sh`.
- Purpose: scale the best current local CAT result from the completed
  1024-row teacher-logprob set to a 2048-row token-prefix-product-mask PARD K5
  drafter, then run the same OpenMath bs32 vLLM gate.
- Method remains the local PARD-2-style approximation:
  - base drafter: `amd/PARD-Qwen3-0.6B`.
  - target: `Qwen/Qwen3-235B-A22B`.
  - K: `5`.
  - CAT mode: `token_prefix_product`.
  - CAT loss mode: `mask`.
  - teacher generation: temperature `0.0`, top_p `1.0`, max tokens `1024`,
    `GENERATION_LOGPROBS=true`, `GENERATION_TOP_LOGPROBS=1`.
- Existing retained teacher-logprob rows:
  offsets `5000,5128,5256,5384,5512,5640,5768,5896`.
- Newly submitted 128-row teacher-logprob chunks:
  - `3189381`: offset `6024`.
  - `3189384`: offset `6152`.
  - `3189387`: offset `6280`.
  - `3189395`: offset `6408`.
  - `3189401`: offset `6536`.
  - `3189409`: offset `6664`.
  - `3189412`: offset `6792`.
  - `3189437`: offset `6920`.
- Dependent train job:
  - `3189439`: `qwen235b-pard2-cat-tpp-2048-train`.
  - waits for all eight new teacher jobs.
  - expected checkpoint:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_tpp_mask_2048_16x128_20260606_042506/checkpoint-32`.
- Dependent gate job:
  - `3189440`: vLLM standalone OpenMath bs32 K5 gate.
  - waits for train job `3189439`.
  - expected metric file:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard2cat_2048_16x128_20260606_042506_local_cat_tpp_openmath_isl1024_osl1024/breakdown.json`.
- Latest queue check:
  - `3189381`, `3189384`, `3189387`, `3189395`, `3189401`,
    `3189409`, and `3189412` were running.
  - `3189437` was pending on priority.
  - `3189439` and `3189440` were pending on dependency.
- Comparison target: the 1024-row TPP-mask checkpoint produced OpenMath bs32
  output throughput `543.08 tok/s/GPU`, speedup `1.122x`, acceptance
  `46.71%`, and mean acceptance length `3.336`. The 2048-row gate should be
  promoted only if it improves throughput and/or acceptance beyond this.

Latest follow-up, 2026-06-06 04:55 PDT:

- Qwen3-235B CAT/TPP 2048-row expansion is still healthy. All eight teacher
  jobs are running, and the dependent train/gate jobs remain queued on
  dependency:
  - running teacher jobs: `3189381`, `3189384`, `3189387`, `3189395`,
    `3189401`, `3189409`, `3189412`, `3189437`.
  - train job waiting on dependency: `3189439`.
  - vLLM OpenMath K5 gate waiting on dependency: `3189440`.
  - latest observed chunk progress: offset `6024` has `96` rows, `6152` has
    `88`, `6280` has `81`, `6408` has `80`, `6536` has `88`, `6664` has `72`,
    `6792` has `88`, and `6920` has `64`.
- Qwen3-32B NeMo-RL focused retry submitted:
  - job `3189630`, public PARD K5, `MAX_STEPS=1`, Full-GRPO no-stop.
  - target fix: `QWEN32_FORCE_LOCAL_TRANSFORMER_SPEC=true` plus fresh Megatron
    checkpoint conversion. This directly targets the r3
    TransformerEngine RMSNorm CUDA invalid-argument failure after generation.
  - status at submit check: pending on priority.
  - poll script: `scripts/poll_qwen32_pard_localtransformer_step1_status.sh`.
- Qwen3-30B-A3B NeMo-RL top-level MoE retry submitted:
  - jobs: baseline `3189658`, public PARD K5 `3189659`, local CAT/TPP K5
    `3189660`, all `MAX_STEPS=1`, Full-GRPO no-stop.
  - target fix: replace the previous nested override
    `policy.generation.vllm_kwargs.kernel_config.moe_backend=triton` with the
    vLLM EngineArgs-compatible top-level override
    `policy.generation.vllm_kwargs.moe_backend=triton`.
  - reason: the installed vLLM exposes `moe_backend` as an `LLM`/`EngineArgs`
    top-level kwarg and then copies it into `KernelConfig`. The old nested key
    could leave runtime selection at `auto`, so it was not a reliable way to
    avoid the Qwen3-MoE packed-weight refit path.
  - status at submit check: all three jobs pending on priority.
  - poll script: `scripts/poll_qwen30ba3b_pard_topmoe_step1_status.sh`.
- DFlash retry28 has completed cleanly (`3189242`, `3189243`) and should not
  be promoted with the current checkpoint:
  - K3 acceptance by batch `1,2,4,8,16,32`:
    `1.216%`, `0.446%`, `0.321%`, `0.733%`, `0.612%`, `0.932%`.
  - K5 acceptance by batch `1,2,4,8,16,32`:
    `0.440%`, `0.247%`, `0.278%`, `0.429%`, `0.349%`, `0.543%`.
  - Mean acceptance length stays near `1.01-1.04`, which means essentially no
    useful multi-token acceptance. Runtime dependencies are solved; drafter
    quality/domain alignment is the blocker.
- Added and submitted a D-PACE-style target-probability ablation for the same
  Qwen3-235B 2048-row teacher block:
  - Code changes:
    - `pard_train_cat_weighted.py` now supports
      `cat_importance_mode=dpace_target`, `dpace_target_cumulative`, and
      `dpace_target_continuation`.
    - `make_pard_train_config.py`, `train_pard_math_k5.sh`, and
      `submit_pard_math_k5_train.sh` now pass `cat_dpace_smoothing`.
    - `submit_qwen235b_pard2_cat_expand_2048.sh` is parameterized for future
      CAT/D-PACE objective reuse.
  - This is not official D-PACE and not official PARD-2. D-PACE computes
    detached weights from the draft confidence on target tokens. This ablation
    uses Qwen3-235B teacher target confidence instead, with D-PACE's cumulative
    prefix and continuation-value form. It is a practical target-prob surrogate
    while official PARD-2 code/checkpoints are unavailable.
  - Submitted jobs:
    - `3189721`: D-PACE-target weighted-CE PARD K5 train, dependent on the same
      teacher jobs `3189381`, `3189384`, `3189387`, `3189395`, `3189401`,
      `3189409`, `3189412`, `3189437`.
    - `3189722`: OpenMath bs32 K5 vLLM gate, dependent on `3189721`.
  - Training settings:
    - `POSITION_LOSS_WEIGHTING=uniform`.
    - `CAT_CONFIDENCE_FLOOR=0.0`.
    - `CAT_IMPORTANCE_MODE=dpace_target`.
    - `CAT_LOSS_MODE=weighted_ce`.
    - `CAT_DPACE_SMOOTHING=0.5`.
  - Poll script: `scripts/poll_qwen235b_pard2_dpace_target_2048_status.sh`.
- Latest 2048 teacher status at 2026-06-06 05:06 PDT:
  - Completed: offsets `6024`, `6152`, `6280`, `6408`, `6536`, `6792`.
  - Running: offsets `6664`, `6920`.
  - Current row counts: `6024=128`, `6152=128`, `6280=128`,
    `6408=128`, `6536=128`, `6664=121`, `6792=128`, `6920=120`.
- Pre-train dependency fix applied before queued train jobs started:
  - `pard_train_cat_weighted.py` previously imported `click` only for command
    line parsing.
  - The retained train venv used by the queued jobs does not expose all Python
    packages outside the container, and `click` was not a safe dependency.
  - Replaced `click` with `argparse`, verified local `py_compile`, and copied
    the updated trainer/config/train scripts to the remote runtime path:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_work/experiments/pard_qwen3_235b_math/`.
  - Remote syntax checks passed and `grep` confirms no `import click` remains.
- NeMo-RL focused retries remain queued on priority, with no new failure logs:
  - Qwen3-32B local-transformer public PARD K5: `3189630`.
  - Qwen3-30B-A3B top-level `moe_backend=triton` baseline/public/local:
    `3189658`, `3189659`, `3189660`.

Practical PARD-2-style training scale estimate:

- The public PARD repo has PARD training/inference and Qwen3 weights, but it
  still states that PARD-2 code/checkpoints are pending release. Our current
  path is therefore a PARD-2-style approximation on top of public PARD.
- For this approximation, sample count is not the main bottleneck. Qwen3-235B
  teacher logprob/hidden extraction is the expensive step.
- Suggested staged scale:
  - `128-512` rows: smoke and loss/acceptance sanity.
  - `2K-10K` rows: quick math-domain adaptation signal.
  - `30K-100K+` rows: credible targeted drafter quality evaluation.
  - larger multi-domain runs only make sense after OpenMath speedup improves.
- Observed drafter fine-tune cost is small once teacher rows exist:
  - 1K CAT prefix-product: `train_runtime=90.27s`.
  - 2K CAT/TPP mask: `4m45s` elapsed, `train_runtime=160.467s`.
  - 2K D-PACE draft-probability CE: `4m50s` elapsed,
    `train_runtime=162.373s`.
  - Rough extrapolation on the same 4-GPU trainer is `5-7m` for 4K rows,
    `13-15m` for 10K, `40-50m` for 30K, and `2-2.5h` for 100K, excluding
    queue time and teacher generation.
- Current 2048-row expansion should be treated as the first real signal beyond
  smoke. If it remains near the 1024-row result, the next move should be a
  better accepted-prefix/CAT or LK/D-PACE-style objective, not just another
  small row-count increase.

Qwen3-235B 2048-row CAT/D-PACE training completion, 2026-06-06 05:16 PDT:

- Public implementation check:
  - Shallow-cloned `https://github.com/AMD-AGI/PARD` at
    `77eee0a12a729aaa4cc38b2a30fd544e11a8173b`.
  - Browser re-check on 2026-06-06 matched the same public status: the AMD
    PARD README lists PARD-2, but its update log still says PARD-2 code and
    checkpoints will be released soon.
  - The current repo tree only exposes the existing PARD trainer/runtime
    (`pard/train.py`, `pard/pard_train.py`, config examples, and vLLM helper).
    There is no official PARD-2 CAT/D-PACE implementation file to run yet.
  - vLLM current docs continue to expose PARD as draft-model speculative
    decoding with `parallel_drafting=true`, matching our standalone gate path.
- CAT/TPP 2048-row train:
  - job `3189439` completed, elapsed `00:04:45`, exit `0:0`.
  - teacher data: 16 x 128 rows, total `2048` rows, train data size `219M`.
  - checkpoint ready:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_cat_tpp_mask_2048_16x128_20260606_042506/checkpoint-32`.
  - train summary: `train_runtime=160.467s`,
    `train_samples_per_second=12.763`, `train_loss=1.093600869178772`.
  - gate job `3189440` is now pending on priority, not dependency.
- D-PACE-target 2048-row train:
  - job `3189721` completed, elapsed `00:05:04`, exit `0:0`.
  - checkpoint ready:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_target_weightedce_2048_20260606_050431/checkpoint-32`.
  - gate job `3189722` is pending; `breakdown.json` not emitted yet.
- Interpretation:
  - The practical training-time estimate is now validated: once teacher data is
    available, 2048-row PARD-style training takes about five minutes on 4 GPUs.
  - The expensive part remains Qwen3-235B teacher continuation/logprob
    generation, not drafter fine-tuning.
  - The next decision point is the two OpenMath bs32 K5 gate results. If CAT/TPP
    and D-PACE-target stay near public PARD's `~44-45%` acceptance and
    `~1.2-1.3x` high-batch speedup, then more rows alone are unlikely to solve
    Qwen3-235B; the objective needs to become closer to true accepted-prefix
    optimization or official PARD-2 once released.

Follow-up job state, 2026-06-06 05:17 PDT:

- Qwen3-235B standalone gates:
  - `3189440` CAT/TPP 2048 K5 gate is pending on priority.
  - `3189722` D-PACE-target 2048 K5 gate is pending on priority.
  - Both dependencies are satisfied; missing artifact is only the final
    `breakdown.json` from the vLLM gate runs.
- NeMo-RL smaller-model PARD validation:
  - Qwen3-32B local-transformer public PARD K5 retry `3189630` is running on
    4 nodes.
  - Qwen3-30B-A3B top-level `moe_backend=triton` baseline `3189658` and public
    PARD K5 `3189659` are running.
  - Qwen3-30B-A3B local CAT/PARD-2-style K5 `3189660` remains pending on
    priority.

Follow-up poll, 2026-06-06 05:26 PDT:

- Qwen3-235B standalone gates:
  - CAT/TPP 2048 K5 gate `3189440` is running, elapsed `00:07:47`.
  - D-PACE-target 2048 K5 gate `3189722` is running, elapsed `00:07:45`.
  - Neither gate has emitted `breakdown.json` yet.
  - CAT/TPP gate startup confirms the intended vLLM path:
    - vLLM engine version reported as `v0.20.2`.
    - target `Qwen/Qwen3-235B-A22B`.
    - `tensor_parallel_size=4`.
    - speculative config uses `method='draft_model'`,
      `num_speculative_tokens=5`, `draft_tensor_parallel_size=4`,
      `parallel_drafting=True`.
    - current MoE backend selection remained `auto` and selected
      FlashInfer TRTLLM unquantized MoE.
- NeMo-RL PARD retries:
  - Qwen3-32B local-transformer public PARD K5 retry `3189630` is running,
    elapsed `00:13:12`. It has installed the driver/actor packages and is in
    setup; no rollout metric yet.
  - Qwen3-30B-A3B baseline `3189658`, public PARD K5 `3189659`, and local
    CAT/PARD-2-style K5 `3189660` are all running.
  - The earlier `Traceback` seen in `3189658` was from the wrapper polling
    `ray status` and hit `AttributeError: 'NoneType' object has no attribute
    'decode'` while extracting actor count. It did not terminate the job; the
    same slurm log later shows `All workers connected!`.
  - No Qwen3-30B-A3B rollout metric or model/runtime failure has been observed
    yet at this poll.

Follow-up poll, 2026-06-06 05:34 PDT:

- Qwen3-235B standalone CAT/TPP 2048 gate `3189440` completed:
  - elapsed `00:12:09`, exit `0:0`.
  - vLLM engine `v0.20.2`, target `Qwen/Qwen3-235B-A22B`, target `TP=4`,
    draft `TP=4`, `parallel_drafting=True`, K5.
  - OpenMath held-out gate shape: `ISL=1024`, `OSL=1024`, `bs=32`,
    `max_num_batched_tokens=131072`.
  - result: `584.42` output tok/s/GPU, `1.207x` vs OpenMath baseline,
    `1.206x` vs current public-PARD recal, acceptance `45.40%`,
    mean acceptance length `3.270`.
  - per-position acceptance: `75.12%`, `55.31%`, `41.04%`, `31.34%`,
    `24.18%`.
- Qwen3-235B standalone D-PACE-target 2048 gate `3189722` completed:
  - elapsed `00:09:34`, exit `0:0`.
  - same OpenMath `ISL=1024`, `OSL=1024`, `bs=32`, target/draft `TP=4`
    K5 gate shape.
  - result: `511.58` output tok/s/GPU, `1.057x` vs OpenMath baseline,
    acceptance `45.46%`, mean acceptance length `3.273`.

Current Qwen3-235B local drafter gate comparison:

| Checkpoint | Train rows | Job | Output tok/s/GPU | Speedup vs baseline | Speedup vs current public recal | Acceptance | Mean acceptance length | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| public PARD current-harness recal | 0 | `3171868` | `484.48` | `1.001x` | `1.000x` | `45.99%` | `3.299` | reference |
| 1K CAT/TPP mask | 1024 | `3174175` | `543.08` | `1.122x` | `1.121x` | `46.71%` | `3.336` | modest current-harness win |
| 2K CAT/TPP mask | 2048 | `3189440` | `584.42` | `1.207x` | `1.206x` | `45.40%` | `3.270` | best local current-harness gate, but acceptance did not improve |
| 2K D-PACE target weighted CE | 2048 | `3189722` | `511.58` | `1.057x` | `1.056x` | `45.46%` | `3.273` | do not promote |

Training scale evidence from the completed 2K run:

| Stage | Observed time | Notes |
|---|---:|---|
| 128-row Qwen3-235B teacher chunks | `34-39 min` each | 8 chunks ran in parallel for the 1024-row expansion; each chunk used target `TP=4` |
| 2048-row merged train data | `219M` | 16 x 128-row teacher-logprob chunks |
| 2048-row CAT/TPP train | `4m45s` elapsed, `160.467s` trainer runtime | 4 GPUs, 64 optimizer steps, `train_loss=1.0936` |
| 2048-row D-PACE target train | `5m04s` elapsed | same 2048 teacher data reused |
| OpenMath bs32 gate | `9-12 min` | target/draft `TP=4`, K5 |

Interpretation:

- The practical bottleneck is Qwen3-235B teacher continuation/logprob
  generation, not 0.6B drafter fine-tuning.
- Scaling CAT/TPP from 1K to 2K improved current-harness throughput from
  `1.122x` to `1.207x`, but acceptance fell from `46.71%` to `45.40%`.
  This is a throughput/runtime effect, not evidence that the local objective
  learned a better accepted-prefix drafter.
- D-PACE-target weighted CE did not improve either acceptance or throughput.
  Do not spend more large-scale training budget on this exact loss.
- More rows alone are unlikely to solve Qwen3-235B. The next training-side
  improvement should change the objective: closer official PARD-2 CAT when
  released, a direct accepted-prefix/LK-style loss, or a better confidence
  weighting that targets realized multi-token acceptance rather than target
  probability alone.

NeMo-RL retry state at the same poll:

- Qwen3-32B public PARD local-transformer retry `3189630` is running on 4
  nodes, elapsed `00:23:08`, still in setup/env-builder stage with no rollout
  metric yet.
- Qwen3-30B-A3B top-level `moe_backend=triton` jobs remain running:
  baseline `3189658`, public PARD `3189659`, local CAT/PARD-2-style `3189660`.
  They have emitted config/log-dir evidence but no rollout metric yet.

Follow-up poll, 2026-06-06 05:49 PDT:

- Qwen3-235B dynamic D-PACE/PARD-2-style draft-probability run is active:
  - train job `3190562` is running, elapsed `00:03:40`, exit `0:0`.
  - gate job `3190567` is dependency-pending on train success.
  - objective: `cat_loss_mode=dpace_draft_ce`, smoothing `0.5`, K5 PARD
    parallel drafting, using the same 2048 OpenMath teacher rows.
  - the trainer reached forward/backward and logged first metrics at step
    `20/64`: `loss=5.7968`, `grad_norm=11.0830`, `learning_rate=3e-6`.
  - checkpoint `checkpoint-64` is not available yet; gate
    `breakdown.json` is not available yet.
- This dynamic loss is closer to the PARD-2/D-PACE direction than the failed
  D-PACE-target surrogate because it computes acceptance-length weights from
  the current draft probability `q_i`, then applies detached accepted-prefix
  weights to CE. The previous D-PACE-target row used target probability only;
  D-PACE explicitly warns that this surrogate is weak because it is independent
  of the draft distribution.

NeMo-RL retry failure classification:

| Model/run | Job | Final state | Root cause marker | Performance conclusion |
|---|---:|---|---|---|
| Qwen3-32B public PARD K5 local-transformer retry | `3189630` | failed after `00:24:30` | `AssertionError: sequence parallel not supported by torch LayerNorm` during `MegatronPolicyWorker.__init__` | no rollout metric; not a PARD throughput result |
| Qwen3-30B-A3B baseline | `3189658` | failed after `00:28:12` | TransformerEngine RMSNorm CUDA invalid argument during Megatron policy `model_forward` | baseline failed too, so no specdec comparison |
| Qwen3-30B-A3B public PARD K5 | `3189659` | failed after `00:25:17` | same TransformerEngine RMSNorm CUDA invalid argument | setup/runtime failure, not PARD quality |
| Qwen3-30B-A3B local CAT/PARD-2-style K5 | `3189660` | failed after `00:25:20` | same TransformerEngine RMSNorm CUDA invalid argument | setup/runtime failure, not PARD quality |

Interpretation:

- The smaller-model NeMo-RL retries did not produce generation/E2E speedup
  evidence. Qwen3-32B died while constructing Megatron policy workers with the
  local LayerNorm path; Qwen3-30B-A3B died in Megatron forward even for the
  baseline.
- For Qwen3-30B-A3B, the next NeMo-RL action should fix or avoid the
  TransformerEngine RMSNorm path before comparing PARD variants. Since the
  baseline failed, resubmitting more PARD variants on the same setup is not
  informative.
- For Qwen3-235B, the current actionable path remains standalone PARD/PARD-2
  objective quality first, then NeMo-RL integration once a standalone gate
  clearly beats public PARD on OpenMath.

Follow-up poll, 2026-06-06 05:55 PDT:

- Qwen3-235B dynamic D-PACE/PARD-2-style draft-probability train completed:
  - job `3190562`, elapsed `00:04:50`, exit `0:0`.
  - checkpoint:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_2048_20260606_054509/checkpoint-64`.
  - train summary: `train_runtime=162.3726s`,
    `train_samples_per_second=12.613`,
    `train_steps_per_second=0.394`, `train_loss=5.469364166259766`.
  - training was slower per effective step than mask-mode CAT only because the
    dynamic loss computes current draft probabilities and accepted-prefix
    products inside the trainer. Wall-clock remained about five minutes.
- Gate job `3190567` is now dependency-free and pending only on Slurm priority.
  Output JSON is expected at:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard2_dpace_draft_2048_20260606_054509_openmath_isl1024_osl1024/breakdown.json`.
- Updated artifacts:
  - CSV row `local_pard_k5_dpace_draft_ce_2048_train` is marked completed in
    `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`.
  - `scripts/plot_qwen3_235b_pard_followup.py` now includes the dynamic
    D-PACE gate automatically once the gate row has numeric throughput. Pending
    rows are skipped, so the current PNG still shows only the completed public,
    1K CAT/TPP, 2K CAT/TPP, and 2K D-PACE-target bars.

Follow-up poll, 2026-06-06 06:07 PDT:

- Qwen3-235B dynamic D-PACE/PARD-2-style draft-probability gate completed:
  - gate job `3190567`, elapsed `00:11:46`, exit `0:0`.
  - vLLM engine `v0.20.2`, target `Qwen/Qwen3-235B-A22B`, target `TP=4`,
    draft `TP=4`, K5, `parallel_drafting=True`.
  - OpenMath held-out gate shape: `ISL=1024`, `OSL=1024`, `bs=32`,
    `max_num_batched_tokens=131072`, `max_model_len=4096`.
  - result: `627.1401420985972` output tok/s/GPU,
    `1.2955111767006888x` vs OpenMath baseline,
    `1.294453366213392x` vs current public-PARD recal,
    `0.9866107948184755x` vs the historical public-PARD reference.
  - acceptance `47.0078659720094%`, mean acceptance length
    `3.35039329860047`, accepted tokens per draft `2.35039329860047`.
  - per-position acceptance: `76.25%`, `57.24%`, `42.76%`, `32.90%`,
    `25.89%`.
  - final JSON:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard2_dpace_draft_2048_20260606_054509_openmath_isl1024_osl1024/breakdown.json`.

Updated Qwen3-235B local drafter gate comparison:

| Checkpoint | Train rows | Job | Output tok/s/GPU | Speedup vs baseline | Speedup vs current public recal | Ratio vs historical public | Acceptance | Mean acceptance length | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| public PARD current-harness recal | 0 | `3171868` | `484.48` | `1.001x` | `1.000x` | `0.762x` | `45.99%` | `3.299` | current-harness reference |
| 1K CAT/TPP mask | 1024 | `3174175` | `543.08` | `1.122x` | `1.121x` | `0.854x` | `46.71%` | `3.336` | modest current-harness win |
| 2K CAT/TPP mask | 2048 | `3189440` | `584.42` | `1.207x` | `1.206x` | `0.919x` | `45.40%` | `3.270` | throughput improved, acceptance did not |
| 2K D-PACE target weighted CE | 2048 | `3189722` | `511.58` | `1.057x` | `1.056x` | `0.805x` | `45.46%` | `3.273` | do not promote |
| 2K D-PACE draft-prob CE | 2048 | `3190567` | `627.14` | `1.296x` | `1.294x` | `0.987x` | `47.01%` | `3.350` | best local current-harness gate; promote to NeMo-RL test |

Interpretation:

- This is the first local Qwen3-235B PARD/PARD-2-style result that improves
  both current-harness throughput and acceptance over the current public-PARD
  recalibration. It is still slightly below the older historical public-PARD
  throughput row, so the fair claim is not "beats all public PARD results";
  the fair claim is "beats the current public-PARD rerun on the same harness
  by about `1.294x` and nearly recovers the historical public-PARD ceiling."
- The result supports the D-PACE/PARD-2 direction: using the current drafter's
  `q_i` to weight accepted-prefix CE is better than using target probability
  alone. The D-PACE-target surrogate stayed around `45.5%` acceptance and only
  `1.057x` baseline speedup; draft-probability D-PACE reached `47.0%`
  acceptance and `1.296x` baseline speedup.
- The next required step is NeMo-RL validation with this exact checkpoint:
  first a VllmGeneration/direct gate to isolate rollout behavior, then a
  full-GRPO run if the direct gate is healthy. Qwen3-30B-A3B and Qwen3-32B
  NeMo failures remain separate setup issues and should not be mixed with this
  Qwen3-235B standalone result.

Follow-up submission, 2026-06-06 06:13 PDT:

- Submitted the exact dynamic D-PACE/PARD-2-style checkpoint into the NeMo-RL
  VllmGeneration sync smoke path:
  - baseline job `3192176`
  - dynamic D-PACE K5 job `3192177`
  - checkpoint:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_2048_20260606_054509/checkpoint-64`
  - shape: `1n4g`, target `TP=4`, draft `TP=4`, sync vLLM engine,
    batch `32`, `max_new_tokens=256`, natural EOS, `temperature=1.0`,
    `top_p=1.0`, `top_k=-1`, K5, `parallel_drafting=true`.
  - job file:
    `latest_qwen235b_pard2_dpace_nemorl_vllmgeneration_sync_smoke_jobs.txt`.
  - first poll: both jobs are running; JSON outputs are not emitted yet.
- Also submitted a 32-node Full-GRPO step4 validation job with the same dynamic
  D-PACE checkpoint:
  - dynamic K5 job `3192180`
  - matched baseline job is the existing step4 baseline `3186510`
  - shape: `32n4g`, `generation_tp=4`, `TRAIN_GLOBAL_BATCH_SIZE=256`,
    `NUM_PROMPTS=8`, `NUM_GENERATIONS=32`, `MAX_STEPS=4`,
    natural EOS, `max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`,
    `top_k=-1`, `STOP_AFTER_GENERATION=false`.
  - job file:
    `latest_qwen235b_pard2_dpace_nemorl_fullgrpo_sampling_step4_jobs.txt`.
  - first poll: dynamic Full-GRPO job is Slurm priority-pending.
- New local reproducibility scripts:
  - `experiments/eagle3_qwen3_235b/submit_qwen235b_pard2_dpace_nemorl_vllmgeneration_sync_smoke.sh`
  - `scripts/poll_qwen235b_pard2_dpace_nemorl_sync_status.sh`
  - `experiments/eagle3_online/submit_qwen235b_pard2_dpace_fullgrpo_sampling_step4.sh`

Follow-up poll, 2026-06-06 06:17 PDT:

- VllmGeneration sync baseline `3192176` is still running and has reached
  target checkpoint loading. Log marker: `Loading safetensors checkpoint shards`
  for `Qwen/Qwen3-235B-A22B`.
- Initial dynamic D-PACE K5 sync job `3192177` failed before model load or
  speculative decoding:
  - JSON status: `fail`
  - error type: `LocalRayletDiedError`
  - root-cause marker: `runtime_env_agent timed out in 30000ms`, then the
    node was marked dead.
  - affected node: `nvl72165-T13`
  - conclusion: bootstrap/node runtime failure, not a drafter acceptance or
    throughput result.
- Submitted dynamic-only retry `3192211` with
  `SBATCH_EXTRA_ARGS=--exclude=nvl72165-T13`; it is pending at first poll.
- Updated the VllmGeneration sync submit wrapper so `SBATCH_EXTRA_ARGS` is
  passed through to the underlying Slurm submission.

Follow-up poll, 2026-06-06 06:18 PDT:

- Dynamic-only retry `3192211` started on `nvl72069-T09` and is running.
- Matched sync baseline `3192176` is still running on `nvl72094-T17`; target
  model load reached shard `79/118`.
- Full-GRPO dynamic step4 job `3192180` remains priority-pending.
- Added `docs/qwen3_235b_nemorl_dpace_validation_20260606.csv` to track these
  NeMo-RL dynamic D-PACE validation rows.

Follow-up poll, 2026-06-06 06:23 PDT:

- Matched VllmGeneration sync baseline `3192176` emitted a pass JSON:
  - generated tokens: `8192`
  - generation elapsed: `31.789949417114258s`
  - generation throughput: `257.6915078571909 tok/s`
  - acceptance fields are zero as expected for the no-draft baseline.
- Dynamic retry `3192211` is running and has reached target checkpoint loading
  on `nvl72069-T09`.

Follow-up poll, 2026-06-06 06:30 PDT:

- Dynamic D-PACE/PARD-2-style K5 retry `3192211` completed successfully:
  - generated tokens: `8192`
  - generation elapsed: `23.04767346382141s`
  - generation throughput: `355.4371773298166 tok/s`
  - matched baseline speedup: `1.3793127305025317x`
  - acceptance: `44.24882629107981%`
  - mean acceptance length: `3.2124413145539905`
  - accepted/draft tokens: `5655 / 12780`
  - num drafts: `2556`
  - per-position acceptance: `74.30%`, `53.91%`, `39.71%`, `29.85%`,
    `23.47%`.
- Interpretation:
  - This is a real NeMo-RL VllmGeneration speedup for the dynamic
    D-PACE/PARD-2-style Qwen3-235B checkpoint, not just standalone vLLM.
  - It is not yet the best NeMo-RL sync smoke row: earlier fixed256 sync rows
    had public PARD K5 at about `1.522x` and local CAT/TPP K5 at about
    `1.502x`. Dynamic D-PACE has higher acceptance than those rows but lower
    throughput, so the next question is not just acceptance quality; it is
    the NeMo-RL/vLLM overhead profile for this checkpoint and fixed256 shape.
  - Full-GRPO validation is still required. Dynamic step4 job `3192180`
    remains Slurm priority-pending.

Follow-up update, 2026-06-06 06:32 PDT:

- Full-GRPO jobs are still Slurm priority-pending:
  - matched baseline `3186510`
  - earlier local CAT/TPP K5 `3186511`
  - dynamic D-PACE K5 `3192180`
- Added a compact NeMo-RL sync fixed256 comparison plot:
  `docs/qwen3_235b_nemorl_sync_dpace_comparison.png`.

Current Qwen3-235B NeMo-RL sync fixed256 comparison:

| Row | Job | Generation speedup | Acceptance | Mean acceptance length | Notes |
|---|---:|---:|---:|---:|---|
| Baseline | `3192176` | `1.000x` | `0.00%` | `1.000` | matched baseline for dynamic D-PACE |
| Public PARD K5 | `3186417` | `1.522x` | `42.29%` | `3.115` | prior sync smoke reference |
| Local CAT/TPP K5 | `3186340` | `1.502x` | `43.77%` | `3.188` | prior local PARD-2-style reference |
| Dynamic D-PACE K3 | `3192349` | `1.454x` | `57.50%` | `2.725` | better throughput than dynamic K5; lower draft cost |
| Dynamic D-PACE K5 | `3192211` | `1.379x` | `44.25%` | `3.212` | higher accepted tokens/draft, but high runtime cost |

Follow-up efficiency analysis, 2026-06-06 06:36 PDT:

- Added JSON-derived efficiency CSV:
  `docs/qwen3_235b_nemorl_sync_efficiency_20260606.csv`.
- Added compact efficiency plot:
  `docs/qwen3_235b_nemorl_sync_efficiency.png`.
- Key observation from the completed NeMo-RL sync fixed256 JSONs:

| Row | Accepted tokens / draft | ms / draft | Throughput | Interpretation |
|---|---:|---:|---:|---|
| Public PARD K5 | `2.115` | `7.94` | `391.49 tok/s` | lowest runtime cost |
| Local CAT/TPP K5 | `2.188` | `8.24` | `386.50 tok/s` | slightly better acceptance efficiency, modestly higher draft cost |
| Dynamic D-PACE K3 | `1.725` | `7.27` | `374.62 tok/s` | lower accepted/draft due to K3, but cheapest draft cycle |
| Dynamic D-PACE K5 | `2.212` | `9.02` | `355.44 tok/s` | best acceptance efficiency, but draft/runtime cost dominates |

- Root-cause implication:
  - Dynamic D-PACE is not losing because acceptance quality is low. It has the
    best accepted tokens per draft and the highest overall acceptance.
  - It is losing NeMo-RL sync throughput because each draft cycle is more
    expensive: `9.02 ms/draft` vs `7.94 ms/draft` for public PARD and
    `8.24 ms/draft` for local CAT/TPP.
  - Reducing dynamic D-PACE from K5 to K3 improved throughput from `1.379x` to
    `1.454x`, because the draft-cycle cost dropped from `9.02 ms/draft` to
    `7.27 ms/draft`. This confirms that K/value scheduling is a useful lever
    for this checkpoint.
  - The next optimization target should be runtime cost of the dynamic local
    checkpoint path or vLLM draft execution profile, not simply more
    acceptance-rate training.
- Architecture/config check:
  - Public PARD, local CAT/TPP, and dynamic D-PACE all have the same
    `config.json` SHA256:
    `9d315ab34541feb4e0f6f29c15ae194a7b5b3b1a90add32882f4c077fa710d16`.
  - They share the same structure: `Qwen3ForCausalLM`, hidden size `1024`,
    `28` layers, `16` attention heads, `8` KV heads, vocab `151936`,
    `bfloat16`.
  - Therefore the dynamic D-PACE runtime gap is not explained by a different
    drafter architecture. It is either weight-dependent draft behavior,
    scheduling shape, or vLLM execution overhead around this local checkpoint.

Follow-up submission and result, 2026-06-06 06:40-06:51 PDT:

- Submitted a dynamic D-PACE K3 NeMo-RL sync fixed256 runtime-cost probe:
  - job `3192349`
  - same checkpoint as K5:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_2048_20260606_054509/checkpoint-64`
  - `NUM_SPECULATIVE_TOKENS=3`, target `TP=4`, draft `TP=4`, sync engine,
    batch `32`, `max_new_tokens=256`.
  - bad node exclusion retained: `SBATCH_EXTRA_ARGS=--exclude=nvl72165-T13`.
  - job file:
    `latest_qwen235b_pard2_dpace_nemorl_vllmgeneration_sync_k3_jobs.txt`.
  - result: pass.
  - generated tokens: `8192`
  - generation elapsed: `21.867478847503662s`
  - generation throughput: `374.6202320408408 tok/s`
  - matched baseline speedup: `1.4537546664069743x`
  - acceptance: `57.50221631205674%`
  - mean acceptance length: `2.7250664893617023`
  - accepted/draft tokens: `5189 / 9024`
  - num drafts: `3008`
  - per-position acceptance: `74.27%`, `55.49%`, `42.75%`.
- Motivation: K5 dynamic D-PACE has the best acceptance efficiency but higher
  `ms/draft`; K3 trades accepted tokens per draft for lower runtime cost and
  improves dynamic D-PACE speedup from `1.379x` to `1.454x`.

Follow-up Full-GRPO submission, 2026-06-06 06:57 PDT:

- Patched `experiments/eagle3_online/submit_qwen235b_pard_local_tpp_mask_gbs256_worker32_step1.sh`
  so `NUM_SPECULATIVE_TOKENS` is configurable instead of hardcoded to `5`.
  Default remains K5, so existing K5 wrappers are unchanged.
- Added K3 Full-GRPO wrapper:
  `experiments/eagle3_online/submit_qwen235b_pard2_dpace_fullgrpo_sampling_step4_k3.sh`.
- Submitted dynamic D-PACE K3 Full-GRPO step4 job:
  - job `3192438`
  - matched baseline remains `3186510`
  - same dynamic checkpoint as K5
  - `NUM_SPECULATIVE_TOKENS=3`
  - `32n4g`, `generation_tp=4`, `TRAIN_GLOBAL_BATCH_SIZE=256`,
    `MAX_STEPS=4`, natural EOS, `max_new_tokens=256`,
    `STOP_AFTER_GENERATION=false`.
  - job file:
    `latest_qwen235b_pard2_dpace_k3_nemorl_fullgrpo_sampling_step4_jobs.txt`.
  - first poll: Slurm priority-pending.

Current Full-GRPO queue:

| Row | Job | State | Purpose |
|---|---:|---|---|
| Baseline | `3186510` | pending | matched E2E baseline |
| Local CAT/TPP K5 | `3186511` | pending | prior local PARD-2-style reference |
| Dynamic D-PACE K5 | `3192180` | pending | original dynamic full-GRPO validation |
| Dynamic D-PACE K3 | `3192438` | pending | K3 runtime-cost-optimized validation |

Queue poll, 2026-06-06 07:02 PDT:

- All four 32-node Qwen3-235B Full-GRPO validation jobs are still
  `PENDING (Priority)`.
- Slurm currently reports the same candidate start time for the matrix:
  `2026-06-06T11:05:20` PDT, with a 4 hour time limit.
- No `ray-driver.log` exists yet for any of the four jobs, so there are no
  Full-GRPO generation/E2E metrics to parse.
- Added and validated a reusable poll/parser script for this exact matrix:
  `scripts/poll_qwen235b_pard2_dpace_fullgrpo_step4_status.sh`.

| Row | Job | State | Candidate start | Nodes | Notes |
|---|---:|---|---|---:|---|
| Baseline | `3186510` | pending, Priority | `2026-06-06T11:05:20` | `32` | matched E2E baseline |
| Local CAT/TPP K5 | `3186511` | pending, Priority | `2026-06-06T11:05:20` | `32` | prior local PARD-2-style reference |
| Dynamic D-PACE K5 | `3192180` | pending, Priority | `2026-06-06T11:05:20` | `32` | original dynamic validation |
| Dynamic D-PACE K3 | `3192438` | pending, Priority | `2026-06-06T11:05:20` | `32` | K3 runtime-cost-optimized validation |

External method scan, 2026-06-06:

- AMD PARD repo still lists PARD-2 code and checkpoints as "released soon";
  therefore there is no official public PARD-2 checkpoint to drop into vLLM or
  NeMo-RL yet. The local CAT/D-PACE trainer remains an approximation, not the
  official PARD-2 implementation.
- Official vLLM docs now include `parallel_drafting=true` PARD examples with
  `amd/PARD-Qwen3-0.6B`, which matches our current standalone and NeMo-RL
  runtime path.
- vLLM Speculators v0.5.0 added DFlash training and online/offline hidden-state
  extraction support, but our Qwen3-235B DFlash checkpoint gate showed near-zero
  OpenMath acceptance. DFlash remains a training-quality problem, not a runtime
  blocker.
- SPECTRE is the most relevant newly found systems paper because it reports
  Qwen3-235B-A22B TP=8 results up to `2.28x` over autoregressive decoding by
  overlapping remote drafting and target verification. It is implemented in
  SGLang, so it is not an immediate vLLM/NeMo-RL drop-in, but it directly
  supports the root-cause hypothesis that target/draft overlap and draft-side
  scheduling can matter more than raw acceptance on Qwen3-235B.

Sources:

- AMD PARD repo: https://github.com/AMD-AGI/PARD
- PARD-2 paper: https://arxiv.org/abs/2605.08632
- vLLM PARD docs:
  https://docs.vllm.ai/en/latest/features/speculative_decoding/parallel_draft_model/
- vLLM Speculators DFlash/online training:
  https://vllm.ai/blog/2026-05-28-speculators-v050
- SPECTRE: https://arxiv.org/abs/2605.08151

4K math-domain D-PACE expansion, 2026-06-06 07:10-07:25 PDT:

- Motivation:
  - The 2K D-PACE checkpoint produced a real NeMo-RL sync signal, but it did
    not beat public PARD on the fixed256 sync gate.
  - If the gap is domain adaptation / objective quality, increasing the
    held-out-nonoverlapping math teacher rows should improve OpenMath
    acceptance and/or throughput.
  - If the gap is runtime overhead, more rows may improve acceptance but still
    fail to improve speedup, which would push the next fix toward draft runtime
    scheduling rather than more training.
- Added a reusable 4K expansion wrapper:
  `experiments/pard_qwen3_235b_math/submit_qwen235b_pard2_dpace_draft_4096_expand.sh`.
- The wrapper reuses the completed 16 chunks at offsets `5000..6920` and
  submits only the missing 16 chunks at offsets `7048..8968`, avoiding a
  duplicate 2K teacher pass.
- Submitted chain:
  - teacher chunks:
    `3192508`, `3192509`, `3192516`, `3192538`, `3192544`, `3192545`,
    `3192551`, `3192573`, `3192576`, `3192577`, `3192578`, `3192579`,
    `3192581`, `3192602`, `3192604`, `3192610`
  - train job: `3192611`
  - standalone OpenMath K5 gate: `3192612`
  - standalone OpenMath K3 gate: `3192613`
- Training config:
  - train rows: `4096`
  - objective: `cat_loss_mode=dpace_draft_ce`
  - smoothing: `0.5`
  - base drafter: `amd/PARD-Qwen3-0.6B`
  - `para_num=5`, PARD token id `151670`
  - `TRAIN_MAX_SEQ_LENGTH=2048`
  - `TRAIN_GRAD_ACCUM=8`
  - `TRAIN_LR=3e-6`
  - `TRAIN_SAVE_STEPS=128`
  - target checkpoint:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_4096_20260606_070951/checkpoint-128`
- Gate config:
  - target `Qwen/Qwen3-235B-A22B`
  - held-out OpenMath prompt file:
    `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/openmath_reasoning_cot_conversations_50k.jsonl`
  - `ISL=1024`, `OSL=1024`, batch `32`
  - target `TP=4`, draft `TP=4`
  - `parallel_drafting=true`
  - K5 and K3 are both submitted against the same 4K checkpoint.
- First queue poll:
  - most teacher jobs are already `RUNNING`
  - teacher jobs `3192604` and `3192610` are still `PENDING (Priority)`
  - train job `3192611` is `PENDING (Dependency)`
  - gates `3192612` and `3192613` are `PENDING (Dependency)`
- Job file:
  `latest_qwen235b_pard2_dpace_draft_4096_jobs.txt`.
- Added and validated a reusable chain poll script:
  `scripts/poll_qwen235b_pard2_dpace_4096_status.sh`.
- Follow-up poll at 2026-06-06 07:26 PDT:
  - all 16 teacher jobs are now `RUNNING`
  - train `3192611` remains `PENDING (Dependency)`
  - K5 gate `3192612` remains `PENDING (Dependency)`
  - K3 gate `3192613` remains `PENDING (Dependency)`
  - first five teacher output files had already started writing `8` rows each;
    later chunks were still empty/missing, as expected early in generation.

Follow-up poll, 2026-06-06 07:28 PDT:

- Full-GRPO step4 matrix is still waiting for allocation:
  - baseline `3186510`: `PENDING (Priority)`
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`
  - candidate start shifted to `2026-06-06T11:26:10` PDT
  - no `ray-driver.log` exists yet, so no Full-GRPO generation/E2E metrics are
    available.
- 4K D-PACE chain remains healthy:
  - all 16 new teacher chunks are `RUNNING`
  - train `3192611` remains `PENDING (Dependency)`
  - gates `3192612` and `3192613` remain `PENDING (Dependency)`
  - early row-count snapshot:
    - offsets `7048`, `7176`, and `7560`: `16` rows each
    - offsets `7304` and `7432`: `8` rows each
    - later chunks are still empty or not yet materialized
  - assembled 4K train JSON and gate JSONs are not expected to exist until all
    teacher chunks finish and train/gate dependencies run.
- Added pending rows for `3192611`, `3192612`, and `3192613` to
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`; CSV validation
  passes with `34` rows and `11` fields.

Follow-up poll, 2026-06-06 07:31 PDT:

- Full-GRPO step4 matrix is still allocation-pending; no driver logs exist:
  - baseline `3186510`: `PENDING (Priority)`, candidate start
    `2026-06-06T11:01:30` PDT
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`, candidate start
    `2026-06-06T11:30:22` PDT
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`, candidate start
    `2026-06-06T11:30:22` PDT
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`, candidate start
    `2026-06-06T11:30:22` PDT
- 4K D-PACE chain is still healthy:
  - all 16 new teacher chunks remain `RUNNING`
  - train `3192611` is still `PENDING (Dependency)`
  - K5 gate `3192612` and K3 gate `3192613` are still
    `PENDING (Dependency)`
  - row progress now includes:
    - offsets `7048`, `7176`, `7432`, `7560`: `32` rows
    - offsets `7304`, `7688`, `7816`: `24` rows
    - later chunks are still empty or not materialized yet.
- Fixed `scripts/poll_qwen235b_pard2_dpace_4096_status.sh` so completed gate
  JSONs are parsed from the actual `results[*].spec_decode_metrics` structure.
  Validation against the completed 2K D-PACE gate JSON reads:
  `bs=32`, `tok_s_gpu=627.1401420985972`,
  `acceptance=0.470078659720094`, `mean_accept_len=3.35039329860047`.

Follow-up poll, 2026-06-06 07:33 PDT:

- Full-GRPO step4 matrix is still `PENDING (Priority)` with no driver logs:
  `3186510`, `3186511`, `3192180`, `3192438`.
- 4K D-PACE teacher generation is still progressing normally:
  - all 16 new teacher jobs remain `RUNNING`
  - train `3192611` and gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
  - row-count progress:
    - offsets `7048`, `7176`, `7304`, `7432`, `7560`: `40` rows each
    - offsets `7688`, `7816`: `32` rows each
    - offsets `8072`, `8200`: file exists at `0` rows
    - later chunks are still not materialized yet.
- No 4K train JSON or gate `breakdown.json` exists yet, as expected before all
  teacher chunks finish.

Follow-up poll, 2026-06-06 07:37 PDT:

- Full-GRPO step4 matrix remains allocation-pending, with no driver logs or
  parsed generation/E2E metrics:
  - baseline `3186510`: `PENDING (Priority)`
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`
- 4K D-PACE teacher chain remains healthy:
  - all 16 new teacher chunks are `RUNNING`
  - train `3192611` remains `PENDING (Dependency)`
  - K5 gate `3192612` and K3 gate `3192613` remain
    `PENDING (Dependency)`
  - row-count snapshot:
    - offset `7048`: `56` rows
    - offset `7176`: `64` rows
    - offsets `7304`, `7560`: `56` rows
    - offset `7432`: `57` rows
    - offsets `7688`, `7816`: `48` rows
    - offset `7944`: `8` rows
    - offsets `8072`, `8200`: `24` rows
    - offsets `8328`, `8456`, `8584`, `8712`: `8` rows
    - offsets `8840`, `8968`: file exists at `0` rows
  - assembled 4K train JSON and both gate `breakdown.json` files are still
    missing, as expected before dependencies complete.
- Inspected logs for the slow or empty late-offset chunks
  (`3192573`, `3192576`, `3192577`, `3192578`, `3192579`, `3192581`,
  `3192602`, `3192604`, `3192610`). No traceback, CUDA OOM, vLLM engine
  failure, or schema error was present. The only repeated warning is the known
  pip resolver warning around `flashinfer-python 0.6.5` vs precompiled vLLM
  `0.17.0`; these jobs still reached "vLLM server is ready" and generation.
  Offsets `8840` and `8968` are simply waiting for the first completed prompt
  write after server startup.

Follow-up poll, 2026-06-06 07:40-07:41 PDT:

- Enhanced `scripts/poll_qwen235b_pard2_dpace_4096_status.sh` to print an
  aggregate teacher-row total, expected row count, percent completion, and
  missing-file count. This avoids manual summing during the 4K expansion.
- 4K D-PACE teacher chain continues to progress with no missing chunk files:
  - new expansion teacher rows: `779/2048` complete (`38.0%`)
  - combined logical training set progress: existing `2048` rows plus `779`
    new rows, or `2827/4096` rows if all new chunks finish without skips
  - train `3192611` remains `PENDING (Dependency)`
  - K5 gate `3192612` and K3 gate `3192613` remain
    `PENDING (Dependency)`
  - no assembled 4K train JSON or gate `breakdown.json` exists yet.
- Full-GRPO step4 matrix remains allocation-pending, but Slurm now reports a
  concrete planned start time:
  - baseline `3186510`: scheduled start `2026-06-06T11:38:55` PDT
  - local CAT/TPP K5 `3186511`: scheduled start `2026-06-06T11:38:55` PDT
  - dynamic D-PACE K5 `3192180`: scheduled start `2026-06-06T11:38:55` PDT
  - dynamic D-PACE K3 `3192438`: scheduled start `2026-06-06T11:38:55` PDT
  - no `ray-driver.log` exists yet, so no generation/E2E metric has been
    emitted.

Follow-up poll, 2026-06-06 07:42 PDT:

- 4K D-PACE expansion is still progressing and no failure is visible:
  - all 16 new teacher chunks remain `RUNNING`
  - new expansion teacher rows: `875/2048` complete (`42.7%`)
  - combined logical training set progress: existing `2048` rows plus `875`
    new rows, or `2923/4096` rows before any final skip accounting
  - no missing teacher files
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gates `3192612`/`3192613` remain `PENDING (Dependency)`
  - assembled 4K train JSON and gate JSONs are still absent, as expected.
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs yet.
  Slurm still reports a concrete planned start, now
  `2026-06-06T11:41:09` PDT for all four jobs:
  baseline `3186510`, local CAT/TPP K5 `3186511`, dynamic D-PACE K5
  `3192180`, and dynamic D-PACE K3 `3192438`.

Follow-up poll, 2026-06-06 07:44 PDT:

- 4K D-PACE teacher generation is still healthy and close to halfway through
  the new 2K expansion:
  - all 16 new teacher chunks remain `RUNNING`
  - new expansion teacher rows: `985/2048` complete (`48.1%`)
  - combined logical training set progress: existing `2048` rows plus `985`
    new rows, or `3033/4096` rows before final skip accounting
  - no missing teacher files
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gates `3192612`/`3192613` remain `PENDING (Dependency)`
  - no assembled 4K train JSON or gate JSONs exist yet.
- Full-GRPO step4 matrix remains pending with no driver logs. Slurm's planned
  start moved slightly to `2026-06-06T11:43:09` PDT for all four jobs.

Follow-up poll, 2026-06-06 07:45 PDT:

- 4K D-PACE teacher expansion has crossed the halfway point:
  - all 16 new teacher chunks remain `RUNNING`
  - new expansion teacher rows: `1081/2048` complete (`52.8%`)
  - combined logical training set progress: existing `2048` rows plus `1081`
    new rows, or `3129/4096` rows before final skip accounting
  - no missing teacher files
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gates `3192612`/`3192613` remain `PENDING (Dependency)`
  - no assembled 4K train JSON or gate JSONs exist yet.
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Planned start is now `2026-06-06T11:44:40` PDT for all four jobs.

Follow-up poll, 2026-06-06 07:46 PDT:

- 4K D-PACE teacher expansion continues without visible failure:
  - all 16 new teacher chunks remain `RUNNING`
  - new expansion teacher rows: `1171/2048` complete (`57.2%`)
  - combined logical training set progress: existing `2048` rows plus `1171`
    new rows, or `3219/4096` rows before final skip accounting
  - no missing teacher files
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gates `3192612`/`3192613` remain `PENDING (Dependency)`
  - no assembled 4K train JSON or gate JSONs exist yet.
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Planned start remains `2026-06-06T11:44:40` PDT for all four jobs.

Follow-up poll, 2026-06-06 07:48 PDT:

- 4K D-PACE teacher expansion continues to make steady progress:
  - new expansion teacher rows: `1336/2048` complete (`65.2%`)
  - combined logical training set progress: existing `2048` rows plus `1336`
    new rows, or `3384/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm now reports `2026-06-06T11:24:46` PDT for baseline/local CAT-TPP
  (`3186510`, `3186511`) and `2026-06-06T11:47:59` PDT for dynamic D-PACE
  K5/K3 (`3192180`, `3192438`).

Follow-up poll, 2026-06-06 07:50 PDT:

- 4K D-PACE teacher expansion continues normally:
  - new expansion teacher rows: `1464/2048` complete (`71.5%`)
  - combined logical training set progress: existing `2048` rows plus `1464`
    new rows, or `3512/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Baseline/local CAT-TPP are still planned for `2026-06-06T11:24:46` PDT;
  dynamic D-PACE K5/K3 are still planned for `2026-06-06T11:47:59` PDT.

Follow-up poll, 2026-06-06 07:51 PDT:

- 4K D-PACE teacher expansion is still progressing:
  - new expansion teacher rows: `1547/2048` complete (`75.5%`)
  - combined logical training set progress: existing `2048` rows plus `1547`
    new rows, or `3595/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm no longer reports concrete start times for the four jobs at this poll;
  all show `StartTime=Unknown`.

Follow-up poll, 2026-06-06 07:52 PDT:

- 4K D-PACE teacher expansion crossed `80%`:
  - new expansion teacher rows: `1649/2048` complete (`80.5%`)
  - combined logical training set progress: existing `2048` rows plus `1649`
    new rows, or `3697/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm again reports planned start times for all four jobs at
  `2026-06-06T11:51:26` PDT.

Follow-up poll, 2026-06-06 07:54 PDT:

- 4K D-PACE teacher expansion is in the final stretch:
  - new expansion teacher rows: `1698/2048` complete (`82.9%`)
  - combined logical training set progress: existing `2048` rows plus `1698`
    new rows, or `3746/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm planned start time moved to `2026-06-06T11:54:02` PDT for all four
  jobs.

Follow-up poll, 2026-06-06 07:55 PDT:

- 4K D-PACE teacher expansion continues in the final stretch:
  - new expansion teacher rows: `1752/2048` complete (`85.5%`)
  - combined logical training set progress: existing `2048` rows plus `1752`
    new rows, or `3800/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm still reports planned start `2026-06-06T11:54:02` PDT for all four
  jobs.

Follow-up poll, 2026-06-06 07:56 PDT:

- 4K D-PACE teacher expansion remains healthy and is nearing completion:
  - new expansion teacher rows: `1809/2048` complete (`88.3%`)
  - combined logical training set progress: existing `2048` rows plus `1809`
    new rows, or `3857/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm planned start time moved slightly to `2026-06-06T11:55:54` PDT for all
  four jobs.

Follow-up poll, 2026-06-06 07:58 PDT:

- 4K D-PACE teacher expansion is nearly complete:
  - new expansion teacher rows: `1881/2048` complete (`91.8%`)
  - combined logical training set progress: existing `2048` rows plus `1881`
    new rows, or `3929/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm returned to `StartTime=Unknown` for all four jobs at this poll.

Follow-up poll, 2026-06-06 07:59 PDT:

- 4K D-PACE teacher expansion is almost complete:
  - new expansion teacher rows: `1938/2048` complete (`94.6%`)
  - combined logical training set progress: existing `2048` rows plus `1938`
    new rows, or `3986/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm now reports planned start `2026-06-06T11:57:59` PDT for all four jobs.

Follow-up poll, 2026-06-06 08:00 PDT:

- 4K D-PACE teacher expansion is effectively at the tail:
  - new expansion teacher rows: `1978/2048` complete (`96.6%`)
  - combined logical training set progress: existing `2048` rows plus `1978`
    new rows, or `4026/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Slow-tail log check covered offsets `7944`, `8584`, `8712`, `8840`, and
  `8968`. All were still writing rows; no traceback, CUDA OOM, vLLM server
  failure, or schema error was present. No resubmission is warranted.
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.
  Slurm planned start time moved to `2026-06-06T11:59:58` PDT for all four
  jobs.

Follow-up poll, 2026-06-06 08:02 PDT:

- 4K D-PACE teacher expansion is one tail batch from completion:
  - new expansion teacher rows: `2032/2048` complete (`99.2%`)
  - combined logical training set progress: existing `2048` rows plus `2032`
    new rows, or `4080/4096` rows before final skip accounting
  - no missing teacher files
  - assembled 4K train JSON is still absent
  - train `3192611` remains `PENDING (Dependency)`
  - K5/K3 gate JSONs are still absent; gates `3192612`/`3192613` remain
    `PENDING (Dependency)`
- Full-GRPO step4 matrix remains `PENDING (Priority)` with no driver logs.

Follow-up poll, 2026-06-06 08:06-08:07 PDT:

- 4K D-PACE teacher expansion completed cleanly:
  - all 16 new teacher chunk jobs completed with Slurm exit `0:0`
  - all new chunk files have `128` rows, for `2048/2048` new rows
  - the reused existing offset `5000..6920` chunks were rechecked and each has
    `128` rows, so the raw teacher pool is `4096` rows total
- The train job `3192611` moved from dependency/pending to `RUNNING` on
  `nvl72036-T15`.
- Train assembly is confirmed healthy from the training log:
  - output train JSON:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/data/pard_train_math_teacher_logprobs_4096_dpace_draft_offset5000_20260606_070951.jsonl`
  - `written=4096`, `skipped=0`
  - effective training JSON:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/runs/qwen235b_math_k5_dpace_draft_ce_4096_20260606_070951_20260606_072353/pard_train_math_first_4096.jsonl`
- The earlier poll that showed `3250` assembled rows was a partial write during
  train setup, not a data loss condition.
- K5 and K3 OpenMath gates remain queued on `afterok:3192611`:
  - K5 gate `3192612`
  - K3 gate `3192613`
- Full-GRPO step4 matrix remains allocation-pending with no ray-driver logs:
  - baseline `3186510`
  - local CAT/TPP K5 `3186511`
  - dynamic D-PACE K5 `3192180`
  - dynamic D-PACE K3 `3192438`
  - Slurm currently reports planned start around `2026-06-06T12:05:24` PDT for
    all four jobs.
  Slurm returned to `StartTime=Unknown` for all four jobs at this poll.

Follow-up poll, 2026-06-06 08:15 PDT:

- 4K D-PACE train job `3192611` completed successfully:
  - Slurm elapsed: `7m17s`
  - train runtime: `283.5377s`
  - train loss: `5.330507278442383`
  - checkpoint root:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_4096_20260606_070951`
- Both held-out OpenMath gates started immediately after the successful train:
  - K5 gate `3192612`, running on `nvl72036-T15`
  - K3 gate `3192613`, running on `nvl72091-T14`
- Gate `breakdown.json` files are still pending at this timestamp, so no 4K
  speedup or acceptance number is available yet.

Follow-up result, 2026-06-06 08:27 PDT:

- Both 4K D-PACE OpenMath gates completed successfully.

| Gate | Job | Throughput / GPU | Speedup vs bs32 baseline | Acceptance | Mean acceptance length | Per-position acceptance |
|---|---:|---:|---:|---:|---:|---|
| 4K D-PACE K5 | `3192612` | `586.70 tok/s` | `1.212x` | `46.92%` | `3.346` | `76.19%, 56.92%, 42.55%, 33.04%, 25.93%` |
| 4K D-PACE K3 | `3192613` | `576.69 tok/s` | `1.191x` | `61.73%` | `2.852` | `78.48%, 60.04%, 46.67%` |

Comparison against the current best 2K D-PACE draft-probability checkpoint:

| Gate | Throughput / GPU | Speedup | Acceptance | Delta vs 2K D-PACE K5 |
|---|---:|---:|---:|---:|
| 2K D-PACE K5 `3190567` | `627.14 tok/s` | `1.296x` | `47.01%` | reference |
| 4K D-PACE K5 `3192612` | `586.70 tok/s` | `1.212x` | `46.92%` | `0.936x` of 2K throughput |
| 4K D-PACE K3 `3192613` | `576.69 tok/s` | `1.191x` | `61.73%` | `0.920x` of 2K throughput |

Interpretation:

- Scaling this D-PACE draft-probability CE recipe from 2K to 4K rows did not
  improve the OpenMath bs32 gate. The best 4K speedup is `1.212x`, below the
  2K D-PACE K5 speedup of `1.296x`.
- K3 shows a much higher aggregate acceptance rate, but that rate is computed
  over only three draft positions. Its throughput is still lower than K5 and
  lower than the 2K K5 checkpoint, so the higher K3 acceptance alone is not a
  promotion signal.
- This result strengthens the current diagnosis: for Qwen3-235B OpenMath,
  simply adding more teacher rows to this local objective is not enough. The
  remaining bottleneck is likely objective/runtime efficiency rather than raw
  sample count.
- Updated CSV:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`
- Updated plot:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.png`

Full-GRPO config audit, 2026-06-06 08:31 PDT:

- The pending dynamic D-PACE Full-GRPO jobs already use the current best 2K
  checkpoint, not the regressed 4K checkpoint:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_2048_20260606_054509/checkpoint-64`
- Submitted dynamic jobs:
  - K5 `3192180`, `num_speculative_tokens=5`
  - K3 `3192438`, `num_speculative_tokens=3`
- Shape for both dynamic jobs:
  - `32` nodes x `4` GPUs
  - generation TP `4`
  - GBS/samples `256`
  - `temperature=1.0`, `top_p=1.0`, `top_k=-1`
  - `max_new_tokens=256`
  - `max_steps=4`
  - `stop_after_generation=false`
  - `policy_draft_enabled=false`
- Current Slurm state:
  - baseline `3186510`: `PENDING (Priority)`
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`
  - latest candidate start at this poll: `2026-06-06T12:16:51` PDT
- Decision: do not submit a 4K D-PACE Full-GRPO job. The standalone gate
  showed the 4K checkpoint is worse than the 2K checkpoint, so adding it to
  NeMo-RL would consume 32-node queue capacity without a stronger hypothesis.

2K D-PACE K3 standalone fill-in submission, 2026-06-06 08:32 PDT:

- Gap: the 2K D-PACE checkpoint had a standalone K5 gate and NeMo-RL sync
  K3/K5 gates, but no standalone K3 gate on the same checkpoint.
- Submitted OpenMath standalone K3 gate for the best 2K D-PACE checkpoint:
  - job `3193047`
  - checkpoint:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/qwen3_235b_pard_math_artifacts/checkpoints/PARD-Qwen3-0.6B_qwen235b_math_k5_dpace_draft_ce_2048_20260606_054509/checkpoint-64`
  - `num_speculative_tokens=3`
  - OpenMath held-out prompts, `ISL=1024`, `OSL=1024`, bs32
  - target TP `4`, draft TP `4`, `parallel_drafting=true`
  - output JSON:
    `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard2_dpace_draft_2048_20260606_054509_k3_openmath_isl1024_osl1024/breakdown.json`
- Purpose: determine whether the K3 advantage seen in NeMo-RL sync also holds
  in the standalone OpenMath gate for the same 2K checkpoint.

Queue follow-up, 2026-06-06 08:33 PDT:

- 2K D-PACE K3 standalone gate `3193047` is `PENDING (Priority)` with no
  dependency and `StartTime=Unknown`.
- Full-GRPO jobs remain `PENDING (Priority)`:
  - baseline `3186510`: planned `2026-06-06T11:41:09` PDT
  - local CAT/TPP K5 `3186511`: planned `2026-06-06T12:33:59` PDT
  - dynamic D-PACE K5 `3192180`: planned `2026-06-06T12:33:59` PDT
  - dynamic D-PACE K3 `3192438`: planned `2026-06-06T12:33:59` PDT
- No ray-driver logs or Full-GRPO E2E metrics exist yet.

Standalone JSON diagnosis and queue follow-up, 2026-06-06 08:44 PDT:

- 2K D-PACE K3 standalone gate `3193047` has started and is running on
  `nvl72086-T16`. The K3 `breakdown.json` is not present yet.
- Full-GRPO jobs remain `PENDING (Priority)`:
  - baseline `3186510`
  - local CAT/TPP K5 `3186511`
  - dynamic D-PACE K5 `3192180`
  - dynamic D-PACE K3 `3192438`
- Existing standalone JSONs show that the 4K regression is not explained by a
  lower K5 acceptance rate:

| Gate | Latency | Throughput / GPU | Acceptance | Accepted tokens / draft |
|---|---:|---:|---:|---:|
| 2K D-PACE K5 `3190567` | `13.062s` | `627.14 tok/s` | `47.01%` | `2.350` |
| 4K D-PACE K5 `3192612` | `13.963s` | `586.70 tok/s` | `46.92%` | `2.346` |
| 4K D-PACE K3 `3192613` | `14.205s` | `576.69 tok/s` | `61.73%` | `1.852` |

Interpretation:

- 2K K5 and 4K K5 have essentially the same accepted tokens per draft
  (`2.350` vs `2.346`), but 4K K5 is about `6.9%` slower in measured latency.
  This is why throughput regressed from `627.14` to `586.70 tok/s/GPU`.
- The available breakdown files have no useful trace attribution coverage for
  drafting vs verification buckets, so do not over-claim which internal vLLM
  subcomponent caused the latency gap from these JSONs alone.
- The evidence still points away from raw sample count as the limiting factor.
  The next useful training-side move is a closer accepted-prefix/CAT or LK-loss
  objective, while the next systems-side move is reducing runtime overhead in
  the PARD verification/drafting path.

2K D-PACE K3 standalone completion, 2026-06-06 08:43 PDT:

- Job `3193047` completed successfully in `6m35s`.
- Result JSON:
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs/qwen235b_pard2_dpace_draft_2048_20260606_054509_k3_openmath_isl1024_osl1024/breakdown.json`

| Gate | Latency | Throughput / GPU | Speedup | Acceptance | Accepted tokens / draft | Per-position acceptance |
|---|---:|---:|---:|---:|---:|---|
| 2K D-PACE K5 `3190567` | `13.062s` | `627.14 tok/s` | `1.296x` | `47.01%` | `2.350` | `76.25%, 57.24%, 42.76%, 32.90%, 25.89%` |
| 2K D-PACE K3 `3193047` | `14.016s` | `584.47 tok/s` | `1.207x` | `61.55%` | `1.847` | `78.06%, 60.43%, 46.16%` |

Interpretation:

- On the same best 2K D-PACE checkpoint, standalone K3 does not beat K5.
  Although K3 aggregate acceptance is higher, it emits fewer accepted tokens
  per draft and measured latency is higher than K5.
- This means the earlier NeMo-RL sync K3-over-K5 signal should not be treated
  as a standalone engine preference. It may be specific to the NeMo-RL sync
  generation shape, scheduling behavior, or measurement window.
- Updated CSV:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`
- Updated plot:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.png`

Full-GRPO queue follow-up, 2026-06-06 08:43 PDT:

- The no-stop Full-GRPO matrix is still pending with no E2E metrics:
  - baseline `3186510`: `PENDING (Priority)`
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`

External method and next-loss follow-up, 2026-06-06 08:50 PDT:

- Rechecked the official AMD PARD repo README:
  https://github.com/AMD-AGI/PARD
  - The README still lists the 2026-05-09 PARD-2 paper announcement and says
    code/model checkpoints will be released soon.
  - Therefore, our current PARD-2 work remains a runnable local approximation
    using public PARD plus CAT/D-PACE/LK-style objective ablations; it is not
    an official PARD-2 checkpoint reproduction.
- Rechecked Test-Time Speculation:
  https://arxiv.org/abs/2605.09329
  - This is relevant to the Qwen3-235B failure mode because it explicitly
    targets long-response acceptance degradation for DFlash, EAGLE-3, and PARD.
  - It suggests a systems/training direction: use verification-time target
    outputs to adapt the drafter online during long generations. That is a
    larger runtime integration than the current offline PARD trainer, so it is
    tracked as a next method direction rather than a quick gate.
- Immediate low-cost ablation submitted: an LK-loss-like direct acceptance-rate
  objective using the existing 2048-row math teacher set, without another
  Qwen3-235B teacher pass.

| Item | Job | State | Notes |
|---|---:|---|---|
| Accept-rate train | `3193161` | `COMPLETED` in `4m51s` | `cat_loss_mode=accept_rate`, same 2K teacher chunks, `train_runtime=164.3043s`, `train_loss=-4.4168701171875`; checkpoint-64 is present |
| Accept-rate OpenMath K5 gate | `3193162` | `COMPLETED` in `9m58s` | bs32, `ISL=1024`, `OSL=1024`, target TP4, draft TP4, `parallel_drafting=true`; `613.14 tok/s/GPU`, `1.267x`, `46.80%` acceptance |

Code changes made for this ablation:

- `experiments/pard_qwen3_235b_math/submit_qwen235b_pard2_dpace_draft_2048_from_existing.sh`
  now accepts `TRAIN_CAT_LOSS_MODE`, `TRAIN_CAT_IMPORTANCE_MODE`, and
  `GATE_MODEL_LABEL` env overrides instead of hardcoding the D-PACE draft CE
  mode.
- Updated CSV:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`
- Updated plot:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.png`

Accept-rate result interpretation:

- The accept-rate objective is better than the regressed 4K D-PACE gates, but
  still below the current best 2K D-PACE K5 checkpoint:

| Gate | Throughput / GPU | Speedup | Acceptance | Accepted tokens / draft |
|---|---:|---:|---:|---:|
| 2K D-PACE K5 `3190567` | `627.14 tok/s` | `1.296x` | `47.01%` | `2.350` |
| 2K accept-rate K5 `3193162` | `613.14 tok/s` | `1.267x` | `46.80%` | `2.340` |
| 4K D-PACE K5 `3192612` | `586.70 tok/s` | `1.212x` | `46.92%` | `2.346` |

- This suggests direct acceptance-rate maximization is a useful ablation but
  not the answer by itself. The best current checkpoint remains 2K D-PACE K5.
  A stronger next training objective should combine accepted-prefix utility
  with stable CE anchoring or target-confidence weighting, rather than pure
  acceptance maximization.

Hybrid D-PACE + accept-rate CE follow-up, 2026-06-06 09:16 PDT:

- Implemented `cat_loss_mode=dpace_accept_rate_ce` in
  `experiments/pard_qwen3_235b_math/pard_train_cat_weighted.py`.
  - Loss shape: D-PACE weighted CE anchor minus
    `cat_accept_reward_weight * mean(accepted_prefix_probability)`.
  - This is meant to test whether the best 2K D-PACE K5 checkpoint can be
    improved with a bounded accepted-prefix reward without the drift risk of
    pure `accept_rate`.
- Added config plumbing:
  - `experiments/pard_qwen3_235b_math/make_pard_train_config.py`
  - `experiments/pard_qwen3_235b_math/train_pard_math_k5.sh`
  - `experiments/pard_qwen3_235b_math/submit_pard_math_k5_train.sh`
  - `experiments/pard_qwen3_235b_math/submit_qwen235b_pard2_dpace_draft_2048_from_existing.sh`
- Syntax checks passed:
  - `python3 -m py_compile experiments/pard_qwen3_235b_math/pard_train_cat_weighted.py experiments/pard_qwen3_235b_math/make_pard_train_config.py`
  - `bash -n experiments/pard_qwen3_235b_math/train_pard_math_k5.sh experiments/pard_qwen3_235b_math/submit_pard_math_k5_train.sh experiments/pard_qwen3_235b_math/submit_qwen235b_pard2_dpace_draft_2048_from_existing.sh`
- Submitted 2K hybrid train + OpenMath K5 gate using the same 2048 teacher rows:

| Item | Job | State | Notes |
|---|---:|---|---|
| Hybrid train | `3193361` | `COMPLETED` in `4m50s` | `cat_loss_mode=dpace_accept_rate_ce`, `cat_accept_reward_weight=1.0`, `train_runtime=165.5486s`, `train_loss=1.296966552734375` |
| Hybrid OpenMath K5 gate | `3193362` | `COMPLETED` in `11m11s` | bs32, `ISL=1024`, `OSL=1024`, target TP4, draft TP4; `600.98 tok/s/GPU`, `1.241x`, `46.86%` acceptance |

- Updated CSV:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.csv`
- Updated plot:
  `docs/qwen3_235b_pard_math_local_checkpoint_gates.png`

Hybrid result interpretation:

| Gate | Throughput / GPU | Speedup | Acceptance | Accepted tokens / draft |
|---|---:|---:|---:|---:|
| 2K D-PACE K5 `3190567` | `627.14 tok/s` | `1.296x` | `47.01%` | `2.350` |
| 2K accept-rate K5 `3193162` | `613.14 tok/s` | `1.267x` | `46.80%` | `2.340` |
| 2K hybrid D-PACE + reward K5 `3193362` | `600.98 tok/s` | `1.241x` | `46.86%` | `2.343` |
| 4K D-PACE K5 `3192612` | `586.70 tok/s` | `1.212x` | `46.92%` | `2.346` |

- The hybrid objective did not improve acceptance or throughput over the 2K
  D-PACE K5 checkpoint.
- Because 4K D-PACE and the 2K hybrid both regress, the next useful step is
  not blindly adding rows or adding another simple scalar reward. The likely
  missing pieces are a closer PARD-2 target-confidence/CAT implementation,
  test-time/online adaptation during long generation, or runtime overhead
  reduction in the PARD draft/verification path.

PARD-2-style training sample/time guidance, 2026-06-06 PDT:

- Official PARD-2 code/checkpoints are not released yet in the AMD PARD repo,
  so this guidance is for our local PARD-2-style approximations, not an
  official reproduction.
- Current evidence says sample count is not the main limiter:
  - 1K CAT prefix-product: `train_runtime=90.27s`.
  - 2K CAT/TPP mask: `4m45s` elapsed, `train_runtime=160.467s`.
  - 2K D-PACE draft CE: `4m50s` elapsed, `train_runtime=162.373s`,
    best current standalone gate at `1.296x`.
  - 2K accept-rate: `4m51s` elapsed, `train_runtime=164.304s`,
    gate `1.267x`.
  - 2K hybrid D-PACE + reward: `4m50s` elapsed,
    `train_runtime=165.549s`, gate `1.241x`.
  - 4K D-PACE draft CE: `7m17s` elapsed, `train_runtime=283.538s`,
    gate regressed to `1.212x`.
- Practical planning:
  - 1K-2K rows are enough for smoke/objective ranking.
  - 8K-16K rows are reasonable for a stronger domain ablation once the
    objective is fixed.
  - 50K rows may be useful for a serious domain drafter candidate, but should
    wait until the 2K/4K objective trend improves.
  - 500K rows are not justified yet; the costly part is collecting
    Qwen3-235B teacher continuations/logprobs and gating them, not the 0.6B
    drafter fine-tune itself.
- Cost split: 128-row Qwen3-235B teacher-logprob chunks took about `34-39 min`
  each on 4 GB200 GPUs with target `TP=4`. When chunks are submitted in
  parallel, 2K teacher collection is roughly the slowest chunk plus queue time;
  when serialized, teacher generation dominates by hours. Drafter fine-tuning
  remains only minutes.

Full-GRPO queue follow-up, 2026-06-06 10:12 PDT:

- The current no-stop Full-GRPO matrix still has no E2E metrics. All active
  jobs are queued as `PENDING (Priority)` with elapsed `00:00:00`.
- Real-sampling `MAX_STEPS=4` jobs:
  - baseline `3186510`: `PENDING (Priority)`, elapsed `00:00:00`.
  - local CAT/TPP K5 `3186511`: `PENDING (Priority)`, elapsed `00:00:00`.
  - dynamic D-PACE K5 `3192180`: `PENDING (Priority)`, elapsed `00:00:00`.
  - dynamic D-PACE K3 `3192438`: `PENDING (Priority)`, elapsed `00:00:00`.
- Fixed-256 diagnostic `MAX_STEPS=20` jobs:
  - baseline `3186342`: `PENDING (Priority)`, elapsed `00:00:00`.
  - local CAT/TPP K5 `3186343`: `PENDING (Priority)`, elapsed `00:00:00`.
  - public PARD K5 `3186344`: `PENDING (Priority)`, elapsed `00:00:00`.
- Superseded queued jobs `3177855`, `3177856`, `3182758`, `3185571`,
  `3185572`, `3185573`, `3186173`, `3186174`, `3186175`, `3186176`,
  `3186177`, and `3186178` are cancelled and should not be reported as active.
- A smaller no-stop Full-GRPO validation was not submitted. The Qwen3-235B
  policy/training shape is configured for `32n4g` with generation `TP=4`,
  Megatron `TP=2`, `PP=8`, `CP=2`, `EP=16`, and GBS `256`. Reducing node count
  would change the policy shard shape rather than just making a cheaper E2E
  check. For cheap validation, use the 1-node `VllmGeneration` direct/sync
  gates; for true E2E, keep the current 32-node no-stop jobs.
- The Full-GRPO summarizer now supports active presets:
  - `python3 scripts/summarize_qwen235b_fullgrpo_pard.py --preset sampling-step4`
    includes baseline `3186510`, local CAT/TPP K5 `3186511`, dynamic D-PACE K5
    `3192180`, and dynamic D-PACE K3 `3192438`.
  - `python3 scripts/summarize_qwen235b_fullgrpo_pard.py --preset fixed256-step20`
    includes baseline `3186342`, local CAT/TPP K5 `3186343`, and public PARD K5
    `3186344`.
  - It currently emits `missing_log` rows for all active jobs, which is expected
    because no driver logs exist yet.
- Therefore, the evidence boundary remains unchanged:
  - vLLM standalone and NeMo-RL generation gates show PARD/PARD-2-style
    generation benefit.
  - True NeMo-RL no-stop E2E Full-GRPO speedup is still unmeasured.

D-PACE smoothing ablations submitted, 2026-06-06 09:47 PDT:

- Rationale: the current best 2K D-PACE K5 checkpoint used
  `cat_dpace_smoothing=0.5`. Since 4K row scaling and the hybrid reward both
  regressed, the next low-cost objective check is the D-PACE weighting shape,
  not more teacher rows.
- Both ablations reuse the same 2048 Qwen3-235B math teacher-logprob rows and
  require no new teacher generation pass.

| Ablation | Train job | Gate job | State | Notes |
|---|---:|---:|---|---|
| alpha `0.2` | `3193575` | `3193576` | train `COMPLETED` in `4m51s`; gate `COMPLETED` in `9m42s` | `601.72 tok/s/GPU`, `1.243x`, `47.84%` acceptance |
| alpha `0.8` | `3193588` | `3193589` | train `COMPLETED` in `5m00s`; gate `COMPLETED` in `11m45s` | `591.09 tok/s/GPU`, `1.221x`, `47.43%` acceptance |

- Train summaries:
  - alpha `0.2`: `train_runtime=164.236s`, `train_loss=2.493776321411133`.
  - alpha `0.8`: `train_runtime=174.6134s`, `train_loss=9.099517822265625`.
- Gate interpretation:
  - alpha `0.2` improved aggregate acceptance versus alpha `0.5`
    (`47.84%` vs `47.01%`) but reduced throughput (`1.243x` vs `1.296x`).
  - alpha `0.8` also regressed throughput (`1.221x`).
  - Therefore the current best remains 2K D-PACE K5 with
    `cat_dpace_smoothing=0.5`; no smoothing ablation should be promoted to
    NeMo-RL Full-GRPO.

- These are vLLM standalone gates only, and both are now negative. The current
  NeMo-RL Full-GRPO validation candidate remains the 2K D-PACE K5 checkpoint
  from job `3190567`, with K5 job `3192180` and K3 companion job `3192438`
  queued against the matched sampling baseline `3186510`.

NeMo-RL sync D-PACE K1/K2 fill-in, 2026-06-06 10:48 PDT:

- Completed the low-K runtime-cost fill-in using the same 2K D-PACE checkpoint,
  1-node sync `VllmGeneration` shape, fixed 256-token decode, target `TP=4`,
  draft `TP=4`, and matched baseline job `3192176`.

| K | Job | Throughput | Speedup vs baseline | Acceptance | Mean acceptance length | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `3194117` | `322.23 tok/s` | `1.250x` | `76.42%` | `1.764` | Highest acceptance, but too few speculative tokens to amortize overhead |
| 2 | `3194118` | `335.88 tok/s` | `1.303x` | `67.61%` | `2.352` | Better than K1, still below K3 |
| 3 | `3192349` | `374.62 tok/s` | `1.454x` | `57.50%` | `2.725` | Current best NeMo-RL sync generation point |
| 5 | `3192211` | `355.44 tok/s` | `1.379x` | `44.25%` | `3.212` | More draft tokens than K3, but extra verification/draft overhead reduces net throughput |

- K3 remains the best systems tradeoff for the NeMo-RL sync generation path.
  K1/K2 prove that higher aggregate acceptance alone is not sufficient; the
  accepted-token budget per draft and runtime overhead matter more.
- No K2 Full-GRPO companion is justified from this fill-in. Keep the active
  K3/K5 Full-GRPO jobs pending against the matched baseline until E2E metrics
  are available.

Full-GRPO queue follow-up, 2026-06-06 11:13 PDT:

- The no-stop Full-GRPO jobs still have no E2E metrics. Slurm reports all
  active real-sampling step4 jobs as `PENDING (Priority)` with elapsed
  `00:00:00`, and no `ray-driver.log` exists yet. It now also reports planned
  starts, although these times have been moving as Slurm reschedules:
  - baseline `3186510`: planned `13:44 PDT`
  - local CAT/TPP K5 `3186511`: planned `13:44 PDT`
  - dynamic D-PACE K5 `3192180`: planned `14:37 PDT`
  - dynamic D-PACE K3 `3192438`: planned `14:40 PDT`
- The fixed-256 `MAX_STEPS=20` diagnostic summarizer also still emits
  `missing_log` rows for baseline `3186342`, local CAT/TPP K5 `3186343`, and
  public PARD K5 `3186344`. Latest scheduler snapshot: baseline `3186342`
  planned `12:56 PDT`, local CAT/TPP K5 `3186343` and public PARD K5
  `3186344` planned `13:44 PDT`.
- Added remote refresh wrapper
  `scripts/refresh_qwen235b_fullgrpo_status_from_remote.sh`; it runs the
  summarizer on the remote filesystem, copies the CSV/MD summaries back into
  `docs/`, and writes
  `docs/qwen3_235b_fullgrpo_scheduler_status_20260606.csv` with the current
  Slurm state and `ray-driver.log` existence bit.
- Older sampling-smoke step2 jobs are also still queued as secondary evidence:
  baseline `3186345` and local K5 `3186354` planned for `13:34 PDT`, public
  PARD K5 `3186355` planned for `13:48 PDT`.
- Upstream PARD check at 11:13 PDT still reports AMD-AGI/PARD master HEAD
  `77eee0a12a729aaa4cc38b2a30fd544e11a8173b`; no official PARD-2 code/checkpoint
  release appeared in that repo during this refresh.
- Evidence boundary remains: vLLM standalone and 1-node NeMo-RL generation
  gates show usable PARD/PARD-2-style generation speedup, but true no-stop
  Full-GRPO E2E speedup is still unmeasured.

Full-GRPO queue follow-up, 2026-06-06 11:20 PDT:

- Refreshed with `scripts/refresh_qwen235b_fullgrpo_status_from_remote.sh`.
  No `ray-driver.log` exists for any active no-stop job yet, so both summary
  presets still emit `missing_log` rows and no E2E metrics.
- Current Slurm snapshot:
  - fixed-256 baseline `3186342`: `PENDING (Priority)`, planned
    `12:23:46 PDT`.
  - fixed-256 local CAT/TPP K5 `3186343`: `PENDING (Priority)`, planned
    `12:56:08 PDT`.
  - fixed-256 public PARD K5 `3186344`: `PENDING (Priority)`, planned
    `13:43:48 PDT`.
  - real-sampling baseline `3186510`: `PENDING (Priority)`, planned
    `13:43:48 PDT`.
  - real-sampling local CAT/TPP K5 `3186511`: `PENDING (Priority)`, planned
    `13:43:48 PDT`.
  - real-sampling dynamic D-PACE K5 `3192180`: `PENDING (Priority)`, planned
    `14:35:57 PDT`.
  - real-sampling dynamic D-PACE K3 `3192438`: `PENDING (Priority)`, planned
    `14:35:57 PDT`.
- Upstream PARD check at 11:20 PDT still reports AMD-AGI/PARD master HEAD
  `77eee0a12a729aaa4cc38b2a30fd544e11a8173b`; no official PARD-2
  code/checkpoint release is visible in that repo.

Full-GRPO queue follow-up, 2026-06-06 11:25 PDT:

- Refreshed again at 11:25 PDT. All active no-stop Full-GRPO jobs remain
  `PENDING (Priority)` with elapsed `00:00:00` and no `ray-driver.log`; summary
  rows therefore remain `missing_log`.
- Current planned starts:
  - fixed-256 diagnostic baseline/local/public PARD jobs `3186342`, `3186343`,
    `3186344`: planned `11:59:58 PDT`.
  - real-sampling baseline `3186510`: planned `13:17:25 PDT`.
  - real-sampling local CAT/TPP K5 `3186511`, dynamic D-PACE K5 `3192180`, and
    dynamic D-PACE K3 `3192438`: planned `13:58:26 PDT`.
- This still proves only that the Full-GRPO validation is queued, not that PARD
  has produced NeMo-RL E2E speedup. The proven NeMo-RL result remains the
  1-node `VllmGeneration` generation gate/sync speedup.

Full-GRPO queue follow-up, 2026-06-06 11:27 PDT:

- Refreshed at 11:27 PDT and manually checked `squeue` plus the expected
  `ray-driver.log` paths. All seven active jobs are still `PENDING (Priority)`;
  no driver logs exist yet, so there are no Full-GRPO E2E metrics.
- Current planned starts:
  - fixed-256 diagnostic baseline/local/public PARD jobs `3186342`, `3186343`,
    `3186344`: planned `11:59:58 PDT`.
  - real-sampling baseline `3186510`, local CAT/TPP K5 `3186511`, dynamic
    D-PACE K5 `3192180`, and dynamic D-PACE K3 `3192438`: planned
    `13:58:26 PDT`.
- AMD-AGI/PARD master remains at
  `77eee0a12a729aaa4cc38b2a30fd544e11a8173b`; no official PARD-2 code/checkpoint
  release is visible in that repo.

Full-GRPO queue follow-up, 2026-06-06 11:30 PDT:

- Refreshed at 11:30 PDT. All active Qwen3-235B no-stop Full-GRPO jobs are still
  `PENDING (Priority)` with elapsed `00:00:00`; manual driver-log checks also
  found no `ray-driver.log` for any active job.
- Current planned starts:
  - fixed-256 baseline `3186342` and local CAT/TPP K5 `3186343`: planned
    `12:02:22 PDT`.
  - fixed-256 public PARD K5 `3186344`: planned `13:22:15 PDT`.
  - real-sampling baseline `3186510` and local CAT/TPP K5 `3186511`: planned
    `13:36:34 PDT`.
  - real-sampling dynamic D-PACE K5 `3192180`: planned `14:09:25 PDT`.
  - real-sampling dynamic D-PACE K3 `3192438`: planned `14:27:06 PDT`.
- This remains a queue wait, not a runtime failure. There is still no NeMo-RL
  Full-GRPO E2E speedup number to report.

Full-GRPO queue follow-up, 2026-06-06 11:33 PDT:

- Refreshed at 11:33 PDT and inspected `scontrol show job` for every active
  no-stop Full-GRPO job. All seven jobs remain `PENDING` with
  `Reason=Priority`, `Dependency=(null)`, elapsed `00:00:00`, and missing
  `ray-driver.log`.
- This rules out a dependency deadlock or launch/log failure at the current
  snapshot. The jobs are waiting for 32-node allocation priority.
- Current planned starts:
  - fixed-256 baseline `3186342` and local CAT/TPP K5 `3186343`: planned
    `12:04:17 PDT`.
  - fixed-256 public PARD K5 `3186344`: planned `13:23:06 PDT`.
  - real-sampling baseline `3186510` and local CAT/TPP K5 `3186511`: planned
    `13:34:53 PDT`.
  - real-sampling dynamic D-PACE K5 `3192180`: planned `14:10:00 PDT`.
  - real-sampling dynamic D-PACE K3 `3192438`: planned `14:27:06 PDT`.
- Evidence boundary remains unchanged: vLLM standalone and 1-node NeMo-RL
  generation gates prove generation benefit; true 32-node no-stop Full-GRPO E2E
  benefit is still unmeasured.

Full-GRPO integration follow-up, 2026-06-06 12:48 PDT:

- Current focus moved from queue wait to making PARD/PARD-style Full-GRPO
  actually run through NeMo-RL actor creation, generation, logprob, and train
  step without stop-after-generation.
- Fixed two concrete integration blockers found while running smaller
  Qwen3-32B and Qwen3-30B-A3B jobs:
  - Qwen3-30B-A3B MoE actor import failed with
    `Grouped GEMM is not available`. The launcher now sets
    `++policy.megatron_cfg.moe_grouped_gemm=false`.
  - The remote `nemo_rl/models/megatron/community_import.py` now applies
    `moe_grouped_gemm`, `moe_enable_deepep`, `moe_token_dispatcher_type`,
    `moe_shared_expert_overlap`, and `moe_permute_fusion` onto the Megatron
    bridge provider before `finalize()`. Backup on remote:
    `community_import.py.before_moe_provider_override_20260606`.
  - Qwen3-32B local-transformer actor forward then failed with
    `Packed sequence is not supported by DotProductAttention`. The retry
    launcher now defaults `SEQUENCE_PACKING_ENABLED=false`; sequence parallel is
    also off for local-transformer retries.
- Qwen3-235B K5 Full-GRPO was also refreshed to avoid stale environment/config
  reuse:
  - `SOURCE_VLLM_SITE=` remains blank, avoiding the old vLLM 0.17 Python-site
    contamination that caused `rpds.rpds` import failure.
  - `MEGATRON_MOE_GROUPED_GEMM=false` is explicitly passed.
  - Both top-level `++policy.generation.vllm_kwargs.moe_backend=triton` and
    nested `kernel_config.moe_backend=triton` are passed.
  - Driver venv path now includes the run label instead of reusing the old
    fixed `..._k5_r2` path.
- Active jobs as of 12:48 PDT:
  - Qwen3-32B baseline seqpack-off Full-GRPO step1: `3195288`, `RUNNING`,
    driver venv creation stage.
  - Qwen3-32B public PARD K5 seqpack-off Full-GRPO step1: `3195289`,
    `RUNNING`, driver venv creation stage.
  - Qwen3-30B-A3B baseline provider-MoE/seqpack-off Full-GRPO step1:
    `3195290`, `RUNNING`, driver venv creation stage.
  - Qwen3-30B-A3B public PARD K5 provider-MoE/seqpack-off Full-GRPO step1:
    `3195291`, `RUNNING`, just started.
  - Qwen3-30B-A3B local PARD2CAT K5 provider-MoE/seqpack-off Full-GRPO step1:
    `3195292`, `RUNNING`, driver venv creation stage.
  - Qwen3-235B local CAT/TPP-mask PARD K5 Full-GRPO step20:
    `3195285`, `PENDING (Priority)`.
- Evidence boundary: no new throughput, E2E step-time, or acceptance metric has
  been emitted yet by the fresh seqpack-off/provider-MoE retries. The latest
  confirmed progress is that the previous failure signatures are understood and
  were removed from the new submissions.

Full-GRPO integration follow-up, 2026-06-06 13:15 PDT:

- Qwen3-32B public PARD K5 `3195289` proved the NeMo-RL PARD path can initialize
  through vLLM 0.20.0, load the PARD drafter, reach `SETUP COMPLETE`, and enter
  the first GRPO generation rollout. It failed during rollout at GBS 2048 with
  CUDA OOM. This is a memory-sizing failure, not a PARD/drafter compatibility
  failure.
- Qwen3-32B baseline `3195288` reached generation, reward, logprob, and training
  update, then failed with Megatron DDP async-overlap assertion
  `Communication call has not been issued for this bucket`. This matches the
  earlier Qwen3-8B online training blocker that was avoided by disabling
  `overlap_grad_reduce` and `overlap_param_gather`.
- Qwen3-30B-A3B baseline `3195290` and local PARD2-style `3195292` reached vLLM
  backend setup. Both then failed in training-side MoE model creation when
  `NRL_FORCE_LOCAL_TRANSFORMER_SPEC=true`: `SequentialMLP` attempted to
  `deepcopy(config)` and hit `TypeError: cannot pickle
  torch._C._distributed_c10d.ProcessGroup`. This points to the local torch
  transformer spec being unsafe for MoE training workers with custom process
  groups.
- Corrective action:
  - `submit_qwen32_qwen30ba3b_pard2style_fullgrpo5.sh` now always passes
    `policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false`
    and
    `policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false`.
  - The smaller-model reruns were resubmitted with GBS 256:
    - Qwen3-32B baseline no-overlap: `3195498`
    - Qwen3-32B public PARD K5 no-overlap: `3195499`
    - Qwen3-30B-A3B baseline no-local-spec/no-overlap: `3195500`
    - Qwen3-30B-A3B local PARD2-style K5 no-local-spec/no-overlap: `3195501`
- Qwen3-235B local CAT/TPP-mask PARD K5 `3195285` remains `PENDING (Priority)`.
  Its launcher already uses the defensive settings learned above:
  `FORCE_LOCAL_TRANSFORMER_SPEC=false`, `MEGATRON_MOE_GROUPED_GEMM=false`,
  no-overlap DDP, and vLLM MoE backend `triton`.
- Current evidence boundary: Qwen3-235B Full-GRPO has not yet started, so there
  is still no verified 235B E2E speedup number. The strongest positive evidence
  is that Qwen3-32B PARD reached NeMo-RL `SETUP COMPLETE` and first rollout; the
  remaining blockers are training memory/DDP/MoE integration settings, not PARD
  model loading.

Full-GRPO integration follow-up, 2026-06-06 13:46 PDT:

- Qwen3-32B dense Full-GRPO now has a completed matched baseline/PARD pair at
  GBS 256, 4 nodes x 4 GB200, generation `TP=2`, training `TP=2`,
  `MAX_STEPS=1`, natural EOS, `max_new_tokens=256`, `temperature=1.0`,
  `top_p=1.0`, `top_k=-1`, sequence packing off, and DDP overlap disabled.

| Model | Job | SpecDec | Result | Step time | Generation time | E2E tok/s/GPU | Generation worker tok/s/GPU | Acceptance |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Qwen3-32B | `3195498` | baseline | `COMPLETED 0:0` | `58.69s` | `16.66s` | `100.91` | `355.47` | n/a |
| Qwen3-32B | `3195499` | public PARD K5 | `COMPLETED 0:0` | `56.14s` | `14.21s` | `105.49` | `416.75` | `49.4%` avg draft, mean accepted length `3.47` |

- This gives Qwen3-32B public PARD K5 speedups of:
  - E2E step-time speedup: `58.69 / 56.14 = 1.045x`.
  - E2E throughput speedup: `105.49 / 100.91 = 1.045x`.
  - Generation-time speedup: `16.66 / 14.21 = 1.18x`.
  - Generation-worker throughput speedup: `416.75 / 355.47 = 1.17x`.
- Interpretation: PARD is reducing generation cost in real NeMo-RL
  Full-GRPO, but at this small one-step Qwen3-32B shape, generation is only
  `25-28%` of the total step. Policy training, logprob, and preparation time
  dominate E2E, so the E2E speedup is much smaller than generation speedup.
- Qwen3-30B-A3B no-local-spec retries `3195500` and `3195501` still failed
  before useful metrics with the same MoE-side error:
  `SequentialMLP` calls `deepcopy(config)`, and the `TransformerConfig`
  contains distributed `ProcessGroup` objects that are not pickleable. This is
  not a PARD model-loading problem; it is a Megatron MoE model construction
  problem that also matters for Qwen3-235B-A22B.
- Corrective patch applied on the active remote checkout:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
  now uses `copy.copy(config)` instead of `deepcopy(config)` only in
  `SequentialMLP`'s `moe_ffn_hidden_size != ffn_hidden_size` branch. The only
  required mutation is a top-level `ffn_hidden_size` override, so a shallow copy
  preserves the intended behavior while avoiding recursive copies of
  `ProcessGroup`.
- Reproducibility artifact:
  `experiments/eagle3_online/remote_patch_files/megatron_moe_sequential_mlp_shallowcopy_processgroup.patch`.
- Reapply helper:
  `scripts/apply_remote_megatron_moe_pg_shallowcopy_patch.sh`.
- Poll helper for the current Qwen3-235B/Qwen3-30B-A3B validation set:
  `scripts/poll_qwen235b_pard_fullgrpo_current_status.sh`.
- Recovery/status summary:
  `docs/qwen3_235b_pard_fullgrpo_recovery_20260606.md`.
- Machine-readable current validation table:
  `docs/qwen3_235b_fullgrpo_validation_status_20260606.csv`.
- The remote file passed `python3 -m py_compile`. A timestamped backup was left
  next to the original as
  `experts.py.before_processgroup_shallowcopy_20260606_*`.
- Post-patch Qwen3-30B-A3B MoE validation jobs were submitted:
  - baseline: `3195815`
  - local PARD2-style/CAT K5: `3195816`
  Both use GBS 256, no-local-transformer-spec, no-overlap DDP,
  `moe_grouped_gemm=false`, vLLM MoE backend `triton`, and a fresh
  `pgshallow` cache/checkpoint suffix.
- Qwen3-235B local CAT/TPP-mask PARD K5 Full-GRPO `3195285` remains
  `PENDING (Priority)` as of this update. Because it has not started yet, the
  `SequentialMLP` shallow-copy patch should be picked up before its MoE policy
  workers are created.

Operational runbook added, 2026-06-06 PDT:

- Next experiment runbook:
  `docs/qwen3_235b_next_experiment_runbook_20260606.md`.
- Machine-readable experiment matrix:
  `docs/qwen3_235b_next_experiment_matrix_20260606.csv`.
- First command after internal DNS/VPN recovery:

```bash
TRY_REMOTE=true scripts/refresh_qwen235b_report_bundle.sh
```
