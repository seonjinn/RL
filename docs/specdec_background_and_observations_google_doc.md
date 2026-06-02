# Speculative Decoding for NeMo-RL Rollout Acceleration

Audience: NeMo-RL, vLLM, and training-performance collaborators  
Status: internal working draft, updated June 2, 2026  
Primary reference: arXiv 2601.11580v2, especially the execution-time breakdown style used around Figure 4.

## Executive Summary

Speculative decoding can improve rollout generation throughput when a small drafter proposes tokens that are accepted by the verifier often enough to offset drafter, verification, sampling, scheduling, and cache-management overheads. The key point from recent production-oriented speculative decoding analysis is that acceptance rate alone is not a speedup guarantee. Speedup depends on where the time goes: drafter generation, target verification, rejection sampling, scheduler overhead, CUDA Graph behavior, memory bandwidth, and whether the target model is already well utilized at the current batch size.

Our current observations match that framing.

- vLLM standalone can show clear generation-only speedup. Qwen3-8B with RedHatAI/Qwen3-8B-speculator.eagle3 and K=3 reaches 2.006x at batch size 4 and 2.145x at batch size 32 in static-batch standalone vLLM.
- Qwen3-235B-A22B with nvidia/Qwen3-235B-A22B-Eagle3 and K=1 reaches 1.125x to 1.275x speedup across batch sizes 1 to 32 in standalone vLLM with CUDA Graph enabled and custom all-reduce disabled.
- NeMo-RL full GRPO is harder. Generation-only speedup does not automatically become end-to-end step speedup because logprobs, refit, reward, and policy training still dominate part of the step.
- The best confirmed Qwen3-30B-A3B NeMo-RL generation result so far is the in-house 500K Speculators Eagle3 K=1 token-pressure run, job 3060379: 5696.58 generated tok/s/GPU, 1.366x generation speedup, 57.40% acceptance.
- The Qwen3-235B PublicHF always-on NeMo-RL run has high acceptance, 65.71%, but is slightly slower than its matched baseline: 0.986x generation throughput and 0.975x end-to-end throughput. This is the strongest evidence that high acceptance is not enough when SpecDec is enabled in the wrong load regime.
- Request-count-only long-tail gating is functional but too conservative in our Qwen3-30B-A3B test. Job 3116841 only enables drafting for a tiny fraction of scheduler checks, so speedup stays around 1.051x even with 54.67% acceptance on drafted tokens.

The practical conclusion is that our next optimization target should not be "maximize acceptance" in isolation. It should be "enable SpecDec only when the expected target-side savings exceed the drafter plus verification overhead." That means runtime gating based on active requests, scheduled-token pressure, draft-position acceptance, and measured generation-time contribution.

## Background: What Speculative Decoding Does

Speculative decoding uses two models:

- Target/verifier model: the real policy model whose output distribution must be preserved.
- Drafter model: a smaller or cheaper model that proposes one or more future tokens.

The drafter proposes K tokens. The target model verifies those tokens in parallel. Accepted tokens can skip autoregressive target-model calls. Rejected tokens fall back to target-model sampling, preserving correctness.

For rollout acceleration, this is attractive because generation can be a large part of RL wall time. However, the drafter is not free. Every speculative step adds:

- drafter forward time,
- target verification work,
- acceptance/rejection sampling,
- scheduler and metadata overhead,
- KV-cache and CUDA Graph interaction costs,
- possible loss of efficiency at larger batch sizes.

## What Determines SpecDec Performance Benefit

The arXiv 2601.11580v2 analysis is useful because it frames speculative decoding as an execution-time tradeoff, not just an acceptance-rate metric. A Figure 4-style breakdown should separate at least:

- Drafting: time spent running the drafter model.
- Verification: time spent running the target model over proposed tokens.
- Rejection sampling: time spent accepting/rejecting draft tokens and sampling fallback tokens.
- Other vLLM overheads: scheduler, cache, metadata, graph replay/warmup, and runtime framework overhead.

The main performance drivers are:

1. Acceptance rate

   Higher acceptance increases skipped target-model work. But acceptance must be interpreted per draft position. For K=3, the third token can have near-zero acceptance even when the first token is good. Drafting positions that are almost never accepted wastes time.

2. Proposal length K

   K=1 has a simple upper-bound intuition: if acceptance is a, the ideal no-overhead speedup is roughly 1 + a. With 57.40% acceptance, the no-overhead ceiling is about 1.574x. Our best measured Qwen3-30B-A3B generation speedup of 1.366x is below that ceiling but not inconsistent with it. K=2 or K=3 can exceed this ceiling only if later draft positions have meaningful acceptance and overhead stays low.

3. Drafter cost versus target cost

   SpecDec helps when the drafter is cheap relative to the target. It hurts when drafter work plus verification overhead approaches or exceeds the saved target work.

4. Batch size and load regime

   If the target model is already saturated at a large batch size, speculative decoding can reduce the number of target iterations but still lose to overhead. This is why recent long-tail work argues for enabling SpecDec only in underfilled or tail phases.

5. System overhead

   CUDA Graph mode, custom all-reduce behavior, KV-cache layout, scheduler implementation, and metric aggregation all affect whether a theoretical token-saving mechanism appears as a real throughput gain.

6. RL end-to-end composition

   NeMo-RL step time is not pure generation. Even if generation improves, total step time may not if logprobs, training, refit, or reward computation dominate the step.

## Acceptance Rate to Speedup Projection

Acceptance rate is useful, but only after separating an ideal upper bound from the measured system overhead. For a simple uniform-acceptance model with K drafted tokens, the ideal no-overhead generation speedup is:

```text
Ideal speedup(K) ~= 1 + p + p^2 + ... + p^K
```

where p is the probability that each draft position is accepted. For K=1 this simplifies to:

```text
Ideal speedup(K=1) ~= 1 + p
```

This is an upper bound. It assumes the drafter is free, verification overhead is free, batch scheduling does not hurt target utilization, and every draft position has the same acceptance probability. Real systems are below this bound.

Our best Qwen3-30B-A3B K=1 token-pressure result gives a useful calibration point:

- Job 3060379 acceptance: 57.40%
- Ideal K=1 upper bound: 1.574x
- Observed NeMo-RL generation speedup: 1.366x
- Observed gain efficiency: `(1.366 - 1) / (1.574 - 1) ~= 0.64`

So for Qwen3-30B-A3B K=1 in the same favorable token-pressure regime, a rough practical projection is:

```text
Practical K=1 speedup ~= 1 + 0.64 * acceptance_rate
```

This is not universal. It should not be applied to always-on 235B, where acceptance was 65.71% but generation throughput was 0.986x versus matched baseline. That run proves that the load regime and overhead can erase the acceptance benefit.

### K=1 Projection

| Acceptance | Ideal K=1 upper bound | 30B-calibrated K=1 projection | Interpretation |
|---:|---:|---:|---|
| 30% | 1.300x | 1.191x | Too low for large gains |
| 40% | 1.400x | 1.255x | Useful only if overhead is low |
| 50% | 1.500x | 1.319x | Good but still far from 2x |
| 57.40% | 1.574x | 1.366x | Matches our best 30B K=1 observation |
| 60% | 1.600x | 1.383x | Marginally better than current best |
| 65.71% | 1.657x | 1.419x | Acceptance alone would predict gain, but 235B always-on measured 0.986x |
| 70% | 1.700x | 1.447x | Still below 2x for K=1 |
| 80% | 1.800x | 1.511x | K=1 cannot reach 2x under this overhead model |

The main takeaway is that K=1 cannot realistically produce 2x generation speedup unless overhead is almost zero and acceptance is near 100%. For >2x, we need useful K=2/K=3 acceptance or a much lower overhead path.

### Deeper Drafting Upper Bounds

| Uniform acceptance p | Ideal K=1 | Ideal K=2 | Ideal K=3 |
|---:|---:|---:|---:|
| 30% | 1.300x | 1.390x | 1.417x |
| 40% | 1.400x | 1.560x | 1.624x |
| 50% | 1.500x | 1.750x | 1.875x |
| 55% | 1.550x | 1.853x | 2.019x |
| 60% | 1.600x | 1.960x | 2.176x |
| 65% | 1.650x | 2.073x | 2.347x |
| 70% | 1.700x | 2.190x | 2.533x |
| 80% | 1.800x | 2.440x | 2.952x |

These are still upper bounds. If we reuse the 30B K=1 overhead efficiency of about 0.64, then reaching a measured 2x would require approximately:

| Draft depth | Uniform per-position acceptance needed for projected 2x |
|---:|---:|
| K=1 | Not reachable under this overhead model |
| K=2 | ~84.8% |
| K=3 | ~70.9% |
| K=4 | ~65.9% |

This explains why static K=3 is not automatically beneficial. It can cross 2x in theory, but only if later positions are accepted often enough and overhead does not grow. In our Qwen3-30B-A3B observations, later draft positions were much weaker than the first position, so static deeper drafting wasted work.

### How to Use the Projection

Use the projection as a decision rule, not as a claim of guaranteed speedup:

- If K=1 acceptance is below 50%, expect at most a modest generation gain.
- If K=1 acceptance is around 57-60%, expect roughly 1.35-1.40x generation speedup in a favorable token-pressure regime.
- If K=1 acceptance is high but the batch is already saturated, speedup can still be below 1.0x.
- To target >2x, prioritize K=2/K=3 only when per-position acceptance remains high beyond the first token.
- Track per-position acceptance, not only aggregate acceptance.
- Track gate enabled ratio. A high acceptance rate on a tiny number of draft attempts will not move average throughput.

## Current Measurements

### vLLM Standalone: Qwen3-8B Public Eagle3

Target: Qwen/Qwen3-8B  
Drafter: RedHatAI/Qwen3-8B-speculator.eagle3  
Scope: static-batch vLLM standalone generation, not NeMo-RL end-to-end.

| Batch size | Baseline tok/s/GPU | K=1 tok/s/GPU | K=1 speedup | K=3 tok/s/GPU | K=3 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 245.68 | 241.20 | 0.982x | 434.45 | 1.768x |
| 2 | 482.62 | 326.17 | 0.676x | 437.35 | 0.906x |
| 4 | 855.50 | 959.64 | 1.122x | 1716.56 | 2.006x |
| 8 | 1792.27 | 1903.83 | 1.062x | 3372.17 | 1.882x |
| 16 | 3219.52 | 3764.53 | 1.169x | 6388.47 | 1.984x |
| 32 | 5349.54 | 7121.34 | 1.331x | 11475.60 | 2.145x |

Observation: K=3 is strong for batch size 4 and above in this standalone setup. Batch size 2 is worse than baseline, so the batch boundary is not monotonic or universal.

### vLLM Standalone: Qwen3-235B Public Eagle3

Target: Qwen/Qwen3-235B-A22B  
Drafter: nvidia/Qwen3-235B-A22B-Eagle3  
Scope: standalone vLLM LLM.generate wall-clock sweep, CUDA Graph on, custom all-reduce disabled, not NeMo-RL end-to-end.

| Batch size | K=1 output tok/s/GPU | K=1 speedup vs matched baseline |
|---:|---:|---:|
| 1 | 32.71 | 1.194x |
| 2 | 62.70 | 1.210x |
| 4 | 116.99 | 1.125x |
| 8 | 236.19 | 1.181x |
| 16 | 455.80 | 1.220x |
| 32 | 843.53 | 1.275x |

Observation: standalone 235B K=1 does show generation-only benefit across tested batch sizes. This is useful boundary evidence, but it does not prove NeMo-RL GRPO speedup.

### NeMo-RL: Qwen3-30B-A3B In-House 500K Eagle3

Scope: NeMo-RL GRPO measurements. These include rollout generation and other RL step components.

| Run | Job | Gate / K | Generation tok/s/GPU | Generation speedup | E2E tok/s/GPU | Acceptance | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| Historical baseline | 3056050 | no drafter | 4169.19 | 1.000x | 1758.32 | n/a | 20 completed-step mean |
| Best token-pressure K=1 | 3060379 | K=1, token gate 2048 | 5696.58 | 1.366x | 1926.58 | 57.40% | Best confirmed generation result |
| Request-bucket long-tail K=1 | 3116841 | K=1, request <= 8 | 4380.97 | 1.051x | 1736.27 | 54.67% | Gate works, but enables too rarely |
| Strict alltoall full-shape comparison | 3078630 vs 3078629 | K=1 | 6776.01 vs 6068.62 | 1.117x | 1001.70 vs 1004.69 | 57.28% | Generation improves, E2E is flat |

Observation: Qwen3-30B-A3B proves that SpecDec can help NeMo-RL generation, but the best confirmed generation speedup is 1.366x, not 2x. E2E speedup is much smaller because non-generation work remains significant.

### NeMo-RL: Qwen3-235B PublicHF Eagle3

Target: Qwen/Qwen3-235B-A22B-Thinking-2507  
Drafter: nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3  
Scope: corrected NeMo-RL diagnostic with the matching target/drafter pair.

| Run | Job | K | Generation tok/s/GPU | E2E tok/s/GPU | Acceptance | Speedup vs matched baseline |
|---|---:|---:|---:|---:|---:|---|
| Matched baseline | 3088636 | n/a | 61.48 | 44.72 | n/a | 1.000x |
| PublicHF always-on SpecDec | 3084952 | 1 | 60.61 | 43.60 | 65.71% | Generation 0.986x, E2E 0.975x |

Observation: This is the most important negative result. Acceptance is high, but throughput is lower. Gate metrics showed enabled_ratio=1.0 and disabled=0, so this run was effectively always-on. The result supports long-tail or load-aware gating rather than unconditional SpecDec.

### NeMo-RL: Qwen3-8B Public Eagle3

Target: Qwen/Qwen3-8B  
Drafter: RedHatAI/Qwen3-8B-speculator.eagle3  
Status: NeMo-RL baseline job 3119388 and SpecDec K=3 job 3119389 are running as of June 2, 2026. No GRPO step, acceptance, or throughput metric has been emitted yet. The only completed 8B results so far are standalone vLLM results.

## Why We Are Not Seeing 2x in NeMo-RL Yet

1. K=1 has a limited ceiling.

   At 57.40% acceptance, the ideal K=1 no-overhead speedup intuition is about 1.574x. A 1.366x observed generation speedup is plausible after overhead. To get beyond 2x, we likely need useful K=2/K=3 acceptance, lower drafter overhead, or a regime where target decoding is heavily underfilled.

2. Deeper drafting is not yet efficient enough.

   In our 30B K=3 dynamic run, later draft positions were weak, with the third position effectively near zero acceptance in some steps. Static K=3 can waste work if later tokens are rarely accepted.

3. Always-on SpecDec can lose.

   Qwen3-235B PublicHF shows 65.71% acceptance but still slows down in NeMo-RL. This indicates that the batch/load regime and overhead dominate.

4. Request-only long-tail gating can be too conservative.

   Job 3116841 verifies that request-bucket gating works, but scheduler enabled ratio is only 0.0137%. That is too little drafting to create a large average speedup.

5. NeMo-RL E2E is not generation-only.

   Even when generation improves, policy logprobs, training, reward computation, and refit can flatten the end-to-end gain. This is visible in the 30B alltoall full-shape comparison: generation improves by 1.117x, but E2E throughput is flat.

## Recommended Next Steps

1. Build a Figure 4-style timing breakdown for our vLLM standalone runs.

   We should report drafting, verification, rejection sampling, and other vLLM overheads separately. This will show whether the bottleneck is drafter cost, verification cost, or scheduler/runtime overhead.

2. Keep CUDA Graph enabled.

   The standalone 235B boundary run preserved CUDA Graph and only disabled custom all-reduce after it caused initialization failure. We should avoid enforce-eager for performance measurements unless debugging requires it.

3. Use dynamic gating, not always-on SpecDec.

   The gate should consider active request count, scheduled token count, and observed acceptance by draft position. Request-count-only gates are too blunt; token-pressure-only gates can improve throughput but may enable too broadly. The next policy should combine both.

4. Prefer K=1/K=2 before static K=3 in NeMo-RL.

   K=3 only makes sense when later draft positions are accepted enough. Otherwise K=3 burns drafter and verification overhead.

5. Separate generation speedup from E2E speedup in every report.

   Every table should show generation time, generation tok/s/GPU, E2E step time, E2E tok/s/GPU, acceptance, and gate enabled ratio. This prevents standalone generation wins from being confused with full RL wins.

6. Continue 8B NeMo-RL K=3 as a low-cost sanity check.

   Since standalone 8B K=3 is strong, the running NeMo-RL K=3 job is a useful way to test whether our NeMo-RL integration and long-tail gating can convert standalone benefits into rollout benefits at small model scale.

## Bottom Line

SpecDec is working mechanically: drafters attach, tokens are accepted, and standalone vLLM speedups are real. The main open problem is system-level selectivity. We need to enable SpecDec in the batch/load regime where it reduces target-model work more than it adds drafter and verification overhead. For NeMo-RL, the current evidence says that load-aware gating and per-position dynamic K are more important than simply increasing acceptance rate or training-set size.

## References

1. arXiv 2601.11580v2, HTML version: https://arxiv.org/html/2601.11580v2
2. Internal report: `experiments/eagle3_qwen3_235b/specdec_math_progress_report.html`
3. Internal status files: `latest_qwen3_8b_specdec_jobs.txt`, `latest_vllm_qwen235b_boundary_specdec_k1_jobs.txt`, `latest_qwen30ba3b_500k_specdec_req8_batchlog_jobs.txt`, `experiments/eagle3_qwen3_235b/qwen30ba3b_500k_live_summary.json`
