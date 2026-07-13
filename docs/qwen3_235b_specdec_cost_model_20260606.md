# Qwen3-235B SpecDec Cost Model

Date: 2026-06-06 PDT

This uses existing measured CSVs only. It is not a kernel-level
decomposition. The acceptance-only ideal factor is the observed
`mean_acceptance_length`; the realized fraction is
`observed speedup / mean_acceptance_length`.

Plot:

```text
docs/qwen3_235b_specdec_cost_model_20260606.png
```

## Observed Cost Pattern

| Area | Shape | Method | K | Speedup | Acceptance | Mean len | Realized / ideal | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| vLLM standalone | short ISL1000 OSL512 bs32 | public PARD | 12 | 3.290x | 92.95% | 12.154 | 0.271 | High acceptance and high mean length; useful sanity proof. |
| vLLM standalone | short ISL1000 OSL512 bs32 | public PARD | 5 | 1.707x | 95.51% | 5.775 | 0.296 | High acceptance but smaller K limits total batching factor. |
| vLLM standalone | OpenMath ISL1024 OSL1024 bs32 | public PARD | 5 | 1.313x | 45.51% | 3.276 | 0.401 | Domain mismatch lowers acceptance and mean length. |
| vLLM standalone | OpenMath ISL1024 OSL1024 bs32 | public PARD | 12 | 1.220x | 22.84% | 3.741 | 0.326 | Larger K increases overhead and deeper-position rejection. |
| vLLM standalone | long ISL10000 OSL1000 bs32 | public PARD | 5 | 0.392x | 13.68% | 1.684 | 0.233 | Long context turns PARD into a slowdown despite nonzero acceptance. |
| vLLM standalone | long ISL10000 OSL1000 bs32 | public PARD | 12 | 0.553x | 13.42% | 2.610 | 0.212 | Large K partly recovers batching but still below baseline. |
| vLLM standalone | decode-heavy 10k bs32 | K1 | 1 | 0.823x | 93.59% | 1.936 | 0.425 | High K1 acceptance still loses because one draft token cannot amortize long-context verify/draft overhead. |
| NeMo-RL sync generation | 1n4g fixed256 batch32 | dynamic D-PACE | 1 | 1.250x | 76.42% | 1.764 | 0.709 | K1 is positive in sync generation but slower than K2/K3. |
| NeMo-RL sync generation | 1n4g fixed256 batch32 | dynamic D-PACE | 3 | 1.454x | 57.50% | 2.725 | 0.533 | Best current sync generation systems tradeoff. |
| NeMo-RL sync generation | 1n4g fixed256 batch32 | public PARD | 5 | 1.522x | 42.29% | 3.115 | 0.489 | Best public sync smoke result; generation-only evidence. |
| NeMo-RL worker32 generation-only | 32n4g fixed256 req/engine32 | local CAT/TPP-mask | 5 | 1.718x | 53.53% | 3.677 | 0.467 | Generation segment only; not Full-GRPO E2E. |

## Interpretation

- High acceptance is insufficient when K is too small or context verification
  cost is high. The decode-heavy K1 case has `93.59%` acceptance but only
  `0.823x` throughput because one speculative token cannot amortize the
  additional draft and verification work.
- OpenMath reduces accepted length sharply relative to short synthetic
  prompts. The public PARD K5 bs32 case drops from `5.775` mean length on
  short synthetic to `3.276` on OpenMath.
- Long-context verification is the clearest standalone failure mode: PARD
  K5 at `ISL=10000/OSL=1000` has nonzero acceptance but only `0.392x`
  throughput. This points to attention/verification cost and drafter
  domain mismatch, not simply missing acceptance logging.
- NeMo-RL sync generation currently prefers K3 over K1/K2/K5 for the local
  D-PACE checkpoint. K1/K2 have higher acceptance, but issue too few
  speculative tokens to win throughput.
- The unresolved evidence is no-stop Full-GRPO E2E. Generation-only results
  cannot prove total step speedup because policy logprob/reference/update
  work may dominate the step.

## Action

Prioritize Qwen3-235B no-stop Full-GRPO with the current best two gates:
dynamic D-PACE K3 for NeMo-RL systems tradeoff and dynamic D-PACE K5 for
vLLM standalone OpenMath. Keep public PARD K5 as the public baseline.
Avoid spending more nodes on DFlash until the held-out OpenMath acceptance
checkpoint is improved.
