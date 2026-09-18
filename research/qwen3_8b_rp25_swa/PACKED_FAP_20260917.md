# Q8 GBS512 / 32K packed FAP gate

## Purpose

Measure whether sequence packing reduces policy-training and logprob time enough
to make generation the dominant Qwen3-8B GRPO stage. This is a new three-step
cohort. It does not overwrite the completed non-packed or earlier PIECEWISE
experiments.

The matched workload remains DAPOMath17K, 64 prompts x 8 generations, GBS512,
input cap 2048, output cap 30720, total cap 32768, BF16, training TP2/CP1 and
generation TP1 on one four-GPU GB200 node. Baseline uses engine-default request
concurrency. DFlash and DSpark use the rp25-44000 B8 exports, K5 serving and S8.
All drafters are frozen for this correctness/performance gate.

The only policy-side workload change from the completed FAP cohort is:

```yaml
policy:
  sequence_packing:
    enabled: true
    train_mb_tokens: 32768
    logprob_mb_tokens: 32768
```

Generation retains Inductor, `FULL_AND_PIECEWISE`, the target/draft CUDA Graph
audit, and the existing capture buckets. The W&B group is
`q8-gbs512-32k-packed-fap-20260917`.

## Baseline for comparison

The completed non-packed FAP jobs used three valid policy steps:

| Arm | Generation | Total step | Generation ratio |
|---|---:|---:|---:|
| Baseline, default concurrency | 417.69 s | 965.45 s | 43.26% |
| DFlash K5 frozen, S8 | 399.28 s | 935.45 s | 42.68% |
| DSpark K5 frozen, S8 | 424.69 s | 959.41 s | 44.27% |

The baseline averaged 9,289 tokens/sample. Its policy-training and logprob
means were 350.08 s and 185.92 s, so packing must reduce those stages without
changing generated lengths, rewards, entropy or KL behavior. A higher ratio
caused by slower generation is a failure, not a successful packing result.

## Acceptance gate

- All three arms reach policy step 3 with finite reward, entropy, generation KL
  and policy KL metrics.
- The resolved config reports sequence packing enabled with both token budgets
  equal to 32768.
- Mean output/total token lengths remain comparable to the non-packed FAP cohort.
- Baseline policy-training plus logprob time falls materially; generation time
  does not regress.
- Target and, for SpecDec arms, draft FULL graph capture and replay are present.
- Report generation ratio as generation time divided by total step time. Do not
  infer success from allocation wall time or process exit alone.

## Status

Renderer contract and resolved-config validation are implemented locally.
Submission receipts and GPU results are pending.
