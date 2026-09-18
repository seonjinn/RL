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

The baseline averaged 9,286 tokens/sample. Its policy-training and logprob
means were 351.97 s and 188.46 s, so packing must reduce those stages without
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

The renderer contract, 18 study/renderer unit tests, three fully resolved config
tests, Ruff, shell syntax and diff checks pass at source commit
`680c939327531ccb959da3b3a3bebe52a4caefad`.

Three independent jobs were submitted on OCI-HSG at 2026-09-18 06:09 UTC,
using account `coreai_dlalgo_nemorl`, partition `batch`, one exclusive four-GPU
GB200 node each and a four-hour limit. All three exact commands passed
`sbatch --test-only`; planning IDs 7246588-7246590 are not real jobs. No job
dependencies were used.

| Arm | Job | Start |
|---|---:|---|
| Baseline, default concurrency | 7246595 | 2026-09-18 06:10:02 UTC |
| DFlash K5 frozen, S8 | 7246596 | 2026-09-18 06:10:05 UTC |
| DSpark K5 frozen, S8 | 7246597 | 2026-09-18 06:10:05 UTC |

Immutable bundle SHA256:
`badf0923a5848b15fd41dd62b236c1b8a4eea0ccaee614f0a4a8a6bfa38f7ba2`.
Artifact parent:
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q8-rp25-swa-20260914/gbs512-32k-packed-fap-680c93932-20260917`.
The first five minutes of scheduler monitoring completed with all jobs still
running on separate nodes. Each verified the immutable source and entered the
node-local dependency build. No traceback, OOM or early process failure was
present.

All three jobs subsequently completed three policy steps with exit code zero.
The resolved runtime configs show packing enabled with both token budgets at
32768. Target FULL capture/replay is present in every arm; DFlash and DSpark
also show draft FULL capture/replay. Rewards, entropy and KL metrics are finite.

## Three-step results

The table uses the arithmetic mean of all three valid policy steps. Throughput
is the NeMo-RL logged per-GPU metric; it is not reconstructed from averaged
times.

| Arm | Total step | Generation | Policy | Logprob | Generation ratio | E2E tok/s/GPU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, default concurrency | 571.47 s | 404.68 s | 101.03 s | 58.53 s | 70.81% | 2,125.97 |
| DFlash K5 frozen, S8 | 586.72 s | 417.93 s | 101.72 s | 59.61 s | 71.23% | 2,061.94 |
| DSpark K5 frozen, S8 | 595.23 s | 429.64 s | 100.02 s | 57.99 s | 72.18% | 2,022.67 |

Packing versus the corresponding non-packed arm:

| Arm | E2E time speedup | Policy speedup | Logprob speedup | Generation-time speedup |
|---|---:|---:|---:|---:|
| Baseline | 1.689x | 3.484x | 3.220x | 1.032x |
| DFlash K5 frozen, S8 | 1.594x | 3.391x | 3.084x | 0.955x |
| DSpark K5 frozen, S8 | 1.612x | 3.435x | 3.173x | 0.988x |

Packing therefore passes the functional and policy/logprob performance gates.
It makes generation dominant without materially changing baseline generation:
the baseline mean length is 9,256 tokens versus 9,286 non-packed, and the mean
reward is 0.8424 versus 0.8431.

The current S8 SpecDec comparison does not beat the packed baseline. Relative
to that baseline, DFlash is 0.974x in E2E time and DSpark is 0.960x. Their mean
generation lengths remain comparable, but their generation-time speedups are
0.968x and 0.942x. Logs show eight running requests and as many as 120 waiting
for both SpecDec arms, while the default baseline can run 128 requests per
engine. This gate therefore identifies S8 concurrency as the immediate
performance constraint; it does not establish that K5 SpecDec is intrinsically
slower. The prepared S32/S64 variants are the next controlled test.

Final-step quality diagnostics remain in the same range: entropy is
0.2597/0.2655/0.2584 and policy KL is
0.000761/0.000705/0.000749 for baseline/DFlash/DSpark, respectively. No quality
collapse is visible in this three-step gate.
