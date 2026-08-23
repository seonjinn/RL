# Qwen3-30B-A3B 20-step performance results

HTML dashboard: `public/reports/qwen3_30ba3b_specdec_results_20260823.html`

## Methodology

- Source: W&B run history, arithmetic mean over completed steps 3 through 20.
- All current-cohort rows use 4 OCI GB200 nodes, 4 GPUs per node, synchronous GRPO,
  OpenMathInstruct-2, 16 prompts per step, 32 generations per prompt, GBS 512,
  max new tokens 1024, and the same policy/training topology.
- Throughput is the metric logged by NeMo-RL in tokens/s/GPU; it is not reconstructed
  from averaged times.
- Refit has no logged token-throughput metric. The table reports total refit time and
  its transfer/update subcomponent.

## Current matched cohort

Every cell contains `measured value (baseline-relative speed)`. For time, the ratio is
`baseline time / method time`; for throughput, it is `method throughput / baseline
throughput`. A ratio below `1.000x` is slower than baseline for that metric.

### Stage time

| Method | Policy train | Logprob | Generation | Refit total | Refit transfer/update | E2E step |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 11.133 s (1.000x) | 4.058 s (1.000x) | 51.656 s (1.000x) | 5.728 s (1.000x) | 2.009 s (1.000x) | 77.249 s (1.000x) |
| DFlash K5, original drafter | 12.017 s (0.926x) | 4.246 s (0.956x) | 37.894 s (1.363x) | 5.995 s (0.955x) | 2.067 s (0.972x) | 64.749 s (1.193x) |
| DSpark K5, original drafter | 12.100 s (0.920x) | 4.247 s (0.956x) | 38.854 s (1.329x) | 5.753 s (0.996x) | 2.071 s (0.970x) | 65.591 s (1.178x) |
| DFlash K5, Lyris14500 drafter | 11.842 s (0.940x) | 4.248 s (0.955x) | 31.651 s (1.632x) | 6.117 s (0.936x) | 2.072 s (0.969x) | 58.652 s (1.317x) |
| DFlash K7, Lyris14500 drafter | 11.963 s (0.931x) | 4.358 s (0.931x) | 31.854 s (1.622x) | 5.799 s (0.988x) | 2.068 s (0.971x) | 58.696 s (1.316x) |
| DSpark K5, Lyris14500 drafter | 11.894 s (0.936x) | 4.311 s (0.941x) | 33.109 s (1.560x) | 5.896 s (0.972x) | 2.043 s (0.983x) | 59.504 s (1.298x) |
| DSpark K7, Lyris14500 drafter | 11.967 s (0.930x) | 4.250 s (0.955x) | 33.186 s (1.557x) | 5.768 s (0.993x) | 2.061 s (0.975x) | 59.958 s (1.288x) |
| Eagle-3 K3 | 11.025 s (1.010x) | 4.163 s (0.975x) | 40.276 s (1.283x) | 5.867 s (0.976x) | 2.008 s (1.001x) | 65.841 s (1.173x) |
| Eagle-3 K5 | 11.279 s (0.987x) | 4.117 s (0.986x) | 41.623 s (1.241x) | 5.810 s (0.986x) | 2.054 s (0.978x) | 67.579 s (1.143x) |
| Eagle-3 K5, wide buckets | 11.131 s (1.000x) | 4.152 s (0.977x) | 41.556 s (1.243x) | 5.790 s (0.989x) | 2.044 s (0.983x) | 67.050 s (1.152x) |

### Stage throughput

| Method | Train tok/s/GPU | Logprob tok/s/GPU | Generation tok/s/GPU | E2E tok/s/GPU |
|---|---:|---:|---:|---:|
| Baseline | 3199.58 (1.000x) | 8769.75 (1.000x) | 695.90 (1.000x) | 462.62 (1.000x) |
| DFlash K5, original drafter | 2965.64 (0.927x) | 8387.58 (0.956x) | 945.01 (1.358x) | 550.87 (1.191x) |
| DSpark K5, original drafter | 2945.47 (0.921x) | 8390.88 (0.957x) | 920.36 (1.323x) | 543.55 (1.175x) |
| DFlash K5, Lyris14500 drafter | 3011.33 (0.941x) | 8384.56 (0.956x) | 1131.21 (1.626x) | 607.72 (1.314x) |
| DFlash K7, Lyris14500 drafter | 2984.23 (0.933x) | 8182.31 (0.933x) | 1135.81 (1.632x) | 609.49 (1.317x) |
| DSpark K5, Lyris14500 drafter | 2997.38 (0.937x) | 8265.39 (0.942x) | 1086.60 (1.561x) | 600.13 (1.297x) |
| DSpark K7, Lyris14500 drafter | 2979.74 (0.931x) | 8377.63 (0.955x) | 1088.90 (1.565x) | 596.13 (1.289x) |
| Eagle-3 K3 | 3233.62 (1.011x) | 8556.01 (0.976x) | 892.72 (1.283x) | 542.96 (1.174x) |
| Eagle-3 K5 | 3162.43 (0.988x) | 8640.43 (0.985x) | 856.77 (1.231x) | 527.00 (1.139x) |
| Eagle-3 K5, wide buckets | 3203.81 (1.001x) | 8594.19 (0.980x) | 862.92 (1.240x) | 532.36 (1.151x) |

Time and throughput ratios need not be exact reciprocals because generated token counts
vary slightly by run; the throughput columns use the metrics logged directly by NeMo-RL.

Eagle-3 K3 job 6471189 completed all 20 steps with Slurm exit `0:0` in 32:04.
Eagle-3 K5 job 6471693 completed all 20 steps with Slurm exit `0:0` in 33:01.
Eagle-3 K5 wide-bucket job 6471694 completed all 20 steps with Slurm exit `0:0`
in 32:25.

## Eagle-3 K5 bucket ablation

The wide arm added `[64, 128, 256, 512]` to the standard K5 capture list while
leaving the config and the runtime shape-to-bucket mapping for shapes 1 through 48
unchanged. Compared with standard K5, the single wide run measured:

- Generation throughput: 862.92 vs 856.77 tok/s/GPU (`+0.72%`).
- Generation time: 41.556 vs 41.623 seconds (`-0.16%`).
- E2E throughput: 532.36 vs 527.00 tok/s/GPU (`+1.02%`).
- E2E step time: 67.050 vs 67.579 seconds (`-0.78%`).
- Mean tokens per sample: 1003.211 vs 1002.674 (`+0.05%`).
- CUDA Graph initialization: about 5 vs 3 seconds and 0.16 vs 0.11 GiB per
  worker.

Because the additional buckets are never selected by runtime shapes 1 through 48,
the sub-1% generation-time difference is consistent with run noise rather than a
mechanism-based speedup. The only clear causal change is higher capture startup cost
and memory.

## K semantics

- Evaluation K is vLLM `speculative_config.num_speculative_tokens`. DFlash and
  DSpark K5 therefore both propose 5 tokens, and their K7 variants both propose 7.
- DFlash co-training uses `gamma=K`; its training window includes an anchor and is
  therefore K+1 positions wide.
- The migrated DSpark checkpoint was trained with `block_size=8`; both DSpark
  evaluation arms retain that checkpoint/training-head width while vLLM truncates
  inference proposals to K5 or K7. DSpark internally constructs its shared DFlash
  body with `gamma=block_size-1`, but that bookkeeping does not change inference K.

## Historical Eagle-3 K5 control

Historical Eagle-3 K5 job 2250930 used a different 4K-OSL, 64-prompt, TP1/EP16
cohort and is therefore not directly comparable with the current table. Against its
own matched baseline over steps 2 through 20, generation throughput was 4695.63 vs
5173.60 tok/s/GPU (`0.908x`), generation time was 87.578 vs 79.513 seconds
(`0.908x` speedup), and E2E throughput was 1810.25 vs 1874.75 tok/s/GPU (`0.966x`).
Its actual CUDA Graph capture still completed successfully; the regression was
performance rather than fallback to eager execution.

## Eagle-3 CUDA Graph evidence

- Historical Eagle-3 K5 job 2250930 recorded actual mixed PIECEWISE 48/48 and decode
  FULL 48/48 capture completion on all workers, followed by `Graph capturing finished`.
- Current Eagle-3 K3 job 6471189 uses capture sizes
  `[1, 2, 4, 8, 12, 16, 24, 32]`, which cover every runtime shape from 1 through
  `max_num_seqs * (K + 1) = 8 * 4 = 32`. Its runtime gate recorded actual PIECEWISE
  8/8 capture completion before Step 1 and Step 2.
- Current Eagle-3 K5 standard job 6471693 recorded actual PIECEWISE 10/10 capture
  with sizes `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48]`, followed by Step 1, Step 2,
  and all 20 steps.
- Current Eagle-3 K5 wide job 6471694 recorded actual PIECEWISE 14/14 capture with
  the four additional sizes through 512 on all 16 generation workers, followed by
  Step 1, Step 2, and all 20 steps.

## W&B

- Baseline: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-baseline-k0-346bf3ea6ece41308116d897245c6dd1
- DFlash K5, original drafter: https://wandb.ai/nvidia/nemo-rl/runs/q30-20step-dflash-fee67fa4f77d4217b631412d5cf02931
- DSpark K5, original drafter: https://wandb.ai/nvidia/nemo-rl/runs/q30-20step-dspark-2bd78689016e4c1c927877d25a683e0d
- DFlash K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k5-lyris14500-be73db5620ed42d0a94a140ee278c719
- DFlash K7: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k7-lyris14500-c630b91c18434e3bb9af2c6a61a2d107
- DSpark K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k5-lyris14500-dc7cc1e70d0c4cab92f9f40bb57c9d07
- DSpark K7: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k7-lyris14500-4ece4886d4e545aab96ed09b561301eb
- Eagle-3 K3: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k3-6694481b873c4dbbb15d56218dfad131
- Eagle-3 K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k5-9d9477bec023431ab1e91dc95f2a8bea
- Eagle-3 K5 wide buckets: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k5-wide-6d66173c1e9b446cadd2b21dadccd2c9
