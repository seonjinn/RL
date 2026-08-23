# Qwen3-30B-A3B 20-step performance results

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

| Method | Policy train s | Train tok/s/GPU | Logprob s | Logprob tok/s/GPU | Generation s | Generation tok/s/GPU | Refit s (transfer/update s) | E2E step s | E2E tok/s/GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 11.133 | 3199.58 | 4.058 | 8769.75 | 51.656 | 695.90 | 5.728 (2.009) | 77.249 | 462.62 |
| DFlash K5 | 11.842 | 3011.33 | 4.248 | 8384.56 | 31.651 | 1131.21 | 6.117 (2.072) | 58.652 | 607.72 |
| DFlash K7 | 11.963 | 2984.23 | 4.358 | 8182.31 | 31.854 | 1135.81 | 5.799 (2.068) | 58.696 | 609.49 |
| DSpark K5 | 11.894 | 2997.38 | 4.311 | 8265.39 | 33.109 | 1086.60 | 5.896 (2.043) | 59.504 | 600.13 |
| DSpark K7 | 11.967 | 2979.74 | 4.250 | 8377.63 | 33.186 | 1088.90 | 5.768 (2.061) | 59.958 | 596.13 |
| Eagle-3 K3 | 11.025 | 3233.62 | 4.163 | 8556.01 | 40.276 | 892.72 | 5.867 (2.008) | 65.841 | 542.96 |
| Eagle-3 K5 | 11.279 | 3162.43 | 4.117 | 8640.43 | 41.623 | 856.77 | 5.810 (2.054) | 67.579 | 527.00 |
| Eagle-3 K5 wide buckets | 11.131 | 3203.81 | 4.152 | 8594.19 | 41.556 | 862.92 | 5.790 (2.044) | 67.050 | 532.36 |

## Baseline-relative result

| Method | Generation throughput | Generation-time speedup | E2E throughput | E2E step-time speedup |
|---|---:|---:|---:|---:|
| DFlash K5 | 1.626x | 1.632x | 1.314x | 1.317x |
| DFlash K7 | 1.632x | 1.622x | 1.317x | 1.316x |
| DSpark K5 | 1.561x | 1.560x | 1.297x | 1.298x |
| DSpark K7 | 1.565x | 1.557x | 1.289x | 1.288x |
| Eagle-3 K3 | 1.283x | 1.283x | 1.174x | 1.173x |
| Eagle-3 K5 | 1.231x | 1.241x | 1.139x | 1.143x |
| Eagle-3 K5 wide buckets | 1.240x | 1.243x | 1.151x | 1.152x |

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
- DFlash K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k5-lyris14500-be73db5620ed42d0a94a140ee278c719
- DFlash K7: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dflash-k7-lyris14500-c630b91c18434e3bb9af2c6a61a2d107
- DSpark K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k5-lyris14500-dc7cc1e70d0c4cab92f9f40bb57c9d07
- DSpark K7: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-dspark-k7-lyris14500-4ece4886d4e545aab96ed09b561301eb
- Eagle-3 K3: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k3-6694481b873c4dbbb15d56218dfad131
- Eagle-3 K5: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k5-9d9477bec023431ab1e91dc95f2a8bea
- Eagle-3 K5 wide buckets: https://wandb.ai/nvidia/sna-specdec/runs/q30ba3b-20step-eagle3-k5-wide-6d66173c1e9b446cadd2b21dadccd2c9
