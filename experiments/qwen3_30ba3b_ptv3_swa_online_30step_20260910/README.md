# Qwen3-30B-A3B PTV3-SWA Online Drafter Cadence

This experiment isolates online drafter-update cadence for the PTV3-SWA
step-44000 DFlash and DSpark checkpoints. It preserves the official 4-node,
4-GPU Qwen3-30B-A3B Math performance recipe and the frozen cohort's runtime,
target model, FAP CUDA Graph mode, FlashInfer TRTLLM MoE backend, and sequence
packing configuration.

## Matrix

| Arm | Runtime K | Fixed interval | Optimizer steps |
|---|---:|---:|---:|
| DFlash | 5 | 5 | 30 |
| DFlash | 5 | 10 | 30 |
| DSpark | 7 | 5 | 30 |
| DSpark | 7 | 10 | 30 |

The four jobs have no SLURM dependencies and may run in parallel. Results must
be compared with the matched PTV3-SWA frozen and no-SpecDec runs over a common
closed step window. Report generation and E2E throughput, step-time breakdown,
acceptance, reward, approximate entropy, generation KL error, policy KL error,
draft loss, update requests, and refit requests.

## Commands

```bash
python3 matrix.py --json
bash submit_online_matrix.sh --test-only
bash submit_online_matrix.sh --submit
```

The launcher fails closed on source SHA, source cleanliness, container,
checkpoint, target-model, W&B authentication, and SLURM `--test-only` checks.
