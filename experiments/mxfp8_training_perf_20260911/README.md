# MXFP8 Training Parameter-Storage Comparison

Ten 20-step Async-1off runs: five models, each with fp8_param=false/true.
Both arms use routed-expert-only MXFP8 training and MXFP8 rollout, FP32
routing, FlashInfer TRTLLM, CUDA Graphs and the configured nccl_reshard path.
The pinned nightly uses the existing Python exact-transfer fallback over NCCL;
these are not native nccl.m2n performance claims. Receiver batching is enabled.

| Model | Nodes x GPUs | Training batch | First/last BF16 |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 4 x 4 | 2048 | 0 / 0 |
| Qwen3-235B-A22B | 32 x 4 | 512 | 0 / 0 |
| Qwen3.5-35B-A3B-Base | 8 x 4 | 128 | 2 / 6 |
| Nemotron 3.5 Lightning 30B-A3B | 8 x 4 | 128 | 2 / 6 |
| Nemotron 3 Super 120B-A12B | 32 x 4 | 256 | 2 / 6 |

These are matched pairs within each model, not a cross-model throughput ranking.
The Qwen3.5/Lightning batch is larger than the previous batch-16 functional check.
No model is declared validated merely because submission succeeds.

Use the shared launcher with MODEL, ARM=mxfp8-false-mxfp8 or
mxfp8-true-mxfp8, MODE=async, MAX_STEPS=20, WALLTIME=04:00:00,
CONFIG_OVERRIDE=experiments/mxfp8_training_perf_20260911/MODEL-async.yaml.
Set REPO, immutable CONTAINER, SLURM_ACCOUNT and RUN_GROUP explicitly.
Run ACTION=test-only before ACTION=submit. The launcher pulls and checks pinned
recursive submodules, archives source once per source ID, and expands working
files and compile caches on node-local storage. No per-job shared virtualenvs.

Report steps 2-20 inclusive with sample counts, validation time separately,
and policy/logprob/generation/E2E logged tokens/sec/GPU. Async generation wait
is not total generation duration. Refit is weight_sync plus a separately
reported overlapping refit bubble, never their sum. Include gen_kl_error,
reward and entropy ranges. Compare true/false ratios only for matching inputs,
code, image, scope and topology; keep failed/OOM runs out of speedup tables.
