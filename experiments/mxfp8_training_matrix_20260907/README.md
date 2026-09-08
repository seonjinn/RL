# MXFP8 training and rollout matrix

This experiment validates 20 GRPO steps on GB200 for:

- Qwen3-30B-A3B
- Qwen3.5-35B-A3B
- NVIDIA Nemotron-3.5-Lightning-30B-A3B
- synchronous colocated rollout and asynchronous 1-off rollout
- MXFP8 training with `fp8_param=false` and `fp8_param=true`
- MXFP8 rollout with the FlashInfer TRTLLM MoE backend

The per-module Transformer Engine recipes quantize routed MoE FC1 and FC2
only. The Qwen3.5 and Nemotron 3.5 recipes keep the first two and last six
transformer layers in BF16 for both training and rollout.

Run a scheduler preflight before submission:

```bash
CLUSTER=ptyche MODEL=qwen30 MODE=sync FP8_PARAM=false ACTION=test-only \
  bash experiments/mxfp8_training_matrix_20260907/submit.sh

CLUSTER=ptyche MODEL=qwen30 MODE=sync FP8_PARAM=false ACTION=submit \
  bash experiments/mxfp8_training_matrix_20260907/submit.sh
```

Supported values are:

- `CLUSTER=ptyche|lyris`
- `MODEL=qwen30|qwen35|nano35`
- `MODE=sync|async`
- `FP8_PARAM=false|true`
- `ACTION=test-only|submit`

The launcher requires a clean remote checkout at `EXPECTED_HEAD`. Source code
lives under `/home`; node-local caches live under `/raid/scratch`; only durable
logs are written to `/lustre`.
