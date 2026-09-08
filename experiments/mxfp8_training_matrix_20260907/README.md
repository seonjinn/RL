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

For `fp8_param=true`, the launcher disables rollout refit prequantization.
The trainer already exports native MXFP8 values and scales in this mode, while
refit prequantization accepts BF16 trainer weights only.

Run a scheduler preflight before submission:

```bash
CLUSTER=ptyche MODEL=qwen30 MODE=sync FP8_PARAM=false ACTION=test-only \
  EXPECTED_HEAD=$(git rev-parse HEAD) \
  bash experiments/mxfp8_training_matrix_20260907/submit.sh

CLUSTER=ptyche MODEL=qwen30 MODE=sync FP8_PARAM=false ACTION=submit \
  EXPECTED_HEAD=$(git rev-parse HEAD) \
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
logs and the cross-node Hugging Face dataset cache are written to `/lustre`.
The driver uses the container's shared Python path so Ray system actors can
start on every node. NeMo-RL rebuilds each tier-specific actor environment on
the node where that actor runs. Each node stages the pinned source under
`/raid/scratch` first, which prevents concurrent editable builds from writing
to the same checkout. `PYTHONPATH` pins system actors to that staged source,
while `uv` reuses the container's preloaded cache. A setup failure on any node
terminates the batched worker step and signals the full Ray allocation to stop.
The copy does not preserve shared-filesystem metadata that node-local storage
cannot represent. Shared storage holds model snapshots, datasets, and durable
logs. Writable Python, Hugging Face module, and compiler caches stay
node-local.
