# Nemotron 3 Nano MXFP8 QKVO A/B

This suite measures Nemotron 3 Nano rollout and refit performance with the
standard MXFP8 quantization scope and with Q/K/V/O projections included.

QKVO enables q/k/v/o relative to the standard MXFP8 scope, while all other
ModelOpt-eligible layers stay unchanged. A prior QKV run showed
probability-ratio outliers, so this is performance/correctness validation, not
a recommended default.

## Matrix

| Arm | Rollout precision and scope | PR refit optimizations |
| --- | --- | --- |
| `bf16` | BF16 | off |
| `moe-baseline` | standard MXFP8 | off |
| `moe-optimized` | standard MXFP8 | on |
| `qkvo-baseline` | QKVO-inclusive MXFP8 | off |
| `qkvo-optimized` | QKVO-inclusive MXFP8 | on |

All arms use TP2, PP2, CP2, and EP8 from the canonical
`grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml` trainer recipe. The MXFP8
overlays use `4n4g` filenames to reflect the effective Lyris allocation.
The Lyris allocation is 4 nodes with 4 GB200 GPUs per node. Every arm uses
vLLM TP1, `gpu_memory_utilization=0.5`, train GBS 16, seed 42, real importance
sampling, and checkpointing disabled.

The BF16 arm uses Triton for the MoE kernel because that is the refit-compatible
BF16 path in the current vLLM container. MXFP8 uses its supported ModelOpt
FlashInfer path. Therefore BF16 is a matched precision-path baseline, not a
kernel-controlled generation baseline.

The optimized arms enable trainer-side prequantization, persistent IPC staging
buffers, slim post-refit offload, batched MoE shuffle, and cached weight-loader
replay. The baseline arms explicitly disable all five paths.
`pinned_reference_swap` remains disabled in every arm.

The launcher uses the nightly container's prebuilt Python and Ray environments,
prepends the local checkout to `PYTHONPATH`, and verifies the Ray CLI/driver
versions, imported NeMo-RL source, and W&B authentication before model
initialization.

## Run

The model defaults to:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/models/nemotron-nano3/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf
```

Override it with `NANO_MODEL_PATH`. The suite defaults to a 20-step Lyris
test-only submission:

```bash
./experiments/mxfp8_qkvo_nano/submit_suite.sh
ACTION=submit ./experiments/mxfp8_qkvo_nano/submit_suite.sh
```

Select one or more comma-separated arms:

```bash
ARM_FILTER=bf16,qkvo-optimized \
ACTION=submit \
MAX_STEPS=20 \
./experiments/mxfp8_qkvo_nano/submit_suite.sh
```

Logs, manifests, and results default to:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/mxfp8-qkvo-nano
```
