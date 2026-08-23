# Sync MXFP8 Quantization Scope

This experiment measures the effect of widening MXFP8 rollout quantization in
synchronous colocated GRPO on GB200. All arms use BF16 policy training, vLLM
rollout with CUDA Graphs, CUDA IPC refit, trainer-side MXFP8 prequantization,
the batched TRTLLM expert shuffle, and the same seed.

## Arms

Qwen3-30B-A3B:

- `moe_only`: routed expert FC weights.
- `qkvo`: routed expert FC weights and attention QKVO projections.

Nemotron3 Nano:

- `moe_only`: routed expert FC weights.
- `qkvo`: routed expert FC weights and attention QKVO projections.
- `qkvo_mamba`: routed expert FC weights, attention QKVO projections, and the
  Mamba `in_proj` and `out_proj` linear projections.

The Nano Mamba arm does not quantize convolution, state-space parameters,
routers, shared experts, latent helper tensors, embeddings, or `lm_head`.

## Method

- Hardware: 4 nodes, 4 GB200 GPUs per node.
- Execution: synchronous colocated GRPO.
- Refit: CUDA IPC, 4 GiB persistent buffer, trainer prequantization.
- Runtime: CUDA Graphs enabled and FlashInfer TRTLLM MoE.
- Duration: 20 steps with seed 42.
- Reporting window: steps 3 through 20.

Run the scope audit before every arm. The audit records the intended quantized
and excluded layer families in each result directory.

The pre-submit audit expects the following projection-family counts:

| Model | Arm | Quantized families |
| --- | --- | ---: |
| Qwen3-30B-A3B | MoE only | 48 |
| Qwen3-30B-A3B | MoE + QKVO | 240 |
| Nemotron3 Nano | MoE only | 23 |
| Nemotron3 Nano | MoE + QKVO | 47 |
| Nemotron3 Nano | MoE + QKVO + Mamba | 93 |

```bash
ACTION=test-only MODEL=qwen30 ARM=moe_only ./experiments/mxfp8_sync_scope/submit_oci_hsg.sh
ACTION=submit MODEL=qwen30 ARM=moe_only ./experiments/mxfp8_sync_scope/submit_oci_hsg.sh
```
