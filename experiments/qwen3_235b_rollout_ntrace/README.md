# Qwen3-235B-A22B rollout ntrace comparison

This experiment captures matched rank-0 rollout traces for BF16 and
expert-only MXFP8 generation from the same Qwen3-235B-A22B policy checkpoint.

## Measurement contract

- Recipe: `grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml`
- Hardware: 16 GB200 nodes, four GPUs per node
- Generation: TP4, asynchronous vLLM engine, FlashInfer TRTLLM MoE
- Training: BF16, zero learning rate, no checkpoints
- MXFP8 scope: routed expert FC1/FC2 weights only
- Profiler: ntrace rank 0, one warm-up rollout, three captured rollouts
- CUDA graphs: captured during engine initialization

The two arms differ only in rollout precision. Attention, router gate, and
`lm_head` remain BF16 in the MXFP8 arm. The rank-0 capture supports module,
operation, and kernel-family breakdowns. It does not measure cross-rank skew.

## Launch

Run scheduler validation first, then submit the two tracked wrappers through
`tools/launch` using the cluster's pinned container and Hugging Face cache.
Each result directory includes source, model, container, and profiler
provenance.
