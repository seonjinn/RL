# Qwen3-30B-A3B MXFP8 Linear Backend Comparison

This experiment compares the dense MXFP8 linear backends used by a Qwen3-30B-A3B NeMo-RL rollout:

- `flashinfer_cutedsl`
- `flashinfer_cutlass`
- `flashinfer_trtllm`
- `flashinfer_trtllm_adaptive`

All arms use the shipped four-node performance recipe, 4 GPUs per node, a global batch size of 2,048, a 4,096-token maximum sequence length, and CUDA Graphs. The MoE backend is pinned to `flashinfer_trtllm`; the dense linear backend is the only experimental variable. The model, seed, generation policy, and training configuration remain unchanged.

The stock recipe leaves Q/K/V/O projections in BF16. This comparison removes those four exclusions so the selected dense MXFP8 backend is exercised by the projection layers. `lm_head` and `mlp.gate` remain unquantized.

The plain TRTLLM arm uses the custom vLLM implementation at commit `a76062edee3a3ac23d47a93c7ce466f06a19111f` without tactic hints. The Adaptive arm uses the same TRTLLM path, but pins a previously qualified exact-shape table for the Qwen3-30B output projection and routes unqualified layer families to CuTeDSL. An exact-shape miss uses TRTLLM's default tactic and is not an error. The table is loaded before CUDA Graph capture; no shmoo runs in the rollout request path. The table fingerprints the NeMo-RL worker package version `0.25.1+precompiled`, which is built from the same pinned custom vLLM commit, and retains strict SHA256 validation.

## Workflow

Prepare the custom vLLM checkout once in the remote NeMo-RL experiment checkout:

```bash
ACTION=test-only ./experiments/qwen30b_mxfp8_linear_backends/prepare_custom_vllm_ptyche.sh
ACTION=submit ./experiments/qwen30b_mxfp8_linear_backends/prepare_custom_vllm_ptyche.sh
```

Then validate scheduling and submit a short smoke matrix:

```bash
ACTION=test-only MAX_STEPS=2 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
ACTION=submit MAX_STEPS=2 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
```

After all three arms complete two steps without initialization, refit, NCCL, CUDA Graph, or token-validity errors, submit the eight-step measurement matrix:

```bash
ACTION=submit MAX_STEPS=8 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
```

Report the mean of steps 3-8. Primary metrics are rollout generation time and generated tokens/s/GPU. Secondary metrics are total step time, refit time, log-probability time, and training time.

## 32K Output-Length Study

`DAPO` is an RL recipe family, not a fixed context length. The repository's DAPO recipes use different limits, including 16K, 30K, and 49K total contexts. This experiment defines a 32K output cap explicitly:

- maximum input length: 2,048 tokens
- maximum generated length: 32,768 tokens
- vLLM and policy context limit: 34,816 tokens
- rollouts per step: 48 prompts x 4 generations = 192
- measured training steps: 20
- CUDA Graphs: enabled

The smaller rollout count keeps the maximum token volume per step near the original 2,048-sample x 4K experiment while allowing individual responses to reach 32K. It also limits each generation worker to 12 concurrent rollouts, preserving KV-cache headroom for the prepared TRTLLM weights. Activation checkpointing is enabled, sequence packing is disabled, and log-probability execution uses batch size one with 2,048-token chunks and deferred FP32 logits to reduce long-sequence training memory pressure.

Run a two-step scheduling and runtime smoke first:

```bash
ACTION=test-only MAX_STEPS=2 RUN_ID=q30-long32k-smoke \
  ./experiments/qwen30b_mxfp8_linear_backends/submit_long32k_ptyche.sh
ACTION=submit MAX_STEPS=2 RUN_ID=q30-long32k-smoke \
  ./experiments/qwen30b_mxfp8_linear_backends/submit_long32k_ptyche.sh
```

After both backends complete CUDA Graph capture, generation, refit, log-probability inference, and one training update, submit the 20-step comparison without dependencies:

```bash
ACTION=test-only RUN_ID=q30-long32k-20step \
  ./experiments/qwen30b_mxfp8_linear_backends/submit_long32k_ptyche.sh
ACTION=submit RUN_ID=q30-long32k-20step \
  ./experiments/qwen30b_mxfp8_linear_backends/submit_long32k_ptyche.sh
```

Report steady-state steps 3-20. This is a long-output-cap experiment: the realized response length still depends on model EOS behavior and must be reported from the run rather than assumed to be 32K.
