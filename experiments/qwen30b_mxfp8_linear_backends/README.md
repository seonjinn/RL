# Qwen3-30B-A3B MXFP8 Linear Backend Comparison

The model-matrix workflow compares the dense MXFP8 linear backends used by a
Qwen3-30B-A3B NeMo-RL rollout:

- `flashinfer_cutedsl`
- `flashinfer_cutlass`

Both arms use the shipped four-node performance recipe, 4 GPUs per node, a
global batch size of 2,048, a 4,096-token maximum sequence length, and CUDA
Graphs. The MoE backend is pinned to `flashinfer_trtllm`; the dense linear
backend is the only experimental variable. The model, seed, generation policy,
and training configuration remain unchanged. The single-arm launcher retains
the TRTLLM modes used by the separate 32K study, but the default model matrix
submits exactly CuTeDSL and CUTLASS.

The stock recipe leaves Q/K/V/O projections in BF16. This comparison removes those four exclusions so the selected dense MXFP8 backend is exercised by the projection layers. `lm_head` and `mlp.gate` remain unquantized.

The launcher requires the custom vLLM implementation at commit
`a76062edee3a3ac23d47a93c7ce466f06a19111f`, sources its `nemo-rl.env`, and
rejects a runtime import whose `vllm.__file__` is outside that checkout. The
plain TRTLLM mode uses this implementation without tactic hints. The Adaptive
mode uses the same TRTLLM path, but pins a previously qualified exact-shape
table for the Qwen3-30B output projection and routes unqualified layer families
to CuTeDSL.

## Workflow

Prepare the custom vLLM checkout once in the remote NeMo-RL experiment checkout:

```bash
ACTION=test-only ./experiments/qwen30b_mxfp8_linear_backends/prepare_custom_vllm_ptyche.sh
ACTION=submit ./experiments/qwen30b_mxfp8_linear_backends/prepare_custom_vllm_ptyche.sh
```

Preparation may change only root `pyproject.toml`, root `uv.lock`, and
`3rdparty/vllm`. It rejects any other tracked or untracked NeMo-RL source
change before and after preparation. Replaced vLLM checkouts are moved under
the external preparation output root, not elsewhere in the repository.

Then validate scheduling and submit a short smoke matrix:

```bash
ACTION=test-only MAX_STEPS=2 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
ACTION=submit MAX_STEPS=2 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
```

After both arms complete two steps without initialization, refit, NCCL, CUDA
Graph, or token-validity errors, submit the eight-step measurement matrix:

```bash
ACTION=submit MAX_STEPS=8 ./experiments/qwen30b_mxfp8_linear_backends/submit_matrix_ptyche.sh
```

Report the mean of steps 3-8. Primary metrics are rollout generation time and generated tokens/s/GPU. Secondary metrics are total step time, refit time, log-probability time, and training time.

`ACTION=test-only` and `ACTION=submit` reject all NeMo-RL source changes except
the preparation-owned `pyproject.toml`, `uv.lock`, and `3rdparty/vllm` state.
The launcher fingerprints the two dependency files, the recipe content, and
the clean tracked vLLM source at submission, then rechecks them with the exact
NeMo-RL and vLLM commits when the job starts. Untracked vLLM build artifacts
may remain, but tracked and staged vLLM changes are rejected. Each backend
writes the complete validated configuration to
`<run-root>/<backend>/run_manifest.json`.

## 32K Output-Length Study

`DAPO` is an RL recipe family, not a fixed context length. The repository's DAPO recipes use different limits, including 16K, 30K, and 49K total contexts. This experiment defines a 32K output cap explicitly:

- maximum input length: 2,048 tokens
- maximum generated length: 32,768 tokens
- vLLM and policy context limit: 34,816 tokens
- rollouts per step: 48 prompts x 4 generations = 192
- measured training steps: 20
- CUDA Graphs: enabled

The smaller rollout count keeps the maximum token volume per step near the original 2,048-sample x 4K experiment while allowing individual responses to reach 32K. It also limits each generation worker to 12 concurrent rollouts. The long-context launcher reserves 50% of GPU memory for vLLM instead of the recipe's 60% default, leaving wake-up headroom for the prepared TRTLLM weights in the colocated policy/generation process. Activation checkpointing is enabled, sequence packing is disabled, and log-probability execution uses batch size one with 2,048-token chunks and deferred FP32 logits to reduce long-sequence training memory pressure.

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
