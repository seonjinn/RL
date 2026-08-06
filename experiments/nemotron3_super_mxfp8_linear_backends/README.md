# Nemotron3 Super MXFP8 Linear Backend Comparison

This experiment compares the FlashInfer CuTeDSL and CUTLASS dense MXFP8 linear
backends with the shipped 32-node Nemotron3 Super performance recipe. Both arms
use four GPUs per node, segment size eight, TP4 generation, 256 rollouts per
step, an 8,192-token sequence cap, CUDA Graphs, and the FlashInfer TRTLLM MoE
backend. The selected dense linear backend is the only experiment variable.

The launcher overrides the BF16 generation path with `precision=fp8` and
`is_mx=true`. Quantization ignores only `lm_head` and `mlp.gate`, so Q/K/V/O
projections use the selected backend. It relies on the recipe's default
colocated IPC refit path and does not enable `nccl_reshard`.

Prepare the custom vLLM checkout at commit
`a76062edee3a3ac23d47a93c7ce466f06a19111f` with the existing Qwen preparation
job before using `ACTION=test-only` or `ACTION=submit`:

```bash
ACTION=submit ./experiments/qwen30b_mxfp8_linear_backends/prepare_custom_vllm_ptyche.sh
```

The launcher sources `3rdparty/vllm/nemo-rl.env` and fails before GRPO starts
unless the runtime `vllm.__file__` resolves under that custom checkout.
Submission also requires a clean NeMo-RL checkout. The launcher captures its
exact commit and the custom-vLLM commit at submission, then rechecks both
commits and NeMo-RL cleanliness when the queued job starts.

Run a two-step scheduler and CUDA Graph/refit smoke validation first:

```bash
ACTION=test-only MAX_STEPS=2 ./submit_matrix_ptyche.sh
ACTION=submit MAX_STEPS=2 RUN_ID=$(date +%Y%m%d-%H%M%S) ./submit_matrix_ptyche.sh
```

After both arms complete the smoke run, submit the eight-step measurement:

```bash
ACTION=test-only MAX_STEPS=8 ./submit_matrix_ptyche.sh
ACTION=submit MAX_STEPS=8 RUN_ID=$(date +%Y%m%d-%H%M%S) ./submit_matrix_ptyche.sh
```

The matrix submits both arms independently, with no inter-arm dependency.
W&B and checkpoint writes are disabled; TensorBoard logging remains enabled.
An explicit `EXPERIMENT_ROOT` is split into backend-specific subdirectories.
Each arm writes `run_manifest.json` there after runtime validation, recording
the model, exact source/runtime commits, container, recipe, CUDA Graph mode,
quantization scope, MoE backend, and linear backend.
