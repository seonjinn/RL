# Qwen3-235B MXFP8 Linear Backend Comparison

This experiment compares FlashInfer CuTeDSL and CUTLASS for MXFP8 dense
linear layers in the Qwen3-235B-A22B NeMo-RL performance recipe. Both arms use
the same 16-node, four-GB200-per-node configuration, FlashInfer TRTLLM MoE
backend, CUDA Graph execution, and MXFP8 quantization scope. The scope includes
Q/K/V/O projections by ignoring only `lm_head` and the router gate.

Prepare the custom vLLM checkout at commit
`a76062edee3a3ac23d47a93c7ce466f06a19111f` before using `ACTION=test-only` or
`ACTION=submit`:

```bash
git -C ../../3rdparty/vllm rev-parse HEAD
```

The command must print `a76062edee3a3ac23d47a93c7ce466f06a19111f`.

Run the two-step smoke validation before measurement:

```bash
ACTION=test-only MAX_STEPS=2 ./submit_matrix.sh
ACTION=submit MAX_STEPS=2 RUN_ID=$(date +%Y%m%d-%H%M%S) ./submit_matrix.sh
```

After both smoke arms qualify, validate and submit the eight-step measurement:

```bash
ACTION=test-only MAX_STEPS=8 ./submit_matrix.sh
ACTION=submit MAX_STEPS=8 RUN_ID=$(date +%Y%m%d-%H%M%S) ./submit_matrix.sh
```

Each matrix arm is submitted independently with no inter-arm `afterok`
dependency. Use two steps for CUDA Graph and refit smoke validation. Use eight
steps for the measurement run and report the steady-state mean from steps 3
through 8.
