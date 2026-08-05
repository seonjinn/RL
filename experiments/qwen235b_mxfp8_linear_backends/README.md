# Qwen3-235B MXFP8 Linear Backend Comparison

This experiment compares FlashInfer CuTeDSL and CUTLASS for MXFP8 dense
linear layers in the Qwen3-235B-A22B NeMo-RL performance recipe. Both arms use
the same 16-node, four-GB200-per-node configuration, FlashInfer TRTLLM MoE
backend, CUDA Graph execution, and MXFP8 quantization scope. The scope includes
Q/K/V/O projections by ignoring only `lm_head` and the router gate.

Run scheduler validation before submission:

```bash
ACTION=test-only BACKEND=flashinfer_cutedsl ./submit_cluster.sh
ACTION=test-only BACKEND=flashinfer_cutlass ./submit_cluster.sh
```

Submit both arms with a shared run ID:

```bash
ACTION=submit RUN_ID=$(date +%Y%m%d-%H%M%S) ./submit_matrix.sh
```

Use two steps for CUDA Graph and refit smoke validation. Use eight steps for
the measurement run and report the steady-state mean from steps 3 through 8.
