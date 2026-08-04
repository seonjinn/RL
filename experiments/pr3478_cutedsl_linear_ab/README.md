# MXFP8 Linear Backend A/B

This experiment compares FlashInfer CUTLASS and CuTeDSL MXFP8 dense linear
GEMMs during NeMo-RL rollout. Training remains BF16. The MXFP8 scope includes
the MoE experts and Q/K/V/O projections so the selected dense linear backend
is exercised by generation.

## Matrix

| Model | GCP-NRT topology | Generation topology | vLLM TP |
|---|---|---|---:|
| Qwen3-30B-A3B | 4 x 8 B200 | 2 dedicated nodes | 1 |
| Qwen3-235B-A22B | 8 x 8 B200 | colocated | 4 |

## Results

### Qwen3-30B-A3B

Mean over steps 2-5 after excluding the first-step warmup:

| Metric | CUTLASS | CuTeDSL | CuTeDSL delta |
|---|---:|---:|---:|
| Generation time | 51.317 s | 51.524 s | +0.40% |
| Generation throughput | 9990.60 tok/s/GPU | 9951.95 tok/s/GPU | -0.39% |
| E2E step time | 303.949 s | 305.032 s | +0.36% |
| E2E throughput | 843.10 tok/s/GPU | 840.06 tok/s/GPU | -0.36% |
| Logprob time | 117.174 s | 117.653 s | +0.41% |
| Policy training time | 128.058 s | 128.166 s | +0.08% |
| Refit time | 5.277 s | 5.320 s | +0.82% |

The steady-state result is parity within normal run-to-run noise, with no
measured CuTeDSL generation speedup for this workload. CuTeDSL increased the
vLLM-specific cold initialization from 137.5 s to 172.0 s, but the complete
setup times were similar: 264.2 s for CUTLASS and 266.8 s for CuTeDSL.

- CUTLASS: job `493177`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/v856fdxy)
- CuTeDSL: job `493284`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/u2dkfhxx)

### Qwen3-235B-A22B

The matched runs are in progress. Final job IDs and W&B links will be added
after both arms complete.

Each model runs two otherwise matched arms:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`

The MoE backend remains vLLM's MXFP8 auto selection, FlashInfer TRTLLM on
B200. Only `policy.generation.vllm_kwargs.linear_backend` changes.

The trainer remains BF16 (`fp8_param=false`), so these runs use the legacy
refit transport (`refit_transport=null`). Current NCCL Reshard validates only
matching BF16 or blockwise-FP8 trainer and rollout storage and rejects
BF16-to-MXFP8 conversion.

vLLM 0.25.1 enables PyTorch symmetric-memory all-reduce by default. On the
GCP-NRT B200 nodes used here, its multicast allocation failed during KV-cache
profiling and terminated an engine. Both arms explicitly set
`VLLM_ALLREDUCE_USE_SYMM_MEM=0` and `DISABLE_CUSTOM_ALL_REDUCE=true`, selecting
the NCCL fallback for the common all-reduce path. The dense linear GEMM backend
remains the only A/B difference.

## Submit

Run scheduling validation first:

```bash
MODEL=qwen30b LINEAR_BACKEND=flashinfer_cutlass ACTION=test-only \
  experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
MODEL=qwen30b LINEAR_BACKEND=flashinfer_cutedsl ACTION=test-only \
  experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
```

Submit either model/backend pair with `ACTION=submit`. `MAX_STEPS=5` is the
default smoke; use `MAX_STEPS=20` for the final comparison.

The matched 235B submissions use:

```bash
MODEL=qwen235b TOTAL_NODES=8 GPUS_PER_NODE=8 \
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  DISABLE_CUSTOM_ALL_REDUCE=true LINEAR_BACKEND=flashinfer_cutlass \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
MODEL=qwen235b TOTAL_NODES=8 GPUS_PER_NODE=8 \
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  DISABLE_CUSTOM_ALL_REDUCE=true LINEAR_BACKEND=flashinfer_cutedsl \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
```

The primary metric is generation step time and derived generation
tokens/s/GPU. E2E, logprob, policy training, and refit metrics are retained as
controls.
