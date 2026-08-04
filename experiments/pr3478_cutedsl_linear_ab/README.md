# MXFP8 Linear Backend A/B

This experiment compares FlashInfer CUTLASS and CuTeDSL MXFP8 dense linear
GEMMs during NeMo-RL rollout. Training remains BF16. The MXFP8 scope includes
the MoE experts and Q/K/V/O projections so the selected dense linear backend
is exercised by generation.

## Matrix

| Model | GCP-NRT topology | Generation topology | vLLM TP |
|---|---|---|---:|
| Qwen3-30B-A3B | 4 x 8 B200 | 2 dedicated nodes | 1 |
| Qwen3-235B-A22B | 16 x 4 allocated B200 GPUs | colocated | 4 |

Each model runs two otherwise matched arms:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`

The MoE backend remains vLLM's MXFP8 auto selection, FlashInfer TRTLLM on
B200. Only `policy.generation.vllm_kwargs.linear_backend` changes.

The trainer remains BF16 (`fp8_param=false`), so these runs use the legacy
refit transport (`refit_transport=null`). Current NCCL Reshard validates only
matching BF16 or blockwise-FP8 trainer and rollout storage and rejects
BF16-to-MXFP8 conversion.

The 235B recipe intentionally allocates four of each B200 node's eight GPUs.
This preserves the source 16-node, four-GPU topology and places one TP4 vLLM
engine per node. Folding the same 64 GPUs onto eight B200 nodes places two TP4
engines per NVSwitch domain and caused nondeterministic engine-startup failures.
Set `DISABLE_CUSTOM_ALL_REDUCE=true` for both matched arms as an additional
startup guard; the linear GEMM backend remains the only A/B difference.

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
MODEL=qwen235b TOTAL_NODES=16 GPUS_PER_NODE=4 \
  DISABLE_CUSTOM_ALL_REDUCE=true LINEAR_BACKEND=flashinfer_cutlass \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
MODEL=qwen235b TOTAL_NODES=16 GPUS_PER_NODE=4 \
  DISABLE_CUSTOM_ALL_REDUCE=true LINEAR_BACKEND=flashinfer_cutedsl \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
```

The primary metric is generation step time and derived generation
tokens/s/GPU. E2E, logprob, policy training, and refit metrics are retained as
controls.
