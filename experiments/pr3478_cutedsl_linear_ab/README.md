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

Each model runs two otherwise matched arms:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`

The MoE backend remains vLLM's MXFP8 auto selection, FlashInfer TRTLLM on
B200. Only `policy.generation.vllm_kwargs.linear_backend` changes.

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

The primary metric is generation step time and derived generation
tokens/s/GPU. E2E, logprob, policy training, and refit metrics are retained as
controls.
