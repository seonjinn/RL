# MXFP8 Linear Backend A/B

This experiment compares FlashInfer CUTLASS and CuTeDSL MXFP8 dense linear
GEMMs during NeMo-RL rollout. Training remains BF16. The MXFP8 scope includes
the MoE experts and Q/K/V/O projections so the selected dense linear backend
is exercised by generation. The vocabulary-parallel `lm_head` remains BF16.

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

Jobs `493496` and `493497` passed the earlier vLLM startup failure after
disabling the common FlashInfer fused all-reduce/RMS path. The CUTLASS arm then
exposed an independent first-refit issue. Excluding the vocabulary-parallel
`lm_head` correctly narrowed the QKVO quantization scope, but jobs `493608` and
`493609` reproduced the same shape mismatch on the BF16
`model.embed_tokens.weight`: its full `[151936, 4096]` HF tensor reached a
TP4-local `[37984, 4096]` parameter through vLLM's default loader. TP1 had
masked the issue because its full and local vocab shapes are equal.

The refit loader now restores vLLM's TP-aware
`VocabParallelEmbedding.weight_loader` plus its `input_dim=1` and
`output_dim=0` sharding metadata when parameter post-processing has removed
them. The strengthened targeted test passed in job `493867`, and the complete
FP8 quantization unit file passed `14/14` tests in job `493889`.

CUTLASS job `493935` then passed the former vocab shape assertion and reached
Step 1. It exposed a separate recipe-scope issue: the QKVO overlay had also
quantized the MoE router `mlp.gate`. That `ReplicatedLinear` keeps its scale in
`weight_scale`, while the generic MXFP8 refit path attempted to load
`weight_scale_from_checkpoint`. The QKVO overlays now keep both `lm_head` and
`mlp.gate` in BF16, preserving the intended MXFP8 scope of MoE experts plus
Q/K/V/O projections. A recipe regression test covers both model overlays.

Replacement CUTLASS job `494003` is pending GCP-NRT resources. Latest run
references:

- CUTLASS: job `493935`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/gmm2z6ia)
- Replacement CUTLASS: job `494003`, W&B link available after startup
- Earlier CUTLASS: job `493743`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/ddut411d)
- Earlier CuTeDSL: job `493745`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/6l1emf1q)

Earlier failed-run references:

- CUTLASS: job `493496`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/bo9eflyp)
- CuTeDSL: job `493497`, [W&B run](https://wandb.ai/nvidia/sna-pr3478-cutedsl-linear-ab/runs/e3u8i38o)

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
profiling and terminated an engine. Both arms set
`VLLM_ALLREDUCE_USE_SYMM_MEM=0` and
`compilation_config.pass_config.fuse_allreduce_rms=false`. Custom all-reduce
remains enabled. The dense linear GEMM backend remains the only A/B difference.

## Async Non-Colocated Refit Support

The current branch contains PR 3478's batched MXFP8 MoE post-load shuffle but
does not contain PR 3477's BF16-to-MXFP8 NCCL Reshard conversion. Therefore the
support boundary is:

| Component | Async non-colocated legacy transport | Async non-colocated NCCL Reshard |
|---|---|---|
| BF16 trainer to MXFP8 rollout | Supported | Requires PR 3477 |
| Batched MXFP8 MoE shuffle | Supported | Runs after load when PRs 3477 and 3478 are combined |
| Trainer-side prequantization | Legacy PR 3294 path only | Not used; PR 3477 quantizes on receiver |
| Persistent CUDA IPC buffers | Not used across nodes | Not applicable |
| Loader metadata reuse | Legacy loader cache | NCCL Reshard builds its parameter map during initialization |

The stock Async 1-off MXFP8 recipes leave `refit_transport=null`; being async
and non-colocated does not select NCCL Reshard automatically. The exact
combination must explicitly set `policy.generation.refit_transport=nccl_reshard`
and run on a branch containing both PRs 3477 and 3478. That exact asynchronous
combination does not yet have an end-to-end test result. A patched synchronous
BF16-to-MXFP8 NCCL Reshard smoke reached two training steps, which validates the
receiver conversion path but not Async GRPO scheduling.

`policy.generation.vllm_cfg.async_engine=true` only selects vLLM's asynchronous
engine process. It is not an Async GRPO 1-off result unless
`grpo.async_grpo.enabled=true` as well. The 235B backend A/B above uses the
asynchronous vLLM engine but synchronous GRPO and colocated resources.

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
  VLLM_FUSE_ALLREDUCE_RMS=false LINEAR_BACKEND=flashinfer_cutlass \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
MODEL=qwen235b TOTAL_NODES=8 GPUS_PER_NODE=8 \
  VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  VLLM_FUSE_ALLREDUCE_RMS=false LINEAR_BACKEND=flashinfer_cutedsl \
  ACTION=submit experiments/pr3478_cutedsl_linear_ab/submit_gcp_nrt.sh
```

The primary metric is generation step time and derived generation
tokens/s/GPU. E2E, logprob, policy training, and refit metrics are retained as
controls.
