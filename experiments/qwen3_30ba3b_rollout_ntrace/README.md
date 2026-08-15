# Qwen3-30B-A3B rollout ntrace comparison

This experiment compares the GPU-time breakdown of BF16 and MoE-only MXFP8
rollout generation. Both arms use the same NeMo-RL source, Qwen3-30B-A3B
recipe, four-node allocation, asynchronous vLLM engine, CUDA Graph mode, and
matched generation workload. The BF16 arm uses the Qwen-compatible sparse
BF16 refit path. The MXFP8 arm uses the NCCL reshard receiver that converts
BF16 wire tensors to the MXFP8 destination layout. The optimizer learning rate
is zero in both arms so the captured rollout steps use the same policy weights.
Refit time is reported separately from the generation-kernel breakdown.

The outer rollout profiler hook currently runs in the synchronous GRPO trainer.
The experiment therefore disables `async_grpo` while retaining
`policy.generation.vllm_cfg.async_engine=true`. Results from this directory are
rollout-kernel comparisons, not async 1-off pipeline throughput results.

## Capture window

- generation workers: 8 GPUs, TP1/PP1/EP1
- warm-up rollouts: 1
- measured steady-state rollouts: 3
- captured generation ranks: 0-7
- CUDA graph provenance: engine initialization
- model scope: routed MoE experts only for MXFP8; attention, MLP gate, and
  `lm_head` stay BF16

The ntrace source and runtime commits are recorded in each result directory.
After capture, run the graph replay audit before producing breakdowns.

## Launch

`run_capture.sh` expects these environment variables:

```bash
NTRACE_ARM=bf16|mxfp8
NTRACE_RUNTIME=/shared/path/to/ntrace-runtime
NTRACE_SOURCE_COMMIT=<commit>
NEMO_SOURCE_COMMIT=<commit>
NTRACE_RESULTS_ROOT=/shared/path/to/results
```

Use `tools/launch` so the NeMo-RL code snapshot and exact git revision are
stored with the job.

After a capture completes, run the artifact and CUDA Graph gates before using
the profile:

```bash
uv run experiments/qwen3_30ba3b_rollout_ntrace/analyze_capture.sh \
  /path/to/run-root \
  /path/to/ntrace-source \
  /path/to/ntrace-runtime
```
