# Qwen3-30B-A3B rollout ntrace comparison

This experiment compares the GPU-time breakdown of BF16 and MoE-only MXFP8
rollout generation. Both arms use the same NeMo-RL source, Qwen3-30B-A3B
recipe, four-node allocation, asynchronous vLLM engine, CUDA Graph mode, and
matched generation workload. Both arms use the FlashInfer TRTLLM MoE backend
and NCCL Reshard refit. The MXFP8 receiver converts BF16 wire tensors to the
MXFP8 destination layout. The optimizer learning rate is zero so the captured
rollout steps use fixed policy weights. This replaces the earlier BF16 sparse
refit control, which failed before generation on Qwen's expert W2 layout.

The outer rollout profiler hook currently runs in the synchronous GRPO trainer.
The experiment therefore disables `async_grpo` while retaining
`policy.generation.vllm_cfg.async_engine=true`. Results from this directory are
rollout-kernel comparisons, not async 1-off pipeline throughput results.

## Capture window

- generation workers: 8 GPUs, TP1/PP1/EP1
- warm-up rollouts: 1
- measured rollouts: 3; the first includes stack-capture overhead, so the
  throughput comparison uses the later two windows
- captured generation ranks: 0-7
- CUDA graph provenance: engine initialization
- model scope: routed MoE experts only for MXFP8; attention, MLP gate, and
  `lm_head` stay BF16

The ntrace source and runtime commits are recorded in each result directory.
After capture, run the graph replay audit before producing breakdowns.

The current result and its compact source data are in
[`report/rollout_bottleneck_analysis.md`](report/rollout_bottleneck_analysis.md).

## Launch

`run_capture.sh` expects these environment variables:

```bash
NTRACE_SOURCE=/shared/path/to/pinned-ntrace-source
NTRACE_RUNTIME=/shared/path/to/ntrace-runtime
NTRACE_SOURCE_COMMIT=<commit>
NEMO_SOURCE_COMMIT=<commit>
NTRACE_RESULTS_ROOT=/shared/path/to/results
```

Launch `run_capture_bf16.sh` and
`run_capture_mxfp8_moe_only_exact_scope.sh` as separate jobs. The wrappers set
the arm and give each `tools/launch` job its own code snapshot. The exact-scope
wrapper locks attention, the router gate, and `lm_head` to BF16 after all
forwarded command-line overrides.

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
