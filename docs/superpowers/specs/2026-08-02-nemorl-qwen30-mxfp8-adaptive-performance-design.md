# NeMo-RL Qwen3-30B-A3B MXFP8 Adaptive Performance Design

## Objective

Measure whether the vLLM 0.25.1 FlashInfer TRTLLM adaptive dense MXFP8 path
improves Qwen3-30B-A3B rollout generation over the stock CuTeDSL linear
backend under a NeMo-RL performance workload.

## Workload Contract

Derive the generation workload from
`grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml` without creating training or refit
workers. Use the asynchronous performance recipe's generation slice:

- model: `Qwen/Qwen3-30B-A3B`
- generation precision: MXFP8 (`precision=fp8`, `is_mx=true`)
- BF16 exclusions: `q_proj`, `k_proj`, `v_proj`, and `o_proj`
- workload: 64 prompts and 32 generations per prompt, or 2,048 rollouts
- maximum sequence length and output budget: 4,096 tokens
- topology: two GB200 nodes, four GPUs per node, eight TP1/EP1 engines
- CUDA Graphs enabled (`enforce_eager=false`)
- identical prompts, seed, sampling parameters, and runtime for every arm

The primary metrics are generation tokens/s/GPU and generation time. Output
token counts and successful completion are validity gates.

## Staged Evaluation

### 1. Shape Trace

Run the direct TRTLLM path without a tactic table and record the real dense
MXFP8 execution signatures observed during the 2,048-rollout workload. A
signature consists of `(M, N_logical, N_physical, K, layout)` plus the runtime
fingerprint.

The stock Qwen performance recipe excludes the attention projections from
MXFP8. The trace therefore acts as a scope gate: if no dense MXFP8 signatures
are observed, the adaptive dense kernel mechanism has no eligible work in this
recipe and no offline shmoo is required.

### 2. Offline Profiling

If the trace contains eligible signatures, query the FlashInfer TRTLLM runner
for legal tactics and profile every legal tactic on the same GB200 software
stack. Build a Qwen-specific exact lookup table only from candidates that pass
the existing numerical, repeatability, CUDA Graph, and minimum-speedup gates.

Do not reuse the Nemotron3 Ultra table. Its projection dimensions and qualified
layer allowlist are model-specific even though its runtime fingerprint matches.

### 3. Matched A/B

Compare these two arms:

- baseline: `linear_backend=flashinfer_cutedsl`
- adaptive: `linear_backend=flashinfer_trtllm`, adaptive 8x4/128x4 layout,
  Qwen exact tactic table, and CuTeDSL fallback for unseen or unqualified work

Only the linear backend and adaptive lookup variables may differ. Exact misses
must fall back safely and are not failures.

### 4. Full GRPO Confirmation

Run the complete four-node synchronous Qwen performance recipe only if the
generation-only A/B reports a repeatable improvement. Compare steady-state
steps 5 through 9 and report generation tokens/s/GPU separately from end-to-end
training throughput.

## Failure Handling

- Treat zero output tokens, engine failures, OOM, NCCL failures, and incomplete
  jobs as invalid.
- Preserve logs and the exact resolved configuration for every arm.
- Retry infrastructure failures after identifying and fixing the root cause.
- Do not replace a failed configuration with a different topology or workload.
- Report an empty shape trace as a valid negative result, not as a profiler
  failure.

## Acceptance Criteria

1. The trace job completes with valid output tokens and records either a
   non-empty exact signature set or an explicit zero-eligible-shape result.
2. Any generated tactic table is tied to the exact Qwen signatures and runtime
   fingerprint.
3. Baseline and adaptive runs use the same workload and CUDA Graph settings.
4. The report states whether the adaptive path is exercised, its exact-match
   coverage, and the measured generation throughput ratio.
