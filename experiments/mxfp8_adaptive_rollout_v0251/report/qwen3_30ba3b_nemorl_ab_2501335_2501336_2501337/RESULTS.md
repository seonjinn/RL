# Qwen3-30B-A3B NeMo-RL MXFP8 Adaptive A/B

## Configuration

- Model: `Qwen/Qwen3-30B-A3B`
- Hardware: 2 nodes, 4 GB200 GPUs per node, 8 TP1/EP1 engines
- Workload: 64 prompts, 32 samples per prompt, 2,048 responses
- Per-engine capacity: `max_num_seqs=256`
- Token budget: `max_num_batched_tokens=16384`, `max_new_tokens=4096`
- Sampling: temperature 1.0, top-p 1.0, top-k disabled
- Execution: CUDA Graph enabled, chunked prefill enabled, prefix caching disabled
- Measurement: one direct generation call per arm; model initialization excluded
- Jobs: `2501335`, `2501336`, `2501337`

The baseline uses NeMo-RL's refit-compatible FlashInfer CUTLASS MXFP8 linear path.
Adaptive selection routes the completely qualified output-projection family
`(N=151936, K=2048)` through the direct TRTLLM path and uses CuTeDSL fallback
for the unqualified gate family `(N=128, K=2048)`.

## Results

| Repeat | Baseline tok/s/GPU | Adaptive tok/s/GPU | Adaptive / baseline |
|---:|---:|---:|---:|
| 1 | 1,911.61 | 1,889.90 | 0.9886x |
| 2 | 1,891.85 | 1,892.23 | 1.0002x |
| 3 | 1,924.66 | 1,902.75 | 0.9886x |
| Median | 1,911.61 | 1,892.23 | 0.9886x paired median |

Both arms produced exactly 1,046,528 output tokens in every repeat. All six
arms completed CUDA Graph capture, and no OOM, NCCL, or runtime error was
observed.

Adaptive selection did not improve this Qwen NeMo-RL workload. The paired
median is a 1.1% throughput regression. The one-time median model setup time
was also longer for Adaptive selection (557.0 s versus 398.8 s) because the
non-allowlisted CuTeDSL fallback still performs startup autotuning. Setup time
is excluded from the throughput result above.

## Interpretation

The offline qualification reference and the end-to-end baseline are not the
same path. The 23 output-projection tactics were qualified against stock
CuTeDSL, while NeMo-RL replaces the stock path with CUTLASS to preserve the
refit `[N,K]` weight contract. A tactic that beats CuTeDSL is therefore not
guaranteed to beat the actual NeMo-RL CUTLASS baseline.

Only the output projection is allowlisted. The gate projection executes once
per layer and falls back to CuTeDSL, while MoE expert kernels and the rest of
the rollout remain unchanged. The optimized family therefore has limited
end-to-end weight, and its microbenchmark savings do not offset the additional
layout and dispatch costs in this run.

## Next Decision

Do not enable this table for Qwen NeMo-RL production based on the current
result. First profile the exact observed shapes against the actual NeMo-RL
CUTLASS baseline and measure the call-weighted GPU-time share with NSys. Build
a new table only for shapes that repeatedly beat CUTLASS under CUDA Graph.

The current workload already assigns about 256 requests to each engine and
fills `max_num_seqs=256`. A larger stress test would require 4,096 responses,
`max_num_seqs=512`, a new CUDA Graph shape trace, and a new profiling pass. It
should follow only if the CUTLASS-referenced microbenchmark shows a credible
end-to-end opportunity.
