# Qwen3-30B-A3B NeMo-RL Refit CUTLASS Shmoo Validation

## Question

Does an offline, exact-shape TRTLLM tactic table improve Qwen3-30B-A3B
NeMo-RL generation after accounting for both the backend change and the actual
refit-compatible baseline?

## Configuration

- Model: `Qwen/Qwen3-30B-A3B`
- Hardware: two Ptyche nodes with four GB200 GPUs per node
- Parallelism: eight independent TP1/EP1 vLLM engines
- Workload: 64 prompts, eight samples per prompt, 512 generations per arm
- Sampling: temperature 1.0, top-p 1.0, top-k disabled
- Limits: `max_new_tokens=32768`, `max_model_len=36864`
- Scheduler: `max_num_seqs=32`, `max_num_batched_tokens=16384`
- Execution: CUDA Graph enabled, chunked prefill enabled, prefix caching disabled
- Measurement: one generation call per arm; model setup time excluded
- Jobs: `2502042`, `2502477`, and `2502478`
- vLLM runtime commit: `658d7b1571a914bee7df48f717c2a428ee7c45ad`

The three arms isolate two effects:

1. **Refit CUTLASS baseline:** NeMo-RL's refit-compatible MXFP8 linear path.
2. **TRTLLM default:** direct TRTLLM for the allowlisted output-projection
   family, using the runner's default tactic selection.
3. **Adaptive exact shmoo:** the same TRTLLM routing and layout policy, plus
   exact tactics selected by repeated offline profiling.

The qualified table contains tactics 70, 71, and 69 for traced physical shapes
with `M=1`, `M=31`, and `M=32`, respectively, in the
`(N=151936, K=2048)` family. Its SHA-256 is
`04eace273bc56e952e1c418ae4f5b3f0481c9280154d80c0e5406287aca95daa`.
Unseen shapes, including an `M=10` shape observed during this evaluation, use
the backend default instead of failing.

## Results

All nine arms completed one generation call and produced exactly 2,097,152
output tokens. CUDA Graph capture completed for every arm, and the logs contain
no OOM, NCCL, Ray, or vLLM engine failure.

| Repeat | Refit CUTLASS baseline | TRTLLM default | Adaptive exact shmoo | TRTLLM / baseline | Adaptive / baseline | Adaptive / TRTLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 349.61 tok/s/GPU | 352.60 tok/s/GPU | 355.58 tok/s/GPU | 1.0085x | 1.0171x | 1.0085x |
| 2 | 349.35 tok/s/GPU | 352.28 tok/s/GPU | 359.22 tok/s/GPU | 1.0084x | 1.0283x | 1.0197x |
| 3 | 351.56 tok/s/GPU | 353.46 tok/s/GPU | 360.10 tok/s/GPU | 1.0054x | 1.0243x | 1.0188x |
| Paired median | 349.61 tok/s/GPU | 352.60 tok/s/GPU | 359.22 tok/s/GPU | **1.0084x** | **1.0243x** | **1.0188x** |

Across the three repeats, switching from refit CUTLASS to TRTLLM default
improved generation throughput by 0.54--0.85%. Adding the exact offline tactic
table improved throughput by another 0.85--1.97% over TRTLLM default. The
Adaptive arm improved by 1.71--2.83% over the refit CUTLASS baseline, with a
paired median gain of 2.43%.

## Interpretation

The refit-CUTLASS-referenced shmoo produces a small but repeatable end-to-end
gain. The large per-GEMM microbenchmark speedups do not translate directly to
rollout throughput because the allowlisted output projection is only one part
of generation. MoE expert kernels, attention, sampling, scheduling, and
unmatched dense shapes are unchanged. The observed end-to-end gain is therefore
about 2.4%, not the roughly 2x speedup measured for selected isolated GEMMs.

The offline table itself adds no profiling work to a rollout request. It is
loaded during worker initialization, and runtime performs an exact lookup.
Unknown signatures safely retain the backend default behavior.

## Scope Limitation

This experiment configured a 32K output cap, but it did not exercise 32K output
sequences. The 512 responses in every arm contained exactly 4,096 output tokens
each. The result therefore validates the three-arm mechanism under a
`max_new_tokens=32768` configuration, not throughput for actual 32K long-tail
responses. A true 32K-output study needs requests that suppress early stopping
or otherwise force the intended generated length, followed by a new shape trace
and qualification pass.

The output-token equality and existing numerical/CUDA-Graph microbenchmark
gates establish performance comparability and kernel-level safety. This run is
not a semantic-accuracy evaluation; a matched task-level correctness gate is
still required before production enablement.

## Decision

Retain the refit-CUTLASS-qualified Adaptive policy as a promising opt-in path.
The three-repeat result shows a consistent but modest gain. Do not enable it by
default until a matched correctness evaluation and an actual long-tail rollout
study confirm the benefit under the target NeMo-RL distribution.
