# Experiment Plan

## Objective

Validate vLLM 0.24 DynamicSD on GB200 and determine whether adaptive K improves
Qwen3-32B Eagle-3 generation throughput across batch size at temperature 0 and
temperature 1.

## Validation Gates

1. Stage and checksum the official ARM64 image.
2. Confirm `vllm==0.24.0`, `aarch64`, target load, drafter load, and active
   speculative-decoding counters.
3. Materialize and checksum pinned DAPO and OpenMathInstruct-2 prompt JSONLs.
4. Complete baseline/static/dynamic BS 1/2 smoke rows at both temperatures.
5. Complete the barriered synchronous-rollout smoke with active SpecDec metrics.
6. Run the full BS 1-64 and real-dataset rollout matrices only after smoke.
7. Profile one regressed and one improved batch size with NSys.
8. Compare only rows matched on model, TP/PP, ISL/OSL, sampling, graph mode,
   prefix caching, chunked prefill, and token budget.

## Primary Metrics

- Output tok/s/GPU and speedup over the matched baseline.
- Acceptance rate and mean acceptance length.
- Latency per fixed-size generation batch.
- NSys GPU-kernel time and CPU/API launch gaps by NVTX range.

## Exit Criteria

- DynamicSD is operational when draft-token counters are positive and K changes
  at configured scheduler batch-size boundaries.
- A performance claim requires three measured repeats and token-complete output.
- A root-cause claim requires an NSys trace with initialization and graph capture
  excluded from the measurement window.
