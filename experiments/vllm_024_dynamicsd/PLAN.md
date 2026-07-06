# Experiment Plan

## Objective

Validate vLLM 0.24 DynamicSD on GB200, extend the long-tail Sync-RL coverage to
Qwen SWE 32K/64K request-plan profiles, and document SPEED-Bench official and
overlay cohorts without merging their protocols or inventing pending results.

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
9. Keep SPEED-Bench official and Sync-RL overlay rows in separate cohorts with
   explicit provenance matching and separate sampling defaults.
10. Report current completed local evidence accurately: Qwen3-32B Math
    DynamicSD summaries are complete, while pending SWE 32K/64K and
    SPEED-Bench rows stay unclaimed until their own `result.json` artifacts
    exist locally.

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
- Task 6 documentation is complete only when it distinguishes supported,
  integration-only, unsupported, pending, and completed rows without mixing
  official SPEED-Bench protocol with the NeMo-RL-matched overlay cohort.
