# Qwen3-235B MXFP8 Adaptive Canary Plan

## Goal

Measure whether the corrected FlashInfer TRTLLM dense MXFP8 adaptive path improves Qwen3-235B generation on eight GB200 GPUs without changing output validity or CUDA Graph execution.

## Global Constraints

- Run on Ptyche with the pinned NeMo-RL branch, custom vLLM 0.25.1 source, container, and runtime overlay used by the qualified Qwen3-30B canary.
- Start with a generation-only shape-trace gate. Do not submit the tactic sweep or A/B comparison until startup succeeds and eligible dense MXFP8 calls are observed.
- Use TP4 and validate the expert-parallel configuration required by the FlashInfer TRTLLM MoE path.
- Keep Q/K/V/O projections and router gates out of MXFP8 where required by the existing refit contract.
- Keep CUDA Graph enabled (`enforce_eager: false`).
- Treat tactic-table misses as fallback, not errors.
- Require expected output-token counts before reporting performance.
- Do not change existing Qwen3-30B artifacts or production recipes.

## Task 1: Add Qwen3-235B Generation-Only Trace Gate

Add a dedicated eval configuration and Ptyche submitter by reusing the existing Qwen3-30B trace workflow. The configuration must identify Qwen3-235B explicitly, use an eight-GPU TP4 topology, preserve the current MXFP8/refit exclusions, and write to a separate result root. Add focused static tests for the configuration and submitter contract.

Verification:

- YAML parsing succeeds.
- Shell syntax checks succeed.
- Focused tests assert model, topology, CUDA Graph, quantization exclusions, clean provenance checks, and SLURM segment/time settings.

## Task 2: Submit And Validate The Trace Gate

Commit and push Task 1, pull the clean branch on Ptyche, run `--test-only`, then submit. Monitor for at least five minutes after the job starts. Validate model availability, runtime-overlay provenance, TP4/EP startup, CUDA Graph preparation, and shape-trace output.

## Task 3: Produce Qualified Tactic Artifacts

If eligible dense MXFP8 shapes are observed, run the existing exact-shape TRTLLM tactic sweep for each physical `(M,N,K,layout)` with repeated timing and correctness checks. Build a Qwen3-235B-specific exact table and layer allowlist. If no eligible shapes are observed, stop and report that dense adaptive selection does not apply to this quantization scope.

## Task 4: Run Three-Arm Performance Comparison

Run matched repeats of:

1. CuTeDSL/refit baseline.
2. Direct TRTLLM with default tactic.
3. Direct TRTLLM with the qualified exact table.

Require identical request/input settings, expected token counts, and CUDA Graph mode. Report output tokens/s/GPU, TTFT, TPOT, table hit/miss counts, and paired ratios.

## Task 5: Correctness And Report

Run a small correctness smoke test followed by the matched 1,319-example GSM8K gate if the adaptive arm shows a useful performance result. Record exact revisions, container, topology, artifact digests, command lines, job IDs, and all validation outcomes in Markdown and HTML reports.
