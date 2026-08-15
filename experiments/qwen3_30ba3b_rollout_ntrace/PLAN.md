# Plan

1. Run the profiler/MXFP8 unit gates and build the native ntrace runtime from
   the pinned ntrace commit.
2. Capture one warm-up and three steady-state BF16 rollouts.
3. Capture the same window with MoE-only MXFP8 and `flashinfer_trtllm`.
4. Audit CUDA Graph replay provenance and reject incomplete captures.
5. Generate per-rank and multi-rank breakdowns.
6. Compare MoE FC1/FC2, dense projections, attention, quantization/layout,
   collectives, scheduler work, and idle time.
7. Report the measured speedup and the Amdahl upper bound from the MoE share.
