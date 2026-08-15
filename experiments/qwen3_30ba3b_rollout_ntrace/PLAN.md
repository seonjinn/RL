# Plan

1. [x] Run the profiler/MXFP8 unit gates and build the native ntrace runtime from
   the pinned ntrace commit.
2. [ ] Capture a matched BF16 control. The current Qwen sparse-refit path fails
   on the expert W2 2D-to-3D destination layout before generation.
3. [x] Capture one warm-up and three measured rollouts with MoE-only MXFP8 and
   `flashinfer_trtllm`.
4. [x] Audit CUDA Graph replay provenance and reject incomplete captures.
5. [x] Generate per-rank and multi-rank breakdowns.
6. [x] Compare MoE FC1/FC2, dense projections, attention, quantization/layout,
   collectives, scheduler work, and idle time.
7. [x] Report the Amdahl upper bound from the MoE share. Do not report a matched
   BF16 speedup until step 2 succeeds.
