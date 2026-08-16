# Plan

1. [x] Run the profiler/MXFP8 unit gates and build the native ntrace runtime from
   the pinned ntrace commit.
2. [x] Capture a matched BF16 control with native batched TRTLLM expert reload.
3. [x] Capture one warm-up and three measured rollouts with exact expert-only
   MXFP8 and `flashinfer_trtllm`.
4. [x] Audit CUDA Graph replay provenance and reject incomplete captures.
5. [x] Generate per-rank and multi-rank breakdowns.
6. [x] Compare MoE FC1/FC2, dense projections, attention, quantization/layout,
   collectives, scheduler work, and idle time.
7. [x] Report token-normalized matched throughput and the remaining optimization
   targets.
