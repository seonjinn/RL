# Plan

1. Capture one steady-state synchronous MXFP8 refit step with CUDA Graphs on.
2. Attribute GPU time and memory traffic to quantization, transfer/load,
   weight permutation, scale interleave, and destination copies.
3. Benchmark the smallest memory-bounded change against the current batched
   implementation.
4. Run value parity, CUDA Graph pointer-stability, and peak-memory gates.
5. Run a matched 20-step end-to-end A/B only after the microbenchmark gate.

