# Execution plan

1. Remeasure the no-SpecDec baseline for 20 steps with the official 16n4g
   recipe and no scheduler-concurrency override.
2. Validate DSpark K3/K5/K7 with SpecDec-only changes, initially at S64, using
   CUDA Graph widths derived from each K.
3. Run an isolated DFlash2 recognition and one-step execution gate. Do not mix
   an unsupported DFlash2 runtime with the controlled baseline or DSpark runs.
4. Promote successful arms to matched 20-step runs and compare steps 3–20.
5. Report generation throughput, generation time, E2E step time and throughput,
   reward, generation length, acceptance rate, and KL/error metrics with W&B
   links and exact receipts.
