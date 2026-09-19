# Execution plan

1. Remeasure the no-SpecDec baseline for 20 steps with the official 16n4g
   recipe and no scheduler-concurrency override.
2. Stage the matched base-target PTV2-en step-25391 DFlash and DSpark B8/B16
   exports from OCI-HSG to Lyris with checksums.
3. Gate DFlash and DSpark B8 at K5/K7, B16 at K11/K13, and EAGLE-3 at K3/K5.
   Keep the official workload fixed; add only frozen SpecDec runtime settings,
   S64, and method-aware FAP CUDA Graph shapes.
4. Promote every successful one-step gate to a matched 20-step run and compare
   steps 3–20 against the no-SpecDec baseline.
5. Report generation throughput, generation time, E2E step time and throughput,
   reward, generation length, acceptance rate, and KL/error metrics with W&B
   links and exact receipts.
