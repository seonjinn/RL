# Validation plan

1. Run the focused unit suite and formatting checks.
2. Render both Ptyche jobs and run `sbatch --test-only`.
3. Submit the legacy baseline and DFlash K7 deep-refit gates independently.
4. Monitor both jobs for at least five minutes using one filtered scheduler
   query per interval.
5. Validate refit phase ordering, host/CUDA memory snapshots, acceptance,
   output-integrity metrics, and W&B run completion.
6. If both gates pass, submit matched 20-step runs without changing the recipe,
   concurrency, CUDA Graph buckets, target, drafter, or Ray threshold.
7. Summarize Steps 3–20 and update the cross-model Generation/E2E figure.
