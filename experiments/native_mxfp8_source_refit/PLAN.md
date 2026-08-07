# Validation Plan

1. Run repository static and dependency-light unit tests.
2. Run `sbatch --test-only` on GCP-NRT with the four-node B200 topology.
3. Submit a two-step run and monitor initialization and the first refit.
4. Require two completed steps, finite loss/reward metrics, and no refit or
   weight-layout errors.
5. Submit the identical configuration for 20 steps.
6. Record job IDs, W&B links, timing, and correctness metrics in the report.
