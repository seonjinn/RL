# DFlash Policy and Refit Optimization Gate

This directory contains the OCI-HSG correctness gate for the DFlash policy and
refit optimizations described in
`docs/superpowers/specs/2026-08-21-dflash-policy-refit-optimization.md`.

The gate validates:

- hidden-state capture materialization for microbatch sizes one and two;
- exact logical DFlash export reconstruction under TP2;
- one NCCL payload gather per `(device, dtype)` bucket;
- synchronized rejection of rank-asymmetric export manifests before payload
  collectives.

Submit only from a clean, pushed commit after an `sbatch --test-only` probe:

```bash
sbatch --test-only \
  --account=<best-fairshare-account> \
  --export=ALL,SOURCE=<clean-source>,EXPECTED_SHA=<full-sha>,FINAL_ROOT=<durable-root> \
  research/dflash_policy_refit_opt/submit_oci_correctness.sh
```

The job writes live state to node-local `/raid/scratch` and copies its final log
and result manifest to `FINAL_ROOT` on exit.
