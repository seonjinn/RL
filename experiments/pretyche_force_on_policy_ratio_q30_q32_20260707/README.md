# Qwen3 Force-On-Policy-Ratio Benchmark

This experiment measures the isolated performance effect of
`loss_fn.force_on_policy_ratio=true` for the existing synchronous Qwen3-30B-A3B
and Qwen3-32B 4n4g performance recipes on Pre-Tyche.

## Fixed inputs

- NeMo-RL source: `d4cfecf90db41cdf142629963b54b67ab479ab02`
- Container SHA-256:
  `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`
- Immutable container path:
  `containers/nemo_rl_nightly_20260630_0215.sqsh`
- Cluster: Pre-Tyche GB200-NVL36, partition `36x2-a01r`
- Topology: 4 nodes, 4 GPUs per node, segment size 4
- Recipes: `grpo-qwen3-30ba3b-4n4g.yaml` and
  `grpo-qwen3-32b-4n4g.yaml`
- Global batch size: 2048 for both control and treatment
- Steps: 2-step smoke, followed by 20-step performance only after all smoke
  jobs pass

The paired runs differ only in `loss_fn.force_on_policy_ratio`. Reference-policy
logprobs remain enabled because `loss_fn.reference_policy_kl_penalty=0.01`.

## Layout

- `manifests/config_contract.tsv`: four-case benchmark contract
- `scripts/validate_config_contract.py`: resolved-config and paired-equality checks
- `scripts/validate_config_contract.sbatch`: containerized unit/config validation
- `scripts/run_force_on_policy_benchmark.sbatch`: immutable benchmark runner
- `scripts/submit_force_on_policy_matrix.sh`: test-only, smoke, and gated performance submission
- `results/`: job manifests and collected metrics after remote execution

## Submission sequence

Run config validation first, then check schedulability and submit smoke jobs:

```bash
sbatch scripts/validate_config_contract.sbatch
TEST_ONLY=1 SMOKE_ONLY=1 scripts/submit_force_on_policy_matrix.sh
TEST_ONLY=0 SMOKE_ONLY=1 AFTEROK_JOB_ID=<validation-job-id> \
  scripts/submit_force_on_policy_matrix.sh
```

`AFTEROK_JOB_ID` is optional. When set, every submitted case records and uses a
SLURM `afterok` dependency so queued smoke jobs cannot start before validation
completes successfully.

The validation job starts from a clean source checkout and removes only the
`tests/unit/unit_results*` artifacts that the selected pytest tests create, so
the immutable-source guard remains clean for the benchmark jobs.

After all four smoke jobs complete successfully, the submitter independently
checks the smoke manifest, SLURM state, step-2 marker, fatal signatures, and
force-on-policy skip markers before allowing performance submission:

```bash
TEST_ONLY=1 SMOKE_ONLY=0 scripts/submit_force_on_policy_matrix.sh
TEST_ONLY=0 SMOKE_ONLY=0 scripts/submit_force_on_policy_matrix.sh
```

## Analysis contract

Matched performance metrics exclude warm-up step 1 and validation-bearing steps
10 and 20. The final report will compare E2E, generation, policy training,
logprob time, duration-weighted tokens/s/GPU, and peak-memory behavior.
