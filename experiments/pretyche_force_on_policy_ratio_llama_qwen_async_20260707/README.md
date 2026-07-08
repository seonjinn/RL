# Llama and Qwen Async Force-On-Policy Benchmark

This experiment compares `loss_fn.force_on_policy_ratio=false` and `true` for
the native four-GPU-per-node Pre-Tyche performance recipes approved in
`docs/superpowers/specs/2026-07-07-llama-qwen-async-force-on-policy-benchmark-design.md`.

The original matrix contained eight direct 20-step jobs: Llama 3.1 8B 2n4g
sync and async-1off, Qwen3-30B-A3B 4n4g async-1off, and Qwen3-32B 8n4g
async-1off. The corrected retry contains only the six Async-1off jobs. Both
sides use global batch size 2048. No 8g topology is included.

## Fixed identities

- Source SHA: `d4cfecf90db41cdf142629963b54b67ab479ab02`
- Container: `nemo_rl_nightly_20260630_0215.sqsh`
- Container SHA-256:
  `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`
- Cluster: Pre-Tyche `36x2-a01r`

The retry performs resolved-config validation and six `sbatch --test-only`
checks before real submission. Qwen3-30B-A3B uses application and Slurm segment
2; Llama and Qwen3-32B omit both segment settings. It writes remote artifacts
to `pretyche_force_on_policy_ratio_async_retry_20260707` so the failed original
jobs and completed synchronous results remain unchanged.

`scripts/validate_config_contract.sbatch` runs the force-ratio unit tests and
the six-case resolved-config validator in the pinned nightly container before
the retry matrix is submitted.

See [REPORT.md](REPORT.md) for the completed synchronous results and the
terminal classification of the Async-1off topology failures.
