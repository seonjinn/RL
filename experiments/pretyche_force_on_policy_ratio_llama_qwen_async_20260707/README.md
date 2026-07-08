# Llama and Qwen Async Force-On-Policy Benchmark

This experiment compares `loss_fn.force_on_policy_ratio=false` and `true` for
the native four-GPU-per-node Pre-Tyche performance recipes approved in
`docs/superpowers/specs/2026-07-07-llama-qwen-async-force-on-policy-benchmark-design.md`.

The matrix contains eight direct 20-step jobs: Llama 3.1 8B 2n4g sync and
async-1off, Qwen3-30B-A3B 4n4g async-1off, and Qwen3-32B 8n4g async-1off.
Both sides use global batch size 2048. No 8g topology is included.

## Fixed identities

- Source SHA: `d4cfecf90db41cdf142629963b54b67ab479ab02`
- Container: `nemo_rl_nightly_20260630_0215.sqsh`
- Container SHA-256:
  `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`
- Cluster: Pre-Tyche `36x2-a01r`

The experiment performs resolved-config validation and eight
`sbatch --test-only` checks before real submission. It intentionally skips a
two-step model smoke at the user's request.
