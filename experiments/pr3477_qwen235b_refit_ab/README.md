# PR 3477 Qwen3-235B Refit A/B

This experiment measures whether PR 3477's NCCL-Reshard path works for BF16
training plus MXFP8 rollout on Qwen3-235B-A22B, and how much refit time it
saves versus the legacy non-colocated collective path.

The pair uses 8 GCP-NRT B200 nodes (64 GPUs), preserving the GPU budget and
parallelism of the upstream `16n4g` performance recipe. Only
`policy.generation.refit_transport` differs between arms.

See [PLAN.md](PLAN.md) for the fixed setup and commands. Runtime metadata,
SLURM logs, and W&B run identifiers are written under the remote experiment
root printed by the submission script.
