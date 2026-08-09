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

## Execution History

| Jobs | Outcome | Interpretation |
| --- | --- | --- |
| `507182`, `507183` | Failed during environment setup | The original container did not provide the lockfile-required Python 3.13.14 interpreter. |
| `507329`, `507330` | Cancelled | The direct container interpreter used NCCL 2.30.4 while the source lock required NCCL 2.30.7, so this pair was not a valid source/runtime comparison. |
| `507350`, `507351` | Failed during worker-venv setup | Builders on multiple nodes concurrently rebuilt the same Lustre venv and raced in `rmtree` and package installation. No model, refit, or training step ran. |
| `508251`, `508252` | Cancelled during worker-venv setup | Node-local venvs removed the directory race but repeated dependency fetches on every node. One `TransferQueue` fetch failed with `curl 56`/early EOF and another `uv sync` stalled. |
| `508298`, `508299` | Cancelled after topology audit | The coordinated venv path was valid, but the inherited EP16 and PP4 required 64 trainer ranks while the non-colocated split supplied 32. |
| `508312`, `508313` | Submitted | Uses the coordinated shared runtime plus trainer TP2/PP4/CP2/EP8/ETP1, which is valid for the 32-rank trainer partition. |

Only runs that reach measured GRPO steps are eligible for the performance
comparison.
