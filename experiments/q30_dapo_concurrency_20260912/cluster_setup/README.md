# Lyris / Ptyche parity setup

This is infrastructure preparation, not a new performance cohort. Preserve
OCI jobs and all existing remote worktrees and mutable nightly symlinks.

| Site | SSH | New source directory | Account | Partition |
|---|---|---|---|---|
| Lyris | login-lyris | /home/sna/nemorl-q30-dapo-lyris-20260914 | coreai_dlalgo_llm | gb200 |
| Ptyche | login-ptyche | /home/sna/nemorl-q30-dapo-ptyche-20260914 | coreai_dlalgo_llm | batch |

Source baseline: c9775f033 (same runtime code as the OCI DAPO jobs), followed
only by cluster-setup artifacts. Both worktrees initialize submodules recursively.
The old generic setup helper was not run unmodified: it resets branches,
defaults code/caches to Lustre, and selects an old runtime branch. Its sequence
is applied with non-destructive git worktree creation, current pinned source,
node-local caches and explicit image paths instead.

The transfer follows the PBSS SLURM skill with bounded task-specific sbatch
files, moderate parallelism, immutable copies and SHA256 validation. Upload
the exact OCI image, target and both 44K Base drafters once, then download
to `/lustre/fsw/coreai_dlalgo_llm/users/sna/q30-dapo-parity-20260914` on each
cluster. No credentials are copied across clusters. The manifest appears
only after successful upload. Download jobs wait up to 45 minutes for it;
if upload is still queued, leave downloads unsubmitted until upload starts.

Image SHA256: `60cfa5cb28834f1a4c154103a993fb87fdf5d38a0fa29ebd3ae37b63b492093c`.
Existing September 9/12 images are not presumed equivalent by date or filename.

For every submission: commit/push, remote pull, `sbatch --test-only`, then
submit with explicit account, partition, job name and output in the experiment
directory. Smoke depends on successful local download, not another cluster's
job ID. No cross-cluster scheduler dependencies are valid.

Smoke checks four GPUs, NeMo-RL/Megatron/TE/vLLM imports and the DSpark overlay.
It does not establish model execution, DAPO dataset availability, W&B ingestion,
refit correctness or multi-node graph coverage. Those require a subsequent
one-step recipe canary and site-specific launch adaptation before benchmarking.
