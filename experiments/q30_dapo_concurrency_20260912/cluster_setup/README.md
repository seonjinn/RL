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

## Submission receipts (2026-09-14 UTC)

| Site | Purpose | Job | Runtime source |
|---|---|---:|---|
| OCI-HSG | Publish pinned assets to PBSS | 7134539 | 70bc1adbb |
| Ptyche | Download + SHA256 check | 2819327 | cee6668e2 |
| Ptyche | Four-GPU import/overlay smoke | 2819329 | cee6668e2 |
| Lyris | Download + SHA256 check | 3049520 | cee6668e2 |
| Lyris | Four-GPU import/overlay smoke | 3049522 | cee6668e2 |

All submitted jobs passed `sbatch --test-only`; synthetic IDs are excluded.
Downloads were initially requested with a 30-minute begin delay while the
upload was queued; after upload started, StartTime was changed to now.
Each smoke depends on its own download using afterok + kill-on-invalid-dep.
Downloads use their cluster's existing private rclone config; no keys copied.

At 05:54:55 UTC upload 7134539 was RUNNING for 3m23s, not yet verified complete.
The OCI GPU-partition dry-run initially rejected a missing GPU request; the
actual submitted upload uses `--gpus-per-node=4 --exclusive`, batch, n3_post.
A planned move to cpu_datamover did NOT happen: git pull for the CPU scratch
fallback encountered `Disk quota exceeded` writing loose objects, and a
single pack-mode retry failed at fsync too. Global /home showed 36T free;
the user/backend quota cause needs further diagnosis. No files were deleted,
no original upload was cancelled, and no replacement upload was submitted.
The running original uses its previously committed source and /raid/scratch.
Do not change it while it is running. OCI runtime-source code is unchanged.

Ptyche and Lyris clean worktrees and all four recursive submodule SHAs were
verified. The older worktrees and mutable nightly links were preserved.
Setup is still IN PROGRESS: transfer completion, smoke results, DAPO and W&B
checks and a multi-node recipe canary remain outstanding. Do not use the
OCI-hardcoded submit.sh directly on either new cluster before adapting paths
and scheduler flags. Cross-cluster performance needs a local matched baseline.

Logs:
- OCI: existing DAPO experiment root, `cluster-setup-20260914/upload-7134539.log`.
- Ptyche: `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q30-dapo-ptyche-20260914/setup/`.
- Lyris: `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q30-dapo-lyris-20260914/setup/`.

At 06:01 UTC both download tasks were running and waiting for the publication
manifest. A read-only diagnostic inside OCI allocation 7134539 confirmed a
29-line SHA256 manifest on node-local scratch and an active rclone process;
the upload was past manifest generation but not yet confirmed complete.
Neither image download integrity nor GPU smoke is declared passing yet.
