# Timeline

## 2026-08-05 03:09 PDT

- User requested the cumulative PR 3477/3478 step-time effect and asked whether
  BF16 training with NVFP4 rollout had been tested.
- Verified that the implementation and smoke matrix exist but no GPU job has
  been submitted for this exact precision combination.
- Selected GCP-NRT B200 for the runtime validation.

## 2026-08-05 03:13 PDT

- `bash -n` and both smoke scripts' static dry-run contracts passed.
- Local pytest stopped before collection because the Gym workspace submodule is
  not initialized in the local worktree. Deferred the focused pytest to the
  recursive-submodule GCP clone and target container.

## 2026-08-05 03:19 PDT

- Cloned commit `441dc40df` recursively to GCP-NRT.
- Added and syntax-checked `submit_gcp_nrt.sh` with unique code snapshots,
  commit-versioned caches, `--gpus-per-node=8`, and `sbatch --test-only`.
