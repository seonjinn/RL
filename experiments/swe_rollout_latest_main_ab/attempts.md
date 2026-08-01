# SWE Rollout Latest-Main A/B Attempts

This log distinguishes environment and harness failures from optimization
results. None of the attempts below reached the SWE rollout workload, so they
must not be interpreted as performance regressions.

## 2026-07-31 — local PR unit-test setup

The first `uv run pytest` attempt failed before collection because the new
worktree's Gym submodule was uninitialized:

```text
Failed to parse entry: `nemo-gym`
`nemo-gym` references a workspace in `tool.uv.sources`, but is not a workspace member
```

After Gym initialization, Automodel was also missing. Recursive submodule
initialization resolved both issues. The macOS retry then stopped before
collection because the repository lock supports Linux x86_64/aarch64 only.
Linux verification was therefore moved to OCI-HSG.

## 2026-07-31 — OCI-HSG Linux verification

Frozen source under test:

- NeMo-RL arm: `f4b6e964302b692f4930b30f5f43f21c5fe1b523`
- Gym gitlink: `473f446f71ec7c1243eb1517fe2440a2b37fe68b`
- image: `nemo_rl_nightly_20260731_17d53441_20260731_5752984.sqsh`
- image source commit: `1afc767cd6c003ec58f9c2dfdfba448ade161f41`
- image SHA256: `1d0ab3df4ceccbc8e91b6954dc2ba2b0b3ddbe9d05124111fbbfdbc1017a2294`

Attempts:

1. Direct `sbatch --container-image=...` was rejected because OCI-HSG exposes
   the Pyxis container options on `srun`, not on `sbatch`. No job was created.
2. Job `5753915` passed `--chdir=/workspace` to the allocation before the
   container mount existed. It was cancelled after the log proved the working
   directory was invalid.
3. Job `5753938` changed to `cd /workspace` inside the container. It reached
   the correct NeMo-RL and Gym commits but `/opt/nemo_rl_venv` could not import
   vLLM.
4. Job `5754012` inspected the image. `/opt/ray_venvs/*/bin/python` symlinks
   were unresolved and the visible `/opt/nemo_rl_venv` lacked the expected
   backend imports.
5. Job `5754111` attempted a copied writable venv. The visible Python was
   3.13.11 while latest main requires `>=3.13.13,<3.14`, so `uv sync` stopped
   before installing dependencies.
6. Job `5754165` explicitly selected `/opt/nemo_rl_venv/bin/python`; it still
   resolved to Python 3.13.11 and stopped on the same project requirement.
7. Job `5754256` probed `uv python install 3.13.13`. The log exposed uv 0.11.1
   and `/root/.local` Python 3.13.11, although the image Dockerfile pins uv
   0.11.18 and Python 3.13.13. This discrepancy identified the root cause:
   OCI-HSG's default home bind mount was shadowing the image's `/root/.local`
   tree, which also broke the venv symlinks created from that interpreter.

Corrective action: all subsequent OCI-HSG container steps use
`srun --no-container-mount-home`. Image-integrity probe `5754338` then exposed
uv 0.11.18 and Python 3.13.13 as pinned by the image and completed successfully.
The image's default venv intentionally omitted vLLM after warming backend
caches, so Linux verification job `5754510` created the locked `vllm` extra
environment before the focused test suite. That attempt was cancelled after
five minutes of monitoring because `UV_LINK_MODE=copy` was physically copying
the image's roughly 100 GB backend cache onto Lustre. The partial, ignored
`.venv` was the only path removed. The retry uses the image's intended
`symlink` link mode, matching `docker/Dockerfile`. Rollout canaries remain
pending until the focused tests pass.
