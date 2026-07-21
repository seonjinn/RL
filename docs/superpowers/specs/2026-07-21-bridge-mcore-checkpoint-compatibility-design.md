# Bridge–MCore Checkpoint Compatibility Design

## Decision

Forward-port the narrow checkpoint strategy migration from current Megatron-Bridge
into a dedicated `seonjinn/Megatron-Bridge` branch, then pin NeMo-RL to that
Bridge commit. Do not restore the removed deprecated MCore helper.

## Evidence

- Packed CUDA Graph MCore head `5be63d9f4` descends from the MCore change that
  removed `get_default_save_sharded_strategy`.
- The pinned Bridge commit `554c7b93` imports that helper unconditionally in
  `training/checkpointing.py`, preventing even a no-CG policy worker from
  initializing.
- Current MCore main also lacks the helper, so rebasing the CUDA Graph branch
  alone cannot resolve the API skew.
- Current Bridge replaces the helper with direct
  `TorchDistSaveShardedStrategy` construction. That is the only production
  behavior to forward-port.

## Alternatives Considered

1. Restore the deprecated MCore helper. Rejected: current MCore intentionally
   removed it, so this creates a divergent compatibility shim in the CUDA
   Graph branch.
2. Rebase MCore onto current main. Rejected as a standalone fix: current main
   also lacks the helper.
3. Forward-port Bridge's direct strategy migration. Chosen: minimal behavior
   change, aligned with current upstream Bridge, and keeps MCore current API
   intact.

## Implementation Boundary

1. Create a Bridge worktree at NeMo-RL's pinned Bridge base. Add a failing
   checkpointing import/unit test if current Bridge coverage does not exercise
   the strategy construction path.
2. Create and push a signed branch on `seonjinn/Megatron-Bridge` containing
   only the direct-strategy checkpoint migration and its test evidence.
3. Change the NeMo-RL Bridge submodule URL to the seonjinn fork and pin its
   gitlink to the verified Bridge commit. Preserve the verified MCore gitlink
   `5be63d9f4`.
4. Make runtime source provenance explicit: the fresh remote worktree's
   `Megatron-Bridge/src` must precede the container copy, and the strict probe
   must print the resolved Bridge checkpointing module path and require it to
   be under that fresh worktree.
5. Regenerate `uv.lock` only if `uv lock --check` proves the Bridge pin changes
   lock metadata. Reject a broad package-resolution drift.
6. Repeat the strict locked combined import, no-CG five-step smoke, Mamba
   five-step smoke, attn/router preflight, then matched 20/40-step matrix in
   the existing order. Checkpoint saving remains disabled and CUDA Graph
   warmup remains exactly three steps.

## Acceptance Criteria

- The checkpointing module imports successfully with MCore `5be63d9f4`.
- Fresh recursive clone resolves both MCore and Bridge gitlinks from seonjinn
  forks at the intended commits.
- Strict `uv run --locked --extra mcore` probe imports NeMo-RL, MCore, Mamba,
  Transformer Engine, and Bridge checkpointing from the fresh Bridge source.
- No-CG and Mamba five-step smokes run for at least five minutes; rejected
  attn/router cases fail before graph construction.
- Only then are matched 20/40-step jobs submitted.
