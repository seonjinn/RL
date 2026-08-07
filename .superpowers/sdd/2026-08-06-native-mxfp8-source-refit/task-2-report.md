# Task 2 Report: Component-Aware NCCL Refit Metadata

## Status

Implemented in the requested worktree.

Modified files:

- `nemo_rl/weight_sync/nccl_reshard_utils.py`
- `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- This report file

No transfer loops or backend maps were changed. The unrelated untracked
`docs/superpowers/` content was not staged.

## Implementation

- Added normalized ordered component metadata to every `param_info`.
- Preserved legacy metadata by generating an implicit `weight` component.
- Added serialized dtype validation, including `torch.uint8` and `uint8` for
  native MXFP8 scales.
- Added validation for nonempty components, unique roles, positive logical
  shapes, required `weight`, weight shape matching, and compact
  `weight_scale` shape matching (`[..., K / 32]`).
- Derived component placements from the parent HF weight name, so gate/up
  scales use column-parallel dimension 0 and down scales use compressed
  dimension 1.
- Validated generated shard dimensions against each component rank.
- Sorted individual experts by numeric expert index and prepended the global
  expert dimension to every component shape in the same order.
- Preserved explicit components for pre-grouped gate/up expert metadata while
  splitting the fused dimension for both values and scales.
- Restored nested component placements after collective RPC serialization.

## TDD Evidence

The initial focused component test failed with `KeyError: 'components'`.
The nested placement integration test initially failed because nested
`src_placements` remained `{"dim": 1}` instead of becoming `Shard(1)`.
Both failures were addressed by production changes, then the tests passed.

## Verification

The normal project command was attempted exactly as specified:

`uv run pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py`

It is blocked before test collection by the local dependency graph:

`Failed to parse entry: nemo-gym; nemo-gym references a workspace in tool.uv.sources, but is not a workspace member`

The dependency-light source-loader harness was run with:

`uv run --no-project --with pytest --with torch --with numpy python - <<'PY' ... PY`

The harness loaded only `nemo_rl/weight_sync/nccl_reshard_utils.py`, avoided the
project package and test conftest dependency graph, and ran the target unit
file with `--confcutdir=tests/unit/weight_sync`.

Result: `63 passed in 0.03s`.

Static checks:

`uv run --no-project --with ruff==0.9.9 ruff check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`

Result: `All checks passed!`

`uv run --no-project --with ruff==0.9.9 ruff format --check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`

Passed. `git diff --check` also passed.

## Concerns

- The full project pytest suite remains unverified locally because the
  `nemo-gym` workspace/dependency graph is unavailable.
- Task 3 must consume nested component placement fields and call
  `restore_refit_info_placements` after collective RPC deserialization.
