# Task 3 Fix 1 Report: Native MXFP8 Source Review

## Status

Implemented all mandatory round-1 review fixes in the Task 3 worktree.

Modified files:

- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `tests/unit/models/megatron/test_group_experts.py`
- This report

The plan documents, unrelated untracked `docs/superpowers/`, and all other files
were left unchanged.

## Fixes

- Singular TE grouped sources no longer materialize or retain value/scale stacks
  while the source map is built. Each role-aware `LocalParamSpec` has
  `base=None`; its `pre` hook re-fetches cached members with
  `get_grouped_quantized_members(..., create_if_missing=False)`, extracts current
  compact storage, selects the requested projection and role, and stacks only
  that component for the current refit.
- Native extraction eligibility is determined from emitted HF names before
  extraction. Dict-valued gate/up mappings are filtered per output, and shared
  experts remain entirely on the misc Bridge path.
- Native bulk metadata is built from real conversion-task mapping names, local
  logical parameter shapes, model TP/EP topology, and
  `model.config.num_moe_experts`. PP placeholders use shape-dictionary broadcast;
  no native bulk task enters normal Bridge export or MXFP8 dequantization.
- Dense projection dimensions expand by model TP. Expert projection dimensions
  do not: the current train expert mesh carries EP only and validation requires
  ETP=1. Expert metadata uses the global expert count and validates singular
  grouped local count against EP.
- Tests now use real `WeightConversionTask` instances, Parameter-wrapped grouped
  fakes with the pinned cache-only API signature, actual synthetic task-builder
  logic, PP placeholders, TP>1/EP>1 shapes, and mutation between two refits.

## TDD Evidence

- The new stale-source test failed against the original code because grouped
  specs had retained `base` stacks and fetched members during map construction.
- The shared-expert regression would call native extraction before filtering in
  the original compound-mapping branch.
- The shape-only test initially failed because the helper did not exist.
- The TP=2/EP=4 expert regression then failed with `[8, 8, 64]`; after applying
  the ETP=1 source-mesh rule it passes with `[8, 4, 64]` while the dense gate
  shape remains TP-expanded at `[8, 64]`.

## Verification

- Dynamic grouped mutation and shared-expert source-loader harness: `2 passed`
- Shape-only dense TP / expert ETP-EP source-loader harness: `1 passed`
- Native metadata/extraction AST contract harness: `5 passed`
- Dependency-light `test_nccl_reshard_utils.py` source-loader run: `65 passed`
- Ruff check: passed
- Ruff format check: four files already formatted
- `py_compile`: passed
- `git diff --check`: passed

The normal focused project pytest command remains blocked before collection by
the existing workspace configuration error:

`nemo-gym references a workspace in tool.uv.sources, but is not a workspace member`

No dependency synchronization was performed.

## Concerns

- Real Transformer Engine `GroupedTensor` construction and distributed PP/EP
  execution remain unverified on this CPU-only worktree. The faithful harness
  exercises the pinned task fields and grouped API signatures, but a CUDA
  integration run is still required for end-to-end confidence.
- GEMM-swizzled MXFP8 scales remain intentionally rejected by the Task 1
  extractor.
