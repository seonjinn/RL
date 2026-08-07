# Task 4 Report: vLLM Native MXFP8 Destination Binding

## Status

Implemented native MXFP8 value/scale destination binding and component-aware
generation receipt.

Modified files:

- `nemo_rl/models/generation/vllm/vllm_backend.py`
- `tests/unit/models/generation/test_nccl_reshard_backend.py`
- This report

The plan documents, unrelated untracked `docs/superpowers/`, and all other files
were left unchanged.

## Implementation

- Native `weight` components bind to live E4M3 value parameters or their dense
  gate/up and grouped-MoE W13 regions.
- Native `weight_scale` components bind to the matching canonical
  `*_scale_from_checkpoint` parameter or the identical fused region.
- Map construction validates the ordered value/scale pair, canonical dtypes,
  global value/scale relationship, destination dtypes, and exact local target
  shapes derived from each component's destination placements.
- Missing checkpoint-scale parameters fail with a message requiring the first
  MXFP8 post-load processing before NCCL reshard preparation.
- Native specs use direct or merged copy receipt and never call
  `quantize_mxfp8_weight`.
- The legacy BF16 receiver quantization hook and matching blockwise-FP8 direct
  path remain unchanged.
- The generation loop receives every ordered role with its own global shape and
  source/destination placements. The existing single
  `process_weights_after_loading` call remains after all component receives.

## TDD Evidence

- The first native destination test failed because no `weight_scale` role spec
  existed.
- The receive-loop test then failed because only one parent-level transfer ran
  with parent placements.
- After the implementation, native coverage passes for dense gate/up/down,
  grouped W13/W2, nested `routed_experts` names, TP=2 slices, incomplete pairs,
  wrong value/scale dtype and shape, missing initialized checkpoint scales,
  absence of native quantization, per-component transfer metadata, and one
  finalizer call.

## Verification

- Dependency-light AST receiver suite:
  `18 passed`.
- Ruff check: passed.
- Ruff format check: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Pyrefly found no new Task 4 diagnostics after fixes. Its remaining output is
  two pre-existing mapper inference errors plus unavailable optional `zmq` and
  `safetensors` imports in the minimal environment.

The normal project `uv run pytest` command remains blocked before collection by
the existing `nemo-gym` workspace-source configuration. The requested combined
dependency-light command reached `18 passed, 5 skipped, 1 failed`; the sole
failure is an untouched quantization test that directly imports unavailable
vLLM. No dependency synchronization was performed.

## Concerns

- Real vLLM ModelOpt post-load execution and CUDA/NCCL transfer remain
  unverified on this macOS worktree. The focused harness executes the production
  receiver method bodies and validates transfer/finalizer ordering without those
  dependencies.
