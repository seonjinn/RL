# Task 7 Report: NVFP4 NCCL Receiver and Validation

## Result

Implemented BF16-to-NVFP4 NCCL receiver conversion for real ModelOpt W4A16 and
W4A4 rollout workers. The trainer remains a plain BF16 source. NCCL transfers
one BF16 wire tensor into destination-owned scratch, and the receiver expands
grouped experts and invokes the canonical `serialize_bf16_nvfp4_group()` path
before loading checkpoint-layout components through vLLM.

The refit handshake is now target-aware end to end. Real-quant workers request
`nvfp4_w4a16` or `nvfp4_w4a4`; legacy string responses remain mapped to
`mxfp8_e4m3_e8m0`. Both synchronous and asynchronous outer workers forward and
return the inner collective RPC results instead of discarding them.

## RED Evidence

Focused tests were added before implementation and produced these failures:

- W4A16 handshake: `prepare_refit_info()` returned `None`, causing
  `TypeError: object of type 'NoneType' has no len()`.
- W4A4 handshake: the same `None` response prevented an `nvfp4_w4a4` request.
- Grouped receiver: BF16 wire metadata followed the identity path and failed
  because the resolved runtime parameter was `torch.uint8`, not BF16.
- Collective lifecycle: NCCL refit bypassed `_weight_update_lifecycle()` and
  reached direct vLLM post-processing instead of the test lifecycle.
- Static validation: unsupported real-NVFP4 source configurations were accepted.

During review, tests were corrected to use the real schema hierarchy. The
original implementation incorrectly read real-quant fields from `vllm_cfg`,
which made the new gates no-ops. The fixed tests failed against that version and
now drive the generation-level lookups.

## Implementation

### Handshake

- Added `RefitTransformResponse` and deterministic request merging. Legacy
  `list[str]` results retain their MXFP8 meaning; typed requests preserve their
  explicit source and target formats and reject format conflicts.
- Real-quant internal workers return a typed BF16-to-NVFP4 request after source
  classification and return no request for existing prepacked ModelOpt/QARL
  manifests.
- `VllmGenerationWorkerImpl` and `VllmAsyncGenerationWorkerImpl` now return
  merged collective RPC results. `VllmGeneration` merges those typed actor
  responses without hardcoding MXFP8.
- Forwarded the small FP8 configuration needed for the existing MXFP8
  `refit_prequantize` request path. The legacy one-argument internal call still
  records metadata and returns `None`, avoiding new optional imports for
  prepacked QARL tests.

### Receiver and lifecycle

- Added a backend receiver-transform hook to the existing HF-to-local mapping.
  Identity and MXFP8 mappings continue through their prior paths.
- NVFP4 metadata must describe exactly one BF16 weight wire component, the
  exact W4A16/W4A4 destination component family and shapes, model-scoped
  finalization, and a destination-local K divisible by 16.
- Each transformed parameter receives into an owned BF16 scratch tensor on the
  destination device. Collective bytes never target packed Marlin or fused-MoE
  runtime parameters.
- Grouped `[E, M, K]` gate/up/down tensors are expanded into per-expert 2-D HF
  names. Gate and up complete together per expert and therefore share the
  serializer's weight `amax`; W2 is serialized separately.
- Completed groups use the existing `_load_weights()` BF16 path, which calls
  `serialize_bf16_nvfp4_group()`. W4A4 `input_scale` remains calibration-owned,
  is never a wire component, and reuses the first cached value on later refits.
- Bulk receive, its synchronization, misc loading, finalization, and the final
  completion fence now run inside exactly one
  `_weight_update_lifecycle("collective")`. Finalization occurs once after all
  components and groups are complete.

### Static validation

For real NVFP4 NCCL refit, validation now requires:

- `policy.quant_cfg: null` and policy precision `bf16`/`bfloat16`;
- Megatron enabled with plain BF16 storage and DTensor disabled;
- vLLM generation, non-colocated execution, EP=1, and PP=1;
- an effective W4A16 or W4A4 NVFP4 quantization mode;
- a non-empty W4A4 calibration artifact path.

Schema ownership is explicit: `real_quant`, `quant_cfg`, and
`real_quant_calibration_path` are read from `policy.generation`; vLLM
TP/EP/PP and precision remain under `policy.generation.vllm_cfg`. K and source
tensor checks remain in transform negotiation/receiver metadata validation.

## Verification

All commands used the reusable rollout worktree environment because this
worktree's local environment lacks required runtime packages.

- `test_refit_transforms.py`, `test_nccl_reshard_utils.py`, and
  `test_nccl_reshard_weight_synchronizer.py`: **99 passed** after formatting.
- Independent focused weight-sync verification: **93 passed**.
- `test_vllm_modelopt_real_quant_config.py`: **114 passed, 1 skipped**. The skip
  is the expected missing optional
  `modelopt.torch.quantization.calib` dependency.
- `test_nvfp4_refit.py`: **24 passed**.
- `test_calibration_artifact.py`: **29 passed**.
- Focused static validation matrix: **14 passed**.
- Ruff format: passed; 7 files reformatted and the final check reported no
  remaining format changes.
- Ruff check: passed for every changed Python file.
- `py_compile`: passed for every changed Python file.
- `git diff --check`: passed.

## Local limitations

- `tests/unit/models/generation/test_nccl_reshard_backend.py`: **1 skipped at
  module collection**, exact reason: `could not import 'vllm': No module named
  'vllm'`.
- `tests/unit/models/generation/test_vllm_backend.py -k prepare_refit_info`:
  blocked at import because vLLM is absent. These tests contain the sync/async
  MXFP8 and W4A16/W4A4 forwarding coverage for the Linux container run.
- `tests/unit/models/generation/test_vllm_generation.py`: collection is blocked
  in the local environment because `torchdata` is absent.
- A run without the narrow `--confcutdir` reached the repository Ray fixture and
  failed because the minimal Ray installation lacks `aiohttp_cors`; the focused
  reruns used the documented confcut boundary.
- No CUDA/NCCL or real vLLM/ModelOpt kernel execution is possible on this macOS
  host. Linux container validation remains required.

The checked-in NVFP4 recipes inherit colocated generation. An NCCL smoke run
must explicitly set `policy.generation.colocated.enabled=false` (or use a
dedicated NCCL recipe); the validator now rejects the inherited colocated value.

## Changed files

- `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- `nemo_rl/models/generation/vllm/quantization/fp8.py`
- `nemo_rl/models/generation/vllm/vllm_backend.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py`
- `nemo_rl/models/generation/vllm/vllm_worker.py`
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- `nemo_rl/weight_sync/nccl_reshard_utils.py`
- `nemo_rl/weight_sync/refit_transforms.py`
- `tests/unit/models/generation/test_nccl_reshard_backend.py`
- `tests/unit/models/generation/test_vllm_backend.py`
- `tests/unit/models/generation/test_vllm_generation.py`
- `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
- `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- `.superpowers/sdd/2026-08-04-bf16-nvfp4-rollout/task-7-report.md`
