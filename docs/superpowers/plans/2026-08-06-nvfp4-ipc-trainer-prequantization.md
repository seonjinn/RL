# NVFP4 Colocated IPC Trainer-Side Prequantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing trainer-side refit transform handshake so BF16 Megatron training can send prepacked W4A16 or W4A4 tensors directly to colocated vLLM generation through the CUDA-IPC path, while preserving MXFP8 and receiver-side NVFP4 behavior.

**Architecture:** The metadata handshake explicitly records whether a conversion executes at the source or destination, which keeps the existing NCCL-Reshard BF16 wire contract separate from the new IPC prequantized wire contract. A lazy trainer iterator converts complete logical HF groups with the canonical ModelOpt serializer, and the existing IPC streamer remains format-agnostic. After the second handshake, vLLM classifies the manifest as prepacked ModelOpt and uses its current direct-load path without receiver-side BF16 quantization.

**Tech Stack:** Python 3.13.13+, PyTorch, ModelOpt, Megatron-Bridge, vLLM 0.25.1, Ray, CUDA IPC over ZMQ, safetensors, YAML recipes, pytest, pyrefly, Ruff, SLURM on GCP-NRT B200.

## Global Constraints

- Implement from branch `sna/nvfp4-ipc-trainer-prequant-design`, design commit `cb9e09a44`, based on runtime commit `8524a0a0cc6f7a6e67fc35cc4afc3c936ccdb9bb`.
- Create a fresh isolated implementation worktree; do not modify the runtime baseline worktree in place.
- Production scope is Megatron policy plus vLLM real-quant generation with `colocated.enabled=true` and `refit_transport=null`.
- Preserve the receiver-side BF16-to-NVFP4 path when `refit_prequantize=false`.
- Preserve MXFP8 trainer-side prequantization and its `_scale_from_checkpoint` names.
- Preserve the NCCL-Reshard BF16 wire contract; source-side NVFP4 conversion must not activate for destination-stage requests.
- W4A16 and W4A4 use `serialize_bf16_nvfp4_group()`; do not duplicate FP4 packing or scale math.
- W13 gate/up projections share one group weight amax. W2 and eligible dense projections remain independent.
- W4A4 reuses the same frozen calibration artifact as the receiver baseline. Online recalibration, dummy scales, and W4A16 fallback are forbidden.
- Metadata negotiation uses descriptors only; it must not execute quantization kernels or allocate packed model tensors.
- The IPC streamer continues to consume only `(name, tensor)` records and must not branch on quantization format.
- New functions require complete type hints. Keep ModelOpt and vLLM imports lazy on plain Megatron workers.
- Follow red-green-refactor and use `git commit -s` for every commit.
- GPU validation uses committed code, an immutable container, `batch`, four-hour limits, dedicated experiment directories, and five minutes of early monitoring.

## File Map

- `nemo_rl/weight_sync/refit_transforms.py`: source-versus-destination ownership and deterministic wire names.
- `nemo_rl/modelopt/models/generation/nvfp4_refit.py`: lazy group-aware trainer iterator.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`: request validation, calibration load, descriptor metadata, and export transforms.
- `nemo_rl/models/generation/vllm/config.py`: supported colocated IPC configuration gate.
- `nemo_rl/models/generation/vllm/vllm_worker.py`: sync worker negotiation.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`: async worker negotiation.
- `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`: source-stage request selection and prepacked direct load.
- `tests/unit/weight_sync/test_refit_transforms.py`: protocol and metadata tests.
- `tests/unit/models/generation/test_nvfp4_refit.py`: iterator and serializer parity tests.
- `tests/unit/models/policy/test_megatron_worker.py`: source worker integration tests.
- `tests/unit/models/generation/test_vllm_config.py`: configuration matrix.
- `tests/unit/models/generation/test_vllm_backend.py`: outer worker forwarding.
- `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`: receiver negotiation and direct-load tests.
- `tests/unit/models/generation/test_vllm_quant_backend.py`: existing quantized policy-to-vLLM integration regression.
- `tests/test_nvfp4_rollout_recipes.py`: composed recipe tests.
- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant.yaml`: W4A16 opt-in.
- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout-prequant.yaml`: W4A4 opt-in.
- `experiments/nvfp4_ipc_prequant/`: reproducible GCP-NRT launcher and report.

---

### Task 1: Make Transform Ownership Explicit

**Files:**
- Modify: `nemo_rl/weight_sync/refit_transforms.py`
- Modify: `tests/unit/weight_sync/test_refit_transforms.py`
- Modify: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Modify: `tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py`
- Modify: `tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py`

**Interfaces:**
- Produces: `RefitTransformLocation = Literal["source", "destination"]`.
- Produces: `RefitTransformRequest.transform_location: RefitTransformLocation`, defaulting to `"source"` for legacy MXFP8.
- Produces: `transform_component_name(parameter_name: str, target_format: str, role: str) -> str`.
- Produces: `describe_refit_wire_metadata(source_info, requests) -> dict[str, tuple[tuple[int, ...], torch.dtype]]`.
- Preserves: destination-stage `bf16 -> nvfp4_*` carries BF16 for NCCL-Reshard.

- [ ] **Step 1: Write failing ownership tests**

Add source-versus-destination assertions:

```python
@pytest.mark.parametrize("target_format", ["nvfp4_w4a16", "nvfp4_w4a4"])
def test_nvfp4_wire_contract_depends_on_transform_location(
    target_format: str,
) -> None:
    codec = resolve_transform("bf16", target_format)

    destination_wire = codec.describe_outputs(
        (64, 128), "torch.bfloat16", transform_location="destination"
    )
    source_wire = codec.describe_outputs(
        (64, 128), "torch.bfloat16", transform_location="source"
    )

    assert destination_wire == (
        TransformComponentSpec("weight", (64, 128), "torch.bfloat16"),
    )
    assert source_wire[:3] == (
        TransformComponentSpec("weight", (64, 64), "torch.uint8"),
        TransformComponentSpec(
            "weight_scale", (64, 8), "torch.float8_e4m3fn"
        ),
        TransformComponentSpec("weight_scale_2", (), "torch.float32"),
    )
    expected_roles = ["weight", "weight_scale", "weight_scale_2"]
    if target_format == "nvfp4_w4a4":
        expected_roles.append("input_scale")
    assert [component.role for component in source_wire] == expected_roles
```

Add exact naming tests:

```python
assert transform_component_name(name, "mxfp8_e4m3_e8m0", "weight") == name
assert (
    transform_component_name(name, "mxfp8_e4m3_e8m0", "weight_scale")
    == name + "_scale_from_checkpoint"
)
assert (
    transform_component_name(name, "nvfp4_w4a16", "weight_scale")
    == base + ".weight_scale"
)
assert (
    transform_component_name(name, "nvfp4_w4a4", "input_scale")
    == base + ".input_scale"
)
```

Patch runtime quantizers to raise and verify `describe_refit_wire_metadata()` still produces W4A4 shapes and dtypes. This proves setup is descriptor-only.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest \
  tests/unit/weight_sync/test_refit_transforms.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py -q
```

Expected: requests have no transform location and NVFP4 has only the destination-stage BF16 wire description.

- [ ] **Step 3: Add location-aware request and codec contracts**

Extend the request:

```python
RefitTransformLocation = Literal["source", "destination"]

@dataclass(frozen=True)
class RefitTransformRequest:
    parameter_names: tuple[str, ...]
    source_format: str
    target_format: str
    transform_location: RefitTransformLocation = "source"
```

Make `merge_refit_transform_requests()` group by `(source_format, target_format, transform_location)` and reject any parameter whose tuple conflicts. Legacy string responses remain source-side BF16-to-MXFP8.

Add this keyword-only codec argument:

```python
def describe_outputs(
    self,
    global_shape: tuple[int, ...],
    input_dtype_name: str,
    *,
    transform_location: RefitTransformLocation = "destination",
) -> tuple[TransformComponentSpec, ...]:
    raise NotImplementedError
```

The default is deliberately `"destination"`: existing NCCL metadata builders call the codec without a location and must continue to see the current BF16 wire contract. MXFP8 validates that only `"source"` is accepted. For NVFP4, return BF16 for `destination`; for `source`, return packed weight, block scale, second-level scale, and W4A4 input scale. `describe_destination()` always describes the packed family, but marks W4A4 `input_scale` as `"codec"` for a source transform and `"calibration"` for a destination transform.

- [ ] **Step 4: Add deterministic metadata construction**

Implement `transform_component_name()` and `describe_refit_wire_metadata()`. The helper must validate names once, preserve unrequested and destination-stage entries, expand source-stage entries in codec order, reject duplicate output names, and map only these dtype names:

```python
_TORCH_DTYPES_BY_NAME = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float32": torch.float32,
    "torch.uint8": torch.uint8,
}
```

- [ ] **Step 5: Preserve NCCL destination-stage behavior**

Set existing BF16-to-NVFP4 NCCL requests to:

```python
transform_location="destination"
```

Assert their serialized plans still advertise one BF16 wire component and packed destination components. Existing MXFP8 requests retain the source default. Also retain a direct codec regression test without a location argument; it must return BF16 for NVFP4 so `nccl_reshard_utils.py` requires no behavioral change.

- [ ] **Step 6: Verify GREEN and commit**

```bash
uv run --frozen pytest \
  tests/unit/weight_sync/test_refit_transforms.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py -q
uv run --frozen pyrefly check nemo_rl/weight_sync/refit_transforms.py
uv run --frozen ruff check nemo_rl/weight_sync/refit_transforms.py tests/unit/weight_sync
git diff --check
git add nemo_rl/weight_sync/refit_transforms.py tests/unit/weight_sync
git commit -s -m "refactor(refit): record transform execution location"
```

Expected: all pass.

### Task 2: Add a Lazy Canonical NVFP4 Source Iterator

**Files:**
- Modify: `nemo_rl/modelopt/models/generation/nvfp4_refit.py`
- Modify: `tests/unit/models/generation/test_nvfp4_refit.py`

**Interfaces:**
- Consumes: `serialize_bf16_nvfp4_group()` and `nvfp4_refit_group()`.
- Produces: `iter_bf16_nvfp4_refit_weights(weights: Iterable[tuple[str, torch.Tensor]], *, selected_names: Collection[str], mode: NVFP4RefitMode, calibration: NVFP4Calibration | None) -> Iterator[tuple[str, torch.Tensor]]`.
- Guarantees: passthrough tensors preserve identity; selected groups emit canonical ModelOpt families; completed groups are released immediately.

- [ ] **Step 1: Write failing iterator tests**

Use the existing fake exporter:

```python
def test_iter_nvfp4_refit_weights_emits_passthrough_and_complete_w13_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = list(
        iter_bf16_nvfp4_refit_weights(
            iter([(norm, norm_tensor), (gate, gate_tensor), (up, up_tensor)]),
            selected_names={gate, up},
            mode="w4a16",
            calibration=None,
        )
    )
    assert output[0] == (norm, norm_tensor)
    assert [name for name, _ in output[1:]] == [
        gate,
        gate_base + ".weight_scale",
        gate_base + ".weight_scale_2",
        up,
        up_base + ".weight_scale",
        up_base + ".weight_scale_2",
    ]
```

Also test bitwise equality with direct serialization, W4A4 frozen input scales, duplicate selected names, a requested name missing from export, incomplete gate/up groups, non-BF16 selected tensors, and no serializer call when the selected set is empty.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen pytest tests/unit/models/generation/test_nvfp4_refit.py -q
```

Expected: import failure for the new iterator.

- [ ] **Step 3: Implement the single-pass iterator**

Use `pending_by_group`, `seen_selected`, and `remaining_selected`. Do not clone selected tensors. Emit a group through `serialize_bf16_nvfp4_group()` as soon as every expected member arrives, then delete the group.

Finish with explicit checks:

```python
missing = sorted(remaining_selected)
if missing:
    raise ValueError(
        "NVFP4 refit source iterator did not export requested weights: "
        f"{missing}"
    )
if pending_by_group:
    details = _format_incomplete_groups(pending_by_group)
    raise ValueError(
        "NVFP4 refit source iterator ended with incomplete groups: "
        f"{details}"
    )
```

This reuses the receiver baseline serializer, preserving shared W13 amax and exact component order.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run --frozen pytest tests/unit/models/generation/test_nvfp4_refit.py -q
uv run --frozen pyrefly check nemo_rl/modelopt/models/generation/nvfp4_refit.py
uv run --frozen ruff check \
  nemo_rl/modelopt/models/generation/nvfp4_refit.py \
  tests/unit/models/generation/test_nvfp4_refit.py
git diff --check
git add \
  nemo_rl/modelopt/models/generation/nvfp4_refit.py \
  tests/unit/models/generation/test_nvfp4_refit.py
git commit -s -m "feat(modelopt): stream canonical NVFP4 refit groups"
```

Expected: all pass.

### Task 3: Integrate Source Transforms into Megatron Export

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`
- Modify: `tests/unit/modelopt/test_calibration_artifact.py`

**Interfaces:**
- Consumes: Task 1 metadata helper and Task 2 NVFP4 iterator.
- Produces: `_iter_refit_transformed_params(weights) -> Iterator[tuple[str, torch.Tensor]]`.
- Produces: `_load_nvfp4_refit_calibration(mode, selected_names) -> NVFP4Calibration | None`.
- Preserves: `_iter_params_with_optional_kv_scales()` remains the common source for IPC, packed collective, and checkpoint-engine callers.

- [ ] **Step 1: Write failing metadata-only and parity tests**

Preserve the existing MXFP8 checkpoint-engine test. Add W4A16 and W4A4 tests:

```python
request = RefitTransformRequest(
    parameter_names=(gate, up),
    source_format="bf16",
    target_format="nvfp4_w4a16",
    transform_location="source",
)
updated = worker.enable_refit_transforms([request])

assert updated[gate] == ((32, 8), torch.uint8)
assert updated[gate_base + ".weight_scale"] == (
    (32, 1),
    torch.float8_e4m3fn,
)
assert updated[gate_base + ".weight_scale_2"] == ((), torch.float32)
assert list(worker._iter_params_with_optional_kv_scales()) == expected
```

Patch MXFP8 and NVFP4 quantizers to raise during `enable_refit_transforms()` and assert setup still succeeds.

For W4A4, create a temporary artifact with `save_nvfp4_calibration()`, configure matching model id, explicit revision, quant config, and path, then compare every emitted component bit-for-bit with direct artifact load plus serializer output.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen pytest \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/modelopt/test_calibration_artifact.py -q
```

Expected: NVFP4 metadata remains BF16 and the Bridge iterator does not apply installed transforms.

- [ ] **Step 3: Store validated request state**

Initialize:

```python
self._refit_transform_requests_by_name: dict[str, RefitTransformRequest] = {}
self._nvfp4_refit_calibration: NVFP4Calibration | None = None
```

Validate the complete request set before mutating state. Replace metadata enumeration through the Bridge with:

```python
return describe_refit_wire_metadata(source_info, requests)
```

Retain `_refit_prequant_names` only if a remaining caller requires it; otherwise remove it after all references migrate.

- [ ] **Step 4: Load W4A4 calibration once**

Read:

```python
generation_cfg = cast(VllmConfig, self.cfg["generation"])
artifact_path = generation_cfg.get("real_quant_calibration_path")
quant_cfg = generation_cfg.get("quant_cfg")
model_id = self.cfg["model_name"]
revision = generation_cfg.get("vllm_kwargs", {}).get("revision")
```

Require all values for W4A4, then call:

```python
load_nvfp4_calibration(
    artifact_path,
    model_id=model_id,
    model_revision=revision,
    quant_cfg=quant_cfg,
    expected_projection_names=selected_names,
)
```

The receiver independently validates the same artifact. Any provenance or projection mismatch fails before transfer.

- [ ] **Step 5: Apply the common export transform**

Wrap Bridge output:

```python
base_iter = self.megatron_bridge.export_hf_weights(
    [self.model],
    show_progress=False,
    conversion_tasks=conversion_tasks,
)
yield from self._iter_refit_transformed_params(base_iter)
```

Behavior:

- no source-stage request: yield unchanged;
- MXFP8: call the existing quantizer and preserve names;
- W4A16/W4A4: call Task 2's group iterator;
- destination-stage request: yield BF16 unchanged;
- multiple source target formats in one run: fail before transfer with all formats listed.

Draft weights and KV/Q scale records remain unselected and unchanged.

- [ ] **Step 6: Verify GREEN and commit**

```bash
uv run --frozen pytest \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/modelopt/test_calibration_artifact.py \
  tests/unit/models/generation/test_nvfp4_refit.py -q
uv run --frozen pyrefly check \
  nemo_rl/models/policy/workers/megatron_policy_worker.py
uv run --frozen ruff check \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_worker.py
git diff --check
git add \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/modelopt/test_calibration_artifact.py
git commit -s -m "feat(refit): prequantize NVFP4 on Megatron workers"
```

Expected: all pass and setup invokes no quantization kernel.

### Task 4: Negotiate Prepacked NVFP4 in vLLM

**Files:**
- Modify: `nemo_rl/models/generation/vllm/config.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `tests/unit/models/generation/test_vllm_config.py`
- Modify: `tests/unit/models/generation/test_vllm_backend.py`
- Modify: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

**Interfaces:**
- Consumes: location-aware requests from Task 1.
- Produces: `VllmQuantInternalWorkerExtension.prepare_refit_info(state_dict_info: dict[str, Any], serialized_fp8_config: dict[str, Any] | None = None, refit_prequantize: bool = False) -> RefitTransformResponse`.
- Guarantees: source-stage request only for supported opt-in IPC; second handshake classifies `modelopt` and returns no request.

- [ ] **Step 1: Write failing config matrix tests**

Define an accepted NVFP4 config:

```python
def _nvfp4_config(**overrides: object) -> VllmConfig:
    return cast(VllmConfig, {
        "vllm_cfg": {
            "precision": "bfloat16",
            "refit_prequantize": True,
        },
        "quant_cfg": (
            "examples/modelopt/quant_configs/"
            "nvfp4_experts_weightonly.yaml"
        ),
        "real_quant": True,
        "refit_transport": None,
        "colocated": {"enabled": True, "resources": {}},
        **overrides,
    })
```

Reject `real_quant=false`, missing `quant_cfg`, non-colocated generation, `refit_transport="nccl_reshard"`, W4A4 without calibration, and non-NVFP4 quant config. Keep all existing MXFP8 tests.

- [ ] **Step 2: Write failing negotiation tests**

For BF16 W4A16:

```python
assert extension.prepare_refit_info(
    bf16_info,
    serialized_fp8_config=None,
    refit_prequantize=True,
) == [
    RefitTransformRequest(
        parameter_names=expected_names,
        source_format="bf16",
        target_format="nvfp4_w4a16",
        transform_location="source",
    )
]
```

With the flag false, assert `transform_location="destination"`. Feed a complete packed manifest into the second call and assert source `modelopt`, no request, and no `_load_bf16_weights()` call.

Add sync and async forwarding assertions, including the explicit boolean RPC argument.

- [ ] **Step 3: Verify RED**

```bash
uv run --frozen pytest \
  tests/unit/models/generation/test_vllm_config.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  -k 'prepare_refit or prepacked or ipc_payload' -q
```

Expected: validation permits only MXFP8 and the quant backend always requests receiver-side NVFP4.

- [ ] **Step 4: Generalize config validation**

Allow `refit_prequantize=true` for exactly one of:

1. MXFP8: `precision == "fp8"` and `is_mx is True`.
2. NVFP4: `real_quant is True`, quant config resolves to W4A16/W4A4, generation is colocated, and `refit_transport is None`.

Require a non-empty calibration path for W4A4. Errors name the invalid setting and supported topology.

- [ ] **Step 5: Pass the flag through sync and async workers**

Both outer workers compute:

```python
refit_prequantize = bool(
    self.cfg["vllm_cfg"].get("refit_prequantize", False)
)
```

Pass it as the third internal RPC argument. The internal quant extension selects `source` only when true; otherwise it keeps `destination`.

On the second handshake the packed manifest sets `_nrl_real_quant_source = "modelopt"`. Reuse the existing direct path for filtering ignored scales, batching fused MoE components, loader route reuse, and IPC reference detachment.

- [ ] **Step 6: Verify GREEN and commit**

```bash
uv run --frozen pytest \
  tests/unit/models/generation/test_vllm_config.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_quant_backend.py -q
uv run --frozen pyrefly check \
  nemo_rl/models/generation/vllm/config.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py
uv run --frozen ruff check \
  nemo_rl/models/generation/vllm \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_config.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
git diff --check
git add \
  nemo_rl/models/generation/vllm/config.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_config.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
git commit -s -m "feat(vllm): negotiate prepacked NVFP4 IPC refits"
```

Expected: all pass.

### Task 5: Add Opt-In Recipes and Complete CPU Validation

**Files:**
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout-prequant.yaml`
- Modify: `tests/test_nvfp4_rollout_recipes.py`

**Interfaces:**
- Consumes: existing W4A16 and W4A4 rollout recipes.
- Produces: opt-in variants changing only refit optimizations and output names.

- [ ] **Step 1: Write failing recipe tests**

For both composed configs assert:

```python
assert config["policy"]["quant_cfg"] is None
assert config["policy"]["generation"]["real_quant"] is True
assert config["policy"]["generation"]["refit_transport"] is None
assert config["policy"]["generation"]["colocated"]["enabled"] is True
assert (
    config["policy"]["generation"]["vllm_cfg"]["refit_prequantize"]
    is True
)
assert config["policy"]["refit_persistent_ipc_buffers"] is True
assert (
    config["policy"]["megatron_cfg"]["refit_slim_offload_after"]
    is True
)
```

For W4A4, assert the checked-in calibration path is null and validation fails until launch supplies the immutable path.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen pytest tests/test_nvfp4_rollout_recipes.py -q
```

Expected: optimized recipe paths are missing.

- [ ] **Step 3: Add minimal derived recipes**

W4A16:

```yaml
defaults: ./grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.yaml
checkpointing:
  checkpoint_dir: results/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant
policy:
  refit_persistent_ipc_buffers: true
  megatron_cfg:
    refit_slim_offload_after: true
  generation:
    refit_transport: null
    vllm_cfg:
      refit_prequantize: true
logger:
  log_dir: logs/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant
  wandb:
    project: nemo-rl
    name: grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant
```

W4A4 extends its existing recipe with the same fields and retains:

```yaml
policy:
  generation:
    real_quant_calibration_path: null
```

Do not enable `refit_batched_moe_shuffle`; PR 3478 remains an independent optimization.

- [ ] **Step 4: Run focused and broad CPU validation**

```bash
uv run --frozen pytest \
  tests/unit/weight_sync/test_refit_transforms.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py \
  tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/generation/test_nvfp4_refit.py \
  tests/unit/models/generation/test_vllm_config.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/test_nvfp4_rollout_recipes.py -q
uv run --frozen pytest \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  -k 'prepare_refit or prepacked or ipc_payload' -q
uv run --frozen pytest tests/unit/models/generation/test_vllm_quant_backend.py -q
uv run --frozen pytest tests/unit/weight_sync tests/unit/modelopt -q
uv run --frozen pyrefly check nemo_rl
uv run --frozen ruff check nemo_rl tests
uv run --frozen ruff format --check nemo_rl tests
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 5: Commit**

```bash
git add \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout-prequant.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout-prequant.yaml \
  tests/test_nvfp4_rollout_recipes.py
git commit -s -m "test(perf): add NVFP4 IPC prequant recipes"
```

### Task 6: Validate Correctness and Performance on GCP-NRT

**Files:**
- Create: `experiments/nvfp4_ipc_prequant/README.md`
- Create: `experiments/nvfp4_ipc_prequant/submit_gcp_nrt.sh`
- Create after runs: `experiments/nvfp4_ipc_prequant/results.md`

**Interfaces:**
- Consumes: committed implementation and immutable GCP-NRT artifacts.
- Produces: matched W4A16/W4A4 OFF/ON W&B runs, logs, source metadata, and steps 3-20 comparisons.

- [ ] **Step 1: Write the experiment contract**

Record and enforce:

```text
Cluster: GCP-NRT B200, 8 GPUs/node
Source: exact git commit, clean worktree, pushed branch
Container: absolute immutable .sqsh path and SHA256
Model: exact Qwen3-30B-A3B path/revision
Dataset: exact path and SHA256
Calibration: exact W4A4 safetensors path and SHA256
Topology: identical node/GPU/TP/EP/DP settings within each OFF/ON pair
Comparison: only refit_prequantize changes
Window: W&B steps 3-20 for 20-step runs
```

Launcher inputs:

```bash
MODE=w4a16|w4a4
PREQUANT=0|1
MAX_STEPS=5|20|100
RUN_SUFFIX=20260806-a
```

The script rejects a dirty tree, verifies `SOURCE_COMMIT`, runs scheduling preflight, submits through `ray.sub`, and writes `metadata.env`, resolved config, job id, and log paths under one run directory.

- [ ] **Step 2: Validate and commit the launcher**

```bash
bash -n experiments/nvfp4_ipc_prequant/submit_gcp_nrt.sh
shellcheck experiments/nvfp4_ipc_prequant/submit_gcp_nrt.sh
git diff --check
git add experiments/nvfp4_ipc_prequant
git commit -s -m "bench(refit): add NVFP4 IPC prequant harness"
git push -u origin HEAD
```

Expected: shell checks pass, the tree is clean, and the exact commit is remote.

- [ ] **Step 3: Submit four five-step smoke runs**

Submit W4A16 OFF/ON and W4A4 OFF/ON under W&B project `sna-nvfp4-ipc-prequant`. Monitor each running job for five minutes and stop on the first deterministic failure.

Every ON run must show a complete packed manifest, receiver source `modelopt`, no receiver `_load_bf16_weights()` call, five finite steps, stable manifest shape/dtype, and no monotonic GPU memory growth.

- [ ] **Step 4: Submit matched 20-step OFF/ON runs**

Use the same seed, data order, model revision, topology, quantization scope, container, calibration, and importance-sampling settings within each pair.

Use the `nemo-rl-wandb-reporting` skill for steps 3-20 and report:

```text
transfer/update seconds
total refit seconds
generation seconds and tokens/s/GPU
logprob seconds and tokens/s/GPU
policy training seconds and tokens/s/GPU
E2E step seconds and tokens/s/GPU
peak allocated/reserved GPU memory
mean rollout reward
policy loss, KL, entropy, and token_mult_prob_error when logged
```

The primary gate is lower transfer/update and total refit time. Finite metrics and no suspicious short-window reward/loss divergence are required, but do not establish training-quality equivalence.

- [ ] **Step 5: Record results and decide the 100-step gate**

Write raw W&B links, job ids, commit, sample count, mean, standard deviation, and deltas into `results.md`. If either ON mode is performance-positive and intended for upstreaming, submit a matched 100-step correctness pair before opening a production PR.

Do not start NCCL-Reshard implementation in this worktree. Create a branch from the verified IPC commit and a separate implementation plan. Its acceptance threshold is at least 5 percent lower total refit time or at least 1 percent lower E2E step time, with serializer parity and no topology-changing memory regression.

- [ ] **Step 6: Commit the measured report**

```bash
git add \
  experiments/nvfp4_ipc_prequant/results.md \
  experiments/nvfp4_ipc_prequant/README.md
git commit -s -m "docs(refit): report NVFP4 IPC prequant results"
git push
```

Expected: every aggregate links to raw logs or W&B and the report contains no unverified claims.
