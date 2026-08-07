# Native MXFP8 Source Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Megatron-LM to vLLM NCCL Reshard path so native Transformer Engine MXFP8 parameters (`fp8_param=true`, `fp8_recipe=mxfp8`) transfer canonical E4M3 values and E8M0 scales without a BF16 round trip.

**Architecture:** Preserve the current logical HF parameter plan and add ordered component roles beneath each parameter. Existing BF16 and matching blockwise-FP8 transfers continue to use the implicit `weight` component. Native MXFP8 source adapters expose compact `weight` and `weight_scale` tensors; NCCL Reshard transfers each with its own global shape and placement; the vLLM adapter binds them to the existing value and `_scale_from_checkpoint` destinations.

**Tech Stack:** Python 3.13, PyTorch DTensor placements, Transformer Engine MXFP8 metadata, Megatron-Bridge mappings, vLLM ModelOpt MXFP8, pytest.

## Global Constraints

- Base revision is PR 3477 head `6f57c1b79504245fc8211028e504465045315f34`.
- Do not create a GitHub pull request during this implementation cycle.
- Initial support is Megatron-LM policy to vLLM generation, non-colocated NCCL Reshard, FFN and routed-MoE parameters only.
- SGLang, QKVO, native-MXFP8-to-BF16, and NVFP4 are excluded.
- Existing BF16, BF16-to-MXFP8, and matching blockwise-FP8 behavior must remain unchanged.
- Reject GEMM-swizzled TE source scales until a verified inverse conversion is available.
- Use type hints, Google-style docstrings, `uv run`, specific exceptions, and signed commits.

---

### Task 1: Canonical Native MXFP8 Storage Adapter

**Files:**
- Create: `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
- Test: `tests/unit/models/policy/test_mxfp8_refit_source.py`

**Interfaces:**
- Consumes: an object with `shape` and `get_metadata()` returning TE keys `rowwise_data`, `rowwise_scale_inv`, and `with_gemm_swizzled_scales`.
- Produces: `NativeMXFP8Components(weight: torch.Tensor, weight_scale: torch.Tensor)` through `extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components`.

- [x] **Step 1: Write failing extraction tests**

```python
def test_extract_native_mxfp8_components_crops_padding() -> None:
    source = FakeMXFP8Tensor(
        shape=(64, 256),
        rowwise_data=torch.arange(64 * 256, dtype=torch.uint8).reshape(64, 256),
        rowwise_scale_inv=torch.arange(128 * 8, dtype=torch.uint8).reshape(128, 8),
        with_gemm_swizzled_scales=False,
    )

    components = extract_native_mxfp8_components(source)

    assert components.weight.shape == (64, 256)
    assert components.weight.dtype == torch.float8_e4m3fn
    assert components.weight_scale.shape == (64, 8)
    assert components.weight_scale.dtype == torch.uint8
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py`

Expected: collection fails because `mxfp8_refit_source` does not exist.

- [x] **Step 3: Implement compact extraction and validation**

```python
@dataclass(frozen=True)
class NativeMXFP8Components:
    weight: torch.Tensor
    weight_scale: torch.Tensor


def extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components:
    metadata = tensor.get_metadata()
    if metadata.get("with_gemm_swizzled_scales"):
        raise ValueError("Native MXFP8 refit requires compact rowwise scales")
    shape = tuple(int(size) for size in tensor.shape)
    rows = math.prod(shape[:-1])
    if not shape or shape[-1] % 32:
        raise ValueError(f"Native MXFP8 refit requires K divisible by 32; got {shape}")
    data = metadata["rowwise_data"]
    scale = metadata["rowwise_scale_inv"]
    weight = data.reshape(shape).view(torch.float8_e4m3fn)
    compact_scale = scale[:rows, : shape[-1] // 32].reshape(*shape[:-1], shape[-1] // 32)
    return NativeMXFP8Components(weight=weight, weight_scale=compact_scale)
```

- [x] **Step 4: Add missing-data, dtype, K-alignment, and swizzled-layout tests**

- [x] **Step 5: Run focused tests and commit**

```bash
uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py
git add nemo_rl/models/policy/workers/mxfp8_refit_source.py tests/unit/models/policy/test_mxfp8_refit_source.py
git commit -s -m "feat(refit): extract native MXFP8 components"
```

### Task 2: Component-Aware NCCL Refit Metadata

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: optional serialized `components` on each logical parameter metadata entry.
- Produces: every `param_info` contains ordered components with `role`, `global_shape`, `dtype`, `src_placements`, and `dst_placements`; legacy entries produce one implicit `weight` component.

- [x] **Step 1: Write failing component metadata tests**

```python
def test_build_refit_info_describes_native_mxfp8_components() -> None:
    metadata = {
        "model.layers.0.mlp.down_proj.weight": {
            "shape": [64, 256],
            "dtype": "torch.bfloat16",
            "components": [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"},
            ],
        }
    }
    info = build_nccl_reshard_refit_info(metadata, TRAIN, GEN, 4, 4)
    components = info["per_layer_params"]["model.layers.0"][0]["components"]
    assert [component["role"] for component in components] == ["weight", "weight_scale"]
    assert tuple(components[1]["global_shape"]) == (64, 8)
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py -k native_mxfp8_components`

Expected: component metadata is absent.

- [x] **Step 3: Build ordered component metadata with placement validation**

Legacy parameters use `[{"role": "weight", ...}]`. Native scale placement is derived from the parent FFN name, so column-parallel gate/up scales shard dimension 0 and row-parallel down scales shard their compressed final dimension.

- [x] **Step 4: Preserve component shapes while grouping routed experts**

When individual expert metadata is grouped along expert dimension, prepend global expert count to every component shape as well as the logical shape.

- [x] **Step 5: Add duplicate-role, empty-component, dtype, and scale-shape rejection tests**

- [x] **Step 6: Run focused tests and commit**

```bash
uv run pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py
git add nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "refactor(refit): describe ordered weight components"
```

### Task 3: Megatron Native MXFP8 Source Mapping

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Test: `tests/unit/models/megatron/test_group_experts.py`

**Interfaces:**
- Consumes: Task 1 extraction and Task 2 component metadata.
- Produces: `HFToLocalParamMap.get(hf_name, role="weight") -> LocalParamSpec | None`; native source maps contain both `weight` and `weight_scale` specs.

- [x] **Step 1: Write failing map lookup and fused gate/up tests**

```python
assert source_map.get(gate_name, role="weight").base.shape == (intermediate, hidden)
assert source_map.get(gate_name, role="weight_scale").base.shape == (
    intermediate,
    hidden // 32,
)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -q tests/unit/models/policy/test_megatron_worker.py tests/unit/models/megatron/test_group_experts.py -k native_mxfp8`

Expected: `HFToLocalParamMap.get` does not accept `role` and no native source components are exported.

- [x] **Step 3: Extend `HFToLocalParamMap` with backward-compatible roles**

```python
def get(
    self,
    hf_name: str,
    default: LocalParamSpec | None = None,
    *,
    role: str = "weight",
) -> LocalParamSpec | None:
    return self.specs.get((hf_name, role), default)
```

Legacy constructors normalize string keys to `(name, "weight")` in `__post_init__`.

- [x] **Step 4: Detect native MXFP8 training storage**

The native path is enabled only for `fp8_param=true`, `fp8_recipe=mxfp8`, generation `precision=fp8`, and `is_mx=true`. Existing `_is_fp8_export()` remains blockwise-only.

- [x] **Step 5: Split fused Megatron value and scale tensors consistently**

For `GatedMLPMapping` and `FusedGatedExpertMapping`, split value and compact scale on the same output dimension. For down projections, keep both tensors direct. Group routed-expert values and scales independently but in identical expert-index order.

For `moe_single_grouped_weight=True`, the live parameter is a Transformer
Engine `GroupedTensor`, not an `MXFP8Tensor`. Detect it through Megatron
Core, obtain the already-cached per-expert members with
`get_grouped_quantized_members(..., create_if_missing=False)`, extract each
member independently, and stack in numeric local-expert order. Never pass the
`GroupedTensor` container itself to the Task 1 extractor or split its padded
raw scale storage.

- [x] **Step 6: Advertise native component metadata during prepare**

Use the existing logical HF metadata for global shapes, replace only validated FFN/MoE entries with native value/scale component descriptors, and leave misc parameters on their existing path.

- [x] **Step 7: Transfer every ordered component**

Update train-side `nccl_reshard_refit` to fetch `source_map.get(name, role=component["role"])` and pass the component global shape to `DTensorRef`.

- [x] **Step 8: Run focused tests and commit**

```bash
uv run pytest -q tests/unit/models/policy/test_megatron_worker.py tests/unit/models/megatron/test_group_experts.py tests/unit/weight_sync/test_nccl_reshard_utils.py
git add nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/megatron/test_group_experts.py
git commit -s -m "feat(refit): export native MXFP8 source components"
```

### Task 4: vLLM Native MXFP8 Destination Binding

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`

**Interfaces:**
- Consumes: ordered component metadata and role-aware `HFToLocalParamMap`.
- Produces: destination specs for canonical E4M3 values and E8M0 scales without invoking `quantize_mxfp8_weight` on the native path.

- [ ] **Step 1: Write a failing native destination test**

```python
mapping = backend.build_hf_to_local_param_map(native_refit_info)
assert mapping.get(hf_name, role="weight").base.data_ptr() == value.data_ptr()
assert mapping.get(hf_name, role="weight_scale").base.data_ptr() == scale.data_ptr()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest -q tests/unit/models/generation/test_nccl_reshard_backend.py -k native_mxfp8`

Expected: no `weight_scale` destination spec exists.

- [ ] **Step 3: Bind value and scale destinations**

Resolve each logical HF name once, derive `<vllm_name>_scale_from_checkpoint`, apply the same fused gate/up or w13 region to both tensors, and validate exact component shape and dtype.

- [ ] **Step 4: Receive every ordered component**

Update generation-side `nccl_reshard_refit` to use each component's role and global shape. Keep BF16 receiver quantization in the existing `weight` post-hook.

- [ ] **Step 5: Add incomplete-pair and wrong-scale-layout tests**

- [ ] **Step 6: Run focused tests and commit**

```bash
uv run pytest -q tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/models/generation/test_vllm_fp8_quantization.py
git add nemo_rl/models/generation/vllm/vllm_backend.py tests/unit/models/generation/test_nccl_reshard_backend.py
git commit -s -m "feat(refit): load native MXFP8 reshard components"
```

### Task 5: Validation and Draft Report Update

**Files:**
- Modify: `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- Modify: `/Users/sna/MXFP8_generation/deliverables/pr_native_mxfp8_source_draft.html`

**Interfaces:**
- Consumes: completed implementation and test outputs.
- Produces: validated branch and an evidence-backed draft page; no GitHub PR.

- [ ] **Step 1: Update validator tests**

Accept native MXFP8 only for the exact source/target pairing and reject native-MXFP8-to-BF16, blockwise-FP8-to-MXFP8, SGLang, QKVO, and swizzled source scale routes.

- [ ] **Step 2: Run focused local or GCP-NRT container tests**

```bash
uv run pytest -q \
  tests/unit/models/policy/test_mxfp8_refit_source.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
uv run pre-commit run --all-files
```

- [ ] **Step 3: Run a two-step GCP-NRT native MXFP8 smoke**

Use the existing Qwen3-30B-A3B performance recipe with `fp8_param=true`, `fp8_recipe=mxfp8`, MXFP8 rollout, non-colocated generation, and `refit_transport=nccl_reshard`.

- [ ] **Step 4: Run a matched 20-step comparison after smoke success**

Record refit, generation, logprob, policy training, E2E step time, tokens/s/GPU, reward, loss, and W&B URLs for the native path and the current BF16-round-trip path.

- [ ] **Step 5: Update and validate the draft HTML**

```bash
python3 /Users/sna/.codex/skills/explain-diff-html/scripts/validate_explainer.py \
  /Users/sna/MXFP8_generation/deliverables/pr_native_mxfp8_source_draft.html
```

- [ ] **Step 6: Commit the implementation state without opening a PR**

```bash
git add tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "test(refit): validate native MXFP8 reshard"
```
