# MXFP8 Linear Backend Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let NeMo-RL retain an explicitly selected refit-safe FlashInfer CUTLASS, CuTeDSL, or TRTLLM MXFP8 linear backend instead of silently replacing non-CUTLASS kernels.

**Architecture:** vLLM remains responsible for mapping `vllm_kwargs.linear_backend` to a concrete kernel and preparing the backend-specific runtime weight and scale layouts. NeMo-RL checks the selected kernel for the `preserves_checkpoint_weight_scale_for_refit` capability and delegates post-load/refit preparation when present; the existing CUTLASS transformation remains as a legacy compatibility path.

**Tech Stack:** Python 3.12, PyTorch, vLLM 0.25.1 integration APIs, pytest, YAML documentation

## Global Constraints

- Support recipe values `flashinfer_cutlass`, `flashinfer_cutedsl`, and `flashinfer_trtllm` through the existing `policy.generation.vllm_kwargs.linear_backend` field.
- Do not introduce a NeMo-RL backend enum that duplicates vLLM's version-dependent backend registry.
- Do not silently substitute CUTLASS when CuTeDSL or TRTLLM was requested.
- Keep the existing legacy FlashInfer CUTLASS refit path unchanged.
- Only retain a non-CUTLASS backend when its vLLM kernel declares `preserves_checkpoint_weight_scale_for_refit = True`.
- Exact TRTLLM tactic misses remain backend-local fallbacks to `tactic=-1`; NeMo-RL does not parse tactic tables.
- All commits must use `seonjinn <sna@nvidia.com>` as the sole author and include a Signed-off-by line.

---

## File Structure

- Modify `nemo_rl/models/generation/vllm/quantization/fp8.py`: implement capability-based MXFP8 linear post-load/refit dispatch and remove the silent CuTeDSL-to-CUTLASS substitution.
- Modify `tests/unit/models/generation/test_vllm_fp8_quantization.py`: pin delegation, legacy fallback, and unsupported-kernel failures.
- Modify `tests/unit/models/generation/test_vllm_fp8_hf_overrides.py`: prove `linear_backend` survives FP8 argument merging for all three backend values.
- Modify `docs/guides/use-custom-vllm.md`: document backend selection, custom-vLLM capability requirements, and fallback semantics.

### Task 1: Refit-Safe Native MXFP8 Kernel Dispatch

**Files:**
- Modify: `nemo_rl/models/generation/vllm/quantization/fp8.py:656-719`
- Test: `tests/unit/models/generation/test_vllm_fp8_quantization.py`

**Interfaces:**
- Consumes: `self.kernel`, `kernel.preserves_checkpoint_weight_scale_for_refit`, `kernel.process_weights_after_loading(layer)`, and the existing CUTLASS compatibility path.
- Produces: `process_weights_after_loading_mxfp8_linear(self, layer) -> None` that delegates to refit-safe native kernels, preserves legacy CUTLASS, and rejects every unsupported non-CUTLASS kernel.

- [ ] **Step 1: Write failing delegation and rejection tests**

Append tests that identify the concrete backend family through representative kernel class names and verify native capability dispatch:

```python
@pytest.mark.parametrize(
    "kernel_name",
    [
        "FlashInferCutedslMxfp8LinearKernel",
        "FlashInferTrtllmMxfp8LinearKernel",
    ],
)
def test_mxfp8_linear_delegates_to_refit_safe_native_kernel(
    fp8_module, kernel_name
):
    calls = []

    def process_weights_after_loading(self, layer):
        calls.append(layer)

    kernel_type = type(
        kernel_name,
        (),
        {
            "preserves_checkpoint_weight_scale_for_refit": True,
            "process_weights_after_loading": process_weights_after_loading,
        },
    )
    layer = types.SimpleNamespace(weight=types.SimpleNamespace(ndim=2))
    method = types.SimpleNamespace(kernel=kernel_type())

    fp8_module.process_weights_after_loading_mxfp8_linear(method, layer)

    assert calls == [layer]
    assert method.kernel.__class__.__name__ == kernel_name
    assert not hasattr(layer, "weight_scale_from_checkpoint")


def test_mxfp8_linear_rejects_refit_unsafe_cutedsl_kernel(fp8_module):
    kernel_type = type("FlashInferCutedslMxfp8LinearKernel", (), {})
    layer = types.SimpleNamespace(weight=types.SimpleNamespace(ndim=2))
    method = types.SimpleNamespace(kernel=kernel_type())

    with pytest.raises(
        RuntimeError,
        match=(
            "FlashInferCutedslMxfp8LinearKernel.*"
            "preserves_checkpoint_weight_scale_for_refit"
        ),
    ):
        fp8_module.process_weights_after_loading_mxfp8_linear(method, layer)

    assert method.kernel.__class__.__name__ == "FlashInferCutedslMxfp8LinearKernel"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --extra vllm pytest \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  -k 'mxfp8_linear' -v
```

Expected: the native capability test fails because current main does not delegate, and the rejection test fails because current main silently replaces CuTeDSL with CUTLASS or raises a different assertion.

- [ ] **Step 3: Implement capability dispatch before legacy imports**

Change the start of `process_weights_after_loading_mxfp8_linear` to:

```python
def process_weights_after_loading_mxfp8_linear(self, layer) -> None:
    if layer.weight.ndim != 2:
        raise ValueError(
            f"MXFP8 linear layer weight must be 2D, but got {layer.weight.ndim}D"
        )

    kernel = getattr(self, "kernel", None)
    if getattr(kernel, "preserves_checkpoint_weight_scale_for_refit", False):
        kernel.process_weights_after_loading(layer)
        return

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    from vllm.model_executor.parameter import ModelWeightParameter
```

Replace the current CuTeDSL-to-CUTLASS mutation with a rejection that names the kernel and required capability:

```python
    else:
        kernel_name = type(kernel).__name__ if kernel is not None else None
        if kernel_name != "FlashInferCutlassMxfp8LinearKernel":
            raise RuntimeError(
                "Unsupported MXFP8 linear kernel for refit: "
                f"{kernel_name}. Non-CUTLASS kernels must declare "
                "preserves_checkpoint_weight_scale_for_refit=True and implement "
                "process_weights_after_loading(layer)."
            )
```

Keep the legacy backend-enum branch and all CUTLASS scale-swizzle code below it unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run --extra vllm pytest \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  -k 'mxfp8_linear or apply_fp8_patches_registers_modelopt' -v
```

Expected: all selected tests pass; the native kernel object remains unchanged.

- [ ] **Step 5: Run formatting checks for changed Python files**

Run:

```bash
uv run ruff check \
  nemo_rl/models/generation/vllm/quantization/fp8.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py
uv run ruff format --check \
  nemo_rl/models/generation/vllm/quantization/fp8.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit the dispatch change**

```bash
git add \
  nemo_rl/models/generation/vllm/quantization/fp8.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py
git commit -s -m "feat(vllm): allow refit-safe MXFP8 linear backends"
```

### Task 2: Preserve Backend Configuration Through FP8 Initialization

**Files:**
- Test: `tests/unit/models/generation/test_vllm_fp8_hf_overrides.py`

**Interfaces:**
- Consumes: `_merge_fp8_kwargs(vllm_kwargs: dict[str, Any], fp8_kwargs: dict[str, Any]) -> None`.
- Produces: regression coverage proving that user-selected `linear_backend` remains unchanged after FP8 initialization.

- [ ] **Step 1: Add a parameterized backend-preservation test**

Append:

```python
import pytest


@pytest.mark.parametrize(
    "linear_backend",
    [
        "flashinfer_cutlass",
        "flashinfer_cutedsl",
        "flashinfer_trtllm",
    ],
)
def test_fp8_merge_preserves_linear_backend(linear_backend):
    vllm_kwargs = {
        "linear_backend": linear_backend,
        "hf_overrides": {"max_position_embeddings": 8192},
    }
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {
            "quantization_config": {"weight_block_size": [32, 16]}
        },
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["linear_backend"] == linear_backend
    assert vllm_kwargs["quantization"] == "fp8"
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
uv run pytest \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py \
  -k linear_backend -v
```

Expected: PASS because `_merge_fp8_kwargs` already preserves unrelated user keys. This test records the supported configuration contract without adding a second config implementation.

- [ ] **Step 3: Run the complete FP8 merge test module**

Run:

```bash
uv run pytest tests/unit/models/generation/test_vllm_fp8_hf_overrides.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit the configuration regression test**

```bash
git add tests/unit/models/generation/test_vllm_fp8_hf_overrides.py
git commit -s -m "test(vllm): preserve MXFP8 linear backend selection"
```

### Task 3: Document Custom vLLM Backend Requirements

**Files:**
- Modify: `docs/guides/use-custom-vllm.md`

**Interfaces:**
- Consumes: the recipe key `policy.generation.vllm_kwargs.linear_backend` and vLLM kernel capability `preserves_checkpoint_weight_scale_for_refit`.
- Produces: user-facing setup instructions for CUTLASS, CuTeDSL, and TRTLLM with explicit refit-safety behavior.

- [ ] **Step 1: Add the MXFP8 backend selection section**

Insert the following section after “Running NeMo RL Apps with Custom vLLM”:

```markdown
## Selecting an MXFP8 Linear Backend

MXFP8 rollout recipes can pass vLLM's linear backend option directly:

```yaml
policy:
  generation:
    vllm_kwargs:
      linear_backend: flashinfer_trtllm
```

The supported FlashInfer values are `flashinfer_cutlass`,
`flashinfer_cutedsl`, and `flashinfer_trtllm`. CUTLASS retains NeMo-RL's
legacy refit path. CuTeDSL and TRTLLM require a custom vLLM kernel that declares
`preserves_checkpoint_weight_scale_for_refit = True` and refreshes its prepared
weight and scale buffers after every refit. A stock kernel without this contract
fails during model preparation; NeMo-RL does not silently replace the requested
backend.

An exact TRTLLM tactic-table miss is not a refit error. The TRTLLM kernel stays
selected and uses its default tactic (`tactic=-1`). Backend or layer-family
fallback remains a vLLM policy decision.
```

- [ ] **Step 2: Validate Markdown and whitespace**

Run:

```bash
git diff --check
rg -n "Selecting an MXFP8 Linear Backend|flashinfer_(cutlass|cutedsl|trtllm)" \
  docs/guides/use-custom-vllm.md
```

Expected: `git diff --check` exits zero and the search shows the new section plus all three values.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/guides/use-custom-vllm.md
git commit -s -m "docs(vllm): explain MXFP8 linear backend selection"
```

### Task 4: Full Verification

**Files:**
- Verify: `nemo_rl/models/generation/vllm/quantization/fp8.py`
- Verify: `tests/unit/models/generation/test_vllm_fp8_quantization.py`
- Verify: `tests/unit/models/generation/test_vllm_fp8_hf_overrides.py`
- Verify: `docs/guides/use-custom-vllm.md`

**Interfaces:**
- Consumes: all deliverables from Tasks 1-3.
- Produces: a clean, reviewable branch whose CPU-level contracts are verified and whose remaining GPU requirement is explicit.

- [ ] **Step 1: Run the complete affected unit-test set**

Run:

```bash
uv run --extra vllm pytest \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 2: Run static and formatting checks**

Run:

```bash
uv run ruff check \
  nemo_rl/models/generation/vllm/quantization/fp8.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py
uv run ruff format --check \
  nemo_rl/models/generation/vllm/quantization/fp8.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/models/generation/test_vllm_fp8_hf_overrides.py
git diff --check upstream/main...HEAD
```

Expected: every command exits zero.

- [ ] **Step 3: Verify authorship and branch scope**

Run:

```bash
git log upstream/main..HEAD --format='%h %an <%ae> %s'
git diff --stat upstream/main...HEAD
git status --short
```

Expected: every commit is authored only by `seonjinn <sna@nvidia.com>`, the diff contains only the design, plan, source, tests, and guide listed above, and the worktree is clean.

- [ ] **Step 4: Record the GPU validation gate**

Do not claim CuTeDSL or TRTLLM production support from CPU unit tests alone. The companion custom-vLLM build must pass, in order:

```text
initial load -> repeated refit -> prepared-buffer pointer stability
-> numerical comparison -> CUDA Graph capture/replay -> rollout smoke
```

For an exact TRTLLM tactic miss, verify from logs that execution stays on TRTLLM with `tactic=-1`. For an unqualified layer family, verify that any CuTeDSL fallback is emitted by vLLM and is not a NeMo-RL substitution.
