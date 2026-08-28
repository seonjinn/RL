# DeepSeek Custom PP NCCL Reshard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let NeMo-RL derive NCCL-reshard PP-stage metadata from a validated, non-interleaved MCore custom pipeline layout.

**Architecture:** Add a pure protocol-based layout mapping helper to the existing reshard utility module. The Megatron policy worker delegates to it only when its runtime model config contains a custom layout; the current standard uneven-PP calculation remains unchanged.

**Tech Stack:** Python 3.13, PyTorch, Megatron-Core runtime layout API, pytest, Ruff, Pyrefly.

**Spec:** `docs/superpowers/specs/2026-08-28-deepseek-custom-pp-nccl-reshard-design.md`

## Global Constraints

- Support only `virtual_pipeline_model_parallel_size == 1`.
- Do not parse MCore layout strings or duplicate its layout grammar.
- Do not change the existing FFN-only NCCL-reshard parameter whitelist.
- Do not enable FlashInfer TRTLLM destination conversion in this change.
- Preserve the standard non-custom PP mapping path.
- New functions require complete type annotations.
- Every commit must use `git commit -s`.

---

### Task 1: Pure Custom-Layout Mapping

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: a runtime object satisfying `PipelineLayerLayout`.
- Produces: `build_layer_to_pp_stage_from_custom_layout(...) -> dict[str, int]`.

- [ ] **Step 1: Write the failing DeepSeek PP8 mapping test**

Add a fake layout whose `get_layer_id_list(vp_stage=0, pp_rank=rank)` returns
`8, 8, 8, 8, 8, 8, 8, 5` decoder IDs across eight stages. Assert the helper
returns all 61 `model.layers.N` keys with stages `0-7` and no duplicate keys.

```python
layout = _FakePipelineLayout([8, 8, 8, 8, 8, 8, 8, 5])
mapping = build_layer_to_pp_stage_from_custom_layout(
    layout,
    pp_size=8,
    layer_prefix="model",
    num_layers=61,
)
assert mapping["model.layers.0"] == 0
assert mapping["model.layers.60"] == 7
assert len(mapping) == 61
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run in the Linux NeMo-RL container:

```bash
uv run --no-sync pytest \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  -k custom_layout -q
```

Expected: import or attribute failure because the helper is not defined.

- [ ] **Step 3: Implement the protocol and mapping helper**

Define a local `Protocol` exposing `pipeline_model_parallel_size`,
`virtual_pipeline_model_parallel_size`, and `get_layer_id_list`. Implement the
keyword-only helper. Validate PP/VPP sizes, integer/range constraints,
duplicates, and exact coverage before constructing exported HF layer names.

- [ ] **Step 4: Add malformed-layout tests**

Parameterize tests for mismatched PP size, VPP2, duplicate IDs, out-of-range
IDs, and missing IDs. Each case must assert the diagnostic substring in the
raised `ValueError`.

- [ ] **Step 5: Run the focused utility suite**

```bash
uv run --no-sync pytest tests/unit/weight_sync/test_nccl_reshard_utils.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the independently tested helper**

```bash
git add nemo_rl/weight_sync/nccl_reshard_utils.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "feat(refit): map custom PP layouts for NCCL reshard"
```

### Task 2: Policy Worker Integration and Config Gate

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: `build_layer_to_pp_stage_from_custom_layout` from Task 1.
- Produces: custom-layout-aware `MegatronPolicyWorkerImpl._build_layer_to_pp_stage` behavior.

- [ ] **Step 1: Write failing worker delegation and support-gate tests**

Create a worker with a fake runtime custom layout and assert
`_build_layer_to_pp_stage(8, "model")` returns the helper's PP8 mapping. Add a
config test proving `check_nccl_reshard_refit_support` accepts a custom layout
with VPP1 and still rejects VPP2.

- [ ] **Step 2: Run the two focused tests and verify failure**

```bash
uv run --no-sync pytest \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  -k "layer_to_pp_stage or custom_pipeline_layout" -q
```

Expected: the existing runtime assertion and config validation reject the
custom layout.

- [ ] **Step 3: Integrate the helper**

Import the helper at module scope. In `_build_layer_to_pp_stage`, delegate when
`config.pipeline_model_parallel_layout` is non-null; otherwise execute the
existing standard path. Remove only the custom-layout assertion and update the
docstring to keep VPP, embedding-split, and loss-split exclusions explicit.

- [ ] **Step 4: Relax only the custom-layout config rejection**

Remove the `pipeline_model_parallel_layout must be unset` violation from
`check_nccl_reshard_refit_support`. Keep the existing VPP and split-accounting
checks. Runtime validation remains authoritative because it sees MCore's parsed
layout object.

- [ ] **Step 5: Run focused and neighboring regression tests**

```bash
uv run --no-sync pytest \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/models/policy/test_megatron_worker.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the integration**

```bash
git add nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "feat(refit): support custom PP ownership metadata"
```

### Task 3: Static and Container Verification

**Files:**
- Verify only; no planned source changes.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a clean, pushed commit suitable for the GPU canary.

- [ ] **Step 1: Run formatting and static checks**

```bash
uv run --no-sync ruff format --check \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/models/policy/test_megatron_worker.py
uv run --no-sync ruff check \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/models/policy/test_megatron_worker.py
```

Expected: zero findings.

- [ ] **Step 2: Run the unit suites in the pinned Linux container**

Use the OCI-HSG nightly container and source mounted from `/home`. Run the two
focused pytest files. If the image lacks a working pytest entrypoint, install
only the test dependency in a node-local `/raid/scratch` venv and rerun there.

- [ ] **Step 3: Verify branch cleanliness and commit provenance**

```bash
git diff --check
git status --short
git log --oneline --show-signature -3
```

Expected: no uncommitted changes and signed commits.

- [ ] **Step 4: Push the exact branch**

```bash
git push -u seonjinn feature/deepseek-custom-pp-nccl-reshard-20260828
```

### Task 4: 48-Node Triton Functional Canary

**Files:**
- Modify after results: investigator experiment `manifest.yaml`
- Modify after results: investigator experiment `results.md`

**Interfaces:**
- Consumes: the pushed source commit from Task 3.
- Produces: functional evidence for custom-PP NCCL reshard independent of FlashInfer layout conversion.

- [ ] **Step 1: Validate scheduling and submit**

Use the existing 48-node DeepSeek non-colocated Triton launcher with these
additional overrides:

```text
policy.generation.refit_transport=nccl_reshard
grpo.max_num_steps=4
checkpointing.enabled=false
```

Run `sbatch --test-only`, commit and push the launcher/report metadata, then
submit the exact commit.

- [ ] **Step 2: Monitor initialization for five minutes**

Check only the exact job ID at intervals of at least 60 seconds. Verify the log
prints the xfer payload split, creates all eight PP-stage communication groups,
and has no missing stage or parameter mapping.

- [ ] **Step 3: Validate completed-run correctness**

Require four completed steps, finite loss/reward/logprob metrics, at least two
successful refits, and no missing or unexpected parameter diagnostics.

- [ ] **Step 4: Record the result**

Add the source commit, job ID, topology, effective config, payload split,
status, and timing to the investigator manifest and report. Do not claim a
FlashInfer performance improvement from this Triton-only transport canary.

