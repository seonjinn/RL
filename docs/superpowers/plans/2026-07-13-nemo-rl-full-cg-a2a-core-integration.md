# NeMo-RL Full-CG and A2A Core Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the reviewed fixed-shape PolicyTraining CUDA Graph implementation into the current CuTeDSL/A2A branch, preserve combined-1F1B A2A schedule construction, and add trustworthy replay and storage-stability evidence.

**Architecture:** Semantically port the implementation from commits `5ee358abb` and `690fc74da` in dependency order. Self-contained adapter and loss code is reused; overlapping train, setup, and worker code is reconciled against the current branch rather than mechanically selected from either history. The graph captures only synchronous PolicyTraining forward/backward. Optimizer, scheduler, Logprob, generation, refit, and evaluation remain eager or fail closed.

**Tech Stack:** Python 3.13, PyTorch 2.11, Megatron Core `002255075`, Megatron-Bridge `3e3cdf11`, Transformer Engine `42b8400516`, pytest, Pyrefly, Ruff.

## Global Constraints

- Preserve current CuTeDSL settings, A2A `return_schedule_plan` support, lifecycle telemetry, and failure diagnostics.
- Support only optimizer-backed synchronous PolicyTraining with dynamic batching disabled, sequence packing disabled, and context parallel size one.
- Accept only `ClippedPGLossFn` and `NLLLossFn`; reject other losses before graph warm-up or capture.
- Keep forward-only Logprob, reference Logprob, evaluation, top-k/QKV calibration, split/async PolicyTraining, colocated refit/offload, router replay, and draft/hidden capture fail closed.
- Keep optimizer and scheduler steps outside the graph and describe that boundary in every report.
- Port `690fc74da` after `5ee358abb`; the second donor is required for graph-stable MoE/MTP auxiliary-loss scaling.
- Add `Based-on-commit:` trailers when a donor implementation is semantically rewritten. Do not claim an exact cherry-pick for rewritten hunks.
- Do not modify or advance `sna/nemo-2606-cutedsl-a2a-factorial-20260712`, the immutable source branch for jobs `2373273` through `2373278`.
- Do not claim replay, parity, temporal overlap, or speedup from configuration flags or mocked tests.
- Core PyTorch/MCore tests cannot run in the macOS harness-only environment. Run RED and GREEN commands in the pinned Linux environment. The Mac may run only source/config harness tests with `uv run --no-sync`.

---

### Task 1: Port the self-contained full-CG adapter

**Files:**
- Create: `nemo_rl/models/megatron/full_cuda_graph.py`
- Create: `tests/unit/models/megatron/test_full_cuda_graph.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Produces: `TensorSignature`, `StaticMicrobatchSignature`, `FullCudaGraphCallSignature`, `ProcessedMicrobatchStaticBufferLoader`, `NemoRLFullCudaGraphWrapper`, `validate_full_cuda_graph_policy_config`, `require_supported_full_cuda_graph_operation`, `full_cuda_graph_loss_signature`, `attach_full_cuda_graph_normalizers`, `materialize_full_cuda_graph_metrics`, and `build_full_cuda_graph_schedule`.

- [ ] **Step 1: Port donor tests first**

Use commit `5ee358abb` as the exact source for `tests/unit/models/megatron/test_full_cuda_graph.py`, excluding the four loss-metric tests assigned to Task 2. Do not include `FullCudaGraphAuxLossScaleBuffer` tests until Task 6.

- [ ] **Step 2: Verify RED in Linux**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_full_cuda_graph.py
```

Expected: collection fails because `nemo_rl.models.megatron.full_cuda_graph` is absent.

- [ ] **Step 3: Port the adapter implementation**

Use `5ee358abb:nemo_rl/models/megatron/full_cuda_graph.py` as the donor. Retain the exact supported-operation guards and static-buffer rules. Exclude `FullCudaGraphAuxLossScaleBuffer`, which belongs to Task 6. Add the donor module to `pyrefly.toml` using the same module override as the donor commit.

- [ ] **Step 4: Verify GREEN**

```bash
uv run --group test pytest -q tests/unit/models/megatron/test_full_cuda_graph.py
```

Expected: adapter tests pass without CUDA hardware.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/full_cuda_graph.py \
  tests/unit/models/megatron/test_full_cuda_graph.py pyrefly.toml
git commit -s -m "feat: port fixed-shape full CUDA graph adapter" \
  -m "Based-on-commit: 5ee358abb06c8ecc57469812fb9f7a56793c46eb"
```

---

### Task 2: Port graph-safe loss metrics

**Files:**
- Modify: `nemo_rl/algorithms/loss/interfaces.py`
- Modify: `nemo_rl/algorithms/loss/loss_functions.py`
- Modify: `tests/unit/models/megatron/test_full_cuda_graph.py`

**Interfaces:**
- Produces: `full_cuda_graph_metrics()` and graph-aware scalar metric conversion for `ClippedPGLossFn` and `NLLLossFn`.

- [ ] **Step 1: Add the four donor loss tests**

Port these nodes from `5ee358abb`:

- `test_full_cuda_graph_metric_context_keeps_scalar_tensors_on_device`;
- `test_clipped_pg_loss_emits_tensor_metrics_in_full_cuda_graph_context`;
- `test_nll_loss_emits_tensor_metrics_in_full_cuda_graph_context`; and
- `test_materialize_full_cuda_graph_metrics_restores_python_scalars`.

- [ ] **Step 2: Verify RED**

```bash
uv run --group test pytest -q tests/unit/models/megatron/test_full_cuda_graph.py \
  -k 'metric_context or emits_tensor_metrics or materialize_full_cuda_graph_metrics'
```

Expected: tests fail because the graph metric context is absent or eager `.item()` conversion occurs.

- [ ] **Step 3: Port the minimal loss changes**

Port only the loss-interface and `ClippedPGLossFn`/`NLLLossFn` changes from `5ee358abb`. Preserve eager Python metric types when the graph context is inactive. Avoid data-dependent Python branches on captured tensors when the context is active.

- [ ] **Step 4: Verify graph and eager behavior**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_full_cuda_graph.py \
  tests/unit/algorithms/test_loss_functions.py
```

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/algorithms/loss/interfaces.py \
  nemo_rl/algorithms/loss/loss_functions.py \
  tests/unit/models/megatron/test_full_cuda_graph.py
git commit -s -m "feat: preserve graph-safe loss metrics" \
  -m "Based-on-commit: 5ee358abb06c8ecc57469812fb9f7a56793c46eb"
```

---

### Task 3: Add typed full-CG and paged-stash configuration

**Files:**
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`

**Interfaces:**
- Adds: `cuda_graph_warmup_steps`, `cuda_graph_use_single_mempool`, `moe_expert_rank_capacity_factor`, `moe_paged_stash`, `moe_paged_stash_page_size`, `moe_paged_stash_buffer_size_factor_cuda`, and `moe_paged_stash_buffer_size_factor_cpu`.

- [ ] **Step 1: Add donor setup tests**

Port `TestApplyMoeConfig::test_full_cuda_graph_paged_stash_fields_are_applied` and `TestApplyPerformanceConfig::test_full_iteration_cuda_graph_fields_are_applied` from `5ee358abb`.

- [ ] **Step 2: Verify RED in the Linux MCore environment**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_full_cuda_graph_paged_stash_fields_are_applied \
  tests/unit/models/megatron/test_megatron_setup.py::TestApplyPerformanceConfig::test_full_iteration_cuda_graph_fields_are_applied
```

- [ ] **Step 3: Add typed fields and propagation**

Port the donor fields additively. Preserve the current CuTeDSL validation, all A2A fields, upstream-default behavior when fields are absent, and MTP-zero-to-None normalization.

- [ ] **Step 4: Verify GREEN plus retained features**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_megatron_setup.py \
  -k 'full_cuda_graph or full_iteration or a2a or cutedsl or delay_wgrad'
```

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/policy/__init__.py nemo_rl/models/megatron/setup.py \
  tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "feat: expose full CUDA graph policy settings" \
  -m "Based-on-commit: 5ee358abb06c8ecc57469812fb9f7a56793c46eb"
```

---

### Task 4: Compose the graph runner with the current A2A train adapter

**Files:**
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `tests/unit/models/megatron/test_train.py`
- Create: `tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py`

**Interfaces:**
- Produces: `forward_with_post_processing_fn` with keyword-only `return_schedule_plan` and `full_cuda_graph`; `megatron_forward_backward` with optional injected `forward_backward_func`; graph-aware `LossPostProcessor`.

- [ ] **Step 1: Add donor graph tests and one integration test**

Port the two donor `TestMegatronForwardBackward` full-CG tests. Add `test_full_graph_forward_step_preserves_a2a_schedule_plan` to the integration file. Its fake injected raw schedule must invoke the provided `forward_step_func` with `return_schedule_plan=True`. Assert that:

```python
assert output is schedule_plan
model.build_schedule_plan.assert_called_once_with(
    input_ids=static_microbatch.input_ids_cp_sharded,
    position_ids=static_microbatch.position_ids,
    attention_mask=static_microbatch.attention_mask,
)
assert observed_valid_seqs is static_microbatch.data_dict[
    FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS
]
assert observed_valid_toks is static_microbatch.data_dict[
    FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS
]
```

- [ ] **Step 2: Verify RED**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py::test_full_graph_forward_step_preserves_a2a_schedule_plan \
  tests/unit/models/megatron/test_train.py::TestMegatronForwardBackward::test_full_cuda_graph_schedule_receives_static_signature_and_normalizers \
  tests/unit/models/megatron/test_train.py::TestMegatronForwardBackward::test_full_cuda_graph_schedule_rejects_forward_only
```

- [ ] **Step 3: Semantically merge the train path**

Use the donor implementation but preserve the current `_build_post_processing_fn` and complete `if return_schedule_plan:` block. Load `FULL_CUDA_GRAPH_GLOBAL_VALID_SEQS` and `FULL_CUDA_GRAPH_GLOBAL_VALID_TOKS` immediately after extracting the processed microbatch and before schedule-plan construction. The forward-step partial must receive both `full_cuda_graph` and the MCore-supplied `return_schedule_plan`. The injected graph schedule is `forward_backward_func`; eager mode continues to call `get_forward_backward_func()`.

- [ ] **Step 4: Verify GREEN and A2A regressions**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py \
  tests/unit/models/megatron/test_train.py \
  -k 'full_cuda_graph or schedule_plan or a2a'
```

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/train.py \
  tests/unit/models/megatron/test_train.py \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py
git commit -s -m "feat: compose full CUDA graph with A2A schedules" \
  -m "Based-on-commit: 5ee358abb06c8ecc57469812fb9f7a56793c46eb"
```

---

### Task 5: Integrate synchronous worker ownership and fail-closed lifecycle

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`

**Interfaces:**
- Worker owns `_full_cuda_graph_enabled`, `_full_cuda_graph_schedule`, and `_full_cuda_graph_wrapper`.

- [ ] **Step 1: Port donor worker tests first**

Port the runtime-operation rejection, eval rejection, and graph-backed train tests from `5ee358abb`. Keep current lifecycle and host-memory tests unchanged.

- [ ] **Step 2: Verify RED**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/policy/test_megatron_worker.py \
  -k 'full_cuda_graph and not aux_loss'
```

- [ ] **Step 3: Integrate without replacing current lifecycle code**

Validate configuration before CUDA setup. Build the graph wrapper after Megatron setup. In synchronous `train`, inject the graph schedule and materialize graph-safe metrics afterward. Add fail-closed guards to Logprob, reference Logprob, eval/top-k, split/async begin, QKV calibration, inference preparation/finish, and refit offload entry points. Graph-disabled behavior and all current telemetry must remain byte-for-byte equivalent outside the new guards.

- [ ] **Step 4: Verify GREEN and lifecycle regressions**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/policy/test_megatron_worker.py \
  -k 'full_cuda_graph or prepare_for_training or offload or refit'
```

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_worker.py
git commit -s -m "feat: own full CUDA graph policy lifecycle" \
  -m "Based-on-commit: 5ee358abb06c8ecc57469812fb9f7a56793c46eb"
```

---

### Task 6: Stabilize MoE and MTP auxiliary-loss scale storage

**Files:**
- Modify: `nemo_rl/models/megatron/full_cuda_graph.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_full_cuda_graph.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`

- [ ] **Step 1: Port the three donor tests from `690fc74da`**

Add the scalar buffer stability, nonscalar rejection, and worker repeated-update tensor-identity tests.

- [ ] **Step 2: Verify RED**

```bash
uv run --group test pytest -q tests/unit/models/megatron/test_full_cuda_graph.py \
  -k 'aux_loss_scale_buffer'
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/policy/test_megatron_worker.py \
  -k 'full_cuda_graph_aux_loss_scale'
```

- [ ] **Step 3: Port `FullCudaGraphAuxLossScaleBuffer` and worker integration**

Reuse one scalar tensor allocation and update it in place. Both MoE and MTP scale callbacks must return the same graph-visible tensor in graph mode. Preserve current eager normalization and zero-token clamping outside graph mode.

- [ ] **Step 4: Verify GREEN**

Run the RED commands plus current `compute_moe_grad_scale` tests.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/full_cuda_graph.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_full_cuda_graph.py \
  tests/unit/models/policy/test_megatron_worker.py
git commit -s -m "fix: stabilize full CUDA graph auxiliary scales" \
  -m "Based-on-commit: 690fc74daf705b47fcd4151fe82c1727b3afa0cc"
```

---

### Task 7: Add replay counters and storage-pointer safety

**Files:**
- Modify: `nemo_rl/models/megatron/full_cuda_graph.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py`

**Interfaces:**
- Adds: `FullCudaGraphExecutionStats`, `NemoRLFullCudaGraphWrapper.execution_stats()`, `FullCudaGraphStorageSignature.capture(model, optimizer)`, `require_match(model, optimizer)`, and `digest()`.
- Worker emits flat metrics `full_cuda_graph_warmup_calls`, `full_cuda_graph_capture_calls`, `full_cuda_graph_replay_calls`, `full_cuda_graph_reset_calls`, and `full_cuda_graph_storage_signature_sha256`.

- [ ] **Step 1: Add integration-only failing tests**

Add nodes proving warmup/capture/replay/reset counts, rejection after parameter pointer change, rejection after gradient pointer change, rejection after optimizer-state pointer change, and SHA-256 digest change after pointer replacement.

- [ ] **Step 2: Verify RED**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py \
  -k 'stats or storage_guard or storage_signature_digest'
```

- [ ] **Step 3: Implement evidence and validation**

Counters are instance-local. With one warm-up call and three total calls, expected counts are one warm-up, one capture, and two replays. The storage signature deterministically includes model parameters, allocated `grad` or `main_grad` tensors, and recursively discovered optimizer tensors. Each entry records name, shape, dtype, device, and `untyped_storage().data_ptr()`. Deduplicate identical tensor objects. `digest()` hashes the canonical entry serialization and publishes only the digest. Capture after optimizer state is first initialized; validate immediately before later graph-backed training calls.

- [ ] **Step 4: Verify GREEN**

Run the RED command and the complete full-CG adapter and worker test files.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/full_cuda_graph.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py
git commit -s -m "feat: expose full CUDA graph replay evidence"
```

---

### Task 8: Add defaults and a non-colocated recipe contract

**Files:**
- Modify: `examples/configs/grpo_math_1B.yaml`
- Modify: `examples/configs/grpo_math_1B_megatron.yaml`
- Modify: `tests/unit/reference_configs/grpo_math_1B.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-full-cg-noncolocated.yaml`
- Create: `tests/test_full_cuda_graph_policy_recipe.py`

- [ ] **Step 1: Add failing resolved-config tests**

Require CP1, dynamic batching and packing disabled, vLLM generation, colocated generation disabled with explicit generation resources, force-on-policy ratio enabled, reference Logprob skipped, validation disabled, `cuda_graph_impl=full_iteration`, warm-up three, single mempool enabled, and CuTeDSL enabled.

- [ ] **Step 2: Verify RED on macOS**

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --group test pytest \
  -p no:cacheprovider -q tests/test_full_cuda_graph_policy_recipe.py
```

- [ ] **Step 3: Add defaults and recipe**

Port donor defaults additively beside current A2A keys. The recipe is a configuration contract only; it is not considered runnable until the separate policy-only/non-colocated GPU harness plan creates and validates its allocation and launch path. Keep the existing colocated factorial submitter fail-closed.

- [ ] **Step 4: Verify GREEN and fail-closed harness behavior**

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --group test pytest \
  -p no:cacheprovider -q \
  tests/test_full_cuda_graph_policy_recipe.py \
  tests/test_nemo2606_multinode_factorial_harness.py::test_official_performance_recipe_accepts_full_iteration_overrides \
  tests/test_nemo2606_multinode_factorial_harness.py::test_matrix_payload_fails_closed_on_missing_feature_implementations
```

- [ ] **Step 5: Commit**

```bash
git add examples/configs/grpo_math_1B.yaml \
  examples/configs/grpo_math_1B_megatron.yaml \
  tests/unit/reference_configs/grpo_math_1B.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-full-cg-noncolocated.yaml \
  tests/test_full_cuda_graph_policy_recipe.py
git commit -s -m "feat: define non-colocated full CUDA graph recipe"
```

---

### Task 9: Verify and review the integrated core

- [ ] **Step 1: Run Linux non-MCore tests**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_full_cuda_graph.py \
  tests/unit/algorithms/test_loss_functions.py \
  tests/test_full_cuda_graph_policy_recipe.py \
  tests/test_nemo2606_multinode_factorial_harness.py
```

- [ ] **Step 2: Run Linux MCore tests**

```bash
uv run --extra mcore --group test pytest --mcore-only -q \
  tests/unit/models/megatron/test_full_cuda_graph_a2a_integration.py \
  tests/unit/models/megatron/test_train.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py
```

- [ ] **Step 3: Run locked static checks in Linux**

```bash
uv run --group dev ruff check \
  nemo_rl/models/megatron/full_cuda_graph.py \
  nemo_rl/models/megatron/train.py \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/algorithms/loss/interfaces.py \
  nemo_rl/algorithms/loss/loss_functions.py
uv run --group dev pyrefly check \
  nemo_rl/models/megatron/full_cuda_graph.py \
  nemo_rl/models/megatron/train.py \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/algorithms/loss/interfaces.py \
  nemo_rl/algorithms/loss/loss_functions.py
git diff --check
```

- [ ] **Step 4: Obtain independent task and whole-branch reviews**

Generate review packages from each task base and from merge-base to HEAD. Resolve every Critical and Important finding with focused tests and re-review.

- [ ] **Step 5: Record durable progress**

Append task commit ranges and review verdicts to `.superpowers/sdd/progress.md`, commit with sign-off, and push only `sna/nemo-2606-full-cg-a2a-integration-20260713`.

## Deferred GPU Harness Plan

After this plan passes review, create a separate plan for:

- `run_full_cg_policy_training.py`;
- `run_full_cg_policy_training.sbatch`;
- `submit_full_cg_policy_training.sh`; and
- `tests/test_full_cuda_graph_policy_harness.py`.

That subsystem must validate eager/replay loss and update parity, one capture plus at least two replays, the storage digest, Nsight correlation, CuTeDSL kernels in replay, and A2A temporal overlap before running performance replicas. The core recipe added here is not evidence that such a driver exists or works.
