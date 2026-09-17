# SpecDec-Aware Refit Memory Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Qwen3-235B SpecDec host-memory refit stalls while preserving the exact default behavior and accuracy path of no-SpecDec baselines.

**Architecture:** Add an opt-in, typed deep-refit lifecycle to the existing vLLM refit configuration. Legacy runs keep level-1 sleep; supported SpecDec runs snapshot only the small static drafter, discard stale target weights with level-2 sleep, stream updated target weights, restore the drafter, and wake the KV cache. Split the aggregate refit timer into child phases and fail before generation if draft restoration is incomplete.

**Tech Stack:** Python 3.13, PyTorch, Ray, vLLM 0.25.1, Pydantic v2, pytest, OmegaConf, SLURM, W&B.

**Spec:** `docs/superpowers/specs/2026-09-17-specdec-refit-memory-lifecycle-design.md`

## Global Constraints

- The no-SpecDec baseline remains on the existing `legacy_level1` lifecycle and makes no drafter RPC.
- The new lifecycle is opt-in and must fail loudly for unsupported speculative backends.
- The target model, drafter, and KV cache must be restored before generation resumes.
- `RAY_memory_usage_threshold=0.98` is not an accepted fix.
- Accuracy checks include reward, generated length, approximate entropy, policy KL, generation KL, and output integrity.
- Performance comparisons use time-weighted Steps 3-20 totals from matched runs.
- Production changes use tests-first development and signed, narrowly scoped commits.

---

### Task 1: Typed lifecycle configuration and legacy default

**Files:**
- Modify: `nemo_rl/models/generation/vllm/config.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`
- Modify: `examples/configs/grpo_math_1B.yaml`
- Modify: `tests/unit/reference_configs/grpo_math_1B.yaml`

**Interfaces:**
- Produces: `VllmRefitMemoryLifecycleConfig(mode: Literal["legacy_level1", "specdec_deep_refit"])`
- Produces: `VllmRefitConfig.memory_lifecycle`
- Produces: `resolve_vllm_refit_memory_lifecycle(config: VllmConfig) -> VllmRefitMemoryLifecycleConfig`

- [ ] **Step 1: Write failing configuration tests**

Add tests that assert the default mode is `legacy_level1`, an explicit
`specdec_deep_refit` value parses, and an unknown value raises a Pydantic
validation error.

```python
def test_refit_memory_lifecycle_defaults_to_legacy_level1():
    config = VllmRefitConfig()
    assert config.memory_lifecycle.mode == "legacy_level1"


def test_refit_memory_lifecycle_accepts_specdec_deep_refit():
    config = VllmRefitConfig.model_validate(
        {"memory_lifecycle": {"mode": "specdec_deep_refit"}}
    )
    assert config.memory_lifecycle.mode == "specdec_deep_refit"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/unit/models/generation/test_vllm_generation.py -k refit_memory_lifecycle
```

Expected: failure because `memory_lifecycle` is not defined.

- [ ] **Step 3: Implement the typed configuration**

Add a Pydantic model with `extra="forbid"`, add it to `VllmRefitConfig` with a
`default_factory`, and expose a resolver that validates `config.get("refit_cfg")
or {}` exactly once.

```python
class VllmRefitMemoryLifecycleConfig(BaseModel, extra="forbid"):
    mode: Literal["legacy_level1", "specdec_deep_refit"] = "legacy_level1"


class VllmRefitConfig(BaseModel, extra="allow"):
    sparse: VllmSparseRefitConfig = Field(default_factory=VllmSparseRefitConfig)
    nixl: VllmNixlRefitConfig = Field(default_factory=VllmNixlRefitConfig)
    memory_lifecycle: VllmRefitMemoryLifecycleConfig = Field(
        default_factory=VllmRefitMemoryLifecycleConfig
    )
```

- [ ] **Step 4: Update exemplar and reference configs**

Set the documented/default value to `legacy_level1`; do not edit performance
recipes in this task.

- [ ] **Step 5: Run focused and schema tests**

```bash
pytest -q tests/unit/models/generation/test_vllm_generation.py -k refit_memory_lifecycle
pytest -q tests/unit/test_config_v2.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models/generation/vllm/config.py \
  tests/unit/models/generation/test_vllm_generation.py \
  examples/configs/grpo_math_1B.yaml \
  tests/unit/reference_configs/grpo_math_1B.yaml
git commit -s -m "feat(vllm): configure refit memory lifecycle"
```

### Task 2: Legacy baseline invariance and deep-sleep selection

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `tests/unit/models/generation/test_vllm_worker_helpers.py`

**Interfaces:**
- Consumes: `resolve_vllm_refit_memory_lifecycle`
- Produces: `_refit_sleep_level(config: VllmConfig) -> Literal[1, 2]`
- Produces: `BaseVllmGenerationWorker.uses_specdec_deep_refit: bool`

- [ ] **Step 1: Write failing sleep-selection tests**

Use a fake LLM that records `sleep(level=...)`. Assert that missing/default
configuration calls `sleep(level=1)`, explicit deep refit calls level 2, and
the legacy path does not call `collective_rpc`.

```python
def test_legacy_worker_sleep_remains_level_one(worker, fake_llm):
    worker.llm = fake_llm
    worker.sleep()
    assert fake_llm.sleep_calls == [1]
    assert fake_llm.collective_rpc_calls == []
```

- [ ] **Step 2: Run the test and verify RED**

```bash
pytest -q tests/unit/models/generation/test_vllm_worker_helpers.py -k refit_sleep
```

Expected: explicit deep refit still invokes level 1.

- [ ] **Step 3: Implement sleep selection with no legacy side effects**

Resolve the lifecycle once during worker initialization and use the stored
boolean in `sleep()`. The legacy branch must contain only the existing prefix
cache reset, `llm.sleep(level=1)`, GC, and CUDA cache release.

- [ ] **Step 4: Run focused worker tests**

```bash
pytest -q tests/unit/models/generation/test_vllm_worker_helpers.py -k 'refit_sleep or sleep'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/vllm/vllm_worker.py \
  tests/unit/models/generation/test_vllm_worker_helpers.py
git commit -s -m "feat(vllm): select refit sleep level explicitly"
```

### Task 3: Static drafter snapshot and restore contract

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `tests/unit/models/generation/test_vllm_backend.py`

**Interfaces:**
- Produces: `VllmInternalWorkerExtension.snapshot_static_drafter() -> bool`
- Produces: `VllmInternalWorkerExtension.restore_static_drafter() -> bool`
- Produces: `VllmInternalWorkerExtension.describe_static_drafter_snapshot() -> dict[str, int]`

- [ ] **Step 1: Write failing snapshot tests**

Construct a fake drafter module with parameters and buffers. Assert that the
snapshot is a detached CPU clone, mutation of live tensors does not mutate the
snapshot, restore copies every tensor back, and the returned description
contains exact tensor and byte counts.

```python
def test_static_drafter_snapshot_restores_parameters_and_buffers(extension):
    assert extension.snapshot_static_drafter()
    expected = {name: value.clone() for name, value in extension._draft_state()}
    for _, value in extension._draft_state():
        value.zero_()
    assert extension.restore_static_drafter()
    for name, value in extension._draft_state():
        assert torch.equal(value.cpu(), expected[name].cpu())
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pytest -q tests/unit/models/generation/test_vllm_backend.py -k static_drafter
```

Expected: missing-method failures.

- [ ] **Step 3: Implement the snapshot contract**

Use `_get_drafter_model()` and store post-load runtime parameters and persistent
buffers by fully qualified name. Validate name, shape, dtype, and byte count on
restore. Copy into the existing tensors under `torch.no_grad()`; do not replace
Parameter objects because compiled graphs retain their identities. Return
`False` only on pipeline stages that legitimately do not own a drafter; raise
on the owning stage when state is absent or incomplete.

- [ ] **Step 4: Add failure tests**

Cover missing tensor, mismatched shape, mismatched dtype, and restore without a
snapshot. Each condition must raise `RuntimeError` before generation resumes.

- [ ] **Step 5: Run backend tests**

```bash
pytest -q tests/unit/models/generation/test_vllm_backend.py -k static_drafter
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models/generation/vllm/vllm_backend.py \
  tests/unit/models/generation/test_vllm_backend.py
git commit -s -m "feat(vllm): preserve static drafter across deep refit"
```

### Task 4: Validate and initialize the deep-refit worker

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `tests/unit/models/generation/test_vllm_worker_helpers.py`

**Interfaces:**
- Consumes: `snapshot_static_drafter`, `restore_static_drafter`
- Produces: `BaseVllmGenerationWorker.restore_drafter_after_refit() -> bool`
- Produces: setup-time validation of SpecDec configuration and drafter capability

- [ ] **Step 1: Write failing setup validation tests**

Assert that deep refit without `vllm_kwargs.speculative_config` fails, deep
refit with a missing drafter fails, and legacy mode accepts both baseline and
SpecDec configurations without snapshot RPCs.

- [ ] **Step 2: Run the tests and verify RED**

```bash
pytest -q tests/unit/models/generation/test_vllm_worker_helpers.py -k deep_refit
```

Expected: no validation or snapshot behavior exists.

- [ ] **Step 3: Initialize static draft ownership**

During `post_init()`, call `snapshot_static_drafter` on vLLM internal workers
only in `specdec_deep_refit` when `_draft_weights_from_refit` and
`_mtp_weights_from_refit` are both false. Require at least one owning stage and
retain the per-rank results. When either internal flag is true, register the
existing target/draft refit stream as the drafter provider and do not allocate
a frozen snapshot. Add `restore_drafter_after_refit()` that invokes the restore
RPC only for the frozen provider and treats successful streamed draft loading
as the online provider's completion acknowledgement.

- [ ] **Step 4: Add baseline no-RPC regression test**

Explicitly assert that `post_init()`, `sleep()`, and the refit restore hook make
zero draft-related RPCs in `legacy_level1` mode.

- [ ] **Step 5: Run worker tests**

```bash
pytest -q tests/unit/models/generation/test_vllm_worker_helpers.py -k 'deep_refit or legacy'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models/generation/vllm/vllm_worker.py \
  tests/unit/models/generation/test_vllm_worker_helpers.py
git commit -s -m "feat(vllm): initialize specdec deep refit safely"
```

### Task 5: Driver-level restore ordering and subphase timers

**Files:**
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `tests/unit/algorithms/test_grpo.py`
- Modify: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**
- Produces: `VllmGeneration.requires_drafter_restore_after_refit: bool`
- Produces: `VllmGeneration.restore_drafter_after_refit() -> bool`
- Produces: six child refit timers under `prepare_for_generation/*`

- [ ] **Step 1: Write a failing call-order test**

Use recording fake policy and generation objects. Deep refit must produce:

```python
[
    "policy.offload_before_refit",
    "generation.wake.weights",
    "target.transfer",
    "generation.restore_drafter",
    "policy.offload_after_refit",
    "generation.wake.kv_cache",
]
```

Legacy mode must preserve the current list without
`generation.restore_drafter`.

- [ ] **Step 2: Run the test and verify RED**

```bash
pytest -q tests/unit/algorithms/test_grpo.py -k refit_call_order
```

Expected: the deep-refit restore operation is absent.

- [ ] **Step 3: Implement driver-level restore and timers**

Wrap each existing operation in its child timer. After successful target
transfer, call drafter restore only when the local generation configuration
requires it. If restore returns false, raise before policy offload completion
or KV wake. Retain the existing
`prepare_for_generation/transfer_and_update_weights` metric name.

- [ ] **Step 4: Test timer coverage and baseline invariance**

Assert that child timers are entered in call order and that legacy/no-SpecDec
does not execute the drafter timer body or remote restore call.

- [ ] **Step 5: Run focused tests**

```bash
pytest -q tests/unit/algorithms/test_grpo.py -k 'refit_call_order or prepare_for_generation'
pytest -q tests/unit/models/generation/test_vllm_generation.py -k drafter_restore
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/algorithms/grpo.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/models/generation/test_vllm_generation.py
git commit -s -m "perf(grpo): restore drafter within timed refit transaction"
```

### Task 6: Bounded memory diagnostics

**Files:**
- Create: `nemo_rl/models/generation/vllm/refit_memory_diagnostics.py`
- Create: `tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`

**Interfaces:**
- Produces: `RefitMemorySnapshot` dataclass
- Produces: `capture_refit_memory_snapshot(phase: str) -> RefitMemorySnapshot`
- Produces: one structured log record per deep-refit phase boundary

- [ ] **Step 1: Write failing snapshot tests**

Patch `/proc/self/statm`, `os.sysconf`, and CUDA memory functions to fixed
values. Assert exact RSS, available-memory, allocated, and reserved-byte fields.

- [ ] **Step 2: Run the test and verify RED**

```bash
pytest -q tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py
```

Expected: module import failure.

- [ ] **Step 3: Implement bounded local snapshots**

Read only the current process and one `/proc/meminfo` record; do not enumerate
processes or nodes. Log snapshots only in deep-refit mode so the baseline path
performs no additional system calls.

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py
pytest -q tests/unit/models/generation/test_vllm_worker_helpers.py -k deep_refit
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/vllm/refit_memory_diagnostics.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py
git commit -s -m "perf(vllm): record deep-refit memory phases"
```

### Task 7: Local verification and static checks

**Files:**
- Modify only files required by formatter or type checker from Tasks 1-6

**Interfaces:**
- Consumes all preceding tasks
- Produces a locally verified commit ready for GPU gating

- [ ] **Step 1: Run the focused suite**

```bash
pytest -q \
  tests/unit/models/generation/test_vllm_worker_helpers.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py \
  tests/unit/algorithms/test_grpo.py
```

- [ ] **Step 2: Run formatting and lint checks on touched files**

```bash
pre-commit run --files \
  nemo_rl/models/generation/vllm/config.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/refit_memory_diagnostics.py \
  nemo_rl/algorithms/grpo.py
```

- [ ] **Step 3: Review the baseline branch diff**

Verify mechanically that `legacy_level1` makes no new draft RPC and retains
`llm.sleep(level=1)`.

- [ ] **Step 4: Commit verification-only fixes if required**

```bash
git add nemo_rl/models/generation/vllm/config.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/refit_memory_diagnostics.py \
  nemo_rl/algorithms/grpo.py \
  tests/unit/models/generation/test_vllm_worker_helpers.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_refit_memory_diagnostics.py \
  tests/unit/algorithms/test_grpo.py
git commit -s -m "test(vllm): verify specdec deep refit"
```

### Task 8: GPU smoke and matched baseline gate

**Files:**
- Create: `experiments/q235_specdec_deep_refit_20260917/README.md`
- Create: `experiments/q235_specdec_deep_refit_20260917/PLAN.md`
- Create: `experiments/q235_specdec_deep_refit_20260917/configs/`
- Create: `experiments/q235_specdec_deep_refit_20260917/scripts/`
- Create: `experiments/q235_specdec_deep_refit_20260917/results/`

**Interfaces:**
- Consumes the signed, pushed implementation commit
- Produces matched baseline and DFlash K7 three-step evidence

- [ ] **Step 1: Create reproducible experiment manifests**

Record commit SHA, immutable container path/hash, target and drafter paths,
account, partition, recipe, overrides, seeds, CUDA Graph mode/sizes, and W&B
names. The baseline omits the deep-refit override. DFlash K7 adds:

```text
++policy.generation.refit_cfg.memory_lifecycle.mode=specdec_deep_refit
```

- [ ] **Step 2: Commit and push before submission**

```bash
git add experiments/q235_specdec_deep_refit_20260917
git commit -s -m "test(q235): add deep-refit GPU gate"
git push
```

- [ ] **Step 3: Run SLURM test-only checks**

Use the cluster launcher with the selected account and partition. Verify node,
GPU, time, container, mounts, and output paths without allocating GPUs.

- [ ] **Step 4: Submit matched three-step baseline and DFlash K7 gates**

Use filtered scheduler queries for only the submitted job IDs and monitor for
at least five minutes. Do not raise the Ray threshold above 0.95.

- [ ] **Step 5: Validate correctness and memory gates**

Require nonzero acceptance after each wake, finite reward/KL/entropy metrics,
complete child timing, no host-memory growth across cycles, and no OOM.

- [ ] **Step 6: Record results and commit**

Write job IDs, W&B URLs, metric windows, failure evidence, and the go/no-go
decision into `results/` and `README.md`, then commit with sign-off.

### Task 9: Twenty-step performance cohort and cross-model figure

**Files:**
- Modify: `experiments/q235_specdec_deep_refit_20260917/README.md`
- Create: `experiments/q235_specdec_deep_refit_20260917/results/steps3_20.csv`
- Create: `experiments/q235_specdec_deep_refit_20260917/results/summary.json`
- Modify: `docs/blog/figures/render_cross_model_speedups.py` in the technical-blog worktree
- Create: `docs/blog/figures/data/q30_q235_refit_speedups.json` in the technical-blog worktree
- Create: `docs/blog/figures/specdec_q30_q235_speedups.png` in the technical-blog worktree
- Create: `docs/blog/figures/specdec_q30_q235_speedups.pdf` in the technical-blog worktree

**Interfaces:**
- Consumes matched Qwen3-30B-A3B results and completed Qwen3-235B results
- Produces actual generation speedup, actual E2E speedup, and projected
  generation-only E2E speedup

- [ ] **Step 1: Submit the 20-step matched cohort after the smoke gate passes**

Submit baseline, DFlash K5/K7, and DSpark K5/K7 with identical target recipe,
hardware, image, data, and seeds. Only SpecDec-specific configuration may
differ.

- [ ] **Step 2: Extract time-weighted Steps 3-20 metrics**

Compute:

```python
generation_speedup = baseline_generation_seconds / arm_generation_seconds
actual_e2e_speedup = baseline_total_seconds / arm_total_seconds
generation_fraction = baseline_generation_seconds / baseline_total_seconds
projected_e2e_speedup = 1 / (
    (1 - generation_fraction) + generation_fraction / generation_speedup
)
```

Reject incomplete or quality-invalid windows.

- [ ] **Step 3: Render the publication figure**

Use the mandatory Paired palette, navy edges, dashed y-grid, black 1.0x
reference, top-centered legend, 300-DPI PNG, and vector PDF. Keep the existing
Qwen3-30B-A3B measured Generation/E2E panel. Add a Qwen3-235B panel with
`Generation`, `Projected E2E (generation-only)`, and `Actual E2E` bars. Label
provisional or failed-refit rows as diagnostics rather than claims.

- [ ] **Step 4: Validate quality metrics**

Compare reward, mean generated length, approximate entropy, policy KL,
generation KL, failure count, and acceptance alongside the timing table.

- [ ] **Step 5: Update HTML/Markdown and commit assets**

Add a concise caption explaining the projection formula and the measured gap.
Commit source data, renderer, PNG, PDF, and manifest together.
