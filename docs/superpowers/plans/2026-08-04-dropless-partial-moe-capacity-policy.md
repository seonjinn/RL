# Dropless Partial MoE and Fixed-Capacity Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make packed THD Transformer Engine CUDA Graph execution correct and measurable for production dropless `moe_router+moe_preprocess`, while keeping whole-MoE and whole-layer capture behind an explicit, zero-drop fixed-capacity experiment gate.

**Architecture:** Megatron-Core owns dispatcher replay state and device-resident MoE capacity telemetry. NeMo-RL forwards only explicitly configured capacity fields, permits dropless partial MoE only for code paths proven by a candidate-SHA-bound distributed gate, and rejects any overflow or valid-token drop collectively before `optimizer.step()`. Experiment tooling distinguishes missing replay evidence from missing fixed capacity and never silently changes dropless recipe semantics.

**Tech Stack:** Python 3.12, PyTorch distributed, Megatron-Core, Megatron-Bridge, Transformer Engine partial CUDA Graphs, NeMo-RL/Ray, Hydra YAML, pytest, SLURM, W&B/TensorBoard, static HTML reporting.

## Global Constraints

- Preserve the exact packed THD physical signature: shape, dtype, device, layout, stride, structural mask, sequence capacity, token capacity, and normalized schedule key.
- Use exactly three successful optimizer steps for CUDA Graph warmup.
- Keep dispatch communication, expert compute, combine, and postprocess eager for production dropless partial MoE.
- Do not add a rank-local mid-forward fallback, routing-value graph bank, output padding repair, silent token drop, or automatic in-step capacity growth.
- Reject or repack unknown physical geometry before model execution. Recapture is allowed only at a globally drained pre-forward boundary.
- Whole-MoE and whole-layer rows are experimental and require a matched eager arm with identical capacity knobs plus an original dropless reference arm.
- Any expert-capacity drop, HybridEP rank overflow, or valid token losing a route aborts the step collectively before the optimizer update.
- Keep checkpointing disabled in experiment jobs, use W&B project `sna-cg-study`, use batch partitions, and store artifacts under `experiments/cuda_graph/nemotron_thd_te_graph_20260731/`.
- Before every SLURM submission: update the relevant branch, run scheduler `--test-only`, commit with `git commit -s`, push all dependency repositories, and bind the job to full source/container/runtime digests.
- Monitor each submitted row for at least five minutes and cancel allocations that are idle because of a failed bootstrap or missing rank.
- Preserve the existing uncommitted harness/report edits and commit them separately from production changes.

---

## Current Root-Cause Map

1. Megatron-Core commit `22919a3d7` already contains dispatcher-owned AlltoAll and Flex/HybridEP replay state. Its dropless HybridEP state deliberately stores `num_permuted_tokens=None`, leaving routing-dependent allocation to eager dispatch. The NeMo-RL gitlink is still at this commit, while the pushed MCore development branch is five verification commits ahead at `32d79616a`; implementation must start from that branch worktree and pin the final SHA only after verification.
2. NeMo-RL still rejects dropless Flex/HybridEP `moe_preprocess` unless expert or rank capacity is fixed. Removing that check without evidence would make an unverified path appear supported.
3. `mcore_test_matrix.json` names packed-eval and fixed-capacity HybridEP pytest nodes that do not exist in the current candidate. Therefore those manifest rows are not correctness evidence.
4. NeMo-RL does not declare or forward `moe_expert_capacity_factor`, `moe_pad_expert_input_to_capacity`, or `moe_expert_rank_capacity_factor`.
5. HybridEP exposes an `over_budget` device flag only through dispatcher internals, while AlltoAll router dropping has no step-level public zero-drop contract. NeMo-RL currently reaches `optimizer.step()` without consuming either condition.
6. `scope_matrix.py` calls unproven dropless `moe_preprocess` capacity-blocked. Missing partial-replay proof is a dependency block; only whole-MoE without a validated capacity profile is capacity-blocked.
7. OCI-HSG stage job `5852837` stopped after 40 passing tests because a subprocess test inherited ambient `RUNTIME_STAGE_CAPABILITY`. The local helper fix already proves 245 tests pass and must be committed before another immutable runtime gate is submitted.

## File and Ownership Map

### Megatron-Core candidate branch

- Development root: `/Users/sna/CudaGraph_PR/task2-mcore-candidate`, branch `sj/thd-cg-hybrid-nemotron-20260731`
- The nested NeMo-RL MCore submodule is a detached pin target, not the development worktree.
- Replay and telemetry types: `megatron/core/transformer/moe/cuda_graph_replay.py`, new `megatron/core/transformer/moe/capacity_tracker.py`
- Router/drop accounting: `megatron/core/transformer/moe/router.py`
- HybridEP rank-overflow accounting: `megatron/core/transformer/moe/token_dispatcher.py`
- Capture-boundary reset and continuation ownership: `megatron/core/transformer/transformer_layer.py`, `megatron/core/transformer/cuda_graphs.py`
- Unit tests: new `tests/unit_tests/transformer/moe/test_capacity_tracker.py`, existing `tests/unit_tests/transformer/moe/test_token_dispatcher.py`, existing `tests/unit_tests/transformer/test_cuda_graphs.py`
- Distributed proof: new `tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py`

### NeMo-RL integration branch

- Typed config and forwarding: `nemo_rl/models/policy/__init__.py`, `nemo_rl/models/megatron/setup.py`
- Step guard and worker result: `nemo_rl/models/policy/workers/megatron_policy_worker.py`, `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- DP aggregation/logging: `nemo_rl/algorithms/utils.py`, policy aggregation call sites that already merge CUDA Graph metrics
- Unit tests: `tests/unit/models/megatron/test_megatron_setup.py`, `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`, new `tests/unit/models/megatron/test_moe_capacity_guard.py`, relevant algorithm utility tests

### Persistent experiment harness

- Capability matrix: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json`
- Typed distributed runner: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`
- Classification and command rendering: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`, `run_scope.sh`
- Model topology selectors: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/*.env`
- New proof validation: `validate_partial_moe_gate.py`, `validate_moe_capacity_profile.py`
- New capacity workflow: `submit_moe_capacity_calibration.sh`, `select_moe_capacity.py`, `results/manifests/moe-capacity-profiles.json`
- Collection/reporting: `export_wandb.py`, `export_tensorboard.py`, `collect_results.py`, `render_report.py`, `report_context.json`, generated `results/report.html`
- Harness tests: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`, `test_nemotron_thd_te_graph_reporting.py`, `test_export_wandb.py`, `test_export_tensorboard.py`, `test_container_harness_hardening.py`

### Dependency pin order

1. Update and push Megatron-Core branch `sj/thd-cg-hybrid-nemotron-20260731` from `/Users/sna/CudaGraph_PR/task2-mcore-candidate`.
2. Pin that full SHA in the Megatron-Bridge branch and push Bridge.
3. Pin the Bridge SHA in NeMo-RL branch `experiment/thd-cg-hybrid-nemotron-20260731` and push NeMo-RL.
4. Generate a new immutable runtime attestation against those exact SHAs.
5. Submit correctness gates; performance submission is forbidden until they pass.

---

### Task 1: Land the Runtime-Stage Environment Isolation Fix

**Files:**

- Modify: `tests/unit/experiments/test_container_harness_hardening.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/report_context.json`
- Regenerate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html`

**Interface:** Every subprocess launched by the harness test helper starts without an ambient `RUNTIME_STAGE_CAPABILITY`; an individual test must supply that variable explicitly when it is part of the case.

- [ ] **Step 1: Preserve the red evidence**

Record OCI-HSG job `5852837`, its 40 passed tests, and the inherited-capability failure in `report_context.json`. Do not rerun a GPU job from this failed stage artifact.

Expected RED: the immutable job artifact reports failure after 40 passes and cannot produce a runtime capability marker.

- [ ] **Step 2: Verify the focused regression test**

The implementation line is:

```python
env.pop("RUNTIME_STAGE_CAPABILITY", None)
```

Run:

```bash
RUNTIME_STAGE_CAPABILITY=mcore-test-v1 uv run pytest -q \
  tests/unit/experiments/test_container_harness_hardening.py
```

Expected GREEN: every subprocess case controls capability presence itself and the file passes with the ambient variable set.

- [ ] **Step 3: Run the exact stage suite**

```bash
RUNTIME_STAGE_CAPABILITY=mcore-test-v1 uv run pytest -q \
  tests/unit/experiments/test_validate_te_runtime.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_container_harness_hardening.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
```

Expected GREEN: `245 passed`, matching the verified local result.

- [ ] **Step 4: Commit only the existing harness/report changes**

```bash
git diff --check
git add tests/unit/experiments/test_container_harness_hardening.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/report_context.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html
git commit -s -m "test: isolate CUDA graph stage capability"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 2: Add Device-Resident MoE Capacity Telemetry in Megatron-Core

**Files (relative to `/Users/sna/CudaGraph_PR/task2-mcore-candidate`):**

- Create: `megatron/core/transformer/moe/capacity_tracker.py`
- Modify: `megatron/core/transformer/moe/router.py`
- Modify: `megatron/core/transformer/moe/token_dispatcher.py`
- Create: `tests/unit_tests/transformer/moe/test_capacity_tracker.py`
- Modify: `tests/unit_tests/transformer/moe/test_token_dispatcher.py`
- Modify: `tests/unit_tests/transformer/moe/test_routers.py`

**Interface:** Add a pointer-stable, process-local device tracker initialized before graph capture. It accumulates four `torch.int64` scalars and never performs `.item()` or a collective inside a captured forward:

```python
@dataclass(frozen=True)
class MoECapacitySnapshot:
    selected_assignments: torch.Tensor
    dropped_assignments: torch.Tensor
    valid_token_drops: torch.Tensor
    rank_overflow_events: torch.Tensor


class MoECapacityTracker:
    def __init__(self) -> None:
        self._counters: torch.Tensor | None = None

    def initialize(self, device: torch.device) -> None:
        if self._counters is None:
            self._counters = torch.zeros(4, dtype=torch.int64, device=device)
        elif self._counters.device != device:
            raise RuntimeError(
                f"MoE capacity tracker is on {self._counters.device}, not {device}."
            )

    def reset(self) -> None:
        if self._counters is None:
            raise RuntimeError("MoE capacity tracker is not initialized.")
        self._counters.zero_()

    def snapshot(self) -> MoECapacitySnapshot:
        if self._counters is None:
            raise RuntimeError("MoE capacity tracker is not initialized.")
        values = self._counters.clone()
        return MoECapacitySnapshot(values[0], values[1], values[2], values[3])
```

`initialize()` is idempotent only for the same device and preserves the buffer identity after first initialization. `reset()` must preserve `data_ptr()`.

- [ ] **Step 0: Update the MCore development branch before editing**

```bash
cd /Users/sna/CudaGraph_PR/task2-mcore-candidate
git status --short --branch
git fetch origin upstream
git pull --ff-only origin sj/thd-cg-hybrid-nemotron-20260731
git merge upstream/main
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

Expected GREEN: the worktree is clean before the merge, the branch contains the five verification commits through `32d79616a`, and the remote branch matches the post-merge local SHA.

- [ ] **Step 1: Write failing tracker lifecycle tests**

Add tests that require:

- initialization before record;
- reset preserves the counter tensor pointer;
- snapshots do not alias mutable counters;
- negative counts and device changes fail;
- two records accumulate without a host sync.

Run from the Megatron-Core root:

```bash
uv run pytest -q tests/unit_tests/transformer/moe/test_capacity_tracker.py
```

Expected RED: import failure because `capacity_tracker.py` does not exist.

- [ ] **Step 2: Implement the minimal tracker**

Expose `get_moe_capacity_tracker()` and `destroy_moe_capacity_tracker()` following the existing `moe_logging.py` global-owner pattern. Record methods accept scalar tensors, cast to `int64` on-device, and update the fixed buffer in place.

Expected GREEN:

```bash
uv run pytest -q tests/unit_tests/transformer/moe/test_capacity_tracker.py
```

- [ ] **Step 3: Add red router drop-accounting tests**

Before `apply_router_token_dropping`, retain the original sparse selection map. After capacity masking, define:

```python
kept_routes = original_routing_map & routing_map
dropped_assignments = original_routing_map.sum() - kept_routes.sum()
valid_token_drops = (
    original_routing_map.any(dim=-1) & ~kept_routes.any(dim=-1)
).sum()
```

Tests cover both `pad_to_capacity=True` and `False`, a partial top-k drop, a fully dropped valid token, padding rows, and no-capacity dropless mode.

Run:

```bash
uv run pytest -q \
  tests/unit_tests/transformer/moe/test_capacity_tracker.py \
  tests/unit_tests/transformer/moe/test_routers.py -k capacity
```

Expected RED: the tracker remains zero after router capacity masking.

- [ ] **Step 4: Record AlltoAll and HybridEP events**

Record router assignment/drop counters only when `moe_expert_capacity_factor` is configured. In HybridEP static rank-capacity mode, record `handle[-1] != 0` after dispatch. Dropless HybridEP with both capacity factors absent records no overflow and does not change allocation behavior.

Run:

```bash
uv run pytest -q \
  tests/unit_tests/transformer/moe/test_capacity_tracker.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_routers.py -k 'capacity or dropping or hybridep'
```

Expected GREEN: exact counter values, zero dropless events, and no new GPU-to-host synchronization.

- [ ] **Step 5: Commit and push the MCore telemetry change**

```bash
uv run isort megatron/core/transformer/moe/capacity_tracker.py \
  megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_capacity_tracker.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py
git diff --check
git add megatron/core/transformer/moe/capacity_tracker.py \
  megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  tests/unit_tests/transformer/moe/test_capacity_tracker.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py
git commit -s -m "feat: expose MoE capacity safety telemetry"
git push origin HEAD:sj/thd-cg-hybrid-nemotron-20260731
```

---

### Task 3: Prove Dropless Partial MoE with Real Distributed Replay

**Files:**

- Create in MCore: `tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py`
- Modify in MCore only if a test exposes a real invariant gap: `megatron/core/transformer/moe/cuda_graph_replay.py`, `token_dispatcher.py`, `transformer_layer.py`, `cuda_graphs.py`
- Modify in NeMo-RL: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json`
- Modify in NeMo-RL: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`
- Modify tests: `tests/unit/experiments/test_mcore_standalone_driver.py`, `tests/unit/experiments/test_matrix_submitters.py`

**Interface:** Add four literal, candidate-SHA-bound distributed rows. Each runs 3 capture warmups plus 20 changed-route replays and asserts eager/graph output, loss, route, gradient, and parameter-delta parity; `graph_calls > 0`; exact continuation geometry; and zero capacity events.

| Row | World size | Topology contract | Dispatcher |
|---|---:|---|---|
| `dropless_hybridep_nano16` | 16 | TP2/PP2/CP2/EP8 | Flex/HybridEP |
| `dropless_alltoall_qwen30_16` | 16 | TP1/PP1/CP1/EP16 | AlltoAll |
| `dropless_alltoall_super32` | 32 | TP2/PP1/CP1/EP16 | AlltoAll |
| `dropless_hybridep_qwen235_64` | 64 | TP2/PP4/CP2/EP16 | Flex/HybridEP |

Ultra remains externally blocked until its concrete NeMo-RL model/data/profile topology is resolved; its existing Bridge provider can receive a separate standalone row without claiming NeMo-RL coverage.

- [ ] **Step 1: Add red manifest-integrity tests**

Require every `pytest_nodes` entry to resolve to a collected node in the candidate archive. Extend the allowed allocation set with `(16, 4, 64)` and reject any 64-rank row using another layout.

```bash
uv run pytest -q \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py
```

Expected RED: existing `packed_eval_8`, `packed_tp2_cp2_pp2_8`, `hybrid_ep16`, and `hybrid_ep32` entries reference absent nodes.

- [ ] **Step 2: Replace absent claims with literal executable nodes**

Do not retain a manifest row merely because an older plan named it. Every row must map to a real function in `test_partial_moe_cuda_graph_distributed.py`. The test selects its topology from the literal parameter ID and validates the actual initialized process groups before model construction.

Expected GREEN for static integrity:

```bash
uv run pytest -q \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py
```

- [ ] **Step 3: Add the smallest real distributed parity test**

Start with `dropless_hybridep_nano16`. Alternate two route distributions with the same physical THD signature. Compare eager and graph valid-token outputs, router probabilities/IDs/counts, every local parameter gradient, and the simulated optimizer delta. Assert dispatch/expert/combine/postprocess execute eagerly while router/preprocess graph counters are nonzero.

Run a one-node CUDA developer check only when 16 local GPUs are available; otherwise rely on the typed SLURM row after push. CPU collection must still succeed:

```bash
uv run pytest --collect-only -q \
  tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py
```

Expected GREEN: all four literal nodes collect with exact parameter IDs.

- [ ] **Step 4: Add AlltoAll and larger-topology cases**

Use the same assertions for Qwen3-30B-A3B, Super, and Qwen3-235B topology contracts. Synthetic test dimensions may be small, but process-group sizes, dispatcher, layer type, packed metadata, top-k, shared-expert state, and graph scope must match the named production topology.

- [ ] **Step 5: Commit MCore tests and any required invariant fix**

```bash
uv run isort tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py \
  megatron/core/transformer/moe/cuda_graph_replay.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/cuda_graphs.py
git diff --check
git add tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py
git add -u megatron/core/transformer/moe/cuda_graph_replay.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/cuda_graphs.py
git commit -s -m "test: prove dropless partial MoE graph replay"
git push origin HEAD:sj/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 6: Commit the typed NeMo-RL manifest update separately**

```bash
uv run pytest -q \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py
git commit -s -m "test: define partial MoE distributed gates"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 4: Forward Capacity Knobs and Correct the NeMo-RL Scope Gate

**Files:**

- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`

**Interface:** Add optional typed fields and forward them only when present:

```python
moe_expert_capacity_factor: NotRequired[float | None]
moe_pad_expert_input_to_capacity: NotRequired[bool]
moe_expert_rank_capacity_factor: NotRequired[float | None]
```

The validation matrix is:

- `moe_router` alone: dropless allowed for supported dispatchers.
- `moe_router+moe_preprocess`: dropless AlltoAll or dropless Flex/HybridEP allowed; overlap, uneven-input padding, AllGather, DeepEP, and NCCL-EP remain rejected.
- `moe` or empty whole-layer scope on an MoE model: positive fixed capacity remains mandatory.
- Capacity fields omitted: preserve provider defaults exactly.
- Capacity fields explicitly set: assign them before model `__post_init__()`.

- [ ] **Step 1: Add failing schema and forwarding tests**

Tests cover each field independently, all fields together, explicit `None`, omission, invalid booleans/numbers, and ordering before post-init.

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_setup.py \
  -k 'capacity or moe_preprocess'
```

Expected RED: the three config fields are absent or ignored.

- [ ] **Step 2: Add presence-based forwarding**

Use one explicit allowlist inside `_apply_moe_config`:

```python
for name in (
    "moe_expert_capacity_factor",
    "moe_pad_expert_input_to_capacity",
    "moe_expert_rank_capacity_factor",
):
    if name in config["megatron_cfg"]:
        setattr(model_cfg, name, config["megatron_cfg"][name])
```

Expected GREEN: forwarding tests pass and omission leaves sentinel provider defaults unchanged.

- [ ] **Step 3: Replace the stale HybridEP fixed-capacity rejection**

Change only the partial `moe_preprocess` path. Keep the whole-MoE capacity check unchanged. Add positive tests for dropless HybridEP and negative tests for every unsupported backend/overlap/uneven-input combination.

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_setup.py
```

Expected GREEN: dropless partial HybridEP reaches model post-init; whole-MoE without capacity still fails.

- [ ] **Step 4: Commit the NeMo config/gate change**

```bash
git diff --check
git add nemo_rl/models/policy/__init__.py \
  nemo_rl/models/megatron/setup.py \
  tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "feat: enable proven dropless partial MoE graphs"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 5: Abort Unsafe Capacity Steps Before the Optimizer

**Files:**

- Modify: `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/algorithms/utils.py`
- Create: `tests/unit/models/megatron/test_moe_capacity_guard.py`
- Modify: `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`
- Modify: `tests/unit/algorithms/test_utils.py`

**Interface:** Capacity telemetry is separate from CUDA Graph telemetry so eager calibration rows receive it too:

```python
@dataclass(frozen=True)
class MoECapacityStepMetrics:
    selected_assignments_rank_sum: int
    dropped_assignments_rank_sum: int
    valid_token_drops_rank_sum: int
    rank_overflow_events_rank_sum: int

    @property
    def safe(self) -> bool:
        return (
            self.dropped_assignments_rank_sum == 0
            and self.valid_token_drops_rank_sum == 0
            and self.rank_overflow_events_rank_sum == 0
        )
```

Every rank resets the MCore tracker in `begin_train_step`. After a new bank is captured, reset once more before the real schedule so capture forwards do not count. `finish_train_step` snapshots and sum-reduces the four counters over the world before gradient scaling and before `self.optimizer.step()`.

- [ ] **Step 1: Add red no-optimizer-on-overflow tests**

Test expert drop, fully dropped valid token, HybridEP rank overflow, safe zero counters, absent capacity knobs, capture-time reset, and a rank-asymmetric local flag. The asymmetric test must produce the same failure on all mocked ranks and assert `optimizer.step.call_count == 0`.

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_moe_capacity_guard.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py
```

Expected RED: `finish_train_step` calls the optimizer without inspecting capacity state.

- [ ] **Step 2: Add the fail-closed worker lifecycle**

Place the guard before gradient normalization and optimizer work:

```python
capacity_metrics = self._collect_moe_capacity_step_metrics()
if capacity_metrics is not None and not capacity_metrics.safe:
    raise RuntimeError(
        "MoE fixed-capacity step dropped routing assignments or exceeded rank capacity; "
        "abort_train_step must run before another step."
    )
```

The outer `finish_train_step` error path leaves `_train_step_state` available for the existing idempotent `abort_train_step`; it must not advance the CUDA Graph warmup counter.

- [ ] **Step 3: Aggregate and log replicated metrics**

Worker results use `moe_capacity_metrics`. Policy aggregation treats world-reduced values as replicated, validates exact keys and plain nonnegative integers, and emits:

```text
moe_capacity/selected_assignments_rank_sum
moe_capacity/dropped_assignments_rank_sum
moe_capacity/valid_token_drops_rank_sum
moe_capacity/rank_overflow_events_rank_sum
```

Do not encode `safe` as a free-form boolean; collectors derive it from the three unsafe counters.

- [ ] **Step 4: Run focused and broader worker tests**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_moe_capacity_guard.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/algorithms/test_utils.py
```

Expected GREEN: unsafe steps never update parameters, safe eager and graph steps produce identical metric schemas, and absent capacity instrumentation preserves existing results.

- [ ] **Step 5: Commit and push**

```bash
git diff --check
git add nemo_rl/models/megatron/cuda_graph_lifecycle.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/algorithms/utils.py \
  tests/unit/models/megatron/test_moe_capacity_guard.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/algorithms/test_utils.py
git commit -s -m "fix: reject unsafe MoE capacity steps"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 6: Make Capability and Capacity Evidence Explicit in the Harness

**Files:**

- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/nano.env`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/super.env`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_30ba3b.env`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_235b.env`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_partial_moe_gate.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_moe_capacity_profile.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interface:** Classification consumes validated evidence, not a manual success label.

- Missing partial replay artifact for `moe_preprocess`: `dependency-blocked`.
- Missing fixed-capacity profile for `moe` or whole-layer: `capacity-blocked`.
- Artifact candidate SHA, integration SHA, dispatcher, TP/PP/CP/EP, packed-token capacity, top-k, model, and container/runtime identity must match the launch profile.
- A capacity profile is invalid if any calibration step has a nonzero unsafe counter.

- [ ] **Step 1: Add red classification tests**

Replace assertions that Nano/Qwen235 preprocess is capacity-blocked. Require dependency-blocked without a gate, runnable with a matching gate, and dependency-blocked with a stale SHA or wrong topology. Keep whole-MoE capacity-blocked until a validated profile is supplied.

```bash
uv run pytest -q \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'classifier or selector or capacity or partial_moe'
```

Expected RED: the classifier has no evidence inputs and reports the old capacity reason.

- [ ] **Step 2: Add typed artifact validators**

Validators accept a regular JSON file plus an expected SHA256 and return a frozen dataclass. They reject symlinks, mutable files, malformed numbers, unknown keys, non-passed status, missing ranks, zero graph calls, nonzero fallback, nonzero unsafe capacity counters, or provenance mismatch.

- [ ] **Step 3: Encode exact model topology**

Selectors declare the effective production topology:

```text
nano: TP2 PP2 CP2 EP8, 16 policy GPUs, HybridEP
super: TP2 PP1 CP1 EP16, 32 policy GPUs, AlltoAll
qwen3_30ba3b: TP1 PP1 CP1 EP16, 16 policy GPUs, AlltoAll
qwen3_235b: TP2 PP4 CP2 EP16, 64 policy GPUs, HybridEP
```

All four use packed token capacity 8192 and `thd_max_packed_sequences=16` for the initial gate. If an effective Hydra config differs, update the selector and test to the resolved value before submission; never coerce the runtime to this table silently.

- [ ] **Step 4: Render capacity only for experimental rows**

Production partial commands contain no expert/rank capacity override. Whole-MoE and whole-layer commands load the validated model/topology profile and append one of:

```text
++policy.megatron_cfg.moe_expert_capacity_factor=${CAPACITY_FACTOR_FROM_PROFILE}
++policy.megatron_cfg.moe_pad_expert_input_to_capacity=true
```

or:

```text
++policy.megatron_cfg.moe_expert_rank_capacity_factor=${CAPACITY_FACTOR_FROM_PROFILE}
```

The renderer obtains the numeric value from the parsed artifact and passes it as an argv element; it never evaluates profile text as shell.

Expected GREEN: matching partial-MoE evidence makes only the proven model/topology row runnable, while a matching capacity profile makes only its experimental whole-MoE rows capacity-ready.

- [ ] **Step 5: Run harness tests and commit**

```bash
uv run pytest -q \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_container_harness_hardening.py
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/models \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_partial_moe_gate.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_moe_capacity_profile.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "feat: gate MoE graph scopes on typed evidence"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 7: Add Reproducible Capacity Calibration and Reporting

**Files:**

- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_moe_capacity_calibration.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/select_moe_capacity.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/moe-capacity-profiles.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_wandb.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py`
- Create: `tests/unit/experiments/test_moe_capacity_selection.py`
- Modify: reporting/export tests under `tests/unit/experiments/`

**Interface:** The calibration submitter runs eager only and sweeps the literal factor grid `1.0,1.25,1.5,2.0,3.0,4.0,8.0`. The selector chooses the next tested factor above the smallest zero-event 20-step factor, then requires a separate 100-step eager soak at that selected factor. A profile is published only when every soak step has zero dropped assignments, zero valid-token drops, and zero rank-overflow events.

- [ ] **Step 1: Add red selector tests**

Cover exact topology matching, missing grid rows, unsafe lower factors, unsafe soak, no safe factor, non-finite values, mixed provenance, and deterministic selection.

```bash
uv run pytest -q tests/unit/experiments/test_moe_capacity_selection.py
```

Expected RED: the selector module does not exist.

- [ ] **Step 2: Implement the pure selection function**

The selection algorithm takes normalized result rows and returns a typed profile. It does not submit jobs or mutate selectors. If the first safe grid value is `8.0`, reject the model as capacity-blocked because no tested safety-margin value remains above it.

- [ ] **Step 3: Add persistent eager calibration submission**

For each factor, reuse the committed `run_scope.sh` path with CUDA Graph disabled and explicit capacity overrides. Use separate filesystem-safe run names and submit all factor rows in parallel after one successful `--test-only` pass. The 100-step soak is submitted only for the selected factor.

- [ ] **Step 4: Extend metric collection and HTML**

Add the four `moe_capacity/*` counters, derived `moe_capacity_safe`, selected factor, calibration job IDs, topology fingerprint, and profile digest. The HTML separates:

- original dropless eager;
- fixed-capacity eager;
- fixed-capacity CUDA Graph;
- production dropless partial CUDA Graph.

It must never label a fixed-capacity speed delta as the CUDA Graph delta against the original dropless recipe.

- [ ] **Step 5: Run reporting tests**

```bash
uv run pytest -q \
  tests/unit/experiments/test_moe_capacity_selection.py \
  tests/unit/experiments/test_export_wandb.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py
```

Expected GREEN: safe/unsafe rows remain distinguishable through export, collection, aggregation, and HTML rendering.

- [ ] **Step 6: Commit and push**

```bash
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_moe_capacity_calibration.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/select_moe_capacity.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/moe-capacity-profiles.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_wandb.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py \
  tests/unit/experiments/test_moe_capacity_selection.py \
  tests/unit/experiments/test_export_wandb.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py
git commit -s -m "feat: calibrate and report MoE graph capacity"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

---

### Task 8: Pin Dependency SHAs and Run the Immutable Distributed Gate

**Files:**

- Modify Bridge gitlink: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`
- Modify NeMo-RL gitlink: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Update exact profile files outside git only through the existing profile workflow
- Produce immutable artifacts under the configured `RUN_LOG_ROOT`

**Interface:** A partial-MoE gate is valid only when all rows reference the final pushed MCore SHA, Bridge SHA, NeMo-RL integration SHA, container SHA256, Transformer Engine SHA, runtime attestation, and complete rank/device topology.

- [ ] **Step 1: Verify and push MCore**

```bash
cd /Users/sna/CudaGraph_PR/task2-mcore-candidate
git fetch origin upstream
git status --short
git log -1 --oneline
git push origin HEAD:sj/thd-cg-hybrid-nemotron-20260731
MCORE_FINAL_SHA=$(git rev-parse HEAD)
git ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | \
  grep "$MCORE_FINAL_SHA"
```

Expected GREEN: the worktree is clean and the remote branch resolves to the local full SHA.

- [ ] **Step 2: Pin MCore in Bridge and Bridge in NeMo-RL**

Commit each gitlink in its owning repository, in dependency order:

```bash
cd /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731
MCORE_WORKTREE=/Users/sna/CudaGraph_PR/task2-mcore-candidate
MCORE_PIN=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
MCORE_FINAL_SHA=$(git -C "$MCORE_WORKTREE" rev-parse HEAD)
git -C "$MCORE_PIN" fetch origin
git -C "$MCORE_PIN" switch --detach "$MCORE_FINAL_SHA"

cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git add 3rdparty/Megatron-LM
git commit -s -m "deps: pin partial MoE CUDA graph MCore"
git push origin HEAD:sna/thd-cg-hybrid-nemotron-20260731

cd ../../..
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "deps: pin partial MoE CUDA graph Bridge"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 3: Update from the pushed branch before submission**

```bash
git pull --ff-only seonjinn experiment/thd-cg-hybrid-nemotron-20260731
git submodule update --init --recursive
```

Expected GREEN: all three repositories are clean and their gitlinks match the pushed SHAs.

- [ ] **Step 4: Produce a fresh immutable runtime attestation**

Create the detached source snapshot with `scripts/create_source_snapshot.sh`, then submit `scripts/validate_oci_container_runtime.sub` against the selected nightly container and exact NeMo-RL/Bridge/MCore/TE SHAs. The new artifact must pass the exact six-module 245-test stage suite from Task 1, report all expected GPUs, and bind the managed Python and uv binaries by SHA256. Update the selected private profile with only that successful job ID and its immutable attestation path.

Expected RED before this step: the prior `5852837` artifact is a failure and cannot authorize any GPU row. Expected GREEN after this step: `verify_runtime_attestation.py` accepts the new artifact against the final source and container digests.

- [ ] **Step 5: Run scheduler preflight**

From the NeMo-RL root, use the selected concrete cluster profile:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)
SBATCH_TEST_ONLY=1 CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS='dropless_hybridep_nano16 dropless_alltoall_qwen30_16 dropless_alltoall_super32 dropless_hybridep_qwen235_64' \
  "$EXP/submit_mcore_matrix.sh"
```

Expected GREEN: four valid batch allocations, no backfill partition, no missing runtime field, and no raw command injection.

- [ ] **Step 6: Submit rows in parallel and monitor**

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS='dropless_hybridep_nano16 dropless_alltoall_qwen30_16 dropless_alltoall_super32 dropless_hybridep_qwen235_64' \
  "$EXP/submit_mcore_matrix.sh"
```

Monitor for five minutes with `squeue`, `sacct`, and the per-row SLURM logs. A job that bootstraps on only part of its allocation or leaves GPUs idle is cancelled and diagnosed before resubmission.

- [ ] **Step 7: Validate proof artifacts**

Each row must contain 20 changed-route replays, nonzero graph calls for requested scopes, exact eager parity, zero fallback, zero capacity events, all expected ranks, and exact provenance. Only then may the matching selector's partial preprocess evidence be promoted.

---

### Task 9: Run NeMo-RL Five-Step Functional and Coverage Smokes

**Files:**

- Modify only if the gate exposes a real bug: NeMo-RL/MCore production files owned by Tasks 2-6
- Produce: immutable smoke results and promotion manifests under the experiment result root
- Regenerate: `results/report.html`

**Interface:** Submit model-compatible rows in parallel after one test-only pass. Warmup is exactly 3; each row runs 5 optimizer steps after setup; checkpointing is disabled.

| Model | Required smoke scopes |
|---|---|
| Nano | baseline, `moe_router`, `moe_router+moe_preprocess`, `attn+mamba+moe_router+moe_preprocess` |
| Super | baseline, `moe_router`, `moe_router+moe_preprocess`, `attn+mamba+moe_router+moe_preprocess` |
| Qwen3-30B-A3B | baseline, `moe_router`, `moe_router+moe_preprocess`, `attn+moe_router+moe_preprocess` |
| Qwen3-235B | baseline, `moe_router`, `moe_router+moe_preprocess`, `attn+moe_router+moe_preprocess` |
| Ultra | Bridge/MCore standalone only until the external NeMo-RL profile gate is resolved |

Expected RED before Tasks 3-6: Nano and Qwen3-235B `moe_router+moe_preprocess` are rejected by the stale fixed-capacity validation or classifier, so no NeMo-RL smoke can be promoted.

- [ ] **Step 1: Render and inspect every command**

Run each persistent scope with `TEST_ONLY=1`. Assert production partial rows contain no capacity knobs, CUDA Graph rows contain warmup 3, and baseline contains `cuda_graph_impl=none`.

- [ ] **Step 2: Submit all independent rows together**

Use `submit_smoke_matrix.sh` with batch profiles and the validated partial-MoE artifacts. Do not wait for one model before submitting an independent model.

- [ ] **Step 3: Enforce functional gates**

Every graph row requires:

- no CUDA/NCCL error and no idle-rank reaper event;
- `fallback_count == 0`;
- `graph_calls > 0` and stage-specific coverage above zero;
- zero MoE capacity safety counters;
- finite loss/reward/KL/error/grad-norm values;
- exact requested scope discovery, including Mamba and MTP-present layers where applicable.

If a scope has zero eligible calls, classify it model-incompatible or dependency-blocked; never report it as a successful zero-coverage graph run.

Expected GREEN: every supported row completes five steps with nonzero requested-scope coverage, zero fallback, zero unsafe capacity counters, and a valid immutable promotion artifact.

- [ ] **Step 4: Commit only genuine fixes, then rerun the failed row**

Use a red test reproducing each failure, make the smallest ownership-layer fix, rerun focused tests, commit with `-s`, push dependency pins in order, regenerate runtime provenance, and resubmit only affected rows plus one unchanged control.

---

### Task 10: Calibrate and Test Experimental Whole-MoE Capacity

**Files:**

- Produce eager calibration results and validated capacity profiles
- Produce matched eager/graph smoke artifacts
- Update report context and generated HTML

**Interface:** Whole-MoE never borrows the production partial graph's dropless claim. The three arms are always:

1. original dropless eager recipe;
2. fixed-capacity eager with the selected profile;
3. fixed-capacity CUDA Graph with the identical profile.

Expected RED before calibration: `scope_matrix.py classify --scope moe` returns `capacity-blocked` because no validated model/topology profile exists.

- [ ] **Step 1: Submit 20-step eager factor sweeps in parallel**

Run the fixed grid for each model/topology. Separate expert-capacity and HybridEP rank-capacity profiles; do not set both in one row. A row that records any unsafe event is rejected.

- [ ] **Step 2: Select with margin and run a 100-step eager soak**

Use `select_moe_capacity.py`. The selected factor is the next tested grid point above the first safe 20-step factor. Run one 100-step eager soak at that value. Any unsafe event advances to the next grid value and requires a fresh 100-step soak.

- [ ] **Step 3: Publish immutable profiles**

Profiles contain exact model/topology/packed geometry, capacity mode/value, source SHAs, runtime/container digests, all calibration job IDs, step counts, and zero-event counters. Commit only reviewed profiles whose raw artifacts remain addressable.

Expected GREEN: `validate_moe_capacity_profile.py` accepts the reviewed profile and the matched eager/graph commands render the identical capacity value.

- [ ] **Step 4: Run five-step whole-MoE and whole-layer smokes**

First run matched fixed-capacity eager, then CUDA Graph using the same frozen inputs/profile. Require nonzero graph coverage of `moe`, exact output/gradient/parameter-delta parity, and zero unsafe events. If TE cannot capture a routing-dependent expert kernel even at fixed geometry, keep the row capacity-ready but capability-blocked and report the exact unsupported operation.

- [ ] **Step 5: Decide whether whole-MoE merits optimization work**

Profile the production dropless partial scope first. Continue whole-MoE engineering only if the eager dispatch/expert/combine tail remains a material policy-training bottleneck and the fixed-capacity semantics/memory cost is acceptable. Otherwise retain partial MoE as the production implementation.

---

### Task 11: Run Paired 20-Step Performance and 100-Step Accuracy Gates

**Files:**

- Produce performance and accuracy manifests/results
- Modify collector/report code only for observed schema bugs
- Regenerate and publish `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html`

**Interface:** Compare identical source/runtime/model/topology/data/seed pairs. Use three repeats for 20-step performance and one deterministic 100-step soak for the best correct production partial scope. Report CUDA Graph coverage separately for policy training and logprob; a stage without an implemented graph lifecycle is dependency-blocked, not counted as accelerated.

Expected RED before promotion: the performance and accuracy submitters reject a row without a content-bound passed five-step promotion manifest.

- [ ] **Step 1: Promote only passed five-step rows**

Generate a content-bound promotion manifest. Reject missing graph calls, fallback, unsafe capacity counters, provenance mismatch, NaN/Inf, or fixed-batch parity failure.

- [ ] **Step 2: Submit matched 20-step rows**

For each promoted model, submit the baseline and candidate repeats together. Record step time and tokens/sec/GPU for:

- E2E;
- generation;
- policy training;
- combined policy/reference logprob.

Also record capture/replay/cache/fallback counts, graph/eligible calls, THD capacity utilization, and MoE safety counters.

- [ ] **Step 3: Compare correctness and semantics**

For production dropless partial rows, compare eager/graph valid-token outputs, routes, probabilities, expert counts, loss, gradients, parameter deltas, reward, `gen_kl_error`, token/multi-logprob error, policy KL, JS divergence, sampling importance ratio, masked-sequence count, and grad norm.

For fixed-capacity rows, perform the same graph-versus-matched-eager comparison and separately show the fixed-eager-versus-original-dropless delta.

- [ ] **Step 4: Run the 100-step stability gate**

Run the best correct production partial scope and its baseline for 100 steps. Include Nano combined attention/Mamba/MoE and the best model-compatible scopes for Super and Qwen. Abort on NaN/Inf, nonzero capacity events, fallback, late gradient divergence, or missing graph coverage.

- [ ] **Step 5: Publish the final table and HTML**

The final report table includes model, dispatcher, scope, status/reason, graph calls, eligible calls, coverage, warmup, E2E/generation/policy/logprob step time and throughput, capacity utilization, all MoE safety counters, correctness metrics, repeat median/p95/variance, job IDs, and full provenance.

- [ ] **Step 6: Run final local verification and commit report artifacts**

```bash
uv run pytest -q tests/unit/experiments
git diff --check
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/results \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/report_context.json
git commit -s -m "docs: publish partial MoE CUDA graph results"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

Expected GREEN: the report contains no unsupported success claims, every numerical comparison links to immutable raw evidence, and all production conclusions are based on zero-fallback, zero-drop runs.

---

## Completion Gate

This plan is complete only when all of the following are true:

- Every distributed manifest node exists and is collected from the exact candidate archive.
- Dropless HybridEP and AlltoAll partial MoE pass changed-route forward/backward/parameter-delta parity with nonzero graph calls.
- NeMo-RL accepts dropless `moe_router+moe_preprocess` without capacity knobs only on the supported backends.
- Whole-MoE remains blocked without a validated fixed-capacity profile.
- Capacity fields are optional, typed, presence-forwarded, and applied before model post-init.
- Any dropped assignment, fully dropped valid token, or rank overflow prevents `optimizer.step()` on every rank.
- Nano and Super pass all model-compatible partial scopes; Qwen3-30B-A3B and Qwen3-235B pass their targeted partial scopes; Ultra is either proven with a real profile or explicitly externally blocked.
- The selected production partial scopes complete paired 20-step performance and 100-step stability runs with warmup 3, zero fallback, zero unsafe capacity events, and explicit stage coverage.
- Fixed-capacity results, if any, are labeled experimental and compared both to matched fixed-capacity eager and original dropless eager.
- MCore, Bridge, and NeMo-RL SHAs are pushed and pinned in dependency order, and the HTML report links every claim to immutable job/provenance artifacts.
