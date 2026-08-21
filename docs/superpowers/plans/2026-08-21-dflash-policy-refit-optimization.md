# DFlash Policy and Refit Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce online DFlash hidden-capture memory traffic and TP draft-export collective launches while preserving exact training/refit behavior.

**Architecture:** Hidden hooks retain detached source tensors until one validated materialization creates a shared backing buffer. DFlash refit export builds a deterministic per-dtype/device flat TP bucket, gathers once, and reconstructs logical tensors using the existing split-axis contract.

**Tech Stack:** Python 3.12, PyTorch, torch.distributed, Megatron Core, pytest, Ruff, Pyrefly, SLURM/OCI-HSG.

**Spec:** `docs/superpowers/specs/2026-08-21-dflash-policy-refit-optimization.md`

## Global Constraints

- Base exact `f909e3d124bb663db4099e88f6846e55b0500912`.
- Preserve one draft optimizer update and one draft refit per optimizer step.
- Preserve parameter names/order, logical tensor values, TP rank order, loss/gradient behavior, and generation configuration.
- Use strict RED then GREEN for every production change.
- Do not mutate PR9-11 branches or active 1000-step jobs.

---

### Task 1: Single-allocation hidden capture

**Files:**
- Modify: `nemo_rl/models/megatron/draft/hidden_capture.py`
- Create: `tests/unit/models/megatron/test_hidden_capture.py`

**Interfaces:**
- Consumes: embedding and auxiliary-layer hook outputs shaped `[sequence, micro_batch, hidden]`.
- Produces: `CapturedStates(hidden_states, inputs_embeds)` whose tensors are non-overlapping views into one contiguous backing tensor for PP1.

- [ ] **Step 1: Write the failing value and allocation tests**

Create a minimal fake Megatron model and patch the parallel-state accessors to PP1. Capture one embedding and three auxiliary tensors for MBS1 and MBS2. Assert exact output values and:

```python
assert captured.inputs_embeds.untyped_storage().data_ptr() == captured.hidden_states.untyped_storage().data_ptr()
assert clone_calls == 0
assert cat_calls == 1
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/unit/models/megatron/test_hidden_capture.py
```

Expected: the existing hook clones make `clone_calls == 0` fail and embedding/hidden tensors do not share storage.

- [ ] **Step 3: Write the failing mutation test**

Capture sources, mutate one source in place, then call `get_captured_states()` and assert:

```python
with pytest.raises(RuntimeError, match="modified in place"):
    capture.get_captured_states()
```

- [ ] **Step 4: Implement validated retained captures**

Add a private frozen record:

```python
@dataclass(frozen=True)
class _CapturedTensorRef:
    tensor: Tensor
    version: int

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "_CapturedTensorRef":
        detached = tensor.detach()
        return cls(detached, detached._version)

    def validated(self, name: str) -> Tensor:
        if self.tensor._version != self.version:
            raise RuntimeError(f"captured tensor '{name}' was modified in place")
        return self.tensor
```

Store `_CapturedTensorRef` in hooks. In PP1, validate embeddings and sorted auxiliary tensors, call `torch.cat([embeds, *hidden_chunks], dim=-1)` once, and return `[..., :hidden]` for embeddings plus the remaining slice for hidden states. Cache the assembled `CapturedStates` until hooks are re-registered. Validate before PP sends and preserve existing PP order.

- [ ] **Step 5: Verify GREEN and regressions**

Run:

```bash
pytest -q tests/unit/models/megatron/test_hidden_capture.py tests/unit/models/megatron/test_dflash_training_provider.py
ruff check nemo_rl/models/megatron/draft/hidden_capture.py tests/unit/models/megatron/test_hidden_capture.py
ruff format --check nemo_rl/models/megatron/draft/hidden_capture.py tests/unit/models/megatron/test_hidden_capture.py
```

Expected: all tests pass; Ruff exits 0.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models/megatron/draft/hidden_capture.py tests/unit/models/megatron/test_hidden_capture.py
git commit -s -m "perf(draft): fuse hidden capture materialization"
```

### Task 2: Bucketed DFlash TP export

**Files:**
- Modify: `nemo_rl/models/megatron/draft/utils.py`
- Modify: `tests/unit/models/megatron/test_dflash_export_contract.py`

**Interfaces:**
- Consumes: ordered `state_dict()` entries plus `_dflash_weight_layout(name, config) -> (logical_shape, split_axis)`.
- Produces: the same ordered `(name, logical_tensor)` list as `export_dflash_weights_to_hf` with at most one TP `all_gather` per `(device, dtype)` bucket.

- [ ] **Step 1: Write a failing TP2 bucket test**

Use two Gloo ranks with multiple sharded tensors on axes 0 and 1 plus unsharded norms. Compare export tensors against a tensor-by-tensor reference and count `dist.all_gather` calls:

```python
assert exported_names == reference_names
for actual, expected in zip(exported_tensors, reference_tensors):
    torch.testing.assert_close(actual, expected)
assert gather_calls == 1
```

- [ ] **Step 2: Verify RED**

Run the focused distributed test. Expected: values match but the current exporter invokes one collective per sharded parameter, so `gather_calls == 1` fails.

- [ ] **Step 3: Implement deterministic flat buckets**

Add a private export entry record containing name, tensor, logical shape, split axis, and flat offset. Group only tensors that require TP reconstruction by `(tensor.device, tensor.dtype)`. Concatenate contiguous flattened local shards, gather the flat bucket in rank order, then reconstruct each tensor with:

```python
rank_shards = [
    gathered[rank][offset : offset + local_numel].view(local_shape)
    for rank in range(tp_world_size)
]
logical = torch.cat(rank_shards, dim=split_axis).contiguous()
```

Validate identical entry count, local shapes, dtype/device, offsets, and final logical shape. Preserve TP1/already-logical identity and output ordering.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
pytest -q tests/unit/models/megatron/test_dflash_export_contract.py
ruff check nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py
ruff format --check nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py
```

Expected: TP1 and TP2 tests pass; TP2 uses one collective for BF16 tensors.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py
git commit -s -m "perf(draft): bucket tensor-parallel refit export"
```

### Task 3: Exact verification and matched GPU matrix

**Files:**
- Create: an experiment-only harness branch derived from the verified product head.
- Create: durable result metadata under the existing OCI experiment root.

**Interfaces:**
- Consumes: exact signed product head from Tasks 1-2.
- Produces: matched fixed/online results for GBS32/MBS1, GBS64/MBS1, and GBS64/MBS2.

- [ ] **Step 1: Run the full local/static gate**

```bash
pytest -q tests/unit/models/megatron/test_hidden_capture.py tests/unit/models/megatron/test_dflash_export_contract.py tests/unit/models/megatron/test_dflash_training_provider.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/algorithms/test_loss_wrappers.py
ruff check nemo_rl/models/megatron/draft tests/unit/models/megatron/test_hidden_capture.py tests/unit/models/megatron/test_dflash_export_contract.py
ruff format --check nemo_rl/models/megatron/draft tests/unit/models/megatron/test_hidden_capture.py tests/unit/models/megatron/test_dflash_export_contract.py
git diff --check
```

- [ ] **Step 2: Push exact product head and prepare OCI source**

Create signed+DCO commits only, normal-push the isolated branch, fast-forward a clean recursive OCI checkout, and verify local/remote/OCI SHA equality plus immutable container SHA.

- [ ] **Step 3: Submit one exact-head correctness gate**

Compare FairShare, run `sbatch --test-only`, submit one actual 4-GPU TP2 gate, and monitor at 60-second cadence for at least five minutes. Require terminal zero exit and exact-head result artifact before performance claims.

- [ ] **Step 4: Build the matched matrix harness**

Keep four generations per prompt. Use 8 prompts for GBS32 and 16 prompts for GBS64. Keep seed, prompt order, model revisions, topology, sequence packing, CUDA Graph settings, and optimizer configuration identical. Parameterize only fixed/online, GBS, and MBS.

- [ ] **Step 5: Submit and analyze paired runs**

For every harness head: normal push, clean OCI checkout, parity proof, FairShare, `sbatch --test-only`, actual submission, and five-minute monitoring. Exclude steps 0-4 and report summed time divided by summed samples/tokens, policy/refit/E2E time, generation TPS, acceptance rate, peak memory, draft loss, and update/refit markers.

- [ ] **Step 6: Decide adoption**

Adopt only if loss/gradient/update/refit correctness is unchanged, no OOM occurs, and matched time-per-token improves. State node placement and confidence intervals; do not attribute unmatched-node variation to the patch.
