# Graph-Safe Variable Packed Sequence Auxiliary Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make variable-length packed THD `seq_aux_loss` exactly match padded
eager execution in eager and Transformer Engine partial CUDA Graph paths.

**Architecture:** NeMo-RL materializes fixed-shape CP/SP-local sample ownership
before model execution. MCore keeps that ownership in a dedicated MoE
`PackedSeqParams` namespace, accumulates loss into static `[S_cap, E]` buffers,
and binds the two dynamic Tensors plus static capacity to each graphable MoE
leaf and schedule-owned graph bank. Attention and Mamba keep independent graph
input surfaces.

**Tech Stack:** Python 3.12, PyTorch distributed, Megatron-LM/MCore,
Transformer Engine, NeMo-RL, pytest, Ruff, isort, CUDA Graphs, and NCCL.

## Global Constraints

- Work only in
  `/Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731`.
- NeMo-RL branch is `experiment/thd-cg-hybrid-nemotron-20260731`.
- Megatron-LM branch is `sj/thd-cg-hybrid-nemotron-20260731`.
- PR 6115 commit `4d145f34e1f236dd548499ef22408f0892f22bea`
  is a math/test reference only; do not cherry-pick it.
- Add `seq_aux_loss_sample_ids` as contiguous CUDA `torch.int64` with shape
  `[T_router_capacity]`.
- Add `seq_aux_loss_num_samples` as contiguous CUDA `torch.int64` with scalar
  shape `[]` and exact real count `N`.
- Add `seq_aux_loss_max_samples` as static Python `int` equal to
  `thd_max_packed_sequences - 1`.
- Construct IDs by per-sequence CP zigzag, concatenate, then exactly one
  contiguous TP-rank slice when sequence parallelism is enabled.
- Real and inter-sequence padding slots use their logical sample ID. Appended
  dummy and unused tail slots use ID zero and are structurally masked.
- Use an independent `_moe_packed_seq_params_` graph namespace. Do not add MoE
  ownership to generic attention or Mamba field tuples.
- Use static `[S_cap, E]` router buffers. Runtime count/ID values may change;
  shape, dtype, device, layout, stride, presence, and `S_cap` may not.
- Preserve Task 4 structural-padding, Task 5 dispatcher replay, Task 6 graph
  bank, and Task 7 ordered Hybrid/MTP behavior.
- No `.item()`, `.tolist()`, Python Tensor predicate, or data-dependent
  allocation is permitted inside CUDA capture/replay.
- Missing metadata and signature changes fail before graph launch. Rank-local
  eager fallback is forbidden.
- Reject `moe_router_fusion=True` for variable packed `seq_aux_loss` in this
  implementation. A later enablement requires a separate pinned-TE capability
  test; there is no silent kernel fallback.
- Warmup is exactly three globally successful optimizer updates.
- Local macOS validation uses focused source/AST/isolated tests. Real
  Linux/CUDA/TE and multi-rank NCCL validation remains mandatory before any
  performance submission.
- The frozen MCore/NeMo-RL locks are Linux-only on this macOS host and the
  local runtime has no PyTorch or TE. Run each canonical focused command once
  and record any platform-resolution error verbatim. Then obtain mandatory
  RED/GREEN evidence with a dependency-isolated harness in this plan's ignored
  SDD workspace that executes the changed production functions against literal
  fixtures. A source-text assertion or static check does not count as RED or
  GREEN, and the lock files remain unchanged.
- Commit only owned files. NeMo-RL commits use `git commit -s`; Megatron-LM
  commits use `git commit -s -S`. Do not push to NVIDIA remotes.

---

## File and Responsibility Map

### Megatron-LM packed contract

- `megatron/core/packed_seq_params.py`: define ownership fields, preserve them
  through THD padding, and split/merge the dedicated MoE graph namespace.
- `megatron/core/extensions/transformer_engine.py`: prevent MoE-only fields
  from reaching TE attention.
- `tests/unit_tests/test_sequence_packing.py`: pure CPU ownership split/merge
  and THD-padding preservation tests.
- `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`: verify
  TE-layer namespace isolation and exact graph-facing types.

### NeMo-RL producer

- `nemo_rl/models/megatron/data.py`: build CP-then-SP-local ownership and
  validate it before the microbatch is yielded.
- `tests/unit/models/megatron/test_megatron_data.py`: verify exact CP1, CP2,
  SP, dummy-tail, propagation, and fail-closed behavior.

### Megatron-LM router

- `megatron/core/transformer/moe/router.py`: implement fixed-capacity
  segmented sequence auxiliary loss.
- `megatron/core/transformer/moe/moe_layer.py`: pass ownership through route,
  forward, recompute, and partial callables.
- `megatron/core/transformer/moe/moe_utils.py`: permit an `int | Tensor`
  denominator in the unfused switch-loss type contract.
- `tests/unit_tests/transformer/moe/test_aux_loss.py`: padded-versus-packed
  forward/loss/gradient oracle.

### Megatron-LM graph integration

- `megatron/core/transformer/transformer_layer.py`: propagate ownership to
  eager MoE and flatten/rebuild it only for graphable MoE scopes.
- `megatron/core/transformer/cuda_graphs.py`: add ownership to canonical inner
  MoE-leaf samples, including router-only capture.
- `megatron/core/transformer/te_cuda_graph_bank.py`: fingerprint, validate,
  activate, roll back, and reset the MoE contract.
- `tests/unit_tests/transformer/test_cuda_graphs.py`: verify scope-aware
  sample/replay behavior and exact counters.
- `tests/unit_tests/transformer/test_te_cuda_graph_bank.py`: verify fixed
  signatures, dynamic values, rollback, and reset.

### Integration gates

- `tests/unit_tests/models/test_hybrid_moe_model.py`: Hybrid/MTP ownership.
- `tests/unit_tests/ssm/test_mamba_layer.py`: Mamba namespace isolation.
- `tests/unit_tests/transformer/test_submodule_callables.py`: canonical leaf
  identity/order and fine-grained MoE bounds.
- `tests/unit/algorithms/sequence_packing_gradient_actor.py`: NeMo-RL packed
  policy-gradient ownership.

---

### Task 1: Add the MCore Fixed-Shape Ownership Contract

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/packed_seq_params.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/test_sequence_packing.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`

**Interfaces:**
- Consumes: existing generic and Mamba `PackedSeqParams` split/rebuild helpers.
- Produces: the three exact fields,
  `split_moe_packed_seq_params_for_cuda_graph()`, and
  `merge_moe_packed_seq_params_from_cuda_graph_kwargs()`.

The merge function has this exact public-internal signature:

```python
def merge_moe_packed_seq_params_from_cuda_graph_kwargs(
    packed_seq_params: PackedSeqParams | None,
    kwargs: MutableMapping[str, object],
    static_metadata: Mapping[str, object] | None,
    prefix: str = MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    remove_from_kwargs: bool = True,
) -> PackedSeqParams | None
```

- [ ] **Step 1: Write failing independent-namespace tests**

Add the pure CPU contract test to `test_sequence_packing.py` using a real
`PackedSeqParams` fixture whose generic, Mamba, and MoE fields are all
populated. The ownership assertions must use literal expected keys and must
prove the merge returns a different object:

```python
source = PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
    cu_seqlens_kv=torch.tensor([0, 3, 8], dtype=torch.int32),
    total_tokens=12,
    seq_idx=torch.arange(12, dtype=torch.int32).unsqueeze(0),
    seq_aux_loss_sample_ids=torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int64
    ),
    seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
    seq_aux_loss_max_samples=3,
)
tensor_kwargs, static = split_moe_packed_seq_params_for_cuda_graph(source)
assert set(tensor_kwargs) == {
    "_moe_packed_seq_params_seq_aux_loss_sample_ids",
    "_moe_packed_seq_params_seq_aux_loss_num_samples",
}
assert static == {"seq_aux_loss_max_samples": 3}

rebuilt = merge_moe_packed_seq_params_from_cuda_graph_kwargs(
    source, dict(tensor_kwargs), static
)
assert rebuilt is not source
assert rebuilt.seq_aux_loss_sample_ids is source.seq_aux_loss_sample_ids
assert rebuilt.seq_aux_loss_num_samples is source.seq_aux_loss_num_samples
assert rebuilt.seq_aux_loss_max_samples == 3
```

Also assert generic split keys contain no `_moe_packed_seq_params_` prefix,
Mamba split keys contain no MoE prefix, and changing `rebuilt` cannot change
the three fields on `source`. Repeat merge with `remove_from_kwargs=False` and
assert the input kwargs mapping is unchanged.

- [ ] **Step 2: Run the tests and record the expected red failure**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
uv run python -m pytest \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "moe_packed or seq_aux_loss" -q
```

Expected: collection or assertion failure because the fields and MoE helpers
do not exist.

- [ ] **Step 3: Implement the minimal fixed-shape contract**

Add exactly:

```python
MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_moe_packed_seq_params_"
MOE_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = (
    "seq_aux_loss_sample_ids",
    "seq_aux_loss_num_samples",
)
MOE_PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS = (
    "seq_aux_loss_max_samples",
)
```

The merge helper takes an existing `PackedSeqParams | None`, creates a new
instance or shallow copy, consumes only MoE-prefixed kwargs, validates Tensor
versus static types, and leaves the input object unchanged. Reuse the existing
key/type-validation helpers rather than duplicating the generic loop.

Extend `pad_sequence_for_thd()` so its single `padded_params` reconstruction
preserves the three exact values from the input object. Do not derive count
from padded cumulative endpoints. In `TEDotProductAttention.__init__`, add all
three calls to `self.kept_packed_seq_params.discard` beside the existing
`tokens_per_sample` exclusion.

- [ ] **Step 4: Add padding-preservation and TE-filter behavior tests**

In `test_sequence_packing.py`, use `pad_sequence_for_thd()` with two real
samples, token capacity 16, and sequence capacity four. Assert the returned
object retains the same ID/count Tensor identities and static value. In the
transformer test, exercise the existing TE attention PackedSeqParams filtering
boundary and assert none of the three MoE names is present in the forwarded
object.

- [ ] **Step 5: Run focused tests and static checks**

```bash
uv run python -m pytest \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "packed or seq_aux_loss" -q
uv run isort megatron/core/packed_seq_params.py \
  megatron/core/extensions/transformer_engine.py \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
uv run ruff check megatron/core/packed_seq_params.py \
  megatron/core/extensions/transformer_engine.py \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git diff --check
```

- [ ] **Step 6: Commit only Task 1 files**

```bash
git add megatron/core/packed_seq_params.py \
  megatron/core/extensions/transformer_engine.py \
  tests/unit_tests/test_sequence_packing.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -S -m "feat: add packed MoE sample ownership contract"
git log -1 --show-signature --format=fuller
```

---

### Task 2: Materialize Router-Local Ownership in NeMo-RL

**Files:**
- Modify: `nemo_rl/models/megatron/data.py`
- Test: `tests/unit/models/megatron/test_megatron_data.py`

**Interfaces:**
- Consumes: Task 1's three `PackedSeqParams` fields.
- Produces: validated IDs in final router order, exact scalar `N`, and static
  `S_cap` on every fixed graph-training microbatch.

- [ ] **Step 1: Write the failing CP1 producer test**

Extend the existing fixed packer test for logical lengths `[3, 2]`, physical
lengths `[4, 4]`, token capacity 16, and sequence capacity four:

```python
assert torch.equal(
    model_params.seq_aux_loss_sample_ids,
    torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int64,
    ),
)
assert model_params.seq_aux_loss_num_samples.shape == torch.Size([])
assert model_params.seq_aux_loss_num_samples.dtype == torch.int64
assert int(model_params.seq_aux_loss_num_samples) == 2
assert model_params.seq_aux_loss_max_samples == 3
```

Pack one real sample with the same capacities and assert the ID shape remains
16 while the scalar changes to one. Repeated capacity endpoints must not alter
the scalar.

- [ ] **Step 2: Write failing CP2 and CP2+SP tests**

For two physical eight-token samples and a 32-token capacity, compute each
expected sample segment with the real `_get_tokens_on_this_cp_rank()` helper.
For each CP rank, assert the CP-local IDs have shape 16 and equal:

```python
expected_cp = torch.cat(
    [
        _get_tokens_on_this_cp_rank(
            torch.full((8,), sample_id, dtype=torch.int64),
            cp_rank,
            2,
            seq_dim=0,
        )
        for sample_id in (0, 1)
    ]
    + [torch.zeros(8, dtype=torch.int64)]
)
```

With TP2/SP enabled, TP rank zero receives `expected_cp[:8]` and TP rank one
receives `expected_cp[8:]`; each result is contiguous with shape eight.

- [ ] **Step 3: Run and record the red failures**

```bash
uv run python -m pytest tests/unit/models/megatron/test_megatron_data.py \
  -k "seq_aux_loss_sample_ids or fixed_cp2 or fixed_packing" -q
```

Expected: failures because fixed packs do not populate ownership and packer
entry points do not accept TP/SP geometry.

- [ ] **Step 4: Implement the producer helper and propagation**

Add this exact helper surface:

```python
def _build_packed_seq_aux_loss_sample_ids(
    cu_seqlens_padded: torch.Tensor,
    *,
    capacity_tokens: int,
    real_sequence_count: int,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
    sequence_parallel: bool,
) -> torch.Tensor:
    local_parts: list[torch.Tensor] = []
    for sample_id in range(real_sequence_count):
        physical_len = int(
            cu_seqlens_padded[sample_id + 1] - cu_seqlens_padded[sample_id]
        )
        ids = torch.full(
            (physical_len,),
            sample_id,
            dtype=torch.int64,
            device=cu_seqlens_padded.device,
        )
        local_parts.append(
            _get_tokens_on_this_cp_rank(ids, cp_rank, cp_size, seq_dim=0)
            if cp_size > 1
            else ids
        )

    dummy_tokens = capacity_tokens - int(cu_seqlens_padded[-1])
    dummy = torch.zeros(
        (dummy_tokens,), dtype=torch.int64, device=cu_seqlens_padded.device
    )
    if cp_size > 1 and dummy_tokens > 0:
        dummy = _get_tokens_on_this_cp_rank(
            dummy, cp_rank, cp_size, seq_dim=0
        )
    local = torch.cat((*local_parts, dummy), dim=0)

    if sequence_parallel:
        if local.numel() % tp_size != 0:
            raise ValueError(
                "CP-local sample IDs must divide evenly across TP/SP ranks."
            )
        width = local.numel() // tp_size
        local = local.narrow(0, tp_rank * width, width)
    return local.contiguous()
```

For each real sample, fill its physical segment with its ID and apply the
existing CP selector before concatenation. Build the append-dummy tail as
zeros, CP-shard it once, concatenate, and apply one contiguous TP slice only
for SP. Return contiguous `int64`.

Thread `tp_rank`, `tp_size`, and `sequence_parallel` from
`make_processed_microbatch_iterator()` through `process_microbatch()` into the
fixed packer. Read `get_tensor_model_parallel_rank()` and
`get_tensor_model_parallel_world_size()` beside the existing CP accessors,
require the initialized TP world size to equal
`cfg["megatron_cfg"]["tensor_model_parallel_size"]`, and reject a rank outside
that size. Direct non-SP callers retain explicit defaults `(0, 1, False)`.

After `pad_sequence_for_thd()` returns and `seq_idx` is finalized, assign:

```python
packed_seq_params.seq_aux_loss_sample_ids = sample_ids
packed_seq_params.seq_aux_loss_num_samples = torch.tensor(
    real_sequence_count, dtype=torch.int64, device=input_ids.device
)
packed_seq_params.seq_aux_loss_max_samples = thd_max_packed_sequences - 1
```

- [ ] **Step 5: Extend pre-yield validation with real behavior cases**

Compute expected router capacity as global capacity divided by CP and, for SP,
TP. Validate field presence, exact dtype/shape/device/contiguity, scalar count,
static capacity, and equality between Tensor count and
`PackedGeometry.real_sequence_count`. Add corruption rows for missing IDs,
wrong length, wrong dtype, non-scalar count, and wrong static capacity. Each
must raise before `next(iterator)` returns. Add an exact zero-count corruption
and assert it fails the existing `1 <= N` producer bound.

Rebuild the exact expected IDs from the producer's real physical endpoints,
CP rank/size, TP rank/size, SP flag, and capacity, then compare the yielded
Tensor before model entry. Add negative ID, `id == N`, and a valid-but-nonzero
ID in an appended dummy/tail slot as corruption rows. This enforces both ID
bounds and dummy ownership; literal CP1/CP2/SP tests independently verify the
builder's ordering.

- [ ] **Step 6: Run focused and regression tests**

```bash
uv run python -m pytest tests/unit/models/megatron/test_megatron_data.py \
  -k "seq_aux_loss or fixed_thd or fixed_cp2 or fixed_process" -q
uv run ruff check nemo_rl/models/megatron/data.py \
  tests/unit/models/megatron/test_megatron_data.py
uv run ruff format --check nemo_rl/models/megatron/data.py \
  tests/unit/models/megatron/test_megatron_data.py
git diff --check
```

- [ ] **Step 7: Commit only Task 2 files**

```bash
git add nemo_rl/models/megatron/data.py \
  tests/unit/models/megatron/test_megatron_data.py
git commit -s -m "fix: materialize packed MoE sample ownership"
```

---

### Task 3: Implement Fixed-Capacity Router Parity

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/router.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_aux_loss.py`

**Interfaces:**
- Consumes: Task 1's ownership fields and Task 2's exact producer values.
- Produces: one fixed-capacity sequence-auxiliary-loss path used identically by
  packed eager and graph execution.

Use these exact call surfaces:

```python
def MoELayer.forward(
    self,
    hidden_states: torch.Tensor,
    intermediate_tensors=None,
    padding_mask: torch.Tensor | None = None,
    packed_seq_params: PackedSeqParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]

def MoELayer.route(
    self,
    hidden_states: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    *,
    seq_aux_loss_sample_ids: torch.Tensor | None = None,
    seq_aux_loss_num_samples: torch.Tensor | None = None,
    seq_aux_loss_max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

`TopKRouter.forward()` and `TopKRouter.routing()` accept the same three
keyword-only ownership values after `padding_mask`.

- [ ] **Step 1: Add a failing padded-versus-packed eager router oracle**

Port PR 6115's padded-versus-variable-packed test without its dynamic metadata
derivation. Initialize one router, clone identical logits, and compare:

```text
padded: [L_max, N, E] with explicit padding mask
packed: [T_capacity, 1, E] with IDs/count/S_cap
```

Cover literal geometries `[3, 5]`, `[1, 7, 2]`, one sample, and `N == S_cap`;
equal physical lengths `[8, 8]`; unequal physical lengths `[4, 8]`; exact token
occupancy; extra dummy tail; internal alignment padding; and at least two
unused sample rows. Parametrize top-k one/two and router score function
`softmax`, `sigmoid`, and `sqrtsoftplus`. Assert equal valid-token
probabilities, exact top-k and routing map, `seq_load_balancing_loss`, input
gradient, and router parameter gradient. Assert every padding-row gradient is
zero.

- [ ] **Step 2: Add failing API and validation tests**

Exercise the actual `MoELayer.route()` and recompute closure. Variable packed
`seq_aux_loss` must raise for a missing field, `S_cap < 1`, wrong ID shape, or
non-scalar count. Add a test that `moe_router_fusion=True` always raises a
clear unsupported-configuration error for this variable packed path and never
silently calls the unfused function.

Direct MCore callers must also cover dynamic `N == 0` and `N > S_cap`.
Assert that both hit a device-side bound check without `.item()`, `.tolist()`,
or a Python Tensor predicate. Keep the destructive real-CUDA assertion replay
in the isolated Task 12 subprocess; this unit row verifies the validation
surface and safe-denominator construction.

Add the scheduler-only all-padding case with `N == 1`, IDs all zero, and mask
all true. Through the real router path, assert exact zero auxiliary loss,
valid-token count, input gradient, and router gradient, with no NaN/Inf.

- [ ] **Step 3: Run and record red output**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
uv run python -m pytest \
  tests/unit_tests/transformer/moe/test_aux_loss.py \
  -k "variable_length_packed_seq_aux_loss or fixed_capacity_seq_aux" -q
```

Expected: failures because router/MoELayer do not accept explicit ownership and
still reshape the flattened pack as one sample.

- [ ] **Step 4: Implement static segmented accumulation**

Extend the router/MoELayer call surfaces with the two Tensor values and static
capacity. Allocate `local_counts` and `local_scores` with shape
`[seq_aux_loss_max_samples, num_experts]`, expand IDs to expert width, and use
`scatter_add_`. Reduce counts over the existing `tp_cp_group`.

Use exactly:

```python
num_samples_is_valid = (
    (seq_aux_loss_num_samples >= 1)
    & (seq_aux_loss_num_samples <= seq_aux_loss_max_samples)
)
torch._assert_async(
    num_samples_is_valid,
    "seq_aux_loss_num_samples must be in [1, seq_aux_loss_max_samples]",
)
num_samples_float = seq_aux_loss_num_samples.to(dtype=local_scores.dtype)
safe_num_samples_float = torch.where(
    num_samples_is_valid,
    num_samples_float,
    torch.ones_like(num_samples_float),
)
total_routes = global_counts.sum().to(dtype=local_scores.dtype)
has_valid_routes = total_routes > 0
total_num_tokens = total_routes / (self.topk * safe_num_samples_float)
safe_total_num_tokens = torch.where(
    has_valid_routes,
    total_num_tokens,
    torch.ones_like(total_num_tokens),
)
aux_loss = switch_load_balancing_loss_func(
    probs=local_scores.reshape(1, seq_aux_loss_max_samples * num_experts),
    tokens_per_expert=global_counts.reshape(-1),
    total_num_tokens=safe_total_num_tokens,
    topk=self.topk,
    num_experts=self.config.num_moe_experts,
    moe_aux_loss_coeff=seq_aux_loss_coeff,
    fused=False,
) / safe_num_samples_float
aux_loss = torch.where(
    num_samples_is_valid & has_valid_routes,
    aux_loss,
    torch.zeros_like(aux_loss),
)
valid_token_count = local_counts.sum() / self.topk
```

`torch._assert_async` is the only allowed MCore dynamic `N` bound check. Its
capture/replay support is an explicit pinned-runtime Task 12 gate; lack of
support disables this graph scope instead of permitting host synchronization
or silent fallback.

Widen the `total_num_tokens` parameter of
`switch_load_balancing_loss_func` and its docstring from `int` to
`int | torch.Tensor`. Do not change its formula or fused dispatch. The variable
packed caller has already rejected router fusion and passes the safe Tensor
denominator with `fused=False`.

Preserve current padding masking, HybridEP zeroing, token-dropping order,
expert-bias accounting, and `tp_cp_group`. Keep fixed-width
`tokens_per_sample` behavior separate.

- [ ] **Step 5: Forward metadata through every MoELayer route path**

Extract explicit metadata from `packed_seq_params` once in `MoELayer.forward`
and close over those values in `custom_forward` and recompute. Pass the values
through `route` and router invocation. Expert compute and combine do not
consume them.

- [ ] **Step 6: Run parity and static checks**

```bash
uv run python -m pytest \
  tests/unit_tests/transformer/moe/test_aux_loss.py \
  -k "seq_aux_loss or padding_mask" -q
uv run isort megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/moe_layer.py \
  megatron/core/transformer/moe/moe_utils.py \
  tests/unit_tests/transformer/moe/test_aux_loss.py
uv run ruff check megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/moe_layer.py \
  megatron/core/transformer/moe/moe_utils.py \
  tests/unit_tests/transformer/moe/test_aux_loss.py
git diff --check
```

- [ ] **Step 7: Commit only Task 3 files**

```bash
git add megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/moe_layer.py \
  megatron/core/transformer/moe/moe_utils.py \
  tests/unit_tests/transformer/moe/test_aux_loss.py
git commit -s -S -m "fix: preserve packed sequence auxiliary loss parity"
git log -1 --show-signature --format=fuller
```

---

### Task 4: Bind Ownership to TE Graph Leaves and Banks

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/te_cuda_graph_bank.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Test: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`

**Interfaces:**
- Consumes: Task 1 split/merge helpers and Task 3 MoELayer/router API.
- Produces: scope-aware graph samples, replay reconstruction, and exact
  schedule-bank fingerprints for MoE ownership.

- [ ] **Step 1: Write failing scope-surface tests**

Use actual Task 7 leaf descriptors and assert:

- router-only and router+preprocess samples contain
  `_moe_packed_seq_params_seq_aux_loss_sample_ids` and
  `_moe_packed_seq_params_seq_aux_loss_num_samples` without requiring
  self-attention;
- whole-MoE and whole-layer samples contain them;
- attention-only samples omit them while the eager MoE tail receives the
  original populated `PackedSeqParams`; and
- Mamba samples contain `seq_idx` but no MoE-prefixed key.

- [ ] **Step 2: Write failing bank-signature tests**

Capture one bank fingerprint, then replace ID/count Tensor values with new
Tensors of the same signatures. Validation must pass. Parametrize shape,
dtype, device when available, non-contiguous stride, field presence, and
`S_cap`; each mismatch must fail before a graph callable is invoked. Verify a
failed activation restores the previously active padding, dispatcher, and MoE
metadata.

Inject `_moe_packed_seq_params_unexpected` and assert that any undeclared key
with the MoE prefix fails before a graph callable is invoked. Do not ignore,
forward, or silently strip unknown names.

Add the scheduler-only all-padding case as `N == 1`, IDs all zero, and a fully
true structural mask. The bank must accept its fixed signature, forward the
fully masked inputs, count the exact graph call, and record no fallback. Task 3
owns its zero router-loss/gradient math; the runtime gate checks both together.

- [ ] **Step 3: Run and record red output**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
uv run python -m pytest \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "seq_aux or moe_packed or graph_bank" -q
```

Expected: failures because only generic attention/Mamba packed namespaces are
registered and fingerprinted.

- [ ] **Step 4: Propagate ownership through TransformerLayer**

For eager MoE, pass the original packed metadata to Task 3's MoELayer API.
For TE capture/replay, flatten and rebuild the dedicated namespace only when
the specific inner Transformer leaf captures whole layer, `moe`,
`moe_router`, or `moe_preprocess`. Store MoE static metadata and exact Tensor
key names separately from generic attention metadata. Do not mutate the
producer object and do not add PR 6115's TE rejection.

Every eager, recompute, and partial call to a MoE `self.mlp` receives
`packed_seq_params=packed_seq_params`. When explicit variable ownership is
present, make `should_chunk_mlp_for_training` false so independent chunks do
not normalize `seq_aux_loss` separately.

- [ ] **Step 5: Add canonical sample and bank integration**

Derive MoE sample ownership from Task 7's canonical graphable inner-leaf
descriptors. Never attach it to `HybridStack` or the MTP owner. Extend bank
fingerprints and transactional install/restore/reset with the two Tensor
signatures and static `S_cap`. Values may change between runs; signatures and
capacity may not. Validate that the complete set of keys carrying
`_moe_packed_seq_params_` is exactly the declared set for that scope; reject an
unknown prefixed key before graph invocation. Keep execution counters at the
actual eligible/graph callable boundaries.

- [ ] **Step 6: Run focused local checks**

```bash
uv run python -m pytest \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "seq_aux or moe_packed or graph_bank or mtp or hybrid" -q
uv run isort megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py
uv run ruff check megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py
git diff --check
```

- [ ] **Step 7: Commit only Task 4 files**

```bash
git add megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -S -m "feat: graph packed MoE sample ownership"
git log -1 --show-signature --format=fuller
```

---

### Task 5: Gate Hybrid, Mamba, MTP, and Distributed Correctness

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_aux_loss.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/models/test_hybrid_moe_model.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/ssm/test_mamba_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_submodule_callables.py`
- Modify: `tests/unit/algorithms/sequence_packing_gradient_actor.py`

**Interfaces:**
- Consumes: reviewed Tasks 1–4.
- Produces: deterministic CPU/isolated regressions and executable distributed
  tests for the parent plan's Linux/CUDA gate.

- [ ] **Step 1: Add failing Hybrid/MTP ownership tests**

Use ordered `[MoE Transformer, Mamba, dense Transformer]` fixtures with MTP
depths zero, one, and two and unequal physical samples. Assert ownership stays
unchanged under within-sequence MTP roll, Mamba never registers it, and only
the inner MoE Transformer leaf owns the namespace. Preserve Task 7's one-MoE
fine-grained bound: zero, mixed, and multi-MoE-leaf unsupported layouts fail
before schedule execution.

- [ ] **Step 2: Add the NeMo-RL policy-gradient oracle**

Run the same fixed-capacity pack through padded eager and packed eager with
identical deterministic parameters. Compare policy loss, sequence auxiliary
loss, valid-token logits, input gradient, router gradient, routed-expert
gradient, and zero padding gradient. Use literal sequence lengths `[3, 5]`
and a capacity with both inter-sequence and tail padding.

- [ ] **Step 3: Run local/isolated tests**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
uv run python -m pytest \
  tests/unit_tests/transformer/moe/test_aux_loss.py \
  tests/unit_tests/models/test_hybrid_moe_model.py \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_submodule_callables.py \
  -k "seq_aux or packed or mtp or hybrid" -q
git diff --check

cd /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731
uv run python -m pytest \
  tests/unit/algorithms/sequence_packing_gradient_actor.py \
  -k "seq_aux or packed" -q
git diff --check
```

- [ ] **Step 4: Commit test ownership in dependency order**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
git add tests/unit_tests/transformer/moe/test_aux_loss.py \
  tests/unit_tests/models/test_hybrid_moe_model.py \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_submodule_callables.py
git commit -s -S -m "test: verify packed MoE graph parity"
git log -1 --show-signature --format=fuller

cd /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731
git add tests/unit/algorithms/sequence_packing_gradient_actor.py
git commit -s -m "test: verify packed policy gradient ownership"
```

---

## Runtime Gate After the Parent Task 10 Pin

Tasks 1–5 and their independent reviews finish before parent Task 10. Parent
Task 10 then pins the reviewed MCore commit into Bridge and NeMo-RL. Parent
Task 12 runs the mandatory runtime gate below before Task 13 performance jobs.

In the pinned nightly container, compare padded eager, packed eager, and packed
TE graph for TP1/CP1, TP2, TP2+SP, CP2, TP2+SP+CP2, and PP2/VP where supported.
Run top-k one/two, AllToAll and supported Flex/HybridEP, shared expert off/on,
and Nano/Super/Ultra Hybrid fixtures. Require exactly three warmups, at least
two replays per occupancy, exact forward/loss/all-gradient parity, zero padding
contribution, nonzero graph calls, and zero fallback.

The TP1/CP1 oracle explicitly covers equal `[8, 8]` and unequal `[4, 8]`
physical lengths, exact token occupancy and extra tail capacity, and router
score functions `softmax`, `sigmoid`, and `sqrtsoftplus`. Model rows also use
their production score-function settings.

Graph parity explicitly includes router probabilities, pre/post-drop routing
maps, per-sample/per-expert counts, attached auxiliary gradient scale, valid
input gradients, and padding input gradients; aggregate expert counts and
top-k indices alone are insufficient.

The first runtime row uses one actual `torch.cuda.CUDAGraph` after exactly
three warmups and replays the same graph with `N == 1`, an intermediate `N`,
and `N == S_cap` while every signature stays fixed. It must show no recapture,
no fallback, exact counters, and equal loss/gradients. This behavior is the
no-host-sync gate; no source-text grep test substitutes for it.

Before that row, capture and replay a minimal `torch._assert_async` probe in
the exact pinned PyTorch/TE container. Then run invalid `N == 0` and
`N > S_cap` replays in separate subprocesses and require the expected device
assertion. Each invalid case needs a fresh process because a device assertion
poisons its CUDA context. A failed probe leaves variable packed
`seq_aux_loss` graph scopes unsupported and blocks performance submission.

Record runtime/container provenance and results in the existing persistent
experiment report. Only after this gate passes may parent Task 13 run 20-step
scope attribution, the 40-step baseline-versus-best accuracy comparison, and
the 100-step stability soak. Keep checkpoints disabled and W&B project
`sna-cg-study`.
