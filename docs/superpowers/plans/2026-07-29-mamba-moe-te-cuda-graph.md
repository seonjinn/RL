# Mamba and MoE TE Partial CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add numerically validated Transformer Engine partial CUDA Graph support for packed attention, Mamba, and MoE scopes, including bounded reuse across variable PP microbatch schedules.

**Architecture:** Megatron-LM owns packed graph-input adaptation and a model-scoped graph-bank manager whose banks own TE callable references. NeMo-RL owns a two-entry LRU keyed by PP `num_microbatches`, activates a drained bank before each forward/backward schedule, and preserves the three-successful-step warmup contract. NeMo-RL pins a Megatron-Bridge branch that pins the new Megatron-LM branch.

**Tech Stack:** Python 3.12, PyTorch distributed, Transformer Engine 2.10+, Megatron-LM/Megatron-Core, Megatron-Bridge, NeMo-RL, Ray, SLURM, pytest, Ruff, YAML, Bash.

## Global Constraints

- Use fresh branches from the latest upstream NeMo-RL, Megatron-Bridge, and NVIDIA Megatron-LM main branches.
- Apply the latest Megatron-LM PR 5672 head before adding Mamba support.
- Preserve latest-main CUDA Graph correctness and buffer-lifetime behavior, including PR 5975.
- Use Transformer Engine training partial graphs; do not replace local or full-iteration graphs.
- Use `CudaGraphModule`; do not restore deprecated `CudaGraphScope` behavior.
- Mamba graph inputs always include `seq_idx`; include `cu_seqlens_q` and `cu_seqlens_q_padded` only when CP is active; never graph Mamba-unused KV fields.
- Keep `total_tokens` outside the graph boundary so replay never reconstructs `seq_idx`.
- Cache only PP schedule count after validating fixed seq length, microbatch size, packed metadata, model topology, scope, precision, recompute, overlap, and offload invariants.
- Normalize PP1 to schedule key `1`.
- Default `policy.megatron_cfg.cuda_graph_max_cached_schedules` to `2`.
- Require Transformer Engine 2.10 or newer when eviction is enabled.
- Switch or evict banks only at a drained optimizer-step boundary with no live forward, backward, delayed-wgrad, communication, or MoE dispatcher state.
- Never use an inactive stock `TECudaGraphHelper.delete_cuda_graphs()`; reset bank-owned graph handles by identity.
- Silent eager fallback after warmup is forbidden.
- Use three successful optimizer updates before the first graph capture.
- Use checkpointing disabled for all performance and accuracy experiments.
- Use W&B project `sna-cg-study`.
- Use the installed nightly Transformer Engine directly; do not build TE natively.
- Follow TDD: add one failing behavior test, verify the expected failure, implement the minimum code, verify green, then refactor.
- Commit only task-owned files with `git commit -s`; Megatron-LM commits also use `-S` when the local signing key is available.
- Push in dependency order: Megatron-LM, Megatron-Bridge nested pointer, then NeMo-RL Bridge pointer.

## File and Ownership Map

Megatron-LM:

- `megatron/core/packed_seq_params.py`: attention and Mamba Tensor/static packed schemas and signatures.
- `megatron/core/ssm/mamba_layer.py`: Mamba capture/replay flattening and contract validation.
- `megatron/core/transformer/cuda_graphs.py`: PR 5672 sample wiring, helper capture refactor, and compatibility entrypoints.
- `megatron/core/transformer/te_cuda_graph_bank.py`: bank value object, model-scoped ownership, activation, reset, and exception-safe capture.
- `megatron/core/transformer/transformer_layer.py`: latest-main attention/MoE partial replay integration and MoE drained/schema hooks.
- `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`: pure adapter contract tests.
- `tests/unit_tests/transformer/test_te_cuda_graph_bank.py`: pure bank ownership and lifecycle tests.
- `tests/unit_tests/transformer/test_cuda_graphs.py`: real-GPU eager/graph parity and bank-switch integration.

NeMo-RL:

- `nemo_rl/models/megatron/cuda_graph_lifecycle.py`: new schedule key, LRU, warmup, capture result, and cleanup.
- `nemo_rl/data/packing/algorithms.py`: packed-bin sequence-count bound.
- `nemo_rl/distributed/batched_data_dict.py`: sequence-count bound plumbing.
- `nemo_rl/models/megatron/data.py`: fixed packed token/boundary geometry and capture sample.
- `nemo_rl/models/policy/__init__.py`: user-facing cache setting type.
- `nemo_rl/models/megatron/setup.py`: config validation and provenance logging.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`: helper factory, schedule activation, PP calculator reconfiguration, telemetry, and offload guards.
- `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`: LRU and failure behavior.
- `tests/unit/data/packing/test_algorithms.py`: packed-bin sequence-count bound.
- `tests/unit/models/megatron/test_megatron_data.py`: fixed graph geometry.
- `tests/unit/models/megatron/test_megatron_setup.py`: config validation.
- `tests/unit/models/policy/test_megatron_worker.py`: worker schedule transitions and split-step pinning.
- `experiments/cuda_graph/mamba_moe_te_graph_20260729/`: reusable launchers, manifests, collection, and report generation.
- `tests/unit/experiments/test_mamba_moe_te_graph_launchers.py`: launcher and scope-matrix contract.

Megatron-Bridge:

- `3rdparty/Megatron-LM`: nested submodule pointer only.

---

### Task 0: Create the Fresh Three-Repository Worktree

**Files:**
- Create worktree:
  `/Users/sna/CudaGraph_PR/RL-pr5672-mamba-moe-graph-cache-20260729`
- Initialize: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Initialize:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`

**Interfaces:**
- Consumes: latest NeMo-RL `origin/main` plus this committed design and plan.
- Produces: isolated NeMo-RL, Megatron-Bridge, and Megatron-LM branches without
  modifying any existing dirty worktree.

- [ ] **Step 1: Fetch and create the NeMo-RL worktree**

From `/Users/sna/CudaGraph_PR/RL`:

```bash
git fetch origin main
git worktree add \
  /Users/sna/CudaGraph_PR/RL-pr5672-mamba-moe-graph-cache-20260729 \
  -b experiment/pr5672-mamba-moe-graph-cache-20260729 \
  origin/main
```

- [ ] **Step 2: Import only the approved documentation**

Resolve the plan commit from the source documentation branch, then cherry-pick
the design, provenance update, and plan:

```bash
PLAN_COMMIT="$(
  git rev-parse experiment/latestmain-pr5672-nano-matrix-20260727^{commit}
)"
git cherry-pick 3360024c4 1bd9e182d "${PLAN_COMMIT}"
```

Expected: the new branch contains no old experimental code changes.

- [ ] **Step 3: Initialize the nested repositories**

Run:

```bash
git submodule update --init --recursive
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge fetch upstream main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge switch -c \
  sna/pr5672-mamba-moe-graph-cache-20260729 upstream/main
git -C \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  fetch upstream main
```

If the expected `upstream` remote is absent, add the NVIDIA repository URL
documented by that submodule before fetching. Push targets remain the
`seonjinn` forks.

- [ ] **Step 4: Verify isolation**

Run:

```bash
git status --short
git log --oneline --decorate --max-count=6
git submodule status --recursive
```

Expected: only the three documentation commits and intentional latest-main
nested submodule pointers differ from latest NeMo-RL main; both nested
repositories are initialized and have no uncommitted file changes.

### Task 1: Establish Latest-Main and PR 5672 Foundation

**Files:**
- Modify through cherry-pick: `megatron/core/packed_seq_params.py`
- Modify through cherry-pick: `megatron/core/transformer/cuda_graphs.py`
- Modify through cherry-pick: `megatron/core/transformer/transformer_layer.py`
- Modify through cherry-pick: `megatron/rl/rl_utils.py`
- Modify through cherry-pick: `megatron/rl/sequence_packing_utils.py`
- Modify through cherry-pick: `megatron/training/training.py`
- Modify through cherry-pick: `train_rl.py`
- Test through cherry-pick: `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`
- Test through cherry-pick: `tests/unit_tests/rl/test_rl_utils.py`

**Interfaces:**
- Consumes: latest NVIDIA Megatron-LM main and GitHub PR 5672 head `6ff66f0a000ee65efa4f322c17871a3938f33427` as observed on 2026-07-29.
- Produces: attention `PackedSeqParams` split/build helpers and the
  `sample_packed_seq_params` input on `TECudaGraphHelper`.

- [ ] **Step 1: Record exact source provenance**

Run:

```bash
git fetch upstream main
git fetch upstream pull/5672/head:refs/remotes/upstream/pr5672
git rev-parse upstream/main upstream/pr5672
git merge-base upstream/main upstream/pr5672
git log --oneline --reverse "$(git merge-base upstream/main upstream/pr5672)..upstream/pr5672"
```

Expected: three PR commits ending at `6ff66f0a0`, and latest main contains commit `4b18b260f` or an equivalent descendant for PR 5975.

- [ ] **Step 2: Create the Megatron-LM feature branch and apply PR 5672**

Run:

```bash
git switch -c sj/pr5672-mamba-moe-graph-cache-20260729 upstream/main
git cherry-pick 4cb58d5d6 1ba1418b8 6ff66f0a0
```

Resolve conflicts by retaining latest-main graph memory, MoE explicit-output, and type signatures while preserving the PR's attention-only packed adapter.

- [ ] **Step 3: Verify the imported adapter**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/rl/test_rl_utils.py
```

Expected: all imported PR tests pass.

- [ ] **Step 4: Verify ancestry and clean state**

Run:

```bash
git merge-base --is-ancestor upstream/main HEAD
git log --oneline --decorate --max-count=8
git status --short
```

Expected: exit 0 for the ancestry check and an empty worktree.

### Task 2: Add the Minimal Packed-Mamba Graph Adapter

**Files:**
- Modify: `megatron/core/packed_seq_params.py`
- Modify: `megatron/core/ssm/mamba_layer.py`
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Test: `tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: PR 5672 attention adapter from Task 1.
- Produces:
  - `split_mamba_packed_seq_params_for_cuda_graph(packed_seq_params, include_cp_fields)`.
  - `build_mamba_packed_seq_params_from_cuda_graph_kwargs(kwargs, static_metadata)`.
  - Mamba layer methods that store and validate Tensor signatures.
  - TE helper sample wiring for explicit and whole-layer Mamba capture.

- [ ] **Step 1: Add failing schema tests**

Import `megatron.core.packed_seq_params` as a module so missing helpers produce
an assertion failure instead of a collection error. Add this discovery helper
and the schema assertions:

```python
def _get_mamba_graph_helpers() -> tuple[Callable[..., object], Callable[..., object]]:
    split = getattr(
        packed_seq_module,
        "split_mamba_packed_seq_params_for_cuda_graph",
        None,
    )
    build = getattr(
        packed_seq_module,
        "build_mamba_packed_seq_params_from_cuda_graph_kwargs",
        None,
    )
    assert callable(split)
    assert callable(build)
    return split, build


def test_mamba_graph_schema_uses_only_consumed_tensor_fields() -> None:
    split, _ = _get_mamba_graph_helpers()
    packed = _make_mamba_packed_seq_params()
    tensor_kwargs, static = split(packed, include_cp_fields=True)
    assert set(tensor_kwargs) == {
        "_mamba_packed_seq_params_seq_idx",
        "_mamba_packed_seq_params_cu_seqlens_q",
        "_mamba_packed_seq_params_cu_seqlens_q_padded",
    }
    assert "_mamba_packed_seq_params_cu_seqlens_kv" not in tensor_kwargs
    assert "total_tokens" not in static


def test_mamba_graph_schema_without_cp_uses_only_seq_idx() -> None:
    split, _ = _get_mamba_graph_helpers()
    tensor_kwargs, _ = split(
        _make_mamba_packed_seq_params(), include_cp_fields=False
    )
    assert set(tensor_kwargs) == {"_mamba_packed_seq_params_seq_idx"}


def test_mamba_rebuild_preserves_supplied_seq_idx_identity() -> None:
    split, build = _get_mamba_graph_helpers()
    packed = _make_mamba_packed_seq_params()
    tensor_kwargs, static = split(packed, include_cp_fields=True)
    rebuilt = build(dict(tensor_kwargs), static)
    assert rebuilt.seq_idx is packed.seq_idx
    assert rebuilt.total_tokens is None
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "mamba_graph_schema or mamba_rebuild"
```

Expected: FAIL because the Mamba helpers do not exist.

- [ ] **Step 3: Implement the schema**

Add the two field lists and two typed helpers. The split helper must:

```python
MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_mamba_packed_seq_params_"
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = ("seq_idx",)
MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_CP_TENSOR_FIELDS = (
    "cu_seqlens_q",
    "cu_seqlens_q_padded",
)
```

- Return empty Tensor kwargs and explicit static metadata for a `None` packed
  input.
- Construct the consumed field list from the two constants.
- Validate every non-`None` dynamic field with `isinstance(value, Tensor)`.
- Require `seq_idx` to be a Tensor whenever packed Mamba parameters are
  present, and require its prefixed key during reconstruction.
- Store only reconstruction metadata that Mamba consumes and omit
  `total_tokens`.
- Never include KV fields.
- Rebuild a new `PackedSeqParams` from a copied kwargs dictionary and fail if
  any expected prefixed key is missing.

- [ ] **Step 4: Verify GREEN for schema tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Add failing Mamba layer capture/replay tests**

Add tests that assert:

- Replay with a changed `seq_idx` shape raises `AssertionError` before TE is
  invoked.
- Replay with a changed `seq_idx` dtype raises `AssertionError` before TE is
  invoked.
- Replay with changed static metadata or a changed dynamic field set raises
  `AssertionError` before TE is invoked.
- Explicit `mamba` scope passes the packed Mamba Tensor sample to the graph
  callable.
- Whole-layer Mamba capture passes the same sample and does not add attention
  KV fields.
- A requested explicit `mamba` scope that discovers no `MambaLayer` on any
  pipeline stage raises consistently on every PP rank instead of silently
  using eager execution. A rank with no local Mamba layer remains valid when
  another stage in its pipeline group contains one.

Use signatures containing exact `shape`, `dtype`, `device`, `layout`, and `stride`.

- [ ] **Step 6: Verify RED for layer behavior**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  -k "mamba_layer or helper_adds_mamba"
```

Expected: FAIL because `MambaLayer` does not flatten/rebuild packed inputs.

- [ ] **Step 7: Implement Mamba capture and replay**

In `MambaLayer`:

```python
def _te_cuda_graph_capture(self, *args: object, **kwargs: object) -> Tensor:
    self._rebuild_te_cuda_graph_mamba_packed_seq_params(kwargs)
    return self.forward(*args, **kwargs)


def _te_cuda_graph_replay(self, *args: object, **kwargs: object) -> object:
    assert kwargs.get("inference_context") is None
    self._flatten_te_cuda_graph_mamba_packed_seq_params(kwargs)
    return super()._te_cuda_graph_replay(*args, **kwargs)
```

Store the static metadata and exact Tensor signatures at sample creation. Replay compares signatures before invoking TE. Derive `include_cp_fields` from `layer.config.context_parallel_size > 1`.

- [ ] **Step 8: Verify the complete adapter test file**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
uv run isort \
  megatron/core/packed_seq_params.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
```

Expected: all tests pass and import order is clean.

- [ ] **Step 9: Commit**

Run:

```bash
git add \
  megatron/core/packed_seq_params.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py
git commit -s -S -m "feat: support packed Mamba TE CUDA graphs"
```

### Task 3: Add Explicit TE Graph-Bank Ownership

**Files:**
- Create: `megatron/core/transformer/te_cuda_graph_bank.py`
- Modify: `megatron/core/transformer/cuda_graphs.py`
- Test: `tests/unit_tests/transformer/test_te_cuda_graph_bank.py`

**Interfaces:**
- Consumes: graphable layers and `TECudaGraphHelper` from Tasks 1-2.
- Produces:
  - `TECudaGraphBankFingerprint`.
  - `TECudaGraphBank.activate()` and `TECudaGraphBank.reset()`.
  - `TECudaGraphBankManager.capture(helper, num_microbatches)`.
  - bank-specific activation, uninstall, and reset.
  - backward-compatible single-bank `TECudaGraphHelper.create_cudagraphs()`.

- [ ] **Step 1: Add failing ownership tests**

Import the new module through this assertion helper so the initial failure is
an ordinary test failure:

```python
def _load_bank_module() -> ModuleType:
    spec = importlib.util.find_spec(
        "megatron.core.transformer.te_cuda_graph_bank"
    )
    assert spec is not None
    return importlib.import_module(spec.name)
```

Use fake layers with `cuda_graphs`, fake graphs with counted `reset()`, and a
fake helper that asserts layer lists are empty during capture. Add tests that
assert:

- Capture clears every installed graph list before invoking the helper.
- Activation installs each bank's exact tuple on every layer.
- A bank created by a different manager is rejected.
- The replay guard rejects the wrong `num_microbatches` before selecting a
  graph.
- Resetting an inactive bank leaves every active graph untouched.
- A capture exception restores the previously active bank.
- Reset is idempotent and resets only graph identities owned by the bank.
- Activation and eviction reject a model-drain callback that reports live
  delayed-wgrad or communication work.
- Activation rejects a bank whose capture scope, packed static signature, or
  layer topology differs from the manager fingerprint.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py
```

Expected: FAIL at `_load_bank_module` because `te_cuda_graph_bank` is absent;
test collection succeeds.

- [ ] **Step 3: Implement the bank value objects**

Define:

```python
@dataclass(frozen=True)
class TECudaGraphBankFingerprint:
    num_microbatches: int
    layer_ids: tuple[int, ...]
    graph_counts: tuple[int, ...]
    cuda_graph_modules: tuple[str, ...]
    packed_input_signature: tuple[tuple[str, object], ...]
    moe_attribute_schema: tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(eq=False)
class TECudaGraphBank:
    fingerprint: TECudaGraphBankFingerprint
    graphs_by_layer: tuple[
        tuple[GraphableMegatronModule, tuple[object, ...]], ...
    ]
    _manager: "TECudaGraphBankManager" = field(repr=False)

    def activate(self) -> None:
        self._manager.activate(self)

    def reset(self) -> None:
        self._manager.reset(self)
```

The manager owns one active bank, verifies every layer identity and graph
count, invokes a model-drain assertion before capture/activation/reset, and
synchronizes before explicit reset.

Each `(layer, graphs)` entry also owns the layer replay contract active at
capture time:

```text
cuda_graph_manual_hooks
_te_cuda_graph_packed_seq_params_static_metadata
_te_cuda_graph_packed_seq_params_tensor_kwarg_names
_te_cuda_graph_mamba_packed_seq_params_static_metadata
_te_cuda_graph_mamba_packed_seq_params_tensor_signatures
```

Activation installs the contract and hooks before assigning
`layer.cuda_graphs`. A bank without packed inputs explicitly clears stale
packed attributes from the previously active bank.

- [ ] **Step 4: Refactor helper capture to return owned graph lists**

Split existing capture into a private `_capture_cuda_graph_lists()` method
that returns immutable `(layer, graphs)` pairs and this public method:

```python
def create_cuda_graph_bank(
    self,
    manager: TECudaGraphBankManager,
    *,
    num_microbatches: int,
) -> TECudaGraphBank:
    return manager.capture(self, num_microbatches=num_microbatches)
```

Before capture, install empty manager-owned lists on every graphable layer so
`GraphableMegatronModule.__call__` enters capture instead of replaying the
previous bank. Map TE results with the existing normal and EP-overlap index
formulas into those same list objects; list identity is the ownership token.
Vision wrapping mutates the owned list in place.

Existing `create_cudagraphs()` captures one bank, activates it, and stores that
exact bank for its compatibility `delete_cuda_graphs()`. Compatibility deletion
resets only its owned callables and clears layer-facing state only when the
installed list is the same object.

- [ ] **Step 5: Make capture exception-safe**

Ensure `_finish_capturing` or a dedicated abort cleanup runs from `finally`,
clears both MCore and TE capture flags, unfreezes GC, resets fine-grained
offload capture state, resets partial owned callables under the TE 2.10 guard,
marks the failed helper uncreated, and restores the previously active lists,
hooks, and packed replay contracts before re-raising.

Add mapping tests for two layers/two microbatches:

```text
normal mapping: layer0=[g0,g2], layer1=[g1,g3]
EP-overlap mapping: layer0=[g0,g1], layer1=[g2,g3]
```

- [ ] **Step 6: Verify GREEN**

Run the Step 2 command.

Expected: all ownership tests pass.

- [ ] **Step 7: Run existing helper tests**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "TECudaGraphHelper and not cuda"
```

Expected: CPU-selectable helper tests pass; GPU-only tests skip without a CUDA worker.

- [ ] **Step 8: Commit**

Run:

```bash
git add \
  megatron/core/transformer/te_cuda_graph_bank.py \
  megatron/core/transformer/cuda_graphs.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py
git commit -s -S -m "feat: add owned TE CUDA graph banks"
```

### Task 4: Make MoE State Safe Across Bank Switching

**Files:**
- Modify: `megatron/core/transformer/transformer_layer.py`
- Modify: `megatron/core/transformer/te_cuda_graph_bank.py`
- Test: `tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: bank manager from Task 3 and latest-main explicit MoE graph outputs.
- Produces:
  - `TransformerLayer.te_cuda_graph_bank_schema()`.
  - `TransformerLayer.assert_te_cuda_graph_bank_drained()`.
  - `TransformerLayer.clear_te_cuda_graph_bank_references()`.
  - bank fingerprint validation for dispatcher attribute names.

- [ ] **Step 1: Add failing MoE drained-state tests**

Add tests that assert:

- Switching banks with a nonempty MoE Tensor store raises before uninstall.
- Reset removes stale dispatcher Tensor references for the reset bank.
- Activation rejects a changed `valid_cudagraph_attrs` schema.
- Manual DDP hooks retain object identity after inactive-bank reset.
- An FP64 router graph output retains FP64 without an implicit downcast.
- Shared-expert overlap enabled and disabled both preserve the drained-state
  contract.
- Selective `moe_act` recompute enabled and disabled both preserve the
  dispatcher schema.

`moe_act` and shared expert are configuration dimensions, not members of
`CudaGraphModule`; test them inside `moe` and
`moe_router+moe_preprocess` boundaries rather than inventing unsupported scope
names.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  -k "moe or manual_ddp or fp64"
```

Expected: FAIL because bank switching has no MoE hooks.

- [ ] **Step 3: Implement MoE layer hooks**

Add hooks to ordinary `TransformerLayer`, guarded by `self.is_moe_layer`;
`MoETransformerLayer` inherits them. Standard GPT MoE layers are not instances
of `MoETransformerLayer`.

```python
def te_cuda_graph_bank_schema(self) -> tuple[str, ...]:
    if not self.is_moe_layer:
        return ()
    return tuple(self.mlp.token_dispatcher.valid_cudagraph_attrs or ())
```

After the model-level drain callback and `torch.cuda.synchronize()`, the drain
assertion checks:

- all four `cudagraph_tensor_store` fields are `None`;
- overlapped shared-expert state is `IDLE` and its cached forward fields are
  `None`;
- Flex/HybridEP dispatcher `handle` or NCCL EP `_buffer` is `None`;
- a live grouped-MLP `moe_act` checkpoint has no retained context/output.

Only uninstalling the active bank clears active dispatcher leaf references and
the Tensor store. Resetting an inactive bank never calls a layer-global clear
hook and never touches active bank state. Preserve `valid_cudagraph_attrs` as
structural schema, clear only its resolved leaf values, and do not restore the
old weak-reference side channel.

There is no reliable host-visible pending flag for every delayed-wgrad and
communication stream. The model-level callback therefore proves the
optimizer/schedule boundary, then synchronizes CUDA before these logical
assertions.

- [ ] **Step 4: Verify GREEN and existing MoE tests**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "moe or router or bank"
```

Expected: new tests pass; applicable existing tests pass or GPU-skip.

Run FP64 preservation at the graph boundary or with all-to-all. DeepEP and
HybridEP intentionally cast router probabilities to FP32, so they are not a
universal FP64 end-to-end gate. Do not cross shared-expert overlap with EP
overlap, and enable grouped GEMM for selective `moe_act` tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -S -m "fix: drain MoE state before graph bank switches"
```

### Task 5: Add Distributed MCore Correctness Coverage

**Files:**
- Modify: `tests/unit_tests/transformer/test_cuda_graphs.py`
- Create: `tests/functional_tests/test_cases/hybrid/te_graph_bank/model_config.yaml`
- Modify: `tests/test_utils/recipes/gb200/unit-tests.yaml`

**Interfaces:**
- Consumes: Mamba adapter and graph-bank manager from Tasks 2-4.
- Produces: eager/graph numerical gates for explicit scopes and schedule switching.

- [ ] **Step 1: Add the explicit Mamba parity test**

Create a real-GPU test that initializes identical eager and TE graph hybrid models, uses packed boundaries `[8, 24]` for capture and `[12, 20]` for replay, executes backward, and compares output plus every parameter gradient.

- [ ] **Step 2: Add the `5 -> 3 -> 5` bank test**

The test:

```python
schedule = [5, 3, 5]
for num_microbatches in schedule:
    bank = banks.get(num_microbatches)
    if bank is None:
        bank = capture_bank(num_microbatches)
        banks[num_microbatches] = bank
    bank.activate()
    run_forward_backward(num_microbatches)

assert capture_counts == {5: 1, 3: 1}
assert fallback_count == 0
```

Compare eager and graph outputs, losses, gradients, and one optimizer update. Routing IDs and expert token counts are exact.

- [ ] **Step 3: Verify RED on a GB200 allocation**

Run:

```bash
uv run python -m torch.distributed.run --nproc-per-node 2 -m pytest -q \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "packed_mamba_te_cuda_graph or te_graph_bank_schedule_switch"
```

Expected before Tasks 2-4: failure from absent Mamba adapter/bank API. At this task boundary the tests must pass.

- [ ] **Step 4: Add production-topology functional coverage**

Configure TP2/PP2/CP2/EP8, BF16, dropout 0, packed max sequences 16, and scopes:

```text
[]
[attn]
[mlp]
[attn,mlp]
[mamba]
[mlp,mamba]
[mamba,moe]
[mamba,moe_router,moe_preprocess]
[moe_router,moe_preprocess]
[attn,mamba,moe_router,moe_preprocess]
```

For full `moe` and `moe_router+moe_preprocess`, parameterize:

```text
moe_shared_expert_overlap = false, true
selective recompute module moe_act = false, true
```

Do not advertise `moe_act` or `shared_expert` as independent graph scopes:
verify from graph discovery and replay telemetry that their work executes
inside the selected MoE boundary.

Full `moe` tests explicitly configure expert-capacity drop-and-pad as required
by MCore. Default-model configurations that do not satisfy this contract must
fail preflight and be labeled unsupported rather than silently changing the
recipe.

Use successful warmups 1-3, schedule 5 on step 4, schedule 3 on steps 5-6, and schedule 5 on step 7.

- [ ] **Step 5: Commit tests**

Run:

```bash
git add \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/functional_tests/test_cases/hybrid/te_graph_bank/model_config.yaml \
  tests/test_utils/recipes/gb200/unit-tests.yaml
git commit -s -S -m "test: cover hybrid TE graph bank switching"
```

### Task 6: Refactor NeMo-RL Lifecycle into a Bounded Schedule LRU

**Files:**
- Create: `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- Create: `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`

**Interfaces:**
- Consumes: MCore bank objects with `activate()` and `reset()`.
- Produces:
  - `TECudaGraphScheduleKey`.
  - `TECudaGraphEnsureResult`.
  - `TECudaGraphLifecycle.ensure_active(key, capture_bank)`.

- [ ] **Step 1: Add failing lifecycle tests**

Use fake banks that count `activate()` and `reset()` calls. Add tests that
assert:

- Three globally successful optimizer steps warm the first key, while a
  second key captures immediately afterward.
- A failed optimizer update does not advance the global warmup counter or
  mutate cached banks.
- A cache hit activates without calling the capture callback.
- Capacity two evicts only the least-recently-used inactive bank.
- Capacity one recaptures alternating keys.
- A capture exception leaves the cache and active key unchanged.
- `close()` resets every distinct cached bank exactly once.
- PP1 normalizes all positive runtime counts to key one.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py
```

Expected: FAIL at an assertion-based module discovery helper because latest
NeMo-RL main has no training graph lifecycle module.

- [ ] **Step 3: Implement typed schedule and result objects**

Add:

```python
@dataclass(frozen=True)
class TECudaGraphScheduleKey:
    num_microbatches: int

    @classmethod
    def from_runtime(
        cls, *, pipeline_parallel_size: int, num_microbatches: int
    ) -> "TECudaGraphScheduleKey":
        return cls(1 if pipeline_parallel_size == 1 else num_microbatches)


@dataclass(frozen=True)
class TECudaGraphEnsureResult:
    key: TECudaGraphScheduleKey
    status: Literal["warming", "hit", "captured"]
    evicted_key: TECudaGraphScheduleKey | None
```

- [ ] **Step 4: Implement the LRU**

Use `OrderedDict[TECudaGraphScheduleKey, TECudaGraphBankProtocol]`. Warmup is global. `ensure_active` inserts only after successful capture, activates hits, resets the least-recently-used inactive entry at capacity, and returns telemetry.

- [ ] **Step 5: Verify GREEN**

Run the Step 2 command.

Expected: all lifecycle tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add \
  nemo_rl/models/megatron/cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py
git commit -s -m "feat: cache TE CUDA graphs by PP schedule"
```

### Task 7: Integrate Schedule Banks into the NeMo-RL Worker

**Files:**
- Modify: `nemo_rl/data/packing/algorithms.py`
- Modify: `nemo_rl/distributed/batched_data_dict.py`
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/data/packing/test_algorithms.py`
- Modify: `tests/unit/models/megatron/test_megatron_data.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`

**Interfaces:**
- Consumes: lifecycle from Task 6 and MCore graph-bank manager from Tasks 3-4.
- Produces: training graph scope/config plumbing absent from latest NeMo-RL
  main, fixed packed token/boundary geometry, full worker behavior for
  `5 -> 3 -> 5`, TE-only cache default 2, schedule telemetry, and zero
  fallback.

- [ ] **Step 1: Add failing config tests**

Add config tests that assert:

- A TE training-graph config that omits the cache field is normalized to
  capacity two during setup.
- Zero, negative, Boolean, and non-integer capacities are rejected.
- Any explicitly supplied cache capacity is rejected for a non-TE
  implementation.
- Eviction is rejected when the parsed TE version is below 2.10.
- Packed TE graphs require positive `cuda_graph_max_packed_seqs` and sequence
  packing.
- `moe` and `moe_router` are mutually exclusive; `moe_preprocess` without
  `moe_router` is rejected.

- [ ] **Step 2: Verify RED for config**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/models/megatron/test_megatron_setup.py \
  -k "cached_schedule or graph_cache"
```

Expected: FAIL because the setting is undefined.

- [ ] **Step 3: Add user-facing settings and fixed packed geometry**

Add to the existing TypedDict:

```python
cuda_graph_modules: NotRequired[list[str]]
cuda_graph_packed_seq: NotRequired[bool]
cuda_graph_max_packed_seqs: NotRequired[int | None]
cuda_graph_warmup_steps: NotRequired[int]
cuda_graph_max_cached_schedules: NotRequired[int]
```

When `cuda_graph_impl=transformer_engine`, setup inserts cache capacity two
before the worker is constructed. The worker reads the normalized value
directly; do not add a worker call-site fallback. Graph-specific experiment
scripts set warmup 3 and the model/workload-specific
`cuda_graph_max_packed_seqs`.

Port the fixed packed-geometry contract from the prior experiment with new
tests:

- Split packed bins so no microbatch exceeds
  `cuda_graph_max_packed_seqs`.
- Pad packed tokens to `sequence_packing.train_mb_tokens`.
- Build fixed-length cumulative-boundary Tensors of
  `cuda_graph_max_packed_seqs + 2`, including the terminal capacity sentinel.
- Preserve the true `cu_seqlens` separately for loss slicing.
- Build the helper's sample `PackedSeqParams` on the model device with the same
  token and sequence-count capacities.
- Reject runtime token or sequence counts that exceed either capacity.

- [ ] **Step 4: Add failing worker tests**

Add worker tests that assert:

- Runtime schedule `5 -> 3 -> 5` captures exactly twice and activates key five
  on the last call.
- The global microbatch calculator is reconfigured on a cache hit.
- A split train step pins one key until optimizer-step completion.
- Replay telemetry is logged once per key and counters continue increasing.
- CPU offload is forbidden while any graph bank is cached.
- Worker shutdown resets all cached banks.
- Every normalized combination of `attn`, dense `mlp`, `mamba`, and one valid
  MoE mode maps to the expected `CudaGraphModule` list; `moe_preprocess`
  without `moe_router` is rejected.

- [ ] **Step 5: Verify RED for worker**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/models/policy/test_megatron_worker.py \
  -k "cuda_graph and schedule"
```

Expected: FAIL on the current fixed-count mismatch behavior.

- [ ] **Step 6: Refactor worker initialization**

Keep fixed packed sample state and create one MCore `TECudaGraphBankManager`. Replace single-capture fields with:

```python
self._active_te_cuda_graph_key: TECudaGraphScheduleKey | None = None
self._te_cuda_graph_replay_logged_keys: set[TECudaGraphScheduleKey] = set()
self._te_cuda_graph_capture_counts: dict[TECudaGraphScheduleKey, int] = {}
self._te_cuda_graph_replay_counts: dict[TECudaGraphScheduleKey, int] = {}
self._te_cuda_graph_fallback_count = 0
```

- [ ] **Step 7: Replace fixed-count rejection with schedule activation**

Implement `_ensure_te_cuda_graph_schedule`:

1. Validate fixed seq length and microbatch size.
2. Build the normalized schedule key.
3. Reconfigure MCore's global calculator for every PP request, including hits.
4. Call lifecycle `ensure_active`.
5. Disable registered DDP prehooks only during a capture.
6. Require an MP-wide graph-created result.
7. Install shared manual hooks after capture.
8. Pin the key until the optimizer step drains.
9. Log status, key, capture/hit/eviction counters, and `fallback_count=0`.

The worker-provided model-drain assertion checks that no split train step,
delayed-wgrad queue, asynchronous collective, or MoE dispatcher Tensor store
is live. It is passed to the MCore manager and exercised before every bank
mutation.

Latest NeMo-RL main has no training `TECudaGraphHelper` lifecycle. Reintroduce
helper construction after model finalization, call schedule activation before
both synchronous and split forward/backward schedules, advance warmup only
after the MP-wide optimizer result succeeds, release a split-step pinned key on
finish or abort, guard every CPU relocation/refit path while any bank is
cached, and add worker shutdown cleanup.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py \
  -k "cuda_graph"
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit**

Run:

```bash
git add \
  nemo_rl/models/policy/__init__.py \
  nemo_rl/data/packing/algorithms.py \
  nemo_rl/distributed/batched_data_dict.py \
  nemo_rl/models/megatron/data.py \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/data/packing/test_algorithms.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py
git commit -s -m "feat: switch TE graphs across packed PP schedules"
```

### Task 8: Add Reusable Scope Launchers and Report Pipeline

**Files:**
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/profiles/ptyche.env`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/profiles/oci-hsg.env`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/run_scope.sh`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/scopes/*.sh`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/variants/*.sh`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_all_smokes.sh`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_performance.sh`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/render_report.py`
- Create: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_report.html`
- Create: `tests/unit/experiments/test_mamba_moe_te_graph_launchers.py`

**Interfaces:**
- Consumes: NeMo-RL worker implementation from Task 7.
- Produces: immutable per-scope scripts, test-only submission mode, CSV collection, and static HTML.

- [ ] **Step 1: Add failing launcher matrix tests**

Generate the exact 32-row TE scope matrix from three independent Boolean axes
and one mutually exclusive MoE axis:

```python
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)

VALID_GRAPH_SCOPES = {
    tuple(
        name
        for enabled, name in zip(enabled_dense, DENSE_AXES, strict=True)
        if enabled
    )
    + moe_scope
    for enabled_dense in itertools.product((False, True), repeat=3)
    for moe_scope in MOE_AXES
}

assert len(VALID_GRAPH_SCOPES) == 32
```

The empty tuple is TE whole-layer capture. Add a separate
`00_baseline_no_cg.sh` with `cuda_graph_impl=none`; never represent baseline
with an empty TE module list. Assert there is exactly one persistent script per
TE row plus baseline. Every TE script sets warmup 3 and cache capacity 2.
Every script sets max packed sequences 16, checkpointing false, W&B project
`sna-cg-study`, and a unique run name.

Add persistent variant scripts for both values of
`moe_shared_expert_overlap` and selective `moe_act` recompute under the full
`moe` and `moe_router+moe_preprocess` scopes. These variants reuse the common
runner and must not introduce `moe_act` or `shared_expert` into the graph-scope
list.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
```

Expected: FAIL because the experiment directory is absent.

- [ ] **Step 3: Implement common runner and thin scope scripts**

`run_scope.sh` validates `PHASE=smoke|performance|accuracy`, `SCOPE`, cluster profile, model snapshot, and container. It emits the full `COMMAND` and `sbatch` command under `TEST_ONLY=1`.

Profiles select one immutable base recipe:

```text
Nano hybrid:
  examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml
Qwen3-30B-A3B:
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
Qwen3-235B-A22B:
  examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
```

Cluster profiles must pin the exact HF revision or existing model snapshot and
the staged nightly container digest. Missing values make real submission fail
before `sbatch`; `TEST_ONLY=1` reports every unresolved field.

Every `scopes/*.sh` contains only its scope tuple and delegates:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCOPE='[attn,mamba,moe_router,moe_preprocess]' \
SCOPE_NAME=attn-mamba-moe-router-preprocess \
bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"
```

- [ ] **Step 4: Implement collection and HTML rendering**

Persist:

```text
scope,job_id,status,step,geometry_key,capture_count,replay_count,cache_hit_count,
eviction_count,fallback_count,e2e_step_time,e2e_tokens_per_sec_per_gpu,
generation_time,generation_tokens_per_sec_per_gpu,policy_training_time,
policy_training_tokens_per_sec_per_gpu,logprob_time,logprob_tokens_per_sec_per_gpu,
reward_mean,generation_kl_error,policy_loss,grad_norm,peak_allocated_gib,
peak_reserved_gib
```

The report has separate Correctness, Smoke, Performance, Accuracy, Failures,
and Provenance sections. It distinguishes no-CG baseline from empty-scope TE
whole-layer capture.
It additionally labels `moe_act` and shared-expert overlap as configuration
variants so they cannot be mistaken for graph scopes.

Map W&B metrics exactly:

```text
E2E throughput: performance/tokens_per_sec_per_gpu
generation throughput: performance/generation_tokens_per_sec_per_gpu
policy throughput: performance/policy_training_tokens_per_sec_per_gpu
logprob throughput:
  performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu
E2E time: timing/train/total_step_time
generation time: timing/train/generation
policy time: timing/train/policy_training
logprob time: timing/train/policy_and_reference_logprobs
quality: train/reward, train/accuracy, train/token_mult_prob_error, train/loss
```

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
for script in experiments/cuda_graph/mamba_moe_te_graph_20260729/scopes/*.sh; do
  TEST_ONLY=1 CLUSTER=ptyche bash "${script}"
done
for script in experiments/cuda_graph/mamba_moe_te_graph_20260729/variants/*.sh; do
  TEST_ONLY=1 CLUSTER=ptyche bash "${script}"
done
```

Expected: tests pass and every test-only command is accepted locally without submission.

- [ ] **Step 6: Commit**

Run:

```bash
git add \
  experiments/cuda_graph/mamba_moe_te_graph_20260729 \
  experiments/cuda_graph/results/mamba_moe_te_graph_20260729_report.html \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
git commit -s -m "experiments: add Mamba and MoE graph matrix"
```

### Task 9: Verify, Review, and Push the Three-Repository Stack

**Files:**
- Modify: `Megatron-Bridge/3rdparty/Megatron-LM` submodule pointer.
- Modify: `NeMo-RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` submodule pointer.

**Interfaces:**
- Consumes: all prior implementation commits.
- Produces: pushed MCore, Bridge, and NeMo-RL branches with reproducible pointers.

- [ ] **Step 1: Run focused Megatron-LM verification**

Run:

```bash
uv run python -m pytest -q \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py
uv run isort \
  megatron/core/packed_seq_params.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  megatron/core/transformer/transformer_layer.py
uv run pre-commit run --all-files
```

Expected: all tests and checks pass.

- [ ] **Step 2: Run focused NeMo-RL verification**

Run:

```bash
uv run python -m pytest -q \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
uv run pre-commit run --all-files
```

Expected: all tests and checks pass.

- [ ] **Step 3: Push Megatron-LM**

Run:

```bash
git push -u origin sj/pr5672-mamba-moe-graph-cache-20260729
git rev-parse HEAD
```

Record the full SHA in the experiment README.

- [ ] **Step 4: Update and push Megatron-Bridge**

In Megatron-Bridge:

```bash
git branch --show-current
git merge-base --is-ancestor upstream/main HEAD
git add 3rdparty/Megatron-LM
git commit -s -m "chore: pin MCore Mamba and MoE graph support"
git push -u origin sna/pr5672-mamba-moe-graph-cache-20260729
git rev-parse HEAD
```

- [ ] **Step 5: Update and push NeMo-RL**

In NeMo-RL:

```bash
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "chore: pin Mamba and MoE graph support"
git push -u seonjinn experiment/pr5672-mamba-moe-graph-cache-20260729
git rev-parse HEAD
git submodule status --recursive
```

Record all three SHAs in the experiment README and report.

### Task 10: Run the GB200 Correctness, Performance, and Accuracy Gates

**Files:**
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`
- Modify: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_report.html`
- Create: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_smoke.csv`
- Create: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_performance.csv`
- Create: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_accuracy.csv`

**Interfaces:**
- Consumes: pushed branches, nightly container, Nano model snapshot, ptyche or OCI-HSG.
- Produces: audited correctness, 20-step performance, and accuracy evidence.

- [ ] **Step 1: Read the SLURM skill and cluster reference**

Read completely:

```text
/Users/sna/.codex/plugins/cache/e2etrain-marketplace/e2etrain/1.0.0/skills/ssh-slurm/SKILL.md
/Users/sna/.claude/plugins/marketplaces/e2etrain-container-image-skill/plugins/e2etrain/skills/stage-training-containers/SKILL.md
/Users/sna/.Codex/docs/clusters/slurm/ptyche.md
```

Use the documented login, FairShare, `--test-only`, immutable nightly
container staging, and five-minute monitoring procedures.

- [ ] **Step 2: Submit preflight and all scope/variant five-step smokes**

Run test-only for every theoretical script. Classify model-incompatible scopes
using MCore validation and graph-layer discovery, then submit every
model-compatible job independently without dependencies:

```bash
CLUSTER=ptyche TEST_ONLY=1 \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_all_smokes.sh
CLUSTER=ptyche \
  bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_all_smokes.sh
```

Expected: baseline, all 32 theoretical TE scope rows, and targeted MoE
configuration variants have an audited preflight result. Submitted jobs use
fixed-rollout schedule `[5, 5, 5, 5, 3]`; TE jobs use warmup 3 and cache
capacity 2. Unsupported rows such as dense `mlp` on a model without dense
layers or full `moe` without drop-and-pad are recorded and not treated as
runtime failures.

- [ ] **Step 3: Monitor early failures**

For five minutes, verify Ray driver creation, model loading, step-one completion, and absence of import, auth, TE build, CUDA, NCCL, or graph-capture errors.

- [ ] **Step 4: Gate smoke correctness**

Every graph job must complete 5/5 with:

```text
fallback_count = 0
capture_count[5] = 1
capture_count[3] = 1
router top-k IDs exact
no NaN or Inf
```

Any wrong-N modulo replay, live MoE state, illegal memory access, or silent eager fallback blocks performance submission.

- [ ] **Step 5: Submit the selected 20-step paired matrix**

Run no-CG plus:

```text
attn
mlp
attn+mlp
mamba
attn+mamba
mamba+mlp
mamba+moe
attn+mamba+moe
mamba+moe_router
attn+mamba+moe_router
moe_router+moe_preprocess
mamba+moe_router+moe_preprocess
attn+mamba+moe_router+moe_preprocess
```

For the two best MoE graph boundaries, also run paired
shared-expert-overlap on/off and selective-`moe_act`-recompute on/off variants.

Use fixed-rollout schedule:

```text
[5,5,5,5,3,3,5,3,5,3,5,3,5,3,5,3,5,3,5,3]
```

Discard steps 1-5 and report steps 6-20. Require exactly one capture for keys 5 and 3 and zero later captures.

- [ ] **Step 6: Evaluate numerical accuracy**

Compare fixed-rollout eager and graph runs using:

```text
integer/routing/mask fields: exact
loss and KL: abs <= 2e-3 and rel <= 2e-3
grad norm relative delta: <= 1%
generation_kl_error: abs <= 2e-4 and rel <= 10%
held-out token NLL abs delta: <= 2e-3
held-out perplexity relative delta: <= 0.2%
top-1 token agreement: >= 99.9%
mean KL(eager || graph): <= 1e-4
p99 token KL: <= 1e-3
```

- [ ] **Step 7: Evaluate performance**

Across at least three paired repeats:

```text
correctness merge gate:
  median policy throughput >= 0.98x eager
  median E2E throughput >= 0.98x eager
  p95 policy train time <= 1.05x eager
  repeat CV <= 5%

performance claim gate:
  median policy throughput >= 1.05x eager
  paired/bootstrap 95% CI lower bound > 1.0
  E2E regression <= 2%
```

- [ ] **Step 8: Run model-breadth confirmation**

After the Nano hybrid correctness gate passes, run the no-CG baseline and the
best correct MoE scopes for:

```text
Qwen3-30B-A3B performance recipe: 20 steps
Qwen3-235B-A22B performance recipe: 20 steps
```

Qwen models have no Mamba layers, so an explicit `mamba` request must fail the
empty-match gate. Compare full `moe`, `moe_router`, and
`moe_router+moe_preprocess`, plus attention combinations, using the same
performance and numerical thresholds. Record resource or model-snapshot
blockers as unsupported evidence, never as a passing result.

- [ ] **Step 9: Render and commit the final report**

Run:

```bash
uv run python \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py
uv run python \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/render_report.py
git add \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md \
  experiments/cuda_graph/results/mamba_moe_te_graph_20260729_*.csv \
  experiments/cuda_graph/results/mamba_moe_te_graph_20260729_report.html
git commit -s -m "experiments: report Mamba and MoE graph results"
git push seonjinn experiment/pr5672-mamba-moe-graph-cache-20260729
```

Expected: the HTML distinguishes verified correctness, smoke-only evidence, performance results, accuracy results, failures, and unsupported cases.
