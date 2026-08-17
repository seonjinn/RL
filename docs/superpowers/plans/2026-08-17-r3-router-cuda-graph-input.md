# R3 Router CUDA Graph Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rollout-selected Router Replay expert IDs explicit fixed-signature Tensor inputs to Transformer Engine partial CUDA Graphs that own `moe_router`, while preserving differentiable current-policy routing scores and failing before launch on invalid state.

**Architecture:** Megatron-Core exposes a versioned capability and threads one layer-local route Tensor through the existing TE Tensor-kwarg surface; TE owns the captured static address and copies changing route values before each replay. NeMo-RL keeps ownership of full-model route transport, packing, CP identity, lifecycle, and trace evidence, and enables R3 plus graph-owned router only when the exact MCore capability and tested Nano contract are present. The existing eager R3 probability helper remains the single mathematical implementation.

**Tech Stack:** Python 3.13, PyTorch 2.11, Transformer Engine `04a76c84423d9a4eb2f2010ef6692e347326cc00`, Megatron-Core, Megatron-Bridge, NeMo-RL, HybridEP, packed THD, pytest, uv, TensorBoard, SLURM, and GB200.

**Spec:** `docs/superpowers/specs/2026-08-17-r3-router-cuda-graph-input-design.md`

## Global Constraints

- Capability name is exactly `r3_router_cuda_graph_input_v1`.
- The initial supported row is BF16 Nano TP2/PP2/CP2/EP8, HybridEP, packed THD, with `moe_router` or `attn,mamba,moe_router` capture.
- `moe_preprocess`, whole-MoE capture, FP8, NVFP4, reference-logprob graphs, and fused router top-k remain fail-closed.
- Graph input shape is `[local_fixed_tokens, topk]`, dtype is `torch.long`, and route values never participate in graph-bank selection.
- Structural padding rows use `torch.arange(topk)`; logical rows containing the all-`-1` missing-route sentinel are rejected before graph entry.
- Validation finishes before manual hooks, bank entry, or graph launch. There is no silent fallback inside replay.
- The existing `topk_routing_with_score_function` implementation remains the authority for score function, normalization, scaling, and router gradients.
- Production changes follow TDD. MCore commits use `git commit -s -S`; NeMo-RL and integration commits use `git commit -s`.
- Commit and push MCore first, then create the pinned Bridge integration commit, then commit and push NeMo-RL.
- GPU jobs require a clean immutable source snapshot, exact runtime attestation, `sbatch --test-only`, one actual submission, and at least five minutes of monitoring.

## Planned Interfaces

### Megatron-Core route contract

```python
ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY = "r3_router_cuda_graph_input_v1"
ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG = "router_replay_indices"


@dataclass(frozen=True)
class RouterReplayCudaGraphInputSignature:
    shape: tuple[int, int]
    dtype: torch.dtype
    device_type: str
    topk: int
    num_experts: int


def validate_router_replay_cuda_graph_input(
    indices: torch.Tensor,
    *,
    structural_padding_mask: torch.Tensor,
    expected_tokens: int,
    topk: int,
    num_experts: int,
) -> RouterReplayCudaGraphInputSignature:
    if indices.dtype != torch.long or indices.dim() != 2 or not indices.is_contiguous():
        raise TypeError("router replay CUDA graph indices must be contiguous torch.long [T, K]")
    if tuple(indices.shape) != (expected_tokens, topk):
        raise ValueError("router replay CUDA graph input shape differs from fixed capacity")
    structural = structural_padding_mask.reshape(-1).to(device=indices.device, dtype=torch.bool)
    if structural.numel() != expected_tokens:
        raise ValueError("structural padding mask differs from route token capacity")
    logical = indices[~structural]
    if logical.numel() and bool(logical.lt(0).any().item()):
        raise ValueError("logical route row contains a missing-route sentinel")
    if logical.numel() and bool(logical.ge(num_experts).any().item()):
        raise ValueError("logical route expert id is outside the expert range")
    if logical.shape[-1] > 1 and bool(
        logical.sort(dim=-1).values.diff(dim=-1).eq(0).any().item()
    ):
        raise ValueError("logical route row contains duplicate expert ids")
    dummy = torch.arange(topk, device=indices.device, dtype=indices.dtype)
    structural_count = int(structural.sum().item())
    if structural_count and not torch.equal(
        indices[structural], dummy.expand(structural_count, -1)
    ):
        raise ValueError("structural route row differs from the canonical dummy route")
    return RouterReplayCudaGraphInputSignature(
        shape=(expected_tokens, topk),
        dtype=indices.dtype,
        device_type=indices.device.type,
        topk=topk,
        num_experts=num_experts,
    )
```

The pure validator accepts only a two-dimensional contiguous `torch.long`
Tensor and records its device type so CPU unit tests can exercise the contract.
The TransformerLayer graph boundary additionally requires CUDA. Validation
checks exact token capacity, top-k, range and per-row uniqueness; logical rows
may not contain `-1`, while every structural padding row must equal
`torch.arange(topk)`. The structural mask is flattened in the same
sequence-major order as `_normalize_routed_experts_for_mcore`.

### Transformer Engine Tensor kwarg

```python
def _get_te_cuda_graph_router_replay_input(
    self,
    hidden_states: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    replay = self.mlp.router.router_replay
    indices = replay.target_topk_idx
    if indices is None:
        raise RuntimeError("router replay CUDA graph input is missing")
    if not indices.is_cuda:
        raise RuntimeError("router replay CUDA graph input must be a CUDA Tensor")
    validate_router_replay_cuda_graph_input(
        indices,
        structural_padding_mask=padding_mask.transpose(0, 1).reshape(-1),
        expected_tokens=hidden_states.shape[0] * hidden_states.shape[1],
        topk=self.config.moe_router_topk,
        num_experts=self.config.num_moe_experts,
    )
    return indices
```

`TransformerLayer._get_te_cuda_graph_replay_args` inserts the returned Tensor
under `router_replay_indices` only when the layer is MoE, Router Replay is
enabled, and the selected graph scope owns `moe_router`. The capture sample is
a fixed-capacity canonical dummy route. `_te_cuda_graph_capture` consumes the
kwarg and installs it only for the duration of captured router computation.
TE's existing graphed-callable input surface owns the captured address and
copies new caller values on replay.

### NeMo-RL capability gate

```python
def resolve_router_replay_cuda_graph_capability() -> str | None:
    try:
        from megatron.core.transformer.moe.router_replay import (
            ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY,
        )
    except ImportError:
        return None
    return (
        ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY
        if type(ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY) is str
        else None
    )


def validate_router_replay_cuda_graph_scope(
    *,
    enabled: bool,
    cuda_graph_impl: object,
    cuda_graph_modules: object,
    runtime_capability: str | None,
    validation_enabled: bool,
    router_fusion: bool,
    fixed_thd_capacity: bool,
    bf16: bool,
    hybridep: bool,
) -> None:
    if not enabled or cuda_graph_impl == "none":
        return
    if isinstance(cuda_graph_modules, str):
        modules = {item.strip() for item in cuda_graph_modules.split(",") if item.strip()}
    else:
        modules = {
            getattr(item, "name", item) for item in (cuda_graph_modules or ())
        }
    captures_router = not modules or bool(
        modules.intersection({"moe", "moe_router", "moe_preprocess"})
    )
    if not captures_router:
        return
    if modules not in (
        {"moe_router"},
        {"attn", "mamba", "moe_router"},
    ):
        raise ValueError("R3 v1 supports only tested partial graph-owned router scopes")
    if runtime_capability != "r3_router_cuda_graph_input_v1":
        raise ValueError("runtime lacks r3_router_cuda_graph_input_v1")
    if not validation_enabled:
        raise ValueError("R3 router CUDA graph input requires route validation")
    if router_fusion:
        raise ValueError("R3 router CUDA graph input does not support router fusion")
    if not fixed_thd_capacity:
        raise ValueError("R3 router CUDA graph input requires fixed THD capacity")
    if not bf16 or not hybridep:
        raise ValueError("R3 router CUDA graph input v1 requires BF16 HybridEP")
```

The gate accepts graph-owned router scopes only when the runtime capability is
exact, validation is enabled, router fusion is disabled, BF16 HybridEP is in
use, and fixed THD capacity is configured. All existing eager-router R3 scopes
remain unchanged.

---

### Task 1: Add the MCore Route-Input Contract and Failing Unit Tests

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/router_replay.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_router_replay.py`

**Interfaces:**
- Consumes: existing `RouterReplayAction`, `RouterReplay.get_replay_topk`, and `topk_routing_with_score_function`.
- Produces: `ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY`, `ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG`, `RouterReplayCudaGraphInputSignature`, `validate_router_replay_cuda_graph_input`, and a capture-only context manager `RouterReplay.use_cuda_graph_input(indices)`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_validate_router_replay_cuda_graph_input_accepts_exact_contract():
    indices = torch.tensor([[0, 3], [1, 2], [0, 1]], dtype=torch.long)
    structural = torch.tensor([False, False, True])
    signature = validate_router_replay_cuda_graph_input(
        indices,
        structural_padding_mask=structural,
        expected_tokens=3,
        topk=2,
        num_experts=4,
    )
    assert signature.shape == (3, 2)
    assert signature.topk == 2


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (torch.tensor([[0, 0], [0, 1]]), "duplicate"),
        (torch.tensor([[-1, -1], [0, 1]]), "missing-route"),
        (torch.tensor([[0, 4], [0, 1]]), "outside"),
        (torch.tensor([[0, 1], [1, 2]]), "structural dummy"),
    ],
)
def test_validate_router_replay_cuda_graph_input_rejects_invalid_rows(indices, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_router_replay_cuda_graph_input(
            indices.long(),
            structural_padding_mask=torch.tensor([False, True]),
            expected_tokens=2,
            topk=2,
            num_experts=4,
        )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run from the nested MCore root:

```bash
uv run pytest -q tests/unit_tests/transformer/moe/test_router_replay.py \
  -k 'cuda_graph_input'
```

Expected: collection or import failure because the contract symbols do not yet exist.

- [ ] **Step 3: Implement the strict contract and capture-only context**

```python
@contextmanager
def use_cuda_graph_input(self, indices: torch.Tensor) -> Iterator[None]:
    previous_target = self.target_topk_idx
    previous_action = self.router_replay_action
    self.target_topk_idx = indices
    self.router_replay_action = RouterReplayAction.REPLAY_FORWARD
    try:
        yield
    finally:
        self.target_topk_idx = previous_target
        self.router_replay_action = previous_action
```

The context must not append to `replay_backward_list`. Replace existing
`assert` statements touched by this contract with explicit exceptions.

- [ ] **Step 4: Add semantic and lifetime tests**

```python
def test_cuda_graph_input_context_preserves_router_gradients_and_fifo():
    replay = RouterReplay()
    logits = torch.randn(3, 4, requires_grad=True)
    routes = torch.tensor([[0, 2], [1, 3], [0, 1]])
    before = list(replay.replay_backward_list)
    with replay.use_cuda_graph_input(routes):
        probs, _ = topk_routing_with_score_function(
            logits=logits,
            topk=2,
            router_replay=replay,
            score_function="softmax",
        )
        probs.sum().backward()
    assert logits.grad is not None
    assert replay.replay_backward_list == before
    assert replay.target_topk_idx is None
    assert replay.router_replay_action is None
```

- [ ] **Step 5: Run MCore router tests and formatting**

```bash
uv run pytest -q tests/unit_tests/transformer/moe/test_router_replay.py
uv run isort megatron/core/transformer/moe/router_replay.py \
  tests/unit_tests/transformer/moe/test_router_replay.py
git diff --check
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit the MCore contract**

```bash
git add megatron/core/transformer/moe/router_replay.py \
  tests/unit_tests/transformer/moe/test_router_replay.py
git commit -s -S -m "feat: define router replay CUDA graph inputs"
```

### Task 2: Thread Route IDs Through the TE Graph Input Surface

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/module.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/te_cuda_graph_bank.py`
- Create: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_router_replay_cuda_graph.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: Task 1's kwarg name, validator, signature, and capture-only context.
- Produces: one fixed-signature route kwarg per graph-owned router leaf,
  `router_replay_input_signatures` in `TECudaGraphBankFingerprint`, and a
  post-success graph-launch record that identifies bank/graph index/copy
  generation without exposing TE-private static buffers.

- [ ] **Step 1: Write RED tests for pre-hook validation and dynamic-value reuse**

```python
def test_router_replay_graph_rejects_missing_input_before_manual_hooks(layer, monkeypatch):
    calls = {"hook": 0, "graph": 0}
    layer.cuda_graph_manual_hooks = [lambda: calls.__setitem__("hook", calls["hook"] + 1)]
    layer.cuda_graphs = [lambda *args, **kwargs: calls.__setitem__("graph", calls["graph"] + 1)]
    with pytest.raises(RuntimeError, match="router replay CUDA graph input"):
        layer._te_cuda_graph_replay(hidden_states=torch.randn(4, 1, 8))
    assert calls == {"hook": 0, "graph": 0}


def test_router_replay_graph_values_change_without_signature_change(layer):
    first = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    second = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)
    first_signature = layer._validate_te_cuda_graph_router_replay_input(first)
    second_signature = layer._validate_te_cuda_graph_router_replay_input(second)
    assert first_signature == second_signature
```

- [ ] **Step 2: Run the new graph tests and confirm RED**

```bash
uv run pytest -q tests/unit_tests/transformer/test_router_replay_cuda_graph.py
```

Expected: failures for missing TransformerLayer route-input methods.

- [ ] **Step 3: Add the sample input and replay kwarg**

Implement these concrete rules:

```python
if self._te_cuda_graph_owns_router_replay_input():
    local_tokens = micro_batch_size * slen_per_cptp
    static_inputs[ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG] = torch.arange(
        self.config.moe_router_topk,
        dtype=torch.long,
        device=torch.cuda.current_device(),
    ).expand(local_tokens, -1).clone()
```

At replay, obtain the current layer-local `target_topk_idx`, validate it
against `hidden_states` and `padding_mask`, then insert it into
`cudagraph_kwargs` before `super()._te_cuda_graph_replay`. At capture, pop the
kwarg before attention/MLP forwarding and wrap only router computation in
`use_cuda_graph_input`.

- [ ] **Step 4: Add first-class bank signature ownership**

```python
@dataclass(frozen=True)
class TECudaGraphBankFingerprint:
    # existing fields remain unchanged
    router_replay_input_signatures: Sequence[
        tuple[int, RouterReplayCudaGraphInputSignature | None]
    ]
```

Snapshot, install, validate, rollback, and clear the corresponding layer
contract attribute transactionally. Include signature only, never Tensor
contents or route digest.

- [ ] **Step 5: Test stable TE addresses with changing caller values**

Use the existing fake graphed-callable fixtures in `test_cuda_graphs.py` to
record the sample kwarg address and two replay values. Assert one capture,
stable static address, second-value visibility, and no recapture.

- [ ] **Step 6: Test fusion and unsupported scopes fail closed**

```python
def test_router_replay_cuda_graph_rejects_fused_router(config):
    config.moe_enable_routing_replay = True
    config.moe_router_fusion = True
    config.cuda_graph_modules = [CudaGraphModule.moe_router]
    with pytest.raises(ValueError, match="moe_router_fusion"):
        build_transformer_layer(config)
```

- [ ] **Step 7: Run focused bank, helper, and graph tests**

```bash
uv run pytest -q tests/unit_tests/transformer/test_router_replay_cuda_graph.py
uv run pytest -q tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  -k 'fingerprint or activation or rollback or capture_failure'
uv run pytest -q tests/unit_tests/transformer/test_cuda_graphs.py \
  -k 'router or moe or static'
uv run isort megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/module.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_router_replay_cuda_graph.py
git diff --check
```

- [ ] **Step 8: Commit the MCore graph surface**

```bash
git add megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/module.py \
  megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_router_replay_cuda_graph.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -S -m "feat: replay router IDs through TE graphs"
```

### Task 3: Prove MCore Distributed Output and Gradient Parity

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interfaces:**
- Consumes: Tasks 1-2 route kwarg and bank contract.
- Produces: typed MCore row `dropless_hybridep_nano16_r3_router_graph` with eager-versus-graph parity evidence.

- [ ] **Step 1: Add a failing distributed test mode**

Extend the synthetic Nano model so every replay supplies changed valid routes
with identical physical signature. Use three eager warmups, one capture, and
20 replays. Add typed shared-expert-on and shared-expert-off variants rather
than relying on `MCORE_TEST_DISABLE_NANO_SHARED_EXPERT` ambient state.

```python
assert torch.equal(eager_route_ids, graph_route_ids)
torch.testing.assert_close(graph_output, eager_output, rtol=5e-2, atol=5e-2)
torch.testing.assert_close(graph_loss, eager_loss, rtol=5e-2, atol=5e-2)
for name in eager_parameter_grads:
    torch.testing.assert_close(
        graph_parameter_grads[name], eager_parameter_grads[name], rtol=5e-2, atol=5e-2
    )
assert graph_counters.fallback_count == 0
assert graph_counters.graph_calls == graph_counters.eligible_calls
```

- [ ] **Step 2: Add negative phases**

Parameterize missing `-1`, duplicate, out-of-range, wrong token capacity, and
stale microbatch generation. Each phase must raise before the graph-call spy.

- [ ] **Step 3: Register the typed test row and launcher unit test**

Add the exact row to `mcore_test_matrix.json`; verify the rendered command uses
WORLD_SIZE=16 and the intended test selector without ambient diagnostic flags.

- [ ] **Step 4: Run local source tests**

```bash
uv run pytest -q tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'r3 and router and graph'
python -m py_compile \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py
git diff --check
```

- [ ] **Step 5: Commit MCore distributed coverage and root harness metadata separately**

Commit the MCore test with `-s -S`. Commit root matrix/launcher tests with
`-s`; do not mix the two repository histories.

### Task 4: Replace NeMo-RL's Blanket Gate with the Versioned Capability Gate

**Files:**
- Modify: `nemo_rl/models/megatron/router_replay.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Modify: `tests/unit/models/megatron/test_router_replay.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interfaces:**
- Consumes: MCore capability constant from Task 1.
- Produces: exact capability resolution and supported/unsupported configuration validation.

- [ ] **Step 1: Write RED gate tests**

```python
def test_router_replay_allows_router_graph_with_exact_runtime_capability(monkeypatch):
    monkeypatch.setenv("NRL_ROUTER_REPLAY_VALIDATE", "1")
    validate_router_replay_cuda_graph_scope(
        enabled=True,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["moe_router"],
        runtime_capability="r3_router_cuda_graph_input_v1",
        validation_enabled=True,
        router_fusion=False,
        fixed_thd_capacity=True,
        bf16=True,
        hybridep=True,
    )


@pytest.mark.parametrize(
    ("capability", "validation", "fusion", "fixed", "bf16", "hybridep", "message"),
    [
        (None, True, False, True, True, True, "r3_router_cuda_graph_input_v1"),
        ("r3_router_cuda_graph_input_v1", False, False, True, True, True, "validation"),
        ("r3_router_cuda_graph_input_v1", True, True, True, True, True, "fusion"),
        ("r3_router_cuda_graph_input_v1", True, False, False, True, True, "fixed THD"),
        ("r3_router_cuda_graph_input_v1", True, False, True, False, True, "BF16"),
        ("r3_router_cuda_graph_input_v1", True, False, True, True, False, "HybridEP"),
    ],
)
def test_router_replay_router_graph_gate_fails_closed(
    capability, validation, fusion, fixed, bf16, hybridep, message
):
    with pytest.raises(ValueError, match=message):
        validate_router_replay_cuda_graph_scope(
            enabled=True,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=["moe_router"],
            runtime_capability=capability,
            validation_enabled=validation,
            router_fusion=fusion,
            fixed_thd_capacity=fixed,
            bf16=bf16,
            hybridep=hybridep,
        )
```

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  -k 'router_replay and cuda_graph'
```

- [ ] **Step 3: Implement capability resolution and both validation boundaries**

The early policy-config validator checks the imported MCore constant and
requested fields. `_apply_performance_config` repeats the check after provider
defaults are materialized, so inherited graph ownership cannot bypass it.
Retain unconditional rejection for `moe`, `moe_preprocess`, empty whole-layer
scope, fusion, non-BF16, and missing fixed THD geometry.

The persistent launcher accepts the two v1 scopes only when the selected
runtime attestation is feature-bound to
`dropless_hybridep_nano16_r3_router_graph_v1`; otherwise TEST_ONLY exits before
printing an SBATCH command. Preserve exact rejection for every other
router-owning scope.

- [ ] **Step 4: Run gate and setup tests**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  -k 'router_replay or fixed_te_graph'
uv run pytest -q tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'router_replay and graph'
uv run ruff check nemo_rl/models/megatron/router_replay.py \
  nemo_rl/models/megatron/setup.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_setup.py
git diff --check
```

- [ ] **Step 5: Commit the NeMo-RL gate**

```bash
git add nemo_rl/models/megatron/router_replay.py \
  nemo_rl/models/megatron/setup.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "feat: gate router replay graph inputs"
```

### Task 5: Bind Current-Microbatch Routes to Training and Abort Lifecycles

**Files:**
- Modify: `nemo_rl/models/megatron/router_replay.py`
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/utils/r3_trace.py`
- Modify: `tests/unit/models/megatron/test_router_replay.py`
- Modify: `tests/unit/models/megatron/test_megatron_data.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`
- Modify: `tests/unit/models/policy/test_megatron_split_state.py`
- Modify: `tests/unit/utils/test_r3_trace_semantics.py`

**Interfaces:**
- Consumes: Task 4 gate and MCore route-input contract.
- Produces: one current-microbatch route generation, graph-consumer trace record, and deterministic cleanup on success or abort.

- [ ] **Step 1: Add failing lifecycle tests**

```python
def test_graph_router_replay_replaces_values_for_each_microbatch():
    replay = RouterReplay()
    router = SimpleNamespace(router_replay=replay, layer_number=1)
    model = SimpleNamespace(
        config=SimpleNamespace(num_layers=1, moe_layer_freq=[1]),
        modules=lambda: [router],
    )
    first = torch.tensor([[[0, 1]], [[2, 3]]], dtype=torch.int32)
    second = torch.tensor([[[1, 2]], [[0, 3]]], dtype=torch.int32)
    set_router_replay_forward(model, first, microbatch_generation=7)
    assert torch.equal(replay.target_topk_idx, first[:, 0].long())
    clear_router_replay(model)
    set_router_replay_forward(model, second, microbatch_generation=8)
    assert torch.equal(replay.target_topk_idx, second[:, 0].long())
    assert replay.graph_input_generation == 8


def test_aborted_train_step_clears_graph_route_generation(worker):
    worker.begin_train_step(loss_fn=worker._test_loss_fn, gbs=16, mbs=1)
    worker._active_router_route_generation = 3
    worker.abort_train_step()
    assert worker._active_router_route_generation is None
```

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/policy/test_megatron_split_state.py \
  -k 'graph and route'
```

- [ ] **Step 3: Add typed current-microbatch identity**

Extend `set_router_replay_forward` with keyword-only
`microbatch_generation: int`. Store the generation on every assigned replay
instance, reject reuse or regression, and trace the source/consumer digest.
Extend `_TECudaGraphCallState` and the first-microbatch preflight with the route
signature and generation. Validate routes before `_ensure_te_cuda_graph_schedule`
can activate a bank; the current later `train.py` installation remains the
per-microbatch value handoff, not the first validity check. `clear_router_replay`
clears action, indices, generation, and graph-consumer state. Both
`megatron_forward_backward` finally blocks, collective graph-failure cleanup,
normal split completion, and split-worker abort paths must call it.

- [ ] **Step 4: Emit graph-consumer evidence**

Add an R3 trace record containing stage, action, layer number, payload index,
route digest, physical signature, bank ID, graph index, schedule key, copy
generation, successful graph launch, and capability version. Do not serialize
token or prompt contents. TE static-address ownership is proved by Task 2's
tests rather than by exposing TE-private buffers. Reduce produced/copied/
launched and every unsafe counter across TP/CP/PP/DP; any nonzero missing,
stale, malformed, range, duplicate, or CP-mismatch counter blocks promotion.

- [ ] **Step 5: Run route, split-state, and trace tests**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_megatron_split_state.py \
  tests/unit/utils/test_r3_trace_semantics.py
uv run ruff check nemo_rl/models/megatron/router_replay.py \
  nemo_rl/models/megatron/data.py \
  nemo_rl/models/megatron/train.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/utils/r3_trace.py
git diff --check
```

- [ ] **Step 6: Commit lifecycle and telemetry**

```bash
git add nemo_rl/models/megatron/router_replay.py \
  nemo_rl/models/megatron/data.py \
  nemo_rl/models/megatron/train.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/utils/r3_trace.py \
  tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_megatron_split_state.py \
  tests/unit/utils/test_r3_trace_semantics.py
git commit -s -m "feat: bind replay routes to graph microbatches"
```

### Task 6: Verify Packing, CP Identity, and Full Trace Completeness

**Files:**
- Modify: `tests/unit/models/megatron/test_megatron_data.py`
- Modify: `tools/check_r3_trace.py`
- Modify: `tests/unit/experiments/test_r3_trace_checker.py`
- Modify only if a defect is exposed: `nemo_rl/models/megatron/data.py`

**Interfaces:**
- Consumes: existing route packing and CP helpers plus Task 5 graph trace records.
- Produces: fail-closed proof that rollout, eager prev-logprob, and graph training consumed the same routes and token identity.

- [ ] **Step 1: Add MBS2 fixed-capacity packing tests**

Build two sequences with different logical lengths, CP2 zigzag sharding, one
full-layer payload, and structural tail capacity. Assert exact token/route row
alignment, canonical structural dummy routes, and unchanged expert zero.

- [ ] **Step 2: Add checker adversarial tests**

Reject missing graph-consumer records, stale microbatch generation, wrong
layer/payload mapping, digest mismatch, malformed signature, and graph records
for only a subset of required stages.

- [ ] **Step 3: Run focused data and checker tests**

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_data.py \
  -k 'routed_experts or router_replay'
uv run pytest -q tests/unit/experiments/test_r3_trace_checker.py
uv run ruff check tools/check_r3_trace.py \
  tests/unit/experiments/test_r3_trace_checker.py
git diff --check
```

- [ ] **Step 4: Commit trace completeness**

```bash
git add tests/unit/models/megatron/test_megatron_data.py \
  tools/check_r3_trace.py tests/unit/experiments/test_r3_trace_checker.py
git commit -s -m "test: verify graph router replay identity"
```

### Task 7: Add Frozen-Batch Same-Worker Parity

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_r3_router_graph_parity.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/diagnostics/submit_r3_router_graph_parity.sh`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/policy/test_megatron_split_state.py`

**Interfaces:**
- Consumes: Tasks 1-6 complete route contract and lifecycle.
- Produces: diagnostic-only API that runs eager R3 and graph R3 from identical live-worker state without optimizer/scheduler advancement.

- [ ] **Step 1: Write worker API tests**

Define diagnostic methods to snapshot/restore RNG, zero existing grad storage
in place, execute one prepared microbatch without optimizer step, and return
loss plus local output/input/parameter-grad hashes and selected Tensor values.

- [ ] **Step 2: Implement the parity driver**

The driver loads one frozen `train_data_step*.jsonl`, constructs one packed
batch, runs eager then graph arms on the same worker state, and writes one
immutable JSON artifact. Require exact route/token digests and compare loss,
outputs, input grads, all parameter grads, and simulated parameter deltas at
`rtol=atol=5e-2`.

- [ ] **Step 3: Add launcher gates**

The submitter requires exact source/profile/runtime attestation, Router Replay
validation, capability v1, Triton vLLM, fixed THD capacity, no dependency, and
one 16-GPU allocation. `TEST_ONLY=1` and `SBATCH_TEST_ONLY=1` must render the
same payload.

- [ ] **Step 4: Run unit tests and shell validation**

```bash
uv run pytest -q tests/unit/models/policy/test_megatron_split_state.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'r3_router_graph_parity'
bash -n experiments/cuda_graph/nemotron_thd_te_graph_20260731/diagnostics/submit_r3_router_graph_parity.sh
python -m py_compile experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_r3_router_graph_parity.py
git diff --check
```

- [ ] **Step 5: Commit the parity harness**

```bash
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_r3_router_graph_parity.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/diagnostics/submit_r3_router_graph_parity.sh \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_split_state.py
git commit -s -m "test: add frozen router graph parity"
```

### Task 8: Bind the Capability into Runtime Stage and Attestation

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_container_runtime.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py`
- Modify: `tests/unit/experiments/test_runtime_attestation.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Interfaces:**
- Consumes: MCore capability constant and NeMo launcher feature request.
- Produces: runtime feature `dropless_hybridep_nano16_r3_router_graph_v1` whose passed JSON proves the exact capability import from the staged MCore Python.

- [ ] **Step 1: Write failing attestation tests**

Add fixtures where the staged MCore capability is exact, absent, malformed, or
wrong-version. Require the v1 feature to pass only the exact case. Verify the
leaf verifier rejects an older `dropless_hybridep_nano16` attestation before
SBATCH rendering.

```python
assert artifact["runtime_feature_set"] == (
    "dropless_hybridep_nano16_r3_router_graph_v1"
)
assert artifact["mcore_capabilities"]["router_replay_cuda_graph_input"] == (
    "r3_router_cuda_graph_input_v1"
)
```

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'r3_router or runtime_feature'
```

- [ ] **Step 3: Probe and verify the staged capability**

The runtime validator imports the constant using the staged MCore Python in an
isolated subprocess, writes it to the immutable JSON, and compares it to the
requested v1 feature. The verifier checks exact value, exact staged Python,
source SHAs, feature set, exclusions, and container digest. Missing or unknown
fields fail closed; no legacy default is accepted for this feature.

- [ ] **Step 4: Run the full attestation and launcher unit suites**

```bash
uv run pytest -q tests/unit/experiments/test_runtime_attestation.py
uv run pytest -q tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'runtime or r3_router or router_replay'
bash -n experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub
python -m py_compile \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_container_runtime.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py
git diff --check
```

- [ ] **Step 5: Commit the feature-bound runtime contract**

```bash
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_container_runtime.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "feat: attest R3 router graph capability"
```

### Task 9: Integrate Repository Pins and Publish Reviewable Commits

**Files:**
- Modify in Bridge integration branch: `3rdparty/Megatron-LM`
- Modify in NeMo-RL branch: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Modify: `docs/superpowers/plans/2026-08-17-r3-router-cuda-graph-input.md`

**Interfaces:**
- Consumes: all local passing commits.
- Produces: remotely reachable MCore, Bridge integration, and NeMo-RL SHAs with clean recursive gitlinks.

- [ ] **Step 1: Run final targeted CPU tests**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_split_state.py \
  tests/unit/experiments/test_r3_trace_checker.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
uv run ruff check nemo_rl tests/unit tools/check_r3_trace.py
git diff --check
```

- [ ] **Step 2: Request independent code review**

Use a reviewer agent on the exact MCore and NeMo diffs. Resolve every
blocker/high/medium correctness finding and rerun the affected tests.

- [ ] **Step 3: Push MCore and create a draft MCore PR**

Push the signed MCore commits to the personal fork branch. Verify the remote
tip equals the local SHA and open a draft PR against the correct upstream
branch. Do not push a Bridge PR for the MCore implementation.

- [ ] **Step 4: Create the Bridge integration pin**

On a dedicated experimental Bridge branch, update only the MCore gitlink to
the pushed MCore candidate, commit with `-s`, push, and verify remote reachability.

- [ ] **Step 5: Pin Bridge in NeMo-RL and push**

Update only the Bridge gitlink plus completed NeMo-RL changes. Verify recursive
submodule status is exact and clean, then push `sj/r3-cg-router-input`.

### Task 10: Run the Correctness and Performance Promotion Ladder

**Files:**
- Modify after results: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/report_context.json`
- Modify after results: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/timeline.md`
- Modify after results: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`

**Interfaces:**
- Consumes: exact clean source snapshot and fresh runtime attestation.
- Produces: immutable MCore parity, frozen-batch parity, 5/20/100-step correctness, and cache-hit performance evidence.

- [ ] **Step 1: Stage and attest the exact source**

Run `TEST_ONLY=1`, then `SBATCH_TEST_ONLY=1`, then one CPU stage. Require
`COMPLETED|0:0`, exact marker hash, exact `stage-job-id`, read-only verification,
then one 4-GPU attestation with passed JSON and exact source/MCore/TE identities.

- [ ] **Step 2: Run the 16-GPU MCore distributed row**

Require exact output/loss/route/input-grad/all-parameter-grad/parameter-delta
parity, 20 changed-route replays, 100% graph coverage, zero fallback,
recapture, eviction, and unsafe route events.

- [ ] **Step 3: Run frozen-batch same-worker parity**

Require identical state/RNG/token/route digests and the established BF16
`5e-2` envelope for every compared Tensor and simulated update.

- [ ] **Step 4: Run two five-step Nano smokes**

Submit `moe_router` and `attn,mamba,moe_router` with R3 on, Triton vLLM,
padding=true, warmup=3, cache=4, TensorBoard on, W&B off. Monitor each for at
least five minutes and through terminal status. Reject any unmasked finite
outlier, fallback, recapture, or missing graph consumer trace.

- [ ] **Step 5: Run paired 20-step performance jobs**

Use the same frozen rollout/state/RNG for eager R3 control and graph R3 arms.
Report capture steps separately. Promotion requires correctness-clean
cache-hit policy throughput above eager; E2E throughput remains a separate
metric.

- [ ] **Step 6: Run the selected 100-step soak**

Require zero unsafe route events, zero fallback/eviction/recapture, complete
R3 trace evidence, finite metrics, no masked correctness failures, and stable
cache-hit policy speedup.

- [ ] **Step 7: Update the report with exact evidence**

Record job IDs, source/runtime SHAs, frozen-batch digest, graph telemetry,
correctness maxima, cache-hit timing/TPS, and explicit limitations. Do not
promote independent stochastic-run deltas as causal speedups.
