# Semantic Precision Policy and Transactional Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one positive semantic precision policy that configures MXFP8/BF16 training and rollout, drives mixed-layout refit without name/dtype guesses, fails the whole launcher immediately on refit failure, and preserves the fastest correct refit path.

**Architecture:** A typed policy compiler resolves semantic roles against a complete graph-manifest bundle and emits immutable graph precision intents before backend construction. Megatron/Transformer Engine and versioned vLLM endpoint adapters realize and bind those intents; the refit planner then produces canonical source, wire, destination, and transaction-group plans. A transactional engine transfers canonical components, finalizes each physical owner once, and commits a new generation weight version only after every mutable main/MTP/draft graph and rank is ready.

**Tech Stack:** Python 3.13.13, Pydantic v2, frozen dataclasses, PyTorch, Ray, Megatron-Core/Bridge, Transformer Engine, vLLM 0.25.1 and 0.28.0, ModelOpt MXFP8, FlashInfer TRTLLM, pytest, Pyrefly, Ruff, MyST.

**Spec:** `docs/design-docs/semantic-precision-refit.md`

## Global Constraints

- Implement from immutable baseline `4601ba2c646ec40e5928c780fc0051a842328eba` on branch `codex/refit-semantic-policy-v2-20260903`; do not update any existing pull-request head while this plan is under validation.
- The public schema is version 1, `default` is always `bf16`, and the common recipe is a positive allow-list. Raw checkpoint/runtime parameter patterns are not a stable user interface.
- The same compiled intent group is the only source of truth for training, wire, and rollout precision. Generated TE matchers or vLLM include data are derived artifacts and must pass exact realized-module validation.
- `moe.routed_expert` means only main text-decoder routed expert gate/up/down kernels. It excludes shared experts, routers, latent projections, bias, MTP/draft graphs, attention, and embeddings.
- `attention.qkvo` means only main text-decoder token-attention Q/K/V/O projection kernels. It excludes MLA, KDA/GDN, sparse indexers, output gates, vision, bias, and MTP/draft graphs.
- Layer coordinates are zero-based. The default index space is `global_decoder`; `moe_ordinal` is explicit. `exclude_first` and `exclude_last` count in the selected index space and cannot consume the full domain.
- Every mutable main-model tensor is accounted for and refitted. `out_of_scope` is allowed only for source-proven frozen parameters, immutable auxiliary models, or backend-owned derived state with a typed reason.
- MTP and speculative drafters are separate semantic graphs. Static auxiliary graphs require immutable revision evidence; co-trained MTP/drafters inherit BF16 unless explicitly selected and must commit atomically with the main model.
- A fused physical owner is atomic. Conflicting precision fails by default; explicit expansion computes a transitive closure and may not cross an explicit BF16 layer boundary.
- Equal dtypes do not imply compatible layouts. Direct copy requires identical complete format/layout descriptors.
- Canonical load components remain distinct from padded, permuted, fused, shuffled, or flattened execution storage. A dirty owner is finalized exactly once per transaction.
- vLLM-specific imports and capability probes live only in versioned endpoint adapters. Unsupported versions or missing public capabilities fail before model construction; there is no process-global MXFP8 monkey patch.
- A refit worker returns a typed result. `None`, `False`, malformed results, exceptions, timeouts, and missing acknowledgements are failures.
- A detected refit failure keeps generation quiesced, poisons any partially updated destination, aborts communicators or terminates their owning workers within a bounded teardown budget, preserves the original phase/rank/cause, and makes sync and async launchers exit non-zero.
- Preserve the direct compatible-component path, persistent buffers, cached routes/permutations, batched expert conversion, and overlap. Do not scan model names or rebuild the semantic plan on each refit.
- The 95% upper confidence bound for treatment/baseline refit p50 and p95 latency is at most 1.05. Post-refit generation latency is at most 1.05 and throughput is at least 0.95 of the fastest correct baseline.
- Production end-to-end coverage includes Qwen3-30B-A3B, Qwen3.5-35B-A3B, NVIDIA Nemotron 3.5 Lightning 30B-A3B, Nemotron3 Super, and Nemotron3 Ultra for BF16-training→MXFP8-rollout and MXFP8-training→MXFP8-rollout with BF16 boundaries.
- Conformance coverage includes Nemotron 3 Nano, separate Kimi K2/K2.5/K3 fixtures, Qwen3.8 MoE/Flash-Next/dense-negative fixtures, and GLM-5.2. Unsupported model/runtime combinations fail closed.
- New non-test Python and shell files carry the 2026 NVIDIA copyright header. New public functions and methods are fully typed and new typed modules are listed explicitly in `pyrefly.toml`.
- Follow strict RED/GREEN/refactor TDD. Every test names an observable break and uses literal, independently derived expected values.

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `nemo_rl/precision_policy/config.py` | YAML-loaded Pydantic schema and strict validation |
| `nemo_rl/precision_policy/semantic.py` | Frozen semantic addresses, roles, formats, atomic groups, and manifests |
| `nemo_rl/precision_policy/compiler.py` | Positive selection, layer filtering, coverage/conflict checks, graph-intent generation, canonical intent digests |
| `nemo_rl/precision_policy/topology.py` | Topology-adapter protocol, registry, nested text-config resolution, complete accounting |
| `nemo_rl/precision_policy/adapters/qwen.py` | Qwen3/Qwen3.5/Qwen3.8 semantic classification |
| `nemo_rl/precision_policy/adapters/nemotron.py` | Nano/Lightning/Super/Ultra semantic classification |
| `nemo_rl/precision_policy/adapters/kimi.py` | Kimi K2/K2.5/K3 manifest conformance and encoding declarations |
| `nemo_rl/precision_policy/adapters/glm.py` | GLM-5.2 manifest conformance |
| `nemo_rl/precision_policy/materialize.py` | One-time policy compilation and endpoint artifact injection before workers start |
| `nemo_rl/models/megatron/precision_policy.py` | TE recipe generation and realized source-binding validation |
| `nemo_rl/weight_sync/refit_plan.py` | Extensible ordered component bindings, transform loci, ownership, and execution schedules |
| `nemo_rl/models/generation/vllm/precision_adapter/base.py` | Destination-adapter protocol and capability types |
| `nemo_rl/models/generation/vllm/precision_adapter/registry.py` | Fail-closed version/capability selection |
| `nemo_rl/models/generation/vllm/precision_adapter/v0251.py` | vLLM 0.25.1 construction and realized storage binding |
| `nemo_rl/models/generation/vllm/precision_adapter/v0280.py` | vLLM 0.28.0 construction and realized storage binding |
| `nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py` | NeMo-owned public quantization plugin and canonical-to-runtime MXFP8 transforms |
| `nemo_rl/weight_sync/transaction.py` | Phase state machine, typed results, combined future supervision, abort/poison propagation |
| `tools/config_cli.py` | `explain-precision` entry point using the production compiler |
| `docs/guides/precision-policy.md` | Progressive user guide, choices, examples, diagnostics, migration |
| `tests/fixtures/precision_policy/` | Pinned model topology/config, auxiliary-graph, and destination-layout fixtures |
| `tests/unit/precision_policy/` | Policy, manifest, compiler, adapter, and explanation tests |

---

### Task 1: Typed Positive Precision Policy Schema

**Files:**
- Create: `nemo_rl/precision_policy/__init__.py`
- Create: `nemo_rl/precision_policy/config.py`
- Modify: `nemo_rl/models/policy/__init__.py:569-630`
- Modify: `pyrefly.toml`
- Test: `tests/unit/precision_policy/test_config.py`

**Interfaces:**
- Consumes: YAML mappings under `policy.precision_policy`.
- Produces: `PrecisionPolicyConfig`, `PrecisionScopeConfig`, `LayerSelectorConfig`, `AdvancedMatchConfig`, `SemanticAddressSelectorConfig`, `PrecisionName`, and `parse_precision_policy(value: object) -> PrecisionPolicyConfig | None`.

- [ ] **Step 1: Write failing schema tests**

```python
def test_minimal_routed_scope_defaults_training_to_bf16() -> None:
    policy = PrecisionPolicyConfig.model_validate({
        "scopes": [{
            "id": "routed-middle",
            "role": "moe.routed_expert",
            "layers": {"exclude_first": 2, "exclude_last": 1},
            "rollout": "mxfp8",
        }]
    })
    assert policy.schema_version == 1
    assert policy.default == "bf16"
    assert policy.scopes[0].training is None
    assert policy.scopes[0].layers.index_space == "global_decoder"

@pytest.mark.parametrize("bad", [
    {"default": "mxfp8", "scopes": []},
    {"scopes": [{"id": "x", "role": "moe.routed_expert"}]},
    {"scopes": [{"id": "x", "role": "moe.routed_expert", "advanced_match": {}, "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "role": "moe.routed_expert", "layers": {"exclude_first": -1}, "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "role_typo": "moe.routed_expert", "rollout": "mxfp8"}]},
])
def test_invalid_or_ambiguous_policy_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PrecisionPolicyConfig.model_validate(bad)
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_config.py`

Expected: collection fails because `nemo_rl.precision_policy.config` does not exist.

- [ ] **Step 3: Implement the strict Pydantic schema**

```python
PrecisionName = Literal["bf16", "mxfp8"]
LayerIndexSpace = Literal["global_decoder", "moe_ordinal"]
AtomicConflictMode = Literal["error", "expand"]

class LayerSelectorConfig(BaseModel, extra="allow"):
    index_space: LayerIndexSpace = "global_decoder"
    exclude_first: NonNegativeInt = 0
    exclude_last: NonNegativeInt = 0

class PrecisionScopeConfig(BaseModel, extra="allow"):
    id: str
    role: str | None = None
    advanced_match: AdvancedMatchConfig | None = None
    semantic_addresses: SemanticAddressSelectorConfig | None = None
    layers: LayerSelectorConfig = Field(default_factory=LayerSelectorConfig)
    training: PrecisionName | None = None
    rollout: PrecisionName | None = None
    atomic_conflict: AtomicConflictMode = "error"

class PrecisionPolicyConfig(BaseModel, extra="allow"):
    schema_version: Literal[1] = 1
    default: Literal["bf16"] = "bf16"
    require_match: bool = True
    atomic_conflict: AtomicConflictMode = "error"
    scopes: list[PrecisionScopeConfig]
```

Each model validator rejects undocumented `model_extra`; the scope validator enforces a non-empty unique `id`, exactly one selector, and at least one non-BF16 endpoint request. Add `precision_policy: NotRequired[PrecisionPolicyConfig]` to `PolicyConfig`.

- [ ] **Step 4: Run tests, type checking, and formatting**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_config.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/config.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/__init__.py nemo_rl/precision_policy/config.py nemo_rl/models/policy/__init__.py tests/unit/precision_policy/test_config.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy nemo_rl/models/policy/__init__.py tests/unit/precision_policy/test_config.py pyrefly.toml
git commit -s -m "feat(precision): add semantic policy schema"
```

### Task 2: Semantic Manifest, Roles, Formats, and Complete Accounting

**Files:**
- Create: `nemo_rl/precision_policy/semantic.py`
- Test: `tests/unit/precision_policy/test_semantic.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: normalized logical model component descriptions from topology adapters.
- Produces: frozen `SemanticAddress`, `SemanticTensor`, `SemanticTensorFamily`, `RoleDefinition`, `FormatDescriptor`, `ComponentDescriptor`, `AtomicGroup`, `OutOfScopeReason`, `ParameterInventoryEntry`, `GraphLifecycle`, `ImmutableAuxiliaryEvidence`, `SemanticGraphManifest`, and `SemanticManifestBundle.validate_complete()`.

- [ ] **Step 1: Write failing semantic-contract tests**

```python
def test_routed_expert_role_excludes_shared_router_and_auxiliary() -> None:
    manifest = SemanticGraphManifest(model_family="fixture", model_revision="fixture-rev", graph_id="text.decoder", tensors=(
        tensor("text.decoder.layer.1.moe.routed.0.gate", "moe.expert_ffn", {"expert_kind": "routed", "projection": "gate"}),
        tensor("text.decoder.layer.1.moe.shared.gate", "moe.expert_ffn", {"expert_kind": "shared", "projection": "gate"}),
        tensor("text.decoder.layer.1.moe.router", "moe.router", {}),
        tensor("draft.decoder.layer.1.moe.routed.0.gate", "moe.expert_ffn", {"expert_kind": "routed", "projection": "gate"}, graph="draft.decoder"),
    ))
    role = builtin_role_definitions(1)["moe.routed_expert"]
    assert [item.address.semantic_id for item in manifest.tensors if role.matches(item.address)] == [
        "text.decoder.layer.1.moe.routed.0.gate"
    ]

def test_mutable_main_tensor_cannot_hide_out_of_scope() -> None:
    with pytest.raises(ValueError, match="mutable main-model"):
        SemanticGraphManifest(model_family="fixture", model_revision="fixture-rev", graph_id="text.decoder", tensors=(mutable_tensor(),), out_of_scope=(out_of_scope(mutable=True),)).validate_complete()
```

- [ ] **Step 2: Run the tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py`

Expected: import failure for `nemo_rl.precision_policy.semantic`.

- [ ] **Step 3: Implement immutable semantic records and exact built-in roles**

```python
@dataclass(frozen=True, slots=True)
class SemanticAddress:
    semantic_id: str
    graph: str
    model_part: str
    module_kind: str
    attributes: tuple[tuple[str, str], ...]
    parameter_role: str
    global_decoder_layer: int | None
    moe_ordinal: int | None

@dataclass(frozen=True, slots=True)
class SemanticGraphManifest:
    model_family: str
    model_revision: str
    graph_id: str
    tensors: tuple[SemanticTensor, ...]
    families: tuple[SemanticTensorFamily, ...] = ()
    atomic_groups: tuple[AtomicGroup, ...] = ()
    out_of_scope: tuple[OutOfScopeTensor, ...] = ()
```

`SemanticManifestBundle` contains exactly one main graph, zero or more separately identified mutable auxiliary graphs, and revision-pinned immutable auxiliary evidence. Define MXFP8 as E4M3 values plus E8M0 block-32 scales; keep block-FP8, NVFP4, and MXFP4 as distinct descriptors. Reject duplicate semantic IDs, unknown logical axes, duplicate ownership, untyped exclusions, any mutable main-model exclusion, an immutable auxiliary without revision evidence, or a mutable MTP/draft graph omitted from the bundle.

- [ ] **Step 4: Run unit, type, and format gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/semantic.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/semantic.py tests/unit/precision_policy/test_semantic.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy/semantic.py tests/unit/precision_policy/test_semantic.py pyrefly.toml
git commit -s -m "feat(precision): define semantic model manifest"
```

### Task 3: Deterministic Policy Compiler and Atomic Precision Intents

**Files:**
- Create: `nemo_rl/precision_policy/compiler.py`
- Test: `tests/unit/precision_policy/test_compiler.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `compile_precision_policy(policy: PrecisionPolicyConfig, manifests: SemanticManifestBundle, roles: Mapping[str, RoleDefinition]) -> CompiledPrecisionIntentGroup`.
- Produces: frozen `CompiledGraphPrecisionIntent` records with immutable per-semantic-ID training/rollout assignments, selected layer ranges, full scope expansions, physical atomic closures, endpoint realization requests, canonical graph `intent_id` values, ordered `CompiledPrecisionIntentGroup`, and `intent_group_id`. Actual backend capability, ownership, transform, and local-plan fingerprints are deferred to Task 7 after realized binding.

- [ ] **Step 1: Write failing selection and conflict tests**

```python
def test_global_decoder_boundaries_keep_first_and_last_selected_layers_bf16() -> None:
    plan = compile_fixture(
        layers=range(6),
        moe_layers=(1, 2, 4, 5),
        scope={"role": "moe.routed_expert", "layers": {"exclude_first": 2, "exclude_last": 1}, "rollout": "mxfp8"},
    )
    assert plan.rollout_precision("layer.1.expert.0.gate") == "bf16"
    assert plan.rollout_precision("layer.2.expert.0.gate") == "mxfp8"
    assert plan.rollout_precision("layer.4.expert.0.gate") == "mxfp8"
    assert plan.rollout_precision("layer.5.expert.0.gate") == "bf16"

def test_moe_ordinal_boundary_differs_from_global_decoder() -> None:
    global_plan = compile_fixture(layers=range(3), moe_layers=(1, 2), scope=scope("global_decoder", 1, 0))
    ordinal_plan = compile_fixture(layers=range(3), moe_layers=(1, 2), scope=scope("moe_ordinal", 1, 0))
    assert global_plan.selected_ids == frozenset({"layer.1.expert.0.gate", "layer.2.expert.0.gate"})
    assert ordinal_plan.selected_ids == frozenset({"layer.2.expert.0.gate"})
```

Also add literal tests for zero-match, unknown role, incomplete advertised role coverage, overlapping conflicting scopes, full-range exclusion, atomic fused QKV conflict, allowed fixed-point expansion, expansion crossing BF16 boundary, dictionary-order-independent intent digest, invalid immutable-auxiliary evidence, separately compiled co-trained MTP/draft graphs, and deterministic graph ordering. Backend unsupported-format and rank-local ownership checks belong to Tasks 7-8, after realized capabilities exist.

- [ ] **Step 2: Run the compiler tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_compiler.py`

Expected: import failure for `compile_precision_policy`.

- [ ] **Step 3: Implement compilation in explicit passes**

```python
def compile_precision_policy(...) -> CompiledPrecisionIntentGroup:
    manifests.validate_complete()
    graph_intents = tuple(
        _compile_graph_intent(policy, manifest, roles)
        for manifest in manifests.mutable_graphs_in_canonical_order()
    )
    canonical = _canonical_intent_group_payload(policy, manifests, graph_intents)
    return CompiledPrecisionIntentGroup(..., intent_group_id=sha256(canonical).hexdigest())
```

Sort graph IDs, semantic IDs, attributes, roles, groups, and components before serialization. Each mutable graph gets its own intent; built-in main-model roles leave auxiliary assignments at BF16 unless an explicit structured auxiliary selector applies. Never hash object identity or dictionary insertion order.

- [ ] **Step 4: Run compiler, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_compiler.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/compiler.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/compiler.py tests/unit/precision_policy/test_compiler.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy/compiler.py tests/unit/precision_policy/test_compiler.py pyrefly.toml
git commit -s -m "feat(precision): compile deterministic endpoint plans"
```

### Task 4: Model Topology Adapters and Pinned Conformance Fixtures

**Files:**
- Create: `nemo_rl/precision_policy/topology.py`
- Create: `nemo_rl/precision_policy/adapters/__init__.py`
- Create: `nemo_rl/precision_policy/adapters/qwen.py`
- Create: `nemo_rl/precision_policy/adapters/nemotron.py`
- Create: `nemo_rl/precision_policy/adapters/kimi.py`
- Create: `nemo_rl/precision_policy/adapters/glm.py`
- Create: `tests/fixtures/precision_policy/qwen3_30ba3b.json`
- Create: `tests/fixtures/precision_policy/qwen3_5_35ba3b.json`
- Create: `tests/fixtures/precision_policy/nemotron_3_5_lightning_30ba3b.json`
- Create: `tests/fixtures/precision_policy/nemotron3_super_120ba12b.json`
- Create: `tests/fixtures/precision_policy/nemotron3_ultra_550ba55b.json`
- Create: `tests/fixtures/precision_policy/nemotron3_nano_30ba3b.json`
- Create: `tests/fixtures/precision_policy/kimi_k2.json`
- Create: `tests/fixtures/precision_policy/kimi_k2_5.json`
- Create: `tests/fixtures/precision_policy/kimi_k3.json`
- Create: `tests/fixtures/precision_policy/qwen3_8_2_4t_a95b.json`
- Create: `tests/fixtures/precision_policy/qwen3_8_flash_next.json`
- Create: `tests/fixtures/precision_policy/qwen3_8_27b.json`
- Create: `tests/fixtures/precision_policy/glm_5_2.json`
- Create: `tests/fixtures/precision_policy/auxiliary_graphs.json`
- Test: `tests/unit/precision_policy/test_topology_adapters.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `build_semantic_manifest_bundle(model_config: Mapping[str, object], model_revision: str, parameter_inventory: Sequence[ParameterInventoryEntry], auxiliary_declarations: Sequence[AuxiliaryGraphDeclaration]) -> SemanticManifestBundle`.
- Produces: registered adapters selected by `model_type` and architecture capabilities; separate main/MTP/draft graph manifests with typed lifecycles and alias ownership; `resolve_text_config()` handles nested `text_config` without assuming top-level `num_hidden_layers`.

- [ ] **Step 1: Add pinned literal topology fixtures and failing adapter tests**

```python
def test_qwen35_uses_nested_40_layer_text_config() -> None:
    manifest = build_fixture_manifest("qwen3_5_35ba3b.json")
    assert manifest.decoder_layer_count == 40
    assert manifest.decoder_layers == tuple(range(40))
    assert manifest.find("text.decoder.layer.39.moe.routed.0.down") is not None

def test_kimi_k25_and_k3_exact_routed_domains() -> None:
    k25 = build_fixture_manifest("kimi_k2_5.json")
    k3 = build_fixture_manifest("kimi_k3.json")
    assert k25.role_domain_size("moe.routed_expert") == 60 * 384 * 3
    assert k3.role_domain_size("moe.routed_expert") == 92 * 896 * 3
    assert k3.role_match_count("sequence_mixer.kda.projections", "moe.routed_expert") == 0
```

Add production fixtures for Qwen3-30B-A3B, Qwen3.5-35B-A3B, Lightning, Super, Ultra and conformance fixtures for Nano, Kimi K2, Kimi K2.5, Kimi K3, Qwen3.8 MoE, Qwen3.8 Flash-Next, Qwen3.8-27B dense, and GLM-5.2. The Qwen3.8 dense fixture must fail required routed-expert compilation. Kimi K2 uses `weight + weight_scale_inv`; K2.5 uses `weight_packed + weight_scale + weight_shape`.

Add MTP/draft fixtures for: a static checkpoint-owned MTP, an independent co-trained MTP, MTP tied aliases, a static external drafter, and a co-trained speculative drafter using a different model-family adapter. Assert that main-model roles select none of them; every mutable auxiliary has its own graph manifest and BF16 intent; immutable graphs carry checkpoint-revision evidence; and tied aliases point to an explicit main-graph owner without duplicating transfer ownership.

- [ ] **Step 2: Run adapter tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_adapters.py`

Expected: import failure for the topology registry.

- [ ] **Step 3: Implement adapter selection and semantic classification**

```python
class ModelTopologyAdapter(Protocol):
    adapter_id: str
    def supports(self, model_config: Mapping[str, object]) -> bool: ...
    def role_definitions(self, schema_version: int) -> Mapping[str, RoleDefinition]: ...
    def build_manifest(self, model_config: Mapping[str, object], model_revision: str, inventory: Sequence[ParameterInventoryEntry], graph_id: str) -> SemanticGraphManifest: ...
```

Classifiers may recognize endpoint names internally, but emit only semantic addresses. The bundle builder chooses an adapter independently for an external drafter and orders main, MTP, and draft graphs deterministically. Reject ambiguous names, missing expected role members, inconsistent expert counts, unnormalized one-based layer indices, and unsupported model revisions. Keep dense prefix layers in the global decoder coordinate even when they contain no routed expert.

- [ ] **Step 4: Run adapter, compiler, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_adapters.py tests/unit/precision_policy/test_compiler.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy tests/unit/precision_policy tests/fixtures/precision_policy pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy tests/unit/precision_policy/test_topology_adapters.py tests/fixtures/precision_policy pyrefly.toml
git commit -s -m "feat(precision): add model topology adapters"
```

### Task 5: One-Time Intent Materialization and `explain-precision`

**Files:**
- Create: `nemo_rl/precision_policy/materialize.py`
- Modify: `nemo_rl/models/policy/lm_policy.py:90-180`
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/generation/__init__.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `tools/config_cli.py`
- Test: `tests/unit/precision_policy/test_materialize.py`
- Test: `tests/unit/tools/test_config_cli.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `materialize_precision_policy(policy_config: PolicyConfig) -> MaterializedPrecisionIntents | None` before policy or generation workers are constructed.
- Produces: one frozen `CompiledPrecisionIntentGroup` object under the typed internal `_compiled_precision_intents` key in both endpoint configs and `render_precision_explanation(intents, format: Literal["text", "json"]) -> str`. Serialization is explicit through `intents.to_wire_dict()` only at process or CLI boundaries; no `dict[str, Any]` side channel is introduced. This stage does not claim a final `plan_id` or actual backend capabilities.

- [ ] **Step 1: Write failing one-source-of-truth and CLI behavior tests**

```python
def test_materializer_injects_the_same_intent_group_into_both_endpoints() -> None:
    config = qwen30_policy_config()
    result = materialize_precision_policy(config)
    assert result is not None
    assert config["_compiled_precision_intents"].intent_group_id == result.intent_group_id
    assert config["generation"]["_compiled_precision_intents"].intent_group_id == result.intent_group_id
    assert config["_compiled_precision_intents"] is config["generation"]["_compiled_precision_intents"]

def test_explain_precision_reports_bf16_boundaries_and_mxfp8_middle(tmp_path: Path) -> None:
    completed = run_config_cli("explain-precision", fixture_recipe(tmp_path), "--format", "json")
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["scopes"][0]["selected_global_decoder_layers"] == [2, 3, 4]
    assert payload["summary"]["rollout"]["mxfp8"] == 3 * 8 * 3
```

Add a subprocess test asserting nonzero exit and an actionable message for zero matches, an unsupported topology adapter, conflicting scopes, incomplete checkpoint-inventory accounting, and an invalid immutable auxiliary declaration. Actual endpoint binding gaps are rejected in Tasks 6-8 after realization.

- [ ] **Step 2: Run tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py tests/unit/tools/test_config_cli.py -k precision`

Expected: missing materializer and unrecognized `explain-precision` subcommand.

- [ ] **Step 3: Implement materialization and wire the CLI to the production compiler**

```python
def materialize_precision_policy(policy_config: PolicyConfig) -> MaterializedPrecisionIntents | None:
    raw_policy = policy_config.get("precision_policy")
    if raw_policy is None:
        return None
    policy = parse_precision_policy(raw_policy)
    manifests, roles = resolve_topology(policy_config)
    intents = compile_precision_policy(policy, manifests, roles)
    policy_config["_compiled_precision_intents"] = intents
    policy_config["generation"]["_compiled_precision_intents"] = intents
    return MaterializedPrecisionIntents(intents=intents)
```

Invoke the materializer at the start of `Policy.__init__`, before `resolve_policy_worker_cls()` or generation-class selection, so every algorithm uses the same path and backend dispatch sees requested training/rollout formats. Repeated calls with an identical policy return the existing intents; a different policy or model revision is rejected rather than silently replacing it. `tools/config_cli.py explain-precision RECIPE` resolves inheritance and interpolation exactly as `expand` does, invokes this function once, and prints graph lifecycles, full role expansion, selected/unselected counts, layer ranges, atomic expansion, requested endpoint formats, model revisions, and intent digests. It labels transforms, physical layouts, and final plan IDs as unavailable until realized binding and never reimplements selector logic.

- [ ] **Step 4: Run focused tests and repository config tests**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py tests/unit/tools/test_config_cli.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/materialize.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/materialize.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/__init__.py nemo_rl/models/generation/__init__.py nemo_rl/models/generation/interfaces.py tools/config_cli.py tests/unit/precision_policy/test_materialize.py tests/unit/tools/test_config_cli.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy/materialize.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/__init__.py nemo_rl/models/generation/__init__.py nemo_rl/models/generation/interfaces.py tools/config_cli.py tests/unit/precision_policy/test_materialize.py tests/unit/tools/test_config_cli.py pyrefly.toml
git commit -s -m "feat(precision): materialize and explain compiled plans"
```

### Task 6: Compile the Training Plan into Transformer Engine Configuration

**Files:**
- Create: `nemo_rl/models/megatron/precision_policy.py`
- Modify: `nemo_rl/models/megatron/setup.py:1107-1220`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Test: `tests/unit/models/megatron/test_precision_policy.py`
- Test: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `compile_te_precision_recipe(intents: CompiledPrecisionIntentGroup, source_bindings: Sequence[SourceModuleBinding]) -> TEPrecisionArtifact`.
- Produces: frozen `SourceModuleBinding`, deterministic enabled exact matchers for training-MXFP8 modules, explicit BF16 evaluation recipes, a recipe digest, and `validate_realized_training_precision(intents, artifact, realized_modules) -> None`.

- [ ] **Step 1: Write failing TE artifact tests**

```python
def test_rollout_only_policy_keeps_all_training_modules_bf16() -> None:
    artifact = compile_te_precision_recipe(rollout_only_plan(), qwen_source_bindings())
    assert artifact.mx_training_module_ids == ()
    assert artifact.recipe is None

def test_mxfp8_training_compiles_only_middle_routed_experts() -> None:
    artifact = compile_te_precision_recipe(mxfp8_training_plan(), qwen_source_bindings())
    assert artifact.mx_training_module_ids == (
        "text.decoder.layer.2.moe.routed",
        "text.decoder.layer.3.moe.routed",
    )
    assert all(matcher["enabled"] is True for matcher in artifact.recipe["matchers"].values())
```

Add tests that first/last routed modules remain BF16, QKVO can be a second scope, duplicate/partial physical bindings fail, disabled matcher is rejected, user-supplied `te_precision_config_file` plus semantic policy fails as two sources of truth, and realized TE module precision mismatch fails before refit.

- [ ] **Step 2: Run TE tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/models/megatron/test_precision_policy.py`

Expected: import failure for `nemo_rl.models.megatron.precision_policy`.

- [ ] **Step 3: Implement deterministic TE generation and setup integration**

```python
@dataclass(frozen=True, slots=True)
class TEPrecisionArtifact:
    recipe: Mapping[str, object] | None
    mx_training_module_ids: tuple[str, ...]
    recipe_digest: str

def apply_compiled_training_precision(model_cfg: object, policy_config: PolicyConfig) -> None:
    intents = require_compiled_intents(policy_config)
    artifact = compile_te_precision_recipe(intents, source_bindings_for(policy_config))
    if artifact.recipe is not None:
        model_cfg.quant_recipe = load_quantization_recipe_from_mapping(artifact.recipe)
```

Keep `fp8_cfg` only as backend compute/storage mechanics derived from the compiled intents. Do not apply a global MXFP8 default to unmatched modules. Persist the artifact digest and validate actual module recipes after model construction; any selected/unselected mismatch raises before communicator creation.

- [ ] **Step 4: Run TE, setup, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/models/megatron/test_precision_policy.py tests/unit/models/megatron/test_megatron_setup.py`

Run: `uv run --no-sync bash tests/functional/test_megatron_te_precision_config.sh`

Run: `uv run --no-sync pyrefly check nemo_rl/models/megatron/precision_policy.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/megatron/precision_policy.py nemo_rl/models/megatron/setup.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_precision_policy.py tests/unit/models/megatron/test_megatron_setup.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/megatron/precision_policy.py nemo_rl/models/megatron/setup.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_precision_policy.py tests/unit/models/megatron/test_megatron_setup.py pyrefly.toml
git commit -s -m "feat(megatron): realize semantic training precision"
```

### Task 7: Extensible Refit Components, Ownership, and Execution Plans

**Files:**
- Create: `nemo_rl/weight_sync/refit_plan.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_refit_plan.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: compiled graph intents plus source/destination capability declarations and bindings.
- Produces: `ComponentRole`, `ComponentBinding`, `BindingSet`, `TransformLocus`, `PhysicalOwner`, `BoundPhysicalOwner`, `EndpointCapabilities`, `BoundSourcePlan`, `BoundDestinationPlan`, `BoundComponentBatch`, `DestinationCommitReady`, `DestinationPoisonReason`, `LocalExecutionPlan`, graph-level `CanonicalRefitPlan`, ordered `CanonicalRefitPlanGroup`, `build_canonical_refit_plan_group()`, `validate_refit_plan()`, and ordered wire metadata.

- [ ] **Step 1: Write failing component and ownership tests**

```python
def test_direct_copy_requires_complete_layout_equality() -> None:
    source = bf16_descriptor(shape=(128, 928, 2688), layout="logical_eih")
    destination = bf16_descriptor(shape=(128, 42, 1024, 64), layout="trtllm_block")
    with pytest.raises(ValueError, match="destination_native_loader"):
        plan_transform(source, destination)

def test_mxfp8_component_order_is_values_then_block_scales() -> None:
    binding = mxfp8_binding("layer.2.expert.0.gate")
    assert tuple(component.role for component in binding.components) == ("values", "block_scales")
```

Add tests for arbitrary future component roles, missing/duplicate components, source/destination semantic-set inequality, unsupported endpoint formats, native MXFP8 direct component transfer, BF16→MXFP8 destination transform, canonical BF16→TRTLLM native loader, fused owner atomicity, local TP/EP/PP ownership, canonical versus rank-local digests, one canonical plan per mutable graph, and deterministic transaction-group assembly from ordered plan IDs and target version. Static immutable graphs validate evidence but contribute no transfer plan.

- [ ] **Step 2: Run refit-plan tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_plan.py`

Expected: import failure for `nemo_rl.weight_sync.refit_plan`.

- [ ] **Step 3: Implement typed, ordered refit plans**

```python
class TransformLocus(StrEnum):
    NONE = "none"
    SOURCE = "source"
    DESTINATION = "destination"
    DESTINATION_NATIVE_LOADER = "destination_native_loader"

@dataclass(frozen=True, slots=True)
class BindingSet:
    semantic_id: str
    format: FormatDescriptor
    components: tuple[ComponentBinding, ...]
    physical_owners: tuple[str, ...]
    atomic_group_ids: tuple[str, ...]
```

Validate the full plan before NCCL groups are created. Wire metadata carries semantic ID, component role, dtype, logical/physical shapes, axes, placement, owner, layout, transform, and plan ID; it never encodes a fixed two-field `weight/weight_scale` assumption.

`build_canonical_refit_plan_group()` runs only after both endpoints bind realized capabilities. It creates separate main, MTP, and speculative-draft plans, verifies tied aliases versus independent owners, and hashes their ordered plan IDs plus the target generation version into `transaction_group_id`.

- [ ] **Step 4: Run plan, reshard, type, and format gates**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_plan.py tests/unit/weight_sync/test_nccl_reshard_utils.py tests/unit/weight_sync/test_weight_synchronizer.py`

Run: `uv run --no-sync pyrefly check nemo_rl/weight_sync/refit_plan.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/nccl_reshard_utils.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py tests/unit/weight_sync/test_refit_plan.py tests/unit/weight_sync/test_nccl_reshard_utils.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/nccl_reshard_utils.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py tests/unit/weight_sync/test_refit_plan.py tests/unit/weight_sync/test_nccl_reshard_utils.py pyrefly.toml
git commit -s -m "feat(refit): add semantic component execution plans"
```

### Task 8: Public, Versioned vLLM Precision Adapters

**Files:**
- Create: `nemo_rl/models/generation/vllm/precision_adapter/__init__.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/base.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/registry.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/v0251.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/v0280.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:550-700`
- Modify: `nemo_rl/models/generation/vllm/quantization/fp8.py:58-300`
- Test: `tests/unit/models/generation/test_vllm_precision_adapter.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: serialized compiled rollout intents and actual `vllm.__version__`; common bound-plan records from Task 7.
- Produces: frozen `VllmCapabilityProbes`; `VllmEndpointAdapter` with `capabilities()`, `configure_engine_kwargs()`, `bind_realized_storage()`, `prepare_transaction()`, `load_component_batch()`, `finalize_transaction()`, and `poison()`; `select_vllm_endpoint_adapter(version: str, probes: VllmCapabilityProbes) -> VllmEndpointAdapter`. Realized binding returns Task 7's `BoundDestinationPlan`, whose capability fingerprint is incorporated only when the canonical plan is assembled.

- [ ] **Step 1: Write failing registry and isolation tests**

```python
@pytest.mark.parametrize(("version", "adapter_id"), [("0.25.1", "vllm-0.25.1"), ("0.28.0", "vllm-0.28.0")])
def test_exact_supported_version_selects_dedicated_adapter(version: str, adapter_id: str) -> None:
    assert select_vllm_endpoint_adapter(version, complete_probes()).adapter_id == adapter_id

def test_unknown_or_incomplete_vllm_fails_before_model_construction() -> None:
    with pytest.raises(UnsupportedVllmEndpointError, match="capability"):
        select_vllm_endpoint_adapter("0.29.0", incomplete_probes())
```

Add a test that importing the registry without vLLM installed succeeds, that selecting 0.28 never imports 0.25-only modules, and that two engines with different plans do not share process-global quantization state.

- [ ] **Step 2: Run adapter tests and observe RED**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_precision_adapter.py --vllm-only`

Expected: missing precision-adapter package.

- [ ] **Step 3: Implement lazy version modules and NeMo quantization registration**

```python
class VllmEndpointAdapter(Protocol):
    adapter_id: str
    def configure_engine_kwargs(self, intents: CompiledPrecisionIntentGroup, kwargs: dict[str, object]) -> None: ...
    def bind_realized_storage(self, intents: CompiledPrecisionIntentGroup, model: object) -> BoundDestinationPlan: ...
    def prepare_transaction(self, transaction_id: str) -> None: ...
    def load_component_batch(self, batch: BoundComponentBatch) -> None: ...
    def finalize_transaction(self, transaction_id: str) -> DestinationCommitReady: ...
    def poison(self, reason: DestinationPoisonReason) -> None: ...
```

Register a NeMo MXFP8 quantization config through vLLM's public quantization registry and pass it through normal engine construction. Replace MXFP8 `unittest.mock.patch` installation and global `FP8State` dependence with adapter-owned method instances and worker-extension state. Each version module owns its version-specific imports and public capability probes.

- [ ] **Step 4: Run adapter and existing FP8 regression tests**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_fp8_quantization.py tests/unit/models/generation/test_vllm_fp8_hf_overrides.py --vllm-only`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/precision_adapter`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml`

Expected: all commands pass under the pinned 0.25.1 environment. Repeat the same conformance file in the pinned 0.28.0 environment and require pass before marking this task complete.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml
git commit -s -m "refactor(vllm): isolate public precision adapters"
```

### Task 9: Mixed BF16/MXFP8 TRTLLM Destination Loading

**Files:**
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/v0251.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/v0280.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py:790-1410`
- Modify: `nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py`
- Test: `tests/unit/models/generation/test_vllm_mixed_precision_refit.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `BoundDestinationPlan` whose owners independently request BF16 logical load, BF16→MXFP8 quantization, or compatible native-MXFP8 component copy.
- Produces: batched owner loads, canonical staging lifetime tracking, dirty-owner set, exactly-once destination finalization, and `DestinationCommitReady` only after completion fences.

- [ ] **Step 1: Write failing mixed-layout and numeric tests**

```python
def test_mixed_trtllm_plan_uses_distinct_owner_transforms() -> None:
    bound = bind_lightning_plan(exclude_first=2, exclude_last=1, tp=2)
    assert bound.owner("layer.1.routed.w2").transform == "destination_native_loader"
    assert bound.owner("layer.2.routed.w2").transform == "destination"
    assert bound.owner("layer.51.routed.w2").transform == "destination_native_loader"

def test_lightning_tp2_padding_contract() -> None:
    result = finalize_bf16_trtllm_expert(torch.arange(128 * 928 * 2688, dtype=torch.bfloat16).reshape(128, 928, 2688))
    assert result.shape == (128, 42, 1024, 64)
    assert inverse_trtllm_layout(result, logical_shape=(128, 928, 2688)).shape == (128, 928, 2688)
```

Add literal padding cases: Lightning TP2 `928→1024` and `2688→3072`, Super TP4 `672→768`, Ultra TP16 `320→384`, Qwen3 TP4 `192→256`, Qwen3.5 TP8 `64→128`. Cover gated/non-gated W13/W31, grouped and split sources, zero-value/unit-scale padding, scale flatten/interleave, native MXFP8 component order, A→B→C repeated refits, finalizer failure poisoning, and no commit after partial load.

- [ ] **Step 2: Run mixed refit tests and observe RED**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_nccl_reshard_backend.py -k 'mixed or padding or grouped' --vllm-only`

Expected: BF16 boundary owners take the dtype-equality direct path or grouped MXFP8 input is rejected.

- [ ] **Step 3: Implement owner-dispatched loading and exactly-once finalization**

```python
def load_owner(self, owner: BoundPhysicalOwner, components: Mapping[ComponentRole, torch.Tensor]) -> None:
    if owner.transform is TransformLocus.NONE:
        _copy_compatible_components(owner, components)
    elif owner.transform is TransformLocus.DESTINATION:
        _quantize_bf16_to_mxfp8(owner, components[ComponentRole.LOGICAL_VALUES])
    elif owner.transform is TransformLocus.DESTINATION_NATIVE_LOADER:
        _load_logical_bf16_through_vllm(owner, components[ComponentRole.LOGICAL_VALUES])
    else:
        raise UnsupportedTransformError(owner.transform)
    self._dirty_owner_ids.add(owner.owner_id)
```

Use complete descriptors, never dtype-only dispatch. Keep logical staging alive through deferred native reload. Finalization pads/permutates/shuffles only dirty owners, records one completion event per batch, clears canonical scratch after the fence, and raises if an owner is finalized twice.

- [ ] **Step 4: Run all vLLM refit regression gates**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_fp8_quantization.py --vllm-only`

Run: `uv run --no-sync pytest -q tests/test_mxfp8_flashinfer_padding.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_nccl_reshard_backend.py pyrefly.toml`

Expected: all commands pass in both vLLM environments.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_nccl_reshard_backend.py pyrefly.toml
git commit -s -m "feat(refit): load mixed BF16 and MXFP8 TRTLLM owners"
```

### Task 10: Native MXFP8 Training Source Components

**Files:**
- Create: `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Test: `tests/unit/models/policy/test_mxfp8_refit_source.py`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: compiled graph intents and actual TE/Megatron parameters; common bound-plan records from Task 7.
- Produces: frozen `SourceParameterInventory` and Task 7's `BoundSourcePlan`, `bind_mxfp8_source(intents, named_parameters) -> BoundSourcePlan`, and ordered direct `values`/`block_scales` exports for native-MXFP8 owners; BF16 owners still export logical BF16.

- [ ] **Step 1: Write failing source binding and repeated-export tests**

```python
def test_native_source_exports_values_and_scales_without_requantizing() -> None:
    bound = bind_mxfp8_source(native_training_plan(), realized_te_parameters())
    exported = bound.export("layer.2.expert.0.gate")
    assert tuple(component.role for component in exported) == ("values", "block_scales")
    assert exported[0].tensor.data_ptr() == realized_te_parameters()["gate.values"].data_ptr()

def test_bf16_boundary_source_remains_logical_bf16() -> None:
    exported = bind_mxfp8_source(native_training_plan(), realized_te_parameters()).export("layer.1.expert.0.gate")
    assert tuple(component.role for component in exported) == ("logical_values",)
    assert exported[0].tensor.dtype == torch.bfloat16
```

Add tests for grouped expert views with gradients, forged dtype metadata, mismatched scale geometry, disabled FP8 export, storage alias partitioning, synchronization before export, MTP exclusion, and direct native component compatibility failure.

- [ ] **Step 2: Run source tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py`

Expected: missing source adapter.

- [ ] **Step 3: Implement source binding from the canonical graph intents**

```python
def bind_mxfp8_source(intents: CompiledPrecisionIntentGroup, inventory: SourceParameterInventory) -> BoundSourcePlan:
    bindings = tuple(_bind_semantic_source(item, inventory) for item in intents.mutable_semantic_items)
    _validate_exact_semantic_coverage(intents, bindings)
    _validate_component_geometry(bindings)
    return BoundSourcePlan(intent_group_id=intents.intent_group_id, bindings=bindings)
```

Fence optimizer/TE updates before reading values or scales. Reuse stable component views when safe; copy only when storage reuse or asynchronous transfer requires lifetime extension. Do not identify MXFP8 storage solely from `torch.float8_e4m3fn`.

- [ ] **Step 4: Run policy, reshard, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py tests/unit/models/policy/test_megatron_worker.py tests/unit/weight_sync/test_weight_synchronizer.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/policy/workers/mxfp8_refit_source.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/policy/workers/mxfp8_refit_source.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py tests/unit/models/policy/test_mxfp8_refit_source.py tests/unit/models/policy/test_megatron_worker.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/policy/workers/mxfp8_refit_source.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py tests/unit/models/policy/test_mxfp8_refit_source.py tests/unit/models/policy/test_megatron_worker.py pyrefly.toml
git commit -s -m "feat(refit): export native MXFP8 source components"
```

### Task 11: Fail-Fast Transaction and Combined Future Supervisor

**Files:**
- Create: `nemo_rl/weight_sync/transaction.py`
- Modify: `nemo_rl/distributed/refit_watchdog.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Test: `tests/unit/weight_sync/test_refit_transaction.py`
- Test: `tests/unit/distributed/test_refit_watchdog.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: a validated `CanonicalRefitPlanGroup`, source and destination `ray.ObjectRef` sets, plus registered abort/poison callbacks.
- Produces: `RefitPhase`, `RefitWorkerResult`, `RefitFailure`, `RefitTransaction`, `supervise_refit_futures()`, and bounded `abort_refit_transaction()`.

- [ ] **Step 1: Write failing first-failure and poison tests**

```python
def test_receiver_failure_is_observed_while_sender_is_still_pending(fake_ray: FakeRay) -> None:
    sender = fake_ray.pending("train-rank-0")
    receiver = fake_ray.failed("gen-rank-1", RuntimeError("layout conversion failed"))
    with pytest.raises(RefitTransactionError, match="gen-rank-1.*layout conversion failed"):
        supervise_refit_futures(transaction(), [sender], [receiver])
    assert fake_ray.elapsed < 1.0

@pytest.mark.parametrize("result", [None, False, {}, RefitWorkerResult(ok=False, rank=2, phase="finalize", detail="bad")])
def test_non_success_result_never_commits(result: object) -> None:
    with pytest.raises(RefitTransactionError):
        validate_worker_result(result)
```

Add tests for first and later component failure, finalize/commit failure, timeout, silent peer, abort callback failure, communicator-abort fallback to worker termination, original cause/rank preservation, no partial version commit, watchdog remaining armed until transaction resolution, and bounded teardown.

Add transaction-group tests using Task 7's main, MTP, and speculative-draft plan IDs: every member FINALIZE acknowledgement is required, a draft-only failure prevents the main version from committing, a stale draft target version is rejected, and a static immutable drafter is validated but sends no per-step payload.

- [ ] **Step 2: Run transaction tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Expected: missing transaction module and current train-first wait delays receiver failure.

- [ ] **Step 3: Implement explicit PREPARE→READY→TRANSFER→FINALIZE→COMMIT state transitions**

```python
class RefitPhase(StrEnum):
    PREPARE = "prepare"
    READY = "ready"
    TRANSFER = "transfer"
    FINALIZE = "finalize"
    COMMIT = "commit"
    ABORT = "abort"

@dataclass(frozen=True, slots=True)
class RefitWorkerResult:
    ok: bool
    rank: int
    phase: RefitPhase
    plan_id: str
    transaction_id: str
    detail: str | None = None
```

Use `ray.wait(..., num_returns=1, timeout=remaining_deadline)` across both source and destination refs, validate each ready result immediately, and only commit after the exact expected rank/phase acknowledgement set is complete. On failure, preserve the first exception, run all abort/poison callbacks concurrently with a fixed teardown deadline, terminate owners whose abort does not acknowledge, then re-raise the first cause wrapped with structured context.

- [ ] **Step 4: Run transaction, watchdog, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Run: `uv run --no-sync pyrefly check nemo_rl/weight_sync/transaction.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync/transaction.py nemo_rl/distributed/refit_watchdog.py nemo_rl/weight_sync/interfaces.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync/transaction.py nemo_rl/distributed/refit_watchdog.py nemo_rl/weight_sync/interfaces.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py pyrefly.toml
git commit -s -m "feat(refit): add fail-fast transaction supervisor"
```

### Task 12: Integrate Fatal Transactions into IPC, Collective, Reshard, Sync, and Async RL

**Files:**
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:1270-1350`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py:1420-1515`
- Modify: `nemo_rl/algorithms/grpo.py:2510-2610,4737-4766,5417-5450`
- Modify: `nemo_rl/algorithms/async_utils/trajectory_collector.py:1040-1175`
- Modify: `examples/run_grpo.py:190-270`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Test: `tests/unit/algorithms/test_grpo.py`
- Test: `tests/unit/algorithms/test_async_utils.py`
- Test: `tests/unit/models/generation/test_vllm_backend.py`
- Test: `tests/functional/refit_failure_exit.py`

**Interfaces:**
- Consumes: `RefitTransaction` and typed worker results from Task 11.
- Produces: the same fatal behavior for every refit transport and both training loops; generation weight version changes only on COMMIT.

- [ ] **Step 1: Write failing integration and launcher-exit tests**

```python
def test_initial_async_refit_failure_is_reraised_and_collection_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo, "refit_policy_generation", Mock(side_effect=RuntimeError("receiver load")))
    with pytest.raises(RuntimeError, match="receiver load"):
        run_async_training_fixture()
    assert collector.set_weight_version.call_count == 0
    assert collector.start_collection.call_count == 0
    assert collector.flush_telemetry.call_count == 1

@pytest.mark.parametrize("async_enabled", [False, True])
def test_refit_failure_makes_launcher_exit_nonzero(async_enabled: bool) -> None:
    completed = run_fault_injected_grpo_subprocess(async_enabled=async_enabled, phase="finalize")
    assert completed.returncode != 0
    assert "finalize" in completed.stderr
```

Add transport tests where a generation future fails while a train future remains pending, all-`None` results, malformed results, worker wrapper errors that currently return `False`, cache invalidation failure, failed `prepare_for_generation`, failed shutdown, and async refit after collector pause. Assert no resume/serve after failure and bounded actor cleanup.

- [ ] **Step 2: Run integration tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py -k 'refit and (failure or fatal or transaction)'`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Expected: async initial failure returns normally, receiver failure can be observed late, or malformed success is accepted.

- [ ] **Step 3: Replace sequential waits and swallowed errors with transactions**

```python
try:
    transaction.run(source_refs=futures_train, destination_refs=futures_inference)
except BaseException:
    collector_terminal_cleanup()
    raise
```

Worker methods return `RefitWorkerResult` or raise; they never translate a load/finalize exception to `False`. Thread a positive `refit_timeout_s` through legacy setup and every initial/subsequent refit. Move async startup under terminal cleanup, re-raise after telemetry flush, make cache invalidation failure fatal for every backend, and leave the collector/generation gate closed after a poisoned update. Launcher cleanup is bounded and never replaces the original exception.

Include `has_refit_draft_weights` and `trains_mtp` in preflight construction. When either graph is mutable, register its plan as a transaction-group member and advance its served version only with the main COMMIT. Reject a co-trained auxiliary graph that has no source or destination binding; accept per-step omission only for a revision-pinned immutable auxiliary graph.

- [ ] **Step 4: Run all sync/async/refit regression gates**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync tests/unit/distributed/test_refit_watchdog.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_backend.py --vllm-only`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/async_utils/trajectory_collector.py examples/run_grpo.py tests/unit/weight_sync tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py tests/unit/models/generation/test_vllm_backend.py tests/functional/refit_failure_exit.py`

Expected: all commands pass and both subprocess modes exit nonzero within the test budget.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/async_utils/trajectory_collector.py examples/run_grpo.py tests/unit/weight_sync tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py tests/unit/models/generation/test_vllm_backend.py tests/functional/refit_failure_exit.py
git commit -s -m "fix(refit): make every refit failure fatal"
```

### Task 13: Preserve and Measure the Fast Refit Paths

**Files:**
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
- Modify: `nemo_rl/weight_sync/transaction.py`
- Create: `tools/refit_precision_benchmark.py`
- Test: `tests/unit/models/generation/test_mxfp8_refit_performance_contract.py`
- Test: `tests/unit/weight_sync/test_refit_transaction.py`

**Interfaces:**
- Consumes: immutable bound plans and repeated refit samples.
- Produces: cached route/permutation/buffer schedules, batched conversion, per-phase timing and memory metrics, raw benchmark JSON, and statistical gate evaluation.

- [ ] **Step 1: Write failing no-rescan/reuse and metrics tests**

```python
def test_repeated_refit_reuses_compiled_routes_buffers_and_permutations() -> None:
    adapter = instrumented_mixed_adapter()
    adapter.refit(update_a())
    first = adapter.resource_counters()
    adapter.refit(update_b())
    second = adapter.resource_counters()
    assert second.plan_compilations == first.plan_compilations == 1
    assert second.parameter_name_scans == first.parameter_name_scans == 1
    assert second.persistent_buffer_allocations == first.persistent_buffer_allocations

def test_transaction_metrics_cover_the_collective_critical_path() -> None:
    metrics = completed_transaction_metrics()
    assert set(metrics) >= {"total_s", "source_prepare_s", "wire_s", "destination_finalize_s", "commit_s", "peak_allocated_bytes", "peak_reserved_bytes"}
```

- [ ] **Step 2: Run performance-contract tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py -k metrics`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_mxfp8_refit_performance_contract.py -k 'reuse or metrics' --vllm-only`

Expected: missing counters/metrics or per-refit allocation/scan is observed.

- [ ] **Step 3: Integrate measured optimizations from PRs #3294 and #3669 behind the new contracts**

```python
@dataclass(frozen=True, slots=True)
class RefitPerformanceSample:
    total_s: float
    source_prepare_s: float
    wire_s: float
    destination_finalize_s: float
    commit_s: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
```

Cache the local execution plan, source routes, BF16 boundary buffers, owner slices, and row permutations after binding. Use batched MXFP8 expert quantization/shuffle for homogeneous owners and direct component copy for compatible native MXFP8. Preserve the reference per-owner path only for numeric comparison. The benchmark records paired randomized samples, maximum-rank critical path, warmup/stability condition, environment/SHA/plan digest, and 95% bootstrap bounds for p50/p95 ratios.

- [ ] **Step 4: Run performance-contract and correctness regression tests**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_mxfp8_refit_performance_contract.py tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_vllm_fp8_quantization.py --vllm-only`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/policy/workers/mxfp8_refit_source.py nemo_rl/weight_sync/transaction.py tools/refit_precision_benchmark.py tests/unit/models/generation/test_mxfp8_refit_performance_contract.py tests/unit/weight_sync/test_refit_transaction.py`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/policy/workers/mxfp8_refit_source.py nemo_rl/weight_sync/transaction.py tools/refit_precision_benchmark.py tests/unit/models/generation/test_mxfp8_refit_performance_contract.py tests/unit/weight_sync/test_refit_transaction.py
git commit -s -m "perf(refit): preserve batched semantic fast paths"
```

### Task 14: Positive-Allow-List Production Recipes and Model Matrix

**Files:**
- Modify: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`
- Modify: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml`
- Modify: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mxfp8-rollout.yaml`
- Modify: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-async-1off-mxfp8-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-training-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3.5-35ba3b-8n4g-mxfp8-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3.5-35ba3b-8n4g-mxfp8-training-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3.5-lightning-30ba3b-8n4g-mxfp8-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3.5-lightning-30ba3b-8n4g-mxfp8-training-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mxfp8-training-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3-ultra-550BA55B-64n4g-mxfp8-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3-ultra-550BA55B-64n4g-mxfp8-training-rollout.yaml`
- Modify: `tests/test_mxfp8_rollout_recipes.py`
- Create: `tests/unit/precision_policy/test_production_model_matrix.py`
- Modify: `tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh`
- Modify: `tests/functional/L1_Functional_Tests_GB200_MXFP8.sh`
- Modify: `tests/test_suites/performance_gb200.txt`

**Interfaces:**
- Consumes: the simple schema and compiled model fixtures.
- Produces: routed-expert-only production recipes with recipe-owned N/M values and no user-maintained negative ignore patterns.

- [ ] **Step 1: Rewrite recipe tests to assert semantic outcomes**

```python
@pytest.mark.parametrize("recipe_name", PRODUCTION_RECIPES)
def test_production_recipe_selects_only_middle_routed_experts(recipe_name: str) -> None:
    config = load_recipe(recipe_name)
    plan = compile_recipe_plan(config)
    assert plan.selected_module_kinds("rollout", "mxfp8") == {"moe.expert_ffn"}
    assert plan.selected_attribute_values("expert_kind") == {"routed"}
    assert plan.boundary_precision("rollout") == "bf16"
    assert "quantization_ignore_patterns" not in config["policy"]["generation"]["vllm_cfg"]
```

Add both training modes for all five production families and exact fixture assertions for shared experts, routers, QKVO, MTP/draft, dense layers, and output heads remaining BF16.

For Qwen3.5, Lightning, Ultra, Qwen3.8, and GLM fixtures that advertise MTP, exercise static and co-trained lifecycle declarations. Add one external speculative-drafter recipe and one co-trained drafter fixture; assert coherent main/draft plan-group digests and version targets.

Add `--config-only` handling to `tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh` before it creates or deletes artifact directories. That mode invokes `tools/config_cli.py explain-precision` for the resolved recipe, prints the intent summary, and exits without constructing Ray, allocating GPUs, or launching training.

- [ ] **Step 2: Run recipe tests and observe RED**

Run: `uv run --no-sync pytest -q tests/test_mxfp8_rollout_recipes.py tests/unit/precision_policy/test_production_model_matrix.py`

Expected: existing recipes still contain negative ignore patterns and no MXFP8-training variants exist.

- [ ] **Step 3: Replace each user-maintained ignore list with the positive policy**

```yaml
policy:
  precision_policy:
    schema_version: 1
    default: bf16
    scopes:
      - id: routed-experts-middle
        role: moe.routed_expert
        layers:
          exclude_first: 2
          exclude_last: 1
        rollout: mxfp8
```

MXFP8-training variants add `training: mxfp8` to the same scope. Keep model-specific N/M in the recipe and derive both backend artifacts from the one policy.

- [ ] **Step 4: Run recipe, CLI, and suite-registration gates**

Run: `uv run --no-sync pytest -q tests/test_mxfp8_rollout_recipes.py tests/unit/precision_policy/test_production_model_matrix.py tests/unit/tools/test_config_cli.py`

Run: `uv run --no-sync bash tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh --config-only`

Run: `uv run --no-sync pre-commit run --files examples/configs/recipes/llm/performance tests/test_mxfp8_rollout_recipes.py tests/unit/precision_policy/test_production_model_matrix.py tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh tests/functional/L1_Functional_Tests_GB200_MXFP8.sh tests/test_suites/performance_gb200.txt`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add examples/configs/recipes/llm/performance tests/test_mxfp8_rollout_recipes.py tests/unit/precision_policy/test_production_model_matrix.py tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh tests/functional/L1_Functional_Tests_GB200_MXFP8.sh tests/test_suites/performance_gb200.txt
git commit -s -m "feat(recipes): use positive MXFP8 precision policies"
```

### Task 15: User Documentation, Examples, Choices, and Migration

**Files:**
- Create: `docs/guides/precision-policy.md`
- Modify: `docs/fp8.md`
- Modify: `docs/guides/refit.md`
- Modify: `docs/index.md`
- Modify: `docs/guides/models/qwen/qwen3-5.md`
- Modify: `docs/guides/models/nemotron/nemotron-3.5-lightning.md`
- Modify: `docs/guides/models/nemotron/nemotron-3-super.md`
- Modify: `docs/guides/models/nemotron/nemotron-3-ultra.md`
- Test: `tests/docs/Docs_Tests.sh`

**Interfaces:**
- Consumes: final schema, CLI, recipes, support matrix, and failure/performance contracts.
- Produces: a progressive guide whose examples are copied from tested recipes and whose diagnostic output is generated by the real compiler.

- [ ] **Step 1: Add a documentation example verifier that executes every YAML fragment**

```python
@pytest.mark.parametrize("example", extract_precision_policy_examples(DOC_PATH))
def test_documented_precision_policy_example_compiles(example: dict[str, object]) -> None:
    policy = PrecisionPolicyConfig.model_validate(example)
    intents = compile_precision_policy(policy, documentation_fixture_manifest_bundle(), builtin_roles())
    assert intents.intent_group_id
```

Place this behavioral test in `tests/unit/precision_policy/test_documentation_examples.py`; do not grep prose.

- [ ] **Step 2: Run the documentation example test and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_documentation_examples.py`

Expected: `docs/guides/precision-policy.md` is absent.

- [ ] **Step 3: Write the progressive guide with these exact sections and executable examples**

1. “Choose a mode” table: rollout-only, MXFP8 training+rollout, BF16 everywhere, multiple semantic scopes.
2. Minimal routed-expert middle-layer example.
3. Same example with `training: mxfp8`.
4. Adding a separate `attention.qkvo` scope without changing routed-expert semantics.
5. `global_decoder` versus `moe_ordinal`, including the dense-layer-0 Kimi example.
6. Shared expert, router, MTP/draft, vision, MLA/KDA, and bias exclusions guaranteed by built-in roles.
7. Advanced structured selector and semantic-address escape hatches.
8. Atomic fused-owner conflict/expansion behavior.
9. `tools/config_cli.py explain-precision RECIPE --format text|json` with graph/layer/count/requested-format/intent output and explicit unavailable markers for realized transforms and final plan IDs.
10. The model-construction preflight's realized capability/transform/plan output, BF16→MXFP8 versus native-MXFP8→MXFP8 refit paths, and canonical/runtime layout distinction.
11. Supported/negative model-version matrix for vLLM 0.25.1 and 0.28.0.
12. Migration from `quantization_ignore_patterns`, first/last backend knobs, and hand-written TE recipes; mixing old and new sources fails.
13. Fatal refit behavior and where phase/rank/cause appear in logs.
14. MTP and speculative-drafter choices: static immutable, co-trained BF16 default, explicit auxiliary scope, and atomic versioning.
15. Performance knobs, metrics, and the 5% gate.

Use this minimal public example verbatim:

```yaml
precision_policy:
  default: bf16
  scopes:
    - id: routed-experts-middle
      role: moe.routed_expert
      layers:
        exclude_first: 2
        exclude_last: 1
      rollout: mxfp8
```

- [ ] **Step 4: Run example, link, MyST, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_documentation_examples.py`

Run: `uvx --python 3.13 --from myst-parser myst-docutils-html --myst-highlight-code-blocks=false --halt=warning --exit-status=warning --validate docs/guides/precision-policy.md /dev/null`

Run: `uv run --no-sync bash tests/docs/Docs_Tests.sh`

Run: `git diff --check`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/precision-policy.md docs/fp8.md docs/guides/refit.md docs/index.md docs/guides/models/qwen/qwen3-5.md docs/guides/models/nemotron/nemotron-3.5-lightning.md docs/guides/models/nemotron/nemotron-3-super.md docs/guides/models/nemotron/nemotron-3-ultra.md tests/unit/precision_policy/test_documentation_examples.py
git commit -s -m "docs: explain semantic MXFP8 precision scopes"
```

### Task 16: Full Local Gate, Immutable Cluster Validation, and PR Decomposition Evidence

**Files:**
- Create: `tests/functional/precision_policy_matrix.sh`
- Create: `tests/functional/refit_transaction_fault_matrix.sh`
- Create: `tests/functional/refit_performance_matrix.sh`
- Create: `tests/fixtures/precision_policy/cluster_matrix.yaml`
- Test: `tests/unit/precision_policy/test_cluster_matrix.py`
- Create after runs: `docs/performance/semantic-precision-refit-validation.md`
- Modify: `tests/functional/L1_Functional_Tests_GB200_MXFP8.sh`
- Modify: `tests/test_suites/performance_gb200.txt`

**Interfaces:**
- Consumes: all implementation commits, pinned containers, exact model revisions, Lyris/Ptyche accounts, and the benchmark tool.
- Produces: reproducible correctness/fault/performance artifacts tied to branch, SHA, image digest, model revision, policy/plan digests, topology, raw samples, and job logs.

- [ ] **Step 1: Write dry-run validation scripts and failing metadata tests**

```python
def test_every_cluster_case_pins_reproducibility_metadata() -> None:
    matrix = yaml.safe_load(CLUSTER_MATRIX.read_text())
    for case in matrix["cases"]:
        assert case["model_revision"]
        assert case["container_digest"].startswith("sha256:")
        assert case["vllm_version"] in {"0.25.1", "0.28.0"}
        assert case["training_precision"] in {"bf16", "mxfp8"}
        assert case["rollout_precision"] == "mxfp8"
```

The scripts must support `--dry-run` and print the resolved cluster, account, branch, exact SHA, container digest, nodes/GPUs, model revision, policy/plan digest, command, time limit, and log directory without submitting.

- [ ] **Step 2: Run the complete local gate before any push**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy tests/unit/weight_sync tests/unit/models/megatron/test_precision_policy.py tests/unit/models/policy/test_mxfp8_refit_source.py tests/test_mxfp8_rollout_recipes.py`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_mixed_precision_refit.py --vllm-only`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/transaction.py nemo_rl/models/megatron/precision_policy.py nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/policy/workers/mxfp8_refit_source.py`

Run: `uv run --no-sync pre-commit run --all-files`

Expected: all commands pass. If any command fails, do not push or submit jobs.

- [ ] **Step 3: Create an immutable validation revision without touching PR heads**

```bash
git status --short
git log --show-signature -1
git push fork HEAD:refs/heads/validation/semantic-refit-r1
```

Record the returned SHA. Lyris and Ptyche must `git fetch fork validation/semantic-refit-r1`, `git checkout --detach <exact-sha>`, and verify `git rev-parse HEAD` before submission. If code changes, create `validation/semantic-refit-r2`; never force-update a revision used by a job.

- [ ] **Step 4: Stage pinned vLLM 0.25.1 and 0.28.0 containers and run the matrix**

Run each script with `--dry-run`, review its resolved submission, then submit. Cover both training modes for all five production models, mixed BF16/MXFP8 boundaries, specified TP/EP/PP/padding rows, repeated A→B→C numeric refits, fresh-load logprob comparison, injected binding/transfer/finalize/commit/silent-peer failures, and at least twenty paired steady-state performance samples where p95 is claimed. Monitor each new job for the required first five minutes and cancel/release resources immediately on a fatal failure.

- [ ] **Step 5: Evaluate hard gates and write the evidence report**

```python
assert every_numeric_case_passed
assert every_sync_and_async_fault_case_exited_nonzero_within_budget
assert refit_p50_ratio_upper_95ci <= 1.05
assert refit_p95_ratio_upper_95ci <= 1.05
assert generation_latency_ratio_upper_95ci <= 1.05
assert generation_throughput_ratio_lower_95ci >= 0.95
```

The report maps retained code to PRs #3477/#3630/#3659/#3669/#3907/#3908/#3909/#3294 and lists minimal restack ranges. Correctness/transaction commits remain separate from independently measured performance commits. Do not update an existing PR until every hard gate passes on both clusters.

- [ ] **Step 6: Commit only durable validation scripts and completed evidence**

```bash
git add tests/functional/precision_policy_matrix.sh tests/functional/refit_transaction_fault_matrix.sh tests/functional/refit_performance_matrix.sh tests/fixtures/precision_policy/cluster_matrix.yaml tests/unit/precision_policy/test_cluster_matrix.py tests/functional/L1_Functional_Tests_GB200_MXFP8.sh tests/test_suites/performance_gb200.txt docs/performance/semantic-precision-refit-validation.md
git commit -s -m "test(refit): validate semantic precision matrix"
```
