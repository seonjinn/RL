# Semantic Precision Policy and Transactional Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one positive semantic precision policy that configures MXFP8/BF16 training and rollout, drives mixed-layout refit without name/dtype guesses, fails the whole launcher immediately on refit failure, and preserves the fastest correct refit path.

**Architecture:** Phase 1 resolves each graph from effective configuration into a source-neutral semantic topology with an explicit exact decoder-layer universe, then compiles one immutable precision selection group before backend construction. Megatron/Transformer Engine and versioned vLLM factories construct only the requested BF16/MXFP8 realizations from that group. Phase 2 runs after construction: runtime producers normalize the realized Bridge, Automodel, Transformer Engine, or checkpoint sources and exact-project them onto the frozen topology; only then does the binder emit source ownership, aliases, cadence, and the final immutable precision intent group. The refit planner builds one-shot startup plans for frozen owners and every-version plans for mutable owners, while a transactional engine transfers canonical components, finalizes each physical owner once per cadence execution, and commits only after every required owning rank is ready.

**Tech Stack:** Python 3.13.13, Pydantic v2, frozen dataclasses, PyTorch, Ray, Megatron-Core/Bridge, Transformer Engine, vLLM 0.25.1 and 0.28.0, ModelOpt MXFP8, FlashInfer TRTLLM, pytest, Pyrefly, Ruff, MyST.

**Spec:** `docs/design-docs/semantic-precision-refit.md`

## Global Constraints

- Implement from immutable baseline `4601ba2c646ec40e5928c780fc0051a842328eba` on branch `codex/refit-semantic-policy-v2-20260903`; do not update any existing pull-request head while this plan is under validation.
- The public schema is version 1, `default` is always `bf16`, and the common recipe is a positive allow-list. Raw checkpoint/runtime parameter patterns are not a stable user interface.
- Canonical scopes use non-empty `roles: [...]`; the common routed-expert case is one scope with `roles: [moe.routed_expert]` and first/last exclusions. `advanced_match` and `addresses` are escape hatches. Singular `role` exists only in an explicit one-time legacy translator and never in canonical serialization.
- `CompiledPrecisionSelectionGroup` is the sole pre-construction precision source of truth and retains its frozen `ResolvedSelectionTopology`; the post-construction `CompiledPrecisionIntentGroup` retains that exact selection and preserves every selected domain and BF16 fence byte-for-byte. Generated TE matchers or vLLM include data are derived artifacts and must pass exact realized-module validation.
- `moe.routed_expert` means only main text-decoder routed expert gate/up/down kernels. It excludes shared experts, routers, latent projections, bias, MTP/draft graphs, attention, and embeddings.
- `attention.qkvo` means only main text-decoder token-attention Q/K/V/O projection kernels. It excludes MLA, KDA/GDN, sparse indexers, output gates, vision, bias, and MTP/draft graphs.
- Layer coordinates are zero-based. The default index space is `global_decoder`; `moe_ordinal` is explicit. `exclude_first` and `exclude_last` count in the selected index space and cannot consume the full domain.
- Every graph carries an explicit configuration-derived `DecoderLayerUniverse`. `global_decoder` is the complete physical decoder range even when boundary layers are dense or unselected; `moe_ordinal` is an exact contiguous, monotonic, one-to-one ordinal mapping onto all and only the MoE-bearing subset of that range. Main, MTP, and draft universes are independent and zero-based.
- Every mutable main-model tensor is accounted for and refitted. Every auxiliary graph instantiated by training is also present in the semantic bundle, even when it is mutable but training-only. `out_of_scope` is allowed only for source-proven frozen parameters, immutable auxiliary models, or backend-owned derived state with a typed reason.
- MTP and speculative drafters are separate semantic graphs. Their internal records distinguish graph kind, provenance, source mutability, rollout participation, derived refit requirement, and rank-local endpoint ownership; none of those fields is added to the public precision-policy selector.
- Every served training-parameter authority requires realized source and destination bindings on ranks derived to own its startup-only or every-version cadence. A graph with any mutable served training authority joins every-version atomic transactions, but only mutable training-source contributors repeat on the wire; its independent frozen training owners load once at startup, while checkpoint components and backend-derived values follow their own realization plans. Mixed realized destination groups follow the cadence-closure rule below. A mutable `not_served` auxiliary is valid and has no rollout/refit plan. Missing drafter storage is fatal only on a derived owning rank for a required cadence, and is valid on a non-owning PP rank or for a graph not served by rollout.
- A `served_from_source` graph must have a non-empty resolved semantic domain and reach at least one present training-runtime canonical value authority. Alias-only graphs are valid only when all aliases resolve to compatible existing owners. An empty reached-owner set or an absent required owner fails; a non-empty all-frozen training authority set is valid and derives a one-shot startup requirement.
- Cadence closes over realized destination owner/finalizer groups. Mixed groups cache verified immutable contributors once, refresh only mutable contributors, and require advertised native preservation or split/repack before exactly-once composition/finalization.
- A `served_from_checkpoint` auxiliary requires immutable graph/model identity, pinned resolved revision, checkpoint-content, model-configuration, semantic-domain digests, and typed evidence source. Its directly owned body contributes no source transfer but still owes destination startup/finalizer acknowledgement and artifact attestation. A cross-graph canonical-alias member follows its canonical value authority: training-runtime owners contribute their mutable/frozen cadence, while checkpoint owners never invent a training-source send. `loss_scaling_factor=0`, `detach_heads`, or a missing current gradient is not freeze evidence.
- Every supported checkpoint-serving destination attests the artifact actually loaded and must match every immutable-evidence field before serving; stale tags, caches, paths, or mismatched evidence are fatal for vLLM and supported static draft backends alike.
- Canonical aliases remain explicit graph members and reference one canonical
  source owner, so source export and wire transfer are never duplicated.
  Identical-storage relations need only immutable identity evidence, while
  synchronized replicas additionally require an exact live source-version
  fence before their canonical export. Destination
  load, finalization, and acknowledgement may be de-duplicated only when the
  endpoint adapter proves identical physical storage-owner and finalizer
  identity; otherwise the plan fans out to every distinct main/drafter
  destination owner and requires one acknowledgement from each.
- Phase 1 expands only declared semantic atomic groups when the user explicitly permits it; the fixed-point closure may not cross an explicit BF16 layer boundary. Task 7 later validates realized physical fused owners and either proves exact split/repack or preservation capability or fails preflight; it never changes the selection.
- Equal dtypes do not imply compatible layouts. Direct copy requires identical complete format/layout descriptors.
- Canonical load components remain distinct from padded, permuted, fused, shuffled, or flattened execution storage. A dirty owner is finalized exactly once per transaction.
- vLLM-specific imports and capability probes live only in versioned endpoint adapters. Unsupported versions or missing public capabilities fail before model construction; there is no process-global MXFP8 monkey patch.
- A refit worker returns a typed result. `None`, `False`, malformed results, exceptions, timeouts, and missing acknowledgements are failures.
- A detected refit failure keeps generation quiesced, poisons any partially updated destination, aborts communicators or terminates their owning workers within a bounded teardown budget, preserves the original phase/rank/cause, and makes sync and async launchers exit non-zero.
- Preserve the direct compatible-component path, persistent buffers, cached routes/permutations, batched expert conversion, and overlap. Do not scan model names or rebuild the semantic plan on each refit.
- Topology resolution, policy selection compilation, and runtime source discovery run once before communicators are created. The repeated refit hot path consumes cached bound plans and must never call any of them.
- The 95% upper confidence bound for treatment/baseline refit p50 and p95 latency is at most 1.05. Post-refit generation latency is at most 1.05 and throughput is at least 0.95 of the fastest correct baseline.
- Production end-to-end coverage includes Qwen3-30B-A3B, Qwen3.5-35B-A3B, NVIDIA Nemotron 3.5 Lightning 30B-A3B, Nemotron3 Super, and Nemotron3 Ultra for BF16-training→MXFP8-rollout and MXFP8-training→MXFP8-rollout with BF16 boundaries.
- Conformance coverage includes Nemotron 3 Nano, separate Kimi K2/K2.5/K3 fixtures, Qwen3.8 MoE/Flash-Next/dense-negative fixtures, and GLM-5.2. Unsupported model/runtime combinations fail closed.
- Source discovery is a post-construction, producer-normalized, graph-scoped phase. Exactly one immutable, versioned producer fingerprint, independently derived structurally ID-free expected-contributor authority, and completeness receipt bind each required runtime graph partition; a static checkpoint-served external draft has no runtime partition and instead owes exact destination attestation. The resolver retains the trusted contributor set and its original typed evidence through final validation, while requests and partitions carry only a constant-locator content-addressed evidence commitment; records never repeat the fingerprint, and semantic addresses never contain PP/TP/EP coordinates.
- Initial source schema IDs are exactly `hf.safetensors.header.v1`, `megatron.bridge.state-dict.v1`, `nemo-automodel.state-dict.v1`, and `transformer-engine.quantized-storage.v1`. Producer revision and normalization digest participate in `runtime_source_digest`/`intent_group_id`, but never select a family adapter or alter `semantic_structure_digest`/`selection_group_id`.
- Task 4 distinguishes thirteen logical `topology_case_id` values from fifteen physical `artifact_case_id` values. Lightning BF16/NVFP4 and A95B BF16/FP8 are distinct artifacts and sibling configuration or record evidence cannot be cross-spliced.
- The only Task 4 conformance labels are `topology facts`, `grammar micro-fixture`, and `full metadata conformance`, with the exact non-overclaiming meanings in the design. Task 4C completes at its executed topology/grammar tier; optional Task 4D receipts promote only the artifacts actually run. Production support is claimed only after source producer, TE realization, destination binding, mixed refit, transaction, numeric, and performance gates all pass.
- Family dispatch requires the exact outer/text model-type combination and a one-element architecture tuple/list. Missing, scalar, empty, multi-element, extra, or contradictory architecture data fails closed; revision is evidence, never an allowlist.
- The checked-in Task 2 built-in descriptors still use legacy/implicit encoding fields. Task 4A.1 must migrate `semantic.py` plus its semantic/compiler contract tests to the canonical BF16/MXFP8 serialization and commit that migration before Task 4B constructs or identity-tests `SOURCE_FORMAT_CATALOG`.
- No family classifier is implemented until literal tests for the canonical logical-format catalog, source-storage realization witnesses, and their independent reviews pass. Insufficient local axis/encoding/layout evidence creates a mandatory extraction gate, never a guessed or permissive descriptor.
- `FormatDescriptor` is never a native-storage descriptor. In particular, the canonical MXFP8 component grid does not describe Transformer Engine's padded compact or GEMM-swizzled uint8 carrier buffers. Producer-normalized source views and evidence-bound native storage realizations are separate identities; Task 7 must revalidate the live realization before selecting direct copy or a transform.
- New non-test Python and shell files carry the 2026 NVIDIA copyright header. New public functions and methods are fully typed and new typed modules are listed explicitly in `pyrefly.toml`.
- Follow strict RED/GREEN/refactor TDD. Every test names an observable break and uses literal, independently derived expected values.

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `nemo_rl/precision_policy/config.py` | YAML-loaded Pydantic schema and strict validation |
| `nemo_rl/precision_policy/semantic.py` | Frozen semantic addresses, roles, formats, `DecoderLayerUniverse`, resolved source-neutral graph/selection topology records, runtime-bound manifests, atomic groups, and orthogonal graph-lifecycle declarations |
| `nemo_rl/precision_policy/compiler.py` | Source-neutral positive selection, layer filtering, BF16 fences, atomic closure, selection-group generation and digest |
| `nemo_rl/precision_policy/topology.py` | Source-neutral topology-adapter protocol, registry, nested text-config resolution, runtime-source exact-projection classification, and complete accounting |
| `nemo_rl/precision_policy/source_discovery.py` | Pure source-schema IDs, producer fingerprints, graph partitions, contributor/source completeness receipts |
| `nemo_rl/precision_policy/source_storage.py` | Producer-normalized view to evidence-bound native-storage realization contracts |
| `nemo_rl/precision_policy/source_formats.py` | Evidence-backed canonical logical-format catalog |
| `nemo_rl/precision_policy/discovery_producers/checkpoint.py` | Safe index/header metadata normalization without weight payloads |
| `nemo_rl/precision_policy/discovery_producers/megatron_bridge.py` | Bridge/MCore conversion-task metadata normalization |
| `nemo_rl/precision_policy/discovery_producers/automodel.py` | Native Automodel state-dict metadata normalization before gathers/conversion |
| `nemo_rl/precision_policy/discovery_producers/transformer_engine.py` | Native TE quantized-storage metadata normalization |
| `nemo_rl/precision_policy/topology_resolver.py` | Task 4B-owned, standard-library-only Phase 1 `resolve_selection_topology()`; it never imports runtime source discovery or a framework producer |
| `nemo_rl/precision_policy/runtime_binding.py` | Task 4B-owned Phase 2 bulk request/result construction, selection-derived graph coverage, producer orchestration, exact projection, and `bind_runtime_source_intents()` |
| `nemo_rl/precision_policy/adapters/qwen.py` | Qwen3/Qwen3.5/Qwen3.8 semantic classification |
| `nemo_rl/precision_policy/adapters/nemotron.py` | Nano/Lightning/Super/Ultra semantic classification |
| `nemo_rl/precision_policy/adapters/kimi.py` | Kimi K2/K2.5/K3 manifest conformance and encoding declarations |
| `nemo_rl/precision_policy/adapters/glm.py` | GLM-5.2 manifest conformance |
| `nemo_rl/precision_policy/materialize.py` | One-time pre-construction selection injection and post-construction runtime-intent binding before communicators start |
| `nemo_rl/models/megatron/precision_policy.py` | TE recipe generation and realized source-binding validation |
| `nemo_rl/weight_sync/refit_plan.py` | Extensible ordered component bindings, transform loci, rank-local ownership, alias de-duplication, and execution schedules |
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
| `tests/metadata/precision_policy/test_full_metadata_conformance.py` | Explicit opt-in full-header artifact classification and resource gates |
| `tools/precision_policy_metadata_conformance.py` | Optional per-artifact metadata promotion runner and exact count/resource receipt |

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
- Produces: `PrecisionPolicyConfig`, `PrecisionScopeConfig`, `LayerSelectorConfig`, `AdvancedMatchConfig` with separate `graph_instance_id` and `semantic_graph_path` predicates, qualified `SemanticAddressSelectorConfig`, `PrecisionName`, and `parse_precision_policy(value: object) -> PrecisionPolicyConfig | None`.

- [ ] **Step 1: Write failing schema tests**

```python
def test_minimal_routed_scope_defaults_training_to_bf16() -> None:
    policy = PrecisionPolicyConfig.model_validate({
        "scopes": [{
            "id": "routed-middle",
            "roles": ["moe.routed_expert"],
            "layers": {"exclude_first": 2, "exclude_last": 1},
            "rollout": "mxfp8",
        }]
    })
    assert policy.schema_version == 1
    assert policy.default == "bf16"
    assert policy.scopes[0].training is None
    assert policy.scopes[0].layers is not None
    assert policy.scopes[0].layers.index_space == "global_decoder"

@pytest.mark.parametrize("bad", [
    {"default": "mxfp8", "scopes": []},
    {"scopes": [{"id": "x", "roles": ["moe.routed_expert"]}]},
    {"scopes": [{"id": "x", "roles": ["moe.routed_expert"], "advanced_match": {}, "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "roles": ["moe.routed_expert"], "layers": {"exclude_first": -1}, "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "roles": [], "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "roles": ["moe.routed_expert", "moe.routed_expert"], "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "role": "moe.routed_expert", "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "role_typo": "moe.routed_expert", "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "advanced_match": {"graph": "text.decoder"}, "rollout": "mxfp8"}]},
    {"scopes": [{"id": "x", "semantic_addresses": {"semantic_ids": ["text.decoder.x"]}, "rollout": "mxfp8"}]},
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
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
SemanticAttributeScalar = str | int | FiniteFloat | bool

class SemanticAddressSelectorConfig(BaseModel, extra="allow"):
    graph_instance_id: str
    semantic_graph_path: str
    semantic_id: str

class AdvancedMatchConfig(BaseModel, extra="allow"):
    graph_instance_id: SemanticStringPredicate | None = None
    semantic_graph_path: SemanticStringPredicate | None = None
    model_part: SemanticStringPredicate | None = None
    module_kind: SemanticStringPredicate | None = None
    parameter_role: SemanticStringPredicate | None = None
    attributes: dict[str, SemanticAttributePredicate] = Field(default_factory=dict)

class LayerSelectorConfig(BaseModel, extra="allow"):
    index_space: LayerIndexSpace = "global_decoder"
    exclude_first: StrictNonNegativeInt = 0
    exclude_last: StrictNonNegativeInt = 0

class PrecisionScopeConfig(BaseModel, extra="allow"):
    id: str
    roles: list[str] | None = None
    advanced_match: AdvancedMatchConfig | None = None
    addresses: list[SemanticAddressSelectorConfig] | None = None
    layers: LayerSelectorConfig | None = None
    training: PrecisionName | None = None
    rollout: PrecisionName | None = None
    atomic_conflict: AtomicConflictMode | None = None

class PrecisionPolicyConfig(BaseModel, extra="allow"):
    schema_version: Literal[1] = 1
    default: Literal["bf16"] = "bf16"
    require_match: StrictBool = True
    atomic_conflict: AtomicConflictMode = "error"
    scopes: list[PrecisionScopeConfig]
```

Validate `schema_version` in `mode="before"` and require `type(value) is int`
before applying `Literal[1]`; Pydantic literal equality otherwise accepts
coercive boolean or floating-point values. Omitted `layers` remains `None`,
whereas explicit `{}` remains a structural zero-exclusion selector after
serialization and reparsing. Omitted scope-level `atomic_conflict` remains
`None` and inherits the policy-level default during Task 3 compilation; an
explicit scope value overrides it. Semantic floating-point predicate values
must be finite, while finite floats, integers, and booleans preserve their
distinct runtime types.

Each model validator rejects undocumented `model_extra`; the scope validator enforces a non-empty unique `id`, exactly one of non-empty duplicate-free `roles`, `advanced_match`, or non-empty `addresses`, and at least one non-BF16 endpoint request. `roles` is the canonical positive allow-list even for one common role. The singular `role` spelling is rejected by this schema and may be accepted only by a separately tested legacy migration translator that cannot coexist with `roles`. Address records require `graph_instance_id` equal to `main` or prefixed by `mtp.`/`draft.`, require the semantic ID to use one canonical path-prefixed rendering, require `semantic_graph_path` to match that rendering, and reject duplicate `(graph_instance_id, semantic_id)` pairs. The ambiguous legacy fields `advanced_match.graph` and `semantic_addresses` are rejected. Add `precision_policy: NotRequired[PrecisionPolicyConfig]` to `PolicyConfig`.

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

- [ ] **Step 6: Apply the required selector-identity follow-up before Task 2**

Task 1 commit `89bfb3956041639a86d8baefb791ecfcea93c638` predates the
qualified identity decision. Update `nemo_rl/precision_policy/config.py` and
`tests/unit/precision_policy/test_config.py` to replace `advanced_match.graph`
with the two predicates above and replace the nested unqualified
`semantic_addresses` selector with typed `addresses` records. Run the Step 4
gates and commit the correction separately:

```bash
git add nemo_rl/precision_policy/config.py tests/unit/precision_policy/test_config.py
git commit -s -m "fix(precision): qualify semantic selectors"
```

Do not start Task 2 until this follow-up is green.

- [ ] **Step 7: Migrate the implemented singular selector to canonical `roles` before Phase 1 integration**

The existing implementation predates the approved positive role-list syntax.
Add RED tests proving one-element and multi-role lists parse in canonical order,
duplicates and empty lists fail, `require_match=True` applies independently to
every listed role, and raw singular `role` fails strict parsing. If a legacy
recipe translator accepts singular `role`, test that it rewrites once to
`roles: [value]`, emits only plural serialization, and rejects simultaneous
`role` and `roles`. Update all policy/compiler fixtures and documentation
examples to plural form. Run the Step 4 gates and commit this migration before
Task 3 is rewritten for `CompiledPrecisionSelectionGroup` or Task 5 injects a
selection into endpoint construction.

### Task 2: Semantic Manifest, Roles, Formats, and Complete Accounting

**Files:**
- Create: `nemo_rl/precision_policy/semantic.py`
- Test: `tests/unit/precision_policy/test_semantic.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: source-neutral logical model facts, authoritative compact runtime-bound `ParameterInventory` values, topology-adapter `RoleExpectedDomain` values, and the complete `ExpectedGraphDeclaration` set derived from training/runtime configuration before adaptation.
- Produces: the normative frozen records in Step 3: source-neutral `DecoderLayerUniverse`, `SelectionTopologyEntry`, `ResolvedGraphTopology`, and `ResolvedSelectionTopology`; `SemanticAddress`, `SemanticTensor`, `SemanticTensorFamily`, `SemanticInventoryMember`, `RoleDefinition`/`RoleExpectedDomain`, logical `FormatDescriptor`, typed extensible `ComponentRole` plus `ComponentDescriptor`, `AtomicGroupParticipant`/`AtomicGroup`, typed `OutOfScopeReason`/`OutOfScopeTensor`, compact qualified `OwnerFamilyReference`/`OwnerFamilyBinding`/`SemanticOwnership`, `EvidenceSourceKind`/`EvidenceSource`, `SourceSynchronizationBoundary`/`SourceReplicaSynchronizationEvidence`, the nonnullable `IdenticalStorageSourceAliasContract | SynchronizedReplicaSourceAliasContract` union, `SourceOwnerInventoryEntry`, `ParameterInventoryEntry`/`ParameterInventory`, `GraphKind`, `GraphProvenance`, `ValueProvenance`, `SourceMutability`, `RolloutParticipation`, transient derived `RefitRequirement`, composite `GraphLifecycle`, `ImmutableAuxiliaryEvidence`, `ExpectedGraphDeclaration`, topology-independent `AuxiliaryGraphDeclaration`, `SemanticGraphManifest`, and the schema-bound `SemanticManifestBundle.validate_complete()`.

`ResolvedSelectionTopology` is intentionally not a partially populated
`SemanticManifestBundle`: it has no source format, source owner, mutability,
native realization, producer fingerprint, alias, or cadence field. Runtime
binding may enrich it only through exact projection.

`SemanticGraphManifest.graph_instance_id` is runtime instance identity (`main`,
`mtp.0`, `draft.external`). `SemanticAddress.semantic_graph_path` is the logical
role domain (`text.decoder`, `text.embedding`, `auxiliary.mtp`,
`draft.decoder`). `semantic_id` has one canonical rendering beginning with that
path, and `(graph_instance_id, semantic_id)` is the canonical tensor identity.
Exactly-one-MAIN and bundle-completeness checks use instance
identity/lifecycle; built-in role and layer matching use `GraphKind` plus
semantic graph path. A MAIN manifest can therefore contain both `text.decoder`
and `text.embedding` addresses.

- [ ] **Step 1: Write failing semantic-contract tests**

```python
def test_routed_expert_role_excludes_shared_router_and_auxiliary() -> None:
    topology = compact_explicit_role_topology_fixture()
    role = topology.role_definition(1, "moe.routed_expert")
    assert role.matching_inventory_entry_ids(topology) == (
        "main-routed-gate",
    )
    role.validate_expected_domain(topology)

def test_mutable_main_tensor_cannot_hide_out_of_scope() -> None:
    with pytest.raises(ValueError, match="mutable main-model"):
        bundle_with_out_of_scope_entry(
            ParameterInventoryEntry("main-kernel", "main", mutable_tensor(), ValueProvenance.TRAINING_PARAMETER),
            OutOfScopeTensor("main-kernel", OutOfScopeReason.SOURCE_PROVEN_FROZEN),
        ).validate_complete()

def test_mutable_training_only_mtp_is_complete_but_requires_no_refit() -> None:
    bundle = bundle_with_training_mtp(rollout_participation="not_served")
    bundle.validate_complete()
    assert bundle.inventory.owner_family("mtp.0", "mtp-head").source_mutability == SourceMutability.MUTABLE
    assert bundle.refit_requirement("mtp.0") == RefitRequirement.NONE

def test_kimi_expert_inventory_stays_compact_during_validation() -> None:
    bundle = compact_kimi_k25_expert_bundle()
    assert len(bundle.inventory.entries) == 3  # fixed gate/up/down families
    assert bundle.inventory.logical_cardinality == 60 * 384 * 3
    assert all(isinstance(entry.member, SemanticTensorFamily) for entry in bundle.inventory.entries)
    with forbid_semantic_member_materialization():
        bundle.validate_complete()
    assert len(bundle.inventory.entries) == 3
```

Also test that omitting any instantiated training MTP/drafter fails bundle
validation; exactly one expected instance is `GraphKind.MAIN`; the instance-ID
grammar and path-prefixed semantic-ID invariant; direct canonical-owner alias
binding and rejection of alias-to-alias targets; typed whole-entry out-of-scope
accounting; and exact declaration ↔ manifest ↔ inventory accounting. A
source-served graph with any
mutable owner derives `every_version`, an all-frozen source-served graph derives
`initial_only`, and a mixed graph assigns startup cadence to frozen owners and
every-version cadence to mutable owners. A checkpoint-served graph requires
graph/model identity, pinned revision, content/configuration/semantic-domain
digests, and evidence source. `loss_scaling_factor=0`, `detach_heads`, or absent
current gradients cannot derive `frozen`.

Add explicit failures for a `served_from_source` graph with an empty logically
resolved compact domain, no canonical source owner, or an owner marked
`absent`. Prove that `all([])` does not derive `initial_only`. Add a valid
alias-only graph whose entire non-empty domain binds directly to an existing
compatible canonical source owner, and fail missing targets, alias-to-alias
targets, and projected-domain/shape/axes/dtype/format-incompatible aliases.

Add compact-family tests for a correlated `LayerMember` domain, independent
expert axes, separate fixed-attribute gate/up/down and Q/K/V/O families,
multiple complete families for ragged layer/expert domains, exact overlap with
an explicit tensor or another family, qualified owner-family aliasing, and
complete inventory union. Assert compact entry count and logical cardinality
independently, and make validation/role matching fail the test if it invokes a
full member renderer or stores expanded members. Verify `OutOfScopeTensor`
claims exactly one complete inventory entry ID and cannot describe a partial
family. Reject arbitrary templates/regex/globs, projection as a generic
attribute axis, correlated Cartesian layer coordinates, partial family
coverage, and any implementation that persists full family expansion.

Add descriptor tests that pin BF16 to one
`logical_values/bfloat16/plain_bfloat16` component and MXFP8 to ordered
`values/e4m3/mxfp8_e4m3_values` plus
`block_scales/e8m0/mxfp8_e8m0_scale`, with explicit
`output_features /1 EXACT` and `input_features /32 CEIL` axes. Assert stable
`format_id` uniqueness and canonical
serialized equality; the same ID with any different family, role, dtype,
encoding, axis, divisor, or rounding is rejected. Block-FP8, NVFP4, and MXFP4
must use distinct adapter-advertised format IDs and component families when
supported; do not add invented built-in profiles merely to satisfy the test.
These are the desired canonical assertions, not a statement about the current
checkout: the completed Task 2 implementation still uses `None` for
BF16/MXFP8 value encodings, `mxfp8_scale` for scales, and the default
output-axis divisor/rule. Its MXFP8 test pins those legacy fields, while its
BF16 test omits encoding from the assertion. Task 4A.1 below owns the explicit
compatibility migration before Task 4B.
Add compact pointwise atomic-group tests in which one group domain expresses
gate/up/down per layer/expert and another expresses Q/K/V/O per layer without
rendering every group instance. Reject an empty group or participant domain,
a cross-graph participant, and an incomplete or ambiguous domain projection.

- [ ] **Step 2: Run the tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py`

Expected: import failure for `nemo_rl.precision_policy.semantic`.

- [ ] **Step 3: Implement immutable semantic records and exact built-in roles**

```python
class GraphKind(StrEnum):
    MAIN = "main"
    MTP = "mtp"
    SPECULATIVE_DRAFTER = "speculative_drafter"

class GraphProvenance(StrEnum):
    TRAINING_RUNTIME = "training_runtime"
    MODEL_CHECKPOINT = "model_checkpoint"
    EXTERNAL_CHECKPOINT = "external_checkpoint"

class ValueProvenance(StrEnum):
    TRAINING_PARAMETER = "training_parameter"
    CHECKPOINT_ENCODING_COMPONENT = "checkpoint_encoding_component"
    BACKEND_DERIVED = "backend_derived"
    CANONICAL_ALIAS = "canonical_alias"

class SourceMutability(StrEnum):
    MUTABLE = "mutable"
    FROZEN = "frozen"
    ABSENT = "absent"

class RolloutParticipation(StrEnum):
    NOT_SERVED = "not_served"
    SERVED_FROM_SOURCE = "served_from_source"
    SERVED_FROM_CHECKPOINT = "served_from_checkpoint"

class RefitRequirement(StrEnum):
    NONE = "none"
    INITIAL_ONLY = "initial_only"
    EVERY_VERSION = "every_version"

class SourceSynchronizationBoundary(StrEnum):
    SOURCE_VERSION_READY = "source_version_ready"

class AxisExtentRounding(StrEnum):
    EXACT = "exact"
    CEIL = "ceil"

class EvidenceSourceKind(StrEnum):
    RUNTIME_INVENTORY = "runtime_inventory"
    PINNED_CHECKPOINT_MANIFEST = "pinned_checkpoint_manifest"
    CONTENT_ADDRESS = "content_address"

class OutOfScopeReason(StrEnum):
    SOURCE_PROVEN_FROZEN = "source_proven_frozen"
    IMMUTABLE_AUXILIARY = "immutable_auxiliary"
    BACKEND_DERIVED_STATE = "backend_derived_state"

class AtomicGroupKind(StrEnum):
    PRECISION = "precision"

type PredicateScalar = str | int | float | bool
ComponentRole = NewType("ComponentRole", str)
LOGICAL_VALUES = ComponentRole("logical_values")
VALUES = ComponentRole("values")
BLOCK_SCALES = ComponentRole("block_scales")

@dataclass(frozen=True, slots=True)
class AttributePredicate:
    name: str
    allowed_values: tuple[PredicateScalar, ...]

@dataclass(frozen=True, slots=True)
class SemanticPredicate:
    graph_kinds: tuple[GraphKind, ...]
    semantic_graph_paths: tuple[str, ...]
    model_parts: tuple[str, ...]
    module_kinds: tuple[str, ...]
    attributes: tuple[AttributePredicate, ...]
    parameter_roles: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class LogicalComponentAxisSpec:
    logical_axis: str
    divisor: int = 1
    rounding: AxisExtentRounding = AxisExtentRounding.EXACT

@dataclass(frozen=True, slots=True)
class LiteralComponentAxisSpec:
    axis_name: str
    extent: int

type ComponentAxisSpec = LogicalComponentAxisSpec | LiteralComponentAxisSpec

@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    role: ComponentRole
    dtype: str
    encoding: str | None = None
    component_axes: tuple[ComponentAxisSpec, ...] | None = None

@dataclass(frozen=True, slots=True)
class FormatDescriptor:
    format_id: str
    family: str
    components: tuple[ComponentDescriptor, ...]

# Post-Task-4A.1 canonical target; the checked-in Task 2 baseline is legacy.
BF16_FORMAT = FormatDescriptor(
    format_id="bf16.logical.v1",
    family="bf16",
    components=(
        ComponentDescriptor(
            role=LOGICAL_VALUES,
            dtype="bfloat16",
            encoding="plain_bfloat16",
        ),
    ),
)
MXFP8_FORMAT = FormatDescriptor(
    format_id="mxfp8.e4m3-e8m0-block32-input-features.v1",
    family="mxfp8",
    components=(
        ComponentDescriptor(
            role=VALUES,
            dtype="e4m3",
            encoding="mxfp8_e4m3_values",
        ),
        ComponentDescriptor(
            role=BLOCK_SCALES,
            dtype="e8m0",
            encoding="mxfp8_e8m0_scale",
            component_axes=(
                LogicalComponentAxisSpec(
                    "output_features",
                    divisor=1,
                    rounding=AxisExtentRounding.EXACT,
                ),
                LogicalComponentAxisSpec(
                    "input_features",
                    divisor=32,
                    rounding=AxisExtentRounding.CEIL,
                ),
            ),
        ),
    ),
)

@dataclass(frozen=True, slots=True)
class SemanticAddress:
    semantic_id: str
    semantic_graph_path: str
    model_part: str
    module_kind: str
    attributes: tuple[tuple[str, str | int | float | bool], ...]
    parameter_role: str
    global_decoder_layer: int | None
    moe_ordinal: int | None

@dataclass(frozen=True, slots=True)
class LayerMember:
    global_decoder_layer: int
    moe_ordinal: int | None

@dataclass(frozen=True, slots=True)
class DecoderLayerUniverse:
    global_decoder_layers: tuple[int, ...]
    moe_global_decoder_layers_by_ordinal: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class LayerDomain:
    members: tuple[LayerMember, ...]

@dataclass(frozen=True, slots=True)
class AxisDomain:
    name: str
    members: tuple[int | str, ...]

@dataclass(frozen=True, slots=True)
class FamilyIndexDomain:
    layer_domain: LayerDomain | None
    independent_axes: tuple[AxisDomain, ...]

@dataclass(frozen=True, slots=True)
class LiteralPathSegment:
    value: str

@dataclass(frozen=True, slots=True)
class IndexPathSegment:
    axis_name: str

type SemanticPathSegment = LiteralPathSegment | IndexPathSegment

@dataclass(frozen=True, slots=True)
class SemanticAddressPattern:
    semantic_graph_path: str
    path_segments: tuple[SemanticPathSegment, ...]
    model_part: str
    module_kind: str
    attributes: tuple[tuple[str, str | int | float | bool], ...]
    parameter_role: str

@dataclass(frozen=True, slots=True)
class OwnerFamilyReference:
    graph_instance_id: str
    owner_family_id: str

@dataclass(frozen=True, slots=True)
class AxisProjection:
    member_axis: str
    owner_axis: str

@dataclass(frozen=True, slots=True)
class OwnerFamilyBinding:
    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    member_domain: FamilyIndexDomain
    member_to_owner_axes: tuple[AxisProjection, ...]
    member_to_value_axes: tuple[AxisProjection, ...]

@dataclass(frozen=True, slots=True)
class SemanticOwnership:
    binding: OwnerFamilyBinding

@dataclass(frozen=True, slots=True)
class SemanticTensor:
    address: SemanticAddress
    format: FormatDescriptor
    logical_dtype: str
    logical_shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    ownership: SemanticOwnership

@dataclass(frozen=True, slots=True)
class SemanticTensorFamily:
    pattern: SemanticAddressPattern
    domain: FamilyIndexDomain
    format: FormatDescriptor
    logical_dtype: str
    logical_shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    ownership: SemanticOwnership

type SemanticInventoryMember = SemanticTensor | SemanticTensorFamily

@dataclass(frozen=True, slots=True)
class EvidenceSource:
    kind: EvidenceSourceKind
    locator: str
    digest: str

@dataclass(frozen=True, slots=True)
class SourceReplicaSynchronizationEvidence:
    replica_group_id: str
    boundary: SourceSynchronizationBoundary
    evidence_source: EvidenceSource

@dataclass(frozen=True, slots=True)
class IdenticalStorageSourceAliasContract:
    alias_entry_id: str
    canonical_value_entry_id: str
    canonical_owner_family: OwnerFamilyReference
    component_role: ComponentRole
    alias_domain: FamilyIndexDomain
    canonical_domain: FamilyIndexDomain
    alias_to_canonical_axes: tuple[AxisProjection, ...]
    storage_identity_evidence: EvidenceSource

@dataclass(frozen=True, slots=True)
class SynchronizedReplicaSourceAliasContract:
    alias_entry_id: str
    canonical_value_entry_id: str
    canonical_owner_family: OwnerFamilyReference
    component_role: ComponentRole
    alias_domain: FamilyIndexDomain
    canonical_domain: FamilyIndexDomain
    alias_to_canonical_axes: tuple[AxisProjection, ...]
    synchronization: SourceReplicaSynchronizationEvidence

type SourceAliasContract = (
    IdenticalStorageSourceAliasContract
    | SynchronizedReplicaSourceAliasContract
)

@dataclass(frozen=True, slots=True)
class ParameterInventoryEntry:
    entry_id: str
    graph_instance_id: str
    member: SemanticInventoryMember
    value_provenance: ValueProvenance

@dataclass(frozen=True, slots=True)
class SourceOwnerInventoryEntry:
    owner_family: OwnerFamilyReference
    domain: FamilyIndexDomain
    source_mutability: SourceMutability
    mutability_evidence_source: EvidenceSource

@dataclass(frozen=True, slots=True)
class ParameterInventory:
    owners: tuple[SourceOwnerInventoryEntry, ...]
    entries: tuple[ParameterInventoryEntry, ...]

@dataclass(frozen=True, slots=True)
class OutOfScopeTensor:
    inventory_entry_id: str
    reason: OutOfScopeReason

@dataclass(frozen=True, slots=True)
class RoleExpectedDomain:
    role_name: str
    inventory_entry_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class RoleDefinition:
    schema_version: int
    role_name: str
    predicate: SemanticPredicate
    expected_domain: RoleExpectedDomain

    def matching_inventory_entry_ids(
        self, topology: "ResolvedSelectionTopology"
    ) -> tuple[str, ...]: ...

    def validate_expected_domain(self, topology: "ResolvedSelectionTopology") -> None: ...

@dataclass(frozen=True, slots=True)
class AtomicGroupParticipant:
    inventory_entry_id: str
    participant_domain: FamilyIndexDomain
    group_to_participant_axes: tuple[AxisProjection, ...]

@dataclass(frozen=True, slots=True)
class AtomicGroup:
    group_id: str
    graph_instance_id: str
    kind: AtomicGroupKind
    group_domain: FamilyIndexDomain
    participants: tuple[AtomicGroupParticipant, ...]

@dataclass(frozen=True, slots=True)
class ImmutableAuxiliaryEvidence:
    graph_instance_id: str
    model_identity: str
    pinned_checkpoint_revision: str
    checkpoint_content_digest: str
    model_config_digest: str
    semantic_domain_digest: str
    evidence_source: EvidenceSource

@dataclass(frozen=True, slots=True)
class GraphLifecycle:
    graph_kind: GraphKind
    graph_provenance: GraphProvenance
    rollout_participation: RolloutParticipation
    immutable_evidence: ImmutableAuxiliaryEvidence | None = None

@dataclass(frozen=True, slots=True)
class ExpectedGraphDeclaration:
    graph_instance_id: str
    model_identity: str
    lifecycle: GraphLifecycle

@dataclass(frozen=True, slots=True)
class AuxiliaryGraphDeclaration:
    graph_instance_id: str
    model_identity: str
    lifecycle: GraphLifecycle

@dataclass(frozen=True, slots=True)
class SelectionTopologyEntry:
    entry_id: str
    graph_instance_id: str
    pattern: SemanticAddressPattern
    domain: FamilyIndexDomain
    logical_dtype: str
    logical_shape: tuple[int, ...]
    logical_axes: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ResolvedGraphTopology:
    declaration: ExpectedGraphDeclaration
    model_family: str
    resolved_model_revision: str
    adapter_id: str
    decoder_layer_universe: DecoderLayerUniverse
    entries: tuple[SelectionTopologyEntry, ...]
    role_definitions: tuple[RoleDefinition, ...]
    atomic_groups: tuple[AtomicGroup, ...]

@dataclass(frozen=True, slots=True)
class ResolvedSelectionTopology:
    schema_version: int
    graphs: tuple[ResolvedGraphTopology, ...]
    role_definitions: tuple[RoleDefinition, ...]
    semantic_structure_digest: str

    def role_registry(self) -> tuple[RoleDefinition, ...]: ...
    def role_definition(self, schema_version: int, role_name: str) -> RoleDefinition: ...
    def validate_complete(self) -> None: ...

@dataclass(frozen=True, slots=True)
class SemanticGraphManifest:
    model_family: str
    model_revision: str
    graph_instance_id: str
    lifecycle: GraphLifecycle
    inventory_entry_ids: tuple[str, ...]
    atomic_groups: tuple[AtomicGroup, ...] = ()
    out_of_scope: tuple[OutOfScopeTensor, ...] = ()

@dataclass(frozen=True, slots=True)
class SemanticManifestBundle:
    schema_version: int
    expected_graphs: tuple[ExpectedGraphDeclaration, ...]
    manifests: tuple[SemanticGraphManifest, ...]
    inventory: ParameterInventory
    role_definitions: tuple[RoleDefinition, ...]
    source_alias_contracts: tuple[SourceAliasContract, ...] = ()

    def role_definition(self, schema_version: int, role_name: str) -> RoleDefinition: ...
```

This Step 3 record shape is normative rather than illustrative. A scalar uses
`FamilyIndexDomain(layer_domain=None, independent_axes=())`, whose cardinality
is one; it does not use a separate scalar owner-inventory schema. Literal path
segments are validated canonical atoms and index segments can reference only a
declared domain axis. `entry_id` is unique and stable for accounting but is not
rendered into `semantic_id` and cannot be used as tensor identity.

`DecoderLayerUniverse.global_decoder_layers` equals
`tuple(range(decoder_layer_count))` exactly. Its MoE tuple is ordered by
the implicit contiguous ordinal domain `range(moe_layer_count)`, contains all
and only the MoE-bearing members of that physical universe, and maps them
strictly monotonically and one-to-one with no duplicate or reversal. Only the
ordinal keys must be gap-free; mapped global indices may skip dense layers.
`ResolvedSelectionTopology` contains exactly one graph for every declaration,
orders them canonically, merges the graph contributions into one deterministic
validated `role_definitions` registry, validates all entry/domain/role/atomic
references within that set, and computes `semantic_structure_digest` from
every field except the digest itself. Reflection tests reject the Phase 2-only
concepts `format`, `source_owner`, `source_mutability`, `native_storage`,
`producer_fingerprint`, `source_alias`, and `cadence` anywhere in this frozen
object graph.

After the mandatory Task 4A.1 migration, the enum values and the two built-in
descriptors above are exhaustive for schema version 1. BF16 has exactly one ordered
`logical_values/bfloat16/plain_bfloat16` component. MXFP8 has exactly ordered
`values/e4m3/mxfp8_e4m3_values` then
`block_scales/e8m0/mxfp8_e8m0_scale`, with explicit
`output_features /1 EXACT` and `input_features /32 CEIL` component axes.
`format_id` is a canonical descriptor identity:
validation rejects a repeated ID with any non-equal family or component
contract. Other encodings exist only as complete, distinct adapter-advertised
descriptors. Every `AtomicGroup` has a non-empty logical
group domain and non-empty participants whose compact inventory entry IDs all
belong to its declared graph. Each participant has a non-empty domain within
its inventory entry and a total, unambiguous projection from each group point.
This expresses pointwise gate/up/down or Q/K/V/O atomicity without expansion.
Groups express only semantic precision topology; physical load atomicity,
owners, layouts, and finalizers are realized in Task 7 and cannot appear here.
`RefitRequirement` is a return type of validated derivation and is never a
field of `GraphLifecycle`, a declaration, manifest, or inventory record.
`RoleDefinition` methods require the complete `ResolvedSelectionTopology` so
graph kind, semantic path, model facets, and entry membership cannot be lost.
`FormatDescriptor` describes only the logical encoding and ordered canonical
components. It never contains a backend layout, physical shape, placement,
padding, permutation, or runtime-storage fact; Task 7 owns those records.
For a component, `component_axes=None` is identity over the member's ordered
logical axes and extents, while the explicit empty tuple is a true rank-zero
scalar whose extent product is one. A `LogicalComponentAxisSpec` preserves its
named logical axis and divides its extent: `EXACT` requires zero remainder and
`CEIL` uses integer ceiling division. A `LiteralComponentAxisSpec` adds a
component-only fixed positive axis. Explicit axes retain their declared order.
After Task 4A.2, every classification edge operates on a producer-normalized
source view. Its region cardinality must equal the
output member-domain cardinality multiplied by the product of the resolved
component-axis extents. A scalar metadata component is therefore `()`, never
an invented `(1,)` axis. Raw carrier shape, padding, flattening, and swizzle
are validated separately by the source-storage realization inventory and do
not enter this compact semantic region algebra.

`GraphLifecycle` stores graph facts, not source-owner state. `SourceMutability`
lives on compact qualified owner-family domains in `ParameterInventory`;
canonical-alias semantic members reference those domains through
`SemanticOwnership`. `RefitRequirement` is computed transiently, never stored
as an independent input: first resolve each participating member to its direct
canonical value authority, then join that authority with the lazily resolved
owner domain. A mutable training parameter derives `every_version` and a
proven-frozen training parameter derives `initial_only`; a checkpoint encoding
component never creates a trainer send and instead requires checkpoint
load/attestation, while a backend-derived value requires its advertised
dependency. `not_served` members derive `none`. This authority-first rule also
applies to aliases regardless of whether their member graph is
`served_from_source` or `served_from_checkpoint`. A checkpoint-served graph's
direct body must remain checkpoint/backend-owned; a directly owned training
parameter is an inconsistent lifecycle and fails closed. The graph summary is
the maximum of its served member obligations.
The Phase 2 intent group may carry these typed semantic owner requirements,
while Task 7 alone realizes physical schedules. Neither belongs to the Phase 1
selection, and neither may be inferred from precision policy, loss
configuration, or current gradients.

`SourceMutability.ABSENT` records an explicit producer-normalized source-view
disposition with no native-storage realization; it is not value provenance and
`ValueProvenance` intentionally has no absent member. A validated canonical
semantic owner for `served_from_source` cannot remain `ABSENT`.

`OutOfScopeTensor` is accounting-only: it contributes neither an endpoint
precision assignment nor a source realization request for itself. Its direct
canonical value may still be carried as source metadata when an in-scope
served alias in another graph references it; exclusion is destination-local,
not removal of canonical source authority. A frozen value that must initialize
its own rollout member remains in scope and inherits default BF16. Therefore
`served_from_source` must reach at least one in-scope training authority, every
source request must trace to an in-scope destination member, and Task 7 must
prove or reject partial realization when one physical owner fuses in-scope and
excluded entries.

For `served_from_source`, derive cadence only after logical/lazy family-domain
and alias resolution, never full materialization. Its semantic domain must be
non-empty and must reach at least one present training-runtime canonical value
authority. Reject `SourceMutability.ABSENT`, unresolved targets, and a vacuous
reached-owner set; a non-empty all-frozen set derives `initial_only`. An
alias-only graph is valid only when every compact alias domain
binds directly to an existing canonical source-owner domain and names one
compatible non-alias `canonical_value_entry_id`, with shape, axes, dtype,
format, and exact member-to-owner and member-to-value index mappings.
`ValueProvenance.CANONICAL_ALIAS` is the sole logical alias marker;
alias-to-alias chains are not representable. For every non-alias entry,
`canonical_value_entry_id` is the
entry's own ID and its canonical source owner must also be present. The alias
retains graph membership but does not create a local source owner or transfer.
`SemanticManifestBundle.validate_complete()` additionally requires a bijective
accounting cover between every canonical-alias entry's complete compact domain
for every ordered format component and its normalized source-alias contracts.
Claims for one `(alias entry, component)` must be in-domain, pairwise disjoint,
and gap-free. Each claim must resolve to the binding's exact direct non-alias
target, canonical owner, compatible target subdomain, and total projection.
Reject orphan contracts, contracts attached to direct values, duplicate or
overlapping claims, gaps, target/owner/component/projection mismatches, and
conflicting relation kinds over the same subdomain. Different relation kinds
may describe disjoint components or subdomains when separately evidenced.
The Phase 2 binder revalidates this bundle and its exact projection once before
emitting `runtime_source_digest` or an intent.

`SemanticManifestBundle` contains exactly one `GraphKind.MAIN` instance, every
auxiliary graph instantiated by training (including mutable training-only
graphs), and every rollout-only static graph declaration. Its authoritative
`expected_graphs` field must match manifests bijectively. Define the built-in
BF16 descriptor as one `logical_values/bfloat16` component and MXFP8 as
`values/e4m3` plus `block_scales/e8m0` block-32. When an adapter supports
block-FP8, NVFP4, or MXFP4,
it must advertise distinct exact format IDs and component families; do not
invent generic built-in profiles for unsupported encodings. Reject duplicate
canonical `(graph_instance_id, semantic_id)`
keys, a semantic ID whose rendered prefix disagrees with
`semantic_graph_path`, unknown logical axes, unqualified/duplicate ownership,
an alias binding without a compatible direct member on its canonical owner,
untyped exclusions, any mutable main-model exclusion, any omitted expected
graph or inventory entry, inconsistent lifecycle/provenance combinations, or
incomplete immutable evidence.

Families use `LayerMember`/`LayerDomain`, independent `AxisDomain` values,
`FamilyIndexDomain`, `LiteralPathSegment | IndexPathSegment`, structured
`SemanticAddressPattern`, and qualified `OwnerFamilyBinding`/
`OwnerFamilyReference`. No field accepts a free-form template, regex, glob, or
wildcard. Correlated coordinates live in one `LayerMember`; ragged domains
split into multiple complete families. Each Phase 1 selection family fixes
facets, dtype, shape, and axes; its Phase 2 exact projection additionally fixes
source format and ownership. Any role-changing value is fixed in a separate
family, so gate/up/down and Q/K/V/O are not projection axes. Validate duplicates
by exact domain intersection and prove the compact inventory-entry union equals
the full logical inventory without persisting expanded instances. Out-of-scope
and alias compatibility checks also operate on whole compact domains. Rank-local
realized ownership and materialization are deliberately deferred to Task 7.

`ResolvedSelectionTopology.role_definitions` is the sole policy-compilation
role registry and is bound to `topology.schema_version`. The runtime-bound
`SemanticManifestBundle` carries an exact copy and may not add or alter a
definition. Its unique key is
`(schema_version, role_name)`; `RoleDefinition` stores no bundle back-reference.
Topology validation requires every definition's schema version to equal
`topology.schema_version`; Phase 2 verifies the same registry byte-for-byte.
For built-in names, `builtin_role_definitions(schema_version,
expected_domains)` installs every centrally fixed predicate. It attaches the
independently derived expected domain when present and an empty expected domain
when that known built-in role is absent from the topology; an adapter cannot
replace the predicate. A required scope over such an absent built-in fails as a
zero-match selection, while an undeclared spelling remains an unknown role.
Namespaced adapter roles supply their complete versioned predicates and
non-empty expected domains. The Phase 1 resolver sorts the final registry
deterministically and enforces one final definition per key; the Phase 2 bundle
must reproduce it byte-for-byte. Before layer filtering, validation compares
each predicate's compact-entry result against its expected IDs using the
complete `ResolvedSelectionTopology`; a partial-family match, extra entry,
missing entry, or orphan definition fails.

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

### Task 3: Deterministic Source-Neutral Selection Compiler

**Files:**
- Create: `nemo_rl/precision_policy/compiler.py`
- Test: `tests/unit/precision_policy/test_compiler.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: `compile_precision_selection(policy: PrecisionPolicyConfig, topology: ResolvedSelectionTopology) -> CompiledPrecisionSelectionGroup`. The compiler reads only the topology's validated schema-bound role registry and exact graph-local decoder universes; callers cannot pass a second role mapping or runtime inventory.
- Produces: frozen `CompiledGraphPrecisionSelection` records with lifecycle identity, immutable compact-domain assignments for participating endpoints, selected layer ranges, explicit BF16 fences, complete compact scope results plus logical cardinalities, semantic atomic closures, canonical graph `selection_id` values, and an ordered `CompiledPrecisionSelectionGroup`. The group retains both the exact `ResolvedSelectionTopology` and a canonical deeply immutable policy snapshot. Those are its only constructor inputs; `schema_version`, every digest, graph/scope selections, fences, closure, and IDs are `init=False` derived fields. Retaining the topology, rather than only its non-invertible digest, lets Phase 2 prove whole-graph equality for static graphs with no runtime result, while retaining the canonical policy prevents callers from pairing a stale or invented digest with different self-consistent selections. The group contains no source format, source mutability, native storage, producer fingerprint, source alias, owner cadence, Task 7 schedule, or expanded family members. Phase 2 source binding and actual backend capability, rank-local ownership, physical scheduling, transform, and local-plan fingerprints are deferred until after construction.

- [ ] **Step 1: Write failing selection and conflict tests**

```python
def test_global_decoder_boundaries_keep_first_and_last_selected_layers_bf16() -> None:
    plan = compile_fixture(
        layers=range(6),
        moe_layers=(1, 2, 4, 5),
        scope={"roles": ["moe.routed_expert"], "layers": {"exclude_first": 2, "exclude_last": 1}, "rollout": "mxfp8"},
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

Also add literal tests for per-role zero-match in a multi-role list, unknown role, incomplete advertised role coverage, overlapping conflicting scopes, full-range exclusion, atomic fused QKV conflict, allowed fixed-point expansion, expansion crossing a BF16 boundary, dictionary-order-independent selection digest, invalid immutable-auxiliary declarations, and deterministic graph ordering. Verify qualified `advanced_match` and `addresses` selectors, reject ambiguous `graph`/unqualified semantic IDs, and prove built-in main roles require `GraphKind.MAIN` plus the exact semantic graph path. The auxiliary cases prove only source-neutral structure and endpoint participation: a training-only MTP/draft receives training selection but no rollout construction request; a checkpoint-served graph carries immutable identity/context but no guessed runtime-source request; and a different-family drafter uses the same policy while retaining its own graph-local topology and layer universe. Source mutability, owner cadence, aliases, and startup/every-version requests are deliberately absent and receive RED coverage in `test_phase_one_selection_contains_no_source_mutability_alias_or_cadence`. Backend unsupported-format, runtime source completeness, and rank-local ownership checks belong to Tasks 4B and 7-8 after construction.
Assert separately that the returned selection retains the exact input topology
object in-process and its byte-exact canonical serialization across a wire
round-trip; a digest-only placeholder is forbidden.
Add a version-mismatch test, including a default-only policy with no role
selector: compilation must reject `policy.schema_version !=
topology.schema_version` before any matching or selection hashing.

- [ ] **Step 2: Run the compiler tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_compiler.py`

Expected: import failure for `compile_precision_selection` or missing
`CompiledPrecisionSelectionGroup`.

- [ ] **Step 3: Implement compilation in explicit passes**

```python
@dataclass(frozen=True, slots=True)
class PrecisionBoundaryFence:
    scope_id: str
    graph_instance_id: str
    endpoint: PrecisionEndpoint
    index_space: LayerIndexSpace
    bf16_layer_members: tuple[LayerMember, ...]

@dataclass(frozen=True, slots=True)
class CompiledGraphPrecisionSelection:
    graph_instance_id: str
    model_family: str
    resolved_model_revision: str
    lifecycle: GraphLifecycle
    decoder_layer_universe: DecoderLayerUniverse
    policy_digest: str
    training_plan: EndpointPrecisionPlan | None
    rollout_plan: EndpointPrecisionPlan | None
    scope_results: tuple[CompiledScopeGraphResult, ...]
    bf16_fences: tuple[PrecisionBoundaryFence, ...]
    atomic_expansions: tuple[AtomicExpansion, ...]
    immutable_checkpoint_evidence: ImmutableAuxiliaryEvidence | None
    selection_id: str

@dataclass(frozen=True, slots=True)
class CanonicalPrecisionPolicySnapshot:
    canonical_json: str
    schema_version: int = field(init=False)
    policy_digest: str = field(init=False)

@dataclass(frozen=True, slots=True)
class CompiledPrecisionSelectionGroup:
    policy_snapshot: CanonicalPrecisionPolicySnapshot
    topology: ResolvedSelectionTopology
    schema_version: int = field(init=False)
    semantic_structure_digest: str = field(init=False)
    policy_digest: str = field(init=False)
    graph_selections: tuple[CompiledGraphPrecisionSelection, ...] = field(init=False)
    scope_results: tuple[CompiledSelectionScopeResult, ...] = field(init=False)
    bf16_fences: tuple[PrecisionBoundaryFence, ...] = field(init=False)
    atomic_expansions: tuple[AtomicExpansion, ...] = field(init=False)
    selection_group_id: str = field(init=False)

def compile_precision_selection(
    policy: PrecisionPolicyConfig,
    topology: ResolvedSelectionTopology,
) -> CompiledPrecisionSelectionGroup:
    if policy.schema_version != topology.schema_version:
        raise PrecisionPolicyError("policy and selection topology schema versions differ")
    return CompiledPrecisionSelectionGroup(
        policy_snapshot=CanonicalPrecisionPolicySnapshot.from_policy(policy),
        topology=topology,
    )
```

`CanonicalPrecisionPolicySnapshot` uses an exact typed canonical JSON encoding,
strictly reparses it through `PrecisionPolicyConfig`, and derives its digest. It
does not retain a mutable Pydantic object. The group constructor recompiles all
derived records from the snapshot and topology through one public path; callers
cannot provide derived identities or aggregates through construction or
`dataclasses.replace()`.

Treat the topology as a sealed, versioned value tree: recursively require exact
source-neutral record, enum, and scalar leaf types before invoking validation,
matching, equality, or hashing. Collection fields must likewise be exact tuples;
tuple subclasses are executable behavior and cannot enter the retained
topology. Traverse compact factors independently without materializing any
Cartesian family members. Explicit canonical wire payloads are the only
untrusted process/persistence boundary; pickle is a trusted in-process
round-trip mechanism only.

Sort graph instance IDs, semantic graph paths, semantic IDs, attributes, roles,
groups, lifecycle fields, decoder universes, selections, BF16 fences, and
requested endpoint formats before serialization. Every declared graph instance
gets a selection record, but an endpoint assignment exists only when that graph
participates in the endpoint. Built-in main roles match `GraphKind.MAIN` and
the exact semantic graph path, never an instance-name spelling. Built-in roles
do not select auxiliaries; a participating unselected auxiliary inherits BF16
unless a qualified scope in the same policy applies. Compute semantic atomic
expansion only here and reject any expansion crossing an explicit BF16 fence.
Do not derive source owners or cadence and do not accept a runtime discovery
record. Never hash object identity or dictionary insertion order.

- [ ] **Step 4: Run compiler, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_compiler.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/compiler.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/compiler.py tests/unit/precision_policy/test_compiler.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy/compiler.py tests/unit/precision_policy/test_compiler.py pyrefly.toml
git commit -s -m "feat(precision): compile deterministic endpoint selections"
```

### Task 4A: Graph-Scoped Source Discovery Contracts

**Files:**
- Create: `nemo_rl/precision_policy/source_discovery.py`
- Modify: `nemo_rl/precision_policy/topology.py`
- Modify: `nemo_rl/precision_policy/__init__.py`
- Test: `tests/unit/precision_policy/test_source_discovery.py`
- Test: `tests/unit/precision_policy/test_topology_adapters.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: producer-normalized metadata contributions for one explicitly declared graph, an exact expected opaque contributor set, the graph's effective config/revision/source identity/artifact identity, and the existing strict `CanonicalSourceDType` boundary.
- Produces: strict `SourceSchemaId`; the four initial schema constants; immutable `SourceProducerFingerprint`; trusted `ExpectedContributorSet` and ID-free `ExpectedContributorAuthority`; `DiscoveryContribution`; exact `DiscoveryRequestKind`; `DiscoveryCompletenessReceipt`; factory-created `GraphDiscoveryPartition`; the partitioned `SourceDiscoveryInventory`; legacy-only `assemble_graph_discovery_partition()` / `validate_discovery_inventory()` compatibility entrypoints; runtime-only `assemble_runtime_graph_discovery_partition()` / `validate_runtime_discovery_inventory()` entrypoints; `runtime_source_request_identity_digest()`; and a hardened `RuntimeGraphSourceRequest` carrying the Phase 1 structure/selection identities plus the exact source producer fingerprint, contributor authority, and typed source/artifact identities. `SourceDiscoveryRecord` and `SourceRecordProvenance` move from `topology.py` into this core module and are imported/re-exported rather than duplicated.

- [ ] **Step 1: Write failing partition, receipt, and graph-agreement tests**

```python
def test_one_fingerprint_is_stored_once_per_complete_graph_partition() -> None:
    fingerprint = checkpoint_fingerprint()
    expected = ExpectedContributorSet(
        contributor_ids=("checkpoint-index",),
        authority=checkpoint_index_authority(),
    )
    runtime_request = main_runtime_source_request(
        fingerprint=fingerprint,
        expected_contributor_authority=expected.to_authority(),
    )
    partition = assemble_runtime_graph_discovery_partition(
        runtime_request=runtime_request,
        expected_contributors=expected,
        contributions=(contribution("checkpoint-index", fingerprint, two_records()),),
    )
    assert partition.producer_fingerprint == fingerprint
    assert partition.expected_contributor_authority == expected.to_authority()
    assert partition.completeness_receipt.observed_contributor_count == 1
    assert partition.completeness_receipt.source_count == 2
    assert all("fingerprint" not in {field.name for field in fields(record)} for record in partition.records)
    assert "contributor_id" not in {field.name for field in fields(partition)}

def test_inventory_requires_one_matching_complete_partition_per_runtime_request() -> None:
    main_request, main_partition = complete_graph_pair("main")
    draft_request, draft_partition = complete_graph_pair("draft.external")
    expected = expected_contributor_sets_by_graph("main", "draft.external")
    validate_runtime_discovery_inventory(
        (main_request, draft_request),
        SourceDiscoveryInventory((main_partition, draft_partition)),
        expected,
    )
    with pytest.raises(ValueError, match="producer fingerprint"):
        validate_runtime_discovery_inventory(
            (replace(main_request, source_producer_fingerprint=other_fingerprint()), draft_request),
            SourceDiscoveryInventory((main_partition, draft_partition)),
            expected,
        )
```

Parameterize exact failures for an unknown/malformed source schema, mutable implementation tag, producer-selected expected authority, replaced authority evidence, expected-authority/runtime-request mismatch, a derived authority with non-content-address kind, noncanonical locator, or malformed digest, missing or undeclared trusted contributor mapping entry, missing or duplicate opaque contributor, mixed fingerprints, contribution graph mismatch, incomplete PP/rank union represented by a missing opaque contributor, duplicate source/native name, wrong config/revision/source-identity/artifact-identity digest, forged/replaced observed contributor count or digest, forged source count/digest or canonical-record digest, altered record tuple after receipt construction, duplicate graph partition, missing required runtime partition, and undeclared runtime partition. Prove every original typed authority-evidence field changes the opaque derived commitment and that raw IDs or PP/TP/EP coordinates in the retained trusted evidence never appear outside it. Reject bare string/buffer values and unsupported generators at tuple-backed discovery boundaries while preserving tuple/list/tuple-like `Sequence` snapshots. Re-run `validate_runtime_discovery_inventory()` on `dataclasses.replace()` variants and prove every receipt/authority mutation is rejected before frozen-adapter source classification. Include a coordinated mutation of runtime-request authority, partition authority, and receipt: recomputation from the separately retained trusted set must still reject it. Assert contributor IDs and any producer-private PP/TP/EP coordinates are absent from the authority serialization, verified partition, adapter arguments, semantic addresses, and family domains. Keep the existing strict dtype, normalized-source-view provenance, absent-record, deterministic-ordering, and deep-immutability tests.

- [ ] **Step 2: Run focused tests and observe RED**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_topology_adapters.py -k 'partition or fingerprint or contributor or completeness or graph_agreement'`

Expected: import failure for `nemo_rl.precision_policy.source_discovery` or acceptance of the unreceipted global record inventory.

- [ ] **Step 3: Implement the normative immutable contract**

```python
@dataclass(frozen=True, slots=True, order=True)
class SourceSchemaId:
    value: str

HF_SAFETENSORS_HEADER_V1 = SourceSchemaId("hf.safetensors.header.v1")
MEGATRON_BRIDGE_STATE_DICT_V1 = SourceSchemaId("megatron.bridge.state-dict.v1")
NEMO_AUTOMODEL_STATE_DICT_V1 = SourceSchemaId("nemo-automodel.state-dict.v1")
TRANSFORMER_ENGINE_QUANTIZED_STORAGE_V1 = SourceSchemaId(
    "transformer-engine.quantized-storage.v1"
)

@dataclass(frozen=True, slots=True)
class SourceProducerFingerprint:
    schema_id: SourceSchemaId
    producer_implementation_id: str
    producer_revision: str
    normalization_contract_digest: str
    evidence: EvidenceSource

@dataclass(frozen=True, slots=True)
class ExpectedContributorAuthority:
    contributor_set_digest: str
    contributor_count: int
    authority: EvidenceSource  # CONTENT_ADDRESS, one exact locator, SHA-256

@dataclass(frozen=True, slots=True)
class ExpectedContributorSet:
    contributor_ids: tuple[str, ...]
    authority: EvidenceSource
    def to_authority(self) -> ExpectedContributorAuthority: ...

@dataclass(frozen=True, slots=True)
class DiscoveryContribution:
    contributor_id: str
    graph_instance_id: str
    producer_fingerprint: SourceProducerFingerprint
    records: tuple[SourceDiscoveryRecord, ...]

class DiscoveryRequestKind(StrEnum):
    GRAPH_TOPOLOGY_INPUT = "graph_topology_input"
    RUNTIME_GRAPH_SOURCE_REQUEST = "runtime_graph_source_request"

@dataclass(frozen=True, slots=True)
class DiscoveryCompletenessReceipt:
    graph_instance_id: str
    producer_fingerprint_digest: str
    observed_contributor_set_digest: str
    observed_contributor_count: int
    source_set_digest: str
    source_count: int
    canonical_records_digest: str
    request_kind: DiscoveryRequestKind
    request_digest: str

@dataclass(frozen=True, slots=True)
class GraphDiscoveryPartition:
    graph_instance_id: str
    producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    records: tuple[SourceDiscoveryRecord, ...]
    completeness_receipt: DiscoveryCompletenessReceipt

@dataclass(frozen=True, slots=True)
class SourceDiscoveryInventory:
    partitions: tuple[GraphDiscoveryPartition, ...]

@dataclass(frozen=True, slots=True)
class RuntimeGraphSourceRequest:
    declaration: ExpectedGraphDeclaration
    resolved_graph: ResolvedGraphTopology
    semantic_structure_digest: str
    selection_group_id: str
    model_config: Mapping[str, object]
    resolved_model_revision: str
    source_producer_fingerprint: SourceProducerFingerprint
    expected_contributor_authority: ExpectedContributorAuthority
    source_identity: EvidenceSource
    artifact_identity: EvidenceSource
    source_allocation_generation: str
    runtime_source_request_digest: str
```

`SourceSchemaId` accepts only an exact lowercase namespaced/versioned atom matching `[a-z][a-z0-9-]*(\.[a-z0-9-]+)+\.v[1-9][0-9]*`; no trimming or case folding. Producer revisions are immutable commit or content identities, not branches/tags. The runtime/checkpoint integration—not the producer—constructs one non-empty, duplicate-free `ExpectedContributorSet` from its trusted index-shard list or runtime membership plus typed authority evidence. Contributor IDs and producer-normalized source-view shapes snapshot only supported non-scalar `Sequence` inputs; bare strings, bytes, byte arrays, memory views, and generators are rejected before tuple conversion, while tuple/list/tuple-like inputs remain supported. `to_authority()` canonicalizes the opaque IDs and computes their count/digest. It separately hashes the complete typed original authority-evidence payload into an `EvidenceSource(kind=CONTENT_ADDRESS, locator="precision-policy.expected-contributor-authority.v1", digest="sha256:<64-lowercase-hex>")`. `ExpectedContributorAuthority` enforces that exact structural form, so no raw contributor or placement label can escape and no substring scan is needed. Neither producer output nor stored receipt fields are inputs to either commitment. `RuntimeGraphSourceRequest` binds the derived ID-free authority plus `semantic_structure_digest` and `selection_group_id` before discovery. Freeze the effective config recursively, compute `runtime_source_request_digest` from the resolved graph identity/config/revision/source identity/artifact identity/source allocation generation/fingerprint/expected authority and both Phase 1 digests, and canonicalize all sets before hashing.

`assemble_runtime_graph_discovery_partition()` recomputes the observed contributor set
from `DiscoveryContribution` values, requires exact equality with the trusted
set and the runtime source request's structurally constrained authority, one
common fingerprint/graph, and a unique complete source set, constructs the receipt
itself, then strips contribution objects and contributor IDs from the
factory-created partition. Producers cannot supply or choose the expected
authority. The resolver retains an exact graph-ID → `ExpectedContributorSet`
mapping, including its original typed evidence, through the next boundary.
`validate_runtime_discovery_inventory(runtime_requests, source_discovery,
expected_contributors_by_graph)` runs immediately before frozen-adapter source
classification. It requires exactly one trusted set per runtime request and no
unrequested mapping or partition; the declared static-checkpoint draft
exception has no runtime request, trusted set, or partition.
It re-derives each set and evidence commitment from that trusted input,
independently compares the authority with both the runtime source request and
partition, compares the receipt's assembly-derived observed contributor
digest/count with it, and recomputes source-set count/digest, canonical-record
digest, and runtime-request digest from the partition. It rejects
forged/replaced/stale receipts, incomplete unions, and even coordinated
runtime-request/partition/receipt authority replacement because the trusted
mapping is a separate input.
`validate_runtime_discovery_inventory()` returns only the fully verified inventory; it
does not select an adapter or build semantic topology. Task 4B's Phase 2 binder
requires that validated result, and Task 4C's later exact-projection helper may
then pass only the verified producer-normalized record tuple and frozen
`ResolvedGraphTopology` to the adapter selected in Phase 1. The verified
partition intentionally stores the realization inventory; no normalized
record, adapter argument, or semantic type stores contributor IDs, original
trusted evidence, placement coordinates, or native physical realization
metadata.

- [ ] **Step 4: Run contract, type, import-isolation, and format gates**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_topology_adapters.py`

Run: `.venv/bin/pyrefly check nemo_rl/precision_policy`

Run: `/opt/homebrew/bin/ruff check nemo_rl/precision_policy/source_discovery.py nemo_rl/precision_policy/topology.py nemo_rl/precision_policy/__init__.py tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_topology_adapters.py`

Run: `/opt/homebrew/bin/ruff format --check nemo_rl/precision_policy/source_discovery.py nemo_rl/precision_policy/topology.py nemo_rl/precision_policy/__init__.py tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_topology_adapters.py`

Run: `git diff --check`

Expected: all commands pass and importing `nemo_rl.precision_policy` does not import Megatron, Automodel, Transformer Engine, vLLM, or Torch.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/precision_policy/source_discovery.py nemo_rl/precision_policy/topology.py nemo_rl/precision_policy/__init__.py tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_topology_adapters.py pyrefly.toml
git commit -s -m "feat(precision): bind complete source discovery partitions"
```

### Task 4A.1: Canonical Built-In Format Compatibility Migration

This mandatory migration is the compatibility boundary between the checked-in
Task 2 implementation and Task 4B. The current constants use `encoding=None`
for BF16 and MXFP8 values, `mxfp8_scale` for MXFP8 scales, and an implicit
output-axis `/1 EXACT` default. Preserve the existing stable IDs because the
underlying BF16 and E4M3/E8M0 block-32 contracts do not change; make their
canonical serialization explicit. Pre-migration wire payloads and digests are
invalidated and regenerated. They are not accepted through a second descriptor
meaning or compatibility alias.

**Files:**
- Modify: `nemo_rl/precision_policy/semantic.py:424-448`
- Test: `tests/unit/precision_policy/test_semantic.py:554-742`
- Test: `tests/unit/precision_policy/test_compiler.py`

**Interfaces:**
- Consumes: Task 2's existing `ComponentDescriptor`, `FormatDescriptor`, `LogicalComponentAxisSpec`, `AxisExtentRounding`, `_validate_reserved_format()`, and the compiler's canonical `_format_payload()` serialization.
- Produces: the sole canonical `BF16_FORMAT` object for `bf16.logical.v1` and sole canonical `MXFP8_FORMAT` object for `mxfp8.e4m3-e8m0-block32-input-features.v1`. No new format ID, alias type, or fallback parser is introduced.

- [ ] **Step 1: Write failing canonical and legacy-rejection tests**

Replace the existing built-in descriptor assertions and add reserved-ID
compatibility cases in `test_semantic.py`:

```python
def test_builtin_format_descriptors_have_canonical_serialization() -> None:
    assert tuple(
        (component.role, component.dtype, component.encoding, component.component_axes)
        for component in BF16_FORMAT.components
    ) == ((LOGICAL_VALUES, "bfloat16", "plain_bfloat16", None),)
    assert tuple(
        (component.role, component.dtype, component.encoding, component.component_axes)
        for component in MXFP8_FORMAT.components
    ) == (
        (VALUES, "e4m3", "mxfp8_e4m3_values", None),
        (
            BLOCK_SCALES,
            "e8m0",
            "mxfp8_e8m0_scale",
            (
                LogicalComponentAxisSpec(
                    "output_features",
                    divisor=1,
                    rounding=AxisExtentRounding.EXACT,
                ),
                LogicalComponentAxisSpec(
                    "input_features",
                    divisor=32,
                    rounding=AxisExtentRounding.CEIL,
                ),
            ),
        ),
    )

@pytest.mark.parametrize(
    "legacy_format",
    (
        FormatDescriptor(
            "bf16.logical.v1",
            "bf16",
            (ComponentDescriptor(LOGICAL_VALUES, "bfloat16"),),
        ),
        FormatDescriptor(
            "mxfp8.e4m3-e8m0-block32-input-features.v1",
            "mxfp8",
            (
                ComponentDescriptor(VALUES, "e4m3"),
                ComponentDescriptor(
                    BLOCK_SCALES,
                    "e8m0",
                    encoding="mxfp8_scale",
                    component_axes=(
                        LogicalComponentAxisSpec("output_features"),
                        LogicalComponentAxisSpec(
                            "input_features",
                            divisor=32,
                            rounding=AxisExtentRounding.CEIL,
                        ),
                    ),
                ),
            ),
        ),
    ),
)
def test_reserved_format_ids_reject_precanonical_meanings(
    legacy_format: FormatDescriptor,
) -> None:
    entry = _tensor_entry(
        "legacy-format",
        "main",
        "text.decoder.legacy.kernel",
        format=legacy_format,
    )
    with pytest.raises(ValueError, match="reserved .* format_id"):
        _bundle((entry,), (_owner(entry),)).validate_complete()
```

Add an exact compiler-payload assertion in `test_compiler.py`; this verifies
that descriptor identity, intent digests, and wire output share the migrated
meaning:

```python
def test_builtin_format_wire_payloads_use_canonical_encodings() -> None:
    assert compiler_module._format_payload(BF16_FORMAT) == {
        "format_id": "bf16.logical.v1",
        "family": "bf16",
        "components": [
            {
                "role": "logical_values",
                "dtype": "bfloat16",
                "encoding": "plain_bfloat16",
                "component_axes": {"kind": "identity"},
            }
        ],
    }
    assert compiler_module._format_payload(MXFP8_FORMAT) == {
        "format_id": "mxfp8.e4m3-e8m0-block32-input-features.v1",
        "family": "mxfp8",
        "components": [
            {
                "role": "values",
                "dtype": "e4m3",
                "encoding": "mxfp8_e4m3_values",
                "component_axes": {"kind": "identity"},
            },
            {
                "role": "block_scales",
                "dtype": "e8m0",
                "encoding": "mxfp8_e8m0_scale",
                "component_axes": {
                    "kind": "explicit",
                    "axes": [
                        {
                            "kind": "logical",
                            "logical_axis": "output_features",
                            "divisor": 1,
                            "rounding": "exact",
                        },
                        {
                            "kind": "logical",
                            "logical_axis": "input_features",
                            "divisor": 32,
                            "rounding": "ceil",
                        },
                    ],
                },
            },
        ],
    }
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py tests/unit/precision_policy/test_compiler.py -k 'builtin_format_descriptors_have_canonical_serialization or reserved_format_ids_reject_precanonical_meanings or builtin_format_wire_payloads_use_canonical_encodings'`

Expected: failures show the checked-in `None`, `mxfp8_scale`, and implicit-axis
descriptor still equals the currently reserved same-ID object.

- [ ] **Step 3: Migrate the two canonical constants**

Update only the built-in constant definitions in `semantic.py`:

```python
BF16_FORMAT = FormatDescriptor(
    format_id="bf16.logical.v1",
    family="bf16",
    components=(
        ComponentDescriptor(
            role=LOGICAL_VALUES,
            dtype="bfloat16",
            encoding="plain_bfloat16",
        ),
    ),
)
MXFP8_FORMAT = FormatDescriptor(
    format_id="mxfp8.e4m3-e8m0-block32-input-features.v1",
    family="mxfp8",
    components=(
        ComponentDescriptor(
            role=VALUES,
            dtype="e4m3",
            encoding="mxfp8_e4m3_values",
        ),
        ComponentDescriptor(
            role=BLOCK_SCALES,
            dtype="e8m0",
            encoding="mxfp8_e8m0_scale",
            component_axes=(
                LogicalComponentAxisSpec(
                    "output_features",
                    divisor=1,
                    rounding=AxisExtentRounding.EXACT,
                ),
                LogicalComponentAxisSpec(
                    "input_features",
                    divisor=32,
                    rounding=AxisExtentRounding.CEIL,
                ),
            ),
        ),
    ),
)
```

Do not modify `_validate_reserved_format()`: once the constants migrate, its
existing structural equality checks reject both legacy same-ID descriptors.
Do not add a legacy reader, dual-ID registry entry, or normalization rule.

- [ ] **Step 4: Run GREEN and regression gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py tests/unit/precision_policy/test_compiler.py -k 'builtin_format_descriptors_have_canonical_serialization or reserved_format_ids_reject_precanonical_meanings or builtin_format_wire_payloads_use_canonical_encodings'`

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py tests/unit/precision_policy/test_compiler.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy/semantic.py nemo_rl/precision_policy/compiler.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/semantic.py tests/unit/precision_policy/test_semantic.py tests/unit/precision_policy/test_compiler.py`

Run: `git diff --check`

Expected: all commands pass. The compiler code is unchanged because it already
serializes all descriptor fields; its new test proves the migrated constants
flow into wire/digest identity. Any cached or persisted intent built with the
legacy descriptor is regenerated before use.

- [ ] **Step 5: Commit and independently review the migration**

```bash
git add nemo_rl/precision_policy/semantic.py tests/unit/precision_policy/test_semantic.py tests/unit/precision_policy/test_compiler.py
git commit -s -m "fix(precision): canonicalize built-in format descriptors"
```

The signed commit contains exactly those three files. Do not begin Task 4B or
run its catalog/object-identity acceptance as a passing gate until this commit
and its independent review pass.

### Task 4A.2: Producer-Normalized Views and Native Storage Realization Evidence

This compatibility boundary is required before Task 4B. A
`FormatDescriptor` describes a canonical logical encoding; it is not evidence
that one framework's native buffers have those dtypes, shapes, padding, or
ordering. The distinction is material for Transformer Engine MXFP8: at the
pinned root revision, values are carried by uint8 buffers at the logical tensor
shape, while rowwise scales are stored as
`[round_up(M, 128), round_up(K / 32, 4)]` and may be GEMM-swizzled in 128x4
tiles. Columnwise storage has a different shape and layout. Neither native
form is the canonical unpadded MXFP8 component grid.

**Files:**
- Create: `nemo_rl/precision_policy/source_storage.py`
- Modify: `nemo_rl/precision_policy/source_discovery.py`
- Modify: `nemo_rl/precision_policy/topology.py`
- Modify: `nemo_rl/precision_policy/__init__.py`
- Test: `tests/unit/precision_policy/test_source_storage.py`
- Test: `tests/unit/precision_policy/test_source_discovery.py`
- Test: `tests/unit/precision_policy/test_topology_adapters.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: Task 4A producer-normalized `SourceDiscoveryRecord` values, their exact native carrier metadata, immutable producer normalization fingerprints, and Task 2 logical component descriptors.
- Produces: `SourcePhysicalAxisSpec`, `SourceStorageComponent`,
  `SourcePaddingSemantics`, `SourceNormalizationKind`,
  `SourceNormalizationContract`, `SourceStorageRealization`,
  `SourceDerivedRealization`, a graph-scoped
  `SourceStorageRealizationInventory`, and completeness receipts that commit to
  both normalized views and their native-storage witnesses.

`SourceDiscoveryRecord.dtype` and `.shape` become the exact producer-normalized
view presented to topology classification. They are metadata-only virtual
views, not eagerly materialized payloads and not silently relabeled native
buffers. Each present non-backend-derived record has at least one separately
attested storage realization; an absent record has none. A backend-derived
record instead has one `SourceDerivedRealization` with no raw component and a
versioned derivation capability/digest that cannot authorize a source wire
payload. A storage realization names exactly one `output_record_id` and owns
an ordered non-empty tuple of raw components with exact native component IDs,
names, carrier dtypes, physical shapes, physical-axis formulas, alignment,
storage encoding, typed padding semantics, and permutation/swizzle identity.
Padding semantics distinguish at least deterministic `ZERO_FILLED` padding
from `UNSPECIFIED_IGNORED` padding for compact native buffers whose unused
cells are intentionally uninitialized. A fill encoding exists only when the
selected padding semantics requires one. It
also owns the output's normalized dtype, shape, numeric encoding, and a
versioned normalization capability ID/digest. Multiple alternative
realizations may target one output record only when all normalized output
facts are identical. A raw component identity and its metadata are canonical
within the graph inventory. Alternatives may reference it; cross-record reuse
is valid only when final topology classification proves the corresponding
identical-storage relation. Synchronized replicas require distinct native
component/owner identities. Unexplained cross-record reuse is rejected before
the semantic bundle is exposed.
An identity normalization is legal only when one raw component exactly equals
the normalized view in dtype, shape, order, and encoding. Reinterpretation,
crop, unflatten, unswizzle, repack, dequantization, or quantization is never
identity.

Use a small typed physical-axis formula rather than a backend-shaped field:
one physical extent is `round_up(divide(product(normalized_axis_indices),
divisor, rounding), alignment)`, or one positive literal. This expresses
ordinary identity/packing as well as TE flattened-prefix and aligned scale
storage without putting TE names into the core. The exact realized
`physical_shape` must equal resolution of every formula. Non-axis tile
reordering remains an immutable versioned storage/permutation identity owned
and validated by the source adapter; an unknown identity fails closed.

The producer fingerprint's `normalization_contract_digest` commits to one
canonical allowed-normalizer manifest. Every realization's capability ID and
contract digest must be an exact member of that manifest; assembly never
accepts a new self-asserted normalizer merely because it can hash it. The graph partition and its assembly-created receipt include canonical count
and digest fields for the storage-realization inventory. Assembly rejects a
missing present-record realization, a realization for an absent or unknown
record, duplicate realization/component identities, malformed or unpinned
normalizer identities, unresolved formulas, mismatched exact shapes, and a
producer contribution whose record and realization graphs differ. Final
inventory validation recomputes these facts from the separately retained
trusted contributor set just as it does for normalized records.

Topology classification partitions the normalized record shape and compares
its normalized dtype and numeric encoding to the claimed logical component.
It never compares a
raw carrier dtype or padded native buffer cardinality directly with a
`FormatDescriptor`. Task 7 consumes the attested realization plus the
classifier's semantic axis mapping, deterministically lowers it into the
source-stage `PhysicalRepresentation`, and re-probes the live endpoint. The
realization, evidence, normalizer-manifest, and live capability digests all
enter the bound plan identity; physical equality cannot erase E4M3/E8M0 tags
or normalization provenance. A raw native fast path is
available only when the live source and destination physical descriptors are
exactly equal and a capability proof authorizes that adjacent transfer;
otherwise the named normalization/transform runs. No storage witness itself
contains placement or grants direct-copy authority.

Exact physical equality includes padding semantics and any required fill
encoding. A `ZERO_FILLED` source and an `UNSPECIFIED_IGNORED` destination (or
the reverse) cannot use direct copy merely because their extents, carrier
dtypes, and byte counts match; the planner must select crop/repack or another
capability-proven transform.

- [ ] **Step 1: Add failing source-storage contract tests**

Test identity BF16 and safetensors components, TE rowwise compact-padded and
GEMM-swizzled scale witnesses, exact flattened-prefix/alignment formulas, and
normalizer-manifest membership/digests. Add failures for
dtype/shape/encoding/order/layout/swizzle or
padding changes, identity applied to a uint8 carrier for an E4M3/E8M0 view,
unknown/output-mismatched record IDs, unexplained raw-component sharing, missing
realization coverage, backend-derived wire eligibility, and receipt mutation.

- [ ] **Step 2: Implement and validate the realization inventory**

Keep `source_storage.py` standard-library-only. Extend contribution assembly,
partition identity, receipt recomputation, import-isolation tests, and topology
classification messages from ambiguous `raw` terminology to explicit
`normalized source view` terminology. Preserve Task 4A's no-framework import
boundary.

- [ ] **Step 3: Run gates, independently review, and commit**

Run focused source-storage, source-discovery, topology, compiler, Pyrefly,
Ruff, import-isolation, and `git diff --check` gates. Commit only the exact
owned files with sign-off. Task 4B cannot proceed to producer implementation
until this task is green and independently reviewed.

### Task 4B: Two-Phase Resolver, Runtime Producers, Evidence Gate, and Canonical Format Catalog

**Prerequisite:** Task 4A.1's canonical built-in-format migration and Task
4A.2's source-storage realization contract have both passed their full gates,
signed commits, and independent reviews. Task 4B must not compensate for an
unmigrated Task 2 checkout by recreating, aliasing, or normalizing either
reserved ID, and it must not identify native storage with a logical format.
Task 4B establishes the resolver/binder contracts with literal fake adapters;
Task 4C supplies and gates the production family adapters without changing
those contracts.

**Files:**
- Create: `nemo_rl/precision_policy/source_formats.py`
- Create: `nemo_rl/precision_policy/discovery_producers/__init__.py`
- Create: `nemo_rl/precision_policy/discovery_producers/checkpoint.py`
- Create: `nemo_rl/precision_policy/discovery_producers/megatron_bridge.py`
- Create: `nemo_rl/precision_policy/discovery_producers/automodel.py`
- Create: `nemo_rl/precision_policy/discovery_producers/transformer_engine.py`
- Modify: `nemo_rl/precision_policy/topology_resolver.py`
- Create: `nemo_rl/precision_policy/runtime_binding.py`
- Create: `tools/capture_precision_policy_source_evidence.py`
- Create: `tests/fixtures/precision_policy/producer_implementations.json`
- Create: `tests/fixtures/precision_policy/source_format_evidence.json`
- Test: `tests/unit/precision_policy/test_source_formats.py`
- Test: `tests/unit/precision_policy/test_discovery_producers.py`
- Test: `tests/unit/precision_policy/test_topology_resolver.py`
- Test: `tests/unit/precision_policy/test_runtime_binding.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: Phase 1 graph declarations/effective model configurations and Task 3's `CompiledPrecisionSelectionGroup`; after construction, Task 4A's immutable runtime source request/partition contract, Task 4A.1's committed canonical `BF16_FORMAT` and `MXFP8_FORMAT` objects, realized Bridge/Automodel/TE or checkpoint contexts, and exact expected opaque contributor sets supplied by each runtime integration.
- Produces: `GraphTopologyResolutionRequest`; `resolve_selection_topology(requests, schema_version) -> ResolvedSelectionTopology`; `validate_compiled_precision_selection_group()`; aggregate `RuntimeSourceDiscoveryRequest`; `RuntimeSourceDiscoveryResult`; selection-bound `build_runtime_graph_source_request()`, `build_runtime_source_discovery_request()`, and `build_runtime_source_discovery_result()` factories; public exact `validate_source_producer_fingerprint()` and `source_producer_fingerprint_identity_digest()` boundaries; `SourceMetadataProducer`; positive graph-bound `produce_runtime_source_discovery_results()` bulk orchestration; separate checkpoint normalization/attestation; `produce_megatron_bridge_partition()`; `produce_automodel_partition()`; `produce_transformer_engine_partition()`; `bind_runtime_source_intents(selection, request, results) -> CompiledPrecisionIntentGroup`; pinned producer-implementation evidence; and the reviewed `SOURCE_FORMAT_CATALOG: tuple[FormatDescriptor, ...]`. `runtime_binding.py` orchestrates Task 4A's `RuntimeGraphSourceRequest` rather than redeclaring it. Phase 1 remains isolated in `topology_resolver.py` and is standard-library-only. Framework objects are normalized inside Phase 2 producers and never cross into topology or result records.

`SourceMetadataProducer.discover_contributions(runtime_graph_request,
expected_contributors)` returns normalized `DiscoveryContribution` values, not
a self-certified partition. The Phase 2 Task 4B resolver supplies the trusted
expected set and calls Task 4A's assembly/validation itself. The convenience
`produce_*_partition()` functions are resolver-owned orchestration wrappers
around that sequence; a producer cannot choose its expected authority or
construct a receipt unchecked.

The production runtime entrypoint is
`produce_runtime_source_discovery_results(selection, request,
producers_by_graph)`. `producers_by_graph` is an exact positive binding for all
and only `TRAINING_RUNTIME` requests; it is never a family registry or a
`can_handle()` scan. The dispatcher snapshots the mapping once, validates the
complete selection-bound request, snapshots every unique producer identity,
fingerprint, and bound discovery callable once, and checks all graph bindings
before the first producer runs. Producer property access and fingerprinting are
external-code boundaries, so the dispatcher revalidates the full compiled
selection and aggregate request afterward and rejects changed commitments or
replaced request collections before any discovery. It then discovers and
resolver-assembles each canonical main-first graph exactly once and publishes
only through one bulk result factory. A failure discards all unpublished
partitions and prevents any later producer call. The bulk factory deliberately
repeats its own bounded `O(G)` defensive validation; this startup-only check is
not a per-version refit operation. A static checkpoint graph has no runtime
producer binding or runtime result and instead crosses the independent
checkpoint load-attestation boundary.

- [ ] **Step 1: Capture missing producer and format evidence before implementation**

Task 4A.1 and its semantic/compiler tests are the authority for the two imported
`BF16_FORMAT` and `MXFP8_FORMAT` objects; Task 4B must not manufacture a
second raw-source claim for either built-in. Write
`tests/unit/precision_policy/test_source_formats.py` so it rejects absent
independent evidence for every additionally constructed catalog component
axis, divisor, rounding, dtype, encoding, producer identity, A95B block
geometry, and native storage realization. Write
`tools/capture_precision_policy_source_evidence.py` to derive its output only
from independently staged raw config JSON, safetensors index JSON, decoded
header-manifest metadata, and pinned local source trees. A pre-shaped
observation or a previously generated output fixture is never an accepted
input. Tests construct raw metadata independently and poison any convenient
generated-output file so copying it cannot pass. The capture receipt records
stable artifact-relative opened paths plus exact byte digests and proves
index-to-header tensor/shard equality; machine-absolute paths never enter a
fixture or identity. It must record:

- Megatron Bridge gitlink/HEAD `b11414c71b15e54d333eb49346ed199f20fa9021`;
- NeMo Automodel gitlink/HEAD `1814c6c93a66b9d59d254960ef6a99a64249b671`;
- nested Megatron-Core gitlink/HEAD `7c9c3a027c503ae9ae1e8ad7b14397abb8269378`;
- the two distinct Transformer Engine provenance identities without collapsing them: the NeMo-RL effective root lock/runtime (`42b840051647eef89761a16dfdff87e82bb253ab`, package identity `2.15.0+42b8400`) and the Megatron Bridge source-tree declaration (`4329ff84bfbdaa778a33cba02a15fb0807c64689`, package identity `2.17.1+4329ff84`); fail if either inspected source identity differs from its pin or if the effective runtime differs from the NeMo-RL root lock;
- the exact K2.5 Automodel I32/F16/I64 pack-8, input-group-32, logical-shape-vector contract from `nemo_automodel/components/models/kimi_k25_vl/state_dict_adapter.py` at the pinned Automodel revision. The catalog uses `/32 EXACT`: the current implementation computes `ceil(K/32)` groups and then reshapes into equal-width groups, so `K=40` is two width-20 groups rather than a canonical 32+8 remainder. The producer rejects `K % 32 != 0` instead of falsely claiming a group-32 CEIL layout;
- the pinned TE MXFP8 logical admission and native realization facts separately: `M=product(shape[:-1])` and `K=shape[-1]` must both be divisible by 32; rowwise and columnwise values use uint8 carriers; rowwise scale storage is `[round_up(M,128), round_up(K/32,4)]`; columnwise scale storage is `[round_up(M/32,4), round_up(K,128)]`; and the scale layout may be compact-padded or GEMM-swizzled in 128x4 tiles. These facts attest Task 4A.2 storage realizations and do not claim native equality with `MXFP8_FORMAT`;
- representative gate/up/down orientations for K2, both K2.5 producer variants, K3, Lightning NVFP4, and A95B FP8, including exact raw names, sibling sets, dtype, shape, logical axes, encoding, divisors, and remainder/rounding behavior.

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_source_formats.py`

Expected RED: the evidence fixtures and catalog module do not exist. Then run the capture tool against the staged raw metadata root. It never reads tensor data ranges or downloads weight payloads. Missing staged data, a gitlink/runtime mismatch, index/header disagreement, or inability to prove A95B geometry/remainder behavior is a hard stop before Step 2, not permission to infer values.

- [ ] **Step 2: Define and independently review the literal catalog**

The literal catalog is:

```python
EXPECTED_SOURCE_FORMATS = {
    "bf16.logical.v1": (
        "bf16",
        (("logical_values", "bfloat16", "plain_bfloat16", None),),
    ),
    "mxfp8.e4m3-e8m0-block32-input-features.v1": (
        "mxfp8",
        (
            ("values", "e4m3", "mxfp8_e4m3_values", None),
            ("block_scales", "e8m0", "mxfp8_e8m0_scale", (("output_features", 1, "exact"), ("input_features", 32, "ceil"))),
        ),
    ),
    "block-fp8.e4m3-f32-scale-inv-block128x128.v1": (
        "block_fp8",
        (
            ("values", "e4m3", "float8_e4m3_values", None),
            ("inverse_scales", "float32", "inverse_scale_float32", (("output_features", 128, "exact"), ("input_features", 128, "exact"))),
        ),
    ),
    "block-fp8.e4m3-bf16-scale-inv-block128x128.v1": (
        "block_fp8",
        (
            ("values", "e4m3", "float8_e4m3_values", None),
            ("inverse_scales", "bfloat16", "inverse_scale_bfloat16", (("output_features", 128, "exact"), ("input_features", 128, "exact"))),
        ),
    ),
    "packed-int4.i32-bf16-group32-shape-i32.v1": (
        "packed_int4",
        (
            ("packed_values", "int32", "int4_offset_binary_pack8", (("output_features", 1, "exact"), ("input_features", 8, "exact"))),
            ("group_scales", "bfloat16", "symmetric_group_scale", (("output_features", 1, "exact"), ("input_features", 32, "exact"))),
            ("logical_shape", "int32", "logical_shape_vector", (("literal", 2, "exact"),)),
        ),
    ),
    "packed-int4.i32-f16-group32-shape-i64.v1": (
        "packed_int4",
        (
            ("packed_values", "int32", "int4_offset_binary_pack8", (("output_features", 1, "exact"), ("input_features", 8, "exact"))),
            ("group_scales", "float16", "symmetric_group_scale", (("output_features", 1, "exact"), ("input_features", 32, "exact"))),
            ("logical_shape", "int64", "logical_shape_vector", (("literal", 2, "exact"),)),
        ),
    ),
    "mxfp4.u8-u8-block32-input-features.v1": (
        "mxfp4",
        (
            ("packed_values", "uint8", "mxfp4_pack2", (("output_features", 1, "exact"), ("input_features", 2, "exact"))),
            ("block_scales", "uint8", "mxfp4_block_scale", (("output_features", 1, "exact"), ("input_features", 32, "exact"))),
        ),
    ),
    "nvfp4.u8-e4m3-f32-block16-input-features.v1": (
        "nvfp4",
        (
            ("packed_values", "uint8", "nvfp4_pack2", (("output_features", 1, "exact"), ("input_features", 2, "exact"))),
            ("block_scales", "e4m3", "nvfp4_block_scale", (("output_features", 1, "exact"), ("input_features", 16, "exact"))),
            ("global_scale", "float32", "nvfp4_global_scale", ()),
        ),
    ),
}
```

This mapping is the expected canonical serialization, not a second set of
descriptor constructors. After Task 4A.1 is committed and reviewed,
`source_formats.py` imports its migrated exact `BF16_FORMAT` and
`MXFP8_FORMAT` objects and places those same objects in
`SOURCE_FORMAT_CATALOG`; it constructs only the additional reviewed formats.
Task 4A.1's `tests/unit/precision_policy/test_semantic.py` pins the explicit
encodings and axes above, while `test_source_formats.py` asserts
`catalog_by_id[BF16_FORMAT.format_id] is BF16_FORMAT`, the corresponding MXFP8
identity, canonical serialization equality, and global format-ID uniqueness.
Any repeated stable ID with a different family/component contract fails
catalog construction; no typed alias may carry a second meaning.

K2 and A95B use distinct block-FP8 IDs because immutable evidence proves that K2 stores inverse scales as FP32 while A95B stores them as BF16. A95B's pinned routed-expert tensors are exactly divisible on both 128-wide axes; the catalog therefore records `exact` and rejects a non-divisible tensor rather than inferring an unproven ceil/pad rule. K2.5 checkpoint and Automodel IDs remain distinct. U8-carried K3 MXFP4 and U8-carried Lightning NVFP4 remain distinct. Assert exact catalog order, stable IDs, family, roles, scalar dtype, encoding, component axes, divisor, and rounding; descriptor identity must not contain a model or repository name. Run the source-format unit/type/format gates and obtain an independent task review. Do not begin Step 3 or Task 4C until the catalog review passes.

- [ ] **Step 3: Write failing producer-integration tests**

Test exact index/header equality and traversal-safe shard names for the checkpoint producer; missing/duplicate/mixed contributors and config/revision/artifact mismatch for every producer; Bridge public conversion-task normalization without retaining Bridge/MCore objects; Automodel native state-dict discovery before any `full_tensor()` or HF conversion; and TE wrappers whose nominal dtype is accepted only with validated quantized component metadata. Assert all producer contributions use the schema ID assigned below, and assert the resolver-owned Task 4A assembly creates and revalidates the completeness receipt:

```python
EXPECTED_PRODUCER_SCHEMAS = {
    "checkpoint": "hf.safetensors.header.v1",
    "megatron_bridge": "megatron.bridge.state-dict.v1",
    "automodel": "nemo-automodel.state-dict.v1",
    "transformer_engine": "transformer-engine.quantized-storage.v1",
}
```

Add a subprocess import test that blocks imports of Torch, Megatron, Automodel, Transformer Engine, and vLLM while importing `nemo_rl.precision_policy`, `source_discovery`, `source_formats`, and `discovery_producers`. The producer package `__init__.py` must not eagerly import optional implementations. Assert no source discovery module imports or depends on vLLM.

In `tests/unit/precision_policy/test_topology_resolver.py`, first add these
exact Phase 1 RED tests:

- `test_global_boundary_uses_declared_decoder_universe_without_dense_marker`;
- `test_moe_ordinal_universe_is_exact_contiguous_one_to_one_mapping`;
- `test_main_mtp_and_draft_layer_universes_are_independent_and_zero_based`;
  and
- `test_phase_one_selection_contains_no_source_mutability_alias_or_cadence`.

Then add these exact Phase 2 RED tests in
`tests/unit/precision_policy/test_runtime_binding.py`:

- `test_runtime_bf16_and_mxfp8_sources_project_to_same_semantic_structure`;
- `test_phase_two_preserves_selection_and_bf16_fences_byte_exactly`;
- `test_runtime_graph_request_requires_phase_one_effective_config_digest`;
- `test_runtime_missing_extra_or_reshaped_member_fails_before_cadence`;
- `test_te_primary_accounts_for_bf16_boundaries_and_mxfp8_middle`;
- `test_static_external_draft_needs_no_runtime_partition`;
- `test_cross_graph_mtp_alias_inherits_main_owner_cadence`; and
- `test_missing_extra_or_stale_runtime_graph_result_fails_atomically`.

The Phase 2 preservation test asserts that the in-process intent group retains
the same selection/topology objects and that their canonical wire bytes remain
identical after serialization. The static-draft test asserts that the graph
remains present in both retained artifacts despite having no runtime result.

Also prove the runtime resolver derives trusted expected-contributor authority
before producer invocation, selects exactly one producer per required runtime
graph, uses the adapter identity already frozen by Phase 1 for a
different-family drafter, rejects every partition/authority/receipt mismatch
before source binding, and never imports or probes vLLM. Assert a failure
publishes no partial result or intent group. The separate hot-path test
`test_refit_hot_path_never_calls_topology_or_source_discovery` lands with the
cached-plan performance work in Task 13.

- [ ] **Step 4: Implement producer normalization and the two phase boundaries**

Each producer module owns its optional framework imports and converts native metadata immediately into frozen Task 4A records/contributions. Checkpoint discovery streams every safetensors header and never model weight payloads. Bridge uses public `AutoBridge.get_conversion_tasks()` / `get_export_fp8_tasks()` metadata and preserves opaque complete contributor union evidence. Automodel walks native `state_dict()` metadata before gather/LoRA merge/conversion and uses adapter key metadata only as a cross-check. TE requires explicit component metadata for quantized wrappers and never infers encoding from nominal dtype.

Keep `GraphTopologyResolutionRequest` and `resolve_selection_topology()` in the
standard-library-only Phase 1 module `nemo_rl/precision_policy/topology_resolver.py`.
Implement the frozen Phase 2 contracts and boundaries in the separate Task
4B-owned module `nemo_rl/precision_policy/runtime_binding.py`:

```python
def build_runtime_graph_source_request(
    selection: CompiledPrecisionSelectionGroup,
    graph_instance_id: str,
    model_config: Mapping[str, object],
    source_producer_fingerprint: SourceProducerFingerprint,
    expected_contributors: ExpectedContributorSet,
    source_identity: EvidenceSource,
    artifact_identity: EvidenceSource,
    source_allocation_generation: str,
) -> RuntimeGraphSourceRequest: ...

def validate_compiled_precision_selection_group(
    selection: CompiledPrecisionSelectionGroup,
) -> CompiledPrecisionSelectionGroup: ...

@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryRequest:
    graph_requests: tuple[RuntimeGraphSourceRequest, ...]
    trusted_expected_contributors: tuple[tuple[str, ExpectedContributorSet], ...]
    semantic_structure_digest: str = field(init=False)
    selection_group_id: str = field(init=False)
    request_digest: str = field(init=False)

@dataclass(frozen=True, slots=True)
class RuntimeGraphSourceContext:
    graph_instance_id: str
    model_config: Mapping[str, object]
    source_producer_fingerprint: SourceProducerFingerprint
    expected_contributors: ExpectedContributorSet
    source_identity: EvidenceSource
    artifact_identity: EvidenceSource
    source_allocation_generation: str

def build_runtime_source_discovery_request_from_contexts(
    *,
    selection: CompiledPrecisionSelectionGroup,
    contexts: Mapping[str, RuntimeGraphSourceContext],
) -> RuntimeSourceDiscoveryRequest: ...

def build_runtime_source_discovery_request(
    selection: CompiledPrecisionSelectionGroup,
    graph_requests: tuple[RuntimeGraphSourceRequest, ...],
    trusted_expected_contributors: Mapping[str, ExpectedContributorSet],
) -> RuntimeSourceDiscoveryRequest: ...

@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryResult:
    graph_request: RuntimeGraphSourceRequest
    partition: GraphDiscoveryPartition
    graph_instance_id: str = field(init=False)
    runtime_source_request_digest: str = field(init=False)
    semantic_structure_digest: str = field(init=False)
    selection_group_id: str = field(init=False)
    producer_fingerprint: SourceProducerFingerprint = field(init=False)
    result_digest: str = field(init=False)

def build_runtime_source_discovery_result(
    request: RuntimeSourceDiscoveryRequest,
    graph_request: RuntimeGraphSourceRequest,
    partition: GraphDiscoveryPartition,
) -> RuntimeSourceDiscoveryResult: ...

def build_runtime_source_discovery_results(
    *,
    request: RuntimeSourceDiscoveryRequest,
    partitions: Sequence[GraphDiscoveryPartition],
) -> tuple[RuntimeSourceDiscoveryResult, ...]: ...

class SourceMetadataProducer(Protocol):
    producer_id: str
    schema_id: SourceSchemaId
    def fingerprint(self) -> SourceProducerFingerprint: ...
    def discover_contributions(
        self,
        request: RuntimeGraphSourceRequest,
        expected_contributors: ExpectedContributorSet,
    ) -> Sequence[DiscoveryContribution]: ...

def produce_runtime_source_discovery_results(
    *,
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    producers_by_graph: Mapping[str, SourceMetadataProducer],
) -> tuple[RuntimeSourceDiscoveryResult, ...]: ...

def validate_runtime_source_discovery_results(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: Sequence[RuntimeSourceDiscoveryResult],
) -> SourceDiscoveryInventory: ...

@dataclass(frozen=True, slots=True)
class CompiledGraphPrecisionIntent:
    selection: CompiledGraphPrecisionSelection
    source_owners: tuple[SourceOwnerInventoryEntry, ...]
    owner_refit_requirements: OwnerRefitRequirements
    refit_requirement: RefitRequirement
    startup_owner_requests: tuple[OwnerFamilyReference, ...]
    every_version_owner_requests: tuple[OwnerFamilyReference, ...]
    source_alias_contracts: tuple[SourceAliasContract, ...]
    intent_id: str

@dataclass(frozen=True, slots=True)
class CompiledPrecisionIntentGroup:
    schema_version: int
    selection: CompiledPrecisionSelectionGroup
    semantic_structure_digest: str
    selection_group_id: str
    runtime_source_digest: str
    graph_intents: tuple[CompiledGraphPrecisionIntent, ...]
    startup_source_items: tuple[OwnerRealizationRequest, ...]
    every_version_source_items: tuple[OwnerRealizationRequest, ...]
    immutable_checkpoint_contexts: tuple[ImmutableAuxiliaryEvidence, ...]
    source_alias_contracts: tuple[SourceAliasContract, ...]
    intent_group_id: str

def bind_runtime_source_intents(
    selection: CompiledPrecisionSelectionGroup,
    request: RuntimeSourceDiscoveryRequest,
    results: tuple[RuntimeSourceDiscoveryResult, ...],
) -> CompiledPrecisionIntentGroup: ...
```

Phase 1 resolves effective configuration/declaration/revision and exact layer
universes, selects exactly one pure adapter per graph, and constructs the whole
`ResolvedSelectionTopology` atomically. It neither imports a producer nor
accepts a runtime source object. `semantic_structure_digest` commits every
graph/member/address/domain/shape/role/atomic-group/layer-universe field plus
the canonical digest of the exact effective model configuration used for
adapter selection. The raw configuration need not be retained after this
digest is frozen.
Production Phase 2 uses the two bulk factories above. It validates the
selection once, snapshots each ephemeral runtime configuration once, and
validates the complete result inventory once before publishing results. The
single-graph request/result factories are safe convenience boundaries only;
calling them in a production graph loop is forbidden because it repeats
whole-selection or aggregate validation and becomes quadratic in graph count.
The request builder derives each claimed universe only from the effective
configuration and declared topology facts. The selected family adapter derives
it independently under its pinned contract and must reproduce the request's
universe byte-for-byte; disagreement fails the entire Phase 1 request set.

After endpoint construction, Phase 2 derives the trusted expected contributor
set/authority at each runtime/checkpoint integration boundary, retains the
exact graph-to-trusted-set mapping, invokes exactly one producer for every
training-runtime graph (including a non-served MTP or drafter), and passes that
mapping with the complete partition inventory through Task 4A validation.
This required graph set is derived from the exact frozen selection lifecycle,
never supplied by the caller. A checkpoint-provenance, checkpoint-served static
external draft has no runtime request, partition, or result; its immutable
destination load receipt remains mandatory. The binder uses only the Phase
1-selected adapter identity to classify source records and first verifies every runtime request's
`resolved_graph` against `selection.topology`. It requires an exact projection
onto all frozen semantic members, addresses, domains, shapes, selections, BF16
fences, and atomic closures. It may add source formats, mutability, native
realizations, aliases, and derived cadence. It cannot select an adapter,
recompile policy, expand an atomic group, or alter semantic structure. Missing,
extra, duplicate, reshaped, or stale results fail the whole set before cadence
derivation.
The graph-request factory accepts no declaration, resolved graph, revision,
adapter ID, semantic digest, or selection ID. It looks up the graph once and
derives all of those fields from the exact `CompiledPrecisionSelectionGroup`;
callers cannot self-certify them. It recomputes the runtime model-config digest
with the shared canonical function and requires equality with the digest
retained by the Phase 1 graph before constructing a request. Aggregate
request/result digest fields are `init=False`. The binder independently repeats
the exact graph-set, graph-content, request-digest, selection-digest, authority,
partition-receipt, and result-digest checks, so direct dataclass construction,
pickle mutation, or a self-consistent set of invented IDs cannot reach source
classification.
Static graphs without a runtime request remain available through
`selection.topology`; the final intent group retains the exact selection rather
than attempting to recover topology from `semantic_structure_digest`.
The opaque trusted sets never reach a topology adapter and are removed from
cross-process/public serialization after the resolver has validated results.

`runtime_source_digest` hashes the canonical request/result set and all
producer/completeness/native-realization evidence. `intent_group_id` binds
`semantic_structure_digest`, `selection_group_id`, `runtime_source_digest`, and
the canonical runtime-bound intents. Producer implementation imports are lazy
inside Phase 2; importing the resolver or running Phase 1 does not import
Torch, Megatron, Automodel, TE, or vLLM. Task 5 imports these functions and does
not redefine them.

The final bound plan must not retain `RuntimeGraphSourceContext`, raw
`model_config`, `RuntimeGraphSourceRequest`, or producer inventory objects.
Add a structural no-retention test and make every topology, compiler,
configuration-digest, and discovery function fail if invoked from the repeated
refit path. This turns the startup-only performance property into a gate rather
than relying on a docstring.

- [ ] **Step 5: Run producer/catalog gates and commit**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_runtime_binding.py tests/unit/precision_policy/test_topology_adapters.py`

Run: `.venv/bin/pyrefly check nemo_rl/precision_policy tools/capture_precision_policy_source_evidence.py`

Run: `/opt/homebrew/bin/ruff check nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py nemo_rl/precision_policy/runtime_binding.py tools/capture_precision_policy_source_evidence.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_runtime_binding.py`

Run: `/opt/homebrew/bin/ruff format --check nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py nemo_rl/precision_policy/runtime_binding.py tools/capture_precision_policy_source_evidence.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_runtime_binding.py`

Run: `git diff --check`

Expected: all commands pass with exact producer identities and no unresolved evidence field.

```bash
git add nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py nemo_rl/precision_policy/runtime_binding.py tools/capture_precision_policy_source_evidence.py tests/fixtures/precision_policy/producer_implementations.json tests/fixtures/precision_policy/source_format_evidence.json tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_runtime_binding.py pyrefly.toml
git commit -s -m "feat(precision): normalize versioned source metadata"
```

Do not start Task 4C until Task 4A.1's migration gates/commit/review and every
Task 4B producer/catalog gate pass and the Task 4B commit is independently
reviewed. Passing the migration or catalog review alone is not sufficient.

### Task 4C: Model Topology Adapters and Pinned Conformance Fixtures

**Files:**
- Modify: `nemo_rl/precision_policy/topology.py`
- Modify: `nemo_rl/precision_policy/adapters/__init__.py`
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
- Create: `tests/fixtures/precision_policy/artifact_cases.json`
- Create: `tests/fixtures/precision_policy/auxiliary_graphs.json`
- Test: `tests/unit/precision_policy/test_topology_adapters.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes in Phase 1: `GraphTopologyResolutionRequest` values with effective plain configuration, resolved revision, declaration, and explicit exact `DecoderLayerUniverse`. Consumes in Phase 2: Task 4A's `RuntimeGraphSourceRequest`, `SourceDiscoveryInventory`, and `ExpectedContributorSet` contracts through Task 4C's internal `build_semantic_manifest_bundle(selection: CompiledPrecisionSelectionGroup, runtime_requests: Sequence[RuntimeGraphSourceRequest], source_discovery: SourceDiscoveryInventory, expected_contributors_by_graph: Mapping[str, ExpectedContributorSet]) -> SemanticManifestBundle`. The inventory and retained trusted mapping are complete for exactly `RuntimeSourceDiscoveryRequest.graph_requests`. That runtime subset is derived from the frozen selection and lifecycle; an explicitly static checkpoint-served external draft is absent and owes destination attestation instead. A different-family external drafter never inherits the main graph's adapter inputs.
- Produces atomically in Phase 1: registered adapters selected independently per graph by exact `model_type` and architecture capabilities; source-neutral `ResolvedGraphTopology` records with complete compact semantic entries, exact layer universes, roles, and atomic groups; and `resolve_text_config()` handling nested `text_config` without assuming top-level `num_hidden_layers`. Produces in Phase 2 with the same frozen adapter ID: typed compact discovery edges, runtime-bound semantic `ParameterInventory`/manifests, and normalized identical-storage or synchronized-replica source-alias contracts that exact-project onto the Phase 1 entries. Source aliases and their evidence participate in `runtime_source_digest` and `intent_group_id`, never retroactively in `semantic_structure_digest` or `selection_group_id`. Task 2 tests may construct frozen records directly, but production has no API that reselects a family or reconstructs policy from runtime sources.

- [ ] **Step 1: Add pinned literal topology fixtures and failing adapter tests**

```python
def test_qwen35_uses_nested_40_layer_text_config() -> None:
    topology = resolve_fixture_graph("qwen3_5_35ba3b.json")
    assert topology.decoder_layer_universe is not None
    assert topology.decoder_layer_universe.global_decoder_layers == tuple(range(40))
    assert topology.find("text.decoder.layer.39.moe.routed.0.down") is not None

def test_kimi_k25_and_k3_exact_routed_domains() -> None:
    k25 = resolve_fixture_graph("kimi_k2_5.json")
    k3 = resolve_fixture_graph("kimi_k3.json")
    assert k25.role_domain_size("moe.routed_expert") == 60 * 384 * 3
    assert k3.role_domain_size("moe.routed_expert") == 92 * 896 * 3
    assert k3.role_match_count("sequence_mixer.kda.projections", "moe.routed_expert") == 0
```

Add `topology facts` and bounded `grammar micro-fixture` cases for all thirteen
exact topology IDs in the design. Add fifteen physical artifact cases by
splitting Lightning BF16/NVFP4 and A95B BF16/FP8. Each artifact record copies
its exact revision, config/index/header-manifest SHA256, shard count, tensor
count, source schema, and expected canonical logical format set from the design's
artifact table. It separately records the exact lower
`task4c_conformance_tier` actually executed (`topology facts` or `grammar
micro-fixture`). The test asserts literal equality and that the set of IDs is:

```python
TOPOLOGY_CASE_IDS = {
    "qwen3_30ba3b", "qwen3_5_35ba3b",
    "nemotron3_5_lightning_30ba3b", "nemotron3_super_120ba12b",
    "nemotron3_ultra_550ba55b", "nemotron3_nano_30ba3b",
    "kimi_k2", "kimi_k2_5", "kimi_k3",
    "qwen3_8_2_4t_a95b", "qwen3_8_flash_next", "qwen3_8_27b",
    "glm_5_2",
}
ARTIFACT_CASE_IDS = {
    "qwen3_30ba3b_bf16", "qwen3_5_35ba3b_bf16",
    "nemotron3_5_lightning_30ba3b_bf16",
    "nemotron3_5_lightning_30ba3b_nvfp4",
    "nemotron3_super_120ba12b_bf16", "nemotron3_ultra_550ba55b_bf16",
    "nemotron3_nano_30ba3b_bf16", "kimi_k2_block_fp8",
    "kimi_k2_5_checkpoint_int4", "kimi_k3_mxfp4",
    "qwen3_8_2_4t_a95b_bf16", "qwen3_8_2_4t_a95b_fp8",
    "qwen3_8_flash_next_bf16", "qwen3_8_27b_bf16", "glm_5_2_bf16",
}
```

For every known-family topology fixture, assert complete source-neutral logical
semantics and independently expected role/component domains, not merely the
names used by one selector. In separate Phase 2 grammar cases, exact-project
all normalized source records onto those domains. Cover dense and
routed gate/up/down, Q/K/V/O, embeddings, output heads, norms, biases where
present, non-QKVO mechanisms such as KDA/MLA/SSM with zero QKVO role matches,
and every present format component. Only a true extension namespace may use
generic BF16 classification; a missed known-family namespace is a hard
failure.

Assert literal adapter dispatch on `(model_type, architectures, capability
flags)` only. Missing or ambiguous tuple fields, mutually compatible adapters,
and a capability contradiction fail. Changing only repository ID, revision,
artifact format, or producer schema never selects another adapter. Those
fields remain conformance evidence and may still cause a capability or runtime
partition-validation failure after Phase 1 dispatch.

The Qwen3.8 dense fixture must fail required routed-expert compilation. Kimi K2 uses `weight + weight_scale_inv`; K2.5 uses `weight_packed + weight_scale + weight_shape`. Sibling artifacts share logical facts but must have different physical identities where storage differs, and crossing a sibling's config/revision evidence with the other's normalized source-view records or native-storage realizations fails Phase 2 before intent binding.

Add MTP/draft fixtures for: a static checkpoint-owned MTP; independent mutable training-only and source-served MTP graphs; actual same-storage aliases; MCore-style synchronized source replicas with distinct native owners; a static external drafter; and mutable training-only and source-served speculative drafters using a different model-family adapter. Phase 1 asserts that main-model roles select none of them, every declared auxiliary has its own source-neutral topology even when it is training-only, only participating endpoints receive default BF16 selection, and checkpoint-served graphs carry complete typed immutable identity. Phase 2 asserts qualified aliases point to an explicit main-graph owner without duplicating logical ownership, while same-storage and synchronized-replica evidence remain distinct. Eagle parameters initialized by `.copy_()` are independent owners, not aliases. All instances use one versioned precision policy; a different-family drafter does not carry a separate policy.

Use graph-local `LayerMember.global_decoder_layer` coordinates in every
manifest. Main decoder members have semantic path/model part
`text.decoder`/`main`; MTP members use `auxiliary.mtp`/`mtp`; draft members use
`draft.decoder`/`draft`. The different-family drafter fixture is Qwen3.5 main
plus a synthetic Nemotron 3 Nano draft. Phase 1 gives it its own config,
resolved revision, graph-local universe, and scopes; Phase 2 separately
supplies its producer fingerprint and source records. Label it synthetic and do
not present it as an official trained drafter.

Assert one MTP-local layer zero only when the effective configuration and
pinned source-neutral topology facts declare it for Qwen3.5, A95B, Flash,
Lightning, Super, Ultra, and GLM. Nano and Qwen3 declare none.
Lightning/Super/Ultra `.0` attention plus `.1` MoE/final-norm runtime records
must exact-project onto that one Phase 1 layer, and GLM physical layer 78 must
project only to MTP-local layer zero. A Phase 2 result missing any required
record fails without changing the topology.

Add negative fixtures that omit an instantiated training auxiliary, declare a
mutable checkpoint-served graph, or omit any immutable evidence field. Add
literal configurations with `loss_scaling_factor=0` and `detach_heads=true` and
assert that their owners remain mutable unless the source inventory supplies
independent freeze evidence. Topology adapter tests stop at topology-independent
graph declarations; Task 7 exclusively derives owning/non-owning ranks. Give a
different-family external drafter its own model configuration and resolved
revision and assert independent adapter selection. Fail a present discovery
region with a gap or overlap, while allowing one fused record to classify
through disjoint compact edges into multiple semantic members of one canonical
owner. Also reject a
missing native name/owner unless mutability is `ABSENT`, and reject either
native field on an `ABSENT` discovery record. Add a literal negative fixture
that still accounts for the normalized routed-expert `up` source-view record
but misclassifies it
as `ffn.dense`; its independently topology-derived expected routed domain must
disagree with predicate matching and fail validation.

Add synchronized-replica RED tests with distinct main/MTP native owners and one
canonical logical owner. Reject equal native IDs, a missing or alias canonical
record, dtype/shape/replica-region/canonical-source-region/component/projection mismatch, empty or invalid
runtime synchronization evidence, mutability mismatch, mixed edge variants,
and region gaps/overlaps. Changing replica group, boundary, or evidence digest
must preserve `semantic_structure_digest` and `selection_group_id` while
changing `runtime_source_digest` and `intent_group_id`. Add the corresponding
Task 10 source test requiring a matching topology/group/version/rank completion
fence and proving only one canonical tensor is exported. A copied Eagle LM head
must remain a second independent owner. It creates its own source request only
when that member is in scope, served from a training source, and not excluded;
`not_served`, direct checkpoint-body, and `out_of_scope` copies create none.

Add literal discovery-edge fixtures for a whole tensor, disjoint fused
gate/up regions, a strided grouped-expert family, a region gap, overlapping
regions, an edge that claims an output omitted from the fragment, a semantic
entry or owner invented without an edge, a non-consuming tied-storage alias edge, and an
explicit `ABSENT` zero-output disposition. Include a tied fused QKV or gate/up
record split into multiple fixed-role alias edges. Assert regions and index maps remain
compact and never enumerate source elements. Only the `ABSENT` disposition may
have zero semantic outputs, and it cannot justify a source-served owner. Add a
60-layer-family negative fixture whose 60 normalized source-view records all claim the same
singleton layer-zero output domain; exact per-entry/component output-domain
partitioning must reject the duplicated layer and missing layers.

Add role-registry tests proving built-in predicates are centrally fixed while
adapters attach independently derived expected domains, namespaced roles carry
complete versioned predicates, and registry order is deterministic. Two draft
instances contributing disjoint domains to the same canonical-equal predicate
must merge into one sorted union; a repeated entry ID, overlapping compact
domain, changed predicate, or version conflict fails construction.

Add literal compiler-boundary acceptance fixtures with `N=2, M=1`. Assert the
exact selected routed-layer sequences and logical cardinalities from the
design for Qwen3, Qwen3.5, Lightning, Super, and Ultra. In particular,
Lightning selects layers
`3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49`, keeps routed
layers 1 and 51 BF16, and does not mistake dense layer zero for a routed
boundary. Assert Qwen3 selected/total `17280/18432`, Qwen3.5 `28416/30720`,
Lightning `5376/5888`, Super `38912/40960`, and Ultra `47104/49152`.

- [ ] **Step 2: Run adapter tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_adapters.py`

Expected: missing concrete adapter registrations/classifiers or failure of the
new literal conformance assertions; Task 4A's core topology imports already
exist.

- [ ] **Step 3: Implement Phase 1 adapter selection and Phase 2 exact classification**

Reuse Task 2's `SourceSynchronizationBoundary`,
`SourceReplicaSynchronizationEvidence`, and normalized source-alias contract
types directly; topology must not declare lookalike types with different
runtime identities.

```python
@dataclass(frozen=True, slots=True)
class SynchronizedReplicaAliasClassificationEdge:
    record_id: str
    replica_source_region: SourceRegion
    alias_output: OutputMemberTarget
    canonical_record_id: str
    canonical_source_region: SourceRegion
    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    component_role: ComponentRole
    alias_to_canonical_axes: tuple[AxisProjection, ...]
    synchronization: SourceReplicaSynchronizationEvidence

@dataclass(frozen=True, slots=True)
class SourceIndexSpan:
    start: int
    stop: int
    step: int = 1

@dataclass(frozen=True, slots=True)
class SourceAxisSelection:
    axis_index: int
    spans: tuple[SourceIndexSpan, ...]

@dataclass(frozen=True, slots=True)
class SourceRegion:
    source_shape: tuple[int, ...]
    axis_selections: tuple[SourceAxisSelection, ...]

@dataclass(frozen=True, slots=True)
class SourceOrdinalMapSegment:
    source_span: SourceIndexSpan
    target_ordinal_start: int
    target_ordinal_step: int = 1

@dataclass(frozen=True, slots=True)
class FamilyIndexAxisTarget:
    axis_name: str

@dataclass(frozen=True, slots=True)
class LayerCoordinateTarget:
    coordinate: Literal["global_decoder_layer", "moe_ordinal"]

@dataclass(frozen=True, slots=True)
class ComponentAxisTarget:
    component_role: ComponentRole
    component_axis: str

type SemanticAxisTarget = (
    FamilyIndexAxisTarget | LayerCoordinateTarget | ComponentAxisTarget
)

@dataclass(frozen=True, slots=True)
class FixedFamilyAxisCoordinate:
    axis_name: str
    member: int | str

@dataclass(frozen=True, slots=True)
class FixedLayerCoordinate:
    member: LayerMember

type FixedMemberCoordinate = FixedFamilyAxisCoordinate | FixedLayerCoordinate

@dataclass(frozen=True, slots=True)
class OutputMemberTarget:
    inventory_entry_id: str
    member_domain: FamilyIndexDomain
    fixed_coordinates: tuple[FixedMemberCoordinate, ...]

@dataclass(frozen=True, slots=True)
class SourceToSemanticAxisMapping:
    source_axis_index: int
    target: SemanticAxisTarget
    segments: tuple[SourceOrdinalMapSegment, ...]

@dataclass(frozen=True, slots=True)
class CanonicalValueClassificationEdge:
    record_id: str
    source_region: SourceRegion
    output: OutputMemberTarget
    canonical_owner_family: OwnerFamilyReference
    component_role: ComponentRole
    axis_mappings: tuple[SourceToSemanticAxisMapping, ...]

@dataclass(frozen=True, slots=True)
class TiedAliasClassificationEdge:
    record_id: str
    aliased_source_region: SourceRegion
    alias_output: OutputMemberTarget
    canonical_owner_family: OwnerFamilyReference
    canonical_value_entry_id: str
    component_role: ComponentRole
    alias_to_canonical_axes: tuple[AxisProjection, ...]

@dataclass(frozen=True, slots=True)
class AbsentDiscoveryDispositionEdge:
    record_id: str

type DiscoveryClassificationEdge = (
    CanonicalValueClassificationEdge
    | TiedAliasClassificationEdge
    | SynchronizedReplicaAliasClassificationEdge
    | AbsentDiscoveryDispositionEdge
)

@dataclass(frozen=True, slots=True)
class RoleDefinitionContribution:
    schema_version: int
    role_name: str
    predicate: SemanticPredicate
    expected_inventory_entry_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class SemanticGraphBuildFragment:
    graph_instance_id: str
    classification_edges: tuple[DiscoveryClassificationEdge, ...]
    source_owners: tuple[SourceOwnerInventoryEntry, ...]
    inventory_entries: tuple[ParameterInventoryEntry, ...]
    manifest: SemanticGraphManifest
    role_contributions: tuple[RoleDefinitionContribution, ...]

class ModelTopologyAdapter(Protocol):
    adapter_id: str
    def supports(self, model_config: Mapping[str, object]) -> bool: ...
    def resolve_graph(
        self,
        request: GraphTopologyResolutionRequest,
    ) -> ResolvedGraphTopology: ...
    def classify_graph(
        self,
        schema_version: int,
        resolved_graph: ResolvedGraphTopology,
        records: tuple[SourceDiscoveryRecord, ...],
    ) -> SemanticGraphBuildFragment: ...

def build_semantic_manifest_bundle(
    selection: CompiledPrecisionSelectionGroup,
    runtime_requests: Sequence[RuntimeGraphSourceRequest],
    source_discovery: SourceDiscoveryInventory,
    expected_contributors_by_graph: Mapping[str, ExpectedContributorSet],
) -> SemanticManifestBundle: ...
```

Task 4A is the single owner of `SourceDiscoveryRecord`,
`SourceDiscoveryInventory`, `GraphDiscoveryPartition`, producer fingerprint,
completeness receipt, and `RuntimeGraphSourceRequest`. Task 4C imports those exact
types and must not redeclare lookalikes. The Phase 2 binder is the sole
production caller of `build_semantic_manifest_bundle()`; the helper is not a
public path that can resolve topology or compile policy from runtime records.
The bundle builder requires one
complete partition and separately retained trusted contributor set for each
input, rejects any undeclared value, re-derives expected authority, and verifies
its graph/config/revision/source/artifact identity, fingerprint, independently
trusted expected-contributor authority, and recomputed completeness receipt
before source classification. It passes only the partition's verified
producer-normalized record tuple and graph request to the adapter ID already
selected and digested by Phase 1,
collects compact build fragments, constructs the semantic inventory and
graph-aware bundle, and validates it as one atomic operation. Adapters cannot
observe opaque contributor IDs or native physical realization metadata.
`SourceRegion` is compact exact region algebra: every source
axis occurs once, its ordered spans are non-empty, disjoint, in bounds, and may
be whole, contiguous, or strided. `SourceToSemanticAxisMapping` maps compact
source spans through a typed `FamilyIndexAxisTarget`, `LayerCoordinateTarget`,
or resolved `ComponentAxisTarget`; no bare target-axis string is accepted.
Grouped/fused layouts use multiple spans or edges, never enumerated tensor
elements. `OutputMemberTarget` names an exact family subdomain and any fixed
family/layer coordinates; every family index coordinate appears exactly once
as varying or fixed.

For every present non-alias discovery record, canonical-value edge regions
must partition the complete producer-normalized view shape exactly once with
no gap or overlap. Task 4A.2 has already proved how its native storage realizes
that view; padding or swizzled carrier bytes are not semantic region members.
Each such edge names exactly one output member target, canonical owner family,
component role, and total axis mapping. Independently, for every
inventory entry and every component role required by its `FormatDescriptor`,
edge output-member domains must exactly partition that entry's compact family
domain with no gap or overlap. Within each output target, fixed coordinates and
coordinates supplied by typed mappings are disjoint and together cover every
family and resolved component-axis coordinate exactly once; sixty per-layer records
cannot all claim the same layer of one family, and a required scale component
cannot disappear. A tied-alias edge justifies one exact alias member target and
its exact direct target. Its `aliased_source_region` partitions only the tied
record's declared logical view and never consumes the underlying canonical
storage, so tied storage is not double-counted. A record marked `ABSENT` has exactly one explicit absent
disposition edge; that is the only zero-output case and it cannot justify a
`served_from_source` owner. Conversely, every fragment semantic entry and
canonical owner is justified by an edge; an unknown edge target, claimed but
omitted output, or invented output fails. Every fragment entry and locally
declared owner belongs to its graph. No partially classified bundle escapes on
failure.

Edge variants are provenance-checked: a `TIED_STORAGE` record has a non-empty
set of tied-alias edges, a `SYNCHRONIZED_REPLICA` record has a non-empty set of
synchronized-replica alias edges, an `ABSENT` record has exactly one absent
disposition, and every other present record is covered only by consuming
canonical-value regions. A producer-normalized source-view record cannot mix
those categories. Multiple tied edges may
split fused storage into separate fixed-role semantic entries, but their
coverage-only regions and `(alias entry, component role, output domain)` claims
must be an exact compact partition without gaps, overlaps, or duplicate
targets. Every direct target must be a compatible non-alias member on the same
underlying canonical native owner identified by the tied discovery record;
zero tied edges or mixed edge variants fail.

`SYNCHRONIZED_REPLICA` is a separate non-consuming alias edge, never a synonym
for `TIED_STORAGE`. Its explicit canonical source record must be a consuming
direct record for the same canonical value/component/subdomain. Replica and
canonical normalized-view dtype, shape, and corresponding compact regions must
match in the initial contract, while their native owner IDs must differ.
Replica regions partition the normalized replica view exactly; Task 4A.2
separately validates each side's native realization. Replica mutability
matches the canonical owner, and the immutable synchronization evidence names a non-empty
replica group plus the `SOURCE_VERSION_READY` boundary. The bundle persists a
strongly typed normalized `IdenticalStorageSourceAliasContract |
SynchronizedReplicaSourceAliasContract` union containing the alias/direct
semantic IDs, canonical owner, component role, exact projected domains,
relation evidence, and, for replicas, group/boundary. No nullable discriminator
combination is accepted. Task 4 Phase 2 evidence proves the runtime source
relation without changing Phase 1 semantic topology, but it proves no live
synchronization; Task 7/10 must enforce optimizer/TE update → replica
synchronization → matching per-version `SourceVersionFence` → export.
Missing/stale group, topology, version, rank, or completion-fence proof
is fatal. If the source adapter cannot prove the invariant, classify the copy
as an independent canonical owner.

Canonical native-owner authority is graph-local rather than fragment-local.
Within one graph, all consuming canonical records with the same
`source_native_owner_id` resolve to exactly one qualified
`OwnerFamilyReference` and agree on provenance, provenance evidence,
mutability, and mutability evidence. Independent graphs may reuse local owner
identifiers without coupling their authority. Cross-graph sharing is expressed
only through an explicit validated alias relation whose canonical reference
includes the owning graph identity.

Each Phase 1 graph resolution emits typed role-definition contributions. Their
expected domains are derived independently from
`GraphTopologyResolutionRequest`, effective configuration, and the declared
layer universe, not by reapplying the role predicate or examining runtime
source names. Phase 2 must reproduce the same semantic domains exactly.
Built-in
contributions must exactly match the central schema-versioned predicate;
adapters may attach expected domains but cannot alter it. Namespaced
contributions provide full versioned predicates. For repeated
`(schema_version, role_name)` keys, the builder requires canonical-equal
predicates and pairwise-disjoint expected domains, then deterministically unions
and sorts their entry IDs into one final `RoleDefinition`. A repeated/overlapping
entry, changed predicate, or version conflict fails. Phase 1 installs that
canonical registry in `ResolvedSelectionTopology` and compares every expected
domain with predicate matching over the complete topology. The Phase 2 bundle
must carry the same registry byte-for-byte.

Classifiers may recognize endpoint names internally, but emit canonical
semantic addresses and structured families only. Phase 1 chooses each adapter
independently, including a different-family drafter, and orders graph instances
deterministically; the bundle builder verifies and reuses those frozen adapter
IDs. Reconcile typed auxiliary declarations against normalized source-view
discovery and its separately attested native-storage realizations so every
actually instantiated training auxiliary is present. Do not derive runtime PP
ownership here. Reject ambiguous names, missing built-in role definitions,
empty namespaced-role expected domains, predicate results unequal to their
expected compact entry IDs, inconsistent expert counts, unnormalized one-based
layer indices, revision/config/header capability contradictions, contradictory
declarations, family-domain overlaps, partial inventory coverage, or any
runtime projection that differs from Phase 1. Repository ID and resolved
revision are evidence, never adapter-dispatch or allowlist inputs. An empty
expected domain is valid only for an installed central built-in that the
topology does not contain. Keep dense prefix layers in the explicit decoder
universe even when they contain no routed expert. Emit separate fixed-attribute
families for gate/up/down and Q/K/V/O and split ragged domains into multiple
complete families. Adapter discovery may use lazy generators, but the
resulting inventory and manifest never store an expanded family member list.

- [ ] **Step 4: Run required topology/grammar, compiler, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_adapters.py tests/unit/precision_policy/test_compiler.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy tests/unit/precision_policy tests/fixtures/precision_policy pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit Task 4C at the executed lower tier**

```bash
git add nemo_rl/precision_policy tests/unit/precision_policy/test_topology_adapters.py tests/fixtures/precision_policy pyrefly.toml
git commit -s -m "feat(precision): add model topology adapters"
```

Task 4C completes when its required `topology facts` and bounded `grammar
micro-fixture` tests pass. `artifact_cases.json` records only the exact lower
`task4c_conformance_tier` actually executed per artifact. A later immutable
Task 4D receipt, not a rewritten fixture claim, promotes that one artifact to
`full metadata conformance`; absence of such a receipt leaves its effective
label at the recorded lower tier and does not block Task 4C or Task 5.
This commit establishes semantic/source-classifier conformance only. No model
is production-supported until the producer, Transformer Engine, destination,
mixed-refit, transaction, numerical-correctness, and performance gates in the
later tasks all pass for its exact artifact and deployment path.

### Task 4D (Optional): Promote Individual Artifacts to Full Metadata Conformance

This opt-in tranche is not a prerequisite for Task 4C completion or Task 5.
Execute it for each artifact whose staged metadata and required host evidence
are available. A missing receipt means only that the artifact retains its
exact Task 4C lower-tier label; it is not silently promoted and is not called
adapter/model support.

**Files:**
- Create: `tools/precision_policy_metadata_conformance.py`
- Test: `tests/metadata/precision_policy/test_full_metadata_conformance.py`
- Create on successful promotion: `tests/fixtures/precision_policy/full_metadata_conformance/<artifact_case_id>.json`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: one or more explicitly requested pinned artifact cases, staged local config/index/all-header metadata, Task 4B's producer/resolver, and Task 4C's classifier.
- Produces per successful artifact: an immutable `full metadata conformance` receipt. It does not rewrite or imply receipts for unexecuted artifacts.

- [ ] **Step 1: Write failing receipt, accounting, and tier-promotion tests**

Define one canonical receipt with artifact/topology IDs, source schema,
producer fingerprint, config/index/header-manifest digests,
`semantic_structure_digest`, `selection_group_id`, `runtime_source_digest`,
`intent_group_id`,
`source_count`, `normalized_record_count`, `semantic_member_count`,
`component_count`, `tensor_count`, `shard_count`, per-trial elapsed seconds,
and per-trial incremental peak RSS. `source_count` is the exact canonical
source-set cardinality from producer completeness; `normalized_record_count`
is the classifier input-record cardinality; `semantic_member_count` is the
exact logical semantic cardinality computed with compact domain algebra;
`component_count` is the exact resolved semantic-component cardinality;
`tensor_count` and `shard_count` are the raw pinned header/index counts. These
fields remain distinct even when two happen to be numerically equal.

Test that every count and digest is recomputed/asserted against the producer,
partition, compact semantic manifest, format descriptors, and pinned artifact
evidence. Reject a missing, swapped, borrowed, or self-reported count, compact
family expansion, index/header inequality, weight-payload read, and promotion
without a valid receipt. Test arbitrary requested subsets; an optional
`--require-all` release audit alone requires exact fifteen-artifact coverage.
An unexecuted artifact retains exactly `topology facts` or `grammar
micro-fixture`.

- [ ] **Step 2: Run the metadata tests and observe RED**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/metadata/precision_policy -q tests/metadata/precision_policy/test_full_metadata_conformance.py`

Expected: missing runner/receipt types and no valid promotions.

- [ ] **Step 3: Implement the streaming per-artifact runner**

`tools/precision_policy_metadata_conformance.py` accepts repeatable `--case`
arguments and an explicit output directory. For each requested artifact it
invokes the Task 4B resolver/producer and Task 4C classifier, streams every
shard header but no weight body, proves exact index/header tensor-key equality,
validates complete source/semantic/component accounting, computes every count
above without rendering Cartesian semantic members, and atomically writes only
that artifact's receipt. A header-manifest digest remains metadata identity and
is never labeled a checkpoint-content digest. A failed case writes no success
receipt and cannot change another artifact's tier.

- [ ] **Step 4: Execute only the requested promotion gates**

Kimi K3 establishes the resource baseline: classify exactly 497,220 normalized
records in one untimed warmup followed by five isolated single-process trials
on a Grace CPU node. Its p95 elapsed time must be at most 60 seconds and
incremental peak RSS at most 4 GiB. Persist all five trials. Before promoting
any other artifact, require that valid K3 baseline receipt; run the requested
artifact in an isolated process under both the same absolute limits and K3's
measured time/RSS. Aggregation cannot hide a failed trial or case.

Run K3: `PYTHONPATH=. .venv/bin/python tools/precision_policy_metadata_conformance.py --artifact-cases tests/fixtures/precision_policy/artifact_cases.json --case kimi_k3_mxfp4 --output-dir tests/fixtures/precision_policy/full_metadata_conformance`

Run another available case by replacing `ARTIFACT_CASE_ID`: `PYTHONPATH=. .venv/bin/python tools/precision_policy_metadata_conformance.py --artifact-cases tests/fixtures/precision_policy/artifact_cases.json --case ARTIFACT_CASE_ID --k3-baseline tests/fixtures/precision_policy/full_metadata_conformance/kimi_k3_mxfp4.json --output-dir tests/fixtures/precision_policy/full_metadata_conformance`

If staged metadata or Grace evidence is unavailable, do not run or fabricate
the promotion. Record the lower tier and proceed with Task 5.

- [ ] **Step 5: Validate and commit only successful promotions**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/metadata/precision_policy -q tests/metadata/precision_policy/test_full_metadata_conformance.py`

Run: `.venv/bin/pyrefly check tools/precision_policy_metadata_conformance.py tests/metadata/precision_policy/test_full_metadata_conformance.py`

Run: `/opt/homebrew/bin/ruff check tools/precision_policy_metadata_conformance.py tests/metadata/precision_policy/test_full_metadata_conformance.py`

Run: `/opt/homebrew/bin/ruff format --check tools/precision_policy_metadata_conformance.py tests/metadata/precision_policy/test_full_metadata_conformance.py`

Run: `git diff --check`

Commit the runner/tests plus only receipts produced by successful requested
cases. An all-fifteen release audit is an optional aggregate gate, not the
definition of Task 4C completion.

```bash
git add tools/precision_policy_metadata_conformance.py tests/metadata/precision_policy/test_full_metadata_conformance.py pyrefly.toml
git add tests/fixtures/precision_policy/full_metadata_conformance/ARTIFACT_CASE_ID.json
git commit -s -m "test(precision): promote metadata conformance"
```

### Task 5: Controller-Owned Semantic Precision Bootstrap and `explain-precision`

**Files:**
- Create: `nemo_rl/precision_policy/materialize.py`
- Modify: `nemo_rl/precision_policy/adapters/__init__.py`
- Modify: `nemo_rl/precision_policy/compiler.py`
- Modify: `nemo_rl/precision_policy/topology.py`
- Modify: `nemo_rl/precision_policy/topology_resolver.py`
- Modify: `nemo_rl/precision_policy/runtime_binding.py`
- Modify: `nemo_rl/precision_policy/source_discovery.py`
- Create: `nemo_rl/distributed/fleet_construction.py`
- Modify: `nemo_rl/distributed/worker_groups.py`
- Modify: `nemo_rl/distributed/virtual_cluster.py`
- Modify: `nemo_rl/utils/venvs.py`
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/workers/base_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- Modify: `nemo_rl/models/policy/teacher_worker_group.py`
- Modify: `nemo_rl/models/value/lm_value.py`
- Modify: `nemo_rl/models/value/tq_value.py`
- Modify: `nemo_rl/models/generation/__init__.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/generation/generation_router.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Create: `nemo_rl/algorithms/controller_setup_teardown.py`
- Modify: `nemo_rl/algorithms/opd.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/ppo.py:314-1100`
- Modify: `nemo_rl/algorithms/distillation.py:199-660`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Modify: `nemo_rl/algorithms/single_controller_utils/setup.py:884-1700`
- Modify: `nemo_rl/experience/rollout_reassembler_actor.py`
- Modify: `nemo_rl/experience/sync_rollout_actor.py`
- Modify: `examples/run_grpo.py`
- Modify: `examples/run_vlm_grpo.py`
- Modify: `examples/run_grpo_sliding_puzzle.py`
- Modify: `examples/nemo_gym/run_grpo_nemo_gym.py`
- Modify: `examples/run_ppo.py`
- Modify: `examples/run_distillation.py`
- Modify: `examples/run_grpo_single_controller.py`
- Modify: `nemo_rl/environments/nemo_gym.py`
- Modify: `nemo_rl/environments/interfaces.py`
- Modify: `nemo_rl/environments/utils.py`
- Modify: `nemo_rl/data/utils.py`
- Modify: `nemo_rl/data_plane/factory.py`
- Modify: `nemo_rl/data_plane/interfaces.py`
- Modify: `nemo_rl/data_plane/adapters/transfer_queue.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/factory.py`
- Modify: `nemo_rl/weight_sync/refit_plan.py`
- Create: `nemo_rl/weight_sync/direct_collective.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py`
- Modify: `tools/config_cli.py`
- Test: `tests/unit/precision_policy/test_materialize.py`
- Test: `tests/unit/precision_policy/test_topology_resolver.py`
- Create: `tests/unit/distributed/test_fleet_construction.py`
- Test: `tests/unit/distributed/test_worker_groups.py`
- Test: `tests/unit/distributed/test_virtual_cluster.py`
- Test: `tests/unit/distributed/test_virtual_cluster_batch_ports.py`
- Test: `tests/unit/utils/test_venvs.py`
- Test: `tests/unit/algorithms/test_grpo.py`
- Test: `tests/unit/algorithms/test_ppo.py`
- Test: `tests/unit/algorithms/test_distillation.py`
- Create: `tests/unit/algorithms/test_controller_setup_teardown.py`
- Test: `tests/unit/algorithms/test_opd.py`
- Test: `tests/unit/single_controller/test_setup.py`
- Test: `tests/unit/single_controller/test_single_controller_actor.py`
- Test: `tests/unit/single_controller/test_entrypoint.py`
- Test: `tests/unit/experience/test_rollout_reassembler_actor.py`
- Create: `tests/unit/experience/test_sync_rollout_actor.py`
- Test: `tests/unit/environments/test_nemo_gym_utils.py`
- Test: `tests/unit/environments/test_environment_utils.py`
- Test: `tests/unit/data/test_utils.py`
- Test: `tests/unit/data_plane/test_architecture_invariants.py`
- Test: `tests/unit/data_plane/test_tq_lifecycle.py`
- Test: `tests/unit/data_plane/test_tq_policy_routes.py`
- Create: `tests/unit/models/policy/test_semantic_precision_endpoints.py`
- Test: `tests/unit/models/policy/test_teacher_worker_group.py`
- Create: `tests/unit/models/value/test_lm_value.py`
- Test: `tests/unit/models/value/test_tq_value.py`
- Create: `tests/unit/models/generation/test_semantic_precision_endpoints.py`
- Test: `tests/unit/models/generation/test_generation_router.py`
- Create: `tests/unit/weight_sync/test_semantic_precision_handshake.py`
- Test: `tests/unit/weight_sync/test_refit_plan.py`
- Create: `tests/functional/conftest.py`
- Create: `tests/functional/test_fleet_construction_ray.py`
- Create: `tests/functional/test_single_controller_resource_handoff_ray.py`
- Create: `tests/functional/test_single_controller_tq_handoff_ray.py`
- Create: `tests/functional/test_semantic_precision_initial_sync.py`
- Test: `tests/unit/tools/test_config_cli.py`
- Test: `tests/unit/test_config_validation.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes before any GPU reservation or endpoint builder:
  `build_graph_topology_resolution_requests(policy_config: PolicyConfig) -> tuple[GraphTopologyResolutionRequest, ...]` and
  `SemanticPrecisionBootstrap.materialize(policy_config: PolicyConfig) -> CompiledPrecisionSelectionGroup | None`.
  The request builder freezes the effective configuration, resolves every model
  reference to an immutable revision, and returns exactly one `main` request plus
  every configured MTP and speculative-drafter request. It rejects an unresolved
  mutable revision, an omitted instantiated graph, or incomplete immutable
  checkpoint evidence before construction.
  It reconciles graph declarations from the same effective authorities that
  construct each runtime before emitting any request. The training main/MTP
  configuration is the final Megatron provider. Derive it from the HF config
  plus `hf_config_overrides` for HF/`megatron_lm` construction. Load the selected
  checkpoint's migrated `run_config.yaml` for `megatron_bridge` and for an
  already-complete converted-HF cache. A fresh converted-HF cache miss has no
  `run_config.yaml` to read at pre-resource bootstrap: derive the prospective
  provider from the pinned HF config instead, without triggering conversion,
  then require the post-conversion provider/`run_config.yaml` to be canonically
  identical before Phase 2, remote baseline, or communicator construction.
  In every case apply `megatron_cfg.model_overrides` and the
  explicit YAML `megatron_cfg` fields in the same precedence as
  `setup_model_config()`/`_apply_mtp_config()`. Inspect the resulting provider's
  exact `mtp_num_layers`; a raw YAML lookup is not an authority and omission of
  a YAML key cannot erase MTP present in the effective provider. The
  post-conversion check compares the complete canonical effective provider
  digest, selected adapter, graph universe, and MTP/draft-relevant fields, not
  only `mtp_num_layers`; mismatch tears down the registered endpoint before
  runtime source discovery begins.
  The enabled semantic-precision request builder accepts only the supported
  vLLM destination. Its rollout MTP additionally requires an MTP method (`mtp`
  or `deepseek_mtp`) and
  an exact positive `speculative_config.num_speculative_tokens`;
  `num_speculative_tokens == 0` is the canonical explicit disable and removes
  rollout MTP even if a method remains in the mapping. Negative, boolean, or
  contradictory token counts fail before construction. These declarations form
  one `mtp` graph whose lifecycle records training-only, source-served, or
  checkpoint-served participation.

  The main rollout declaration also follows its supported backend's actual
  model authority rather than overwriting an explicit destination identity with
  `policy.model_name`; its selected adapter, semantic topology, and complete
  layer universe must equal the source main graph or bootstrap fails. Semantic
  SGLang is rejected by the earlier pure capability gate before this resolution.
  The external `draft` graph determines training
  presence from exact `policy.draft.enabled`, not from whether `model_name` is
  populated, and reconciles an enabled training draft with an active vLLM
  `speculative_config.model`. When `draft.enabled` is true and `model_name` is
  absent, build a deterministic synthesized identity from the frozen main
  provider/revision digest plus every effective draft-construction field
  (`num_layers`, auxiliary-layer selection, and family-specific defaults); do
  not silently drop the graph or pretend it is the main graph. Equal resolved
  identities form one graph; unequal non-empty or synthesized-versus-external
  declarations are fatal rather than becoming two draft graphs or choosing one
  side. A graph declared on only one side remains explicit with the
  corresponding lifecycle, and a rollout-only graph requires complete immutable
  checkpoint evidence. Main, MTP, and draft each retain their own zero-based
  decoder-layer universe from their effective graph configuration and selected
  adapter; no auxiliary universe is offset by, inferred from, or folded into
  `main`.
- Produces one immutable paired registry entry per supported family:
  `PrecisionTopologyAdapterBundle(adapter_id: str, selection: SelectionTopologyAdapter, runtime: ModelTopologyAdapter)`.
  The bundle constructor requires both adapter halves to expose the same exact
  canonical `adapter_id`; registry IDs are unique. Phase 1 receives only
  `bundle.selection`, while Phase 2 resolves only `bundle.runtime` by the adapter
  ID already frozen in the selection. Separate independently configurable
  selection/runtime registries are forbidden. Every production resolution and
  replay-validation entrypoint receives its adapter authority explicitly:
  `resolve_selection_topology(..., *, adapters=...)`,
  `bind_runtime_source_intents(..., *, runtime_adapters_by_id=...)`,
  and `validate_compiled_precision_intent_group(..., *,
  runtime_adapters_by_id=...)`. All six Task 5 artifact entrypoints in
  `refit_plan.py`: `bind_refit_contexts()`, `bind_refit_context()`,
  `install_refit_contexts()`, `install_refit_context()`,
  `install_selected_refit_operations()`, and
  `install_selected_refit_operation()` instead require
  `context: BoundSemanticPrecisionRuntimeContext` and obtain the exact mapping
  only from that controller-local object. These authority parameters are
  required and have no empty, `None`, module-global, or lazy-import fallback.
  Refit validation receives one bound runtime context carrying the exact
  selected mapping retained by the bootstrap, not a newly constructed registry
  or a bare intent group. Compatibility wrappers outside the controller path
  may assemble the built-in bundle explicitly at their outer boundary, but
  neither resolver nor any refit-plan revalidation may call
  `_default_adapters()` or monkey-patch it. This makes Phase 1, Phase 2, and
  serialized worker-projection reinstallation prove against one adapter
  identity set while the full context remains local.
- Produces a controller-local `SemanticPrecisionBootstrap` with
  `selection: CompiledPrecisionSelectionGroup | None`,
  `context: BoundSemanticPrecisionRuntimeContext | None`,
  `phase1_requests_by_graph: Mapping[str, GraphTopologyResolutionRequest]`,
  `materialize(policy_config)`, and
  `bind_runtime_sources(request: RuntimeSourceDiscoveryRequest, results: tuple[RuntimeSourceDiscoveryResult, ...]) -> BoundSemanticPrecisionRuntimeContext | None`.
  An enabled bootstrap follows `NEW -> MATERIALIZED -> READY`: an identical
  `materialize()` repeat returns the original selection object, input drift is
  fatal, and Phase 2 may bind exactly once. The first successful materialization
  deep-freezes the canonical effective request/configuration separately for each
  graph, keyed by exact graph ID, and stores both each request digest and the
  ordered aggregate Phase 1 input digest. Immediately before invoking any
  runtime adapter's `supports()`, Phase 2 recomputes those digests, checks the
  retained graph keys against the frozen selection, and passes that graph's
  retained effective configuration. It never rereads the caller's mutable
  `PolicyConfig`, substitutes another graph's config, or validates support from
  a selection digest alone. An absent policy is an inert
  byte-for-byte no-op; it does not resolve graph requests, inspect factory
  capabilities, add kwargs, or mutate either policy or generation config.
- Produces one frozen `BoundSemanticPrecisionRuntimeContext` only after the
  complete Phase 2 bind succeeds. It retains the exact Phase 1 `selection`, the
  exact `RuntimeSourceDiscoveryRequest`, the complete ordered tuple of
  `RuntimeSourceDiscoveryResult` values, the selected adapter IDs and immutable
  `RuntimeAdapterAuthority` manifest, the process-local exact
  `runtime_adapters_by_id` mapping, and the resulting
  `CompiledPrecisionIntentGroup`. Its derived `runtime_context_id` commits to
  all serializable fields and the adapter-authority manifest. Construction
  rejects a request/result object not bound to the retained selection, missing,
  extra, reordered, or replaced adapter authority, or an intent object not
  reproducible from those exact inputs. The bootstrap builds a local candidate,
  runs the full canonical validation/round trip, and only then atomically
  publishes `_context` and enters `READY`; any exception leaves it in
  `MATERIALIZED` with no partial context or intents visible.

  `BoundSemanticPrecisionRuntimeContext` is controller-local, has no
  `to_wire_dict()` or `from_wire_dict()`, and implements an explicit pickle/
  cloudpickle `__reduce_ex__` guard that raises before any field is serialized:
  Python adapter instances, the
  aggregate runtime request, trusted expected-contributor sets, contributor
  identities, and raw discovery results never cross to a worker. Full replay
  validation and every bind/install/selected-operation entrypoint named above
  run controller-side against the retained
  request, results, and exact adapter mapping before a projection is created.
  The only transportable value is a distinct frozen
  `BoundSemanticPrecisionWorkerProjection` containing the already-bound intent
  wire, `runtime_context_id`, `selection_group_id`, exact bound plan-group IDs
  when Task 7 has produced them (otherwise exact phase `phase2_bound` plus an
  empty plan-ID tuple),
  selected adapter IDs, and their immutable implementation fingerprints plus a
  derived projection digest. Its strict `to_wire_dict()` /
  `from_wire_dict(payload, *, expected_adapter_authority=...)` pair requires an
  explicit, independently provisioned exact
  `RuntimeAdapterAuthority` scalar manifest, verifies the adapter IDs,
  implementation fingerprints, and
  canonical intent/plan identity, and rejects unknown/missing fields; it cannot
  invoke or require a `ModelTopologyAdapter`, re-run discovery, or manufacture
  a context from intents. The worker receives that manifest as a separate
  construction argument before any projection is accepted; comparing a
  projection to the authority embedded in that same projection is forbidden.
  A recursive negative
  payload scan proves that no raw contributor ID, trusted authority evidence,
  request/result record, request/result-only source/artifact/allocation locator,
  or adapter object appears in the projection. Canonical content-addressed
  `EvidenceSource.locator` and alias/source-owner evidence already committed in
  the validated intent wire are permitted and must remain byte-identical; they
  are not raw contributor authority. There is no no-argument decoder, global registry, rediscovery,
  family reselection, or intents-only compatibility path. In-process consumers
  receive the same full context object by identity; actual Ray boundaries
  receive only a fully validated minimal projection. Tasks 6-12 consume the
  full controller context or an already-bound plan/projection carrying its exact
  `runtime_context_id`; they may not reconstruct it from selection/intents
  alone.
- Produces a typed/versioned
  `SemanticPrecisionFactoryCapability(schema_version: Literal[1], selection_keyword: Literal["precision_selection"], adapter_authority_keyword: Literal["precision_adapter_authority"], construction_deadline_keyword: Literal["fleet_construction_deadline"], construction_protocol: Literal["two_phase_owner_v1"], worker_readiness_method: Literal["semantic_precision_construction_ready"])`.
  On the enabled path, endpoint constructors receive the exact selection and
  one shared absolute deadline through explicit keyword-only
  `precision_selection`, `precision_adapter_authority`, and
  `fleet_construction_deadline` arguments. The bootstrap derives the immutable
  authority manifest from the exact selected bundle mapping during Phase 1; it
  contains only canonical IDs/fingerprints, never Python adapter objects. A caller-supplied
  factory must expose
  `semantic_precision_factory_capability: SemanticPrecisionFactoryCapability`
  with `type(capability) is SemanticPrecisionFactoryCapability` and exact
  built-in field types/values before any endpoint builder is started. Structural
  lookalikes, mappings, mocks, subclasses, booleans accepted as integer schema
  versions, and objects with forged equality are rejected. The
  repository-owned TransferQueue `make_policy_factory()` returns a lazy callable
  object advertising that same capability and forwards all three keywords unchanged to
  `TQPolicy` without importing it on the disabled path; it is not treated as an
  unsupported external factory. After a
  custom factory begins construction, the controller requires its
  `begin_semantic_precision_build()` result to implement the exact pending-owner
  protocol below. On the absent-policy path controllers neither read the
  capability nor validate the added protocol; they preserve the legacy
  constructor call byte-for-byte and pass none of the three new keywords (not
  even `None`).
- Produces a controller-local, opaque
  `semantic_precision_fleet_identity: object` token and
  `shutdown_semantic_precision_fleet(cleanup_attempt_token: CleanupAttemptToken) -> ResourceTerminationAck` on
  `SemanticPrecisionSourceEndpoint` and `SemanticPrecisionIntentEndpoint`.
  The dependency-neutral definitions of `CleanupAttemptToken`,
  `ResourceTerminationAck`, `CleanupTokenState`, `CleanupTokenJournal`, and
  `TokenAwareShutdown` live only in `nemo_rl/distributed/fleet_construction.py`.
  Low-level environment, data-plane, distributed, generation, policy, worker,
  and weight-sync modules import those values/protocols from that module;
  `nemo_rl/algorithms/controller_setup_teardown.py` imports them in the same
  direction and must not be an import dependency of any low-level resource.
  Controller-only `RuntimeResourceDescriptor`, ledger orchestration, and
  handoff policy remain in the algorithms layer. `CleanupAttemptToken` is an
  exact frozen `(handoff_id, resource_id, epoch, attempt_id)` value and
  `ResourceTerminationAck` is an exact frozen value that repeats those fields
  with `terminated is True`. The underlying fleet—not a
  driver wrapper copy—journals tokens: retrying the same token after a lost ACK
  returns the same ACK without repeating the termination side effect, while a
  different, stale, or cross-resource token is rejected. Every auxiliary and
  synchronizer cleanup is exposed through the same token-aware adapter; no
  zero-argument callback participates in setup cleanup or runtime handoff. The
  common typed cleanup callable is
  `TokenAwareShutdown = Callable[[CleanupAttemptToken], ResourceTerminationAck]`;
  endpoint, pending-build, synchronizer, router, cluster, placement-group,
  queue/channel, Gym, value, teacher, finalizer, port/socket-holder, and other
  auxiliary adapters all implement it. An exact ACK is required before a
  cleanup journal entry becomes complete. A side effect followed by a lost ACK
  is retried with the same token, while a different/stale token is never used as
  a speculative retry.
  `WeightSynchronizer.shutdown()` remains the existing zero-argument legacy
  runtime API. Its interface adds the distinct final token-aware
  `shutdown_semantic_precision_resource(token) -> ResourceTerminationAck`
  wrapper. On enabled paths the wrapper is only a forwarding client for one
  driver-created repo-owned `SynchronizerCleanupAuthorityActor`; that authority
  retains the actual runtime resource handles, journals the stable token,
  performs underlying shutdown once, and exposes exact ACK/status. Both the
  driver wrapper and a cloudpickled Single Controller wrapper retain the same
  authority handle, never copied journal state. It is claim→submit→publish/
  readiness registered before synchronizer-owned resources, survives death of
  the Single Controller caller, and is terminated by the launcher only after
  its resource ACK/status is durable. Setup descriptors bind only this forwarding
  method/authority fingerprint. On enabled paths VllmGeneration retains only a
  non-owning synchronizer link, and its
  endpoint shutdown must not transitively invoke the separately registered
  synchronizer. The ledger/handoff manifest declares the synchronizer-before-
  generation-endpoint cleanup dependency. Absent-policy generation shutdown
  keeps its current zero-argument transitive call byte-for-byte.
  `ControllerSetupResourceLedger` creates the local canonical `handoff_id`
  before any registration and retains that same ID if ownership later transfers
  to Single Controller; setup-failure tokens use its DRIVER epoch, so successful
  transfer never requires a fallible re-enrollment of resource cleanup
  authority. GRPO keeps that same handoff/owner ID and journal across the
  atomic transfer into `GRPORuntimeResourceOwner`, so setup failure and runtime
  `finally` are two lifecycle states of one authority rather than independently
  minted cleanup tokens.
  Enabled repository-owned and v1 custom factories expose
  `begin_semantic_precision_build(..., precision_selection=selection,
  precision_adapter_authority=bootstrap.worker_adapter_authority,
  fleet_construction_deadline=deadline) -> PendingSemanticPrecisionEndpointBuild[T]`.
  `begin_semantic_precision_build()` returns a purely local pending owner with
  its stable fleet identity and token-aware no-throw shutdown adapter; it performs zero Ray,
  placement-group, router, engine, socket, or other remote allocations. The
  controller registers that owner in the setup ledger and only then submits
  `finish()`. As `finish()`'s first claimed allocation, the owner lazily creates
  and publishes a one-shot asynchronous cancellation-latch actor, submits its
  pending wait once, and wraps that pending `ObjectRef` as a nested field in a
  `FleetConstructionCancelToken`. Only after the latch handle and wait ref are
  locally owned may it claim an initializer, worker, model engine, router,
  socket holder, or other child. The shared implementation in
  `nemo_rl/distributed/fleet_construction.py` uses one locked state machine with
  exact `OPEN/FINISHING/SEALING/SEALED/CANCELLING/CLOSED` states and supports multiple parallel
  in-flight claims. Every allocation follows `claim -> submit -> incremental
  publish/fail`: claim an opaque allocation ID under the owner lock, invoke the
  allocator outside the lock, publish each returned handle immediately (without
  waiting for sibling allocations), and close every unsuccessful claim in a
  `BaseException` path. A published handle is registered before its first
  fallible initialization and before another completed future is consumed.
  Parallelism is retained; the protocol orders ownership publication, not all
  allocations globally.

  The controller creates one typed absolute `FleetConstructionDeadline` before
  submitting endpoint builds and passes it through every pending build into
  `RayWorkerGroup`. Its source is
  `policy.fleet_construction_timeout_s`, normalized as an exact finite positive
  non-Boolean number; `DEFAULT_FLEET_CONSTRUCTION_TIMEOUT_S` is exactly 1800.0
  seconds when the key is absent. This is a construction deadline, not the
  five-second best-effort cleanup deadline. Every wait/drain loop uses
  `deadline.remaining_s(phase)` and finite `ray.wait`/future waits. The first
  expiry resolves the cancellation latch, marks the owner cancelling, raises
  typed `FleetConstructionTimeout` naming the phase and outstanding claim IDs,
  and returns through nonjoining cleanup even when there is no sibling failure.

  `shutdown(cleanup_attempt_token)` first validates/journals the exact token,
  then atomically marks the owner cancelling, rejects future claims,
  claims all published handles, and signals every in-flight claim. A late
  allocator completion must call `publish`; the cancelling owner rejects the
  adoption, claims that handle for exact-once shutdown, and wakes `finish()` with
  a typed cancellation failure. A failed submit calls `fail` and cannot leave a
  permanently open claim. Ledger cleanup never waits for a claim or builder
  thread past its shared deadline. Latch-actor creation or wait-submission failure is
  owned by the already registered pending owner and follows the same
  `fail()`/cleanup path. A custom v1 `begin_semantic_precision_build()` that
  attempts a remote allocation is a protocol violation, not an allocation that
  callers are expected to recover. A returned endpoint must carry the pending owner's exact
  token. Constructor compatibility APIs may implement synchronous construction
  as begin/register-local-guard/finish, but may not leak a partially allocated
  fleet when `finish()` fails.

  A successful `finish()` calls `seal_to_runtime_fleet()` exactly once after
  readiness. Sealing is explicitly two phase and never performs a Ray RPC,
  wait, join, or user callback while holding the registry lock. Under the lock,
  it atomically changes `FINISHING -> SEALING`, closes new claims, moves the
  canonical runtime-child entries and their single cleanup lease in one pure
  local operation from the pending owner to the sealed fleet authority, and
  snapshots every helper-teardown action. The pending owner and runtime
  authority are views over one entry/attempt journal, so this move creates
  neither a zero-owner interval nor two independently claimable copies. The
  moved manifest includes every pooled
  `IsolatedWorkerInitializer` that is a required lifetime owner for its
  non-detached workers, plus every truly construction-only latch, pending wait
  ref, and temporary cleanup action in the separate helper snapshot. It then
  releases the lock and uses only bounded operations to
  resolve/cancel the pending latch ref, obtain the latch close ACK, terminate
  the latch actor, and release other construction-only helpers. Cleanup that
  observes `SEALING` atomically changes it to `CANCELLING` and claims runtime
  entries from that same moved journal. The sealer reacquires the lock only to
  publish `SEALED` and clear the cancel token/deadline/helpers when the state is
  still `SEALING`; if cleanup already won, or helper teardown fails/expires, it
  preserves/transitions to `CANCELLING`, never publishes `SEALED`, and dispatches
  bounded cleanup. A late `publish()` that observes `SEALING`,
  `SEALED`, or `CANCELLING` cannot reopen or join the owner; it independently
  claims the returned handle for token-aware shutdown outside the lock. Thus a
  concurrent controller cleanup can acquire the lock and complete against its
  independent five-second deadline even while seal helper teardown is hung.
  The seal does not discard runtime handles needed for shutdown: initializer
  actors remain alive for the training lifetime and runtime shutdown terminates
  them only after their owned workers. No cancellation latch or other temporary
  construction helper remains. Sealing itself
  uses `deadline.remaining_s("construction_seal")`; exhaustion is a construction
  failure that enters bounded owner cleanup rather than returning an endpoint
  with live helpers.
  Every completed endpoint, `RayWorkerGroup`, value/teacher fleet, and Gym
  wrapper rejects serialization
  until sealed and its `__getstate__` omits all construction-only state
  thereafter. Runtime ActorArgs carry only live runtime handles, canonical
  resource descriptors, and the descriptor-bound token-aware shutdown surface.

  A copied `FleetConstructionCancelToken(cancelled=False)` is forbidden because
  it cannot observe cancellation after crossing a process boundary. The real
  token contains the latch's pending cancellation `ObjectRef` inside a nested
  tuple/dataclass field, so Ray transports the reference itself instead of
  automatically dereferencing and blocking actor submission. A remote
  initializer polls only that ref with public
  `ray.wait(..., timeout=0, fetch_local=False)` immediately before child
  allocation; no per-worker registry RPC or cached Boolean is permitted. The
  latch is an async actor backed by one event: its wait method suspends and its
  cancel method sets the event. Do not implement a blocking method on Ray's
  default single-threaded synchronous actor, and do not try to repair it with
  `max_concurrency > 1`, whose concurrent method ordering would make the latch
  ambiguous. Cancellation resolves the one-shot ref for every token holder.

  The pending owner remains the authoritative local child registry. It records
  every pooled initializer immediately and every worker handle as soon as that
  handle reaches the driver. Each `IsolatedWorkerInitializer` retains a newly
  submitted child locally and the child must be non-detached. If cancellation
  races in the tiny interval after the final poll but before the driver receives
  the handle, cleanup kills the registered initializer and Ray actor-owner fate
  sharing kills that unpublished child. If the handle does reach the driver,
  normal late publication makes the cancelling local owner claim and kill it.
  This closes the race without adding one control-plane RPC per worker.

  The enabled path never calls Ray's private `__ray_ready__`. Every repository
  worker that can be constructed for an enabled semantic fleet implements the
  repo-owned lifecycle method
  `semantic_precision_construction_ready() -> Literal[True]`; the call is
  submitted immediately after the worker handle is published locally. Ray queues
  that method behind the actor constructor, so an exact built-in `True` result is
  the public readiness acknowledgement. Missing methods, RPC failure, `None`,
  `1`, a truthy object, or any non-exact Boolean result are construction
  failures. The v1 custom-factory capability contract requires the same method
  on every returned worker. The absent-policy path does not submit this RPC and
  preserves legacy readiness behavior byte-for-byte.

  `RayWorkerGroup` creates each pooled initializer through the same owner gate,
  publishes it immediately, submits worker creations in parallel, and drains
  completions incrementally with `ray.wait`; it never waits for the whole batch
  before ownership publication. Each resolved worker is registered before
  its readiness RPC, the next completed result, or final `_workers`
  assembly. On any submit, decode, construction, or initialization failure it
  first resolves the cancellation latch, cancels outstanding object refs where safe,
  and kills all pooled initializers and all known/late workers exact-once. This
  failure path must not call an unbounded `ray.get`, enter a joining executor
  context, wait for a hung initializer RPC, or leave `_initializer_pool`
  populated. Killing a pooled initializer also closes its non-detached,
  not-yet-published child ownership. Constructor exceptions, a hung initializer,
  a hung worker constructor, partial multi-worker completion, and result-decoding
  failure are all bounded by the propagated construction deadline. With no semantic policy,
  callers omit the cancel-token/owner arguments and `RayWorkerGroup` follows its
  exact legacy constructor/allocation path.

  The owner-aware boundary begins before the current pre-initializer work, not
  at `IsolatedWorkerInitializer`. `FleetConstructionContext` carries the same
  pending owner, nested-ref cancel token, and absolute deadline through
  `RayWorkerGroup._create_workers_from_bundle_indices()`, lazy
  `RayVirtualCluster.get_placement_groups()` / `_init_placement_groups()`,
  master-address/port discovery (including every uniqueness retry),
  `create_local_venv_on_each_node()`, and batched per-worker address/port
  discovery. The controller registers each `RayVirtualCluster` by stable
  auxiliary identity before a call that can lazily allocate its first placement
  group; the cluster's locked child registry then publishes each placement
  group and ready ref into that already-owned aggregate. Venv placement groups,
  `_env_builder` task refs, `_get_gpu_id_info` topology-probe refs, port-probe
  refs, and master-port probe refs are claimed before submission and
  incrementally published with exact cancel, kill, or remove actions before the
  next fallible allocation or wait. Unified-GPU placement-group topology sorting
  is part of construction and may not perform its current unbounded
  `ray.get(info_refs)`.

  Every enabled `ray.wait`, `ray.get`, retry wait, and backoff in these helpers
  receives `deadline.remaining_s(phase)`. The placement-group legacy readiness
  cap remains an upper bound only: use
  `min(180.0, deadline.remaining_s("placement_group_ready"))`, so a short
  semantic construction deadline is never expanded to 180 seconds. Master-port
  retry sleeps and all port batching are likewise deadline- and cancellation-
  aware. Topology-probe collection also drains incrementally under the same
  deadline and validates every bundle result before sorting. On any
  `BaseException`, including timeout, decode/assertion failure,
  or cancellation, cancel all pending refs, force-stop venv tasks where safe,
  remove every partial/ephemeral placement group, and clear the internal
  registries without an unbounded `ray.get` or join. A completion racing cleanup
  is claimed by the same publish-or-fail state machine and cannot resurrect a
  removed placement group or become an untracked worker prerequisite. On venv
  success, normalize and validate every path before atomically releasing result
  refs and removing the temporary placement group; unequal paths take the same
  cleanup path. The disabled-policy branch calls the existing helper signatures
  without a construction keyword and executes their byte-for-byte legacy
  ordering, 180-second cap, and return shapes.

  `GenerationRouterActor` has the same underlying-authority contract. The
  enabled Single Controller setup claims before actor `.remote()`, publishes
  the handle before requesting its base URL/readiness, and publishes/drains each
  ref under the same deadline. Its exact readiness ACK proves the HTTP server
  socket and daemon thread are serving. The actor itself journals token-aware
  shutdown, closes the server/socket and wakes the thread, returns the exact ACK
  idempotently, and rejects stale/cross-resource attempts. It is a persistent
  runtime descriptor, so success transfers it to the runtime handoff; failure,
  timeout, or late publication cleans it without a driver-only `ray.kill`
  wrapper. The absent-policy router construction and calls stay byte-identical.

  `Policy` and `Value` also stop constructing the currently unused
  `ray.util.queue.Queue` on enabled semantic paths. Its constructor hides a
  `_QueueActor` before caller ownership is possible, and no repository worker
  reads `pre_init_communication_queue`; enabled built-in worker builders omit
  that dead keyword and tests make `RayQueue()` fatal. If a future backend needs
  a pre-init channel, it must add a repo-owned explicit actor handle with the
  same claim/publish/deadline contract rather than reintroducing `RayQueue`.
  The absent-policy branch retains the current queue construction and worker
  kwargs byte-for-byte.

  `TQPolicy` and `TQValue` complete data-plane bootstrap under the same pending
  construction ledger rather than after an otherwise finished worker group,
  but they do not pretend their process-global TQ runtime is two fleets. The
  bootstrapping `TQPolicy` creates and publishes exactly one repo-owned
  `TQControllerAuthorityActor` with a deterministic identity before its
  fallible `start()` may invoke `tq.init(conf=...)`. The non-detached authority
  actor is created by the launcher/driver under claim -> submit -> publish,
  owns the external controller/bootstrap process, and is the single location
  of the controller and attachment token journals plus underlying abort/close
  side effects. Policy, Value, controller, or actor-process wrappers contain no
  copied journal. `TQValue` receives that exact authority explicitly,
  verifies/borrows its identity, and invokes only the non-bootstrapping attach;
  it cannot create or independently close a second controller. Each wrapper
  owns a distinct shared-authority `TQClientAttachmentAuthority` entry plus its
  worker fleet. Single Controller's standalone connect-only `dp_client` is a
  third explicit attachment (present independently of optional Value), and any
  rollout/finalizer/worker-local TQ connection is dependency-keyed to its
  already owned actor/fleet descriptor rather than hidden or double-counted.
  Exact attachment identities are de-duplicated before dependency closure. Each
  `setup_data_plane` ref is incrementally published and accepts exact readiness
  only through the shared cancel token and
  `deadline.remaining_s("tq_data_plane_attach")`. No unbounded `ray.get`
  remains. Closing an attachment releases only that client's dependency and
  never calls process-global `tq.close()`; the one controller authority is a
  persistent descriptor owned by Policy and identity-aliased/borrowed by Value,
  and its dependency fence calls `tq.close()` exactly once, last, after both
  live direct and actor/fleet-owned client attachments and worker fleets
  terminate. A partial Value failure closes only Value state while Policy
  remains live; whole-ledger failure first closes every identity-deduplicated
  attachment/fleet and then the shared controller. If the
  installed TQ API cannot distinguish attachment release from final controller
  close or expose deterministic status/close authority before allocation,
  semantic TQ construction fails closed before calling `tq.init`; it must not
  guess a global controller or maintain two driver wrapper token sets. On
  partial failure, lone hang, cancellation, or late worker attach, token-aware
  cleanup cancels refs and applies that dependency ordering without joining.
  The absent-policy TQ constructors retain the exact existing bootstrap calls,
  `ray.get`, kwargs, and results.

  `TQDataPlaneClient` is never cloudpickled as a live process-local client.
  Its enabled `__getstate__` emits only a wire-safe
  `TQAttachmentRebindDescriptor` (authority actor handle/identity, attachment
  role/ID, controller-config fingerprint, and owning runtime descriptor ID),
  and `__setstate__` creates an inert unbound proxy. During Single Controller's
  pre-adoption protocol, the actor asks the shared authority to claim a new
  process-local attachment, connects, proves an exact rebind ACK, and includes
  every rebound attachment ID in the actor adoption ACK before any data-plane
  operation or `run()`. Only after the driver validates that ACK does it release
  the old driver-process Policy/Value/standalone attachments with their stable
  tokens and complete the ownership transfer. Submission/init/rebind/ACK loss
  leaves the driver owner able to abort both old and claimed-new attachments;
  actor cleanup and driver recovery call the same authority actor with the same
  token, so a lost ACK cannot close the controller twice. Raw client handles,
  local journals, process-global TQ objects, or a copied close callback are
  forbidden in `SingleControllerActorArgs`.
  Single Controller token capture is another attachment, not a post-build
  exception. Before `generation.setup_token_capture()` creates the worker-local
  `TQDataPlaneClient`, sink, or source, the owning vLLM worker/fleet claims a
  persistent attachment ID against the same TQ controller authority and
  publishes that claim in its runtime descriptor. Client creation, exact sink/
  source readiness, and token-capture setup consume the original fleet-
  construction deadline/cancel token rather than a fresh
  `GenerationLifecycleDeadline`. Enabled setup must not call the legacy early
  `set_rollout_weight_version`; Task 12's authenticated post-ready/commit
  publication is the first allowed served-version stamp.
  Partial failure, lone hang, or lost ACK releases the claimed attachment with
  the stable cleanup token; success seals it as a child of the worker fleet so
  actor handoff and ordered worker cleanup retain it without serializing a
  process-local client or a copied journal.

  SGLang is deliberately not admitted to the initial semantic-precision
  release. Its current constructor and workers create driver-local loop threads,
  router/engine subprocesses, placement resources, actors, and health waits
  without one repo-owned process-tree authority whose shutdown remains callable
  while initialization is blocked. Therefore the pure backend-capability
  preflight rejects semantic-precision SGLang before Ray, venv, thread,
  subprocess, router, engine, poster, port, or HTTP-client allocation. No
  partial PID wrapper or assumption that `ray.kill()` recursively terminates a
  GPU child process is accepted. The absent-policy SGLang path remains byte-for-
  byte unchanged. A later versioned SGLang adapter may opt in only after it owns
  every router/engine child process, exposes token-aware graceful/force process-
  tree termination independent of blocked readiness, and passes real-process
  kill-tree/failure/hang/late-completion canaries. The generic selection,
  topology, transaction, and endpoint protocols remain capable of that future
  adapter without naming SGLang as supported now.

  Single Controller rollout reassembler/finalizer actors are not an exempt
  aggregate. On the enabled path,
  `create_rollout_reassembler_actors(..., construction=...)` receives the same
  `FleetConstructionContext` and replaces the current list comprehension with
  one claim -> `.remote()` -> immediate publish -> repo-owned
  `semantic_precision_construction_ready()` exact-`True` acknowledgement per
  actor. Each readiness ref is published and drained incrementally under the
  common construction deadline/cancel token before the final list is returned.
  The already registered pending aggregate exposes token-aware shutdown, and
  each completed actor becomes a persistent runtime descriptor entry. If the
  kth submit/readiness/decode fails, or a lone ref hangs, every earlier and late
  actor is claimed and killed without a join; no actor exists only in a local
  partially built list. The absent-policy branch retains the original helper
  signature, list-comprehension order, readiness behavior, and return shape.

  Communicator initialization is construction work under that same absolute
  deadline; it is not allowed to begin an unowned second timeout domain. On the
  enabled path, `create_weight_synchronizer()`, every supported vLLM transport
  synchronizer, and `DirectCollectiveRefitHandle` receive the exact
  `FleetConstructionContext`. Their Python constructors are side-effect-free;
  the controller registers the returned synchronizer/handle and only then calls
  its explicit `init_communicator(construction=...)`. Collective source/destination
  probes, NCCL-reshard rendezvous/unique-ID/group
  RPCs, checkpoint-engine/IPC/remote-sparse initialization and baseline
  refs, and the inline GRPO master-IP/port plus both endpoint `init_collective`
  refs are each claimed and published before the next fallible submission or
  wait. Every wait/get uses `deadline.remaining_s(phase)` and observes the same
  nested cancellation latch. A lone hung communicator ref therefore raises
  `FleetConstructionTimeout`, cancels outstanding refs, invokes the already
  registered token-aware owner cleanup without joining, and preserves the
  original phase/rank error. Failure, malformed result, cancellation, and late
  completion cannot leave a group or rendezvous actor unowned. The absent-policy
  branches omit the construction argument and preserve their exact legacy
  constructor, unbounded wait, metadata, and initial-sync behavior.

  Native Single Controller environment creation is also owned before clusters.
  The enabled controller creates its setup ledger before
  `setup_response_data()` and passes the same construction context through
  `data/utils.py` into `environments/utils.py`. For each configured environment,
  venv PG/task refs and the outer environment actor follow claim -> submit ->
  immediate publish; its repo-owned
  `semantic_precision_construction_ready() -> exact True` ref is published and
  drained under the shared deadline before the loop advances to another env or
  fallible dataset load. `EnvironmentInterface` keeps every existing
  polymorphic zero-argument `shutdown()` method unchanged and adds the distinct
  inherited final wrapper
  `shutdown_semantic_precision_resource(token: CleanupAttemptToken) -> ResourceTerminationAck`.
  That repo-owned wrapper journals and validates the token, invokes
  `self.shutdown()` exactly once, memoizes the exact ACK for same-token retry,
  and rejects stale/different tokens without invoking the legacy method.
  Runtime descriptors allowlist and bind only this final wrapper, never the
  subclass `shutdown()` directly. Enabled-path capability validation requires
  the registered environment class to inherit the exact
  `EnvironmentInterface.shutdown_semantic_precision_resource` implementation;
  a custom class that shadows it is rejected before any Ray/venv allocation.
  Each successful
  outer actor becomes a persistent runtime descriptor. Built-in envs may create
  internal workers only as non-detached children retained by that published
  outer actor, so killing it closes the constructor/publication gap by Ray fate
  sharing; a custom env that cannot advertise this v1 ownership/readiness/
  final-wrapper contract fails before allocation. A later env, dataset, readiness,
  or decode failure, a lone hang, cancellation, and late completion clean every
  earlier/late actor and venv resource without joining. The absent-policy path
  calls the original data/environment helpers with unchanged signatures,
  ordering, and results.
  `setup_response_data()` returns an explicit typed environment setup result;
  Single Controller retains both train and `_val_env_handles` instead of
  discarding the latter. The ledger and runtime handoff use the identity-
  deduplicated union of train plus validation envs, with binding paths for both
  aliases, so a validation-only environment always has a surviving handle and
  descriptor. Unsupported validation ownership is rejected before creation,
  never repaired by dropping its handle.

  NeMo Gym startup is also an owned pre-stage. On the enabled controller path,
  create a purely local `PendingNemoGymStartup` owner, register its token-aware
  aborter in the setup ledger, and only then call
  `start_nemo_gym_actor(pending, ..., construction=...)`; both it and
  `finish_nemo_gym_actor(..., construction=...)` receive the same context and
  absolute deadline. `make_actor_runtime_env()` venv PG/task refs, the Gym actor,
  spinup ref, and tokenizer/config refs follow claim -> submit -> publish/fail
  before any wait. Every wait is finite; failure, lone hang, cancellation, or
  late completion invokes the registered abort without joining and cannot leak
  work before a pending owner is visible. The absent-policy path retains the
  existing helper signatures, ordering, and results byte-for-byte.

  Each independently allocated supported worker fleet creates one stable token
  and every wrapper over that fleet returns the same token by object identity.
  Every allowed alias is declared before registration and delegates shutdown to
  the one owning endpoint. The token has no wire encoding
  and never enters user configuration. Endpoint `__getstate__` paths used to
  ship rollout clients omit the token and all retained selection/intent values;
  those cross only through the explicit worker calls below. Controller
  installation and failure cleanup deduplicate on this token, not on Python
  wrapper identity.
- Consumes after both source and destination endpoints exist, but before every
  special-case or general communicator:
  `source_endpoint.validate_realized_phase1_inputs(selection, phase1_requests_by_graph) -> None`,
  `source_endpoint.build_runtime_source_discovery_request(selection) -> RuntimeSourceDiscoveryRequest`,
  `source_endpoint.discover_runtime_sources(request) -> tuple[RuntimeSourceDiscoveryResult, ...]`, and
  `endpoint.install_precision_context(context: BoundSemanticPrecisionRuntimeContext, projection: BoundSemanticPrecisionWorkerProjection) -> None`.
  `bind_controller_runtime_sources(bootstrap, source_endpoint, destination_endpoint) -> BoundSemanticPrecisionRuntimeContext | None`
  first validates any freshly converted provider against the exact retained
  prospective Phase 1 request, owns that ordering, binds against the exact
  Phase 1 object, retains its exact request/results/adapter authority rather
  than projecting them away to bare intents, and installs once
  per distinct `semantic_precision_fleet_identity`. Endpoint methods may
  serialize only their explicit argument at an actual Ray boundary; user config
  never transports a selection, intent, producer, request, result, or runtime
  handle.
  The driver endpoint creates graph-bound `SourceMetadataProducer` proxies and
  invokes Task 4B's bulk factory locally. A proxy receives the trusted set from
  that local factory but submits only
  `RuntimeGraphSourceRequest.to_wire_dict()` to every required policy worker.
  Megatron, DTensor v1, and DTensor v2 worker entries immediately decode the
  request, emit their own typed contribution wires, and never receive the
  trusted set. The proxy decodes every returned contribution on the driver; the
  Task 4B factory alone combines and validates all PP/TP/EP contributors against
  controller-retained authority. Intent installation follows the same rule:
  on the enabled path each newly allocated policy/generation fleet passes
  `CompiledPrecisionSelectionGroup.to_wire_dict()` and the independently derived
  `RuntimeAdapterAuthority` wire as separate actor-builder arguments, never
  inside backend config. Each worker decodes and retains both before model or
  engine construction, then uses that retained manifest as the required
  `expected_adapter_authority` for every later projection decode. It never
  imports a topology adapter or trusts a manifest only because the incoming
  projection contains it. A supported shared-fleet alias verifies that its
  retained policy has the same selection and does not submit a second
  construction call.
  Every supported concrete policy source and vLLM synchronous/async destination
  endpoint—including vLLM's internal FlashInfer-TRTLLM runtime path—later
  submits only the minimal
  `BoundSemanticPrecisionWorkerProjection.to_wire_dict()` for installation and
  its worker entry decodes against the exact explicitly provisioned scalar adapter authority
  before retaining the typed value. The aggregate request/results and trusted
  contributor authority remain in the controller context. Absent-policy actor-builder
  arguments remain unchanged. No path calls
  `results[0]`, sends raw metadata dictionaries, or reuses
  `prepare_refit_info()` as semantic discovery. These Task 5 worker entries are
  backend-neutral wire/retention boundaries; they do not import, select, or
  probe a version-specific vLLM adapter, which remains exclusively Task 8.
- Extends `create_weight_synchronizer()`, each supported vLLM transport
  synchronizer, and the direct collective handle
  constructor with optional keyword-only
  `precision_context: BoundSemanticPrecisionRuntimeContext | None` and
  `fleet_construction: FleetConstructionContext | None`. Controllers
  pass the exact non-`None` Phase 2 context only on the enabled path and omit the
  keywords entirely on the absent path. The driver synchronizer validates and
  retains that full context only in a controller-local field excluded by every
  `__getstate__`/Ray serialization path; it separately retains the stripped
  worker projection, the independently construction-provisioned scalar adapter-
  authority manifest, and `runtime_context_id`. In Task 5 the value is additive semantic evidence
  only: collective, checkpoint-engine, IPC, NCCL reshard, and
  remote-sparse initialization must still execute their existing operational
  `policy.prepare_refit_info()`, `generation.prepare_refit_info()`, baseline,
  communicator preparation, and first-sync behavior. Phase 2 does not yet
  provide Task 7's bound physical plans, Task 9/10's executable loaders/exporters,
  or Task 11's transaction receipts, so suppressing a legacy action here would
  create a metadata-only path that never transfers valid weights. Task 12 may
  remove a transport's legacy preparation only in the same change that installs
  and tests its complete semantic executable replacement; Task 5 has no such
  branch. `create_weight_synchronizer()` must not forward
  `precision_context=None`, so every concrete absent-path constructor receives
  its exact legacy arguments. Synchronizers that exist only for unsupported
  generation destinations remain legacy-only: they receive neither semantic
  context nor construction authority and are unreachable after the enabled
  pre-resource backend gate.

  Because `SingleControllerActorArgs` cloudpickles its `weight_synchronizer`,
  each supported vLLM transport synchronizer implements an explicit
  serialization contract
  that rejects or removes the full context, adapter objects, request/results,
  and trusted evidence while preserving legacy transport state and the safe
  projection/context ID plus expected scalar authority manifest. Task 5
  actor-side legacy sync consumes only that safe
  evidence. Tasks 7-12 bind executable plans controller-side and serialize only
  authenticated plan/projection wires carrying the same `runtime_context_id`;
  no actor reconstructs the full context.
  `FleetConstructionContext` is consumable construction authority, never runtime
  state: successful or failed `init_communicator()` seals the synchronizer's
  construction phase and clears its owner/ledger/cancel-latch/deadline reference
  after all children are adopted or cleanup is dispatched. `__getstate__`
  rejects serialization while initialization is pending and otherwise omits the
  construction context entirely. A completed `DirectCollectiveRefitHandle`
  follows the same contract; no stale construction owner crosses into
  `SingleControllerActorArgs` after runtime handoff.

  The inline non-colocated collective path is not an implicit exception. It uses
  side-effect-free `build_direct_collective_refit_handle(policy, generation, *,
  precision_context, fleet_construction, ip, port, world_size,
  train_world_size) ->
  DirectCollectiveRefitHandle` from `weight_sync/direct_collective.py`. Building
  the handle only validates and retains exact arguments: it may not call either
  endpoint, create a communicator/group, submit Ray work, or prepare metadata.
  The controller registers the handle's stable identity and token-aware
  `shutdown(cleanup_attempt_token)` with
  the ledger, then explicitly invokes
  `handle.init_communicator(construction=fleet_construction)`. That method
  executes the unchanged source and destination `init_collective()` contract,
  validates every result, and runs the legacy metadata preparation/initial-sync
  sequence. An absent controller calls the original inline functions with
  byte-for-byte arguments and never imports this helper. No controller infers
  direct-collective readiness from an intent object, handle construction, or a
  communicator event.
- Produces `ControllerSetupResourceLedger(timeout_s: float = 5.0)`, shared by
  GRPO and Single Controller, with
  `register_pending_gym(abort: TokenAwareShutdown, *, descriptor: RuntimeResourceDescriptor)`,
  `replace_pending_gym_with_actor(shutdown: TokenAwareShutdown, *, descriptor: RuntimeResourceDescriptor)`,
  `claim_future(kind: SetupResourceKind) -> SetupFutureClaim`,
  `publish_future(claim, future, adopt_result)`, `fail_future(claim, error)`,
  `register_endpoint(endpoint: SemanticPrecisionIntentEndpoint, *, descriptor: RuntimeResourceDescriptor)`,
  `register_auxiliary(identity: object, shutdown: TokenAwareShutdown, *, descriptor: RuntimeResourceDescriptor, lifetime: SetupResourceLifetime = SetupResourceLifetime.PERSISTENT)`,
  `register_synchronizer(synchronizer: WeightSynchronizer, *, descriptor: RuntimeResourceDescriptor)`,
  `complete_transient(identity: object) -> ResourceTerminationAck`,
  `transfer_to_grpo_runtime_owner() -> GRPORuntimeResourceOwner`,
  `transfer_to_driver_runtime_handoff()`, and
  `cleanup_after_failure() -> None`. It registers each completed endpoint and
  constructed synchronizer plus every setup-owned placement group, deferred
  supported-generation reservation,
  value fleet, teacher cluster/fleet, generation router, Gym startup, and other
  auxiliary; atomically replaces a pending Gym aborter with the completed
  actor's shutdown callback; and performs idempotent no-throw failure cleanup.
  `prepare_grpo_controller_setup(config) -> GRPOControllerSetupSession` is a
  pure-before-allocation launcher seam: on the enabled path it performs Phase 1
  materialization, creates this ledger plus the one absolute construction
  deadline/cancel authority, and becomes the sole pre-runtime owner before
  `init_ray()` or `setup_response_data()` can allocate. The launcher passes that
  exact session into owner-aware `setup_response_data(..., construction=...)`
  and `grpo.setup(..., controller_setup_session=...)`; neither may create a
  second bootstrap, ledger, or deadline. Each native outer environment and venv
  resource is claim→submit→publish/readiness registered into the session, and
  the identity-deduplicated union of train plus validation environments enters
  the same runtime manifest. Dataset/environment/setup failure invokes session
  cleanup even when `grpo.setup()` never returns. On the absent-policy path the
  session is an allocation-free exact legacy sentinel, both callees receive
  their original arguments byte-for-byte, and no ledger method is touched.
  Regular GRPO returns a typed `GRPOSetupResult` containing the existing setup
  values plus `runtime_owner: GRPORuntimeResourceOwner | None`. Enabled setup
  sets that field only through `transfer_to_grpo_runtime_owner()`; the owner
  retains the exact persistent descriptor/callback journal and its dependency
  graph, is never Ray-serialized, and is the sole authority until launcher
  `finally`. Disabled setup sets it to `None` and preserves the legacy runtime
  shutdown calls.
  The result is deliberately not tuple-compatible: every in-repository caller
  (`run_grpo.py`, `run_vlm_grpo.py`, `run_grpo_sliding_puzzle.py`, and
  `nemo_gym/run_grpo_nemo_gym.py`) accesses named fields and must retain/close a
  non-`None` owner in an outer `finally`; no 13-value destructuring path can
  silently discard enabled ownership.
  The synchronous TransferQueue rollout actor is part of this setup handoff,
  not an unowned allocation inside `grpo_train_sync()`. When the enabled
  configuration selects the synchronous data-plane trainer, the setup session
  first creates and ledger-registers a purely local
  `PendingSyncRolloutActorBuild`. Only then may it claim the venv/runtime-env
  work, submit `SyncRolloutActor.remote()`, immediately publish the actor and
  its exact readiness ref, and attach the actor-local TQ client to the one
  shared `TQControllerAuthority`. Every step observes the same construction
  deadline/cancel token. The successful actor plus its distinct TQ attachment
  enter the persistent descriptor graph before
  `transfer_to_grpo_runtime_owner()` freezes it; `GRPOSetupResult` carries the
  actor handle and `grpo_sync.py` consumes that handle instead of constructing
  another actor. Actor/attachment teardown precedes the shared TQ controller.
  `SyncRolloutActor.shutdown_semantic_precision_resource(token)` delegates to
  the shared token journal, closes the process-local attachment exactly once,
  and returns the exact termination ACK; it never swallows a close exception.
  Its legacy zero-argument `shutdown()` and in-function construction remain
  byte-identical only on the absent-policy path.
  Nested helpers may not hide one of these allocations behind a returned
  aggregate: they receive an owner/ledger registration seam before allocating
  the first child. Every registry mutation and cleanup-state transition is
  protected by one lock.

  `SetupResourceLifetime` is the exact `PERSISTENT/SETUP_TRANSIENT` enum.
  Deferred supported-backend reservations that exist only until successful
  adoption are registered as
  `SETUP_TRANSIENT`; the resource stays OPEN and failure-cleanable throughout
  its fallible consumer/adoption step. On success the controller calls
  `complete_transient(identity)`, which uses the entry's journaled stable
  `CleanupAttemptToken`, waits only against the setup cleanup deadline, accepts
  only the exact matching termination ACK, and then records the entry ACKED.
  A side effect followed by a lost ACK is retried with the same token and cannot
  close the reservation twice. Failure or timeout leaves the entry owned and
  makes setup fail through bounded ledger cleanup. Transfer is legal only after
  every transient entry is ACKED; the frozen ledger audit history retains those
  ACKs, but the actor adoption descriptor/callback manifest contains only live
  OPEN `PERSISTENT` entries. Thus a successfully adopted port is not kept alive
  by a dead port-holder actor, and no transient becomes unowned between
  registration, adoption, acknowledgement, or failure cleanup.

  A callback cannot be attached before `Executor.submit()` returns its future.
  Parallel controller work therefore uses the same explicit `claim -> submit ->
  publish/fail` helper as fleet construction: register the claim, call
  `submit()`, then as the immediately following operation attach the done
  callback and publish the future to the ledger before awaiting it or executing
  any other fallible setup step. `Future.add_done_callback()` must correctly
  adopt an already-completed future. A submit exception closes the claim; a
  completion racing publication is adopted exactly once; completed results are
  published incrementally instead of through a dict comprehension followed by
  ordered `.result()` calls.
  Cleanup launches independent dependency roots concurrently but executes each
  registered synchronizer before its dependent generation endpoint, with one
  shutdown per distinct `semantic_precision_fleet_identity`, and waits against one shared
  five-second monotonic deadline, records cleanup errors without replacing the
  original setup exception, and can be disarmed only after setup transfers all
  resources to its caller. Registration happens before each pending endpoint's
  fallible `finish()` and immediately for every later completed auxiliary. If a
  parallel future completes after cleanup has begun, registration atomically
  claims its identity and dispatches that one shutdown immediately without
  extending or blocking the original shared deadline. A claimed-identity set
  spans the OPEN, CLEANING, CLEANUP_PENDING, and CLEANED states, so an owner observed by both a
  done callback and normal result collection is shut down exactly once. Late
  registration after either ownership transfer remains a programmer error. Partial construction,
  Phase 2, remote-baseline startup, communicator construction, communicator
  initialization, and initial-refit failures all use this bounded path.

  Single Controller uses one ownership state machine for every supported vLLM
  setup shape. The setup ledger owns every resource through setup and initial
  refit. Every successful setup path then
  atomically transfers the exact same live persistent entries plus the audited
  ACKED transient history—not an ad hoc selected subset—to a
  driver-local `ControllerRuntimeResourceHandoff`; the setup ledger becomes
  `DISARMED` only in that locked operation, so there is no bare disarm or
  unowned interval. The handoff retains the executable cleanup callbacks on the
  driver. As ownership-control metadata, `SingleControllerActorArgs` carries only a frozen
  `ControllerRuntimeAdoptionEnvelope(handoff_id, nonce,
  resource_descriptors, resource_manifest_digest, ownership_cell,
  cleanup_timeout_s)`, never the driver owner or its bound callbacks. Each
  canonical, wire-safe `RuntimeResourceDescriptor` contains a stable cleanup
  resource ID, exact `SetupResourceKind`, one canonical actor-arg binding path,
  all alias binding paths, an allowlisted cleanup-method ID, and its stable
  cleanup authority fingerprint. The ownership cell derives the epoch/attempt
  token from this descriptor; the wire descriptor never embeds a bound callback
  or a pre-authorized future attempt. Descriptor construction happens from ledger registration,
  not by scanning `ActorArgs`; opaque process-local fleet tokens remain omitted.
  Shared-fleet aliases produce one descriptor with multiple paths, while
  clusters, router, finalizers, Gym/env handles, value, and each teacher fleet
  have explicit entries. Each sealed `RayWorkerGroup` descriptor also binds its
  pooled initializer ActorHandles as persistent lifetime owners and declares a
  worker-before-initializer cleanup dependency; they are never classified as
  setup transients. The actor resolves every path through kind-specific
  binders, requires a one-to-one complete descriptor/resource mapping, rejects
  missing/extra paths, duplicate resource IDs, alias drift, or an unsupported
  cleanup method, and only then reconstructs a dormant local callback table.

  A repo-owned, non-detached `ControllerRuntimeOwnershipCell` is allocated under
  the same construction protocol before setup success: the ledger claims the
  control-cell resource before `ControllerRuntimeOwnershipCell.remote()`, fails
  that claim if submission raises, and immediately publishes any returned actor
  before submitting its repo-owned exact-`True` readiness RPC. Readiness is
  bounded by the shared construction deadline. A handle completing while cleanup
  races publication is adopted only for last-in-order termination and cannot
  become an untracked authority. The cell is therefore incrementally published
  under the setup ledger, not merely attached after construction. It is
  the single linearizable lease authority with exact forward transitions
  `DRIVER_OWNED -> ACTOR_OWNED_PENDING_RUN -> ACTOR_RUNNING -> CLEANING ->
  CLOSED`, the sole pre-run rollback edge
  `ACTOR_OWNED_PENDING_RUN -> DRIVER_OWNED`, and a monotonically increasing
  adoption epoch. Rollback invalidates the pending epoch before restoring the
  driver lease; no rollback from `ACTOR_RUNNING` is legal until actor death is
  established and a new recovery epoch is issued. After established actor
  death, the only recovery edges are `ACTOR_RUNNING -> RECOVERY_PENDING ->
  DRIVER_OWNED` and `CLEANING -> RECOVERY_PENDING -> DRIVER_OWNED`; no live
  actor may traverse them. Driver and
  actor cleanup methods must first present the current lease; retaining a Python
  copy of callbacks confers no authority. `SingleControllerActor.remote()` is
  inside the launcher's outer `try`, and the driver preserves the driver lease
  while submission and actor construction run. It next invokes the repo-owned
  `adopt_runtime_resources(envelope) -> ControllerRuntimeAdoptionAck` RPC. That
  RPC queues behind `__init__`, validates the exact manifest, installs the
  dormant owner, and atomically changes the cell from `DRIVER_OWNED` to
  `ACTOR_OWNED_PENDING_RUN`; the exact typed ACK is created at that same
  linearization point. The driver accepts only the exact built-in ACK type with
  matching handoff ID, nonce, manifest digest, epoch, and `adopted is True`.
  Only after validating it does the driver mark its local handoff actor-owned
  and submit `run(adoption_epoch)`, which atomically advances the cell to
  `ACTOR_RUNNING` before inspecting `weight_synchronizer.is_stale`. Adoption is
  therefore complete before `run()`, never deferred to it.

  `.remote()` submission/serialization failure, actor `__init__` failure,
  missing/malformed/non-exact ACK, ACK timeout/loss, and any driver exception
  before `run()` all enter one bounded rollback. Because `run()` has not been
  released, rollback may compare-and-set
  `ACTOR_OWNED_PENDING_RUN -> DRIVER_OWNED`; it then kills any candidate actor,
  cancels pending ACK refs, and has the driver owner clean every entry against
  one shared deadline. If the ownership cell itself is unavailable, the driver
  first kills the never-run candidate and then uses its emergency lease to
  clean; the actor's dormant/pending owner is forbidden from executing cleanup.
  A late valid ACK is rejected by the epoch/closed-state check and cannot revive
  ownership. Abrupt or injected driver failure is covered by the same outer
  launch guard; the ownership cell and candidate actor are non-detached and
  driver-fate-shared until the run lease is released.

  After successful transfer, the actor's outer `try/finally` encloses the
  pre-pump initial `_sync_weights()`, restore steps, pump creation, and loop and
  performs the one lease-authorized cleanup. The launcher `finally` consults the
  handoff state instead of separately shutting down env, generation, trainer,
  value, or teacher resources: before adoption it invokes driver cleanup; after
  adoption it requests the actor's bounded idempotent cleanup ACK. If the actor
  has failed, the driver first establishes actor death, atomically reclaims the
  lease, and runs its retained fallback actions. Thus setup success, actor
  submit, `__init__`, adoption ACK, `run()`, actor failure, and launcher
  `finally` always have exactly one lease holder. The cell also retains a
  manifest-indexed cleanup journal with exact
  `UNCLAIMED -> IN_FLIGHT(attempt_id) -> ACKED` entries. A cleanup executor must
  claim an entry before invoking its shutdown and report the same attempt ID
  after an exact ACK. If the actor dies before ACK, the driver may reclaim only
  after establishing death and retries the same idempotency token for each
  `IN_FLIGHT` entry; `ACKED` entries are never invoked again. Every registered
  resource shutdown therefore accepts a stable cleanup token and is idempotent.
  The guarantee is one fenced live cleanup executor and exactly one externally
  observed resource termination; an RPC whose actor died after applying the
  side effect but before ACK may be retried with the same token, so the plan
  does not make the impossible claim of exactly one callback invocation across
  process death. There is no semantic-precision special Megatron generation
  path; it was rejected before allocation and cannot introduce separate
  lifetime ownership or duplicate manual teardown.

  The ownership cell is a distinguished control-plane ledger/manifest entry,
  not an ordinary actor cleanup-journal member. The actor may never kill the
  cell that authorizes and records its own cleanup. On normal completion the
  actor first reaches `CLOSED` and returns an exact final cleanup ACK; the
  launcher then terminates the cell last. On pre-adoption failure the
  DRIVER-owned launcher terminates it after every data-plane entry, and on actor
  death or a lost final ACK the launcher uses public status/death evidence to
  recover or confirm `CLOSED`, finishes every remaining journal entry, and only
  then terminates the cell. Its descriptor is included in setup-ledger and
  canonical manifest accounting with a dedicated control-cell kind/path, but is
  excluded from actor callback coverage and alias deduplication. Normal run,
  rollback, recovery, and lost-ACK paths therefore neither leak the cell nor
  destroy the sole authority before cleanup is durably resolved.
- Task 5 proves bootstrap ordering, bounded setup cleanup, and propagation of
  failures that the existing legacy transport calls actually raise. It does not
  claim all-transport transactional fail-fast: silent peers, malformed/vacuous
  results, asynchronous NCCL/Ray failures, exact acknowledgement sets, poison,
  and atomic commit remain Task 11 (and their controller integration remains
  Task 12). No production fail-fast support claim may be made from Task 5's
  compatibility tests alone.
- Produces `render_precision_explanation(selection: CompiledPrecisionSelectionGroup, format: Literal["text", "json"]) -> str`.
  Neither bootstrap phase nor the CLI claims a Task 7 `plan_id` or an unobserved
  destination capability.

- [ ] **Step 1: Write failing bootstrap, adapter-bundle, and immutable-request tests**

```python
def test_absent_policy_is_byte_for_byte_noop_and_skips_request_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = legacy_policy_config_without_precision_policy()
    before = json.dumps(config, sort_keys=False, separators=(",", ":")).encode()
    request_builder = Mock(side_effect=AssertionError("request builder called"))
    monkeypatch.setattr(materialize, "build_graph_topology_resolution_requests", request_builder)

    bootstrap = SemanticPrecisionBootstrap(adapter_bundles=test_adapter_bundles())
    assert bootstrap.materialize(config) is None
    assert bootstrap.selection is None
    assert bootstrap.context is None
    assert json.dumps(config, sort_keys=False, separators=(",", ":")).encode() == before
    request_builder.assert_not_called()


def test_materialize_returns_the_same_frozen_selection_on_identical_repeat() -> None:
    bootstrap = SemanticPrecisionBootstrap(adapter_bundles=test_adapter_bundles())
    first = bootstrap.materialize(qwen30_policy_config())
    second = bootstrap.materialize(qwen30_policy_config())
    assert first is not None
    assert second is first
    assert bootstrap.selection is first


def test_materialize_rejects_policy_or_topology_drift() -> None:
    bootstrap = SemanticPrecisionBootstrap(adapter_bundles=test_adapter_bundles())
    assert bootstrap.materialize(qwen30_policy_config()) is not None
    with pytest.raises(ValueError, match="materialized Phase 1 input changed"):
        bootstrap.materialize(qwen30_policy_config(exclude_first=3))
```

Implement these additional literal RED cases in the same file:

- `test_enabled_policy_with_no_matching_adapter_fails_closed` supplies a complete
  unknown-family main graph and asserts zero endpoint calls.
- `test_enabled_policy_with_ambiguous_adapters_fails_closed` supplies two paired
  bundles whose selection halves both claim the main graph and asserts the
  conflicting adapter IDs in the error.
- `test_materialize_requires_complete_main_mtp_and_drafter_graph_set` configures
  one main, one MTP, and one external draft graph, then deletes each request in
  turn and asserts the exact missing graph ID.
- `test_request_builder_reconciles_training_and_rollout_graph_declarations`
  parameterizes training-only, source-served, and checkpoint-served MTP and
  external-draft configurations. It asserts one canonical graph per logical
  auxiliary, exact lifecycle, independently zero-based graph-local layer
  universes, and complete immutable evidence for rollout-only graphs.
- `test_request_builder_uses_effective_megatron_provider_for_mtp` parameterizes
  HF-derived, `megatron_lm`, converted-HF cache-hit `run_config.yaml`, fresh
  converted-HF cache miss, and `megatron_bridge` provider sources. The cache-miss
  case makes every filesystem `run_config.yaml` read and conversion call fail if
  attempted during Phase 1, and proves the prospective provider is derived from
  the pinned HF config. It applies YAML/model overrides through
  the production precedence and asserts the final provider `mtp_num_layers`,
  including checkpoint-enabled/YAML-omitted and explicit-YAML-zero cases; a spy
  makes any raw-only `megatron_cfg.get("mtp_num_layers")` decision fail.
- `test_fresh_conversion_must_match_prospective_phase1_provider_before_phase2`
  completes a fake conversion after endpoint construction and parameterizes
  exact equality plus changed architecture, layer universe, MTP, and adapter
  fields. Equality reaches Phase 2; every mismatch fails
  `validate_realized_phase1_inputs()` before discovery, remote baseline, or any
  communicator event and cleans the registered pending fleet exactly once.
- `test_request_builder_reconciles_training_provider_and_vllm_mtp_enablement`
  parameterizes the effective training provider's exact `mtp_num_layers` and
  vLLM `speculative_config.{method,num_speculative_tokens}`. Exact zero disables
  vLLM MTP even with `method=mtp`; positive counts create the rollout
  declaration; invalid vLLM types/counts and a rollout MTP incompatible with
  the effective training provider fail before construction. A stale
  `mcore_generation_config.num_speculative_tokens` field is not a semantic
  authority on the supported vLLM destination and changing it cannot change the
  request or selection; choosing Megatron generation instead is rejected by
  the earlier backend gate.
- `test_semantic_precision_rejects_unsupported_generation_backends_before_resolution`
  parameterizes SGLang, Dynamo, standalone TRTLLM, and Megatron generation
  backends and proves the pure
  capability gate raises before model-path resolution, adapter import, Ray,
  thread, subprocess, or endpoint construction. The absent-policy fixtures
  retain their complete legacy path, including SGLang `model_path` behavior.
- `test_enabled_draft_without_model_name_gets_synthesized_identity` sets
  `draft.enabled=true` with no model name and proves the graph is present, its
  identity changes when an effective draft-construction field changes, and its
  identity is stable across mapping order. A conflicting external vLLM drafter
  is fatal; `draft.enabled=false` creates no training draft regardless of stale
  optional keys.
- `test_request_builder_rejects_conflicting_draft_or_mtp_authority` expands a
  recipe where `policy.draft.model_name` differs from vLLM
  `speculative_config.model`, plus MTP enablement/configuration contradictions,
  and asserts failure after immutable identity resolution but before adapter
  selection, cluster reservation, or endpoint construction. It also proves an
  auxiliary configured on only one side is retained rather than silently
  dropped.
- `test_unresolved_mutable_revision_fails_before_construction` uses a mutable
  training-runtime main graph with a tag but no resolved immutable revision and
  asserts request construction fails without invoking an adapter.
- `test_bind_preserves_complete_runtime_context` asserts
  `context.selection is bootstrap.selection`, `context.source_request is
  request`, every retained result is the exact supplied frozen object,
  `context.intents.selection is context.selection`, the adapter mapping is the
  bootstrap's exact immutable selected mapping,
  `context.adapter_authority is bootstrap.worker_adapter_authority`, and all Phase 1, authority, and
  runtime-context digests agree.
- `test_bind_rejects_missing_phase1_adapter_runtime_half` calls the defensive
  Phase 2 adapter lookup with the adapter ID frozen by Phase 1 absent from the
  runtime map and asserts failure before source classification.
- `test_bootstrap_filters_production_registry_to_selected_runtime_adapter_ids`
  supplies Qwen, Nemotron, Kimi, and GLM bundles for a Qwen-only selection,
  spies on `bind_runtime_source_intents()`, and asserts its mapping contains
  exactly the frozen Qwen adapter ID. A different-family draft adds exactly its
  independently frozen adapter ID; no unused production bundle is forwarded.
- `test_worker_adapter_authority_is_scalar_selected_and_independent` asserts the
  Phase-1 authority manifest contains exactly those selected IDs/fingerprints,
  is the same object passed to every endpoint constructor, and round-trips
  without importing or serializing the Python adapters. Missing/duplicate/
  malformed fingerprints and a custom factory that drops or substitutes the
  manifest fail before its first remote allocation.
- `test_bind_cannot_run_twice_or_with_stale_results` first reaches `READY`, then
  rejects both an identical second bind and a result carrying the previous
  request digest.
- `test_bound_context_publication_is_transactional` injects failure at result
  validation, adapter replay, intent construction, context digest validation,
  and worker-projection construction. Every case leaves `bootstrap.context is
  None`, preserves `MATERIALIZED`, and permits no endpoint install or
  synchronizer construction.
- `test_worker_projection_excludes_controller_trust_evidence` recursively scans
  the canonical worker payload and proves that the aggregate request, raw
  results, expected-contributor set, every contributor ID, request/result-only
  source/artifact/allocation locator, and Python adapter object are absent. A
  checkpoint-plus-alias fixture proves canonical content-addressed evidence
  locators already present in the intent remain unchanged and are accepted. It decodes only with the exact selected
  independently provisioned scalar adapter-authority manifest and rejects a
  missing/extra/replaced adapter ID or fingerprint, altered intent, context ID,
  plan identity, or unknown field.
  Calling either the decoder or round-trip helper with an omitted, `None`, empty,
  superset, self-derived, or foreign manifest is rejected before intent
  acceptance; no
  helper imports or rebuilds a default registry.
- `test_full_runtime_context_is_not_serializable` requires `pickle.dumps()`,
  `cloudpickle.dumps()`, and direct use as a Ray argument to fail with the
  controller-local error before inspecting an adapter or trusted contributor;
  only `BoundSemanticPrecisionWorkerProjection` round-trips.
- `test_phase2_supports_uses_retained_graph_keyed_phase1_inputs` mutates the
  caller config after materialization, then asserts each runtime adapter receives
  the matching deeply frozen request/config for its own graph. Deleting,
  swapping, or changing any retained graph key/request digest is fatal before
  `supports()` or source discovery, and a runtime half that supports main but
  not the independently configured drafter is rejected.

Test `PrecisionTopologyAdapterBundle` itself with unequal adapter IDs, a missing
half, duplicate registry IDs, and a runtime half whose `supports()` result
contradicts the retained graph-specific Phase 1 effective configuration. The production registry must
fail during bootstrap construction for the first three cases and before source
classification for the capability contradiction. Assert the request builder
does not mutate `PolicyConfig`, snapshots the final effective config exactly
once, sorts graph IDs deterministically, preserves independent graph-local
layer universes, and never folds MTP or a different-family drafter into `main`.

In `tests/unit/precision_policy/test_topology_resolver.py`, add
`test_selection_resolver_requires_explicit_adapter_registry` and prove an
omitted, `None`, empty, duplicate, or superset registry cannot fall back to a
module global or lazy import. In `test_materialize.py`, patch every legacy
`_default_adapters` helper to raise and prove the paired registry still drives
both phases. In `tests/unit/weight_sync/test_refit_plan.py`, remove every
`MonkeyPatch` of `_default_adapters`; pass the complete bootstrap-retained
`BoundSemanticPrecisionRuntimeContext` through `bind_refit_context[s]()`, and
`install_refit_context[s]()`, plus both
`install_selected_refit_operation[s]()` entrypoints. Add literal missing request/results, missing,
extra, wrong-ID, post-bootstrap replacement, another-family adapter, bare
intents, and forged context-ID cases; all fail before reclassification,
projection, or context installation. An exact local context must reinstall
deterministically; its controller-local replay invokes
`validate_compiled_precision_intent_group()` with `context.intents` and the
exact `context.runtime_adapters_by_id` mapping, without calling `supports()` to
choose a family again, then
emit only the stripped worker projection. Assert these functions have no
selection-plus-intents overload, no separately supplied adapter-mapping
keyword, and no full-context wire encoder. Repeat the
bare/missing/foreign-context matrix through single and batch selected-operation
installation so neither transitive public API preserves an authority bypass.

- [ ] **Step 2: Write failing controller-ordering and custom-factory tests**

Use one shared call ledger in each controller test. Endpoint factories append
their own start/done events. Every weight-synchronizer factory or direct
communicator constructor appends `communicator_construct` at entry before it
returns an object, and the returned mock appends `communicator_init` at the
start of `init_communicator()`. Those are distinct mandatory events; neither may
be represented by a single generic `communicator` marker. Pin these exact tests:

```python
@pytest.mark.parametrize(
    "communicator_path",
    ("legacy_collective", "nccl_reshard", "remote_sparse", "checkpoint_engine"),
)
def test_grpo_phase2_binds_after_both_endpoints_before_every_communicator(
    communicator_path: str,
) -> None:
    events = run_grpo_setup_with_precision(communicator_path)
    assert events.index("phase1") < events.index("policy_builder")
    assert events.index("phase1") < events.index("generation_builder")
    assert events.index("policy_builder_done") < events.index("phase2")
    assert events.index("generation_builder_done") < events.index("phase2")
    assert events.index("phase2") < events.index("communicator_construct")
    assert events.index("communicator_construct") < events.index("communicator_init")
```

For `legacy_collective`, additionally assert exact order `phase2 ->
direct_handle_construct -> ledger_register_direct_handle -> communicator_init`.
Make both endpoint `init_collective()` methods raise if called during handle
construction, then fail source init, destination init, result validation, legacy
metadata preparation, and initial sync in turn; every failure must find the
handle already registered and invoke its idempotent shutdown once.

Add `test_grpo_materializes_before_policy_or_generation_builder`,
`test_grpo_generation_first_path_receives_same_selection_handle`,
`test_grpo_absent_policy_preserves_legacy_factory_kwargs`,
`test_grpo_phase1_failure_starts_no_endpoint_builder`, and
`test_grpo_enabled_precision_rejects_custom_factory_without_v1_capability`.
The generation-first test asserts the policy and generation constructor receive
the same selection and adapter-authority manifest objects by identity. The absent test compares complete recorded args,
kwargs, and input-config bytes against the legacy fixture. The custom-factory
test fails before cluster reservation and before either endpoint builder; a
factory whose `semantic_precision_factory_capability` equals
`SemanticPrecisionFactoryCapability(1, "precision_selection",
"precision_adapter_authority", "fleet_construction_deadline", "two_phase_owner_v1",
"semantic_precision_construction_ready")` receives the exact selection and
independently derived authority manifest plus the deadline keywords by identity
and proceeds. Parameterize a
plain dict, `Mock`, subclass, `schema_version=True`, wrong selection/deadline
keyword, wrong authority keyword, wrong construction protocol, wrong readiness method, and a capability object with
forged `__eq__`; every case
must fail the exact-type validator before construction. A custom v1 factory
whose worker lacks `semantic_precision_construction_ready()` or returns anything
other than exact `True` fails under the construction deadline and cleans its
registered pending owner. In
`tests/unit/data_plane/test_architecture_invariants.py`, add
`test_semantic_precision_transfer_queue_factory_advertises_v1_and_forwards_authority`
using the real `make_policy_factory()` with `data_plane.enabled=True`; assert
the returned callable advertises the exact constant and `TQPolicy` receives the
selection, scalar adapter-authority manifest, and shared absolute deadline by identity through
`begin_semantic_precision_build()`. The pending
owner is visible before `finish()` allocates a TQ fleet. The absent data-plane
case still returns `None`.
In the same architecture suite, add
`test_cleanup_protocols_live_in_dependency_neutral_distributed_module` and an
isolated import-smoke test. Import every low-level lifecycle consumer
(`environments.interfaces`, generation/policy interfaces,
data-plane adapters, workers, and weight synchronizers) with
`nemo_rl.algorithms.controller_setup_teardown` blocked; each must import the
exact `CleanupAttemptToken`, `ResourceTerminationAck`, and journal protocol by
identity from `distributed.fleet_construction`. Then import the controller
ledger and prove the dependency direction is low-level -> distributed values
and algorithms -> distributed values, never low-level -> algorithms. AST/import
graph assertions reject duplicate lifecycle class definitions or an algorithms
import from any listed low-level module.

Add `test_single_controller_materializes_before_clusters_or_threadpool`,
`test_single_controller_phase1_failure_starts_no_cluster_or_builder`,
`test_single_controller_generation_first_vllm_path_receives_same_selection_handle`,
`test_single_controller_absent_policy_preserves_legacy_builder_kwargs`,
`test_single_controller_phase2_precedes_supported_vllm_communicator`. The case
pins the order `claim endpoint builds -> submit -> publish futures/callbacks ->
incrementally publish endpoint results -> Phase 2 -> Gym result -> communicator
construction -> communicator initialization -> initial refit`; a Phase 2
exception must surface without waiting on the Gym future. It covers both
construction and initialization of the communicator created later in setup.
Add controller tests where `submit()` raises, a future completes before
`add_done_callback()`, the second endpoint completes before the first, and one
endpoint hangs after its claim while its sibling fails. The completed endpoint
must already be owned, the pending build must be cancelled without a joining
`ThreadPoolExecutor.__exit__`, and the primary error must escape within the
ledger deadline. These tests explicitly reject the impossible event ordering
"callback installed before submit" while proving the real immediate-after-submit
ordering closes the completion race.
Repeat the sibling-hang/failure race through each of GRPO's three current
parallel construction blocks. On the enabled path assert no
`ThreadPoolExecutor.__exit__`/atexit join runs, the published/pending resources
are cancelled through the ledger, and both the controller call and a subprocess
exit within the construction-plus-cleanup budgets. The absent-policy fixture
must still enter each original executor context with the exact legacy calls.

In `tests/unit/experience/test_rollout_reassembler_actor.py`, exercise enabled
Single Controller creation with more than one finalizer. Fail, hang, malform,
cancel, and late-complete each actor submission/readiness position; assert the
pending aggregate is ledger-owned first, each actor and readiness ref is
published before the next submission, and all earlier/late actors receive one
token-aware shutdown within the shared deadline. Accept only the repo-owned
exact-`True` readiness result. On success, assert every finalizer has one
persistent runtime descriptor and transfers through the unified handoff. The
absent-policy snapshot must retain the existing list-comprehension calls and
perform no owner, deadline, readiness, or token operation.

In `tests/unit/single_controller/test_setup.py`, parameterize supported vLLM
setup with and without Gym and assert both return the same unified driver
handoff contract containing every live persistent setup resource identity plus
the same ACKED transient audit history. Completed deferred reservations are
absent from actor descriptors. Pin the sole lease
holder at setup-ledger transfer (`DRIVER`), actor submission (`DRIVER`), actor
`__init__` (`DRIVER`, actor owner dormant), exact adoption linearization
(`ACTOR_PENDING_RUN`), driver ACK validation (`ACTOR_PENDING_RUN`), `run()`
entry (`ACTOR`), and cleanup (`CLOSED`). No state may expose zero or two cleanup
leases, and the special path may not drop resources merely because it already
ran its initial sync. Assert the canonical descriptor manifest contains one
entry for a shared policy/generation fleet with both alias paths and distinct
entries for train/inference clusters, router, every finalizer, Gym/env actors,
value, and each teacher fleet. Each worker fleet descriptor binds its pooled
initializer lifetime-owner handles and enforces workers-before-initializers
shutdown. Opaque Python fleet tokens and bound callbacks
must not serialize. Include the ownership cell as one distinguished manifest
entry and assert it is not present in the actor callback journal. On setup
failure while DRIVER-owned it is terminated last after every data-plane entry.
Inject ownership-cell `.remote()` submit failure, readiness failure/hang, and a
completion racing setup cleanup; assert claim -> submit -> publish/fail ordering,
the common construction deadline, and no leaked late control actor.

In `tests/unit/single_controller/test_entrypoint.py`, add
`test_semantic_precision_actor_launch_handoff_is_failure_atomic`. Parameterize
`SingleControllerActor.remote()` submission failure, argument serialization
failure, actor-constructor failure, adoption-RPC submission failure, timeout,
lost ref, raised RPC, mapping/Mock/subclass ACK, wrong handoff ID/nonce/manifest/
epoch, `adopted=None`, integer `1`, and a driver exception immediately before
and after each launch operation. The launch `try` must already be active, no
`run.remote()` may occur, the candidate actor/ACK ref is cancelled or killed
without joining, the lease is rolled back to the driver, and every resource is
bounded-cleaned exactly once. Release a valid ACK after rollback and prove it
cannot reacquire the closed epoch. Add the success test requiring exact order
`driver owns -> remote submit -> constructor -> adoption RPC -> exact ACK ->
driver disarms -> run submit`; the launcher's `finally` must route cleanup
through the current handoff owner and must not also execute the legacy manual
env/teacher/generation/trainer/value loops.

In `tests/unit/single_controller/test_single_controller_actor.py`, reject a
missing/mismatched adoption envelope, direct `run()` before the adopted epoch,
duplicate adoption, duplicate run release, and cleanup under the driver lease.
Also reject missing/extra/duplicate descriptors, an unknown kind/method, wrong
canonical or alias actor-arg path, a shared-fleet alias split into two cleanup
IDs, and cluster/router/finalizer/Gym/teacher binding drift before changing the
driver lease. A complete manifest binds every descriptor once without scanning
unlisted args or deserializing a driver callback.
After successful adoption, fail the first refit, restore, pump construction,
normal loop, and actor cleanup ACK in turn; the actor owner runs all cleanup
actions and preserves the primary error. For every manifest entry, kill the
actor before callback invocation, after the shutdown side effect but before its
ACK, and after ACK. The launcher must establish death, traverse the explicit
recovery edge, skip `ACKED` entries, and retry `IN_FLIGHT` entries with the same
attempt/idempotency token; the sentinel observes one termination even where the
mock RPC records a retry. A repeated actor/launcher cleanup is a no-op, and the
test explicitly distinguishes exactly-once claim/termination from callback
invocation count under an unknown outcome.
For the distinguished ownership-cell entry, separately cover normal `CLOSED`,
pre-adoption rollback, actor-death recovery, and a lost final cleanup ACK. In
each case the launcher reconciles public state, terminates the cell last, and
leaves neither a live control actor nor an actor-side attempt to self-terminate
the authority.

In `tests/functional/test_single_controller_resource_handoff_ray.py`, use real
Ray CPU actors to prove the ownership-cell claim precedes actor submission,
readiness/publication is bounded, and cleanup racing a late cell handle
terminates it. Then prove the adoption RPC queues behind a blocked constructor,
constructor failure reaches the driver, an adoption timeout/late ACK is revoked
before any run RPC, and an exact ACK permits one run. Kill the driver-side
launch task before and after adoption and kill the Single Controller during
run; the non-detached ownership cell plus epoch protocol must leave exactly one
side able to claim cleanup and all sentinels must terminate within the fixed
handoff/cleanup deadlines. The canary uses only repo-owned ACK/status methods,
never private `__ray_ready__`. Add a real token-aware sentinel that applies
termination and loses its first ACK; driver recovery retries the same token,
observes one termination, rejects a changed token, reaches `CLOSED`, and then
terminates the ownership cell last.

Until PPO and distillation have their own complete source/destination binding
integration, add `test_ppo_semantic_precision_fails_before_any_resource` and
`test_distillation_semantic_precision_fails_before_any_resource`. Their setup
entrypoints inspect `policy.precision_policy`; distillation additionally inspects
`teacher.precision_policy`. Either configured distillation policy raises an
actionable unsupported-algorithm error before config mutation, Logger,
CheckpointManager, dataloader, cluster, Gym, policy, teacher/value, or generation
construction. Byte-snapshot the input and make every resource factory raise if
called. Parameterize distillation with semantic precision on student only,
teacher only, and both; all three stop at the same first-line guard and neither
model config is mutated. The absent-policy counterparts retain the exact legacy call ledger.
Exercise the production launchers too: `examples/run_ppo.py` and
`examples/run_distillation.py` invoke the same pure guard immediately after
resolved config validation and before `init_ray()`, `setup_response_data()`, or
any logger/checkpointer/resource factory. Monkeypatch every downstream call to
raise if reached and cover student-only, teacher-only, and both-model
distillation policies. Launcher absent-policy snapshots retain their original
call arguments/order. The algorithm-entry guards remain as defense in depth.
This fail-closed guard is removed only by a later change that gives the
algorithm the same Phase 1, Phase 2, initial-sync, and teardown tests as GRPO.

Add `test_shared_alias_installs_once_per_fleet_identity` and
`test_distinct_fleets_each_install_once`. The shared fixture uses distinct
supported wrapper aliases that return the exact same opaque identity token and
asserts one underlying install call. The distinct fixture uses two tokens and
asserts one install on each. Repeated property reads
must return the same token, tokens must have no wire encoder, and a custom
factory result missing the token fails closed through setup teardown. Serialize
each rollout-facing wrapper through its production `__getstate__` path and
assert no fleet token, selection, intent, producer, request, or result is present.
Register the shared fleet source-first and generation-first and require the same
one descriptor with both exact ActorArgs binding paths and the same canonical
digest. A duplicate identity with a conflicting cleanup method/fingerprint,
resource kind/ID, lifetime, dependency set, journal/action authority, or alias
claim fails atomically; the original descriptor remains unchanged and cleanup
still owns it.
For every repository-owned policy and generation factory, pause `finish()`
before its first actor allocation and assert the pending owner is already in the
ledger. Inject failures after each actor, router, engine, value fleet, teacher
placement group/fleet, and deferred supported-generation
allocation, and assert the published owner tears down every partial allocation.
Add `test_begin_semantic_precision_build_is_purely_local`: make every Ray,
placement-group, socket, and engine allocator raise if touched and prove begin
returns the local owner/token without calling one. Then fail creation and
wait-submission of `finish()`'s first cancellation latch and prove the
already-ledger-owned pending build is cleaned once. Tests must make the nested router/value/teacher helpers
raise if called without a pre-existing allocation claim so an aggregate return
cannot hide their children.

In `tests/unit/distributed/test_fleet_construction.py`, add
`test_parallel_claim_submit_incremental_publish_and_fail` with two allocations
in flight: complete the second first, prove it is published before the first is
observed, then fail the first and prove cancellation cleans the published handle
once. Add submit-before-return, completion-before-callback, submit-raises,
cleanup-before-publish, cleanup-after-publish, and late-success cases. A sibling
already claimed before cancellation may have been submitted; it must publish or
fail and be cleaned, while every new claim is rejected. Do not assert that all
sibling allocators are serialized or never invoked. Patch any legacy snapshot
token to remain `cancelled=False`, resolve the nested cancellation latch, and
prove a remote poll still observes cancellation. Add a hung builder and show
controller cleanup returns within the one shared deadline without joining it.
Unit-test cancel-before-wait, wait-before-cancel, repeated cancel, and a malformed
or already-failed nested ref. The async one-shot latch must preserve both valid
orders without a synchronous actor or `max_concurrency` setting. Parameterize
the exact 1800-second default, a valid configured override, Boolean/zero/NaN/
infinite rejection, and prove configuration is read only on the enabled path.
Add `test_success_seal_releases_all_construction_only_resources`: after every
claim and exact readiness ACK, sealing closes the owner, resolves/cancels the
pending wait ref, obtains the latch close ACK, terminates the latch and every
truly temporary helper, and transfers both worker handles and their pooled
initializer lifetime owners into the fleet cleanup authority. Repeated
seal/claim and a late publication cannot reopen it; no construction latch/ref,
pending owner, or construction deadline appears in the returned endpoint or
runtime handoff. Extend the real-Ray canary to prove a worker remains alive and
callable after sealing, its initializer remains alive as the required Ray owner,
and ordered runtime shutdown terminates the worker before that initializer;
only then may the initializer handle disappear.
Add `test_success_seal_never_waits_under_registry_lock`: stop helper teardown at
a barrier after the under-lock `SEALING` manifest/lease move but before helper
teardown, concurrently start controller cleanup and publish a late allocator
completion, and prove both acquire the registry and return within their
respective cleanup/construction deadlines without waiting for the blocked
helper RPC. At every barrier observation exactly one journal/lease owns each
worker and initializer; cleanup atomically claims the moved entries once and
the late handle is independently claimed once. Then release the barrier and
assert the seal publishes either `SEALED` or the one canonical `CANCELLING`
failure, never reopens claims or installs already-cleaned handles, and leaves no
temporary helper or late handle alive.

Add `test_pending_owner_cleanup_racing_finish_cancels_future_allocations`: block
one allocator after its claim, start ledger cleanup, release it, and assert the
late handle is shut down exactly once, no post-cancellation claim is admitted,
`finish()` terminates with the typed cancellation, and the original controller
failure remains the raised exception. Repeat with cleanup immediately before
and after publication and with two concurrently claimed allocators to cover all
lock orderings without eliminating supported startup parallelism.

In `tests/unit/distributed/test_worker_groups.py`, add owner-aware construction
tests using the real `RayWorkerGroup` allocation loop with a deterministic fake
Ray API. Assert the live cancel token/owner is installed before the first pooled
initializer allocation, every initializer is registered immediately, worker
creation is parallel, and completions are published in `ray.wait` completion
order before the repo-owned readiness RPC, the next result, or final `_workers`
assembly. Make the constructor block and prove
`semantic_precision_construction_ready.remote()` remains pending behind it;
then release it and accept only exact `True`. Parameterize a missing method,
`None`, integer `1`, truthy mock, raised RPC, and a readiness ref that remains
pending until the construction deadline. Assert the enabled implementation never
touches `__ray_ready__`, while the absent path neither requires nor submits the
new method.
Inject failure and an indefinitely pending future at initializer allocation,
initializer construction, worker submit, worker construction, driver child publication,
worker initialization, and final group assembly. In every case resolve the
cancellation latch, assert outstanding refs are not awaited, and assert all known and late
handles plus every pooled initializer are killed exactly once; `_initializer_pool`
and partial `_workers` are cleared. Add a barrier race where cancellation lands
between the final zero-time `ray.wait` poll and child `.remote()`; kill or stall the initializer that
owns an unpublished non-detached child and prove ownership cleanup does not join
the initializer RPC and Ray fate-sharing removes the child. If the child result
arrives first, prove incremental driver publication claims it instead. The absent-policy test records complete `RayWorkerGroup`
args, actor options, allocation order, `_initializer_pool`, and `_workers`
behavior against the legacy fixture and proves no cancel-token/owner keyword or
cancellation RPC is added.
Add a lone-hang case with no failing sibling: leave the only initializer-create
or worker-ready ref pending, advance the injected monotonic clock to the
construction deadline, and require `FleetConstructionTimeout`, latch resolution,
pooled-initializer/known-worker cleanup, and controller return without a join.
Assert the five-second cleanup budget starts after the construction timeout and
is not subtracted from or substituted for it.

Extend the same worker-group tests across every operation that precedes the
initializer. Require exact order `registered cluster -> published lazy PG/ready
ref -> published topology-probe refs -> published master-port probe -> published venv PG/ready ref -> published
venv task refs -> published batch port refs -> initializer claim`. Inject a
failure, timeout, cancellation, and completion-after-cleanup at each boundary;
the same absolute deadline and nested cancellation token must be observed, no
initializer may start after prerequisite failure, and every PG/ref is removed
or cancelled once without a final `ray.get`. The disabled-policy fixture records
the original no-keyword calls and ordering byte-for-byte.

In `tests/unit/utils/test_venvs.py`, add owner-aware tests for zero eligible
nodes, placement-group creation, `pg.ready()` submission/result, each
`_env_builder` task submission/result, path decoding/normalization, unequal
paths, and success cleanup. Publish the temporary PG before `ready()`, publish
each task ref before the next submit, and remove the PG on every
`BaseException`, including an assertion/decode failure. A lone pending ready or
builder ref under a short injected deadline raises `FleetConstructionTimeout`,
cancels all refs, removes the PG, and returns without waiting for the remote uv
process; a late completion cannot escape the closed claim. Preserve the exact
legacy helper when no `FleetConstructionContext` is supplied.

In `tests/unit/distributed/test_virtual_cluster.py` and
`test_virtual_cluster_batch_ports.py`, cover partial lazy-PG allocation,
readiness submit/get failure, the fixed legacy-180-second cap under a shorter
semantic deadline, batch sizes below/equal/above 256, port-task submit/result/
decode failure, duplicate-master-port retries, cancellation during retry, and a
lone hung port task. For unified GPU placement groups, independently fail,
malform, hang, cancel, and late-complete each `_get_gpu_id_info` submit/result;
assert topology refs are published before the next bundle probe, drained under
the shared deadline, cancelled on failure, and never collected by an unbounded
`ray.get`. Assert every PG and ref is incrementally visible to its
registered cluster owner, cleanup handles late ready/port completions without a
join, and retry/backoff never exceeds `deadline.remaining_s()`. With a
10-millisecond semantic deadline, captured waits must be at most 10
  milliseconds—not 180 seconds—and partial PGs are removed. With no context,
  capture the original 180-second `ray.get`, unbounded legacy batch calls, retry
  count/order, and return shapes unchanged.

In `tests/unit/models/generation/test_generation_router.py`, fail, hang,
malform, cancel, and late-complete actor submission, base-URL, and exact
readiness refs. Assert claim -> submit -> publish ordering and one persistent
descriptor on success. Exercise token-aware underlying shutdown before and
after server start, same-token ACK loss/retry, stale-token rejection, and a
blocked daemon server thread; the server/socket closes exactly once within the
cleanup deadline and the actor survives only on the successful runtime path.
The absent-policy fixture observes no new construction or shutdown method.

In `tests/unit/environments/test_environment_utils.py` and
`tests/unit/data/test_utils.py`, build two native envs plus a validation-only
env. Fail the second env, the following dataset load, venv staging, actor
submission, constructor/readiness, and result decode; also cover a lone hang,
cancellation, and late completion. Assert the ledger exists first, every
PG/ref/outer actor is published before the next fallible step, readiness is
exact `True`, and token-aware cleanup terminates the published outer actor plus
its non-detached child workers without joining. On success retain the
identity-deduplicated union of train and validation handles/descriptors, with a
validation-only actor still reachable in `SingleControllerActorArgs`; shared
train/validation aliases shut down once. Reject a custom env lacking the v1
non-detached-child/readiness/final-wrapper capability before allocation. Use
at least two real built-in subclasses that override legacy zero-argument
`shutdown()` (for example `CodeJaccardEnvironment` and `MathEnvironment`) and
prove the inherited final
`shutdown_semantic_precision_resource(token)` calls each legacy override once,
returns the memoized exact ACK on a same-token lost-ACK retry, and rejects a
different token without another call. Also define a custom subclass that
shadows the final wrapper and prove class/capability validation rejects it
before allocation. The runtime descriptor method name must be exactly
`shutdown_semantic_precision_resource`; binding a bare `shutdown` method is
fatal. The absent policy snapshot retains the exact legacy data/env call ledger
and discarded compatibility value.

In `test_materialize.py`, `test_semantic_precision_endpoints.py`, GRPO, and
Single Controller setup tests, parameterize SGLang, Dynamo, standalone TRTLLM,
and Megatron generation with semantic precision enabled. Patch every backend
import/constructor and Ray, loop-thread, venv/PG, subprocess, router/engine,
poster, port, and worker seam to fail if touched. The pure capability preflight
must raise the documented unsupported-backend error with zero touched seams for
regular sync and async GRPO plus Single Controller. Repeat with the policy
absent and snapshot each complete legacy backend behavior byte-for-byte. A
future-adapter fixture remains rejected until it supplies all versioned
lifecycle and transaction capabilities required by the support gate; a
structural lookalike or `ray.kill`-only claim is insufficient.

In `tests/unit/environments/test_nemo_gym_utils.py`, make every remote allocator
raise until a purely local `PendingNemoGymStartup` has been ledger-registered.
Then inject submit/result/decode failure, cancellation, lone hang, and late
completion for `make_actor_runtime_env()`'s venv PG/tasks, Gym actor creation,
spinup, and tokenizer/config refs. All children must be incrementally published,
bounded by the same absolute construction deadline, and token-aborted without
joining; no failure before `start_nemo_gym_actor()` returns may escape ledger
ownership. The absent-policy case snapshots the old start/finish calls and
unbounded wait behavior byte-for-byte.

In `tests/functional/test_fleet_construction_ray.py`, run a real-Ray CPU canary
that passes the token's pending ref nested inside the dataclass to an initializer
actor. Prove submission does not auto-dereference or block, the initializer sees
the ref become ready through `ray.wait(fetch_local=False, timeout=0)`, and both
cancel-before-poll and poll-before-cancel terminate. Create a non-detached child,
hold its handle only inside a deliberately blocked initializer, kill the
initializer, and prove the child dies by Ray owner fate-sharing. This canary is
required on the pinned cluster image; fake-Ray unit tests alone cannot establish
these runtime semantics. In the same canary, block a worker constructor, submit
`semantic_precision_construction_ready.remote()`, prove its ref is not ready,
then release construction and require the exact `True` acknowledgement without
calling `__ray_ready__`.
In the same real-Ray process, pass a full
`BoundSemanticPrecisionRuntimeContext` as an actor argument and require
submission serialization to fail with the controller-local guard before the
actor method runs; separately provision the scalar adapter-authority manifest,
then pass its worker projection and require a successful strict decode against
that retained authority. Omission or a different manifest fails before intent
acceptance and no Python adapter object crosses Ray.

In `tests/functional/test_single_controller_tq_handoff_ray.py`, use real Ray
with an instrumented process-global TQ adapter. Bootstrap the controller once
inside the repo-owned authority actor, attach Policy and the standalone driver
`dp_client` (plus Value in the value-enabled case), cloudpickle the actual
ActorArgs safe state, rebind each attachment in `SingleControllerActor`, and
perform one data-plane operation there before the exact adoption ACK. Prove the
old driver attachments release only after that ACK. Kill the actor after the
controller close side effect but before its ACK, let the driver recover with
the same cleanup token, and assert the shared journal reports one controller
termination and no duplicate `tq.close()`. Repeat with Value rebind failure and
prove Policy remains usable until whole-ledger cleanup. Assert GRPO and
value/PPO attachment cardinalities, parent-owned rollout/finalizer/worker
dependency de-duplication, and controller-last shutdown. No local copied journal
may appear in either Ray process.

In `tests/unit/algorithms/test_opd.py`,
`tests/unit/models/policy/test_teacher_worker_group.py`, and
`tests/unit/models/value/test_tq_value.py`, assert every enabled teacher/value
cluster, initializer, and worker receives its corresponding registered
owner/token and the common deadline and is
published before setup-data-plane or health validation. A partially completed
multi-teacher build followed by one failure or timeout cleans all earlier and
late fleets. OPD must use the exact-`True` repo-owned readiness ACK and never
access `__ray_ready__`; missing/non-True ACKs are fatal. The absent-policy cases
snapshot the legacy calls and make every new ownership/readiness method raise if
invoked.

In `tests/unit/models/policy/test_semantic_precision_endpoints.py` and the new
`tests/unit/models/value/test_lm_value.py`, monkeypatch `RayQueue` to raise and
prove enabled built-in Policy and Value construction never instantiates the
unused queue or forwards `pre_init_communication_queue`. The absent-policy
fixtures require exactly one legacy queue construction and byte-identical worker
kwargs. Also make any replacement hidden queue actor fatal unless it uses the
explicit owner claim/publication seam.

In `tests/unit/data_plane/test_tq_lifecycle.py`,
`tests/unit/data_plane/test_tq_policy_routes.py`, and
`tests/unit/models/value/test_tq_value.py`, exercise both `TQPolicy` and
`TQValue` after their base worker fleet is ready. Assert Policy owner-publishes
the one pending `TQControllerAuthority` before `tq.init(conf=...)`, Value
receives and borrows that exact identity without another controller/bootstrap,
and each wrapper publishes its distinct `TQClientAttachmentAuthority` before
the first attach RPC. Every worker attach ref is incrementally visible and
accepts only exact readiness under the shared deadline. Inject controller
bootstrap submit/result failure, Value attach failure after Policy is live,
missing controller-versus-attachment close/status capability, a lone hung
attach, malformed result, cancellation, and completion after cleanup. A Value-
only partial failure closes its attachment/fleet but never invokes global
`tq.close()` or disturbs Policy. Whole-ledger failure and normal runtime cleanup
close both worker fleets and attachments first, then the identity-deduplicated
controller exactly once; same-token retry never double-closes it. Assert the
runtime descriptor graph contains one owned controller descriptor, one borrowed
Value alias to it, and the exact distinct attachment descriptors. A GRPO
Single Controller fixture has the Policy attachment plus the standalone
ActorArgs `dp_client`; an enabled synchronous data-plane GRPO fixture adds the
setup-owned `SyncRolloutActor` attachment; a value/PPO fixture adds the Value attachment. Add
actor/finalizer/worker-owned connections only through their identity-
deduplicated parent descriptor dependency and assert every live attachment
precedes controller close. Serialize the full `SingleControllerActorArgs` and
prove it contains only inert `TQAttachmentRebindDescriptor`s plus the shared
authority actor handle—no live driver client, process-global TQ object, local
token journal, or close callback. Patch the external
API so attachment release is indistinguishable from process-global close and
prove the enabled path fails before calling `tq.init`. The absent-policy
fixtures preserve the original bootstrap/client/worker calls and unbounded wait
byte-for-byte.
Add a Single Controller token-capture fixture using the real setup call shape.
Before the vLLM worker creates its `TQDataPlaneClient`, `TQTokenSink`, or source,
assert one worker-owned attachment is claimed/published against the existing
controller authority. Fail or hang client creation, sink/source readiness, and
`setup_token_capture()` separately; each observes the same construction
deadline/cancel token and releases the attachment without joining. Assert the
legacy version setter remains zero-call. Retry a side-effect-before-ACK loss with the same
cleanup token and observe one release; reject stale tokens. On success the
attachment is sealed beneath the worker fleet, survives actor handoff, and is
released before the shared TQ controller during worker cleanup. No fresh
`GenerationLifecycleDeadline` may appear in this setup sequence.

In `tests/unit/algorithms/test_controller_setup_teardown.py`, test the shared
resource ledger directly, naming every case with the `semantic_precision`
substring so the focused gate selects it. Register endpoints with shared and
distinct fleet tokens plus a synchronizer, inject failures during discovery,
binding, first install, second install, communicator construction, communicator
initialization, and initial refit, and assert each constructed fleet and
synchronizer is asked to shut down exactly once. Inject a short test deadline to
prove a hanging shutdown cannot exceed one shared budget, and assert the
production default is exactly five seconds; cleanup exceptions and a second
cleanup call must not replace the original exception. It must never repeat an
ACKED action, but may resume a returned/raised unknown outcome with the same
token; a concurrently hung invocation is not duplicated. At deadline assert a
detectably live resource receives one nonjoining kind-specific force-termination
dispatch, the state remains `CLEANUP_PENDING` until exact ACK or terminal
observation, and a later call revisits only those pending entries. Distinguish
callback invocation count from the sentinel's single observed termination.
Repeat the
Phase 2 failure cases through GRPO and Single Controller and assert pending Gym
is aborted and all completed
endpoints, placement groups, deferred-generation reservations, value workers,
teacher clusters/fleets, and generation routers are shut down. Parameterize a
failure immediately after each hidden allocation and before its aggregate helper
returns. The ledger transfers to a returned local
`GRPORuntimeResourceOwner` only on normal GRPO setup return; Single Controller
instead performs its explicit driver-to-actor runtime-owner transfer. A bare
`disarm()` is never a successful enabled-path terminal state.
Use barriers to race cleanup against endpoint, synchronizer, auxiliary, and Gym
registration. A future completing after CLEANING, CLEANUP_PENDING, or CLEANED must be claimed and
cleaned once without extending the five-second caller deadline; observing that
same object later through normal future collection must not clean it twice.
For late registration after `CLEANED`, assert the atomic transition to
`CLEANUP_PENDING`; inject both side-effect-before-ACK loss and a hung callback,
and prove the daemon uses the same token-aware bounded retry/force path before
restoring `CLEANED`. A prompt valid ACK must never dispatch force termination;
a hung action must receive exactly one force dispatch only after, never before,
the cleanup deadline.
Assert every registry/state access is under the ledger lock and late
registration after DISARMED fails deterministically.
Register a supported-backend deferred reservation sentinel as
`SETUP_TRANSIENT`; keep it OPEN through a blocked adoption, then exercise
successful exact-ACK completion, adoption failure, and a termination side
effect followed by lost ACK. Prove success retries the same token, records one
ACKED audit entry, closes the reservation once, and omits it from the runtime
handoff descriptors; failure cleanup still owns and closes it; transfer rejects
every transient that is not ACKED. An OPEN transient remains ledger-owned while
setup can still fail and therefore blocks transfer; only an ACKED transient is
omitted from the runtime manifest. The actor
binding manifest must contain every live persistent entry and no completed
transient, while the driver audit history retains the latter's ACK/token digest.
For every endpoint, synchronizer, Gym, and auxiliary kind, assert the ledger
passes a matching `CleanupAttemptToken` and records completion only after an
exact `ResourceTerminationAck`. Simulate a termination side effect followed by
ACK loss: the retry must reuse the same token and the underlying sentinel must
observe one termination. Reject mapping/Mock/subclass ACKs, `terminated=1`, and
different-attempt, stale-epoch, or cross-resource tokens without marking the
journal `ACKED`.

In `tests/unit/single_controller/test_setup.py`, prove the supported vLLM
initial sync remains setup-ledger-owned and a failure cleans all resources
before raising, while success transfers one exact runtime owner in
`SingleControllerActorArgs` without running it. In
`tests/unit/single_controller/test_single_controller_actor.py`, add
`test_semantic_precision_initial_sync_failure_cleans_runtime_owner_and_raises`:
make the stale synchronizer fail before pump creation, require the original
exception to escape promptly, and assert synchronizer, endpoints, router,
value/teacher fleets, and other transferred auxiliaries shut down once. Also
cover success followed by restore failure, pump failure, and a second cleanup;
the outer `finally` owns all of them and never masks the primary exception.

In `tests/unit/models/policy/test_semantic_precision_endpoints.py`, parameterize
the Megatron, DTensor v1, and DTensor v2 worker entries. Prove
the policy constructor retains the exact driver selection, actor construction
passes only its explicit selection plus scalar authority wires and decodes them
before model construction, and later intent installation decodes against the
retained authority and keeps the exact typed group. A custom bundle's Python
runtime adapter must never serialize/import in the worker, while its canonical
ID/fingerprint manifest is accepted. Prove
`RuntimeGraphSourceRequest.to_wire_dict()` is evaluated in the immediate
`run_all_workers_single_data()` argument expression, the worker reconstructs
and validates the exact frozen request, and no trusted `ExpectedContributorSet`
is serialized. Return one contribution wire from every required PP/TP/EP
worker; the driver proxy must decode every value and Task 4B's local bulk
factory must assemble it against the retained trusted set. A raw dict bypassing
the decoder, `None`, changed digest, missing nonzero PP rank, duplicate
contributor, or `results[0]` behavior is fatal.

In `tests/unit/models/generation/test_semantic_precision_endpoints.py`,
parameterize vLLM synchronous and asynchronous workers, including the vLLM
internal FlashInfer-TRTLLM runtime-layout path.
Assert every driver constructor receives the same selection object, actor
construction serializes it and the scalar adapter-authority manifest only as
explicit builder arguments, the worker decodes and retains both before
model/engine construction, every later Ray install submission
serializes only its stripped worker projection, and every worker decodes and
retains the exact typed projection with the expected context/intent identity.
Recursively assert that aggregate requests/results, trusted contributor IDs,
and adapter objects never enter the payload. Projection decode fails for an
omitted/foreign construction authority even when the projection's self-embedded
manifest is valid. Block version-specific vLLM adapter imports in a subprocess
and prove this generic Task 5 wiring still imports and runs. The absent-policy half
records complete actor-builder arguments and proves no selection, adapter-
authority, deadline argument, or serialization call was added.
Add inverse unsupported-backend tests rather than pretending Dynamo, SGLang,
standalone `generation.backend=trtllm`, or Megatron generation satisfies the common fleet protocol. With semantic
precision enabled, the pure bootstrap preflight rejects each before `ray.init`,
endpoint/model-path resolution, managed-runtime import, subprocess/thread
creation, reservation, or worker RPC. Patch every would-be constructor and
allocation seam to raise if touched and assert zero calls for regular sync and
async GRPO plus Single Controller. Their absent-policy configurations follow
their existing paths byte-for-byte. Task 5 deliberately does not claim support
until a future versioned adapter supplies the required persistent lifecycle and
transaction authority.
For Policy/generation/worker-group/value/teacher/Gym endpoint wrappers,
serialization before construction sealing is fatal. After readiness, recursively
inspect their production state and the complete `SingleControllerActorArgs`:
runtime handles/descriptors remain, but no pending owner, claim registry or
lock, latch actor/ref, cancel token, construction context, or construction
deadline is reachable. Required `IsolatedWorkerInitializer` handles are runtime
lifetime-owner descriptors, not leaked construction authority, and remain
reachable until ordered shutdown after their workers.
For vLLM—including its internal FlashInfer-TRTLLM runtime path—register the generation endpoint and its synchronizer as
distinct identities, race concurrent ledger cleanup, and prove the dependency
executor calls the synchronizer's token-aware wrapper once before endpoint
shutdown while the enabled endpoint never transitively calls it. Same-token ACK
loss retries observe one underlying legacy `shutdown()` side effect. In the
absent-policy fixture, call the existing zero-argument generation `shutdown()`
and prove it still invokes the synchronizer's zero-argument `shutdown()` exactly
as before; no token parameter or wrapper is consulted.
In `tests/unit/algorithms/test_grpo.py`, exercise the successful enabled setup
return and both ordinary-training and refit exceptions through the real launcher
ownership seam. The setup result contains one non-serializable
`GRPORuntimeResourceOwner`; from setup return through launcher `finally` it is
the sole cleanup owner, and bounded close invokes the synchronizer token-aware
shutdown exactly once before its dependent vLLM generation endpoint,
then every other persistent ledger entry. A primary training/refit exception is
re-raised after no-throw bounded cleanup. Normal completion closes the same set
once. The absent-policy setup result carries no runtime owner and
`examples/run_grpo.py` retains its existing environment plus zero-argument
`policy_generation.shutdown()` sequence exactly. Start the enabled setup session
before `setup_response_data()` and fail the first/second train or validation env
submit/readiness, venv creation, dataset construction, and the later
`grpo.setup()` separately; every published train/validation env union member and
partial venv resource is cleaned although no setup result was returned. Migrate
all four in-repository GRPO launchers to named `GRPOSetupResult` access and an
outer owner `finally`; a source/AST contract in
`tests/unit/test_config_validation.py` rejects legacy 13-tuple destructuring or
any callsite that can drop a non-`None` owner. Disabled fixtures snapshot the
  original env/data/setup calls and shutdown ordering for every launcher.
In `tests/unit/experience/test_sync_rollout_actor.py` and the synchronous
cases in `tests/unit/algorithms/test_grpo.py`, pass the train environment map
and exact data-plane configuration into enabled `grpo.setup()` and assert the
actor is built after its pending owner is registered but before the ledger is
frozen/transferred. Fail runtime-env/venv submission, actor submission,
constructor, exact-`True` readiness, TQ attachment, and completion racing
cleanup separately; a lone hang expires the shared construction deadline and
late completion is adopted without joining. On success, assert one actor and
one identity-deduplicated attachment appear in the frozen
`GRPORuntimeResourceOwner`, `grpo_sync.py` performs zero `.remote()` calls and
uses the returned handle, and shutdown order is actor attachment/actor before
the shared TQ controller. Inject a TQ close side effect followed by ACK loss and
prove same-token recovery observes one close; a real close failure is recorded
and propagated to the bounded runtime cleanup result instead of being swallowed.
The absent-policy fixture keeps the current in-function actor creation,
zero-argument `shutdown()`, and exception behavior byte-for-byte.
Add explicit dependency-wave barriers for synchronizer-before-endpoint,
worker-before-initializer, every TQ attachment-before-controller, and
control-cell-last. Independent eligible roots may run concurrently, but no
dependent callback starts before every predecessor has an exact ACK or a
kind-specific verified terminal observation. If a predecessor hangs, prove its
force termination is dispatched only at the shared cleanup deadline; a
dependent is then force/gracefully dispatched only after that predecessor is
observed terminal. If terminal observation is delayed, the bounded caller
returns with `CLEANUP_PENDING` while one nonjoining cascade retains ownership;
it never violates ordering to make the deadline. Reject cycles and unknown
predecessor IDs before cleanup/transfer. Register more than
`MAX_CONTROLLER_CLEANUP_CONCURRENCY` independent entries and assert the live
daemon-worker count never exceeds the cap while every entry is eventually
scheduled. Inject scheduler submit/thread-start failure in an eligible wave and
prove the entry remains owned, receives the ordered force/cascade fallback, and
does not block unrelated eligible roots.

In `tests/unit/weight_sync/test_semantic_precision_handshake.py`, parameterize
direct collective, synchronizer collective, NCCL reshard, checkpoint engine,
colocated IPC and both remote-sparse transports targeting a
supported vLLM destination. With
non-`None` context, assert construction and initialization receive the same
full controller-local object by identity while remote workers receive only its
validated stripped projection. Every existing transport-specific `prepare_refit_info()`, baseline,
communicator-preparation, and initial-sync call still occurs in its original
relative order. For remote sparse, assert no `start_baseline()` occurs inside
the policy builder: the order is both endpoint completions, Phase 2, pending
synchronizer construction, ledger registration, `start_baseline`, communicator
initialization, baseline-result validation, and initial sync. Fail each baseline
submission/result and prove ledger cleanup. With no semantic policy, assert
exact legacy factory/constructor kwargs and handshake calls byte-for-byte. The
direct collective case must construct its side-effect-free enabled-path handle,
register it, then call `init_communicator()` and retain the exact legacy wire
setup; it may not use intent or handle presence as readiness.
Serialize every supported vLLM transport synchronizer through its production `__getstate__`
and recursively inspect the complete `SingleControllerActorArgs` payload. It
must contain the safe projection, `runtime_context_id`, and (after Task 7) bound
plan identity plus the independently provisioned scalar adapter-authority
manifest required by actor-side decode, but no full context, aggregate
request/results, trusted contributor IDs/evidence, source locator, or adapter
object. Here `source locator` means request/result-only mutable source, artifact,
or allocation locators; canonical content-addressed evidence locators inside
the intent projection are retained. Directly inserting the full context into any ActorArgs field must trip
its pickle guard instead of silently crossing Ray.
Serialization while communicator initialization is pending must fail closed.
After success, recursively prove the ActorArgs payload contains no
`FleetConstructionContext`, pending owner/ledger, cancellation token/latch
handle/ref, construction deadline, or cleanup callback, while retaining the
runtime-safe projection, scalar adapter authority, and (after Task 11)
authenticated transaction authority.
Assert the driver and deserialized actor synchronizer wrappers reference the
identical `SynchronizerCleanupAuthorityActor` and neither serialized state
contains a local cleanup journal. Let the actor-side wrapper trigger underlying
shutdown, kill the Single Controller caller after the authority records the
side effect but before its ACK reaches that caller, then have driver recovery
retry the same token. The shared authority returns the recorded ACK and the
underlying synchronizer terminates once; a different/stale token fails. After
final status is durable, launcher cleanup terminates the authority actor last.
For every parameterized transport, leave each communicator-construction ref in
turn as the sole pending ref, advance the shared construction deadline, and
require `FleetConstructionTimeout` plus bounded token-aware cleanup without a
join. Independently cover submit failure, raised/malformed result, cancellation,
and completion after cleanup for collective source/destination init, Megatron
rendezvous/group RPCs, NCCL-reshard unique-ID/group RPCs, checkpoint-engine,
IPC, both remote-sparse baseline/init paths, and the direct collective's
master-IP/port and endpoint-init refs. Assert every ref/actor/group is claimed
and published before the next submission. With no policy, record exact legacy
arguments and waits and prove no construction context or timeout branch is
consulted.

In `tests/functional/test_semantic_precision_initial_sync.py`, reuse each
transport's production endpoint fixture and execute an actual first weight sync
with a uniquely valued tiny source tensor. Parameterize colocated IPC, direct
collective, synchronizer collective, NCCL reshard, checkpoint engine,
`vllm_s3_sparse`, and `vllm_zmq_sparse`, each terminating in a
supported vLLM destination, with both native-vLLM and internal
FlashInfer-TRTLLM runtime-layout fixtures. Create
`tests/functional/conftest.py` with an exact session fixture reading
`NEMO_RL_SEMANTIC_PRECISION_TEST_BACKEND=vllm` plus a canonical vLLM runtime-mode selector and
a canonical backend-to-transport set. The test is explicitly parameterized over
all transports and skips only cases not supplied by the selected pinned image;
an absent/unknown environment value fails when this test file is explicitly
invoked. The three cluster invocations together must report one PASS, not SKIP,
for every parameter. Read the destination through its public inference/check API and assert
the sentinel value and committed version, then perturb the source and prove a
second sync still works. This is the Task 5 compatibility gate that prevents
semantic metadata wiring from disabling the operational path before Tasks
7-12 replace it. It is a success-path and synchronously-raised-error gate, not
evidence for silent-peer detection, exact transaction acknowledgements, or
all-transport fail-fast; those claims are forbidden until Tasks 11-12 pass.

- [ ] **Step 3: Write failing `explain-precision` tests**

```python
def test_explain_precision_reports_bf16_boundaries_and_mxfp8_middle(
    tmp_path: Path,
) -> None:
    completed = run_config_cli(
        "explain-precision", fixture_recipe(tmp_path), "--format", "json"
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["scopes"][0]["selected_global_decoder_layers"] == [2, 3, 4]
    assert payload["summary"]["rollout"]["mxfp8"] == 3 * 8 * 3
    assert payload["runtime_source_digest"] == "unavailable until runtime binding"
    assert payload["intent_group_id"] == "unavailable until runtime binding"
    assert payload["plan_id"] == "unavailable until destination binding"
```

Add CLI subprocess cases asserting nonzero exit and an actionable message for
zero matches, an unsupported or ambiguous paired adapter, conflicting scopes,
invalid layer universes, unresolved revisions, an omitted main/MTP/draft graph,
and invalid immutable auxiliary evidence. Spy on all producer and endpoint
factories: the CLI may instantiate `SemanticPrecisionBootstrap` and run Phase 1
only; it must not allocate Ray/GPU resources, invoke source discovery, or claim
runtime completeness.

- [ ] **Step 4: Run the focused tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/algorithms/test_controller_setup_teardown.py tests/unit/algorithms/test_opd.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/data_plane/test_architecture_invariants.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py tests/unit/models/generation/test_semantic_precision_endpoints.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py tests/unit/tools/test_config_cli.py -k 'precision or semantic_precision'`

Run the launcher ownership/pre-resource guards unfiltered as RED: `uv run --no-sync pytest -q tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/test_config_validation.py`

Run the new lifecycle suites unfiltered so hidden-allocation, communicator, and
ownership cases whose names describe the failing phase are not deselected:
`uv run --no-sync pytest -q tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_opd.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py`

Run the NeMo Gym construction suite unfiltered as part of RED:
`uv run --no-sync pytest -q tests/unit/environments/test_nemo_gym_utils.py`

Run the finalizer construction suite unfiltered as part of RED:
`uv run --no-sync pytest -q tests/unit/experience/test_rollout_reassembler_actor.py`

Run the synchronous rollout-actor/TQ ownership suite unfiltered as part of
RED: `uv run --no-sync pytest -q tests/unit/experience/test_sync_rollout_actor.py tests/unit/algorithms/test_grpo.py`

Run the generation-router lifecycle suite unfiltered as part of
RED: `uv run --no-sync pytest -q tests/unit/models/generation/test_generation_router.py`

Run the TQ bootstrap/attach suites unfiltered as part of RED:
`uv run --no-sync pytest -q tests/unit/data_plane/test_tq_lifecycle.py tests/unit/data_plane/test_tq_policy_routes.py tests/unit/models/value/test_tq_value.py`

Run the real-Ray TQ handoff canary as RED:
`uv run --no-sync pytest -q tests/functional/test_single_controller_tq_handoff_ray.py`

Run the native environment ownership suites unfiltered as part of RED:
`uv run --no-sync pytest -q tests/unit/environments/test_environment_utils.py tests/unit/data/test_utils.py`

Run the dependency-direction/import-smoke suite unfiltered as part of RED:
`uv run --no-sync pytest -q tests/unit/data_plane/test_architecture_invariants.py`

Expected: imports fail for `SemanticPrecisionBootstrap`, the paired bundle,
`BoundSemanticPrecisionRuntimeContext`, its stripped worker projection, and the
fleet owner/nested-ref cancellation latch; endpoint/worker methods, explicit
adapter injection, pre-initializer owned venv/PG/port seams, and the
setup/runtime handoff owner are absent. Existing tests still reach
`_default_adapters`, discard the Phase 2 request/results/authority after creating
bare intents, pool worker/port results before publishing them, join hung startup
work, or show endpoint, communicator, and Single Controller construction/run
without the required Phase 1/Phase 2, exact adoption ACK, and ownership ordering;
PPO/distillation continue past the required pre-resource fail-closed seam.

- [ ] **Step 5: Implement paired adapter registration and the controller-owned bootstrap**

Start `nemo_rl/distributed/fleet_construction.py` with the dependency-neutral
frozen lifecycle values and token-journal protocol: `CleanupAttemptToken`,
`ResourceTerminationAck`, exact `CleanupTokenState`, `CleanupTokenJournal`, and
`TokenAwareShutdown`. This layer may depend on standard-library typing and Ray
construction primitives but never on `nemo_rl.algorithms`, endpoints, or
controller orchestration. Every low-level resource imports this single
definition; do not duplicate the dataclasses in an endpoint/interface module.
Implement `ControllerSetupResourceLedger` by importing these values upward,
while keeping `RuntimeResourceDescriptor` and transfer policy controller-local.

```python
@dataclass(frozen=True, slots=True)
class PrecisionTopologyAdapterBundle:
    adapter_id: str
    selection: SelectionTopologyAdapter
    runtime: ModelTopologyAdapter

    def __post_init__(self) -> None:
        if type(self.adapter_id) is not str or not self.adapter_id.strip():
            raise ValueError("adapter bundle ID must be canonical non-empty text")
        if self.selection is None or self.runtime is None:
            raise ValueError("adapter bundle requires selection and runtime halves")
        if self.selection.adapter_id != self.adapter_id:
            raise ValueError("selection adapter ID differs from bundle ID")
        if self.runtime.adapter_id != self.adapter_id:
            raise ValueError("runtime adapter ID differs from bundle ID")


@dataclass(frozen=True, slots=True)
class SemanticPrecisionFactoryCapability:
    schema_version: Literal[1]
    selection_keyword: Literal["precision_selection"]
    adapter_authority_keyword: Literal["precision_adapter_authority"]
    construction_deadline_keyword: Literal["fleet_construction_deadline"]
    construction_protocol: Literal["two_phase_owner_v1"]
    worker_readiness_method: Literal["semantic_precision_construction_ready"]


SEMANTIC_PRECISION_FACTORY_CAPABILITY_V1 = SemanticPrecisionFactoryCapability(
    schema_version=1,
    selection_keyword="precision_selection",
    adapter_authority_keyword="precision_adapter_authority",
    construction_deadline_keyword="fleet_construction_deadline",
    construction_protocol="two_phase_owner_v1",
    worker_readiness_method="semantic_precision_construction_ready",
)


def require_semantic_precision_factory_v1(factory: object) -> None:
    capability = getattr(factory, "semantic_precision_factory_capability", None)
    if type(capability) is not SemanticPrecisionFactoryCapability:
        raise TypeError("semantic precision factory capability must have the exact runtime type")
    if (
        type(capability.schema_version) is not int
        or capability.schema_version != 1
        or type(capability.selection_keyword) is not str
        or capability.selection_keyword != "precision_selection"
        or type(capability.adapter_authority_keyword) is not str
        or capability.adapter_authority_keyword != "precision_adapter_authority"
        or type(capability.construction_deadline_keyword) is not str
        or capability.construction_deadline_keyword != "fleet_construction_deadline"
        or type(capability.construction_protocol) is not str
        or capability.construction_protocol != "two_phase_owner_v1"
        or type(capability.worker_readiness_method) is not str
        or capability.worker_readiness_method != "semantic_precision_construction_ready"
    ):
        raise ValueError("semantic precision factory does not implement capability v1")


T_co = TypeVar("T_co", covariant=True)


class PendingSemanticPrecisionEndpointBuild(Protocol[T_co]):
    @property
    def semantic_precision_fleet_identity(self) -> object: ...

    def shutdown_semantic_precision_fleet(
        self, cleanup_attempt_token: CleanupAttemptToken
    ) -> ResourceTerminationAck: ...

    def finish(self) -> T_co: ...
```

Define `DEFAULT_FLEET_CONSTRUCTION_TIMEOUT_S = 1800.0`, strict
`normalize_fleet_construction_timeout_s()`, immutable
`FleetConstructionDeadline`, and typed `FleetConstructionTimeout` in
`distributed/fleet_construction.py`. The deadline is constructed once by the
controller, not restarted per worker, and every diagnostic reports elapsed,
configured total, phase, fleet identity, and still-open claims without exposing
Ray-private IDs.

Define frozen `FleetConstructionContext(owner, cancel_token, deadline)` and add
keyword-only `construction: FleetConstructionContext | None` seams to the venv
and virtual-cluster helpers. `None` takes an immediate exact legacy branch;
enabled callers must provide all three values as one context and cannot mix
owners or deadlines. In `virtual_cluster.py`, register the cluster aggregate
before lazy PG creation, claim/publish each PG plus its readiness ref, and use
the smaller of the legacy 180-second cap and the shared remaining deadline.
Master and batch port probes use the same claim/publication/drain helper and
deadline-aware retry backoff. `_get_sorted_bundle_indices()` threads the context
into GPU-ID topology discovery, claims and publishes each `_get_gpu_id_info`
ref, and drains/decodes incrementally instead of calling unbounded
`ray.get(info_refs)`. In `venvs.py`, publish the temporary strict-spread
PG before readiness, every `_env_builder` ref before submitting the next, and
remove/cancel all children in one `BaseException`-safe `finally`. None of these
enabled paths may call unbounded `ray.get`; late callbacks close their original
claim rather than allocating a replacement resource.

Validate pending-build and completed-endpoint protocols with explicit
`isinstance(..., @runtime_checkable Protocol)` checks plus exact stable-token
  identity; the capability record itself deliberately uses the stricter exact-type
  validator above. Every built-in begin method creates its owner first and wraps
  all purely local validation in a `BaseException` guard that invokes the
  owner's token-aware shutdown with one stable attempt token, validates the exact
  termination ACK, and re-raises; tests enforce that begin cannot call a remote
  allocator.
`finish()` may allocate only through `PendingSemanticPrecisionFleetOwner.claim()`
followed by its `submit()`/`publish()`/`fail()` boundary. Its first claimed
allocation lazily creates and publishes the async cancellation latch, obtains
one pending wait ref, and constructs the nested-ref cancel token before any
other claim. Its condition-protected states are exact
`OPEN/FINISHING/SEALING/SEALED/CANCELLING/CLOSED`; the normal transitions are
`OPEN -> FINISHING -> SEALING -> SEALED`, failure/cancellation may move any of
`OPEN/FINISHING/SEALING` to `CANCELLING -> CLOSED`, and no transition leaves
`SEALED` except the sealed runtime authority's separately journaled shutdown.
The `SEALING -> SEALED` compare-and-set succeeds only if cleanup did not first
move `SEALING -> CANCELLING`. Multiple in-flight entries retain unique
claim IDs, resource kind, eventual identity, shutdown action, and a claimed bit.
Cancellation checks precede each claim and the initializer's zero-time latch
poll precedes each child submit, while late driver publication is the final
race-closing check. No backend constructor may call a Ray actor,
router, placement group, engine, value fleet, or teacher-fleet
allocation outside that boundary on the enabled path.

Replace `BUILTIN_TOPOLOGY_ADAPTERS` as the production authority with one
deterministically ordered `BUILTIN_PRECISION_TOPOLOGY_ADAPTER_BUNDLES`. Keep any
legacy flat adapter constant private to compatibility tests; neither resolver
may default to it in the controller path. Extend Phase 2 classification and
`bind_runtime_source_intents()` with a required exact runtime-adapter mapping.
Before classification, derive the adapter IDs from
`selection.topology.graphs`, require one runtime half per ID, require its ID and
`supports(retained_phase1_request.effective_model_config)` result to agree with
Phase 1 after request-digest revalidation, and reject
missing, extra, or ambiguous entries. `SemanticPrecisionBootstrap` owns the
boundary from the full validated bundle registry to this exact mapping: it
selects only the unique adapter IDs frozen in the retained graphs, orders them
by canonical ID, and never forwards unused production bundles. The binder still
rejects a superset supplied by any other caller. Remove the defaults from
`resolve_selection_topology()` and the runtime classification entrypoints; an
outer legacy compatibility function must pass its registry explicitly. Extend
`validate_compiled_precision_intent_group()` and the raw runtime binder with the
required mapping. The six Task 5 refit-context bind/install/selected-operation
entrypoints accept only the full `BoundSemanticPrecisionRuntimeContext`, extract
its `context.runtime_adapters_by_id` internally, and pass that exact mapping to
the validator/binder; their signatures reject a separately supplied mapping.
No validation path may rediscover adapters, call
`_default_adapters()`, or accept a test monkey-patch as authority.
Each selected runtime adapter exposes one canonical immutable implementation
fingerprint at bundle registration. The controller validates it while building
`RuntimeAdapterAuthority`; a custom bundle supplies the same scalar contract.
Only the selected ID/fingerprint manifest is sent to endpoint workers at
construction. No custom adapter instance, import path, callable, or registry is
serialized, and worker projection decoding performs no family selection.

Implement the bootstrap without writing to `policy_config` or its nested
generation mapping:

```python
@dataclass(frozen=True, slots=True)
class RuntimeAdapterAuthority:
    adapter_ids: tuple[str, ...]
    implementation_fingerprints: tuple[str, ...]
    authority_digest: str = field(init=False)

    def to_wire_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_wire_dict(
        cls, payload: Mapping[str, object]
    ) -> "RuntimeAdapterAuthority": ...


@dataclass(frozen=True, slots=True)
class BoundSemanticPrecisionRuntimeContext:
    selection: CompiledPrecisionSelectionGroup
    source_request: RuntimeSourceDiscoveryRequest
    source_results: tuple[RuntimeSourceDiscoveryResult, ...]
    adapter_authority: RuntimeAdapterAuthority
    runtime_adapters_by_id: Mapping[str, ModelTopologyAdapter]
    intents: CompiledPrecisionIntentGroup
    runtime_context_id: str = field(init=False)

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("bound semantic precision runtime context is controller-local")

    def to_worker_projection(
        self,
        *,
        binding_phase: Literal["phase2_bound", "plan_bound"],
        bound_plan_group_ids: tuple[str, ...],
    ) -> BoundSemanticPrecisionWorkerProjection: ...


@dataclass(frozen=True, slots=True)
class BoundSemanticPrecisionWorkerProjection:
    runtime_context_id: str
    selection_group_id: str
    intent_group: CompiledPrecisionIntentGroup
    binding_phase: Literal["phase2_bound", "plan_bound"]
    bound_plan_group_ids: tuple[str, ...]
    adapter_authority: RuntimeAdapterAuthority
    projection_digest: str = field(init=False)

    def to_wire_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_wire_dict(
        cls,
        payload: Mapping[str, object],
        *,
        expected_adapter_authority: RuntimeAdapterAuthority,
    ) -> "BoundSemanticPrecisionWorkerProjection": ...


class SemanticPrecisionBootstrap:
    def __init__(
        self,
        *,
        adapter_bundles: tuple[PrecisionTopologyAdapterBundle, ...] = (
            BUILTIN_PRECISION_TOPOLOGY_ADAPTER_BUNDLES
        ),
    ) -> None:
        self._bundles = validate_precision_adapter_bundles(adapter_bundles)
        self._selection: CompiledPrecisionSelectionGroup | None = None
        self._context: BoundSemanticPrecisionRuntimeContext | None = None
        self._phase1_requests_by_graph: Mapping[str, GraphTopologyResolutionRequest] = MappingProxyType({})
        self._runtime_adapters_by_id: Mapping[str, ModelTopologyAdapter] = MappingProxyType({})
        self._worker_adapter_authority: RuntimeAdapterAuthority | None = None
        self._phase1_input_digest: str | None = None
        self._saw_absent_policy = False

    @property
    def selection(self) -> CompiledPrecisionSelectionGroup | None:
        return self._selection

    @property
    def context(self) -> BoundSemanticPrecisionRuntimeContext | None:
        return self._context

    @property
    def phase1_requests_by_graph(
        self,
    ) -> Mapping[str, GraphTopologyResolutionRequest]:
        return self._phase1_requests_by_graph

    @property
    def runtime_adapters_by_id(self) -> Mapping[str, ModelTopologyAdapter]:
        return self._runtime_adapters_by_id

    @property
    def worker_adapter_authority(self) -> RuntimeAdapterAuthority:
        if self._worker_adapter_authority is None:
            raise RuntimeError("adapter authority is unavailable before enabled Phase 1")
        return self._worker_adapter_authority

    def materialize(
        self, policy_config: PolicyConfig
    ) -> CompiledPrecisionSelectionGroup | None:
        policy = parse_precision_policy(policy_config.get("precision_policy"))
        if policy is None:
            if self._selection is not None:
                raise ValueError("materialized Phase 1 input changed to absent policy")
            self._saw_absent_policy = True
            return None
        if self._saw_absent_policy:
            raise ValueError("materialized Phase 1 input changed from absent policy")
        requests = build_graph_topology_resolution_requests(policy_config)
        topology = resolve_selection_topology(
            requests,
            policy.schema_version,
            adapters=tuple(bundle.selection for bundle in self._bundles),
        )
        candidate = compile_precision_selection(policy, topology)
        requests_by_graph = freeze_phase1_requests_by_graph(requests)
        input_digest = phase1_request_set_digest(requests_by_graph)
        if self._selection is None:
            validate_phase1_request_retention(candidate, requests_by_graph, input_digest)
            bundles_by_id = {bundle.adapter_id: bundle for bundle in self._bundles}
            selected_adapter_ids = tuple(
                sorted({graph.adapter_id for graph in candidate.topology.graphs})
            )
            self._runtime_adapters_by_id = MappingProxyType(
                {
                    adapter_id: bundles_by_id[adapter_id].runtime
                    for adapter_id in selected_adapter_ids
                }
            )
            self._worker_adapter_authority = build_runtime_adapter_authority(
                self._runtime_adapters_by_id
            )
            self._phase1_requests_by_graph = requests_by_graph
            self._phase1_input_digest = input_digest
            self._selection = candidate
            return candidate
        if (
            input_digest != self._phase1_input_digest
            or candidate.to_wire_dict() != self._selection.to_wire_dict()
        ):
            raise ValueError("materialized Phase 1 input changed")
        return self._selection

    def bind_runtime_sources(
        self,
        request: RuntimeSourceDiscoveryRequest,
        results: tuple[RuntimeSourceDiscoveryResult, ...],
    ) -> BoundSemanticPrecisionRuntimeContext | None:
        if self._selection is None:
            if self._saw_absent_policy:
                return None
            raise RuntimeError("Phase 1 must materialize before Phase 2")
        if self._context is not None:
            raise RuntimeError("runtime sources may bind exactly once")
        assert self._phase1_input_digest is not None
        validate_phase1_request_retention(
            self._selection,
            self._phase1_requests_by_graph,
            self._phase1_input_digest,
        )
        candidate_intents = bind_runtime_source_intents(
            self._selection,
            request,
            results,
            runtime_adapters_by_id=self._runtime_adapters_by_id,
            phase1_requests_by_graph=self._phase1_requests_by_graph,
        )
        candidate = build_bound_semantic_precision_runtime_context(
            selection=self._selection,
            source_request=request,
            source_results=results,
            runtime_adapters_by_id=self._runtime_adapters_by_id,
            adapter_authority=self.worker_adapter_authority,
            intents=candidate_intents,
        )
        validate_bound_semantic_precision_runtime_context(candidate)
        validate_worker_projection_round_trip(
            candidate.to_worker_projection(
                binding_phase="phase2_bound", bound_plan_group_ids=()
            ),
            expected_adapter_authority=self.worker_adapter_authority,
        )
        self._context = candidate
        return candidate
```

`freeze_phase1_requests_by_graph()` accepts only the exact, canonically sorted
tuple returned by the resolver builder, requires one unique key per declaration,
and returns `MappingProxyType` over the already recursively frozen request
records. `phase1_request_set_digest()` hashes ordered graph ID, request wire
fields, and `effective_model_config_digest`; `validate_phase1_request_retention()`
recomputes those values and proves exact graph key, declaration, resolved
revision, universe, effective-config digest, and selection-topology agreement.
The implementation revalidates the retained selection immediately before Phase 2 and publishes
`_context` only after the complete binder and stripped-projection validation
succeed; an exception leaves the bootstrap in `MATERIALIZED`, never partially
`READY`. All properties are
read-only. The complete request builder performs no source discovery and never
loads tensor payloads.
`validate_worker_projection_round_trip()` requires the exact independently
provisioned `expected_adapter_authority` scalar manifest as a keyword-only
argument and has no default, global registry, Python-adapter lookup, or
context-reconstruction branch. Full controller replay continues to require the
exact process-local `runtime_adapters_by_id` mapping; projection decoding never
does.

Add canonical `to_wire_dict()`/`from_wire_dict()` pairs for the per-graph
runtime request, per-worker `DiscoveryContribution` and its nested source
records/storage realization inventory, discovery contribution, selection,
intent, and stripped worker-projection values that actually cross Ray. A decoder reconstructs exact frozen
domain types through their public constructors, recomputes every derived digest,
rejects unknown or missing fields, then requires `decoded.to_wire_dict()` to
equal the detached input. Aggregate requests, combined discovery results, the
full bound context, and their trusted contributor sets remain controller-local
and have no public wire encoding. Direct in-process calls pass typed objects and
must not serialize them speculatively.

- [ ] **Step 6: Wire both controllers at the two lifecycle seams**

Implement `ControllerSetupResourceLedger` with an internal list of endpoint
representatives deduplicated by opaque fleet-token identity, an identity-
deduplicated synchronizer list, and one current Gym cleanup callback. Its
failure method is a no-throw boundary and uses one deadline for all participants:

```python
def _print_cleanup_error_safely(error: BaseException) -> None:
    try:
        print(f"controller setup cleanup failed: {error!r}", flush=True)
    except BaseException:
        pass


MAX_CONTROLLER_CLEANUP_CONCURRENCY = 16


def _run_dependency_ordered_no_throw_actions_until(
    entries: Sequence[CleanupEntry], *, deadline: float
) -> None:
    def invoke(entry: CleanupEntry) -> None:
        while time.monotonic() < deadline and not entry.cleanup_is_acked():
            claim = entry.claim_cleanup_invocation()
            if claim is None:  # another invocation is still running
                return
            token = claim.token
            try:
                ack = entry.action(token)
                require_exact_resource_termination_ack(ack, expected=token)
                entry.record_cleanup_ack(token, ack)
            except BaseException as error:
                entry.record_cleanup_invocation_returned(token)
                _print_cleanup_error_safely(error)
                entry.wait_retry_backoff(deadline=deadline)

    require_acyclic_known_cleanup_dependencies(entries)
    scheduler = DaemonNonJoiningCleanupScheduler(
        max_workers=min(MAX_CONTROLLER_CLEANUP_CONCURRENCY, max(1, len(entries)))
    )
    active: dict[str, CleanupInvocationHandle] = {}
    while time.monotonic() < deadline:
        available_slots = scheduler.max_workers - len(active)
        eligible = tuple(
            entry
            for entry in entries
            if not entry.cleanup_is_started()
            and all(predecessor.is_cleanup_terminal() for predecessor in entry.predecessors)
        )[:available_slots]
        for entry in eligible:  # bounded independent roots in this wave
            entry.record_cleanup_started()
            try:
                active[entry.resource_id] = scheduler.submit(invoke, entry)
            except BaseException as error:
                entry.record_cleanup_thread_start_failed()
                _print_cleanup_error_safely(error)
        if all(entry.is_cleanup_terminal() for entry in entries):
            break
        if not active:
            break
        wait_for_any_cleanup_invocation_no_later_than(active, deadline=deadline)
        active = {
            resource_id: handle
            for resource_id, handle in active.items()
            if not handle.done()
        }

    # The graceful budget is exhausted. Force only currently eligible entries,
    # in topological waves; never force a dependent before its predecessors are
    # ACKED or have a kind-specific verified terminal observation.
    dispatch_eligible_force_waves_no_wait(entries)
    if any(not entry.is_cleanup_terminal() for entry in entries):
        start_owned_nonjoining_dependency_cascade(entries)
    scheduler.abandon_no_join()


class ControllerSetupResourceLedger:
    def __init__(self, timeout_s: float = 5.0) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("setup teardown timeout must be numeric")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("setup teardown timeout must be finite and positive")
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._state = SetupLedgerState.OPEN
        self._entries: list[CleanupEntry] = []
        self._gym_entry: CleanupEntry | None = None

    def _register(
        self,
        identity: object,
        action: TokenAwareShutdown,
        descriptor: RuntimeResourceDescriptor,
        lifetime: SetupResourceLifetime = SetupResourceLifetime.PERSISTENT,
    ) -> None:
        run_late = False
        with self._lock:
            if self._state is SetupLedgerState.DISARMED:
                raise RuntimeError("cannot register after setup ownership transfer")
            existing = next(
                (entry for entry in self._entries if entry.identity is identity),
                None,
            )
            if existing is not None:
                existing.validate_and_merge_alias_registration(
                    action=action,
                    descriptor=descriptor,
                    lifetime=lifetime,
                )
                return
            entry = CleanupEntry(
                identity=identity,
                action=action,
                descriptor=descriptor,
                lifetime=lifetime,
                force_terminate=resolve_force_termination_adapter(
                    descriptor, identity
                ),
            )
            self._entries.append(entry)
            if self._state in (
                SetupLedgerState.CLEANING,
                SetupLedgerState.CLEANUP_PENDING,
                SetupLedgerState.CLEANED,
            ):
                if self._state is SetupLedgerState.CLEANED:
                    self._state = SetupLedgerState.CLEANUP_PENDING
                entry.claimed = True
                run_late = True
        if run_late:
            self._start_late_cleanup_no_wait(entry)

    def register_endpoint(
        self,
        endpoint: SemanticPrecisionIntentEndpoint,
        *,
        descriptor: RuntimeResourceDescriptor,
    ) -> None:
        self._register(
            endpoint.semantic_precision_fleet_identity,
            endpoint.shutdown_semantic_precision_fleet,
            descriptor,
        )

    def cleanup_after_failure(self) -> None:
        with self._lock:
            if self._state in (SetupLedgerState.CLEANING, SetupLedgerState.CLEANED, SetupLedgerState.DISARMED):
                return
            self._state = SetupLedgerState.CLEANING
            deadline = time.monotonic() + self._timeout_s
            entries = tuple(
                entry for entry in self._entries if not entry.cleanup_is_acked()
            )
            for entry in entries:
                entry.claimed = True
        _run_dependency_ordered_no_throw_actions_until(
            entries, deadline=deadline
        )
        with self._lock:
            if self._state is SetupLedgerState.CLEANING:
                self._state = (
                    SetupLedgerState.CLEANED
                    if all(entry.is_cleanup_terminal() for entry in self._entries)
                    else SetupLedgerState.CLEANUP_PENDING
                )
```

The excerpt above is only the cleanup core. The implementation also provides
the repo-owned fixed-cap daemon/nonjoining cleanup scheduler shown above; it
creates at most `MAX_CONTROLLER_CLEANUP_CONCURRENCY` threads regardless of
fleet/finalizer count, schedules entries incrementally only when their
predecessors are terminal, and is never used as a joining context manager. A
submit/thread-start failure leaves the entry unstarted and owned, records the
diagnostic, and routes it through the same dependency-eligible force/cascade
path rather than dropping it. The cap is a repository constant in v1, not a
per-job knob that can accidentally create one thread per worker.
The implementation also provides
the exact `claim_future()` / `publish_future()` / `fail_future()` transition and
`submit_owned_setup_future()`: it claims first, calls the executor's `submit`,
attaches the adoption callback as the immediately following operation, then
publishes the future. An already-completed future invokes the callback during
`add_done_callback`; both callback and normal result collection use the same
claim identity. A failed submit calls `fail_future()` before propagating.

`SetupLedgerState` is the exact enum
`OPEN/CLEANING/CLEANUP_PENDING/CLEANED/DISARMED` and
`CleanupEntry` retains the identity object, exact runtime-resource descriptor,
stable cleanup resource ID, exact `SetupResourceLifetime`,
token-aware action, repo-owned kind-specific nonjoining force-termination
adapter, dispatch flag, and exact
`UNCLAIMED/IN_FLIGHT/ACKED/FORCE_DISPATCHED` attempt journal under the ledger
lock. `begin_or_resume_cleanup_attempt()`
creates one stable token or returns the same in-flight token after an unknown
outcome; `claim_cleanup_invocation()` additionally returns `None` while an
earlier invocation is still running, so neither a later ledger call nor a retry
thread can invoke the same entry concurrently. A returned/raised invocation
clears only that running flag, never its token. `record_cleanup_ack()` accepts only the exact built-in ACK matching all
token fields. A returned/raised call with an unknown outcome is retried
sequentially with that same token and bounded backoff while the shared deadline
remains; a still-running/hung call is never duplicated. At deadline the ledger
dispatches the descriptor's repo-owned `force_terminate_no_wait()` once for any
detectably live actor, placement group, task ref, socket/reservation, or local
helper without joining it or masking the primary error. This records
`FORCE_DISPATCHED`, distinct from an observed exact ACK. A later
`cleanup_after_failure()` call returns immediately only for `CLEANED`/`DISARMED`;
from `CLEANUP_PENDING` it revisits only non-ACKED entries, resumes the same token
after a prior invocation has returned, and never re-invokes or re-terminates an
ACKED entry. It may re-check but not duplicate an already dispatched force
termination. `CLEANED` means every entry has an exact ACK or a verified
kind-specific terminal observation after force dispatch, not merely that the
first caller's five-second deadline elapsed.
`_start_late_cleanup_no_wait()` invokes the same deadline/retry/force helper for
a newly registered late entry on a daemon thread and never joins it on the
failing controller path. Registering after `CLEANED` first changes the ledger to
`CLEANUP_PENDING`; the helper restores `CLEANED` under the lock only after an
exact ACK or verified terminal observation. A lost ACK resumes the stable token
and a hang receives one post-deadline force dispatch, so `CLEANED` never contains
a newly appended nonterminal entry.
`register_pending_gym()` sets the current Gym callback only once;
`replace_pending_gym_with_actor()` atomically replaces it after successful
startup while holding the same lock. If cleanup already claimed the pending
abort, a subsequently published actor is registered as a late identity and is
shut down once instead of replacing an already-run action.
`register_synchronizer()` and `register_auxiliary()` deduplicate by exact object
identity, but duplicate identity is never a silent return.
`validate_and_merge_alias_registration()` requires the same resource ID, kind,
allowlisted cleanup method, cleanup-authority fingerprint, lifetime, token
journal, and exact dependency set/action authority; it atomically canonicalizes
the union of binding/alias paths. A conflict fails before changing the existing
entry. Repository-owned shared Policy/Generation fleets predeclare their full
known alias path set when possible, and source-first versus generation-first
registration yields the identical descriptor digest.
`complete_transient()` is the only success-path deregistration operation. It
requires a registered `SETUP_TRANSIENT` OPEN entry, executes the same
token-aware cleanup journal outside the ledger lock, and records an exact ACK
under the lock. Repetition with the same completed entry returns the recorded
ACK; a different/stale token, completion of a persistent entry, or transfer
while any transient is not ACKED fails closed. Port/socket adoption invokes it
immediately after the consumer has taken ownership of the port and before the
next fallible setup step.
Enabled GRPO never calls a bare `disarm()`. Once every setup transient is ACKED
and the full legacy return payload is ready,
`transfer_to_grpo_runtime_owner()` atomically freezes all live persistent
entries, their token journals, and dependency edges into one driver-local
`GRPORuntimeResourceOwner`, marks the setup ledger DISARMED in the same locked
transition, and installs that owner in the typed setup result. The owner is
non-serializable and exposes one bounded, idempotent, no-throw
`close(primary_error: BaseException | None)` operation using the same dependency
executor: synchronizers precede dependent generation endpoints, and all
remaining entries terminate without masking a primary training/refit exception.
No zero- or two-owner interval exists, and late registration after transfer is
fatal. `examples/run_grpo.py` keeps the owner live across sync/async training and
invokes it in its outer runtime `finally`; it does not directly shut down an
enabled generation endpoint. A setup result without semantic precision carries
`runtime_owner is None` and follows the existing zero-argument endpoint and
environment shutdown path byte-for-byte. Single Controller instead calls
`transfer_to_driver_runtime_handoff()`, which atomically freezes the same
exact-once live persistent entries in `ControllerRuntimeResourceHandoff`,
includes the already registered ownership cell, retains ACKED transient history
for audit but excludes it from the actor binding manifest, marks the setup
ledger DISARMED, and leaves the driver as the sole lease holder until actor
adoption. No full owner or bound cleanup callback is serialized in
`SingleControllerActorArgs`.
The implementation logs cleanup failures safely but never raises them. Alias
tests register both source-first and generation-first orders to prove either
representative delegates shutdown to the same owning fleet.

Implement `ControllerRuntimeResourceHandoff` as a driver-local wrapper over the
frozen ledger entries and a handle to `ControllerRuntimeOwnershipCell`; its
bound callbacks deliberately fail serialization. Implement
`RuntimeResourceDescriptor`, `ControllerRuntimeAdoptionEnvelope`, and
`ControllerRuntimeAdoptionAck` as exact frozen wire values with canonical
descriptor/manifest/epoch validation. Ledger registrations supply both the
driver cleanup action and its explicit descriptor/binding paths; transfer fails
if either side is absent. Actor-side binders are keyed by `SetupResourceKind`
and select only allowlisted methods from the declared `SingleControllerActorArgs`
path, then prove exact descriptor coverage and aliases before computing the
same manifest digest. The cell accepts
only exact nonce-bearing compare-and-set operations, exposes public
`adoption_status()` for lost-ACK reconciliation, permits the pending-to-driver
rollback only before `run` release, and rejects every stale epoch. The actor
death recovery API requires public death evidence, issues a fresh driver epoch,
and returns the manifest-indexed `ACKED`/`IN_FLIGHT` journal; retries reuse each
entry's stable cleanup attempt token. Every resource wrapper deduplicates that
token and reports exact termination ACK, so an unknown RPC outcome is safe to
retry without pretending the invocation itself was exactly once. The actor
builds a dormant cleanup owner during `__init__`, exposes
`adopt_runtime_resources()` and `shutdown_runtime_resources()` as repo-owned
RPCs, and makes `run(adoption_epoch)` validate/advance the lease before any
fallible runtime action. `DEFAULT_CONTROLLER_RUNTIME_ADOPTION_TIMEOUT_S` is
exactly 30.0; ACK/status/rollback waits and candidate termination share one
absolute deadline, while subsequent cleanup uses the existing independent
five-second cleanup budget.
Represent the ownership cell as a dedicated control-plane manifest descriptor
and ledger entry but never install it in the actor's resource callback table.
Driver failure cleanup terminates it last while the DRIVER lease is valid;
normal/recovered cleanup first obtains or reconciles the exact `CLOSED` ACK and
then lets the launcher terminate the cell. A lost final ACK is resolved through
the public status method, never by killing the cell optimistically.

```python
def bind_controller_runtime_sources(
    bootstrap: SemanticPrecisionBootstrap,
    source_endpoint: SemanticPrecisionSourceEndpoint,
    destination_endpoint: SemanticPrecisionIntentEndpoint,
) -> BoundSemanticPrecisionRuntimeContext | None:
    selection = bootstrap.selection
    if selection is None:
        return None
    source_endpoint.validate_realized_phase1_inputs(
        selection, bootstrap.phase1_requests_by_graph
    )
    request = source_endpoint.build_runtime_source_discovery_request(selection)
    results = source_endpoint.discover_runtime_sources(request)
    context = bootstrap.bind_runtime_sources(request, results)
    if context is None:
        raise RuntimeError("enabled semantic precision produced no runtime context")
    projection = context.to_worker_projection(
        binding_phase="phase2_bound", bound_plan_group_ids=()
    )
    installed_fleet_identities: list[object] = []
    for endpoint in (source_endpoint, destination_endpoint):
        fleet_identity = endpoint.semantic_precision_fleet_identity
        if any(fleet_identity is seen for seen in installed_fleet_identities):
            continue
        endpoint.install_precision_context(context, projection)
        installed_fleet_identities.append(fleet_identity)
    return context
```

In every GRPO launcher, finish every pure config normalization required by
topology resolution, call `prepare_grpo_controller_setup()`, and enter its
failure-cleanup scope before `init_ray()` or `setup_response_data()`. The
enabled session creates the bootstrap/ledger/deadline once and calls
`materialize()` before any native env venv/actor, `RayVirtualCluster`, placement
group, policy, generation, teacher, or Gym allocation. Pass its same owner and
construction context into `setup_response_data()` so each train/validation env
outer actor and venv child is claimed, incrementally published, and readiness-
checked before dataset/setup fallibility; retain the identity-deduplicated union
in the eventual runtime owner. Pass the same session plus the already-created
train environment map into `grpo.setup()` rather than rematerializing. For the
enabled synchronous TransferQueue trainer, `grpo.setup()` uses those inputs
after the policy/generation endpoints exist to register and finish the side-
effect-free pending `SyncRolloutActor` build under the same construction
context. It publishes runtime-env/venv work, actor, readiness ref, and shared-
controller TQ attachment incrementally, then includes their persistent
descriptors and handle in `GRPOSetupResult` before the single runtime-owner
transfer. `grpo_sync.py` receives that handle and cannot allocate a second
actor. Its token-aware close reports failure through the owner journal and
never catches-and-discards it; actor/attachment dependencies terminate before
the shared controller. Validate a custom `policy_factory`'s exact
v1 capability at that point; the TransferQueue factory follows the same typed
check and succeeds. Read and normalize
`policy.fleet_construction_timeout_s` once, create the shared construction
deadline, and thread it plus the same `selection` object into every enabled
two-phase policy/generation construction path for the sole enabled destination,
vLLM (including its internal FlashInfer-TRTLLM runtime-layout path). Standalone
TRTLLM, SGLang, Dynamo, and Megatron generation fail in the pure capability
gate before this point. Call each
factory's allocation-free begin method and register its local pending fleet
owner. Then claim and submit fallible `finish()` work, attach the ledger's
late-completion callback immediately to the returned future, and publish that
future before submitting/awaiting another fallible operation;
replace all three enabled GRPO `with ThreadPoolExecutor(...)` construction
blocks with the same repo-owned daemon/nonjoining setup submitter used below by
Single Controller. A failed or hung sibling can therefore be abandoned only
after its claims are cancelled/owned, and neither function return nor Python's
executor atexit hook joins it. The absent-policy branch alone retains the three
legacy executor contexts byte-for-byte.
Register pending Gym and every auxiliary immediately and transfer to the local
GRPO runtime owner only after the full typed setup result is ready for return.
Return that owner beside the existing setup values and keep it driver-local
until the invoking GRPO launcher closes it in `finally`; never discard ledger
authority or delegate enabled cleanup to `policy_generation.shutdown()` alone.
The absent path does not
construct or consult the ledger and retains its legacy exception/cleanup flow.
Before either branch can allocate, reject semantic-precision Dynamo, SGLang,
standalone TRTLLM, and Megatron generation through the same pure materialization/
backend-capability preflight. Do not instantiate those backends to ask whether
they are safe: their constructors already
own driver-local subprocess, reservation/placement, actor, thread, and blocking
readiness side effects, and their serialized state cannot carry the required
durable process-tree cleanup authority.
Migrate `run_grpo.py`, `run_vlm_grpo.py`, `run_grpo_sliding_puzzle.py`, and
`nemo_gym/run_grpo_nemo_gym.py` to named setup-result access and the same outer
session/owner failure scope; legacy tuple unpacking is forbidden because it can
drop ownership. Do not call Task 5 from `Policy.__init__`; that is too late for an
already-built generation endpoint and gives non-GRPO callers a different
authority.

Thread the same owner gate into all enabled setup helpers that can allocate a placement
group, deferred endpoint, generation router, value
fleet, teacher cluster/fleet, or rollout reassembler/finalizer fleet. Replace
the finalizer list-comprehension allocation with the same per-actor claim,
immediate publication, exact readiness, deadline, cancellation, and late-result
adoption protocol. Each helper either accepts a pre-claimed owner
slot and incrementally publishes children or is split into a side-effect-free
plan plus owned execution. Returning a composite after unregistered child
allocations is forbidden. This requirement applies even when that resource is
not itself a semantic source or destination.
Implement the token journal and exact shutdown ACK inside
`GenerationRouterActor`. Add an explicit construction-context parameter to its
caller helper, publish actor/address/readiness refs incrementally, make the
socket/daemon-thread stop path bounded and idempotent, and use the descriptor-
bound underlying method for both setup cleanup and runtime handoff. Driver
lambdas or copied token sets may provide a last-resort force kill only; they
cannot be the graceful cleanup authority.

In enabled `Policy` and `Value` builders, delete construction and forwarding of
the unused `RayQueue`; keep those two statements only in the isolated
absent-policy legacy branch. In `TQPolicy` and `TQValue`, create/publish the
repo-owned controller/attachment lifecycle adapters before external bootstrap,
require deterministic token-aware controller-close and attachment-release
capabilities, and move client plus worker attach into the shared owner/deadline
path. `TQPolicy` alone creates/owns the canonical
`TQControllerAuthority`; `TQValue` must receive and identity-validate a borrowed
reference to it and must never bootstrap or close another process-global TQ
runtime. Each wrapper owns only its distinct client attachment. Encode the
dependency fence so worker fleets and both attachments terminate before the
shared controller, partial Value cleanup cannot close live Policy, and a
successful seal transfers one persistent controller descriptor plus the
distinct attachment descriptors (with Value's controller reference recorded as
an alias, not a second owner). Refuse enabled construction before `tq.init` if
the installed API cannot distinguish attachment release from final global
close. Host all token journals and controller/attachment side effects in the
repo-owned `TQControllerAuthorityActor`, never in serialized clients. Make the
enabled `TQDataPlaneClient` pickle state an inert
`TQAttachmentRebindDescriptor`; Single Controller adoption claims and ACKs new
actor-process attachments against the shared authority, then releases the old
driver attachments before ownership transfer. Register the standalone
ActorArgs `dp_client` and every parent-owned nested connection alongside Policy
and optional Value attachments, identity-deduplicate them, and close the shared
controller last. Route `generation.setup_token_capture()` through that same
authority. Claim the vLLM worker's token-client/sink/source attachment before
creation, publish it in the worker fleet descriptor, and bound attachment
readiness and setup by the existing fleet-construction deadline/cancel token.
Do not stamp the served version on this enabled setup path; leave it gated for
Task 12's authenticated post-ready/commit publication. Do not create a fresh generation
lifecycle deadline or leave worker-local TQ state outside handoff; on successful
seal it becomes a persistent child of the worker fleet, and on failure/late
completion the stable token releases it before controller cleanup. Keep
semantic SGLang on the pure pre-resource unsupported-capability path. Do not
modify its constructor/thread/process topology or invoke it to probe support;
the future adapter described above is a separate follow-up.

Split NeMo Gym startup so `PendingNemoGymStartup` is constructed locally and its
token-aware aborter is registered before `start_nemo_gym_actor()` may call
`make_actor_runtime_env()` or any Ray API. Pass the shared construction context
through start and finish, claim/publish the venv PG/tasks, actor, spinup, and
tokenizer/config refs, and replace unbounded gets with deadline-aware incremental
drains. Preserve the current helpers exactly in the absent-policy branch.

After each exact public readiness ACK, call `seal_to_runtime_fleet()` for every
completed endpoint/worker group and clear all construction-only fields before
returning its wrapper. Apply the same seal/dispose rule to
value/teacher and Gym wrappers. Their runtime shutdown methods delegate to the
sealed underlying fleet authority and do not retain the pending construction
owner, latch, cancel token, or deadline. A `RayWorkerGroup` transfers its pooled
initializer handles into that sealed runtime authority because those actors own
the non-detached workers; shutdown kills workers first and initializers last.

After both endpoints exist, call `bind_controller_runtime_sources()` once.
Only after it returns may any legacy collective, NCCL reshard, remote
sparse, checkpoint-engine, or future supported communicator be constructed or
initialized. Pass the exact bound runtime context into
`create_weight_synchronizer()`, the
remote-sparse constructor, or the side-effect-free direct-collective handle
builder and register the returned pending/constructed object before any
fallible `init_communicator()` or endpoint call.
Give every enabled synchronizer and direct handle a side-effect-free local
construction phase followed by explicit owned initialization. Pass the same
`FleetConstructionContext` into that initialization, and move all rendezvous,
port/address, unique-ID, group, baseline, metadata, and endpoint-init Ray
submissions behind its claim/publish boundary. Replace sequential or unbounded
`ray.get` calls with incremental finite drains using the remaining shared
construction deadline. On the first timeout or `BaseException`, resolve the
same cancellation latch and return through the ledger's nonjoining cleanup;
late completion is adopted only for shutdown. This applies to direct and
synchronizer collective, IPC, NCCL reshard, checkpoint-engine,
and both remote-sparse implementations.
For remote sparse, remove `start_baseline()` from `init_policy`; call it only
after Phase 2 and synchronizer registration, retain its refs in the registered
synchronizer, and validate them before initial sync. Keep every transport's
legacy metadata/baseline/prepare sequence operational in Task 5. The absent
path omits the new keyword/helper and retains its original handshake and exact
kwargs.

In Single Controller, perform the same Phase 1 call after pure config and
`train_iters` normalization but before `_build_clusters()` and before creating
the parallel build executor. Create the setup ledger before
`setup_response_data()`, thread its construction context through generic venv/
environment creation, and publish/readiness-check each outer actor before the
next environment or dataset step. Retain and identity-deduplicate both train and
validation environment handles into persistent descriptors and actor args.
Implement the token journal once in the inherited, `@final`
`EnvironmentInterface.shutdown_semantic_precision_resource()` wrapper; it
calls the existing polymorphic zero-argument `shutdown()` behind the journal,
and descriptor construction may bind only that new method. Validate exact
method inheritance before actor/venv allocation so concrete built-ins need no
edits and a custom override cannot bypass fencing.
Register local pending owners, submit endpoint
builds with `submit_owned_setup_future()`, and publish results in completion
order. The enabled path uses the repo-owned daemon-thread setup submitter in
`fleet_construction.py`; it never enters `with ThreadPoolExecutor` and never
relies on `ThreadPoolExecutor.shutdown(wait=False)`, whose interpreter-exit hook
can still wait for a hung worker. Cancellation abandons the daemon after
claiming its resources, so neither method return nor process exit joins it. The
absent path retains its legacy executor byte-for-byte. Resolve the endpoint futures,
run Phase 2,
then await the Gym result before constructing the supported vLLM communicator.
Run the same Phase 2 seam before that communicator. Never perform Phase 2 inside `Policy.__init__`, a worker actor, or
the Gym future. Use the same setup resource ledger and registration points as
GRPO. Any exception after resource construction invokes its bounded cleanup
before re-raising; executor shutdown, Gym abort, endpoint shutdown, synchronizer
shutdown, or cleanup diagnostics cannot hide or replace the original failure.
After either setup shape succeeds, call
`transfer_to_driver_runtime_handoff()` and return actor args whose only new
ownership-control payload is the adoption envelope; their existing runtime
resource fields/handles remain so the actor can bind every persistent descriptor.
Define
`DEFAULT_CONTROLLER_RUNTIME_ADOPTION_TIMEOUT_S = 30.0`; construct one finite
absolute adoption deadline at launcher entry and do not restart it for submit,
constructor, ACK, rollback, or candidate kill. In
`examples/run_grpo_single_controller.py`, place actor serialization,
`SingleControllerActor.remote()`, exact adoption ACK validation, and run submit
inside one outer `try/except/finally`. `run()` is never the adoption RPC and is
never submitted before the exact ACK. The launcher removes its per-resource
manual teardown loops and calls only the handoff's current-owner cleanup path.
All supported paths transfer the same complete resource manifest of live
persistent resources plus the same ACKED transient audit history. Completed
deferred reservations are token-closed immediately after successful adoption
and never appear as dead handles in actor descriptors.

Add explicit keyword-only `precision_selection`,
`precision_adapter_authority`, and `fleet_construction_deadline` arguments to
`Policy` and `VllmGeneration`. Each supported wrapper retains the typed
driver-side value and stable fleet token outside user config. Add the pending
owner/nested-ref cancel token/shared deadline to every
enabled `RayWorkerGroup` construction in
Policy (Megatron and DTensor) plus vLLM sync/async; no wrapper may wait for a completed group before
giving the common worker-group layer ownership. Omit all three arguments on every
absent-policy call. Pass the same `FleetConstructionContext` into lazy cluster
PG creation, master-port discovery, owned venv staging, and batch port
discovery before any initializer. Add the
selection/per-graph-request/contribution and stripped worker-projection wire
entrypoints to Megatron and DTensor v1/v2 policy workers plus vLLM sync/async
generation workers/worker pools.
Their worker-group methods call the selection/authority/projection
`to_wire_dict()` methods only in the argument expression of a Ray submission
and decode immediately at worker entry. The construction-provisioned authority
manifest—not a Python adapter map—is retained as the independent expected value
for later projection decoding. Runtime source discovery uses controller-local
producer proxies and Task 4B bulk factories across every contributor; it does
not scan parameter names in the controller, select a model family again, import
a version-specific vLLM adapter, rebuild topology on a refit, or serialize the
aggregate request/results/context.

Although value and teacher fleets are not selected quantization destinations in
Task 5, they are setup-owned GPU resources. On the enabled controller path,
`Value`/`TQValue`, `TeacherWorkerGroup`, and
`create_teacher_worker_groups()` create a purely local pending owner per distinct
value/teacher fleet, let the controller ledger register it before submission,
and receive the one shared construction deadline. `reserve_teacher_clusters()`
uses a ledger claim and publishes each cluster immediately. Each fleet passes its own
owner and nested-ref cancel token plus that common deadline to `RayWorkerGroup`;
each teacher/value fleet and cluster is incrementally published before its next
fallible setup call. Replace OPD's private `worker.__ray_ready__.remote()` probe with
`semantic_precision_construction_ready.remote()` and validate exact `True`
under the remaining shared construction deadline. The absent path omits every
new argument and retains its existing constructors and readiness behavior.

Extend `WeightSynchronizer`, `create_weight_synchronizer()`, and each supported
vLLM transport synchronizer/direct handle named in the Task 5 file list with the exact optional
`precision_context` and `fleet_construction` keywords. The enabled branch
validates the complete retained
context, including request/results/adapter authority and intent-group identity, and
then performs the existing metadata preparation and communicator work; the
absent branch is the original code path. Do not change
`Policy.prepare_refit_info()` to return semantic metadata: Phase 2 discovery and
legacy transport preparation are separate contracts. Only Task 12 may atomically
replace the latter after Tasks 7-11 provide an executable transaction.
Give each synchronizer an explicit safe-state serializer that drops its
controller-only context field and retains only the stripped projection/context
ID, legacy runtime state, and the shared cleanup-authority ActorHandle. It
explicitly drops any process-local cleanup journal/callback; creating or using
one is fatal. Validate the actual serialized
`SingleControllerActorArgs`, not only endpoint wrappers, before actor launch;
Task 7 later replaces the Phase-2 marker with exact bound plan-group IDs.
Treat `fleet_construction` as a single-use field: `init_communicator()` consumes
it, publishes/adopts or cancels every child, seals the phase in `finally`, and
clears the field. Refuse `__getstate__` while that phase is not sealed and never
serialize the owner, cancel token/latch, or construction deadline.
Keep `WeightSynchronizer.shutdown()` and other established zero-argument legacy
APIs unchanged. Add their distinct
`shutdown_semantic_precision_resource(cleanup_attempt_token)` wrappers returning
an exact `ResourceTerminationAck`; on enabled synchronizers these forward to
the shared `SynchronizerCleanupAuthorityActor`, which owns the journal and
underlying shutdown side effect across driver/actor serialization. Adapt every
auxiliary cleanup through an equally distinct token-aware surface and reject
stale/different tokens in the underlying resource authority, not in a wrapper
copy. On enabled vLLM
paths detach the synchronizer as a non-owning endpoint link and encode a
synchronizer-before-endpoint dependency in the ledger/handoff manifest; never
invoke it both transitively and directly. The setup ledger and Single
Controller journal call only the typed wrapper, while absent-policy callers
continue to use legacy zero-argument shutdown.

At the first executable line of PPO setup, call
`reject_semantic_precision_for_unsupported_algorithm("ppo", policy_config)`.
At the first executable line of distillation setup, call
`reject_semantic_precision_for_unsupported_algorithm(
"distillation", policy_config, teacher_config)` before applying checkpoint
overrides or extracting other setup state. It performs exact key-presence checks
for every supplied model config and raises before normalization or resource
construction when any semantic precision policy is configured; with none it
returns without reading unrelated nested fields or changing the existing path.
Call the same side-effect-free helper in `examples/run_ppo.py` and
`examples/run_distillation.py` as soon as the resolved config is available and
before Ray/data/environment setup. Keep the algorithm guards too, so direct
Python callers fail closed even when they bypass a launcher.

- [ ] **Step 7: Wire `explain-precision` to the same Phase 1 authority**

`tools/config_cli.py explain-precision RECIPE` resolves inheritance and
interpolation exactly as `expand` does, constructs the same built-in paired
bundle registry, and invokes only `SemanticPrecisionBootstrap.materialize()`.
It prints graph lifecycles, exact decoder universes, full role predicates,
compact matched domains and logical cardinalities, selected/unselected counts,
layer ranges, BF16 fences, atomic expansion, requested endpoint formats,
resolved immutable model revisions, `semantic_structure_digest`, and
`selection_group_id`. It labels producer, runtime source, mutability,
alias/cadence, physical layout, transform, `runtime_source_digest`,
`intent_group_id`, and final plan IDs unavailable at their exact later phase.
It never reimplements selector logic, invokes Phase 2, or imports a runtime
producer.
Add every Python path labeled `Create` in Task 5's Files list explicitly to
`pyrefly.toml`, including the new unit/functional test modules and functional
`conftest.py`; directory discovery is not a substitute.

- [ ] **Step 8: Run focused tests, type checks, and repository config tests**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/algorithms/test_controller_setup_teardown.py tests/unit/algorithms/test_opd.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/data_plane/test_architecture_invariants.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py tests/unit/models/generation/test_semantic_precision_endpoints.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py tests/unit/tools/test_config_cli.py -k 'precision or semantic_precision'`

Run: `uv run --no-sync pytest -q tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/test_config_validation.py`

Run without a name filter so materialization invariants whose test names do not
contain `precision` are also mandatory: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py`

Run without a name filter so hidden-allocation, communicator, and ownership-state
cases are mandatory: `uv run --no-sync pytest -q tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_opd.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py`

Run: `uv run --no-sync pytest -q tests/unit/environments/test_nemo_gym_utils.py`

Run: `uv run --no-sync pytest -q tests/unit/experience/test_rollout_reassembler_actor.py`

Run: `uv run --no-sync pytest -q tests/unit/experience/test_sync_rollout_actor.py tests/unit/algorithms/test_grpo.py`

Run: `uv run --no-sync pytest -q tests/unit/models/generation/test_generation_router.py`

Run: `uv run --no-sync pytest -q tests/unit/data_plane/test_tq_lifecycle.py tests/unit/data_plane/test_tq_policy_routes.py tests/unit/models/value/test_tq_value.py`


Run: `uv run --no-sync pytest -q tests/unit/environments/test_environment_utils.py tests/unit/data/test_utils.py`

Run: `uv run --no-sync pytest -q tests/unit/data_plane/test_architecture_invariants.py`


Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_runtime_binding.py tests/unit/precision_policy/test_topology_adapters.py`

Run on a pinned real-Ray cluster image: `uv run --no-sync pytest -q tests/functional/test_fleet_construction_ray.py`

Run on a pinned real-Ray cluster image: `uv run --no-sync pytest -q tests/functional/test_single_controller_resource_handoff_ray.py`

Run on a pinned real-Ray cluster image: `uv run --no-sync pytest -q tests/functional/test_single_controller_tq_handoff_ray.py`

Run on the pinned vLLM cluster image: `NEMO_RL_SEMANTIC_PRECISION_TEST_BACKEND=vllm uv run --no-sync pytest -q tests/functional/test_semantic_precision_initial_sync.py`



Run: `uv run --no-sync pyrefly check nemo_rl/distributed/fleet_construction.py nemo_rl/distributed/worker_groups.py nemo_rl/distributed/virtual_cluster.py nemo_rl/utils/venvs.py nemo_rl/models/policy/__init__.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo_single_controller.py tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/functional/conftest.py tests/functional/test_fleet_construction_ray.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py`

Run: `uv run --no-sync pyrefly check nemo_rl/environments/nemo_gym.py tests/unit/environments/test_nemo_gym_utils.py`

Run: `uv run --no-sync pyrefly check nemo_rl/experience/rollout_reassembler_actor.py tests/unit/experience/test_rollout_reassembler_actor.py`

Run: `uv run --no-sync pyrefly check nemo_rl/algorithms/grpo_sync.py nemo_rl/experience/sync_rollout_actor.py tests/unit/experience/test_sync_rollout_actor.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/generation_router.py tests/unit/models/generation/test_generation_router.py`

Run: `uv run --no-sync pyrefly check nemo_rl/data_plane/interfaces.py nemo_rl/data_plane/adapters/transfer_queue.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/value/tq_value.py tests/unit/data_plane/test_tq_lifecycle.py tests/unit/data_plane/test_tq_policy_routes.py tests/unit/models/value/test_tq_value.py`

Run: `uv run --no-sync pyrefly check nemo_rl/environments/interfaces.py nemo_rl/environments/utils.py nemo_rl/data/utils.py tests/unit/environments/test_environment_utils.py tests/unit/data/test_utils.py`

Run: `uv run --no-sync pyrefly check nemo_rl/algorithms/opd.py nemo_rl/models/policy/teacher_worker_group.py nemo_rl/models/value/lm_value.py nemo_rl/models/value/tq_value.py tests/unit/algorithms/test_opd.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py`

Run: `uv run --no-sync pyrefly check nemo_rl/precision_policy nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py nemo_rl/models/generation/__init__.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/ppo.py nemo_rl/algorithms/distillation.py nemo_rl/algorithms/single_controller_utils/setup.py nemo_rl/data_plane/factory.py nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/factory.py nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/direct_collective.py nemo_rl/weight_sync/collective_weight_synchronizer.py nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py nemo_rl/weight_sync/ipc_weight_synchronizer.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py tools/config_cli.py tests/unit/precision_policy/test_materialize.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/algorithms/test_controller_setup_teardown.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py`

Run: `uv run --no-sync pyrefly check tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/data_plane/test_architecture_invariants.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/generation/test_semantic_precision_endpoints.py tests/functional/test_semantic_precision_initial_sync.py tests/unit/tools/test_config_cli.py`

Run: `uv run --no-sync pyrefly check examples/run_grpo.py tests/unit/algorithms/test_grpo.py`

Run: `uv run --no-sync pyrefly check examples/run_grpo.py examples/run_vlm_grpo.py examples/run_grpo_sliding_puzzle.py examples/nemo_gym/run_grpo_nemo_gym.py examples/run_ppo.py examples/run_distillation.py tests/unit/test_config_validation.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/distributed/fleet_construction.py nemo_rl/distributed/worker_groups.py nemo_rl/distributed/virtual_cluster.py nemo_rl/utils/venvs.py nemo_rl/models/policy/__init__.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo_single_controller.py tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/functional/test_fleet_construction_ray.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/environments/nemo_gym.py tests/unit/environments/test_nemo_gym_utils.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/experience/rollout_reassembler_actor.py tests/unit/experience/test_rollout_reassembler_actor.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/algorithms/grpo_sync.py nemo_rl/experience/sync_rollout_actor.py tests/unit/experience/test_sync_rollout_actor.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/generation_router.py tests/unit/models/generation/test_generation_router.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/data_plane/interfaces.py nemo_rl/data_plane/adapters/transfer_queue.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/value/tq_value.py tests/unit/data_plane/test_tq_lifecycle.py tests/unit/data_plane/test_tq_policy_routes.py tests/unit/models/value/test_tq_value.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/environments/interfaces.py nemo_rl/environments/utils.py nemo_rl/data/utils.py tests/unit/environments/test_environment_utils.py tests/unit/data/test_utils.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/algorithms/opd.py nemo_rl/models/policy/teacher_worker_group.py nemo_rl/models/value/lm_value.py nemo_rl/models/value/tq_value.py tests/unit/algorithms/test_opd.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py`

Run: `uv run --no-sync pre-commit run --files examples/run_grpo.py tests/unit/algorithms/test_grpo.py`

Run: `uv run --no-sync pre-commit run --files examples/run_grpo.py examples/run_vlm_grpo.py examples/run_grpo_sliding_puzzle.py examples/nemo_gym/run_grpo_nemo_gym.py examples/run_ppo.py examples/run_distillation.py tests/unit/test_config_validation.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/precision_policy/materialize.py nemo_rl/precision_policy/adapters/__init__.py nemo_rl/precision_policy/compiler.py nemo_rl/precision_policy/topology.py nemo_rl/precision_policy/topology_resolver.py nemo_rl/precision_policy/runtime_binding.py nemo_rl/precision_policy/source_discovery.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py nemo_rl/models/generation/__init__.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/ppo.py nemo_rl/algorithms/distillation.py nemo_rl/algorithms/single_controller_utils/setup.py nemo_rl/data_plane/factory.py nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/factory.py nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/direct_collective.py nemo_rl/weight_sync/collective_weight_synchronizer.py nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py nemo_rl/weight_sync/ipc_weight_synchronizer.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py tools/config_cli.py tests/unit/precision_policy/test_materialize.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/algorithms/test_controller_setup_teardown.py tests/unit/single_controller/test_setup.py tests/unit/data_plane/test_architecture_invariants.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/generation/test_semantic_precision_endpoints.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py tests/functional/conftest.py tests/functional/test_semantic_precision_initial_sync.py tests/unit/tools/test_config_cli.py pyrefly.toml`

Run: `git diff --check`

Expected: all commands pass; enabled runs show one Phase 1 and one Phase 2 event,
every communicator construction, remote baseline, and initialization follows
Phase 2 and ledger registration, every enabled transport still performs a real
successful initial sync through its legacy operational preparation, all worker
contributions are accounted for, the complete controller-local runtime context
survives through refit validation while only its trust-safe projection crosses
Ray, the public exact-`True` construction ACK, nested-ref cancellation/fate-sharing,
and Single Controller adoption canaries pass, and PG/topology/venv/generation-
router plus NeMo Gym startup, communicator, worker,
and rollout-finalizer construction timeout/failure teardown (including late
completions), plus synchronous rollout-actor runtime-env/TQ attachment
construction and shutdown, are bounded
by the same deadline, and successful sealing leaves no construction latch,
pending owner, or deadline alive in runtime wrappers while retaining each
required initializer as a runtime lifetime owner until worker-first shutdown. Enabled Policy/Value
create no hidden RayQueue. Single
Controller has one lease holder, terminates its control cell last, and uses
token/ACK-journaled cleanup throughout launch/run/failure. PPO/distillation fail before resources and absent-policy controller
calls are byte-for-byte identical to the legacy fixtures. This result does not
claim silent-peer or all-transport refit fail-fast; that remains gated on Tasks
11-12.

- [ ] **Step 9: Commit**

```bash
git add nemo_rl/distributed/fleet_construction.py nemo_rl/distributed/worker_groups.py nemo_rl/distributed/virtual_cluster.py nemo_rl/utils/venvs.py nemo_rl/models/policy/__init__.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo_single_controller.py tests/unit/distributed/test_fleet_construction.py tests/unit/distributed/test_worker_groups.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/utils/test_venvs.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/functional/conftest.py tests/functional/test_fleet_construction_ray.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py
git add examples/run_grpo.py tests/unit/algorithms/test_grpo.py
git add examples/run_vlm_grpo.py examples/run_grpo_sliding_puzzle.py examples/nemo_gym/run_grpo_nemo_gym.py examples/run_ppo.py examples/run_distillation.py tests/unit/test_config_validation.py
git add nemo_rl/environments/nemo_gym.py tests/unit/environments/test_nemo_gym_utils.py
git add nemo_rl/experience/rollout_reassembler_actor.py tests/unit/experience/test_rollout_reassembler_actor.py
git add nemo_rl/algorithms/grpo_sync.py nemo_rl/experience/sync_rollout_actor.py tests/unit/experience/test_sync_rollout_actor.py
git add nemo_rl/models/generation/generation_router.py tests/unit/models/generation/test_generation_router.py
git add nemo_rl/data_plane/interfaces.py nemo_rl/data_plane/adapters/transfer_queue.py nemo_rl/models/policy/tq_policy.py nemo_rl/models/value/tq_value.py tests/unit/data_plane/test_tq_lifecycle.py tests/unit/data_plane/test_tq_policy_routes.py tests/unit/models/value/test_tq_value.py
git add nemo_rl/environments/interfaces.py nemo_rl/environments/utils.py nemo_rl/data/utils.py tests/unit/environments/test_environment_utils.py tests/unit/data/test_utils.py
git add nemo_rl/algorithms/opd.py nemo_rl/models/policy/teacher_worker_group.py nemo_rl/models/value/lm_value.py nemo_rl/models/value/tq_value.py tests/unit/algorithms/test_opd.py tests/unit/models/policy/test_teacher_worker_group.py tests/unit/models/value/test_lm_value.py tests/unit/models/value/test_tq_value.py
git add nemo_rl/precision_policy/materialize.py nemo_rl/precision_policy/adapters/__init__.py nemo_rl/precision_policy/compiler.py nemo_rl/precision_policy/topology.py nemo_rl/precision_policy/topology_resolver.py nemo_rl/precision_policy/runtime_binding.py nemo_rl/precision_policy/source_discovery.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py nemo_rl/models/generation/__init__.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/algorithms/controller_setup_teardown.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/ppo.py nemo_rl/algorithms/distillation.py nemo_rl/algorithms/single_controller_utils/setup.py nemo_rl/data_plane/factory.py nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/factory.py nemo_rl/weight_sync/refit_plan.py nemo_rl/weight_sync/direct_collective.py nemo_rl/weight_sync/collective_weight_synchronizer.py nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py nemo_rl/weight_sync/ipc_weight_synchronizer.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py tools/config_cli.py tests/unit/precision_policy/test_materialize.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_ppo.py tests/unit/algorithms/test_distillation.py tests/unit/algorithms/test_controller_setup_teardown.py tests/unit/single_controller/test_setup.py tests/unit/data_plane/test_architecture_invariants.py tests/unit/models/policy/test_semantic_precision_endpoints.py tests/unit/models/generation/test_semantic_precision_endpoints.py tests/unit/weight_sync/test_semantic_precision_handshake.py tests/unit/weight_sync/test_refit_plan.py tests/functional/test_semantic_precision_initial_sync.py tests/unit/tools/test_config_cli.py pyrefly.toml
git commit -s -m "feat(precision): add controller-owned semantic bootstrap"
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
- Consumes before construction: `compile_te_precision_recipe(selection: CompiledPrecisionSelectionGroup, construction_bindings: Sequence[SourceModuleConstructionBinding]) -> TEPrecisionArtifact`. Consumes after construction: the complete Task 5 `BoundSemanticPrecisionRuntimeContext`, the same artifact, and realized TE modules/storage; it never accepts a separately supplied selection/intents pair.
- Produces: frozen `SourceModuleConstructionBinding`, deterministic enabled exact matchers for training-MXFP8 modules, explicit BF16 boundary evaluation recipes, a recipe digest, and `validate_realized_training_precision(context, artifact, realized_modules) -> None`. Validation requires exact agreement among `context.selection`, `context.intents`, the retained request/results/adapter authority, generated TE configuration, and realized BF16/MXFP8 storage, and carries `runtime_context_id` into later bound plans; no separate TE file or reconstructed context is trusted as another source of truth.

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

Add tests that first/last routed modules remain BF16, QKVO can be a second scope, duplicate/partial physical bindings fail, disabled matcher is rejected, user-supplied `te_precision_config_file` plus semantic policy fails as two sources of truth, and realized TE module precision mismatch fails before refit. Bare selection/intents, a different `runtime_context_id`, or replaced request/result/adapter authority fail before realized validation.

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
    selection = require_compiled_selection(policy_config)
    artifact = compile_te_precision_recipe(
        selection,
        construction_bindings_for(policy_config),
    )
    if artifact.recipe is not None:
        model_cfg.quant_recipe = load_quantization_recipe_from_mapping(artifact.recipe)
```

Keep `fp8_cfg` only as backend compute/storage mechanics derived from the
compiled selection. Do not apply a global MXFP8 default to unmatched modules.
Persist the artifact digest. After model construction, the TE runtime producer
must account for every BF16 boundary and MXFP8 middle module in one exact
projection; call `validate_realized_training_precision()` against the final
runtime-bound intents. Any selected/unselected, dtype/encoding, shape, or
storage mismatch raises before communicator creation. A user-supplied
`te_precision_config_file` plus semantic policy remains an error unless an
explicit migration path parses it and proves canonical equality with the
generated artifact.

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
- Consumes: the complete Task 5 `BoundSemanticPrecisionRuntimeContext` (never bare compiled intents), including its Task 2 `owner_refit_requirements`, persisted `source_alias_contracts`, retained Phase 2 request/results, exact adapter authority, and `runtime_context_id`; Task 2's single typed `ComponentRole` vocabulary and built-in `LOGICAL_VALUES`/`VALUES`/`BLOCK_SCALES` constants; Task 4A.2 source-storage realizations and allowed-normalizer manifest identity; explicit `SourceRuntimeParallelTopology` and `DestinationRuntimeParallelTopology`; endpoint capabilities; and realized source/destination bindings. Task 4 supplies no rank ownership. `refit_plan.py` imports and may re-export the Task 2 types; it never declares a second `NewType` or requirement enum.
- Produces: `PhysicalFormatStage`, `PhysicalAxisMapping`, `PhysicalPadding`, `PhysicalPermutation`, `PhysicalLayoutDescriptor`, `PhysicalRepresentation`, `EndpointPlacement`, `PhysicalComponentDescriptor`, `RealizedBindingFormat`, `DirectCopyCapabilityProof`, `ComponentBinding`, `BindingSet`, `TransformLocus`, realized `PhysicalOwnerSchedule`, `PhysicalOwner`, `BoundPhysicalOwner`, `RealizedDestinationOwnerGroup`, `ImmutableContributorCacheKey`, `MixedCadenceCompositionPlan`, `SourceVersionFenceRequirement`, `SourceVersionFence`, `EndpointCapabilities`, derived `RankLocalEndpointOwnership`, `BoundSourcePlans`, `BoundDestinationPlans`, `BoundComponentBatch`, `DestinationCommitReady`, `DestinationPoisonReason`, `LocalExecutionPlan`, `CanonicalStartupLoadPlan`/`CanonicalStartupLoadPlanGroup`, graph-level `CanonicalRefitPlan`, alias-aware `GraphTransactionMember`, ordered `CanonicalRefitPlanGroup`, `build_canonical_plan_groups(context, ...)`, validation functions, and ordered wire metadata. Every bound record carries the exact `runtime_context_id`, and Task 7 emits the `plan_bound` stripped worker projection only after the whole plan group validates. A physical schedule maps semantic owner requirements to startup cached components, every-version components, and exactly-once finalization after realized cadence closure. It de-duplicates a canonical-alias source export independently from destination realization: distinct main and drafter allocations receive explicit fan-out bindings, while destination load/finalize/ACK de-duplication requires an adapter proof that physical storage-owner identity and finalizer identity are equal. Identical-storage aliases need no live replica fence; synchronized replicas do.

- [ ] **Step 1: Write failing component and ownership tests**

```python
def test_direct_copy_requires_complete_layout_equality() -> None:
    binding = bf16_trtllm_binding(
        logical_shape=(128, 928, 2688),
        runtime_shape=(128, 42, 1024, 64),
    )
    assert binding.logical_format == BF16_FORMAT
    with pytest.raises(ValueError, match="physical descriptor"):
        require_direct_copy(
            binding.realized_format,
            PhysicalFormatStage.DESTINATION_LOAD_API,
            PhysicalFormatStage.DESTINATION_RUNTIME,
        )
    assert plan_transform(binding).locus is TransformLocus.DESTINATION_NATIVE_LOADER

def test_direct_copy_rejects_non_adjacent_stage_skip() -> None:
    binding = bf16_trtllm_binding(
        logical_shape=(128, 928, 2688),
        runtime_shape=(128, 42, 1024, 64),
    )
    with pytest.raises(ValueError, match="adjacent physical stages"):
        require_direct_copy(
            binding.realized_format,
            PhysicalFormatStage.WIRE,
            PhysicalFormatStage.DESTINATION_RUNTIME,
        )

def test_mxfp8_component_order_is_values_then_block_scales() -> None:
    binding = mxfp8_binding("layer.2.expert.0.gate")
    assert tuple(component.role for component in binding.components) == ("values", "block_scales")
```

Add `test_plan_builder_requires_complete_bound_runtime_context`: bare intents,
selection-plus-intents, a context with replaced request/results/adapter
authority, and realized bindings carrying a foreign `runtime_context_id` all
fail before ownership derivation or endpoint capability probing. The exact
context yields plans whose every nested record carries its context ID; the
`plan_bound` worker projection contains exactly the canonical startup/refit
plan-group IDs and no controller trust evidence.

Add tests for arbitrary future component roles, missing/duplicate components, semantic-set inequality only across endpoints required by the same semantic owner requirement and realized physical schedule, unsupported endpoint formats, native MXFP8 direct component transfer, BF16→MXFP8 destination transform, canonical BF16→TRTLLM native loader, fused owner atomicity, source/destination TP/EP/PP ownership derivation, canonical versus rank-local digests, and deterministic plan-group assembly. Exercise each adjacent realized stage: source storage→wire, wire→destination load API, and load API→destination runtime. DIRECT_COPY requires ordered physical-component equality plus an adapter capability proof for that exact adjacent stage pair; equal dtype or logical `FormatDescriptor` never suffices. Prove that a logical BF16 `[E,I,H]` wire/load tensor can pass through a destination-native loader into padded/permuted TRTLLM `[E,blocks,I_pad,block]` runtime storage but cannot be copied directly to that runtime allocation. Prove that all-frozen source-served graphs produce startup plans; mixed mutable/frozen graphs produce startup plans for frozen independent owners and repeated wire payloads only for mutable owners; startup-owner digests become immutable refit preconditions; and no startup owner appears in an every-version wire payload. A mutable training-only graph and checkpoint-served graph contribute no source-load plan for their directly owned bodies. A checkpoint-served graph's cross-graph canonical aliases inherit their canonical owners' source cadence. An alias-only member adds no duplicate canonical source export or wire payload. Give it distinct main/drafter destination owners and require fan-out load/finalize/ACK; de-duplicate those destination actions only in a separate fixture where the endpoint proves identical storage-owner and finalizer identity. Missing MTP/drafter binding on a derived owning rank fails, while absence on a derived non-owner rank is valid.

Add synchronized-replica fence tests proving that exactly an in-scope served
canonical alias whose resolved training authority contributes a startup or
every-version source realization also contributes a matching fence requirement.
This includes a checkpoint-served graph's cross-graph alias to a training
authority. Training-only, direct checkpoint-body, `out_of_scope`, and
non-training-authority members do not.
For every required replica/rank, enforce the order optimizer update → replica
synchronization → `SOURCE_VERSION_READY` fence → canonical export. Reject a
missing, stale, pre-update, wrong-group, wrong-topology, wrong-version,
wrong-rank, duplicate, or incomplete fence set before any wire operation.
Add paired direct-copy fixtures showing that identical representations on
different ranks are valid with a matching NCCL route/capability proof, while
equal dtypes with different layout, axis mapping, shape, padding, permutation,
or storage encoding fail.

Add mixed-cadence realized-owner tests where frozen and mutable semantic
contributors share one destination physical owner/finalizer. An A→B→C sequence
must transfer the frozen contributors once into a verified persistent startup
cache, transfer only mutable contributors for B and C, combine cached and fresh
canonical components, and compose/finalize the physical owner exactly once per
update. Accept either advertised native partial preservation or split/repack;
fail preflight when neither is supported. Neither capability may expand or
otherwise alter the compiled semantic selection or its BF16 fences. Assert the
cache key/capability fingerprint enters the plan digest and that storage
rebinding, evidence/layout/topology/capability changes, explicit invalidation,
or poison invalidates it.

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

class PhysicalFormatStage(StrEnum):
    SOURCE_STORAGE = "source_storage"
    WIRE = "wire"
    DESTINATION_LOAD_API = "destination_load_api"
    DESTINATION_RUNTIME = "destination_runtime"

@dataclass(frozen=True, slots=True)
class PhysicalAxisMapping:
    logical_axis: str
    physical_axes: tuple[str, ...]
    mapping_id: str

class PhysicalPaddingSemantics(StrEnum):
    ZERO_FILLED = "zero_filled"
    UNSPECIFIED_IGNORED = "unspecified_ignored"

@dataclass(frozen=True, slots=True)
class PhysicalPadding:
    logical_axis: str
    pad_before: int
    pad_after: int
    semantics: PhysicalPaddingSemantics
    fill_encoding: str | None

@dataclass(frozen=True, slots=True)
class PhysicalPermutation:
    permutation_id: str
    input_axis_order: tuple[str, ...]
    output_axis_order: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class EndpointPlacement:
    rank: int
    device_type: str
    memory_space: str

@dataclass(frozen=True, slots=True)
class PhysicalLayoutDescriptor:
    axis_order: tuple[str, ...]
    logical_to_physical_axes: tuple[PhysicalAxisMapping, ...]
    padding: tuple[PhysicalPadding, ...]
    permutation: PhysicalPermutation | None
    storage_encoding: str

@dataclass(frozen=True, slots=True)
class PhysicalRepresentation:
    role: ComponentRole
    physical_dtype: str
    physical_shape: tuple[int, ...]
    layout: PhysicalLayoutDescriptor

@dataclass(frozen=True, slots=True)
class PhysicalComponentDescriptor:
    representation: PhysicalRepresentation
    placement: EndpointPlacement

@dataclass(frozen=True, slots=True)
class RealizedBindingFormat:
    source_storage: tuple[PhysicalComponentDescriptor, ...]
    wire: tuple[PhysicalComponentDescriptor, ...]
    destination_load_api: tuple[PhysicalComponentDescriptor, ...]
    destination_runtime: tuple[PhysicalComponentDescriptor, ...]
    capability_fingerprint: str

@dataclass(frozen=True, slots=True)
class DirectCopyCapabilityProof:
    source_stage: PhysicalFormatStage
    destination_stage: PhysicalFormatStage
    source_representation_digest: str
    destination_representation_digest: str
    source_placement_digest: str
    destination_placement_digest: str
    transport_capability_fingerprint: str

@dataclass(frozen=True, slots=True)
class BindingSet:
    graph_instance_id: str
    semantic_graph_path: str
    semantic_id: str
    logical_format: FormatDescriptor
    realized_format: RealizedBindingFormat
    components: tuple[ComponentBinding, ...]
    source_owner_families: tuple[OwnerFamilyReference, ...]
    destination_physical_owners: tuple[PhysicalOwner, ...]
    semantic_precision_group_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ImmutableContributorCacheKey:
    contributor_digest: str
    destination_owner_group_id: str
    storage_generation: str
    topology_digest: str
    capability_fingerprint: str

@dataclass(frozen=True, slots=True)
class MixedCadenceCompositionPlan:
    destination_owner_group_id: str
    finalizer_group_id: str
    immutable_cache_keys: tuple[ImmutableContributorCacheKey, ...]
    mutable_contributors: tuple[OwnerFamilyReference, ...]
    mode: Literal["native_preserve", "split_repack"]

@dataclass(frozen=True, slots=True)
class SourceVersionFenceRequirement:
    replica_group_id: str
    boundary: SourceSynchronizationBoundary
    synchronization_evidence_digest: str
    topology_digest: str
    required_ranks: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class SourceVersionFence:
    replica_group_id: str
    boundary: SourceSynchronizationBoundary
    synchronization_evidence_digest: str
    topology_digest: str
    source_version: int
    rank: int
    completion_fence_id: str
```

`PhysicalPadding` requires one exact non-empty `fill_encoding` for
`ZERO_FILLED` and requires `None` for `UNSPECIFIED_IGNORED`. Padding semantics
and the conditional fill encoding participate in physical equality and the
plan digest.

`FormatDescriptor` remains logical encoding intent. `RealizedBindingFormat`
separately records ordered component roles and complete physical descriptors at
`SOURCE_STORAGE`, `WIRE`, `DESTINATION_LOAD_API`, and
`DESTINATION_RUNTIME`; a destination-native finalizer may therefore preserve a
logical BF16 load API while producing padded/permuted runtime storage.

The source endpoint deterministically lowers each attested Task 4A.2
realization through the classifier's semantic axis mapping into its
`SOURCE_STORAGE` representation, then revalidates the live buffers. The static
realization digest, producer evidence, allowed-normalizer manifest, selected
normalizer capability, live endpoint capability, and resulting physical
descriptor all participate in the binding and plan digests. A normalized
discovery view is metadata-only; lowering does not crop, unswizzle, repack, or
copy a payload until the selected execution plan requires that transform.
Exact source/destination physical equality cannot erase numeric encoding tags
or normalization provenance.

Plan transforms only across adjacent stage pairs. DIRECT_COPY is legal only
when ordered roles and their `PhysicalRepresentation` values—dtypes/shapes,
axis order/mappings, padding, permutation, and storage encoding—are equal and a
`DirectCopyCapabilityProof` authenticates both representation and placement
digests, the exact adjacent stage pair, route/placement compatibility, and
transport capability fingerprint. Placements need not be equal: a validated
NCCL route can copy the same representation across ranks. Logical
format or dtype equality alone is never proof, and the planner cannot skip the
load API to compare wire directly with derived runtime storage. Validate the
full plan before NCCL groups are created. Wire metadata carries graph instance
ID, semantic graph path, semantic ID, component role, dtype, logical/physical
shapes, axes, placement, owner, layout, transform, and plan ID; it never
encodes a fixed two-field `weight/weight_scale` assumption.

`build_canonical_plan_groups(context, ...)` first validates the complete bound
runtime context through Task 5's controller-local replay path and rejects any
bare or foreign intent/authority. It then derives rank-local ownership
from both runtime parallel topologies and validates every binding required by a
served member's canonical realization authority. It creates source-wire startup
plans for training-runtime frozen owners and every-version plans for
training-runtime mutable owners. Direct checkpoint owners instead create
checkpoint load/attestation plans, and backend-derived owners create the
advertised dependency plan without a wire payload. If a graph contains any
mutable served member, it is an every-version graph member, but its frozen
independent owners stay in the startup group and contribute only a
startup-precondition digest. An all-frozen source graph has only a startup
plan. An alias member references the qualified canonical source plan instead of
producing a duplicate source plan, then binds every separately realized
destination load owner and finalizer group.

Then compute a transitive cadence closure over destination physical-owner and
finalizer groups. For a mixed group, stage frozen canonical contributors once
in a verified persistent destination cache and refresh only mutable contributors
on each version. The update plan composes cached frozen and fresh mutable inputs
and finalizes that owner once. Require an adapter capability for native partial
preservation or split/repack from canonical components; otherwise reject the
plan before communication. The cache identity covers contributor/content
digests, realized owner/finalizer, layout, storage generation, topology, and
capability fingerprint and is part of startup/refit plan identity. Rebind,
covered-input change, explicit invalidation, or poison invalidates it; ordinary
version advance does not.

The builder rejects an owning-rank binding gap, accepts derived non-owner
absence, and excludes training-only graphs and directly owned checkpoint bodies
from source-wire load. A checkpoint-served graph's cross-graph canonical alias still
inherits the canonical training-runtime owner's source cadence.
The static startup plan group hashes ordered source-owner startup plans,
checkpoint load/attestation plans, each checkpoint graph's expected immutable
evidence and exact bound component/domain, load-operation, finalizer-group,
rank, and fence receipt sets, immutable-contributor cache keys, alias mappings,
and synchronized-replica fence requirements into `startup_plan_group_id`.
Changing any checkpoint consumption or receipt obligation therefore changes
the static and runtime startup identities. The static refit plan group
hashes ordered graph-member records, unique mutable-owner plan IDs, mixed-owner
composition/finalizer plans, alias mappings, active synchronized-replica fence
requirements, and the required startup/cache precondition identity into
`refit_plan_group_id`. Neither static plan is rebuilt per update. Task 11
derives the one-shot startup transaction identity from
`startup_plan_group_id`, the initial source version, the explicit initial
target generation version, and exact live fence set; it cannot include the
successful startup receipt digest that it has not produced yet. It derives
each update `transaction_group_id` from `refit_plan_group_id`, the
successful startup/cache precondition digest, source weight version, target
generation version, and exact live fence set. Checkpoint evidence remains
serving context, not a source transaction member.

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

### Task 8: Public, Versioned vLLM Precision Adapters and Checkpoint Attestation

**Files:**
- Create: `nemo_rl/models/generation/vllm/precision_adapter/__init__.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/base.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/registry.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/v0251.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/v0280.py`
- Create: `nemo_rl/models/generation/vllm/precision_adapter/mxfp8.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:550-700`
- Modify: `nemo_rl/models/generation/vllm/quantization/fp8.py:58-300`
- Test: `tests/unit/models/generation/test_checkpoint_evidence.py`
- Test: `tests/unit/models/generation/test_vllm_precision_adapter.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes before construction: serialized compiled rollout selection and actual `vllm.__version__`. Consumes after construction: the controller-validated Task 5 full context at the endpoint boundary and only its Task 7 `plan_bound` worker projection inside vLLM, including already-bound startup/every-version realization requests and common plan records. It never receives raw Phase 2 request/results or reconstructs context from intents.
- Produces: generic frozen `RealizedCheckpointEvidence`, `CheckpointDestinationLoadReceipt`, `CheckpointDestinationFinalizeReceipt`, complete `CheckpointLoadReceipt`, `DestinationCheckpointAttestor`, `verify_realized_checkpoint_evidence()`, and `verify_checkpoint_load_receipt()` in `nemo_rl.models.generation.interfaces`; frozen `VllmCapabilityProbes`; `VllmEndpointAdapter` with `capabilities()`, `configure_engine_kwargs()`, `describe_runtime_parallel_topology()`, `bind_realized_storage()`, `attest_checkpoint_realization()`, startup/update prepare/finalize methods, `load_component_batch()`, and `poison()`; `select_vllm_endpoint_adapter(version: str, probes: VllmCapabilityProbes) -> VllmEndpointAdapter`. Realized binding returns Task 7's `BoundDestinationPlans`; capability, immutable-contributor-cache, and placement fingerprints enter plan assembly only after realization. vLLM implements the attestor contract for the initial release; a backend-neutral fake proves the protocol remains extensible without admitting another production backend. `DestinationStartupReady` and `DestinationCommitReady` are adapter-local fenced proofs, not Task 11 worker results; Task 12 validates and converts them.

- [ ] **Step 1: Write failing registry and isolation tests**

```python
@pytest.mark.parametrize(("version", "adapter_id"), [("0.25.1", "vllm-0.25.1"), ("0.28.0", "vllm-0.28.0")])
def test_exact_supported_version_selects_dedicated_adapter(version: str, adapter_id: str) -> None:
    assert select_vllm_endpoint_adapter(version, complete_probes()).adapter_id == adapter_id

def test_unknown_or_incomplete_vllm_fails_before_model_construction() -> None:
    with pytest.raises(UnsupportedVllmEndpointError, match="capability"):
        select_vllm_endpoint_adapter("0.29.0", incomplete_probes())
```

Add a test that importing the registry without vLLM installed succeeds, that selecting 0.28 never imports 0.25-only modules, and that two engines with different plans do not share process-global quantization state. Reject a bare intent group, Phase-2-only projection, altered `runtime_context_id`, foreign plan-group ID, or adapter-authority fingerprint before storage binding; recursively prove the accepted projection contains no controller request/result/trusted contributor evidence. Verify that a training-only graph creates no vLLM realization request, while startup-only and every-version owners both expose realized storage and placement for Task 7. A checkpoint-served graph uses its pinned native load context rather than a source-load binding, but serving remains closed until its `CheckpointLoadReceipt` contains `RealizedCheckpointEvidence` matching every field of `ImmutableAuxiliaryEvidence` and exactly accounts for the bound checkpoint component/domain, destination load-operation, finalizer-group, rank, and completion-fence sets. Add fatal stale-tag, stale-cache/path, resolved-revision, checkpoint-content, model-config, semantic-domain, evidence-source, missing/extra/duplicate component domain, load operation, finalizer, rank, covered-member digest, and incomplete-fence tests. Run the same generic receipt verifier against a backend-neutral fake static-drafter attestor so the contract is not vLLM-specific.

- [ ] **Step 2: Run adapter tests and observe RED**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py --vllm-only`

Expected: missing precision-adapter package.

- [ ] **Step 3: Implement lazy version modules and NeMo quantization registration**

```python
@dataclass(frozen=True, slots=True)
class RealizedCheckpointEvidence:
    graph_instance_id: str
    model_identity: str
    resolved_checkpoint_revision: str
    checkpoint_content_digest: str
    model_config_digest: str
    semantic_domain_digest: str
    evidence_source: EvidenceSource

@dataclass(frozen=True, slots=True)
class CheckpointDestinationLoadReceipt:
    rank: int
    load_operation_id: str
    covered_physical_owner_member_digest: str

@dataclass(frozen=True, slots=True)
class CheckpointDestinationFinalizeReceipt:
    rank: int
    finalizer_group_id: str
    covered_load_owner_member_digest: str
    completion_fence_id: str

@dataclass(frozen=True, slots=True)
class CheckpointLoadReceipt:
    evidence: RealizedCheckpointEvidence
    consumed_component_domain_digest: str
    load_receipts: tuple[CheckpointDestinationLoadReceipt, ...]
    finalizer_receipts: tuple[CheckpointDestinationFinalizeReceipt, ...]
    engine_completion_fence_id: str

class DestinationCheckpointAttestor(Protocol):
    def attest_checkpoint_realization(
        self, graph_instance_id: str
    ) -> CheckpointLoadReceipt: ...

def verify_realized_checkpoint_evidence(
    expected: ImmutableAuxiliaryEvidence,
    realized: RealizedCheckpointEvidence,
) -> None: ...

def verify_checkpoint_load_receipt(
    expected: ImmutableAuxiliaryEvidence,
    bound: BoundDestinationPlans,
    receipt: CheckpointLoadReceipt,
) -> None: ...

class VllmEndpointAdapter(DestinationCheckpointAttestor, Protocol):
    adapter_id: str
    def configure_engine_kwargs(self, selection: CompiledPrecisionSelectionGroup, kwargs: dict[str, object]) -> None: ...
    def describe_runtime_parallel_topology(self) -> DestinationRuntimeParallelTopology: ...
    def bind_realized_storage(self, projection: BoundSemanticPrecisionWorkerProjection, model: object) -> BoundDestinationPlans: ...
    def attest_checkpoint_realization(self, graph_instance_id: str) -> CheckpointLoadReceipt: ...
    def prepare_startup_load(self, startup_id: str) -> None: ...
    def finalize_startup_load(self, startup_id: str) -> DestinationStartupReady: ...
    def prepare_transaction(self, transaction_id: str) -> None: ...
    def load_component_batch(self, batch: BoundComponentBatch) -> None: ...
    def finalize_transaction(self, transaction_id: str) -> DestinationCommitReady: ...
    def poison(self, reason: DestinationPoisonReason) -> None: ...
```

Register a NeMo MXFP8 quantization config through vLLM's public quantization registry and pass it through normal engine construction. Replace MXFP8 `unittest.mock.patch` installation and global `FP8State` dependence with adapter-owned method instances and worker-extension state. Each version module owns its version-specific imports and public capability probes.

After each native checkpoint load, construct the attestation from the resolved
artifact actually opened, not the requested model string or cache key. The
generic serving-gate verifier compares graph/model identity, immutable resolved
revision, content/configuration/semantic-domain digests, and typed evidence
source field-for-field, then requires an exact non-empty partition of every
bound checkpoint component/domain across the reported load operations and
finalizer groups with complete rank and fence coverage. The receipt is built
from loader observations or normalized native loader reports, never by copying
the expected plan. A stale tag, local cache entry, path, partial/duplicate
consumption, ignored loader return, or any mismatch poisons construction and
fails the launcher. Every future destination backend or static external drafter
must implement this generic proof before the capability gate admits it, even
when its loading mechanics differ.

- [ ] **Step 4: Run adapter and existing FP8 regression tests**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_fp8_quantization.py tests/unit/models/generation/test_vllm_fp8_hf_overrides.py --vllm-only`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/precision_adapter`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml`

Expected: all commands pass under the pinned 0.25.1 environment. Repeat the same conformance file in the pinned 0.28.0 environment and require pass before marking this task complete.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml
git commit -s -m "refactor(vllm): isolate public precision adapters"
```

### Task 9: Mixed BF16/MXFP8 Loading for vLLM's Internal FlashInfer-TRTLLM Runtime

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
- Consumes: the Task 7 `plan_bound` worker projection plus startup and every-version members of `BoundDestinationPlans` carrying its exact `runtime_context_id` and plan-group IDs; owners independently request BF16 logical load, BF16→MXFP8 quantization, or compatible native-MXFP8 component copy. Bare plans or reconstructed intents are not accepted.
- Produces: cadence-preserving batched owner loads, canonical staging lifetime tracking, dirty-owner sets, exactly-once destination finalization, adapter-local `DestinationStartupReady` for startup plans, and adapter-local `DestinationCommitReady` for every-version plans only after completion fences. These proofs never flow directly to the transaction supervisor.

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

Add literal padding cases: Lightning TP2 `928→1024` and `2688→3072`, Super TP4 `672→768`, Ultra TP16 `320→384`, Qwen3 TP4 `192→256`, Qwen3.5 TP8 `64→128`. Reject a missing projection, Phase-2-only marker, foreign `runtime_context_id`, or altered plan-group identity before any staging allocation. Cover gated/non-gated W13/W31, grouped and split sources, zero-value/unit-scale padding, scale flatten/interleave, native MXFP8 component order, A→B→C repeated refits, finalizer failure poisoning, and no commit after partial load. Run the same owner-dispatched load/finalize primitives for an independent frozen startup-only owner, assert it becomes startup-ready before serving, and assert later every-version loads never dirty or finalize it again. Separately cover Task 7's mixed-cadence fused group: immutable contributors remain in the verified destination cache without wire retransfers while the shared physical owner is composed and finalized once for each mutable update.

- [ ] **Step 2: Run mixed refit tests and observe RED**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_mixed_precision_refit.py tests/unit/models/generation/test_nccl_reshard_backend.py -k 'mixed or padding or grouped' --vllm-only`

Expected: BF16 boundary owners take the dtype-equality direct path or grouped MXFP8 input is rejected.

- [ ] **Step 3: Implement owner-dispatched loading and exactly-once finalization**

```python
def load_owner(self, owner: BoundPhysicalOwner, components: Mapping[ComponentRole, torch.Tensor]) -> None:
    if owner.transform is TransformLocus.NONE:
        _copy_compatible_components(owner, components)
    elif owner.transform is TransformLocus.DESTINATION:
        _quantize_bf16_to_mxfp8(owner, components[LOGICAL_VALUES])
    elif owner.transform is TransformLocus.DESTINATION_NATIVE_LOADER:
        _load_logical_bf16_through_vllm(owner, components[LOGICAL_VALUES])
    else:
        raise UnsupportedTransformError(owner.transform)
    self._dirty_owner_ids.add(owner.owner_id)
```

Use the logical descriptor together with the adjacent-stage realized physical
representations and route proof, never dtype-only dispatch. Keep logical staging alive through deferred native reload. Finalization pads/permutates/shuffles only dirty owners, records one completion event per batch, clears mutable canonical scratch after the fence, and raises if an owner is finalized twice within its cadence execution. Independent startup-finalized owners are sealed with their digest and rejected from later every-version batches. A mixed-cadence group instead retains verified immutable canonical inputs, accepts only its mutable inputs from the wire, and invokes its advertised preserve-or-repack composition/finalizer once per update.

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
- Consumes: the complete controller-local Task 5 bound runtime context, actual TE/Megatron parameters, and Task 7's common bound-plan and `SourceVersionFenceRequirement` records carrying the same `runtime_context_id`; it never accepts reconstructed graph intents/source aliases.
- Produces: frozen `RealizedSourceParameterInventory` with runtime tensor accessors, source runtime parallel topology, Task 7's `BoundSourcePlans`, `bind_mxfp8_source(context, inventory, bound_plans) -> BoundSourcePlans`, exact live `SourceVersionFence` proofs for required synchronized replicas, startup exports for source-proven frozen owners, and every-version exports for mutable owners. This runtime inventory is distinct from Task 4A's partitioned, metadata-only `SourceDiscoveryInventory`; it neither replaces producer completeness receipts nor reclassifies native names. Native-MXFP8 owners export ordered direct `values`/`block_scales`; BF16 owners export logical BF16.

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

Add tests for grouped expert views with gradients, forged dtype metadata, mismatched scale geometry, disabled FP8 export, storage alias partitioning, synchronization before export, lifecycle-based inclusion of a served mutable MTP/drafter, exclusion of a mutable training-only auxiliary from source load, alias-owner de-duplication, and direct native component compatibility failure. Reject bare intents, a foreign bound plan/context ID, and replaced context authority before tensor access. Verify that a frozen source-served owner exports exactly once through the startup plan, a mutable owner exports only through the every-version plan, and a mixed graph returns both without duplicating either owner. For synchronized replicas, prove optimizer update happens before replica synchronization, the matching rank-local `SOURCE_VERSION_READY` fence happens after synchronization, and export happens only after Task 7 validates the complete live fence set. Reject pre-update/stale/wrong-group/wrong-topology/wrong-version/wrong-rank/duplicate fences. Identical-storage aliases require no live replica fence. Exactly in-scope served canonical aliases resolving to a training source require fences, including aliases inside checkpoint-served graphs; training-only, direct checkpoint-body, non-training-authority, and destination-local `out_of_scope` members do not. Frozen synchronized aliases bind their initial source version and live fence set into the startup plan and startup-precondition digest.

- [ ] **Step 2: Run source tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py`

Expected: missing source adapter.

- [ ] **Step 3: Implement source binding from the canonical graph intents**

```python
def bind_mxfp8_source(
    context: BoundSemanticPrecisionRuntimeContext,
    inventory: RealizedSourceParameterInventory,
    bound_plans: CanonicalPlanGroupSet,
) -> BoundSourcePlans:
    require_same_runtime_context(context, bound_plans)
    startup = _bind_semantic_sources(context.intents.startup_source_items, inventory)
    every_version = _bind_semantic_sources(context.intents.every_version_source_items, inventory)
    _validate_exact_cadence_coverage(context.intents, startup, every_version)
    _validate_component_geometry((*startup, *every_version))
    return BoundSourcePlans(
        runtime_context_id=context.runtime_context_id,
        intent_group_id=context.intents.intent_group_id,
        startup=startup,
        every_version=every_version,
    )
```

For each refit source version, fence optimizer/TE writes, synchronize every
required replica group, then emit the exact rank-local `SourceVersionFence`
set and validate it against the bound plan before reading values or scales.
This update → synchronize → fence → export order is mandatory; a topology
evidence digest alone is never a live completion proof. Reuse stable component
views when safe; copy only when storage reuse or asynchronous transfer requires
lifetime extension. Do not identify MXFP8 storage solely from
`torch.float8_e4m3fn`.

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
- Modify: `nemo_rl/weight_sync/refit_supervisor.py`
- Create: `nemo_rl/weight_sync/transaction.py`
- Modify: `nemo_rl/distributed/refit_watchdog.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Test: `tests/unit/weight_sync/test_refit_supervisor.py`
- Create: `tests/unit/weight_sync/test_refit_transaction.py`
- Test: `tests/unit/distributed/test_refit_watchdog.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes controller-side: the exact Task 5
  `BoundSemanticPrecisionRuntimeContext`, validated
  `CanonicalStartupLoadPlanGroup` and `CanonicalRefitPlanGroup` carrying that
  same `runtime_context_id`, expected checkpoint evidence and bound checkpoint
  consumption sets, their exact source version, target version, active
  synchronized-replica fence requirement/live-proof digest, group-phase,
  source-send batch, destination-receive batch, checkpoint-receipt, and
  destination-finalizer rank acknowledgement sets, source and destination
  `ray.ObjectRef` sets, plus registered abort/poison callbacks. The driver-only
  `bind_refit_transaction_authority(context, startup_plans, refit_plans,
  projection, signer, *, expected_runtime_owner_id) -> BoundRefitTransactionAuthority`
  reruns Task 5's full
  controller-local context/authority replay validation and rejects bare intents,
  a projection in place of the context, or a plan carrying a foreign context
  ID. It is the only minting API. Workers receive only its authenticated safe
  template/projection and already-bound execution records; the aggregate
  discovery request/results, adapter objects, and private signing key never
  enter a transaction wire payload.
- Produces: `RefitPhase`, `RefitExecutionKind`, `RefitResultStatus`,
  `TransferDirection`, `RefitLifecycleOperationKind`, `RefitActivationMode`,
  `RefitCapacityKind`, `RefitCapacityParticipantRole`, immutable
  `BoundRefitTransactionAuthority` plus its
  strictly decoded `BoundRefitTransactionAuthorityWire`, immutable
  `BoundRefitResultAuthorization`, authenticated
  `BoundRefitExecutionDeadlineWire`/`BoundRefitCancellationWire`, and nonempty
  ordered `BoundRefitWorkerExecutionBundle`, immutable discriminated
  `BoundRefitDirectDispatchBranch | BoundRefitBroadcastDispatchBranch |
  BoundRefitDispatchLeaf` values aliased as
  `BoundRefitWorkerDispatchNode`, and a root
  per-operation `BoundRefitFacadeDispatchManifest`, immutable dependency-neutral
  `BoundRefitBroadcastGroupEnvelope`, immutable
  `BoundRefitDispatchStep` values containing exactly one ready manifest or one
  controller-only deferred-manifest authority, plus a nonempty ordered
  `BoundRefitPrecommitDispatchSchedule` of exact operation-ID steps
  and a distinct one-shot `BoundRefitActivationDispatchSchedule` plus the
  controller-local `RefitActivationModeDispatcher` protocol,
  minted only from that verified authority plus fresh
  transaction/version/fence inputs and the still-live original absolute
  transaction deadline/cancellation authority, the discriminated
  `GroupPhaseResult | LifecycleOperationResult | CapacityDiscoveryResult |
  RendezvousDiscoveryResult | PreparedMetadataOperationResult | TransferWorkerResult | DestinationLoadResult |
  DestinationFinalizeResult | CheckpointReceiptResult` union named
  `RefitWorkerResult`, non-empty `RefitWorkerResultBatch`, `RefitFailure`,
  one-shot `StartupLoadTransaction`, every-version `RefitTransaction`,
  `supervise_refit_futures()`, and bounded abort. The authority commits to the
  canonical nonempty expected runtime-owner ID, exact static plan wires/digests,
  plan-bound projection, `runtime_context_id`, adapter
  authority fingerprint, and a per-run public verifier; its controller-local
  signing key and full context are excluded from serialization. Actor-side
  transaction construction accepts only a verified authority plus fresh exact
  source/target versions and live fence proofs. Every transaction identity and
  typed result commits to the exact authority/template ID,
  `runtime_context_id`, and plan-bound projection digest. Startup publishes a
  precondition digest before serving; every-version execution contains mutable
  owners only. Verified checkpoint receipt, destination load-owner completion,
  finalizer-group completion, and engine transaction-envelope completion are
  separate proof sets.
  Task 11's only modification to `weight_sync/interfaces.py` is to define these
  dependency-neutral enums, frozen result/authorization/dispatch values, and
  result aliases. It snapshots every existing abstract synchronizer/direct-
  handle method signature unchanged. Task 12 alone changes the abstract call
  signatures atomically with all concrete overrides.
  `GroupPhaseResult` remains coordinator-owned and represents only global
  PREPARE/COMMIT/ABORT; it cannot acknowledge a worker sub-operation. Every
  setup/refit lifecycle RPC instead owes a `LifecycleOperationResult` carrying
  the authority-bound canonical `operation_kind` and `operation_id`, including
  `INIT_COMMUNICATOR`, `REBUILD_COMMUNICATOR`, `PREPARE_METADATA`, `PAUSE`,
  `DISCOVER_CAPACITY`, `DISCOVER_RENDEZVOUS`, `SOURCE_PREPARE`,
  `PREPARE_DESTINATION_WEIGHT_MEMORY`, `SOURCE_RESTORE`, `PREPARE_KV_MEMORY`, `BEGIN_UPDATE`,
  `INVALIDATE`, `END_UPDATE`, `PUBLISH_SERVED_VERSION`, `RESUME`, and
  the mode-specific `OPEN_COLLECTION` or `OPEN_ROLLOUT`. A fused backend may
  execute several operations atomically, but its ordered result batch must
  consume the matching ordered authorizations one-to-one; a global COMMIT proof
  can never substitute for those ACKs. Neither `PUBLISH_SERVED_VERSION` nor
  `RESUME` nor either mode-specific open operation is part of the proof set that earns COMMIT or
  STARTUP_READY. After durable COMMIT—or after STARTUP_READY only when the
  canonical every-version group is empty—the controller mints a one-shot
  ordered activation schedule under the still-live original transaction
  deadline/cancellation authority. Its exact `RefitActivationMode` selects one
  validated shape: async uses publication, backend resume, then one
  `OPEN_COLLECTION` collector leaf; Single Controller uses publication,
  backend resume, then one `OPEN_ROLLOUT` leaf that sets `_rollout_permitted`;
  synchronous driver GRPO has no collector actor and therefore uses the
  canonical two-step publication/resume variant with no fabricated open ACK.
  The synchronous transaction returns serving permission only after every
  resume ACK validates. In a mixed startup+mutable graph, STARTUP_READY leaves the gate closed
  and mints no activation; the initial refit must COMMIT first. Generation and
  collection stay gated until every exact ACTIVATE ACK succeeds; any activation
  failure is terminal even though the coordinator record is already durable.
  Activation never creates a second timeout: its waits consume
  `original_absolute_deadline.remaining_s(...)`; if COMMIT/STARTUP_READY leaves
  no time, activation fails immediately and cleanup uses only its separately
  declared cleanup budget.

The framework-light supervisor foundation lives in
`weight_sync/refit_supervisor.py`. It requires both a positive finite shared
timeout and an explicit result normalizer, places consumer futures first in one
producer/consumer wait set, validates every `ray.wait` partition exactly, and
checks the monotonic deadline through result normalization. Each blocking-ready
result is merged with at most one zero-timeout ready-wave drain before anything
is resolved, then processed in linear consumer-first canonical order. Ray waits
fetch ready values locally, and each `ray.get` receives the remaining shared
timeout. Result normalizers are bounded nonblocking local validators and may not
perform I/O or distributed synchronization. Current producer RPCs that return
exact `None` are supported only by a named migration normalizer that converts
`None` or exact `True` to canonical exact `True`; consumers already require
exact `True`. Typed transaction workers never use that migration contract.
For a real plan with work on only one side, either producer or consumer refs may
be empty, but total refs must be positive and the declared per-side summary
counts must be exact nonnegative integers matching the submitted refs. Empty
producer and consumer sets together are rejected. Exact plan-derived
source/destination acknowledgement sets are validated independently of those
future-list partitions. The `source_refs`/`destination_refs` labels control only
consumer-first wait priority and diagnostics; each item inside a returned typed
batch is classified and credited solely by its authenticated result
discriminant/authorization. Thus one source-side nested worker future may return
both source and nested destination/load/finalize proofs while
`destination_refs=()`, but every expected destination proof remains mandatory
and cannot be vacuously waived by the empty ref list. The declared per-side ref
counts still match submitted refs, not the heterogeneous proof cardinalities.
Those acknowledgement sets, not an artificial peer ref, determine completeness.
The supervisor owns no communicator cancellation, cache mutation, version
commit, or recovery policy. Its structured wrapper preserves the original
exception as `__cause__`; integrations with an existing `RefitAborted` or
`RayActorError` recovery branch must classify that cause chain explicitly
rather than blindly replacing current waits.

- [ ] **Step 1: Write failing first-failure and poison tests**

```python
def test_receiver_failure_is_observed_while_sender_is_still_pending(fake_ray: FakeRay) -> None:
    sender = fake_ray.pending("train-rank-0")
    receiver = fake_ray.failed("gen-rank-1", RuntimeError("layout conversion failed"))
    transaction = refit_transaction_from_bound_authority(fake_ray)
    with pytest.raises(RefitTransactionError, match="gen-rank-1.*layout conversion failed"):
        transaction.run(source_refs=(sender,), destination_refs=(receiver,))
    assert fake_ray.elapsed < 1.0

@pytest.mark.parametrize(
    "result",
    [
        None,
        False,
        {},
        GroupPhaseResult(
            status=RefitResultStatus.FAILED,
            rank=2,
            target_id="coordinator:2",
            result_authorization_id="auth:abort:2",
            phase=RefitPhase.ABORT,
            execution_kind=RefitExecutionKind.UPDATE,
            runtime_context_id="runtime-context",
            transaction_authority_id="authority",
            plan_group_id="refit-plan-group",
            transaction_group_id="tx-7",
            source_version=6,
            target_version=7,
            source_fence_set_digest="fences-6",
            detail="bad",
        ),
    ],
)
def test_non_success_result_never_commits(result: object) -> None:
    with pytest.raises(RefitTransactionError):
        validate_worker_result(result)
```

Add tests for first and later component failure, finalize/commit failure, timeout, silent peer, abort callback failure, communicator-abort fallback to worker termination, original cause/rank preservation, no partial version commit, watchdog remaining armed until transaction resolution, and bounded teardown.
Add supervisor cases for producer-only, consumer-only, and checkpoint-only
startup work; each valid one-sided set completes from its exact expected ACKs.
Reject empty-both, negative/non-integer summary counts, or a count/ref mismatch.
An all-mutable graph has no startup transaction/supervisor call and instead
derives the canonical no-startup precondition digest from its bound authority;
later every-version execution must require that exact digest.
Add exact-context tests that reject a missing full context, bare selection/intents,
a Phase-2-only projection, a context from another bootstrap, changed adapter
authority, and a plan/result/projection with a different `runtime_context_id`
before authority minting, PREPARE, or any Ray submission. A bare/forged
projection, copied template with one changed field, wrong verifier, reused
signature from another run, or direct public construction cannot mint or decode
a `BoundRefitTransactionAuthorityWire`. The valid mint uses the same local
context object by identity; direct pickle/cloudpickle of the full context still
fails, while the authenticated authority round-trips with no request/results,
trusted contributor authority, adapter objects, or signing key. From one
verified authority, instantiate one startup transaction and multiple later
per-version transactions with distinct valid version/fence inputs; reject stale
or mismatched fences/results before commit.
Mint worker execution bundles only inside those transactions. Unit-test a
nonempty ordered tuple of heterogeneous `BoundRefitResultAuthorization` records
covering checkpoint receipt, source and destination transfer, two load
operations, and two finalizer groups. The bundle validator maps one returned
result to each authorization exactly once and rejects empty authorization sets,
reordering, missing/extra/duplicate results, cross-rank or cross-phase reuse,
and direct/worker-side authorization construction. Give a collector publisher
and an engine leaf the same numeric rank zero; swapped results/batches must fail
their globally unique target/result-authorization IDs before credit. Apply that
identity check to every result variant, not only lifecycle results. Its safe wire retains only
authenticated bound identities and never the full context or signing key.
Every bundle also carries the exact authenticated
`BoundRefitExecutionDeadlineWire` and `BoundRefitCancellationWire` for its
transaction. The deadline wire binds the transaction deadline ID, original
positive finite timeout, operation release sequence, and controller-computed
remaining budget at release. The cancellation wire binds that same deadline/
transaction ID to a nested pending cancellation ObjectRef whose identity is
covered by the authority digest; Ray must not dereference it while serializing
the outer call. A facade, direct dispatcher, broadcast envelope, nested policy
helper, and final leaf all receive/validate the same IDs. A dispatcher may
shorten its local remaining budget but may neither replace the cancellation ref
nor mint/reset/extend a budget. Every nested `ray.wait`, `ray.get`, future,
lock, HTTP/ZMQ operation, and awaitable races work against that cancellation
ref and uses at most the received remaining duration. The controller resolves
the latch at the one original absolute deadline even if an outer actor is
silent, so routing delay cannot create another full timeout. Reject a missing,
foreign, already-resolved-as-success, mismatched-ID, nonfinite, negative, or
increased child budget before nested work, and prove cancellation while a leaf
future is pending prevents late mutation/ACK.
For a multi-DP/multi-endpoint and TP/PP nested fan-out, mint one
per-operation `BoundRefitFacadeDispatchManifest` whose ordered root nodes match the
controller's exact canonical outer-target identities and cardinality. Each
node is an exact discriminated sum. `BoundRefitDirectDispatchBranch` has one
target identity, the exact deadline/cancellation wires, a nonempty ordered
child tuple, and no local execution bundle/envelope.
`BoundRefitBroadcastDispatchBranch` has one target identity and exactly one
controller-minted authenticated group envelope carrying those same wires, with
no local execution bundle or separately mutable child tuple.
`BoundRefitDispatchLeaf` has one rank/endpoint-bound nonempty execution bundle
containing those wires and no children/envelope.
A branch can route but cannot acknowledge an operation, and a leaf can execute
but cannot dispatch. Any branch-with-bundle, leaf-with-child, empty branch,
unknown branch mode, or ambiguous structural lookalike fails strict decode.
Distinct outer RPC targets never receive one another's subtree. Inside a
runtime that exposes only public group broadcast, the controller pre-mints and
signs the immediate child nodes into one immutable authenticated
`BoundRefitBroadcastGroupEnvelope`; confidentiality is not required, but each
leaf can select/decode exactly one entry matching its immutable runtime target
identity and cannot consume or acknowledge a sibling entry. An ancestor
dispatcher may route authenticated descendant wires but cannot consume them
because every bundle is bound to its exact target identity/rank.
Reject a missing, extra, duplicate, reordered, or aliased target at any depth,
reuse of one bundle/node for two targets, cycles, or a tree from a foreign
transaction before any facade/collective RPC is submitted. The root, node, and
group-envelope wires retain no minting authority; a direct RPC sends only the
matched node, while an unavoidable public group broadcast sends the exact
group envelope and each leaf derives only its singular bundle.
For a multi-wave flow, mint a `BoundRefitPrecommitDispatchSchedule` whose ordered
`BoundRefitDispatchStep`s bind one exact lifecycle/transfer `operation_kind`,
nonempty `dispatch_operation_id`, and exactly one already-resolved manifest or
controller-only deferred-manifest authority. Target identity uniqueness is required
within one operation step and keyed as `(dispatch_operation_id, target)` across
the schedule, so the same target may legally recur for communicator-init,
metadata, process-group-init, pause, begin, update, and end only with a
fresh operation-bound node/bundle each time. Reject duplicate/omitted/reordered
steps, reuse of an operation ID or node across steps, consuming a later step
early, under-consumption, and any leftover precommit entry before COMMIT.
`PUBLISH_SERVED_VERSION`, `RESUME`, `OPEN_COLLECTION`, and `OPEN_ROLLOUT` are forbidden in that
schedule. Only after durable COMMIT—or STARTUP_READY for a startup-only graph—
may the controller mint a distinct
one-shot `BoundRefitActivationDispatchSchedule` using the exact mode shape and
the original transaction deadline/cancellation authority: all modes publish
the exact destination/owner version and resume all backends; async alone then
uses `OPEN_COLLECTION`, Single Controller alone then uses `OPEN_ROLLOUT`, and
synchronous driver mode ends after resume with no open step or synthetic ACK.
It is never handed to the synchronizer's precommit sync method. Task 11's
signature snapshot test proves adding these values does not
yet alter an abstract synchronizer method; that happens in Task 12.
`RefitActivationModeDispatcher` is a Task 11 controller-local Protocol in
`weight_sync/transaction.py`, not an
ambient variable: it exposes its exact activation mode and runtime-owner ID and
dispatches one already-authorized step with the schedule's deadline/cancellation
wires. The authority and schedule digest bind the same canonical
`expected_runtime_owner_id`; `run_to_serving_permission()` recomputes that
digest and rejects a missing dispatcher, mode mismatch, dispatcher owner ID
different from the bound value, mutated owner/digest, extra capability, foreign
deadline/cancellation, or an untyped result before the first dispatch. Task 11
uses explicit fake dispatchers for all three
shapes; concrete construction and runtime ownership land atomically in Task 12.
Capacity/NIXL metadata-dependent steps are never guessed up front. A
`DISCOVER_CAPACITY` step returns one authenticated positive target-bound
`CapacityDiscoveryResult` per expected participant, including an exact
`SOURCE`/`DESTINATION` participant role. For
`CHECKPOINT_ENGINE_TOTAL_MEMORY`, the required participant set is the union of
every source and every destination rank in the bound checkpoint-engine plan;
neither side may be sampled or inferred from the other. After exact aggregate
validation, the controller derives the canonical minimum across that full union and its capacity digest, then
uses the deferred authority to mint the next communicator manifest. A
`PREPARE_METADATA` step similarly returns immutable
`PreparedRefitMetadataArtifact` values inside
`PreparedMetadataOperationResult`; only the controller validates exact
rank/cardinality/context/transaction/operation/digest coverage, constructs the
canonical metadata aggregate, and resolves the process-group-init manifest
bound to that aggregate digest. Missing, extra, duplicate, malformed,
nonpositive, foreign, or timed-out artifacts poison the transaction; a worker
or synchronizer cannot mint or fill a deferred step.
For all-mutable initial initialization, startup-only initialization, and a later
communicator rebuild, require distinct `INIT_COMMUNICATOR`/
`REBUILD_COMMUNICATOR`, source prepare/restore, capacity discovery, metadata,
pause, begin, invalidate, end, and resume
operation IDs and reject a missing, reordered, replayed, or COMMIT-substituted
sub-operation. Add an end-to-end supervisor case with one source ref returning
an authorized mixed source + nested destination/load/finalize batch and
`destination_refs=()`; it commits only with all expected destination proofs,
and missing/forged destination proofs fail despite the empty destination ref
partition.
Add capacity/metadata fixtures with multiple source and destination ranks:
validate exact positive capacities and derive the canonical minimum across the
full role-tagged union, then validate immutable NIXL
metadata artifacts and bind their aggregate digest into the deferred process-
group manifest. Reject boolean/zero/negative capacity, malformed canonical
bytes/digest, missing/extra/duplicate/foreign rank or operation, timeout, and a
worker attempt to resolve the deferred authority. For checkpoint-engine
total-memory, separately omit, duplicate, and corrupt one source and one
destination result and prove neither a generation-only nor source-only minimum
can advance the schedule. Include IPC policy-free-memory and prove an explicitly
configured buffer skips capacity discovery entirely. For colocated IPC,
assert the exact hot-path order `SOURCE_PREPARE ->
PREPARE_DESTINATION_WEIGHT_MEMORY -> DISCOVER_CAPACITY -> transfer/finalize ->
SOURCE_RESTORE -> PREPARE_KV_MEMORY`; discovering before destination weight
memory is prepared or waking KV memory before transfer is fatal. Explicit size
skips only `DISCOVER_CAPACITY`, not the surrounding lifecycle. Add rendezvous fixtures
for collective and NCCL-reshard communicator rebuild: a valid
address/port attempt resolves the deferred init manifest; malformed address or
port, missing/duplicate/foreign attempt, collision retry, or a lone hung probe
uses the same transaction deadline and can never fall back to the construction
deadline or legacy unbounded wait. Add source lifecycle cases where
`SOURCE_PREPARE` or `SOURCE_RESTORE` fails or is silent; both share the one
transaction deadline/cancel authority, prevent COMMIT, and poison/clean partial
destination work.
Block COMMIT (and separately STARTUP_READY) while an in-flight generation
request watches the publication and backend resume spies: neither may be
submitted. After the coordinator state is durable, mint the activation schedule
and assert exact destination/owner `PUBLISH_SERVED_VERSION` results carrying
the target version complete before the RESUME step is released. Only after all
backend RESUME ACKs succeed may the selected mode's open step run: async calls
only `OPEN_COLLECTION`, Single Controller calls only `OPEN_ROLLOUT`, and
synchronous driver mode completes without either. A silent, malformed,
wrong-version, or failed publisher, backend-resume failure, or mode-specific
open failure is terminal, leaves collection/serving closed, and never relabels
an old-version request as new. Delay each open gate while a sibling backend
resume fails immediately and assert the open RPC is never submitted. Parameterize
all three exact schedule shapes and reject a missing/extra/wrong-mode open step.
Advance the precommit phases to just before the original absolute deadline and
prove activation receives only the remaining fraction rather than a fresh
`refit_timeout_s`; expiry resolves the shared cancellation ref, prevents every
later step, and enters bounded cleanup.
Reject a string phase, any phase inconsistent with its result variant, missing
plan-group/batch/direction/covered-graph-member/finalizer identity, and any
extra execution-unit identity on a group result. Require independent source-send and
destination-receive transfer acknowledgements for each expected execution batch,
and verify their direction, batch, static plan group, covered graph/plan-member
digest, transaction-group, and rank
against Task 7's acknowledgement sets. Reject empty or duplicate result batches
and malformed/mismatched destination startup or commit proofs before treating
their per-owner wrapper results as acknowledgements.

Add transaction-group tests using Task 7's main, MTP, and speculative-draft member records: every expected owning-rank FINALIZE acknowledgement is required, a served-draft-only failure prevents the main version from committing, and a stale served-draft target version is rejected. A mixed-cadence fused destination group owes exactly one finalizer acknowledgement per update even though only its mutable contributors appear in the repeated wire payload. Prove that mutable training-only graphs and static checkpoint drafter bodies are not every-version or trainer source-transfer members; static checkpoint bodies remain mandatory startup-attestation members, and non-owning PP ranks owe no drafter acknowledgement. A checkpoint-served drafter's cross-graph canonical aliases inherit their canonical source cadence; synchronized replicas also owe the exact live source-version fence set. Alias-only MTP members reuse one canonical source transfer but owe one finalizer acknowledgement per distinct realized physical destination owner; they reuse a destination acknowledgement only when Task 7 carries the endpoint's identical storage-owner/finalizer proof. A constructed group that omits a declared owning-rank binding is rejected before PREPARE.

Add startup-load tests proving that every startup owner binds, transfers, and
finalizes exactly once before the serving gate opens; its digest is required by
later refits; it is absent from every-version payload/acknowledgement sets; and
binding, transfer, finalize, timeout, or malformed-result failure is propagated
immediately, poisons partial destination state, leaves serving closed, and exits
the launcher non-zero. For each checkpoint-served direct body, independently
re-run `verify_checkpoint_load_receipt()` against expected immutable evidence
and bound consumption sets, require exactly one `CheckpointReceiptResult` per
expected graph/rank, and reject missing, extra, duplicate, altered, or digest-
mismatched receipts before `STARTUP_READY`.
A static checkpoint body is not a trainer source-transfer member, but its
verified checkpoint receipt is still a mandatory startup serving-gate proof.

- [ ] **Step 2: Run transaction tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_supervisor.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Expected: the transaction/typed-authority module, one-sided-set rules, and
context-bound result fields are missing or RED; the existing consumer-first
supervisor foundation is modified in place rather than replaced.

- [ ] **Step 3: Implement explicit PREPARE→READY→TRANSFER→FINALIZE→COMMIT/STARTUP_READY→ACTIVATE state transitions**

```python
class RefitPhase(StrEnum):
    PREPARE = "prepare"
    READY = "ready"
    TRANSFER = "transfer"
    LOAD = "load"
    FINALIZE = "finalize"
    CHECKPOINT_ATTEST = "checkpoint_attest"
    STARTUP_READY = "startup_ready"
    COMMIT = "commit"
    ACTIVATE = "activate"
    ABORT = "abort"

class RefitExecutionKind(StrEnum):
    STARTUP = "startup"
    UPDATE = "update"

class RefitActivationMode(StrEnum):
    ASYNC_COLLECTOR = "async_collector"
    SINGLE_CONTROLLER = "single_controller"
    SYNCHRONOUS_DRIVER = "synchronous_driver"

class RefitResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class TransferDirection(StrEnum):
    SOURCE_SEND = "source_send"
    DESTINATION_RECEIVE = "destination_receive"

class RefitCapacityKind(StrEnum):
    CHECKPOINT_ENGINE_TOTAL_MEMORY = "checkpoint_engine_total_memory"
    POLICY_FREE_MEMORY = "policy_free_memory"

class RefitCapacityParticipantRole(StrEnum):
    SOURCE = "source"
    DESTINATION = "destination"

class RefitLifecycleOperationKind(StrEnum):
    DISCOVER_CAPACITY = "discover_capacity"
    DISCOVER_RENDEZVOUS = "discover_rendezvous"
    SOURCE_PREPARE = "source_prepare"
    PREPARE_DESTINATION_WEIGHT_MEMORY = "prepare_destination_weight_memory"
    SOURCE_RESTORE = "source_restore"
    PREPARE_KV_MEMORY = "prepare_kv_memory"
    INIT_COMMUNICATOR = "init_communicator"
    REBUILD_COMMUNICATOR = "rebuild_communicator"
    PREPARE_METADATA = "prepare_metadata"
    PAUSE = "pause"
    BEGIN_UPDATE = "begin_update"
    INVALIDATE = "invalidate"
    END_UPDATE = "end_update"
    PUBLISH_SERVED_VERSION = "publish_served_version"
    RESUME = "resume"
    OPEN_COLLECTION = "open_collection"
    OPEN_ROLLOUT = "open_rollout"

@dataclass(frozen=True, slots=True)
class GroupPhaseResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class LifecycleOperationResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    operation_kind: RefitLifecycleOperationKind
    operation_id: str
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class CapacityDiscoveryResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    operation_kind: Literal[RefitLifecycleOperationKind.DISCOVER_CAPACITY]
    operation_id: str
    capacity_kind: RefitCapacityKind
    participant_role: RefitCapacityParticipantRole
    capacity_bytes: int
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class RendezvousDiscoveryResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    operation_kind: Literal[RefitLifecycleOperationKind.DISCOVER_RENDEZVOUS]
    operation_id: str
    attempt_index: int
    address: str
    port: int
    candidate_digest: str
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class PreparedRefitMetadataArtifact:
    schema_version: Literal[1]
    metadata_kind: str
    canonical_payload: bytes
    payload_digest: str

@dataclass(frozen=True, slots=True)
class PreparedMetadataOperationResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    operation_kind: Literal[RefitLifecycleOperationKind.PREPARE_METADATA]
    operation_id: str
    artifact: PreparedRefitMetadataArtifact
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class TransferWorkerResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    execution_batch_id: str
    direction: TransferDirection
    transaction_group_id: str
    covered_graph_member_digest: str
    covered_component_set_digest: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class DestinationLoadResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    destination_plan_id: str
    load_operation_id: str
    covered_graph_member_digest: str
    covered_physical_owner_member_digest: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    completion_fence_id: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class DestinationFinalizeResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    plan_group_id: str
    destination_plan_id: str
    finalizer_group_id: str
    covered_graph_member_digest: str
    covered_load_owner_member_digest: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    completion_fence_id: str
    detail: str | None = None

@dataclass(frozen=True, slots=True)
class CheckpointReceiptResult:
    status: RefitResultStatus
    rank: int
    target_id: str
    result_authorization_id: str
    phase: RefitPhase
    execution_kind: RefitExecutionKind
    runtime_context_id: str
    transaction_authority_id: str
    graph_instance_id: str
    plan_group_id: str
    transaction_group_id: str
    source_version: int
    target_version: int
    source_fence_set_digest: str
    receipt: CheckpointLoadReceipt
    receipt_digest: str
    detail: str | None = None

type RefitWorkerResult = (
    GroupPhaseResult
    | LifecycleOperationResult
    | CapacityDiscoveryResult
    | RendezvousDiscoveryResult
    | PreparedMetadataOperationResult
    | TransferWorkerResult
    | DestinationLoadResult
    | DestinationFinalizeResult
    | CheckpointReceiptResult
)
type RefitWorkerResultBatch = tuple[RefitWorkerResult, ...]

@dataclass(frozen=True, slots=True)
class BoundRefitExecutionDeadlineWire:
    deadline_id: str
    original_timeout_s: float
    operation_release_sequence: int
    remaining_timeout_s_at_release: float
    authority_binding_digest: str

@dataclass(frozen=True, slots=True)
class BoundRefitCancellationWire:
    deadline_id: str
    cancellation_id: str
    pending_cancellation_ref: ray.ObjectRef[object]
    authority_binding_digest: str

@dataclass(frozen=True, slots=True)
class BoundRefitWorkerExecutionBundle:
    result_authorizations: tuple[BoundRefitResultAuthorization, ...]
    execution_deadline: BoundRefitExecutionDeadlineWire
    cancellation: BoundRefitCancellationWire
    bundle_digest: str

class RefitActivationModeDispatcher(Protocol):
    @property
    def activation_mode(self) -> RefitActivationMode: ...

    @property
    def runtime_owner_id(self) -> str: ...

    def dispatch(
        self,
        step: BoundRefitDispatchStep,
        *,
        execution_deadline: BoundRefitExecutionDeadlineWire,
        cancellation: BoundRefitCancellationWire,
    ) -> tuple[ray.ObjectRef[RefitWorkerResultBatch], ...]: ...

@dataclass(frozen=True, slots=True)
class BoundRefitActivationDispatchSchedule:
    activation_mode: RefitActivationMode
    expected_runtime_owner_id: str
    original_execution_deadline: BoundRefitExecutionDeadlineWire
    cancellation: BoundRefitCancellationWire
    ordered_steps: tuple[BoundRefitDispatchStep, ...]
    authority_binding_digest: str

    def run_to_serving_permission(
        self, *, dispatcher: RefitActivationModeDispatcher
    ) -> None: ...

@dataclass(frozen=True, slots=True)
class BoundRefitDirectDispatchBranch:
    node_kind: Literal["direct_branch"]
    target_id: str
    execution_deadline: BoundRefitExecutionDeadlineWire
    cancellation: BoundRefitCancellationWire
    ordered_children: tuple["BoundRefitWorkerDispatchNode", ...]

@dataclass(frozen=True, slots=True)
class BoundRefitBroadcastDispatchBranch:
    node_kind: Literal["broadcast_branch"]
    target_id: str
    execution_deadline: BoundRefitExecutionDeadlineWire
    cancellation: BoundRefitCancellationWire
    group_envelope: "BoundRefitBroadcastGroupEnvelope"

@dataclass(frozen=True, slots=True)
class BoundRefitDispatchLeaf:
    node_kind: Literal["leaf"]
    target_id: str
    worker_execution_bundle: BoundRefitWorkerExecutionBundle

type BoundRefitDispatchBranch = (
    BoundRefitDirectDispatchBranch | BoundRefitBroadcastDispatchBranch
)
type BoundRefitWorkerDispatchNode = BoundRefitDispatchBranch | BoundRefitDispatchLeaf

@dataclass(frozen=True, slots=True)
class BoundRefitBroadcastGroupEnvelope:
    schema_version: Literal[1]
    runtime_context_id: str
    transaction_authority_id: str
    transaction_group_id: str
    dispatch_operation_id: str
    parent_target_id: str
    execution_deadline: BoundRefitExecutionDeadlineWire
    cancellation: BoundRefitCancellationWire
    ordered_child_target_ids: tuple[str, ...]
    ordered_child_node_wires: tuple[bytes, ...]
    ordered_child_node_digests: tuple[str, ...]
    envelope_digest: str
    authority_binding_digest: str
```

The three child tuples have the same positive length and are positionally
paired. Each target ID commits to the exact physical owner plus global/TP/PP
rank identity; each canonical node wire commits to one singular child node and
cannot contain the parent/root manifest. The transaction mint constructs the
envelope and its enclosing broadcast branch together before either reaches an
endpoint. The branch, envelope, every child node, and every leaf bundle must
carry the same deadline/cancellation IDs and exact nested ObjectRef identity;
a child may only reduce its remaining cap. It recomputes every
node/envelope/authority-binding digest, and the
already authenticated enclosing branch/schedule transitively commits those
exact canonical bytes; the envelope requires no fresh worker-side signature.
Strict decode verifies that enclosing authority commitment, order, cardinality, transaction/context/
operation identity, and round-trip canonical bytes before returning the frozen
type. A leaf must derive its target ID from its repo-owned runtime identity,
obtain exactly one positional match, decode only that node, and then validate
its singular bundle. Zero/multiple match and attempted sibling-node decode are
fatal. The envelope carries no secret, signing key, adapter object, or raw
source-discovery/trusted-contributor evidence.
There is no public endpoint/worker envelope constructor: an outer dispatcher
has only verifier authority, verifies and forwards the exact pre-minted
envelope, and cannot alter/re-sign children. Unit tests remove the driver's
signer/mint API from the worker fixture and prove constructing, changing, or
substituting an envelope fails before broadcast.

The result validator accepts `GroupPhaseResult` only for `PREPARE`, `COMMIT`,
and `ABORT`. `LifecycleOperationResult` accepts only the exact enum (never a
string), a canonical nonempty `operation_id`, and the phase mapping
`SOURCE_PREPARE | PREPARE_DESTINATION_WEIGHT_MEMORY | INIT_COMMUNICATOR | REBUILD_COMMUNICATOR | PAUSE |
BEGIN_UPDATE -> PREPARE`, `SOURCE_RESTORE | PREPARE_KV_MEMORY | INVALIDATE | END_UPDATE -> FINALIZE`, and
`PUBLISH_SERVED_VERSION | RESUME | OPEN_COLLECTION | OPEN_ROLLOUT -> ACTIVATE`;
`SUCCEEDED` may satisfy its matching authorization, while `FAILED` immediately
raises with the original detail/cause and is never credited. The validator
matches globally unique `(result_authorization_id, target_id, operation_kind,
operation_id)` one-to-one against the exact transaction-minted
`BoundRefitResultAuthorization` and also checks rank only as an additional
target-bound field. Every result-union variant carries the same nonempty
`target_id` and `result_authorization_id`; neither a numeric rank nor result
kind is authorization identity. The validator also checks execution
kind, context/authority/plan/transaction IDs, source/target versions, and fence
digest. Missing, extra, duplicate, reordered, replayed, wrong-phase, or COMMIT-
substituted lifecycle proofs are fatal. `DISCOVER_CAPACITY`,
`DISCOVER_RENDEZVOUS`, and `PREPARE_METADATA` are rejected as generic lifecycle
results and require their dedicated variants. `CapacityDiscoveryResult`
accepts only exact positive non-boolean bytes, its authorized enum capacity
kind, and exact source/destination participant role. For checkpoint-engine
capacity the controller requires the plan-derived full source-plus-destination
participant set before deriving the canonical minimum/digest.
`RendezvousDiscoveryResult` accepts only an exact
authorized attempt index, canonical nonempty address, non-boolean port in
`1..65535`, and recomputed candidate digest. Collision retry consumes the next
pre-authorized attempt under the same deadline; a worker cannot invent an
attempt or extend the budget.
`PreparedMetadataOperationResult` validates the exact schema/kind, canonical
bytes, recomputed payload digest, rank and operation authorization before its
artifact can resolve a deferred next step. `TransferWorkerResult` is accepted only for `TRANSFER`;
`DestinationLoadResult` only for `LOAD`; and `DestinationFinalizeResult` only
for `FINALIZE`; and `CheckpointReceiptResult` only with phase
`CHECKPOINT_ATTEST` and `execution_kind=STARTUP`. `READY`, `STARTUP_READY`, and
`COMMIT` are coordinator states, while `ACTIVATE` is reserved for a
post-commit/post-startup-ready lifecycle result; the former are not worker
result variants. A source/destination transfer proof is keyed by
`(rank, execution_batch_id, direction)`, a load proof by
`(rank, load_operation_id)`, and a finalizer proof by
`(rank, finalizer_group_id)`. The covered-set digest must equal the exact Task 7
member set for that operation. The validator checks every typed ID, execution
kind, static plan-group identity, exact covered graph/plan-member digest,
runtime-context and transaction-authority identity, source version, target
version, canonical live source-fence-set digest,
and completion fence against the canonical plan group;
no field is blanket-optional, one graph ID cannot stand in for a cross-graph
operation, and no fake physical owner is attached to a group phase. Two target
domains may both report rank zero, but their target and result-authorization
IDs remain distinct; swapping their otherwise identical batches is fatal. The startup
execution uses its concrete initial serving version. A resolved future may
contain one result or a non-empty result batch; validation flattens the batch
only after rejecting duplicates and then checks lifecycle-operation,
checkpoint-receipt, transfer, load, and finalizer acknowledgement sets plus the
separate engine-envelope phase proof. Each checkpoint receipt is independently verified
against immutable evidence and bound destination consumption before its
canonical digest can satisfy the expected `(graph, rank)` proof.

Implement authority binding as a two-level API. The driver validates the full
context and static plans, canonicalizes the safe template bytes, and signs them
with a per-run repo-owned asymmetric signing key that is never serialized. The
wire includes only the template, signature, and public verifier identity;
decoding verifies the signature and all nested canonical digests before
returning the nominal frozen authority type. Its constructors are private to the
binder/verified decoder. `StartupLoadTransaction.from_authority(...)` and
`RefitTransaction.from_authority(...)` accept that exact verified type plus
fresh versions/fences, recheck their plan/context/template IDs, and derive the
runtime transaction identity. They never accept a context, bare plan/projection,
or unchecked mapping at the actor boundary. This lets a Single Controller actor
create multiple concrete versioned transactions without serializing or
reconstructing the full context and without letting a projection self-authorize.

Use `ray.wait(..., num_returns=1, timeout=remaining_deadline)` across both source and destination refs and validate each ready result immediately. Before the first transfer, validate the complete active `SourceVersionFence` set. Derive a fresh runtime transaction identity from the static plan-group ID, source version, target version, canonical live fence-set digest, and, for updates, successful startup/cache precondition digest; bind those fields into every result. Static execution plans, routes, and buffers are not rebuilt. `StartupLoadTransaction` publishes `STARTUP_READY` only after its exact source-fence, transfer, load-owner, finalizer-group, and checkpoint-receipt sets complete; it has no generation-version COMMIT and cannot run twice. `RefitTransaction` verifies the startup/cache digest and commits only after the exact source-fence and acknowledgement sets for every realized load owner and finalizer group affected by a mutable contributor complete. Aliases are de-duplicated independently in each equivalence relation: source transfer, destination load owner, and finalizer group. Non-participating graphs/ranks remain absent. On failure, preserve the first exception, run all abort/poison callbacks concurrently with a fixed teardown deadline, terminate owners whose abort does not acknowledge, then re-raise the first cause wrapped with structured context.

Neither startup readiness nor update commit includes version publication,
backend RESUME, or a mode-specific serving-gate open in the proof set that earns it. Persist
COMMIT, or persist STARTUP_READY for a startup-only graph, then mint the one-shot
activation schedule and supervise the exact shape selected by
`RefitActivationMode`: publication and RESUME in every mode, followed by
`OPEN_COLLECTION` only for async or `OPEN_ROLLOUT` only for Single Controller;
synchronous driver mode ends after RESUME. All waits consume the remaining
time on the original absolute transaction deadline and its same cancellation
authority—there is no new activation timeout. A mixed startup+mutable graph must not mint or submit
any activation after STARTUP_READY; it first runs the initial refit and activates
once after COMMIT. Keep serving closed until every exact target-version
publication, backend resume, and gate-open ACK succeeds; on failure poison and
terminate without rolling back or misreporting the durable coordinator version.

- [ ] **Step 4: Run transaction, watchdog, type, and formatting gates**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_supervisor.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Run: `uv run --no-sync pyrefly check nemo_rl/weight_sync/refit_supervisor.py nemo_rl/weight_sync/transaction.py nemo_rl/distributed/refit_watchdog.py nemo_rl/weight_sync/interfaces.py tests/unit/weight_sync/test_refit_supervisor.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync/refit_supervisor.py nemo_rl/weight_sync/transaction.py nemo_rl/distributed/refit_watchdog.py nemo_rl/weight_sync/interfaces.py tests/unit/weight_sync/test_refit_supervisor.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py pyrefly.toml`

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync/refit_supervisor.py nemo_rl/weight_sync/transaction.py nemo_rl/distributed/refit_watchdog.py nemo_rl/weight_sync/interfaces.py tests/unit/weight_sync/test_refit_supervisor.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py pyrefly.toml
git commit -s -m "feat(refit): add fail-fast transaction supervisor"
```

### Task 12: Integrate Fatal Transactions into IPC, Collective, Reshard, Checkpoint Engine, Sync, and Async RL

**Files:**
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/transaction.py`
- Modify: `nemo_rl/weight_sync/factory.py`
- Modify: `nemo_rl/weight_sync/direct_collective.py`
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py`
- Modify: `nemo_rl/distributed/virtual_cluster.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Modify: `nemo_rl/models/generation/vllm/collective_rpc.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/base.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/registry.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/v0251.py`
- Modify: `nemo_rl/models/generation/vllm/precision_adapter/v0280.py`
- Modify: `nemo_rl/models/generation/vllm/checkpoint_engine.py`
- Modify: `nemo_rl/models/policy/workers/checkpoint_engine.py`
- Modify: `nemo_rl/models/generation/vllm/refit_loader.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_sparse_refit.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_sparse_delta.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/utils.py`
- Modify: `nemo_rl/models/policy/workers/base_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/megatron_remote_sparse_refit.py`
- Modify: `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/async_utils/trajectory_collector.py`
- Modify: `nemo_rl/experience/rollout_manager.py`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Modify: `nemo_rl/algorithms/single_controller_utils/setup.py`
- Modify: `examples/run_grpo.py:190-270`
- Modify: `examples/run_grpo_single_controller.py`
- Modify: `nemo_rl/utils/weight_transfer_http.py`
- Modify: `nemo_rl/utils/weight_transfer_sparse_codec.py`
- Modify: `nemo_rl/utils/weight_transfer_stream.py`
- Modify: `nemo_rl/utils/weight_transfer_zmq.py`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_refit_transaction.py`
- Test: `tests/unit/weight_sync/test_collective_refit_supervision.py`
- Test: `tests/unit/weight_sync/test_reshard_rebuild.py`
- Test: `tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py`
- Test: `tests/unit/algorithms/test_grpo.py`
- Create: `tests/unit/algorithms/test_grpo_sync.py`
- Test: `tests/unit/algorithms/test_async_utils.py`
- Test: `tests/unit/single_controller/test_setup.py`
- Test: `tests/unit/single_controller/test_single_controller_actor.py`
- Test: `tests/unit/single_controller/test_entrypoint.py`
- Test: `tests/unit/models/generation/test_vllm_backend.py`
- Test: `tests/unit/models/generation/test_vllm_collective_rpc.py`
- Test: `tests/unit/models/generation/test_vllm_precision_adapter.py`
- Test: `tests/unit/models/generation/test_vllm_checkpoint_engine.py`
- Test: `tests/unit/models/generation/test_vllm_sparse_refit.py`
- Test: `tests/unit/models/generation/test_vllm_sparse_delta.py`
- Test: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
- Test: `tests/unit/models/policy/test_worker_refit_signatures.py`
- Test: `tests/unit/models/policy/test_utils.py`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Test: `tests/unit/models/policy/test_megatron_remote_sparse_refit.py`
- Test: `tests/unit/models/policy/test_dtensor_worker.py`
- Test: `tests/unit/models/policy/test_dtensor_worker_v2.py`
- Test: `tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_reconcile_communicator.py`
- Test: `tests/unit/distributed/test_virtual_cluster.py`
- Test: `tests/unit/distributed/test_virtual_cluster_batch_ports.py`
- Test: `tests/unit/algorithms/test_grpo_refit_supervision.py`
- Test: `tests/unit/algorithms/test_grpo_checkpoint_engine.py`
- Test: `tests/unit/single_controller/test_refit_recovery.py`
- Test: `tests/unit/experience/test_rollout_manager.py`
- Test: `tests/unit/models/generation/test_vllm_generation.py`
- Test: `tests/unit/models/generation/test_vllm_refit_lifecycle.py`
- Test: `tests/unit/models/generation/test_vllm_refit_loader.py`
- Test: `tests/unit/models/generation/test_vllm_nixl_worker.py`
- Test: `tests/unit/models/generation/test_vllm_quant_backend.py`
- Test: `tests/unit/models/policy/test_dtensor_checkpoint_engine.py`
- Test: `tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py`
- Test: `tests/unit/models/policy/test_megatron_checkpoint_engine.py`
- Test: `tests/unit/utils/test_weight_transfer_stream.py`
- Create: `tests/functional/refit_failure_exit.py`
- Test: `tests/functional/test_single_controller_resource_handoff_ray.py`
- Test: `tests/functional/test_single_controller_tq_handoff_ray.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes controller-side: the exact Task 5
  `BoundSemanticPrecisionRuntimeContext`, Task 7 plan groups carrying its
  `runtime_context_id`, their authenticated plan-bound worker projection,
  Task 11's driver-minted `BoundRefitTransactionAuthority`,
  `StartupLoadTransaction`, `RefitTransaction`, exact live source-version fence
  proofs, and typed worker results from Task 11. No controller or synchronizer
  may reconstruct authority from bare selection/intents; workers receive only
  the projection and bound execution records.
- Produces: before synchronous, asynchronous, or Single Controller execution
  can serve generation, a nonempty startup group reaches `STARTUP_READY` once.
  If canonical every-version members/ACKs also exist, the initial mutable
  `RefitTransaction` must then COMMIT; if that refit group is empty (a valid
  all-frozen or checkpoint-only deployment), `STARTUP_READY` itself is the
  serving proof and no empty refit transaction/supervisor call is constructed.
  Conversely, an all-mutable graph validates the canonical no-startup
  precondition and must COMMIT its nonempty initial `RefitTransaction` before
  serving—the no-startup digest alone never opens serving. A configured semantic
  policy whose startup and refit groups are both empty is rejected before RPCs.
  It also provides the same fatal behavior for every supported vLLM destination
  path—including vLLM's internal FlashInfer-TRTLLM mixed-layout loader—over
  direct collective, synchronizer collective, IPC, NCCL reshard, checkpoint
  engine and both vLLM remote-sparse transports on every later
  refit. Single Controller's pre-pump and pump refits use the same
  transactions and finite `refit_timeout_s`; generation weight version changes
  only on every-version COMMIT. Backend resume/collection activation happens
  only after durable COMMIT, or after STARTUP_READY only for a startup-only
  graph. A mixed startup+mutable graph emits zero activation RPCs between
  STARTUP_READY and its initial refit COMMIT. The separate activation schedule
  is discriminated by execution mode. Every mode supervises exact target-
  version `PUBLISH_SERVED_VERSION` results and then all backend RESUME results.
  Async adds a final `OPEN_COLLECTION` leaf on the trajectory collector;
  Single Controller adds a final `OPEN_ROLLOUT` leaf that alone sets
  `_rollout_permitted`; synchronous driver GRPO has no collector and validates
  the canonical two-step variant with no final open RPC or facade-created ACK.
  All steps consume the remaining original absolute refit deadline and the
  same cancellation authority; activation cannot start a new timeout. Silent,
  malformed, or wrong-version publication, resume failure, wrong-mode schedule,
  or applicable gate-open failure is terminal and cannot open serving.
  Split the collector's current `set_weight_version()` side effects: the new
  authenticated `publish_refit_served_version()` only stamps the exact target
  version and returns its typed publication ACK; it must not set
  `_generation_limit_cleared` or dispatch collection. Backend-specific resume
  runs next without opening the collector. In async mode only, the separately authorized
  `open_collection_after_refit()` consumes `OPEN_COLLECTION` and sets the gate
  after every publication and backend RESUME ACK has succeeded. Single
  Controller instead exposes `open_rollout_after_refit()` for `OPEN_ROLLOUT`;
  no collector method is called, and `_rollout_permitted` remains false until
  its exact ACK. Synchronous driver mode has neither method.
  Task 12 implements
  `build_refit_activation_mode_dispatcher(*, mode, runtime_owner,
  transaction_authority,
  publication_facades, resume_facades, async_collector=None,
  single_controller=None) -> RefitActivationModeDispatcher` in
  `weight_sync/transaction.py`. The factory is allocation-free, requires
  `runtime_owner.runtime_owner_id ==
  transaction_authority.expected_runtime_owner_id`, and validates exact
  endpoint identities against the frozen runtime-owner descriptor graph:
  async requires exactly one owned collector and forbids a Single Controller
  gate; Single Controller requires exactly its adopted actor gate and forbids a
  collector; synchronous driver mode forbids both. Regular sync/async GRPO
  constructs and retains this controller-local dispatcher beside its
  `GRPORuntimeResourceOwner` before the first transaction. The Single
  Controller actor constructs it from adopted, descriptor-bound handles only
  after the exact ownership ACK and retains it until actor/runtime-owner
  cleanup. It is never serialized from the driver, owns no resources separate
  from that runtime owner, becomes invalid when owner cleanup starts, and has no
  global/default reconstruction path. A mode/owner/endpoint mismatch fails
  before publication.
  Dynamo, SGLang, standalone TRTLLM, and Megatron generation are intentionally
  outside this support claim: Task 5 rejects them before resources, so Task 12
  has no route that may reinterpret an untyped status or boolean as a worker
  proof. Removing a preflight requires a separately
  reviewed destination-process extension that validates authority at the
  mutation boundary and returns worker-produced typed proofs, plus a persistent
  lifecycle authority for its driver-local subprocesses/actors/threads; a
  facade-created proof is never sufficient.
- Changes the polymorphic refit boundary itself, rather than relying on mutable
  backend state: `ColocatablePolicyInterface` source-transfer/NCCL
  methods and `GenerationInterface` refit-only prepare, pause/resume, IPC,
  collective, NCCL, and backend `init_collective`/communicator-rebuild methods
  accept a keyword-only exact per-operation
  `BoundRefitFacadeDispatchManifest`. The corresponding high-level
  enabled-only `SemanticPrecisionWeightSynchronizer` and
  `SemanticPrecisionDirectRefitHandle` protocols accept the complete
  `BoundRefitPrecommitDispatchSchedule` and release exactly one next manifest to
  each operation wave; each facade first
  validates its complete ordered outer-target registry against that manifest,
  then sends exactly one matched `BoundRefitWorkerDispatchNode` to each outer
  backend worker. A direct branch exposes `ordered_children` and is valid only
  for a genuinely direct public route. A broadcast branch exposes no children:
  it contains only the controller-minted `group_envelope` plus committed child
  identities. The vLLM sync/async dispatcher requires that broadcast branch,
  validates its internal TP/PP target registry against the envelope, and passes
  the exact envelope bytes unchanged to public identical-argument
  `collective_rpc`; it never reads `.ordered_children`, rebuilds, or re-signs an
  envelope. A direct branch on this public broadcast path is fatal before RPC.
  each leaf selects exactly its target-bound `BoundRefitDispatchLeaf` and receives
  only that leaf's singular `BoundRefitWorkerExecutionBundle`. The immutable safe worker bundle is strictly
  decoded from Task 11's verified authority and carries shared authority/
  template, runtime-context, transaction, source/target-version, live-fence,
  worker/rank, and dispatch identities, the authenticated execution-deadline
  and nested-cancellation wires, plus an exact nonempty ordered tuple of
  `BoundRefitResultAuthorization`. Each authorization commits to one expected
  result variant and its phase/direction, plan-group/batch, checkpoint receipt,
  load-operation, finalizer-group, and covered-member identities as applicable.
  The worker maps returned proofs one-to-one to that tuple and rejects an
  extra, missing, duplicated, reordered, or locally invented authorization;
  heterogeneous checkpoint + transfer + multi-load + multi-finalizer results
  may remain one atomic native batch. Setup/refit lifecycle authorizations use
  exact operation kind/ID, so `init_collective`, rebuild, metadata prepare,
  pause, begin, invalidate, end, and resume cannot substitute for one another or
  for coordinator COMMIT. Split both input and output into explicit, disjoint
  driver-facade, dispatching-worker, and leaf-worker protocols; do not model
  them as overrides of one method with incompatible narrowed arguments.
  `PolicyRefitFacade` and
  `GenerationRefitFacade` accept the dispatch manifest and return
  `list[ray.ObjectRef[RefitWorkerResultBatch]]`.
  `GenerationRefitDispatchingWorkerEndpoint` accepts one broadcast dispatch branch and
  returns one direct nonempty result batch after the validated v1 broadcast-safe
  group-envelope routing contract.
  `PolicyRefitWorkerEndpoint` (Megatron/DTensor v1/v2) and
  `GenerationRefitLeafWorkerEndpoint` accept the singular worker bundle from a
  strictly validated dispatch leaf and
  return one direct nonempty
  `RefitWorkerResultBatch`. Shared non-refit behavior may remain on the existing
  ABCs, but their refit entrypoints delegate to these separate protocols. No
  interface uses a union, `Any`, or one input/output shape for both layers.
  `None`, booleans, untyped
  refs, or authority recovered from an earlier refit-prepare call are invalid.
  At every branch and leaf entry, validate that the deadline/cancellation IDs
  match the bundle, envelope, transaction, and operation before any nested
  submission. Start a local countdown from the received remaining budget,
  poll/wait on the exact nested cancellation ObjectRef alongside every inner
  future, and pass the same cancellation wire with a non-increased remaining
  cap to deeper helpers. No outer worker may replace the ref, restart
  `refit_timeout_s`, or wait for a child after cancellation/expiry; late child
  completion is poisoned and cannot mutate or return a credited proof.
  Ordinary `prepare_for_generation()` remains the transaction-free wake API on
  enabled and absent-policy paths. Add/use a distinct
  `GenerationRefitFacade.prepare_generation_for_refit(*,
  facade_dispatch_manifest=...)` and
  `GenerationRefitDispatchingWorkerEndpoint.prepare_generation_for_refit(*,
  worker_dispatch_branch=...)` or
  `GenerationRefitLeafWorkerEndpoint.prepare_generation_for_refit(*,
  worker_execution_bundle=...)` entrypoints (and the corresponding distinct
  refit-only pause/resume methods) rather than overloading the ordinary wake
  method. Likewise, keep ordinary `invalidate_kv_cache()` for non-refit
  finish/cache lifecycle and add facade-manifest, dispatch-node, and singular-
  worker-bundle forms of `invalidate_kv_cache_for_refit()` on those separate protocols for the
  transaction-authorized invalidation proof. If the
  absent semantic-policy path must retain its old call shape, expose it as an
  explicit overload guarded by an exact internal legacy sentinel and preserve
  its behavior byte-for-byte; an enabled endpoint rejects the sentinel or an
  omitted/`None`/foreign input of its protocol's exact shape before any worker
  RPC. Do not
  use a global, instance default, or last-prepared transaction as fallback.
  A repeated target across pause/begin/update/end or communicator-init/
  metadata/process-group waves receives a different operation-ID-bound node in
  each schedule step. Consuming a step twice, skipping/reordering a step,
  returning before the schedule is empty, or reusing a prior step's bundle is
  fatal and poisons all unconsumed work. The precommit schedule cannot contain
  `PUBLISH_SERVED_VERSION`, RESUME, `OPEN_COLLECTION`, or `OPEN_ROLLOUT`. Immediately after
  durable COMMIT—or STARTUP_READY for a startup-only graph—the transaction, not
  the synchronizer, mints and
  supervises the separate one-shot `BoundRefitActivationDispatchSchedule`:
  publish the exact served version first and release backend RESUME second;
  then release exactly the selected async `OPEN_COLLECTION` or Single
  Controller `OPEN_ROLLOUT` step, while the synchronous-driver variant ends
  without an open step. Every wait receives only
  `original_absolute_deadline.remaining_s(...)` plus the same cancellation
  wire; no post-COMMIT timeout is created. A sync method
  that publishes/resumes internally or sees activation authority before commit
  is invalid.
  Define explicit activation facades and leaf endpoints rather than wrapping
  legacy setters: `RolloutManager.publish_semantic_precision_served_version()`
  and the vLLM facade accept the operation-bound facade manifest and return
  ObjectRefs to typed batches; the rollout-manager/vLLM leaf methods accept only
  their singular worker bundle and return worker-produced
  `LifecycleOperationResult(PUBLISH_SERVED_VERSION)` for the exact target
  version. The coordinator cannot synthesize an ACK from `True` or `None`.
  Existing `RolloutManager.set_weight_version()` and vLLM
  `set_rollout_weight_version()` remain unchanged for the absent-policy path and
  are forbidden on enabled setup/refit. In particular, the current Single
  Controller setup stamp before synchronizer creation/initialization is skipped
  on the enabled path: its first stamp occurs only through the authenticated
  PUBLISH wave after STARTUP_READY for startup-only graphs or after the initial
  COMMIT otherwise. A failed initial sync therefore cannot leave a future
  served version stamped. `open_collection_after_refit()` is a distinct async
  collector leaf operation; `SingleControllerActor.open_rollout_after_refit()`
  is the distinct `_rollout_permitted` leaf. Neither can be combined with
  publication/backend resume or appear in the other mode, and synchronous
  driver mode exposes neither.
  Source offload belongs to that same precommit state machine rather than an
  out-of-band optimization. Add refit-only facade/worker entrypoints for
  `SOURCE_PREPARE` and `SOURCE_RESTORE`: the former owns
  `policy.offload_before_refit()` and precedes transfer, while the latter owns
  `policy.offload_after_refit()` after source send and before COMMIT. Both take
  their exact operation-bound manifest/node/bundle and the original transaction
  deadline/cancellation authority, return typed lifecycle results, and poison
  on failure or timeout. Their absent-policy zero-argument methods remain
  unchanged behind the legacy protocol. A transaction cannot COMMIT or mint
  activation until every expected source-restore proof is valid.
  Capacity discovery likewise has explicit enabled-only typed entrypoints:
  `PolicyRefitFacade.discover_refit_capacity(*,
  facade_dispatch_manifest=...)` and its leaf worker form return
  `CapacityDiscoveryResult` batches for policy free memory. Checkpoint-engine
  total-memory discovery has both source-policy and destination-generation
  facade/leaf forms; each result carries its exact participant role, and the
  controller validates the complete plan-derived source-plus-destination union
  before taking the minimum. Facades return ObjectRefs and leaves return direct batches under the
  same disjoint protocols. The existing raw `get_free_memory_bytes()` and
  checkpoint total-memory APIs remain only for explicit legacy callers; an
  enabled synchronizer cannot call them or turn an `int` into a proof.
  Existing `WeightSynchronizer.init_communicator()`/sync methods and legacy
  direct calls keep their exact zero-new-argument signatures for absent-policy
  PPO, distillation, and other callers. Enabled factories must advertise and
  return the distinct semantic protocol, whose methods are explicitly named
  `init_semantic_precision_communicator(...)` and
  `sync_semantic_precision_weights(...)`. It also exposes
  `reconcile_semantic_precision_communicator(*,
  facade_dispatch_schedule, deadline, cancellation)` for the authorized
  rebuild path; an enabled path may never fall back
  to the legacy protocol, and an absent path never probes or calls these new
  methods.
  `BoundRefitBroadcastGroupEnvelope` is the dependency-neutral frozen wire
  value created in Task 11's `weight_sync/interfaces.py` alongside the other
  dispatch values. Task 12's `models/generation/interfaces.py` imports it and
  defines the public `BroadcastSafeRefitDispatchCapability` protocol. For
  vLLM's public identical-argument
  `LLM.collective_rpc`, the transaction supplies one pre-minted broadcast
  branch/envelope. The selected endpoint adapter uses verifier authority to
  validate its ordered TP/PP registry and the exact one-target-bound-child-per-
  leaf envelope commitment, then broadcasts those unchanged bytes. Each
  repo-owned leaf extension recomputes the envelope digest and verifies its
  transitive transaction-authority binding, derives
  its immutable runtime target identity, selects exactly one matching entry,
  and validates/consumes only that entry's singular bundle; zero/multiple
  matches or an attempt to use a sibling entry is fatal before mutation.
  Sibling wire visibility is harmless and confidentiality is not claimed—their
  target binding makes them unusable. The 0.25.1 and 0.28.0 vLLM adapters probe
  and conform to this public broadcast-safe capability, including the internal
  FlashInfer-TRTLLM runtime layout. No generic worker reaches into
  a version's private executor, no parent/root node is broadcast, and no facade
  synthesizes a leaf ACK. Missing extension/capability fails adapter preflight
  before serving.
- In async GRPO the driver creates the transaction's authoritative monotonic
  deadline and cancellation authority before submitting
  `trajectory_collector.prepare_for_refit.remote(...)`. That outer ObjectRef is
  itself a supervised lifecycle participant. The collector receives only its
  authenticated pure `BoundRefitDirectDispatchBranch` containing exact nested
  backend children and no collector-local execution bundle, plus a one-shot remaining-
  budget wire and the exact authenticated nested cancellation ObjectRef.
  Collector-local publication/activation is represented by a
  separate leaf child when required; the branch aggregates returned child
  batches but cannot acknowledge an operation itself. It
  validates both wires against every child bundle, creates one local countdown
  capped by the received remaining duration, and races every nested
  pause/update/invalidate/resume ref against the same cancellation ObjectRef;
  it never starts
  a fresh per-phase timeout. The driver continues to bound the outer ref by its
  original deadline, so scheduling delay cannot extend the transaction.

- [ ] **Step 1: Write failing integration and launcher-exit tests**

```python
def test_initial_async_refit_failure_is_reraised_and_collection_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo, "refit_policy_generation", Mock(side_effect=RuntimeError("receiver load")))
    with pytest.raises(RuntimeError, match="receiver load"):
        run_async_training_fixture()
    assert collector.set_weight_version.call_count == 0
    assert collector.start_collection.call_count == 0
    assert collector.flush_telemetry.call_count == 1

@pytest.mark.parametrize("execution_mode", ["sync", "async", "single_controller"])
def test_refit_failure_makes_launcher_exit_nonzero(execution_mode: str) -> None:
    completed = run_fault_injected_grpo_subprocess(
        execution_mode=execution_mode, phase="finalize"
    )
    assert completed.returncode != 0
    assert "finalize" in completed.stderr
```

Add transport tests where a generation future fails while a train future remains
pending, a lone peer remains silent, all-`None` results, malformed results,
worker wrapper errors that currently return `False`, cache invalidation failure,
failed enabled `prepare_generation_for_refit` (with an otherwise valid exact
bundle), failed shutdown, and async refit after collector
pause. Cover direct collective, synchronizer collective, IPC, NCCL reshard,
checkpoint engine, both native-vLLM and internal FlashInfer-
TRTLLM runtime-layout cases, and both `vllm_s3_sparse` and `vllm_zmq_sparse`
paths; no advertised Task 5 transport may retain a sequential
or unbounded wait. In particular, reject checkpoint-engine RPC wrappers that
filter `None` or accept vacuous `all(...)`, and require its batched loader,
main/MTP/Eagle physical destinations, finalizers, poison/abort, and exact
acknowledgements to use the same bound transaction plan. Add auxiliary preflight
cases for a mutable training-only MTP/drafter, a mixed frozen/mutable
source-served MTP/drafter, an all-frozen source-served startup graph, a static
checkpoint drafter, a missing drafter on an owning rank, and an absent drafter
on a non-owning PP rank. When the canonical startup group is non-empty, assert
it runs once before collection. For all-frozen source-only and checkpoint-only
fixtures with an empty every-version group, successful `STARTUP_READY` opens
serving and the transaction/supervisor spies observe zero `RefitTransaction`
calls afterward; checkpoint-only startup accepts an empty producer side but
still requires every destination/attestation ACK. An all-mutable graph instead
emits the canonical no-startup precondition digest without constructing a
startup transaction or calling the startup supervisor, but that digest must not
open serving: its nonempty initial mutable refit must COMMIT first. Reject a
semantic graph with both groups empty. Frozen owners never enter later payloads.
Any startup/refit failure prevents resume/serve with bounded actor cleanup.
Add a mixed frozen+mutable fixture that records durable STARTUP_READY, then runs
the initial refit: no PUBLISH, backend RESUME, or mode-specific open RPC may be
submitted between STARTUP_READY and COMMIT, and activation runs exactly once
after COMMIT. In `tests/unit/experience/test_rollout_manager.py` and Single
Controller setup tests, assert enabled setup never calls the legacy early
`set_weight_version()`/`set_rollout_weight_version()` before synchronizer init.
Exercise exact typed rollout-manager and vLLM publication results, wrong-version
and boolean/malformed ACKs, a silent publisher, and a failed initial sync; none
may stamp a future version. Parameterize async, Single Controller, and
synchronous-driver schedules. Delay async collector open and Single Controller
rollout open while a backend RESUME fails immediately and prove neither
`OPEN_COLLECTION` nor `OPEN_ROLLOUT` is submitted and no generation request is
dispatched; synchronous mode submits no open RPC even on success. Reject every
wrong-mode/missing/extra open step.
Construct the production dispatcher in each fixture and assert exact ownership:
sync/async driver dispatchers reference the same live
`GRPORuntimeResourceOwner`; the Single Controller dispatcher is absent before
adoption, created actor-local only after the exact adoption ACK, and references
that adopted owner ID. Reject a borrowed/closed/foreign owner, missing or extra
collector/controller endpoint, a one-byte mutation of the schedule's bound
owner ID or authority-binding digest, driver-serialized dispatcher, and any fallback
factory/global. Starting runtime-owner cleanup invalidates dispatch before the
first RPC. The schedule tests pass this concrete dispatcher explicitly—no
fixture or implementation may rely on an unbound `activation_dispatcher`
variable.
In `tests/unit/models/generation/test_vllm_sparse_refit.py`,
`test_vllm_sparse_delta.py`,
`tests/unit/models/policy/test_megatron_remote_sparse_refit.py`, and
`tests/unit/utils/test_weight_transfer_stream.py`, bind every HTTP/ZMQ payload,
queue/apply/flush, collective wait, retry, and response to the exact transaction
bundle, source/target version, live fence, operation authorization, shared
deadline, and cancellation authority. Reject stale/cross-transaction replay,
duplicate chunks, and missing load/finalizer proof. Hang a queued apply,
collective, HTTP executor future, retry/backoff, and ZMQ flush separately; each
must stop at the shared deadline, poison/tombstone the transaction, cancel
pending work without joining, and return no metric-dict/boolean success. A late
orphan apply from the failed transaction must remain poisoned and must not be
consumed by the next valid refit. Successful HTTP and ZMQ paths emit the exact
typed load/finalizer result batch. The late-orphan test must enter the real
`vllm_backend.py` collective-RPC endpoint and
`VllmSparseDeltaApplier` in `vllm_sparse_delta.py`: validate authorization,
version/fence, and the shared poison tombstone immediately before the first
weight mutation and again before finalize, so a receiver-side timeout cannot be
raced by an already submitted collective.
In `tests/unit/models/generation/test_vllm_collective_rpc.py`, build nested
list/tuple mixtures of Ray ObjectRefs, concurrent futures, awaitables, and typed
result batches returned by `checkpoint_engine_rpc_async()`. Resolve all leaves
concurrently under the exact deadline/cancellation wires decoded from each
leaf bundle. Assert every inner helper receives the same deadline/cancellation
IDs and nested ObjectRef, only decreases the remaining cap, and races that ref
against its work; a missing/replaced ref or reset/full timeout is fatal before
waiting. Resolve under the transaction's remaining authority,
preserve canonical container/result order, and validate each leaf as it becomes
ready. A first failure or malformed leaf cancels/abandons every sibling without
joining; a lone `asyncio.wrap_future`, `to_thread(ray.get)`, or nested awaitable
hang cannot exceed the shared deadline, and a late completion cannot mutate or
acknowledge the poisoned transaction. Also construct sync and async TP>1/PP>1
target registries and require the repo-owned broadcast-safe collective adapter
to validate the complete ordered child-node manifest and controller-pre-minted
group envelope before submitting any call, then use those unchanged bytes in
the public identical-argument broadcast. Every engine leaf
derives its immutable runtime target identity, selects exactly one matching
entry, validates the envelope and singular bundle, and consumes only that
entry; wire visibility of sibling entries grants no usable authority.
Broadcasting a root node, ancestor subtree, or unbound child map is forbidden.
Missing, extra, duplicate, reordered, aliased, zero-match, multi-match,
sibling-use, and late-completing engine cases fail before mutation.
In `tests/unit/models/generation/test_vllm_precision_adapter.py`, run the
broadcast-safe dispatch conformance contract against the 0.25.1 and 0.28.0
adapters. The executable canary proves the supported public
`LLM.collective_rpc` surface sends the same immutable group-envelope argument
to every TP/PP worker, every leaf selects the one entry bound to its probed
target identity, and every returned ref/result preserves that target mapping.
An absent/incomplete/lying capability, a leaf that can consume a sibling
entry, or an adapter that reaches into a private executor fails preflight.
Assert `precision_adapter.base.BroadcastSafeRefitDispatchCapability is
models.generation.interfaces.BroadcastSafeRefitDispatchCapability`; a second
class/protocol with identical fields is rejected so there is one registry
contract.
Run the same TP>1/PP>1 contract against the vLLM internal FlashInfer-TRTLLM
extension and require validation at its actual mutation/finalize boundary; this
must not instantiate or import standalone `generation.backend=trtllm`.
Reject a missing/foreign full context, bare intents, a Phase-2-only projection,
or a plan/projection/result whose `runtime_context_id` differs before submitting
any refit worker RPC. Recursively prove actual worker payloads contain the
authenticated plan-bound projection but no full request/results/trusted
contributor authority or adapter objects.
Parameterize the source-transfer result contract over Megatron, DTensor v1, and
DTensor v2. `LMPolicy.broadcast_weights_for_collective()` routes through one
common typed execution wrapper, and every backend worker returns the exact
`TransferWorkerResult`/batch with source direction, rank, plan/batch/context/
authority/version/fence identity. Legacy `None`, exact booleans, or a backend
that bypasses the wrapper are fatal before COMMIT.
For those same three source backends, surround transfer with the authorized
`SOURCE_PREPARE` and `SOURCE_RESTORE` waves. Inject a lone hung or failed worker
into each wave and prove the one shared deadline/cancel authority terminates the
transaction, no destination COMMIT occurs, and late offload completion cannot
restore or mutate the next transaction. Successful restore returns one exact
operation-ID-bound result per expected source rank; bare `ray.get`, `None`, and
boolean completion remain invalid on enabled paths.
In `tests/unit/models/policy/test_worker_refit_signatures.py`, reflect the
two disjoint layer interfaces and every concrete implementation separately:
enabled source prepare/restore and source-transfer,
typed policy/checkpoint capacity discovery, refit-only prepare/pause/resume,
IPC, collective, NCCL,
`init_collective`, communicator rebuild, and backend update entrypoints
on each driver facade must expose keyword-only `facade_dispatch_manifest` and
return ObjectRef batches; vLLM dispatching outer workers expose
`worker_dispatch_branch`; and each leaf Megatron/DTensor/backend worker endpoint
exposes `worker_execution_bundle` and returns a direct nonempty typed batch.
Assert none of the three layers accepts another layer's argument or return shape
and no facade class satisfies/inherits either remote-worker refit protocol by
narrowing an override. Omitted/`None`, bare authority wire, wrong phase or
direction, and a foreign plan/context/version/fence bundle fail before the
first mocked RPC; the explicit absent-policy sentinel reaches only the unchanged
legacy overload. In `test_vllm_backend.py`, construct multi-DP and multi-endpoint
facade registries and require `VllmGeneration` to validate the exact ordered
dispatch manifest before its first sync or async RPC, then forward only each
target's fresh matched root node by identity. For sync and async TP>1/PP>1
workers, validate each node's exact ordered internal registry before the first
collective RPC, verify the exact controller-minted
`BoundRefitBroadcastGroupEnvelope`, and use those bytes unchanged in the
runtime's public identical-argument broadcast. Add the same TP>1/PP>1 case for
the vLLM internal FlashInfer-TRTLLM extension. Reject missing, duplicate, extra, reordered, aliased,
zero-match, multi-match, or cross-rank entries and one worker result missing
any bound identity at either hierarchy level. Prove a leaf may observe the
envelope wire but cannot decode, consume, or acknowledge another leaf's
authorization. Apply the same fan-out/cardinality cases to the policy and
synchronizer facade suites. In
`tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`, include
the concrete `VllmInternalWorkerExtension` ModelOpt subclass in signature
reflection. Its real-quant `prepare_refit_info` must accept/validate the same
worker sub-bundle (or delete the override and use one tested common hook);
missing/foreign authority fails before quantization preparation and cannot fall
back to the old one-argument override. In the vLLM facade/worker suites, add an
atomic mixed fixture whose bundle authorizes one checkpoint receipt, source and
destination transfer, multiple loads, and multiple finalizers; require exact
ordered one-to-one result coverage without splitting the native operation.
In `tests/unit/models/policy/test_utils.py`, drive
the supported IPC bucket-transfer helpers with authorization sub-bundles and
the shared deadline/cancel authority. Each nested engine chunk, init, lock
acquire/release, NCCL handle,
and update result must be returned in the outer policy-worker's exact typed
mixed batch. Inject a silent engine, silent lock, silent collective handle,
malformed ACK, and failure after lock acquisition; cancellation must release the
lock/temporary handles without joining and missing/forged nested destination
proofs must fail the outer transaction.
In both facade suites, call ordinary `prepare_for_generation()` on an enabled
semantic-policy endpoint with no transaction and require the existing wake
behavior to succeed unchanged; call `prepare_generation_for_refit()` without a
bundle and with a foreign bundle and require failure before its first RPC.
Exercise ordinary `finish_generation()`/`invalidate_kv_cache()` the same way,
while refit-specific invalidation without the exact bundle must fail.
In `tests/unit/weight_sync/test_weight_synchronizer.py`, parameterize an
all-mutable initial communicator init, startup-only init, and a later rebuild;
each exact lifecycle operation ID must be returned once. Missing, reordered,
replayed, wrong-kind, or coordinator-COMMIT substitution is fatal and leaves
the serving gate closed. Give checkpoint init an ordered
capacity-discovery→communicator-init→metadata→process-group schedule and
auto-sized IPC the exact colocated order `SOURCE_PREPARE ->
PREPARE_DESTINATION_WEIGHT_MEMORY -> DISCOVER_CAPACITY -> transfer/finalize ->
SOURCE_RESTORE -> PREPARE_KV_MEMORY`. Checkpoint capacity derives the minimum
positive `checkpoint_engine_total_memory_bytes` across the exact role-tagged
union of every expected source and destination rank; omitting either side is fatal;
IPC derives the minimum positive `Policy.get_free_memory_bytes()` result.
Missing, malformed, duplicate, foreign, failed, or silent capacity results are
fatal under the original deadline. An explicit configured buffer size submits
zero capacity RPCs but still runs both memory lifecycle steps and applicable
source prepare/restore. Assert capacity cannot run before destination weight
memory is awake and KV memory cannot wake before transfer/finalize/source
restore. Patch the legacy raw-int policy/checkpoint methods to fail if called
on the enabled path and require typed leaf-produced results through the exact
capacity facade/worker signatures; synchronizer-side int-to-proof conversion is
fatal. The absent fixture still exercises the raw API unchanged. Checkpoint metadata results are likewise validated as an
exact rank-bound NIXL aggregate before the deferred process-group manifest can
be resolved; booleans or unbound metadata cannot advance the schedule. The same target recurs under distinct
operation IDs and fresh nodes; assert exact next-step consumption, complete
schedule exhaustion, and fatal duplicate/skip/reorder/early-return or prior-
bundle reuse. Assert the synchronizer neither receives nor submits RESUME; hold
COMMIT/STARTUP_READY pending and prove no activation RPC occurs, then mint a
separate one-shot activation schedule. Supervise its exact target-version
publication step before releasing backend RESUME, and then apply the validated
mode shape: async `OPEN_COLLECTION`, Single Controller `OPEN_ROLLOUT`, or no
open step for synchronous driver mode. Silent/malformed/wrong-version
publication, resume failure, wrong-mode shape, or applicable gate-open failure
keeps serving gated and poisons the runtime. Consume only the remaining
original transaction deadline throughout activation and assert a nearly
exhausted precommit budget is not reset. For collective and NCCL-reshard rebuild, exercise
`RayVirtualCluster.get_master_address_and_port_for_refit()` as an authorized
runtime `DISCOVER_RENDEZVOUS` step. A lone hung port task, malformed candidate,
or address collision retry remains inside the same transaction deadline and
cancel authority; no test may reuse Task 5's sealed construction context or
fall back to the legacy unbounded `get_master_address_and_port()` path. Reflect `weight_sync/interfaces.py` separately from
the policy/generation facade protocols and require every enabled
`SemanticPrecisionWeightSynchronizer`/direct-handle implementation to expose
the same keyword-only `facade_dispatch_schedule`, shared-deadline, and
cancellation inputs on `init_semantic_precision_communicator()` and
`sync_semantic_precision_weights()` for communicator rebuild, initial load, and later refit
dispatch; each operation facade takes one released manifest, while its remote-
worker protocol takes exactly one matched
`worker_execution_bundle`. The abstract driver boundary returns only
`list[ray.ObjectRef[RefitWorkerResultBatch]]`; no concrete override may narrow
or erase the bound inputs or return `None`, `bool`, or a direct worker batch.
In the same reflection suite and existing
`test_collective_refit_supervision.py`/`test_reshard_rebuild.py`, snapshot the
legacy `WeightSynchronizer` and direct-call signatures and exercise their
unchanged absent-policy calls with no schedule. Assert enabled factories reject
a legacy-only implementation before communicator RPC and disabled PPO/
distillation never touch the semantic protocol.
Task 11 introduces the result/authority value types only; Task 12 changes these
abstract signatures atomically with every implementation, so there is no
intermediate abstract/concrete mismatch.
In `tests/unit/distributed/test_virtual_cluster.py` and
`test_virtual_cluster_batch_ports.py`, cover the runtime-only refit rendezvous
API independently: claim and publish every probe ref before waiting, validate
its exact operation/attempt authorization and typed artifact, cap wait plus
collision backoff by the caller's remaining monotonic deadline, and resolve the
same cancellation authority on failure. Test success, submit failure, lone
hang, malformed/duplicate candidate, collision then success, deadline during
backoff, and late completion after poison. The existing construction-time and
absent-policy APIs remain unchanged; a runtime refit call without an exact
bundle/deadline is rejected before a Ray submission.
In `test_reconcile_communicator.py`, `test_refit_recovery.py`, and the Single
Controller actor tests, require every enabled initial/recovery reconcile to call
only `reconcile_semantic_precision_communicator()` with its exact authorized
rebuild step, shared deadline, and cancellation authority. A lone hung or failed
reconcile/port probe is surfaced through the original cause chain and poisons
the transaction. Assert the enabled path never invokes legacy
`reconcile_communicator(absent, force)` or wraps it in unbounded
`asyncio.to_thread`; the absent-policy fixture retains that legacy call and its
existing recovery classification exactly.

In `tests/unit/algorithms/test_grpo.py` and `test_async_utils.py`, assert the
driver creates one transaction/deadline before submitting the collector's outer
`prepare_for_refit` ref and forwards the matching lifecycle authorizations and
authenticated one-shot remaining-budget plus nested-cancellation wires. Assert
each collector/dispatcher/leaf validates the same IDs and no nested call gets a
larger remaining duration. Make the outer collector ref silent/dead, then make
one nested backend pause/update/invalidate/version-publication/resume ref silent; both cases must be
detected by the same driver deadline, resolve the common cancellation
authority, and never enter collection or consume a fresh nested timeout.
Explicit ready failure short-circuits. Serialize the collector and generation
copies recursively and reject a full context, signing key, cached authority, or
locally minted bundle while retaining only the authenticated bundle, budget,
and nested cancellation ObjectRef needed for that transaction. Reject a missing
or replaced ref and resolve it while a nested backend future remains pending;
the leaf must stop without mutation/ACK and the late result stays poisoned.
Hold COMMIT/STARTUP_READY pending and prove no
publication, backend resume, or collection occurs. Once durable, supervise
destination and collector publication as the first ACTIVATE step, reject a
malformed/wrong-version ACK, and release RESUME only after every publisher ACK.
Release async `OPEN_COLLECTION` only after every backend RESUME ACK. Inject a silent
publisher, resume failure, or collector-open failure and prove the serving gate
remains closed and the launcher exits terminally. In the collector fixture, let
`publish_refit_served_version()` succeed and the backend RESUME fail or hang;
assert `_generation_limit_cleared` remains unset and no new rollout/collection
dispatch occurs. The legacy combined `set_weight_version()` must not be called
from an enabled transaction.

In the Single Controller tests, exercise startup-only, all-mutable, and mixed
supported-vLLM initial-sync shapes plus the
actor's pre-pump and repeated `_sync_weights()` calls. A pending sender plus
failed receiver, lone silent peer, `None`/malformed result, pre-pump initial
failure, and later pump refit failure must poison the destination, close serving,
enter the unified resource-owner cleanup, and propagate through the actor run
ref so `run_grpo_single_controller.py` detects a silent peer by the refit
deadline and exits nonzero within `refit_timeout_s + cleanup_timeout_s` plus a
fixed small scheduler/test margin. An explicit ready failure short-circuits
without consuming the remaining refit budget.
Hold `_rollout_permitted` false through COMMIT, version publication, and every
backend RESUME ACK; only the exact `OPEN_ROLLOUT` leaf may set it. A missing,
async `OPEN_COLLECTION`, malformed, silent, or failed Single Controller open
proof leaves it false and is terminal. Conversely, synchronous driver tests
require the two-step activation variant and assert neither open operation nor a
synthetic final ACK exists. All modes consume the remaining original refit
deadline, including activation.
The launcher must release all GPU-owning resources even when cleanup itself
fails or the original refit exception races actor death. Extend
the ActorArgs serialization test to prove the full context/signing key is
absent, a forged or bare projection cannot mint/decode transaction authority,
and one valid driver-minted authority instantiates multiple successive actor
transactions with fresh versions/fences. Create
`tests/functional/refit_failure_exit.py` with sync, async, and Single Controller
modes and require each injected transport/phase to terminate nonzero rather than
hang. Timing assertions distinguish failure detection from teardown: silent-peer
detection—including post-COMMIT publication/resume/open—is bounded by the one
original `refit_timeout_s`, cleanup by its independent deadline,
and total process exit by their sum plus the fixed margin.
In `tests/unit/algorithms/test_grpo_sync.py`, exercise the three synchronous
`grpo_sync.py` refit call sites (startup, ordinary every-version update, and the
later recovery/update path). Each receives one driver-held full context plus the
same startup digest/cadence plans and finite timeout through an explicit shared
state object; no call may reselect or reconstruct from config. A foreign/missing
context, pending sender plus failed receiver, silent peer, or malformed result
must trigger bounded terminal cleanup and escape the synchronous loop. The
functional `sync` mode sets `async_grpo.enabled=False` and proves it executes
the supported vLLM path rather than the async or Single Controller
implementation. PR3294's current `defer_wake_for_save`/
`wake_carries_weight_updates()` branch is Megatron-generation-only and therefore
is not advertised as an enabled semantic path. Add an absent-policy byte-for-
byte regression for its existing offload/save/deferred-wake/version-setter
ordering, and prove semantic Megatron generation is rejected before reaching
that branch or allocating resources.
Add wrapper tests that convert a fenced adapter receipt into one
`CheckpointReceiptResult` after independently running
`verify_checkpoint_load_receipt()`, one
`DestinationLoadResult` per expected load operation and one
`DestinationFinalizeResult` per expected finalizer group. Cover all four valid
load/finalizer cardinalities: shared/shared, shared/separate,
separate/model-wide, and separate/separate. Mismatched plan, transaction,
operation or finalizer ID, covered-set digest, rank, execution kind, source
version, target version, live source-fence-set digest, missing fence completion,
extra/duplicate proof, or an empty proof is
fatal. A separate engine-envelope COMMIT proof cannot substitute for any load
or finalizer proof.

Run the existing API-coupled regression modules—not only the new transaction
fixtures—through both explicit legacy and enabled semantic paths:
`test_grpo_refit_supervision.py`, `test_grpo_checkpoint_engine.py`,
`test_reconcile_communicator.py`, `test_refit_recovery.py`, the vLLM generation/
lifecycle/loader/NIXL/quant-backend tests, and the Megatron/DTensor checkpoint-
engine policy tests. Update direct call sites to the exact disjoint protocol
they exercise; do not loosen the new signatures or add a default schedule to
make an old test pass. Their absent-policy fixtures snapshot the legacy shape,
while enabled fixtures assert the bound schedule/result/deadline contract.

- [ ] **Step 2: Run integration tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_collective_refit_supervision.py tests/unit/weight_sync/test_reshard_rebuild.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_grpo_sync.py tests/unit/algorithms/test_async_utils.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py -k 'refit and (failure or fatal or transaction or timeout or malformed)'`

Run the new synchronous and Single Controller integration suites unfiltered so
startup/update cases with lifecycle-oriented names remain mandatory:
`uv run --no-sync pytest -q tests/unit/algorithms/test_grpo_sync.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_collective_refit_supervision.py tests/unit/weight_sync/test_reshard_rebuild.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py`

Run the existing directly coupled controller/communicator APIs unfiltered:
`uv run --no-sync pytest -q tests/unit/algorithms/test_grpo_refit_supervision.py tests/unit/algorithms/test_grpo_checkpoint_engine.py tests/unit/single_controller/test_refit_recovery.py tests/unit/weight_sync/test_reconcile_communicator.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/models/policy/test_dtensor_checkpoint_engine.py tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py tests/unit/models/policy/test_megatron_checkpoint_engine.py`

Run the backend result-contract suites unfiltered:
`uv run --no-sync pytest -q tests/unit/models/policy/test_worker_refit_signatures.py tests/unit/models/policy/test_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/policy/test_dtensor_worker.py tests/unit/models/policy/test_dtensor_worker_v2.py`

Run the vLLM facade contract as RED in its pinned dependency group:
`uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_collective_rpc.py tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py --vllm-only`

Run the existing vLLM API-coupled modules as RED in that same pinned group:
`uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_refit_lifecycle.py tests/unit/models/generation/test_vllm_refit_loader.py tests/unit/models/generation/test_vllm_nixl_worker.py tests/unit/models/generation/test_vllm_quant_backend.py --vllm-only`

Run the remote-sparse protocol suites unfiltered as RED:
`uv run --no-sync pytest -q tests/unit/models/generation/test_vllm_sparse_refit.py tests/unit/models/generation/test_vllm_sparse_delta.py tests/unit/models/policy/test_megatron_remote_sparse_refit.py tests/unit/utils/test_weight_transfer_stream.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Expected: at least one current sync/async/Single Controller or transport path
returns normally, observes a receiver failure late, hangs on a silent peer, or
accepts malformed success; the new authority/timeout/exit tests are RED.

- [ ] **Step 3: Replace sequential waits and swallowed errors with transactions**

```python
try:
    transaction.run(source_refs=futures_train, destination_refs=futures_inference)
except BaseException:
    collector_terminal_cleanup()
    raise
```

Worker methods return the phase-appropriate `RefitWorkerResult` variant or
raise; source and destination transfer wrappers preserve execution-batch and
direction identity and never translate a send/load/finalize exception to
`False`. Thread a positive `refit_timeout_s` through legacy setup and every initial/subsequent refit. Move async startup under terminal cleanup, re-raise after telemetry flush, make cache invalidation failure fatal for every backend, and leave the collector/generation gate closed after a poisoned update. Launcher cleanup is bounded and never replaces the original exception.
Begin the async transaction and its monotonic deadline before the driver submits
`trajectory_collector.prepare_for_refit.remote()`. Pass its authenticated
lifecycle bundle, one-shot remaining-budget wire, and cancellation authority
into the serialized collector; supervise that outer ref under the authoritative
driver deadline. The collector derives one locally monotonic deadline capped by
the received budget/cancel authority and passes the same object through every
nested backend ref for pause, update, invalidation, and resume. It never invokes
a timeout normalizer per phase. A silent/dead outer collector or nested backend
therefore reaches the same cancellation/poison path without extending the
budget, and serialized collector/generation state never contains full context,
signing key, or cached minting authority.
Update the abstract policy/generation interfaces and add the distinct enabled-
only `SemanticPrecisionWeightSynchronizer`/
`SemanticPrecisionDirectRefitHandle` protocols in `weight_sync/interfaces.py`
with every implementation together. Keep all existing legacy synchronizer/
direct method names and signatures unchanged. The semantic high-level methods
take the exact keyword-only precommit dispatch schedule,
shared transaction deadline, and cancellation authority for communicator init,
rebuild, initial load, and later refit dispatch; they release one exact next-
operation manifest to the corresponding operation facade, whose worker methods
take only their matched node/bundle. Facades return ObjectRef batches of
typed results; no semantic implementation relies on mutable prepared state or
returns legacy `None`/boolean. Enabled factory selection validates the semantic
protocol before any RPC; absent-policy PPO/distillation and direct callers use
only the untouched legacy protocol. Task 11 owns the dependency-neutral value types
and supervisor, while these concrete signature changes land only here with all
overrides.
Make capacity and metadata discovery first-class schedule waves. Before
checkpoint-engine communicator init, dispatch the exact enabled checkpoint
capacity facade/leaf RPC, validate the complete worker-produced
`CapacityDiscoveryResult` set, and derive
the canonical minimum/digest. Before auto-sized IPC allocation, do
the same through the policy capacity facade/leaf RPC. Never call the legacy raw
`Policy.get_free_memory_bytes()` or checkpoint integer method and never wrap an
integer in a synchronizer-created result. For colocated IPC release
the exact schedule `SOURCE_PREPARE`, destination weight-memory prepare,
capacity discovery, transfer/finalize, `SOURCE_RESTORE`, then KV-memory prepare.
An explicit buffer
config bypasses only discovery, never either memory-lifecycle operation or the
applicable source lifecycle. After checkpoint `PREPARE_METADATA`, validate every rank's
`PreparedMetadataOperationResult` and aggregate digest before resolving and
releasing the deferred process-group-init manifest. No synchronizer may inspect
raw values, use sequential bare `ray.get`, or synthesize an artifact result.
For runtime collective/NCCL communicator creation, add
`RayVirtualCluster.get_master_address_and_port_for_refit(...)`, which accepts
the exact rendezvous authorization, transaction deadline, and cancellation
authority, publishes each probe ref before waiting, and returns a validated
`RendezvousDiscoveryResult`. Collision retries/backoff consume the same
remaining budget and pre-authorized attempt IDs. This is a separate runtime
API: it neither retains nor resurrects Task 5's sealed construction owner, and
the old construction/absent-policy method remains unchanged.
Replace enabled Single Controller and driver recovery calls to legacy
`reconcile_communicator()` with
`reconcile_semantic_precision_communicator(schedule, deadline, cancellation)`.
It consumes the authorized rendezvous/rebuild operations through the combined
supervisor and preserves the first typed cause. Do not run enabled reconcile in
`asyncio.to_thread` or allow its old `(absent, force)` shape to bypass the
transaction. Keep the legacy method/call sites only on the explicit absent-
policy branch.
Split the refit contract into explicit driver-facade, dispatching-worker, and
leaf-worker protocols for both policy and generation. Driver methods take the
ordered root manifest and return ObjectRef batches; an outer vLLM actor
takes one matched dispatch branch and returns a direct typed batch after
validated broadcast-safe group routing; a leaf method selects its exact
target-bound child, takes only that singular bundle, and returns a direct typed
batch. Wire `Policy`, Megatron, DTensor v1/v2, and every generation
facade/dispatcher/leaf against the correct protocol
and let pyrefly reject cross-layer argument narrowing, inheritance, or result
substitution. Do not express the split only through a covariant result generic;
the input types differ intentionally and must be statically distinct.
Update the ModelOpt `VllmInternalWorkerExtension` real-quant subclass at the
same time: either remove its `prepare_refit_info` override in favor of the
common authorized hook or give it the exact bundle signature and pre-side-
effect validation. Do not leave a legacy one-argument concrete override.
On enabled paths, pass the transaction-minted
`BoundRefitPrecommitDispatchSchedule` explicitly into each high-level
synchronizer/direct handle. It exposes only the exact next operation's
`BoundRefitFacadeDispatchManifest` to a driver facade, marks that step consumed
once, requires the schedule to be exhausted before success, and rejects RESUME
or any ACTIVATE-phase authorization, including served-version publication.
Existing transport-specific precommit methods
are split so their precommit call returns after end/finalize without resuming.
After the coordinator durably records COMMIT/STARTUP_READY, the transaction
mints a fresh one-shot `BoundRefitActivationDispatchSchedule`, calls the
separate activation facades, and supervises exact destination/owner
`PUBLISH_SERVED_VERSION` ACKs for the committed target version before releasing
and supervising backend RESUME. It then releases the exact mode-specific gate:
async `OPEN_COLLECTION`, Single Controller `OPEN_ROLLOUT`, or no gate RPC for
synchronous driver mode. These are not given a new budget: every operation uses
the remaining original absolute transaction deadline and identical
cancellation authority; a silent or malformed participant is fatal.
The sole runner is the mode-neutral
`BoundRefitActivationDispatchSchedule.run_to_serving_permission()`; its
dispatcher validates the exact schedule discriminant before the first RPC and
routes only async `OPEN_COLLECTION`, Single Controller `OPEN_ROLLOUT`, or the
synchronous no-open terminal shape. No operation-specific async-only shortcut
exists.
Refactor `TrajectoryCollector.set_weight_version()` into an enabled-only stamp RPC and a
separate activation RPC: `publish_refit_served_version()` records only the
authorized target version, backend resume touches no collection gate, and only
`open_collection_after_refit()` sets `_generation_limit_cleared` after all
backend resume proofs. Add the corresponding Single Controller leaf so only
`open_rollout_after_refit()` sets `_rollout_permitted`; synchronous driver mode
has no open leaf. Keep the old
combined method only for the explicit legacy path. Before the
first RPC of each step, compare its ordered outer dispatch identities and
cardinality with the canonical live target registry and atomically reject any
gap, duplicate, alias drift, order change, or foreign transaction. Select one
least-authority root `BoundRefitWorkerDispatchNode` per outer target and reject
the wrong branch/leaf discriminant before RPC. A dispatching worker accepts only
a branch and discriminates its exact node kind before routing. A direct branch
exposes children; a broadcast branch has no `.ordered_children` and instead
exposes only its pre-minted envelope and committed child IDs. For vLLM
0.25.1/0.28.0 public group RPC this means requiring the broadcast branch and
verifying the transaction-pre-minted
`BoundRefitBroadcastGroupEnvelope` against the branch and live registry, then
broadcasting that identical immutable argument unchanged; each leaf recomputes the
envelope digest, derives its exact target identity, and selects exactly one
matching leaf; it passes only that leaf's bundle to the mutation endpoint. Only
that leaf then invokes `init_collective`/communicator-rebuild,
refit-specific prepare/pause/resume, or update with its own singular bundle and
validates it again at the mutation boundary. An adapter without the declared
broadcast-safe leaf selector capability fails
closed during preflight; it is never monkey-patched. Never broadcast the full
root manifest/ancestor subtree or reuse a rank-bound bundle for another worker.
Refit prepare may cache physical buffers but never
caches transaction authority for a later call. Introduce
the single canonical `BroadcastSafeRefitDispatchCapability` protocol in
`models/generation/interfaces.py`. `precision_adapter/base.py` imports and
re-exports/uses that exact nominal object; it must not redefine a lookalike.
Make registry selection require its exact public capability, and implement/version-
test the identical-argument `LLM.collective_rpc` paths in
`v0251.py`/`v0280.py`. The adapter validates the complete ordered group, emits
one group RPC with the authenticated immediate-child envelope, and returns
target-keyed leaf proofs to the combined supervisor. This covers vLLM's internal
FlashInfer-TRTLLM path; it must not import standalone TRTLLM. No implementation
may call a private runtime method, broadcast a root/ancestor authority, permit a
leaf to consume a sibling entry, or wrap an untyped/boolean completion as success.
Introduce `prepare_generation_for_refit()` for this contract. Leave only
ordinary non-refit `prepare_for_generation()` wake call sites unchanged and
transaction-free; migrate every supported vLLM weight/KV-memory wake that
participates in a refit to the authorized refit-only method. The unsupported
Megatron-only PR3294 branch remains unchanged behind the absent-policy path. Add the parallel
`invalidate_kv_cache_for_refit()` entrypoint and leave ordinary cache
invalidation/`finish_generation()` transaction-free; a backend-internal
invalidation fused into an authorized update consumes that update's exact
per-result authorization. Preserve the old absent-policy overload only behind the exact
internal legacy sentinel. `VllmGeneration` forwards the matched root node on
each refit-worker dispatch; dispatching vLLM workers route child nodes by exact
internal identity, and only backend leaves receive a bundle. No facade or
dispatcher may infer a manifest, node, or worker bundle from instance state.
Refactor the supported IPC policy helpers in `models/policy/utils.py` to accept
the exact authorization sub-bundle, shared remaining deadline, and cancellation
authority. Replace per-bucket/init/lock/NCCL/engine unbounded gets with bounded
combined drains, release locks and temporary handles on every `BaseException`,
and return the exact mixed source-plus-destination result batch to the remote
policy worker. The helper cannot return `None`, swallow an engine result, or
derive authority from its caller's mutable state.
Install the same combined-future supervisor in direct collective, vLLM internal
FlashInfer-TRTLLM, and both remote-sparse synchronizers
rather than adapting their current sequential gets into transport-specific
loops. Thread the remote-sparse authorization bundle/deadline/cancel token all
the way through `megatron_remote_sparse_refit`, sparse codec and stream/ZMQ/HTTP
helpers, `vllm_sparse_refit` endpoints, queued apply, and flush. Canonical
payload headers bind transaction/version/fence/chunk identities; the receiver
validates before enqueue and records a poison tombstone on timeout so a late
apply cannot mutate this or a later transaction. Carry the same authorization
into `vllm_backend.update_weights_from_decoded_sparse_payload` and
`VllmSparseDeltaApplier`; both check the tombstone and live fence before the
first write and before finalization, then return direct typed load/finalizer
proofs. Cap cached-executor work,
conditions, collectives, HTTP retries/backoff, and ZMQ waits by the remaining
shared deadline and convert successful loads/finalizers to exact typed proofs,
never metric dicts or booleans. The vLLM internal FlashInfer-TRTLLM mutation
boundary follows the same rule inside `vllm_backend.py`; no standalone TRTLLM or
Megatron generation facade participates. At the source boundary,
`LMPolicy.broadcast_weights_for_collective()` supplies the common execution
envelope and normalizes Megatron, DTensor v1, and DTensor v2 only by requiring
their backend workers to return exact typed source-transfer batches; Task 11's
legacy `None` migration normalizer is forbidden here. Move
`offload_before_refit()` and `offload_after_refit()` into the schedule as the
exact `SOURCE_PREPARE` and `SOURCE_RESTORE` waves. Dispatch their refs through
the combined supervisor with the same absolute deadline and cancellation/
poison state; remove enabled-path bare `ray.get` calls. Restore must complete
before COMMIT, while failure in either wave prevents COMMIT and leaves no
out-of-band late source action. In Single Controller, mint the
immutable `BoundRefitTransactionAuthority` against the driver-held full context
and static plans before actor serialization, send only its authenticated safe
wire plus plan/projection wires and public verifier, and make both the setup-time
special initial sync and actor `_sync_weights()` instantiate the concrete typed
transaction from that verified authority plus the then-current versions/fences
and shared finite timeout. The enabled vLLM initial sync and every later actor
sync use this path; no unsupported special generation path exists. The actor cannot mint an authority from a bare
projection. It re-raises the first failure after poison and unified owner cleanup;
the launcher awaits the run ref and exits nonzero.
Create one immutable synchronous refit runtime state after controller-side
context/plan validation and authority minting, and thread it through all three
`grpo_sync.py` `refit_policy_generation` calls. It carries the full context only
in the driver process, the verified authority, startup precondition digest,
canonical cadence plan groups, safe worker projection, and finite timeout; no
call re-runs discovery or adapter selection.
List both newly typed Python files,
`tests/unit/algorithms/test_grpo_sync.py` and
`tests/functional/refit_failure_exit.py`, explicitly in `pyrefly.toml`; do not
rely on directory discovery to include them.

Update `create_weight_synchronizer()` to pass the exact positive
`refit_timeout_s`, bound authority/projection, and context ID through every
enabled vLLM transport constructor branch. Factory tests inspect the exact
kwargs for every transport and reject a branch that substitutes a default or
drops the authority.

Checkpoint-engine loading uses this same worker execution bundle rather than a
parallel boolean protocol. Each batch is associated with its bound physical
destination owners; main, MTP, and Eagle allocations are loaded and finalized
according to their independent realized identities. Empty or filtered RPC
results never satisfy an acknowledgement set.
Refactor `vllm/collective_rpc.py` so `checkpoint_engine_rpc_async()` recursively
discovers nested ObjectRef/future/awaitable leaves, schedules them concurrently,
and resolves/validates ready leaves under the same remaining deadline and
cancellation authority. Eliminate sequential recursion and unbounded
`asyncio.wrap_future`, `asyncio.to_thread(ray.get, ...)`, and bare awaits. On
first failure it cancels or nonjoining-abandons every sibling, poisons the
transaction, and prevents late leaves from contributing an ACK.

Destination worker wrappers first wait for and validate the adapter-local
completion fence. For a checkpoint-served direct body they run
`verify_checkpoint_load_receipt()` against the expected immutable evidence and
bound destination consumption, retain the complete receipt plus its canonical
digest in `CheckpointReceiptResult`, then compare every startup/destination plan, transaction,
load operation, finalizer group, covered-member digest, rank, execution kind,
source version, target version, and live source-fence-set digest with Task 7's
expected acknowledgement sets and Task 11's runtime envelope. Only then do
they return a non-empty `RefitWorkerResultBatch` with the exact checkpoint,
load, and finalizer proofs. Adapters never return a boolean or unchecked readiness object
directly to `supervise_refit_futures()`.

Replace boolean-driven refit selection with Task 7's validated cadence plans:

```python
startup_plans, refit_plans = build_canonical_plan_groups(context, ...)
authority = bind_refit_transaction_authority(
    context,
    startup_plans=startup_plans,
    refit_plans=refit_plans,
    projection=plan_bound_projection,
    signer=controller_signer,
    expected_runtime_owner_id=runtime_resource_owner.runtime_owner_id,
)
activation_dispatcher = build_refit_activation_mode_dispatcher(
    mode=activation_mode,
    runtime_owner=runtime_resource_owner,
    transaction_authority=authority,
    publication_facades=publication_facades,
    resume_facades=resume_facades,
    async_collector=trajectory_collector_if_async,
    single_controller=adopted_single_controller_if_applicable,
)
if canonical_startup_member_count(startup_plans) > 0:
    initial_source_version, initial_fences = source.bind_startup_version(...)
    startup_transaction = StartupLoadTransaction.from_authority(
        authority,
        plans=startup_plans,
        source_version=initial_source_version,
        target_version=initial_generation_version,
        source_fences=initial_fences,
    )
    startup_digest = startup_transaction.run_to_startup_ready()
else:
    startup_transaction = None
    startup_digest = canonical_no_startup_precondition_digest(authority)
if canonical_refit_member_count(refit_plans) > 0:
    source_version, source_fences = source.bind_refit_version(...)
    committed = RefitTransaction.from_authority(
        authority,
        plans=refit_plans,
        required_startup_digest=startup_digest,
        source_version=source_version,
        target_version=next_generation_version,
        source_fences=source_fences,
    ).run_to_commit(...)
    activation = committed.mint_activation_schedule()
elif canonical_startup_member_count(startup_plans) == 0:
    raise ValueError("startup and refit groups are both empty")
else:
    assert startup_transaction is not None
    activation = startup_transaction.mint_activation_schedule_after_startup_ready()
activation.run_to_serving_permission(dispatcher=activation_dispatcher)
```

`trains_mtp`, `has_refit_draft_weights`, `loss_scaling_factor`, and
`detach_heads` may help discover or cross-check graph declarations, but none is
sufficient to decide owner mutability or cadence. Run a startup group once
before opening collection/generation only when it contains at least one exact
plan member; otherwise emit the canonical no-startup precondition digest and do
not call the startup supervisor. Run the initial refit only when its canonical
member/ACK set is nonempty. A one-sided real startup transaction is valid when
its nonempty side and expected ACK sets are exact; a nonempty successful startup
with no every-version work opens serving directly, while both groups empty is
invalid. Preserve the startup/no-startup digest as a repeated refit precondition
whenever repeated work exists. Register every graph with a mutable served member as an
every-version member, including a checkpoint-served body that contains a
canonical alias to a mutable training-runtime owner, and advance its served version with
the main COMMIT.
Reject a missing source/destination binding on a derived owning rank. Accept a
mutable training-only auxiliary with no destination, a fully evidenced
checkpoint-served auxiliary body with no source-wire plan while preserving any
canonical alias obligations, and absence on a derived non-owning PP rank. Never
silently disable MTP/speculative decoding because an
owning-rank drafter module was not realized.

- [ ] **Step 4: Run all sync/async/refit regression gates**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync tests/unit/distributed/test_refit_watchdog.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_grpo_sync.py tests/unit/algorithms/test_async_utils.py tests/unit/experience/test_rollout_manager.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py`

Run: `uv run --no-sync pytest -q tests/unit/algorithms/test_grpo_refit_supervision.py tests/unit/algorithms/test_grpo_checkpoint_engine.py tests/unit/single_controller/test_refit_recovery.py tests/unit/weight_sync/test_reconcile_communicator.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/models/policy/test_dtensor_checkpoint_engine.py tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py tests/unit/models/policy/test_megatron_checkpoint_engine.py`

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_worker_refit_signatures.py tests/unit/models/policy/test_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/policy/test_dtensor_worker.py tests/unit/models/policy/test_dtensor_worker_v2.py`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_collective_rpc.py tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py --vllm-only`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py --vllm-only`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_refit_lifecycle.py tests/unit/models/generation/test_vllm_refit_loader.py tests/unit/models/generation/test_vllm_nixl_worker.py tests/unit/models/generation/test_vllm_quant_backend.py --vllm-only`

Run: `uv run --no-sync pytest -q tests/unit/models/generation/test_vllm_sparse_refit.py tests/unit/models/generation/test_vllm_sparse_delta.py tests/unit/models/policy/test_megatron_remote_sparse_refit.py tests/unit/utils/test_weight_transfer_stream.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Run on the pinned real-Ray image after the ActorArgs/transaction schema change:
`uv run --no-sync pytest -q tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py`

Run: `uv run --no-sync pyrefly check nemo_rl/weight_sync nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/collective_rpc.py nemo_rl/models/generation/vllm/checkpoint_engine.py nemo_rl/models/generation/vllm/refit_loader.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/workers/checkpoint_engine.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/async_utils/trajectory_collector.py nemo_rl/experience/rollout_manager.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo.py examples/run_grpo_single_controller.py tests/unit/weight_sync tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_grpo_sync.py tests/unit/algorithms/test_async_utils.py tests/unit/experience/test_rollout_manager.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_collective_rpc.py tests/functional/refit_failure_exit.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py`

Run: `uv run --no-sync pyrefly check nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/transaction.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_collective_refit_supervision.py tests/unit/weight_sync/test_reshard_rebuild.py`

Run: `uv run --no-sync pyrefly check nemo_rl/distributed/virtual_cluster.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/weight_sync/test_reconcile_communicator.py tests/unit/algorithms/test_grpo_refit_supervision.py tests/unit/algorithms/test_grpo_checkpoint_engine.py tests/unit/single_controller/test_refit_recovery.py tests/unit/models/policy/test_dtensor_checkpoint_engine.py tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py tests/unit/models/policy/test_megatron_checkpoint_engine.py`

Run: `uv run --no-sync pyrefly check tests/unit/models/generation/test_vllm_checkpoint_engine.py`

Run: `uv run --no-sync pyrefly check tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_refit_lifecycle.py tests/unit/models/generation/test_vllm_refit_loader.py tests/unit/models/generation/test_vllm_nixl_worker.py tests/unit/models/generation/test_vllm_quant_backend.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/precision_adapter/base.py nemo_rl/models/generation/vllm/precision_adapter/registry.py nemo_rl/models/generation/vllm/precision_adapter/v0251.py nemo_rl/models/generation/vllm/precision_adapter/v0280.py tests/unit/models/generation/test_vllm_precision_adapter.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/utils.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py tests/unit/models/policy/test_worker_refit_signatures.py tests/unit/models/policy/test_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/policy/test_dtensor_worker.py tests/unit/models/policy/test_dtensor_worker_v2.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/modelopt/models/generation/vllm_quant_backend.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/vllm_sparse_refit.py nemo_rl/models/generation/vllm/vllm_sparse_delta.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/policy/workers/megatron_remote_sparse_refit.py nemo_rl/utils/weight_transfer_http.py nemo_rl/utils/weight_transfer_sparse_codec.py nemo_rl/utils/weight_transfer_stream.py nemo_rl/utils/weight_transfer_zmq.py tests/unit/models/generation/test_vllm_sparse_refit.py tests/unit/models/generation/test_vllm_sparse_delta.py tests/unit/models/policy/test_megatron_remote_sparse_refit.py tests/unit/utils/test_weight_transfer_stream.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/transaction.py nemo_rl/weight_sync/factory.py nemo_rl/weight_sync/direct_collective.py nemo_rl/weight_sync/ipc_weight_synchronizer.py nemo_rl/weight_sync/collective_weight_synchronizer.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_collective_refit_supervision.py tests/unit/weight_sync/test_reshard_rebuild.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/distributed/virtual_cluster.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/weight_sync/test_reconcile_communicator.py tests/unit/algorithms/test_grpo_refit_supervision.py tests/unit/algorithms/test_grpo_checkpoint_engine.py tests/unit/single_controller/test_refit_recovery.py tests/unit/models/policy/test_dtensor_checkpoint_engine.py tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py tests/unit/models/policy/test_megatron_checkpoint_engine.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/collective_rpc.py nemo_rl/models/generation/vllm/checkpoint_engine.py nemo_rl/models/generation/vllm/refit_loader.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/utils.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/checkpoint_engine.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_collective_rpc.py tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/unit/models/policy/test_worker_refit_signatures.py tests/unit/models/policy/test_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/policy/test_dtensor_worker.py tests/unit/models/policy/test_dtensor_worker_v2.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/async_utils/trajectory_collector.py nemo_rl/experience/rollout_manager.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo.py examples/run_grpo_single_controller.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_grpo_sync.py tests/unit/algorithms/test_async_utils.py tests/unit/experience/test_rollout_manager.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/functional/refit_failure_exit.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py pyrefly.toml`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/modelopt/models/generation/vllm_quant_backend.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/precision_adapter/base.py nemo_rl/models/generation/vllm/precision_adapter/registry.py nemo_rl/models/generation/vllm/precision_adapter/v0251.py nemo_rl/models/generation/vllm/precision_adapter/v0280.py tests/unit/models/generation/test_vllm_precision_adapter.py`

Run: `uv run --no-sync pre-commit run --files tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_refit_lifecycle.py tests/unit/models/generation/test_vllm_refit_loader.py tests/unit/models/generation/test_vllm_nixl_worker.py tests/unit/models/generation/test_vllm_quant_backend.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/vllm/vllm_sparse_refit.py nemo_rl/models/generation/vllm/vllm_sparse_delta.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/policy/workers/megatron_remote_sparse_refit.py nemo_rl/utils/weight_transfer_http.py nemo_rl/utils/weight_transfer_sparse_codec.py nemo_rl/utils/weight_transfer_stream.py nemo_rl/utils/weight_transfer_zmq.py tests/unit/models/generation/test_vllm_sparse_refit.py tests/unit/models/generation/test_vllm_sparse_delta.py tests/unit/models/policy/test_megatron_remote_sparse_refit.py tests/unit/utils/test_weight_transfer_stream.py`

Expected: all commands pass; sync, async, and Single Controller subprocesses
detect silent peers by the refit deadline and exit nonzero after bounded cleanup
within the two deadlines plus the fixed scheduler margin. Ready failures exit
without waiting out the refit timeout.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync/interfaces.py nemo_rl/weight_sync/transaction.py nemo_rl/weight_sync/factory.py nemo_rl/weight_sync/direct_collective.py nemo_rl/weight_sync/ipc_weight_synchronizer.py nemo_rl/weight_sync/collective_weight_synchronizer.py nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py tests/unit/weight_sync/test_refit_transaction.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_collective_refit_supervision.py tests/unit/weight_sync/test_reshard_rebuild.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py
git add nemo_rl/models/generation/vllm/checkpoint_engine.py nemo_rl/models/generation/vllm/refit_loader.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/models/policy/workers/checkpoint_engine.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py nemo_rl/algorithms/async_utils/trajectory_collector.py nemo_rl/experience/rollout_manager.py nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/setup.py examples/run_grpo.py examples/run_grpo_single_controller.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_grpo_sync.py tests/unit/algorithms/test_async_utils.py tests/unit/experience/test_rollout_manager.py tests/unit/single_controller/test_setup.py tests/unit/single_controller/test_single_controller_actor.py tests/unit/single_controller/test_entrypoint.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/functional/refit_failure_exit.py tests/functional/test_single_controller_resource_handoff_ray.py tests/functional/test_single_controller_tq_handoff_ray.py
git add nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/collective_rpc.py nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/utils.py nemo_rl/models/policy/workers/base_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_collective_rpc.py tests/unit/models/policy/test_worker_refit_signatures.py tests/unit/models/policy/test_utils.py tests/unit/models/policy/test_megatron_worker.py tests/unit/models/policy/test_dtensor_worker.py tests/unit/models/policy/test_dtensor_worker_v2.py pyrefly.toml
git add nemo_rl/models/generation/vllm/vllm_sparse_refit.py nemo_rl/models/generation/vllm/vllm_sparse_delta.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/policy/workers/megatron_remote_sparse_refit.py nemo_rl/utils/weight_transfer_http.py nemo_rl/utils/weight_transfer_sparse_codec.py nemo_rl/utils/weight_transfer_stream.py nemo_rl/utils/weight_transfer_zmq.py tests/unit/models/generation/test_vllm_sparse_refit.py tests/unit/models/generation/test_vllm_sparse_delta.py tests/unit/models/policy/test_megatron_remote_sparse_refit.py tests/unit/utils/test_weight_transfer_stream.py
git add nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/modelopt/models/generation/vllm_quant_backend.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
git add nemo_rl/models/generation/vllm/precision_adapter/base.py nemo_rl/models/generation/vllm/precision_adapter/registry.py nemo_rl/models/generation/vllm/precision_adapter/v0251.py nemo_rl/models/generation/vllm/precision_adapter/v0280.py tests/unit/models/generation/test_vllm_precision_adapter.py
git add nemo_rl/distributed/virtual_cluster.py tests/unit/distributed/test_virtual_cluster.py tests/unit/distributed/test_virtual_cluster_batch_ports.py tests/unit/weight_sync/test_reconcile_communicator.py tests/unit/algorithms/test_grpo_refit_supervision.py tests/unit/algorithms/test_grpo_checkpoint_engine.py tests/unit/single_controller/test_refit_recovery.py tests/unit/models/policy/test_dtensor_checkpoint_engine.py tests/unit/models/policy/test_dtensor_v2_checkpoint_engine.py tests/unit/models/policy/test_megatron_checkpoint_engine.py
git add tests/unit/models/generation/test_vllm_generation.py tests/unit/models/generation/test_vllm_refit_lifecycle.py tests/unit/models/generation/test_vllm_refit_loader.py tests/unit/models/generation/test_vllm_nixl_worker.py tests/unit/models/generation/test_vllm_quant_backend.py
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

def test_refit_hot_path_never_calls_topology_or_source_discovery() -> None:
    adapter = instrumented_mixed_adapter(fail_on_resolution_or_discovery=True)
    adapter.refit(update_a())
    adapter.refit(update_b())
    assert adapter.topology_resolution_calls == 0
    assert adapter.source_discovery_calls == 0

def test_mixed_owner_never_retransfers_cached_frozen_contributors() -> None:
    metrics = run_mixed_owner_updates("A", "B", "C")
    assert metrics.frozen_wire_bytes_by_update == (metrics.startup_frozen_bytes, 0, 0)
    assert metrics.compose_count_by_update == (1, 1, 1)
    assert metrics.finalize_count_by_update == (1, 1, 1)

def test_transaction_metrics_cover_the_collective_critical_path() -> None:
    metrics = completed_transaction_metrics()
    assert set(metrics) >= {"total_s", "source_prepare_s", "wire_s", "destination_finalize_s", "commit_s", "peak_allocated_bytes", "peak_reserved_bytes"}
```

- [ ] **Step 2: Run performance-contract tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py -k metrics`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_mxfp8_refit_performance_contract.py -k 'reuse or metrics' --vllm-only`

Expected: missing counters/metrics or per-refit allocation/scan is observed.

- [ ] **Step 3: Integrate measured PR #3669 optimization behind the semantic contracts and preserve PR #3294 through absent-policy regression**

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

Cache the local execution plan, source routes, BF16 boundary buffers, owner slices, row permutations, and Task 7 immutable-contributor buffers after binding. Use batched MXFP8 expert quantization/shuffle for homogeneous owners and direct component copy for compatible native MXFP8. Preserve the reference per-owner path only for numeric comparison. Record bytes by owner cadence and fail the performance gate if any frozen contributor is retransferred after its verified startup stage. The benchmark records paired randomized samples, maximum-rank critical path, warmup/stability condition, environment/SHA/plan/cache digest, and 95% bootstrap bounds for p50/p95 ratios.

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

Add both training modes for all five production families and exact Phase 1 fixture assertions for shared experts, routers, QKVO, dense layers, and output heads remaining BF16. Assert separately that built-in main-model roles never select MTP/draft addresses; BF16 defaults apply only to the training or rollout endpoint in which an auxiliary participates. For every case, preserve the selection and BF16-fence serialization exactly across its BF16-source or MXFP8-source Phase 2 binding.

For Qwen3.5, Lightning, Ultra, Qwen3.8, and GLM fixtures that advertise MTP, exercise the full internal lifecycle matrix: mutable training-only, all-frozen `served_from_source`, mixed mutable/frozen `served_from_source`, and static `served_from_checkpoint` with complete immutable evidence. Add the same external speculative-drafter cases, including one different-family adapter. Assert that training-only graphs and directly owned checkpoint bodies create no source-wire plan, frozen training-runtime owners join the one-shot startup group, only mutable training-runtime owners join each repeated payload, and a checkpoint-served graph's cross-graph aliases inherit canonical authority. A mixed graph's target version remains coherent with main while its startup digest is a commit precondition. Owning-rank absence fails and non-owning PP-rank absence succeeds. Canonical aliases never duplicate source export, but destination load/finalization follows the independently proved load-owner and finalizer-group identities. Include loss-scaling-zero and detached-head fixtures that remain mutable. Keep these declarations in model/runtime configuration and internal manifests; do not add lifecycle fields to the public `precision_policy` example.

Add `--config-only` handling to `tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh` before it creates or deletes artifact directories. That mode invokes `tools/config_cli.py explain-precision` for the resolved recipe, prints the Phase 1 selection summary and Phase 2 unavailable markers, and exits without constructing Ray, allocating GPUs, or launching training.

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
        roles: [moe.routed_expert]
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
    selection = compile_precision_selection(policy, documentation_selection_topology())
    assert selection.selection_group_id
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
9. `tools/config_cli.py explain-precision RECIPE --format text|json` with exact graph/layer-universe/count/requested-format/BF16-fence output, `semantic_structure_digest`/`selection_group_id`, and explicit unavailable markers for Phase 2 source facts and final plans.
10. The two-phase lifecycle: source-neutral selection before construction, runtime source discovery and exact projection after construction, the four digest boundaries, and why Phase 2 cannot reselect policy or move BF16 fences.
11. The post-construction preflight's realized TE/config reconciliation, capability/transform/plan output, BF16→MXFP8 versus native-MXFP8→MXFP8 refit paths, and canonical/runtime layout distinction.
12. Supported/negative model-version matrix for vLLM 0.25.1 and 0.28.0.
13. Migration from singular `role`, `quantization_ignore_patterns`, first/last backend knobs, and hand-written TE recipes; mixing old and new sources fails.
14. Fatal refit behavior and where phase/rank/cause appear in logs.
15. MTP and speculative-drafter lifecycle diagnostics: explain graph-instance ID versus semantic graph path; independent zero-based layer universes; training-only mutable, all-frozen and mixed mutable/frozen served-from-source, and static checkpoint-served cases; graph/model identity, pinned revision, content/configuration/semantic-domain digests, complete checkpoint consumption receipts, and typed evidence source; owning versus non-owning PP ranks; canonical-alias source de-duplication versus destination fan-out; identical-storage versus synchronized-replica evidence; live source-version fences; and why zero loss scaling or detached heads do not prove freezing. State clearly that these are internal/model-runtime declarations, not additional public precision-policy selector fields. Explain that a static external draft has no runtime source partition, the one-shot serving gate covers frozen training authorities reached by any in-scope served member, and only mutable training authorities requested by such members repeat inside atomic every-version refit, including a checkpoint-served graph's cross-graph alias.
16. Performance knobs, one-time resolver/discovery cost, hot-path no-rescan guarantees, metrics, and the 5% gate.

Use this minimal public example verbatim:

```yaml
precision_policy:
  default: bf16
  scopes:
    - id: routed-experts-middle
      roles: [moe.routed_expert]
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
- Produces: reproducible correctness/fault/performance artifacts tied to branch, SHA, image digest, model revision, all four structure/selection/runtime-source/intent digests, final plan digests, auxiliary lifecycle/evidence records, expected rank-local ownership, raw samples, and job logs.

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
        for graph in case.get("auxiliary_graphs", []):
            assert set(graph) >= {"graph_instance_id", "model_identity", "graph_kind", "provenance", "rollout_participation", "derived_refit_requirement", "owners"}
            for owner in graph["owners"]:
                assert set(owner) >= {"owner_id", "source_mutability", "mutability_evidence_source", "derived_cadence", "rank_local_endpoint_ownership"}
            if graph["rollout_participation"] == "served_from_checkpoint":
                assert set(graph["immutable_evidence"]) == {"graph_instance_id", "model_identity", "pinned_checkpoint_revision", "checkpoint_content_digest", "model_config_digest", "semantic_domain_digest", "evidence_source"}
                verify_checkpoint_load_receipt(
                    expected=graph["immutable_evidence"],
                    bound=graph["bound_destination_plans"],
                    receipt=graph["checkpoint_load_receipt"],
                )
```

The scripts must support `--dry-run` and print the resolved cluster, account,
branch, exact SHA, container digest, nodes/GPUs, model revision,
`semantic_structure_digest`, `selection_group_id`, and explicit Phase 2
unavailable markers when no runtime is constructed. A submission dry run that
has bound runtime evidence also prints `runtime_source_digest`,
`intent_group_id`, final plan digest, auxiliary lifecycle/evidence summary,
per-owner mutability and derived cadence, expected rank-local endpoint
ownership, startup-precondition digest, command, time limit, and log directory
without submitting.

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

Run each script with `--dry-run`, review its resolved submission, then submit. Cover both training modes for all five production models, mixed BF16/MXFP8 boundaries, specified TP/EP/PP/padding rows, repeated A→B→C numeric refits, fresh-load logprob comparison, injected binding/transfer/finalize/commit/silent-peer failures, and at least twenty paired steady-state performance samples where p95 is claimed. The auxiliary matrix must include mutable training-only MTP/draft success without a destination plan, all-frozen source-served one-shot startup, mixed frozen/mutable source-served startup plus atomic repeated refit, static checkpoint evidence with exact component/load/finalizer/fence receipt and no direct-body source transfer, identical-storage canonical-alias source de-duplication, synchronized-replica startup and every-version live-fence enforcement, a checkpoint-served cross-graph alias to a mutable training authority, fatal missing owning-rank drafter storage, and valid absence on non-owning PP ranks. Assert frozen owners are never retransferred after startup and their startup digest remains a repeated-refit precondition. Monitor each new job for the required first five minutes and cancel/release resources immediately on a fatal failure.

- [ ] **Step 5: Evaluate hard gates and write the evidence report**

```python
assert every_numeric_case_passed
assert every_sync_and_async_fault_case_exited_nonzero_within_budget
assert refit_p50_ratio_upper_95ci <= 1.05
assert refit_p95_ratio_upper_95ci <= 1.05
assert generation_latency_ratio_upper_95ci <= 1.05
assert generation_throughput_ratio_lower_95ci >= 0.95
assert every_instantiated_training_auxiliary_was_accounted
assert every_phase_two_intent_preserved_phase_one_selection_and_bf16_fences_byte_exactly
assert no_runtime_result_added_removed_or_reshaped_a_phase_one_semantic_member
assert every_runtime_result_bound_the_exact_structure_selection_and_allocation_generation
assert no_refit_hot_path_called_topology_resolution_policy_compilation_or_source_discovery
assert every_source_served_frozen_owner_loaded_once_before_serving
assert only_mutable_training_authorities_requested_by_in_scope_served_members_joined_each_refit_payload
assert every_refit_verified_the_startup_precondition_digest
assert no_frozen_contributor_was_retransferred_after_startup
assert every_expected_owning_rank_bound_and_acknowledged
assert every_active_synchronized_replica_had_an_exact_live_source_version_fence
assert every_checkpoint_load_receipt_exactly_covered_bound_components_and_finalizers
assert no_canonical_source_owner_was_transferred_twice_for_aliases
assert every_distinct_destination_finalizer_ran_exactly_once_per_planned_group
```

The report maps retained code to PRs #3477/#3630/#3659/#3669/#3907/#3908/#3909/#3294 and lists minimal restack ranges. Correctness/transaction commits remain separate from independently measured performance commits. Do not update an existing PR until every hard gate passes on both clusters.

- [ ] **Step 6: Commit only durable validation scripts and completed evidence**

```bash
git add tests/functional/precision_policy_matrix.sh tests/functional/refit_transaction_fault_matrix.sh tests/functional/refit_performance_matrix.sh tests/fixtures/precision_policy/cluster_matrix.yaml tests/unit/precision_policy/test_cluster_matrix.py tests/functional/L1_Functional_Tests_GB200_MXFP8.sh tests/test_suites/performance_gb200.txt docs/performance/semantic-precision-refit-validation.md
git commit -s -m "test(refit): validate semantic precision matrix"
```
