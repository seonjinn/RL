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
- Every checkpoint-serving destination attests the artifact actually loaded and must match every immutable-evidence field before serving; stale tags, caches, paths, or mismatched evidence are fatal for vLLM, SGLang, and static draft backends alike.
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
| `nemo_rl/precision_policy/topology_resolver.py` | Task 4B-owned Phase 1 `resolve_selection_topology()` plus Phase 2 runtime producer selection, exact projection, and `bind_runtime_source_intents()` |
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
- Produces: frozen `CompiledGraphPrecisionSelection` records with lifecycle identity, immutable compact-domain assignments for participating endpoints, selected layer ranges, explicit BF16 fences, complete compact scope results plus logical cardinalities, semantic atomic closures, canonical graph `selection_id` values, and an ordered `CompiledPrecisionSelectionGroup` retaining the exact `ResolvedSelectionTopology` and carrying `semantic_structure_digest` plus `selection_group_id`. Retaining the topology, rather than only its non-invertible digest, lets Phase 2 prove whole-graph equality for static graphs with no runtime result. The group contains no source format, source mutability, native storage, producer fingerprint, source alias, owner cadence, Task 7 schedule, or expanded family members. Phase 2 source binding and actual backend capability, rank-local ownership, physical scheduling, transform, and local-plan fingerprints are deferred until after construction.

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
class CompiledPrecisionSelectionGroup:
    schema_version: int
    topology: ResolvedSelectionTopology
    semantic_structure_digest: str
    policy_digest: str
    graph_selections: tuple[CompiledGraphPrecisionSelection, ...]
    scope_results: tuple[CompiledScopeResult, ...]
    bf16_fences: tuple[PrecisionBoundaryFence, ...]
    atomic_expansions: tuple[AtomicExpansion, ...]
    selection_group_id: str

def compile_precision_selection(
    policy: PrecisionPolicyConfig,
    topology: ResolvedSelectionTopology,
) -> CompiledPrecisionSelectionGroup:
    if policy.schema_version != topology.schema_version:
        raise PrecisionPolicyError("policy and selection topology schema versions differ")
    topology.validate_complete()
    roles = topology.role_registry()
    graph_selections = tuple(
        _compile_graph_selection(policy, graph, roles)
        for graph in topology.graphs
    )
    policy_digest = canonical_policy_digest(policy)
    scope_results = _aggregate_scope_results(graph_selections)
    bf16_fences = _aggregate_bf16_fences(graph_selections)
    atomic_expansions = _aggregate_atomic_expansions(graph_selections)
    canonical = _canonical_selection_group_payload(
        schema_version=topology.schema_version,
        semantic_structure_digest=topology.semantic_structure_digest,
        policy_digest=policy_digest,
        graph_selections=graph_selections,
        scope_results=scope_results,
        bf16_fences=bf16_fences,
        atomic_expansions=atomic_expansions,
    )
    return CompiledPrecisionSelectionGroup(
        schema_version=topology.schema_version,
        topology=topology,
        semantic_structure_digest=topology.semantic_structure_digest,
        policy_digest=policy_digest,
        graph_selections=graph_selections,
        scope_results=scope_results,
        bf16_fences=bf16_fences,
        atomic_expansions=atomic_expansions,
        selection_group_id=sha256(canonical).hexdigest(),
    )
```

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
- Produces: strict `SourceSchemaId`; the four initial schema constants; immutable `SourceProducerFingerprint`; trusted `ExpectedContributorSet` and ID-free `ExpectedContributorAuthority`; `DiscoveryContribution`; `DiscoveryCompletenessReceipt`; factory-created `GraphDiscoveryPartition`; the partitioned `SourceDiscoveryInventory`; `assemble_graph_discovery_partition()`; `validate_discovery_inventory()`; `runtime_source_request_identity_digest()`; and a hardened `RuntimeGraphSourceRequest` carrying the Phase 1 structure/selection identities plus the exact source producer fingerprint, contributor authority, and typed source/artifact identities. `SourceDiscoveryRecord` and `SourceRecordProvenance` move from `topology.py` into this core module and are imported/re-exported rather than duplicated.

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
    partition = assemble_graph_discovery_partition(
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
    validate_discovery_inventory(
        (main_request, draft_request),
        SourceDiscoveryInventory((main_partition, draft_partition)),
        expected,
    )
    with pytest.raises(ValueError, match="producer fingerprint"):
        validate_discovery_inventory(
            (replace(main_request, source_producer_fingerprint=other_fingerprint()), draft_request),
            SourceDiscoveryInventory((main_partition, draft_partition)),
            expected,
        )
```

Parameterize exact failures for an unknown/malformed source schema, mutable implementation tag, producer-selected expected authority, replaced authority evidence, expected-authority/runtime-request mismatch, a derived authority with non-content-address kind, noncanonical locator, or malformed digest, missing or undeclared trusted contributor mapping entry, missing or duplicate opaque contributor, mixed fingerprints, contribution graph mismatch, incomplete PP/rank union represented by a missing opaque contributor, duplicate source/native name, wrong config/revision/source-identity/artifact-identity digest, forged/replaced observed contributor count or digest, forged source count/digest or canonical-record digest, altered record tuple after receipt construction, duplicate graph partition, missing required runtime partition, and undeclared runtime partition. Prove every original typed authority-evidence field changes the opaque derived commitment and that raw IDs or PP/TP/EP coordinates in the retained trusted evidence never appear outside it. Reject bare string/buffer values and unsupported generators at tuple-backed discovery boundaries while preserving tuple/list/tuple-like `Sequence` snapshots. Re-run `validate_discovery_inventory()` on `dataclasses.replace()` variants and prove every receipt/authority mutation is rejected before frozen-adapter source classification. Include a coordinated mutation of runtime-request authority, partition authority, and receipt: recomputation from the separately retained trusted set must still reject it. Assert contributor IDs and any producer-private PP/TP/EP coordinates are absent from the authority serialization, verified partition, adapter arguments, semantic addresses, and family domains. Keep the existing strict dtype, normalized-source-view provenance, absent-record, deterministic-ordering, and deep-immutability tests.

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

@dataclass(frozen=True, slots=True)
class DiscoveryCompletenessReceipt:
    graph_instance_id: str
    producer_fingerprint_digest: str
    observed_contributor_set_digest: str
    observed_contributor_count: int
    source_set_digest: str
    source_count: int
    canonical_records_digest: str
    runtime_source_request_digest: str

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

`assemble_graph_discovery_partition()` recomputes the observed contributor set
from `DiscoveryContribution` values, requires exact equality with the trusted
set and the runtime source request's structurally constrained authority, one
common fingerprint/graph, and a unique complete source set, constructs the receipt
itself, then strips contribution objects and contributor IDs from the
factory-created partition. Producers cannot supply or choose the expected
authority. The resolver retains an exact graph-ID → `ExpectedContributorSet`
mapping, including its original typed evidence, through the next boundary.
`validate_discovery_inventory(runtime_requests, source_discovery,
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
`validate_discovery_inventory()` returns only the fully verified inventory; it
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
- Create: `nemo_rl/precision_policy/topology_resolver.py`
- Create: `tools/capture_precision_policy_source_evidence.py`
- Create: `tests/fixtures/precision_policy/producer_implementations.json`
- Create: `tests/fixtures/precision_policy/source_format_evidence.json`
- Test: `tests/unit/precision_policy/test_source_formats.py`
- Test: `tests/unit/precision_policy/test_discovery_producers.py`
- Test: `tests/unit/precision_policy/test_topology_resolver.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: Phase 1 graph declarations/effective model configurations and Task 3's `CompiledPrecisionSelectionGroup`; after construction, Task 4A's immutable runtime source request/partition contract, Task 4A.1's committed canonical `BF16_FORMAT` and `MXFP8_FORMAT` objects, realized Bridge/Automodel/TE or checkpoint contexts, and exact expected opaque contributor sets supplied by each runtime integration.
- Produces: `GraphTopologyResolutionRequest`; `resolve_selection_topology(requests, schema_version) -> ResolvedSelectionTopology`; `RuntimeSourceDiscoveryRequest`; `RuntimeSourceDiscoveryResult`; `SourceMetadataProducer`; `produce_checkpoint_partition()`; `produce_megatron_bridge_partition()`; `produce_automodel_partition()`; `produce_transformer_engine_partition()`; `bind_runtime_source_intents(selection, request, results) -> CompiledPrecisionIntentGroup`; pinned producer-implementation evidence; and the reviewed `SOURCE_FORMAT_CATALOG: tuple[FormatDescriptor, ...]`. It orchestrates Task 4A's `RuntimeGraphSourceRequest` rather than redeclaring it. Phase 1 is standard-library-only. Framework objects are normalized inside Phase 2 producers and never cross into topology or result records.

`SourceMetadataProducer.discover_contributions(runtime_graph_request,
expected_contributors)` returns normalized `DiscoveryContribution` values, not
a self-certified partition. The Phase 2 Task 4B resolver supplies the trusted
expected set and calls Task 4A's assembly/validation itself. The convenience
`produce_*_partition()` functions are resolver-owned orchestration wrappers
around that sequence; a producer cannot choose its expected authority or
construct a receipt unchecked.

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

Then add these exact Phase 2 RED tests:

- `test_runtime_bf16_and_mxfp8_sources_project_to_same_semantic_structure`;
- `test_phase_two_preserves_selection_and_bf16_fences_byte_exactly`;
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

Implement these frozen contracts and boundaries in the explicit Task 4B-owned
module `nemo_rl/precision_policy/topology_resolver.py`:

```python
@dataclass(frozen=True, slots=True)
class GraphTopologyResolutionRequest:
    declaration: ExpectedGraphDeclaration
    effective_model_config: Mapping[str, object]
    resolved_model_revision: str
    decoder_layer_universe: DecoderLayerUniverse

def resolve_selection_topology(
    requests: tuple[GraphTopologyResolutionRequest, ...],
    schema_version: int,
) -> ResolvedSelectionTopology: ...

@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryRequest:
    graph_requests: tuple[RuntimeGraphSourceRequest, ...]
    trusted_expected_contributors: tuple[tuple[str, ExpectedContributorSet], ...]
    semantic_structure_digest: str
    selection_group_id: str
    request_digest: str

@dataclass(frozen=True, slots=True)
class RuntimeSourceDiscoveryResult:
    graph_instance_id: str
    runtime_source_request_digest: str
    semantic_structure_digest: str
    selection_group_id: str
    producer_fingerprint: SourceProducerFingerprint
    partition: GraphDiscoveryPartition
    result_digest: str

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
graph/member/address/domain/shape/role/atomic-group/layer-universe field.
The request builder derives each claimed universe only from the effective
configuration and declared topology facts. The selected family adapter derives
it independently under its pinned contract and must reproduce the request's
universe byte-for-byte; disagreement fails the entire Phase 1 request set.

After endpoint construction, Phase 2 derives the trusted expected contributor
set/authority at each runtime/checkpoint integration boundary, retains the
exact graph-to-trusted-set mapping, invokes exactly one producer for every
required runtime graph, and passes that mapping with the complete partition
inventory through Task 4A validation. A static checkpoint-served external
draft has no runtime partition/result; its immutable destination load receipt
remains mandatory. The binder uses only the Phase 1-selected adapter identity
to classify source records and first verifies every runtime request's
`resolved_graph` against `selection.topology`. It requires an exact projection
onto all frozen semantic members, addresses, domains, shapes, selections, BF16
fences, and atomic closures. It may add source formats, mutability, native
realizations, aliases, and derived cadence. It cannot select an adapter,
recompile policy, expand an atomic group, or alter semantic structure. Missing,
extra, duplicate, reshaped, or stale results fail the whole set before cadence
derivation.
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

- [ ] **Step 5: Run producer/catalog gates and commit**

Run: `PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/precision_policy -q tests/unit/precision_policy/test_source_discovery.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py tests/unit/precision_policy/test_topology_adapters.py`

Run: `.venv/bin/pyrefly check nemo_rl/precision_policy tools/capture_precision_policy_source_evidence.py`

Run: `/opt/homebrew/bin/ruff check nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py tools/capture_precision_policy_source_evidence.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py`

Run: `/opt/homebrew/bin/ruff format --check nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py tools/capture_precision_policy_source_evidence.py tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py`

Run: `git diff --check`

Expected: all commands pass with exact producer identities and no unresolved evidence field.

```bash
git add nemo_rl/precision_policy/source_formats.py nemo_rl/precision_policy/discovery_producers nemo_rl/precision_policy/topology_resolver.py tools/capture_precision_policy_source_evidence.py tests/fixtures/precision_policy/producer_implementations.json tests/fixtures/precision_policy/source_format_evidence.json tests/unit/precision_policy/test_source_formats.py tests/unit/precision_policy/test_discovery_producers.py tests/unit/precision_policy/test_topology_resolver.py pyrefly.toml
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

Canonical native-owner authority is also global rather than fragment-local.
All consuming canonical records with one `source_native_owner_id` resolve to
exactly one qualified `OwnerFamilyReference` and agree on provenance,
provenance evidence, mutability, and mutability evidence. A second graph may
refer to that owner only through a validated alias relation; it cannot declare
a second canonical owner for the same native storage.

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

### Task 5: Pre-Construction Selection, Post-Construction Intent Binding, and `explain-precision`

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
- Consumes before construction: `materialize_precision_selection(policy_config: PolicyConfig) -> CompiledPrecisionSelectionGroup | None`. Consumes after required endpoints are constructed: `bind_materialized_runtime_sources(policy_config: PolicyConfig, request: RuntimeSourceDiscoveryRequest, results: tuple[RuntimeSourceDiscoveryResult, ...]) -> CompiledPrecisionIntentGroup | None`.
- Produces before construction: one frozen `CompiledPrecisionSelectionGroup` under typed internal `_compiled_precision_selection` keys in both endpoint configs and `render_precision_explanation(selection, format: Literal["text", "json"]) -> str`. Produces after construction and before communicator creation: one frozen `CompiledPrecisionIntentGroup` under `_compiled_precision_intents`, bound to the exact selection object/digests. Serialization uses explicit `to_wire_dict()` methods only at process or CLI boundaries; no `dict[str, Any]` side channel is introduced. Neither stage claims a final Task 7 `plan_id` or unobserved backend capability.

- [ ] **Step 1: Write failing one-source-of-truth and CLI behavior tests**

```python
def test_materializer_injects_the_same_selection_group_into_both_endpoints() -> None:
    config = qwen30_policy_config()
    result = materialize_precision_selection(config)
    assert result is not None
    assert config["_compiled_precision_selection"].selection_group_id == result.selection_group_id
    assert config["generation"]["_compiled_precision_selection"].selection_group_id == result.selection_group_id
    assert config["_compiled_precision_selection"] is config["generation"]["_compiled_precision_selection"]
    assert "_compiled_precision_intents" not in config

def test_postconstruction_binding_preserves_the_exact_selection() -> None:
    config, selection, request, results = constructed_runtime_fixture()
    intents = bind_materialized_runtime_sources(config, request, results)
    assert intents is not None
    assert intents.selection is selection
    assert intents.selection.topology is selection.topology
    assert intents.semantic_structure_digest == selection.semantic_structure_digest
    assert intents.selection_group_id == selection.selection_group_id
    assert config["_compiled_precision_intents"] is intents
    assert config["generation"]["_compiled_precision_intents"] is intents

def test_explain_precision_reports_bf16_boundaries_and_mxfp8_middle(tmp_path: Path) -> None:
    completed = run_config_cli("explain-precision", fixture_recipe(tmp_path), "--format", "json")
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["scopes"][0]["selected_global_decoder_layers"] == [2, 3, 4]
    assert payload["summary"]["rollout"]["mxfp8"] == 3 * 8 * 3
```

Add CLI subprocess tests asserting nonzero exit and an actionable message for
zero matches, an unsupported source-neutral topology adapter, conflicting
scopes, invalid layer universes, and an invalid immutable auxiliary
declaration. The CLI must not construct a runtime, invoke a producer, or claim
runtime source completeness. Add post-construction binding tests for every
producer's incomplete/duplicate/mixed-fingerprint graph partition,
graph/config/revision/artifact mismatch, missing/extra/reshaped/stale results,
and incomplete checkpoint-inventory accounting. Prove a different-family
drafter uses its Phase 1 adapter identity and a static checkpoint drafter needs
no runtime result. Actual endpoint binding gaps are rejected in Tasks 6-8.

- [ ] **Step 2: Run tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_materialize.py tests/unit/tools/test_config_cli.py -k precision`

Expected: missing materializer and unrecognized `explain-precision` subcommand.

- [ ] **Step 3: Implement materialization and wire the CLI to the production compiler**

```python
from nemo_rl.precision_policy.topology_resolver import (
    bind_runtime_source_intents,
    resolve_selection_topology,
)

def materialize_precision_selection(
    policy_config: PolicyConfig,
) -> CompiledPrecisionSelectionGroup | None:
    raw_policy = policy_config.get("precision_policy")
    if raw_policy is None:
        return None
    policy = parse_precision_policy(raw_policy)
    requests = graph_topology_resolution_requests(policy_config)
    topology = resolve_selection_topology(requests, policy.schema_version)
    selection = compile_precision_selection(policy, topology)
    policy_config["_compiled_precision_selection"] = selection
    policy_config["generation"]["_compiled_precision_selection"] = selection
    return selection

def bind_materialized_runtime_sources(
    policy_config: PolicyConfig,
    request: RuntimeSourceDiscoveryRequest,
    results: tuple[RuntimeSourceDiscoveryResult, ...],
) -> CompiledPrecisionIntentGroup | None:
    selection = policy_config.get("_compiled_precision_selection")
    if selection is None:
        return None
    intents = bind_runtime_source_intents(selection, request, results)
    policy_config["_compiled_precision_intents"] = intents
    policy_config["generation"]["_compiled_precision_intents"] = intents
    return intents
```

Invoke the selection materializer at the start of `Policy.__init__`, before
`resolve_policy_worker_cls()` or generation-class selection, so every algorithm
constructs endpoints from the same requested training/rollout formats. Repeated
calls with an identical policy/configuration return the existing selection;
different policy, revision, graph declarations, or layer universes are
rejected rather than silently replacing it. After both required endpoint
runtimes exist, collect the exact Phase 2 result set and call the binder once,
before any refit communicator is created. Rebinding, stale request/result
digests, or any source result that changes the selection is fatal.

`tools/config_cli.py explain-precision RECIPE` resolves inheritance and
interpolation exactly as `expand` does, invokes only Phase 1, and prints graph
lifecycles, exact decoder universes, the full role predicates, compact matched
domains and logical cardinalities, selected/unselected counts, layer ranges,
BF16 fences, atomic expansion, requested endpoint formats, model revisions,
`semantic_structure_digest`, and `selection_group_id`. It labels producer,
runtime source, mutability, alias/cadence, physical layout, transform,
`runtime_source_digest`, `intent_group_id`, and final plan IDs unavailable until
runtime binding. It never reimplements selector logic.

Both phases import the Task 4B boundaries rather than redefining them. Phase 1
never invokes a producer. Phase 2 never constructs a flat independently
selected inventory, reselects an adapter, recompiles policy, or inspects vLLM
for source discovery.

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
- Consumes before construction: `compile_te_precision_recipe(selection: CompiledPrecisionSelectionGroup, construction_bindings: Sequence[SourceModuleConstructionBinding]) -> TEPrecisionArtifact`. Consumes after construction: the Phase 2 `CompiledPrecisionIntentGroup`, the same artifact, and realized TE modules/storage.
- Produces: frozen `SourceModuleConstructionBinding`, deterministic enabled exact matchers for training-MXFP8 modules, explicit BF16 boundary evaluation recipes, a recipe digest, and `validate_realized_training_precision(selection, intents, artifact, realized_modules) -> None`. Validation requires exact agreement among the policy selection, generated TE configuration, and realized BF16/MXFP8 storage; no separate TE file is trusted as another source of truth.

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
- Consumes: compiled graph intents, their Task 2 `owner_refit_requirements` and persisted `source_alias_contracts`, Task 2's single typed `ComponentRole` vocabulary and built-in `LOGICAL_VALUES`/`VALUES`/`BLOCK_SCALES` constants, Task 4A.2 source-storage realizations and allowed-normalizer manifest identity, explicit `SourceRuntimeParallelTopology` and `DestinationRuntimeParallelTopology`, endpoint capabilities, and realized source/destination bindings. Task 4 supplies no rank ownership. `refit_plan.py` imports and may re-export the Task 2 types; it never declares a second `NewType` or requirement enum.
- Produces: `PhysicalFormatStage`, `PhysicalAxisMapping`, `PhysicalPadding`, `PhysicalPermutation`, `PhysicalLayoutDescriptor`, `PhysicalRepresentation`, `EndpointPlacement`, `PhysicalComponentDescriptor`, `RealizedBindingFormat`, `DirectCopyCapabilityProof`, `ComponentBinding`, `BindingSet`, `TransformLocus`, realized `PhysicalOwnerSchedule`, `PhysicalOwner`, `BoundPhysicalOwner`, `RealizedDestinationOwnerGroup`, `ImmutableContributorCacheKey`, `MixedCadenceCompositionPlan`, `SourceVersionFenceRequirement`, `SourceVersionFence`, `EndpointCapabilities`, derived `RankLocalEndpointOwnership`, `BoundSourcePlans`, `BoundDestinationPlans`, `BoundComponentBatch`, `DestinationCommitReady`, `DestinationPoisonReason`, `LocalExecutionPlan`, `CanonicalStartupLoadPlan`/`CanonicalStartupLoadPlanGroup`, graph-level `CanonicalRefitPlan`, alias-aware `GraphTransactionMember`, ordered `CanonicalRefitPlanGroup`, `build_canonical_plan_groups()`, validation functions, and ordered wire metadata. A physical schedule maps semantic owner requirements to startup cached components, every-version components, and exactly-once finalization after realized cadence closure. It de-duplicates a canonical-alias source export independently from destination realization: distinct main and drafter allocations receive explicit fan-out bindings, while destination load/finalize/ACK de-duplication requires an adapter proof that physical storage-owner identity and finalizer identity are equal. Identical-storage aliases need no live replica fence; synchronized replicas do.

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

`build_canonical_plan_groups()` runs only after it derives rank-local ownership
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
- Modify: `nemo_rl/models/generation/sglang/sglang_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:550-700`
- Modify: `nemo_rl/models/generation/vllm/quantization/fp8.py:58-300`
- Test: `tests/unit/models/generation/test_checkpoint_evidence.py`
- Test: `tests/unit/models/generation/test_vllm_precision_adapter.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes before construction: serialized compiled rollout selection and actual `vllm.__version__`. Consumes after construction: the exact Phase 2 runtime-bound intents, including startup/every-version realization requests for participating graphs, and common bound-plan records from Task 7.
- Produces: generic frozen `RealizedCheckpointEvidence`, `CheckpointDestinationLoadReceipt`, `CheckpointDestinationFinalizeReceipt`, complete `CheckpointLoadReceipt`, `DestinationCheckpointAttestor`, `verify_realized_checkpoint_evidence()`, and `verify_checkpoint_load_receipt()` in `nemo_rl.models.generation.interfaces`; frozen `VllmCapabilityProbes`; `VllmEndpointAdapter` with `capabilities()`, `configure_engine_kwargs()`, `describe_runtime_parallel_topology()`, `bind_realized_storage()`, `attest_checkpoint_realization()`, startup/update prepare/finalize methods, `load_component_batch()`, and `poison()`; `select_vllm_endpoint_adapter(version: str, probes: VllmCapabilityProbes) -> VllmEndpointAdapter`. Realized binding returns Task 7's `BoundDestinationPlans`; capability, immutable-contributor-cache, and placement fingerprints enter plan assembly only after realization. vLLM and SGLang implement the same attestor contract for checkpoint-served graphs. `DestinationStartupReady` and `DestinationCommitReady` are adapter-local fenced proofs, not Task 11 worker results; Task 12 validates and converts them.

- [ ] **Step 1: Write failing registry and isolation tests**

```python
@pytest.mark.parametrize(("version", "adapter_id"), [("0.25.1", "vllm-0.25.1"), ("0.28.0", "vllm-0.28.0")])
def test_exact_supported_version_selects_dedicated_adapter(version: str, adapter_id: str) -> None:
    assert select_vllm_endpoint_adapter(version, complete_probes()).adapter_id == adapter_id

def test_unknown_or_incomplete_vllm_fails_before_model_construction() -> None:
    with pytest.raises(UnsupportedVllmEndpointError, match="capability"):
        select_vllm_endpoint_adapter("0.29.0", incomplete_probes())
```

Add a test that importing the registry without vLLM installed succeeds, that selecting 0.28 never imports 0.25-only modules, and that two engines with different plans do not share process-global quantization state. Verify that a training-only graph creates no vLLM realization request, while startup-only and every-version owners both expose realized storage and placement for Task 7. A checkpoint-served graph uses its pinned native load context rather than a source-load binding, but serving remains closed until its `CheckpointLoadReceipt` contains `RealizedCheckpointEvidence` matching every field of `ImmutableAuxiliaryEvidence` and exactly accounts for the bound checkpoint component/domain, destination load-operation, finalizer-group, rank, and completion-fence sets. Add fatal stale-tag, stale-cache/path, resolved-revision, checkpoint-content, model-config, semantic-domain, evidence-source, missing/extra/duplicate component domain, load operation, finalizer, rank, covered-member digest, and incomplete-fence tests. Run the same generic receipt verifier against a fake SGLang/static-drafter adapter so the contract is not vLLM-specific.

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
    def bind_realized_storage(self, intents: CompiledPrecisionIntentGroup, model: object) -> BoundDestinationPlans: ...
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
consumption, ignored loader return, or any mismatch poisons construction and fails the launcher. All destination
backends, including SGLang and static external drafters, implement this generic
proof even when their loading mechanics differ.

- [ ] **Step 4: Run adapter and existing FP8 regression tests**

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py tests/unit/models/generation/test_vllm_fp8_quantization.py tests/unit/models/generation/test_vllm_fp8_hf_overrides.py --vllm-only`

Run: `uv run --no-sync pyrefly check nemo_rl/models/generation/vllm/precision_adapter`

Run: `uv run --no-sync pre-commit run --files nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/sglang/sglang_worker.py nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml`

Expected: all commands pass under the pinned 0.25.1 environment. Repeat the same conformance file in the pinned 0.28.0 environment and require pass before marking this task complete.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/sglang/sglang_worker.py nemo_rl/models/generation/vllm/precision_adapter nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/quantization/fp8.py tests/unit/models/generation/test_checkpoint_evidence.py tests/unit/models/generation/test_vllm_precision_adapter.py pyrefly.toml
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
- Consumes: startup and every-version members of `BoundDestinationPlans`, whose owners independently request BF16 logical load, BF16→MXFP8 quantization, or compatible native-MXFP8 component copy.
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

Add literal padding cases: Lightning TP2 `928→1024` and `2688→3072`, Super TP4 `672→768`, Ultra TP16 `320→384`, Qwen3 TP4 `192→256`, Qwen3.5 TP8 `64→128`. Cover gated/non-gated W13/W31, grouped and split sources, zero-value/unit-scale padding, scale flatten/interleave, native MXFP8 component order, A→B→C repeated refits, finalizer failure poisoning, and no commit after partial load. Run the same owner-dispatched load/finalize primitives for an independent frozen startup-only owner, assert it becomes startup-ready before serving, and assert later every-version loads never dirty or finalize it again. Separately cover Task 7's mixed-cadence fused group: immutable contributors remain in the verified destination cache without wire retransfers while the shared physical owner is composed and finalized once for each mutable update.

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
- Consumes: compiled graph intents and persisted source-alias contracts, actual TE/Megatron parameters, and Task 7's common bound-plan and `SourceVersionFenceRequirement` records.
- Produces: frozen `RealizedSourceParameterInventory` with runtime tensor accessors, source runtime parallel topology, Task 7's `BoundSourcePlans`, `bind_mxfp8_source(intents, inventory) -> BoundSourcePlans`, exact live `SourceVersionFence` proofs for required synchronized replicas, startup exports for source-proven frozen owners, and every-version exports for mutable owners. This runtime inventory is distinct from Task 4A's partitioned, metadata-only `SourceDiscoveryInventory`; it neither replaces producer completeness receipts nor reclassifies native names. Native-MXFP8 owners export ordered direct `values`/`block_scales`; BF16 owners export logical BF16.

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

Add tests for grouped expert views with gradients, forged dtype metadata, mismatched scale geometry, disabled FP8 export, storage alias partitioning, synchronization before export, lifecycle-based inclusion of a served mutable MTP/drafter, exclusion of a mutable training-only auxiliary from source load, alias-owner de-duplication, and direct native component compatibility failure. Verify that a frozen source-served owner exports exactly once through the startup plan, a mutable owner exports only through the every-version plan, and a mixed graph returns both without duplicating either owner. For synchronized replicas, prove optimizer update happens before replica synchronization, the matching rank-local `SOURCE_VERSION_READY` fence happens after synchronization, and export happens only after Task 7 validates the complete live fence set. Reject pre-update/stale/wrong-group/wrong-topology/wrong-version/wrong-rank/duplicate fences. Identical-storage aliases require no live replica fence. Exactly in-scope served canonical aliases resolving to a training source require fences, including aliases inside checkpoint-served graphs; training-only, direct checkpoint-body, non-training-authority, and destination-local `out_of_scope` members do not. Frozen synchronized aliases bind their initial source version and live fence set into the startup plan and startup-precondition digest.

- [ ] **Step 2: Run source tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py`

Expected: missing source adapter.

- [ ] **Step 3: Implement source binding from the canonical graph intents**

```python
def bind_mxfp8_source(intents: CompiledPrecisionIntentGroup, inventory: RealizedSourceParameterInventory) -> BoundSourcePlans:
    startup = _bind_semantic_sources(intents.startup_source_items, inventory)
    every_version = _bind_semantic_sources(intents.every_version_source_items, inventory)
    _validate_exact_cadence_coverage(intents, startup, every_version)
    _validate_component_geometry((*startup, *every_version))
    return BoundSourcePlans(intent_group_id=intents.intent_group_id, startup=startup, every_version=every_version)
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
- Create: `nemo_rl/weight_sync/transaction.py`
- Modify: `nemo_rl/distributed/refit_watchdog.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Test: `tests/unit/weight_sync/test_refit_transaction.py`
- Test: `tests/unit/distributed/test_refit_watchdog.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: validated `CanonicalStartupLoadPlanGroup` and `CanonicalRefitPlanGroup`, expected checkpoint evidence and bound checkpoint consumption sets, their exact source version, target version, active synchronized-replica fence requirement/live-proof digest, group-phase, source-send batch, destination-receive batch, checkpoint-receipt, and destination-finalizer rank acknowledgement sets, source and destination `ray.ObjectRef` sets, plus registered abort/poison callbacks.
- Produces: `RefitPhase`, `RefitExecutionKind`, `RefitResultStatus`, `TransferDirection`, the discriminated `GroupPhaseResult | TransferWorkerResult | DestinationLoadResult | DestinationFinalizeResult | CheckpointReceiptResult` union named `RefitWorkerResult`, non-empty `RefitWorkerResultBatch`, `RefitFailure`, one-shot `StartupLoadTransaction`, every-version `RefitTransaction`, `supervise_refit_futures()`, and bounded abort. Startup publishes a precondition digest before serving; every-version execution contains mutable owners only. Verified checkpoint receipt, destination load-owner completion, finalizer-group completion, and engine transaction-envelope completion are separate proof sets.

- [ ] **Step 1: Write failing first-failure and poison tests**

```python
def test_receiver_failure_is_observed_while_sender_is_still_pending(fake_ray: FakeRay) -> None:
    sender = fake_ray.pending("train-rank-0")
    receiver = fake_ray.failed("gen-rank-1", RuntimeError("layout conversion failed"))
    with pytest.raises(RefitTransactionError, match="gen-rank-1.*layout conversion failed"):
        supervise_refit_futures(transaction(), [sender], [receiver])
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
            phase=RefitPhase.ABORT,
            execution_kind=RefitExecutionKind.UPDATE,
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

Run: `uv run --no-sync pytest -q tests/unit/weight_sync/test_refit_transaction.py tests/unit/distributed/test_refit_watchdog.py`

Expected: missing transaction module and current train-first wait delays receiver failure.

- [ ] **Step 3: Implement explicit PREPARE→READY→TRANSFER→FINALIZE→COMMIT state transitions**

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
    ABORT = "abort"

class RefitExecutionKind(StrEnum):
    STARTUP = "startup"
    UPDATE = "update"

class RefitResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class TransferDirection(StrEnum):
    SOURCE_SEND = "source_send"
    DESTINATION_RECEIVE = "destination_receive"

@dataclass(frozen=True, slots=True)
class GroupPhaseResult:
    status: RefitResultStatus
    rank: int
    phase: RefitPhase
    execution_kind: RefitExecutionKind
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
    phase: RefitPhase
    execution_kind: RefitExecutionKind
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
    phase: RefitPhase
    execution_kind: RefitExecutionKind
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
    phase: RefitPhase
    execution_kind: RefitExecutionKind
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
    phase: RefitPhase
    execution_kind: RefitExecutionKind
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
    | TransferWorkerResult
    | DestinationLoadResult
    | DestinationFinalizeResult
    | CheckpointReceiptResult
)
type RefitWorkerResultBatch = tuple[RefitWorkerResult, ...]
```

The result validator accepts `GroupPhaseResult` only for `PREPARE`, `COMMIT`,
and `ABORT`; `TransferWorkerResult` only for `TRANSFER`;
`DestinationLoadResult` only for `LOAD`; and `DestinationFinalizeResult` only
for `FINALIZE`; and `CheckpointReceiptResult` only with phase
`CHECKPOINT_ATTEST` and `execution_kind=STARTUP`. `READY` and `STARTUP_READY`
are coordinator states, not worker
result variants. A source/destination transfer proof is keyed by
`(rank, execution_batch_id, direction)`, a load proof by
`(rank, load_operation_id)`, and a finalizer proof by
`(rank, finalizer_group_id)`. The covered-set digest must equal the exact Task 7
member set for that operation. The validator checks every typed ID, execution
kind, static plan-group identity, exact covered graph/plan-member digest,
source version, target version, canonical live source-fence-set digest,
and completion fence against the canonical plan group;
no field is blanket-optional, one graph ID cannot stand in for a cross-graph
operation, and no fake physical owner is attached to a group phase. The startup
execution uses its concrete initial serving version. A resolved future may
contain one result or a non-empty result batch; validation flattens the batch
only after rejecting duplicates and then checks the checkpoint receipt,
transfer, load, and finalizer acknowledgement sets plus the separate
engine-envelope phase proof. Each checkpoint receipt is independently verified
against immutable evidence and bound destination consumption before its
canonical digest can satisfy the expected `(graph, rank)` proof.

Use `ray.wait(..., num_returns=1, timeout=remaining_deadline)` across both source and destination refs and validate each ready result immediately. Before the first transfer, validate the complete active `SourceVersionFence` set. Derive a fresh runtime transaction identity from the static plan-group ID, source version, target version, canonical live fence-set digest, and, for updates, successful startup/cache precondition digest; bind those fields into every result. Static execution plans, routes, and buffers are not rebuilt. `StartupLoadTransaction` publishes `STARTUP_READY` only after its exact source-fence, transfer, load-owner, finalizer-group, and checkpoint-receipt sets complete; it has no generation-version COMMIT and cannot run twice. `RefitTransaction` verifies the startup/cache digest and commits only after the exact source-fence and acknowledgement sets for every realized load owner and finalizer group affected by a mutable contributor complete. Aliases are de-duplicated independently in each equivalence relation: source transfer, destination load owner, and finalizer group. Non-participating graphs/ranks remain absent. On failure, preserve the first exception, run all abort/poison callbacks concurrently with a fixed teardown deadline, terminate owners whose abort does not acknowledge, then re-raise the first cause wrapped with structured context.

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

### Task 12: Integrate Fatal Transactions into IPC, Collective, Reshard, Checkpoint Engine, Sync, and Async RL

**Files:**
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py`
- Modify: `nemo_rl/models/generation/vllm/checkpoint_engine.py`
- Modify: `nemo_rl/models/policy/workers/checkpoint_engine.py`
- Modify: `nemo_rl/models/generation/vllm/refit_loader.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py:1270-1350`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py:1420-1515`
- Modify: `nemo_rl/algorithms/grpo.py:2510-2610,4737-4766,5417-5450`
- Modify: `nemo_rl/algorithms/async_utils/trajectory_collector.py:1040-1175`
- Modify: `examples/run_grpo.py:190-270`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Test: `tests/unit/algorithms/test_grpo.py`
- Test: `tests/unit/algorithms/test_async_utils.py`
- Test: `tests/unit/models/generation/test_vllm_backend.py`
- Test: `tests/unit/models/generation/test_vllm_checkpoint_engine.py`
- Test: `tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py`
- Test: `tests/functional/refit_failure_exit.py`

**Interfaces:**
- Consumes: `StartupLoadTransaction`, `RefitTransaction`, exact live source-version fence proofs, and typed worker results from Task 11.
- Produces: one successful startup-load gate before either training loop can serve generation, plus the same fatal behavior for every later refit transport; generation weight version changes only on every-version COMMIT.

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

Add transport tests where a generation future fails while a train future remains pending, all-`None` results, malformed results, worker wrapper errors that currently return `False`, cache invalidation failure, failed `prepare_for_generation`, failed shutdown, and async refit after collector pause. Cover IPC, collective, NCCL reshard, and checkpoint-engine paths. In particular, reject checkpoint-engine RPC wrappers that filter `None` or accept vacuous `all(...)`, and require its batched loader, main/MTP/Eagle physical destinations, finalizers, poison/abort, and exact acknowledgements to use the same bound transaction plan. Add auxiliary preflight cases for a mutable training-only MTP/drafter, a mixed frozen/mutable source-served MTP/drafter, an all-frozen source-served startup graph, a static checkpoint drafter, a missing drafter on an owning rank, and an absent drafter on a non-owning PP rank. Assert startup load runs once before collection, frozen owners never enter later payloads, and any startup/refit failure prevents resume/serve with bounded actor cleanup.
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

Worker methods return the phase-appropriate `RefitWorkerResult` variant or
raise; source and destination transfer wrappers preserve execution-batch and
direction identity and never translate a send/load/finalize exception to
`False`. Thread a positive `refit_timeout_s` through legacy setup and every initial/subsequent refit. Move async startup under terminal cleanup, re-raise after telemetry flush, make cache invalidation failure fatal for every backend, and leave the collector/generation gate closed after a poisoned update. Launcher cleanup is bounded and never replaces the original exception.

Checkpoint-engine loading uses this same transaction envelope rather than a
parallel boolean protocol. Each batch is associated with its bound physical
destination owners; main, MTP, and Eagle allocations are loaded and finalized
according to their independent realized identities. Empty or filtered RPC
results never satisfy an acknowledgement set.

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
startup_plans, refit_plans = build_canonical_plan_groups(...)
initial_source_version, initial_fences = source.bind_startup_version(...)
startup_digest = StartupLoadTransaction(
    startup_plans,
    source_version=initial_source_version,
    target_version=initial_generation_version,
    source_fences=initial_fences,
).run_before_serving()
source_version, source_fences = source.bind_refit_version(...)
RefitTransaction(
    refit_plans,
    required_startup_digest=startup_digest,
    source_version=source_version,
    target_version=next_generation_version,
    source_fences=source_fences,
).run(...)
```

`trains_mtp`, `has_refit_draft_weights`, `loss_scaling_factor`, and
`detach_heads` may help discover or cross-check graph declarations, but none is
sufficient to decide owner mutability or cadence. Run a non-empty startup group
once before opening collection/generation; preserve its digest as a repeated
refit precondition. Register every graph with a mutable served member as an
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

Run: `uv run --no-sync pytest -q tests/unit/weight_sync tests/unit/distributed/test_refit_watchdog.py tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_backend.py --vllm-only`

Run: `uv run --extra vllm --group test pytest -q tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py --vllm-only`

Run: `uv run --no-sync python tests/functional/refit_failure_exit.py`

Run: `uv run --no-sync pre-commit run --files nemo_rl/weight_sync nemo_rl/models/generation/vllm/checkpoint_engine.py nemo_rl/models/generation/vllm/refit_loader.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/models/policy/workers/checkpoint_engine.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/async_utils/trajectory_collector.py examples/run_grpo.py tests/unit/weight_sync tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/functional/refit_failure_exit.py`

Expected: all commands pass and both subprocess modes exit nonzero within the test budget.

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/weight_sync nemo_rl/models/generation/vllm/checkpoint_engine.py nemo_rl/models/generation/vllm/refit_loader.py nemo_rl/models/generation/vllm/vllm_worker.py nemo_rl/models/generation/vllm/vllm_worker_async.py nemo_rl/models/policy/workers/checkpoint_engine.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/async_utils/trajectory_collector.py examples/run_grpo.py tests/unit/weight_sync tests/unit/algorithms/test_grpo.py tests/unit/algorithms/test_async_utils.py tests/unit/models/generation/test_vllm_backend.py tests/unit/models/generation/test_vllm_checkpoint_engine.py tests/functional/refit_failure_exit.py
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
