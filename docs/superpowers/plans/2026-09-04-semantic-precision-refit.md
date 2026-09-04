# Semantic Precision Policy and Transactional Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one positive semantic precision policy that configures MXFP8/BF16 training and rollout, drives mixed-layout refit without name/dtype guesses, fails the whole launcher immediately on refit failure, and preserves the fastest correct refit path.

**Architecture:** A typed policy compiler resolves semantic roles against a complete graph-manifest bundle and emits immutable graph precision intents before backend construction. Internal lifecycle records independently describe graph kind/provenance and rollout participation; authoritative owner inventory records source mutability and derives refit requirement and owner cadence. Megatron/Transformer Engine and versioned vLLM endpoint adapters realize and bind only the endpoints each graph participates in. The refit planner produces one-shot startup plans for frozen owners served from source and every-version transaction plans for mutable owners served from source. A transactional engine transfers canonical components, finalizes each physical owner once per cadence execution, and commits a new generation weight version only after every expected owning rank of every every-version transaction member is ready.

**Tech Stack:** Python 3.13.13, Pydantic v2, frozen dataclasses, PyTorch, Ray, Megatron-Core/Bridge, Transformer Engine, vLLM 0.25.1 and 0.28.0, ModelOpt MXFP8, FlashInfer TRTLLM, pytest, Pyrefly, Ruff, MyST.

**Spec:** `docs/design-docs/semantic-precision-refit.md`

## Global Constraints

- Implement from immutable baseline `4601ba2c646ec40e5928c780fc0051a842328eba` on branch `codex/refit-semantic-policy-v2-20260903`; do not update any existing pull-request head while this plan is under validation.
- The public schema is version 1, `default` is always `bf16`, and the common recipe is a positive allow-list. Raw checkpoint/runtime parameter patterns are not a stable user interface.
- The same compiled intent group is the only source of truth for training, wire, and rollout precision. Generated TE matchers or vLLM include data are derived artifacts and must pass exact realized-module validation.
- `moe.routed_expert` means only main text-decoder routed expert gate/up/down kernels. It excludes shared experts, routers, latent projections, bias, MTP/draft graphs, attention, and embeddings.
- `attention.qkvo` means only main text-decoder token-attention Q/K/V/O projection kernels. It excludes MLA, KDA/GDN, sparse indexers, output gates, vision, bias, and MTP/draft graphs.
- Layer coordinates are zero-based. The default index space is `global_decoder`; `moe_ordinal` is explicit. `exclude_first` and `exclude_last` count in the selected index space and cannot consume the full domain.
- Every mutable main-model tensor is accounted for and refitted. Every auxiliary graph instantiated by training is also present in the semantic bundle, even when it is mutable but training-only. `out_of_scope` is allowed only for source-proven frozen parameters, immutable auxiliary models, or backend-owned derived state with a typed reason.
- MTP and speculative drafters are separate semantic graphs. Their internal records distinguish graph kind, provenance, source mutability, rollout participation, derived refit requirement, and rank-local endpoint ownership; none of those fields is added to the public precision-policy selector.
- Every `served_from_source` physical owner requires realized source and destination bindings on ranks derived to own its startup-only or every-version cadence. A graph with any mutable owner joins every-version atomic transactions, but only mutable source contributors repeat on the wire; its independent frozen owners load once at startup, while mixed realized destination groups follow the cadence-closure rule below. A mutable `not_served` auxiliary is valid and has no rollout/refit plan. Missing drafter storage is fatal only on a derived owning rank for a required cadence, and is valid on a non-owning PP rank or for a graph not served by rollout.
- A `served_from_source` graph must have a non-empty resolved semantic domain and at least one present canonical source owner. Alias-only graphs are valid only when all aliases resolve to compatible existing owners; absent owners and vacuous all-frozen derivation fail.
- Cadence closes over realized destination owner/finalizer groups. Mixed groups cache verified immutable contributors once, refresh only mutable contributors, and require advertised native preservation or split/repack before exactly-once composition/finalization.
- A `served_from_checkpoint` auxiliary requires immutable graph/model identity, pinned resolved revision, checkpoint-content, model-configuration, semantic-domain digests, and typed evidence source, and contributes no source transfer. `loss_scaling_factor=0`, `detach_heads`, or a missing current gradient is not freeze evidence.
- Every checkpoint-serving destination attests the artifact actually loaded and must match every immutable-evidence field before serving; stale tags, caches, paths, or mismatched evidence are fatal for vLLM, SGLang, and static draft backends alike.
- Tied aliases remain explicit graph members but reference one canonical physical owner; they never duplicate transfer, finalization, or acknowledgement.
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
| `nemo_rl/precision_policy/semantic.py` | Frozen semantic addresses, roles, formats, atomic groups, manifests, and orthogonal graph-lifecycle declarations |
| `nemo_rl/precision_policy/compiler.py` | Positive selection, layer filtering, coverage/conflict checks, graph-intent generation, canonical intent digests |
| `nemo_rl/precision_policy/topology.py` | Topology-adapter protocol, registry, nested text-config resolution, complete accounting |
| `nemo_rl/precision_policy/adapters/qwen.py` | Qwen3/Qwen3.5/Qwen3.8 semantic classification |
| `nemo_rl/precision_policy/adapters/nemotron.py` | Nano/Lightning/Super/Ultra semantic classification |
| `nemo_rl/precision_policy/adapters/kimi.py` | Kimi K2/K2.5/K3 manifest conformance and encoding declarations |
| `nemo_rl/precision_policy/adapters/glm.py` | GLM-5.2 manifest conformance |
| `nemo_rl/precision_policy/materialize.py` | One-time policy compilation and endpoint artifact injection before workers start |
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
    exclude_first: NonNegativeInt = 0
    exclude_last: NonNegativeInt = 0

class PrecisionScopeConfig(BaseModel, extra="allow"):
    id: str
    role: str | None = None
    advanced_match: AdvancedMatchConfig | None = None
    addresses: list[SemanticAddressSelectorConfig] | None = None
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

Each model validator rejects undocumented `model_extra`; the scope validator enforces a non-empty unique `id`, exactly one of `role`, `advanced_match`, or non-empty `addresses`, and at least one non-BF16 endpoint request. Address records require `graph_instance_id` equal to `main` or prefixed by `mtp.`/`draft.`, require the semantic ID to use one canonical path-prefixed rendering, require `semantic_graph_path` to match that rendering, and reject duplicate `(graph_instance_id, semantic_id)` pairs. The ambiguous legacy fields `advanced_match.graph` and `semantic_addresses` are rejected. Add `precision_policy: NotRequired[PrecisionPolicyConfig]` to `PolicyConfig`.

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

### Task 2: Semantic Manifest, Roles, Formats, and Complete Accounting

**Files:**
- Create: `nemo_rl/precision_policy/semantic.py`
- Test: `tests/unit/precision_policy/test_semantic.py`
- Modify: `pyrefly.toml`

**Interfaces:**
- Consumes: normalized logical model components, authoritative compact `ParameterInventory`, topology-adapter `RoleExpectedDomain` values, and the complete `ExpectedGraphDeclaration` set discovered from training/runtime configuration before adaptation.
- Produces: the normative frozen records in Step 3: `SemanticAddress`, `SemanticTensor`, `SemanticTensorFamily`, `SemanticInventoryMember`, `RoleDefinition`/`RoleExpectedDomain`, `FormatDescriptor`, `ComponentDescriptor`, `AtomicGroup`, typed `OutOfScopeReason`/`OutOfScopeTensor`, compact qualified `OwnerFamilyReference`/`OwnerFamilyBinding`/`SemanticOwnership`, `EvidenceSourceKind`/`EvidenceSource`, `SourceOwnerInventoryEntry`, `ParameterInventoryEntry`/`ParameterInventory`, `GraphKind`, `GraphProvenance`, `ValueProvenance`, `SourceMutability`, `RolloutParticipation`, transient derived `RefitRequirement`, composite `GraphLifecycle`, `ImmutableAuxiliaryEvidence`, `ExpectedGraphDeclaration`, topology-independent `AuxiliaryGraphDeclaration`, `SemanticGraphManifest`, and `SemanticManifestBundle.validate_complete()`.

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
    bundle = compact_explicit_role_fixture()
    expected = RoleExpectedDomain(("main-routed-gate",))
    role = builtin_role_definitions(1, {"moe.routed_expert": expected})["moe.routed_expert"]
    assert role.matching_inventory_entry_ids(bundle.inventory) == (
        "main-routed-gate",
    )
    role.validate_expected_domain(bundle.inventory)

def test_mutable_main_tensor_cannot_hide_out_of_scope() -> None:
    with pytest.raises(ValueError, match="mutable main-model"):
        bundle_with_out_of_scope_entry(
            ParameterInventoryEntry("main-kernel", "main", mutable_tensor(), ValueProvenance.TRAINING_PARAMETER),
            OutOfScopeTensor("main-kernel", OutOfScopeReason.FROZEN),
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
grammar and path-prefixed semantic-ID invariant; qualified tied aliases and
alias-cycle rejection; typed whole-entry out-of-scope accounting; and exact
declaration ↔ manifest ↔ inventory accounting. A source-served graph with any
mutable owner derives `every_version`, an all-frozen source-served graph derives
`initial_only`, and a mixed graph assigns startup cadence to frozen owners and
every-version cadence to mutable owners. A checkpoint-served graph requires
graph/model identity, pinned revision, content/configuration/semantic-domain
digests, and evidence source. `loss_scaling_factor=0`, `detach_heads`, or absent
current gradients cannot derive `frozen`.

Add explicit failures for a `served_from_source` graph with an empty logically
resolved compact domain, no canonical source owner, or an owner marked
`absent`. Prove that `all([])` does not derive `initial_only`. Add a valid
alias-only graph whose
entire non-empty domain resolves to an existing compatible canonical owner, and
fail missing targets, cycles, and domain/shape/axes/format-incompatible aliases.

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

Add descriptor tests that pin BF16 to one logical BF16 value component and
MXFP8 to E4M3 values plus E8M0 block-32 scales. Block-FP8, NVFP4, and MXFP4
must use distinct adapter-advertised format IDs and component families when
supported; do not add invented built-in profiles merely to satisfy the test.

- [ ] **Step 2: Run the tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_semantic.py`

Expected: import failure for `nemo_rl.precision_policy.semantic`.

- [ ] **Step 3: Implement immutable semantic records and exact built-in roles**

```python
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
    owner_family: OwnerFamilyReference
    member_domain: FamilyIndexDomain
    member_to_owner_axes: tuple[AxisProjection, ...]
    alias_of: OwnerFamilyReference | None = None

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
    inventory_entry_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class RoleDefinition:
    schema_version: int
    role_name: str
    predicate: SemanticPredicate
    expected_domain: RoleExpectedDomain

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
    expected_graphs: tuple[ExpectedGraphDeclaration, ...]
    manifests: tuple[SemanticGraphManifest, ...]
    inventory: ParameterInventory
```

This Step 3 record shape is normative rather than illustrative. A scalar uses
`FamilyIndexDomain(layer_domain=None, independent_axes=())`, whose cardinality
is one; it does not use a separate scalar owner-inventory schema. Literal path
segments are validated canonical atoms and index segments can reference only a
declared domain axis. `entry_id` is unique and stable for accounting but is not
rendered into `semantic_id` and cannot be used as tensor identity.

`GraphLifecycle` stores graph facts, not source-owner state. `SourceMutability`
lives on compact qualified owner-family domains in `ParameterInventory`; tied
semantic members reference those domains through `SemanticOwnership`.
`RefitRequirement` is computed transiently, never stored as an independent
input: join lifecycle with lazily resolved canonical owner domains, derive
`every_version` when any source owner is mutable, `initial_only` when a non-empty
owner set is entirely proven frozen, and `none` for `not_served` or
`served_from_checkpoint`. Within a mixed every-version graph, mutable owners
have repeated cadence and frozen independent owners have startup cadence. Do
not infer mutability or cadence from precision policy, loss configuration, or
current gradients.

For `served_from_source`, derive cadence only after logical/lazy family-domain
and alias resolution, never full materialization. Its semantic domain must be
non-empty and must reach at least one present canonical source owner. Reject
`SourceMutability.ABSENT`, unresolved targets, and vacuous all-frozen
derivation. An alias-only graph is valid only when every compact alias domain
resolves to an existing canonical owner domain with compatible shape, axes,
format, and exact index mapping; the alias retains graph membership but does
not create another owner.

`SemanticManifestBundle` contains exactly one `GraphKind.MAIN` instance, every
auxiliary graph instantiated by training (including mutable training-only
graphs), and every rollout-only static graph declaration. Its authoritative
`expected_graphs` field must match manifests bijectively. Define the built-in
BF16 descriptor as one logical BF16 value component and MXFP8 as E4M3 values
plus E8M0 block-32 scales. When an adapter supports block-FP8, NVFP4, or MXFP4,
it must advertise distinct exact format IDs and component families; do not
invent generic built-in profiles for unsupported encodings. Reject duplicate
canonical `(graph_instance_id, semantic_id)`
keys, a semantic ID whose rendered prefix disagrees with
`semantic_graph_path`, unknown logical axes, unqualified/duplicate ownership,
alias cycles, untyped exclusions, any mutable main-model exclusion, any omitted
expected graph or inventory entry, inconsistent lifecycle/provenance
combinations, or incomplete immutable evidence.

Families use `LayerMember`/`LayerDomain`, independent `AxisDomain` values,
`FamilyIndexDomain`, `LiteralPathSegment | IndexPathSegment`, structured
`SemanticAddressPattern`, and qualified `OwnerFamilyBinding`/
`OwnerFamilyReference`. No field accepts a free-form template, regex, glob, or
wildcard. Correlated coordinates live in one `LayerMember`; ragged domains
split into multiple complete families. Each family fixes facets, format, dtype,
shape, axes, and ownership, and any role-changing value is fixed in a separate
family, so gate/up/down and Q/K/V/O are not projection axes. Validate duplicates
by exact domain intersection and prove the compact inventory-entry union equals
the full logical inventory without persisting expanded instances. Out-of-scope
and alias compatibility checks also operate on whole compact domains. Rank-local
realized ownership and materialization are deliberately deferred to Task 7.

`RoleExpectedDomain` contains a non-empty tuple of inventory entry IDs supplied
by the topology adapter. `builtin_role_definitions(schema_version,
expected_domains)` attaches that domain to each predicate. Before layer
filtering, validation compares the predicate's compact-entry result exactly to
the expected IDs; a partial-family match, extra entry, or missing entry fails.

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
- Produces: frozen `CompiledGraphPrecisionIntent` records with lifecycle identity, immutable compact-domain assignments for participating endpoints, selected layer ranges, complete compact scope results plus logical cardinalities, semantic atomic closures, owner cadence, startup and every-version endpoint realization requests, canonical graph `intent_id` values, ordered `CompiledPrecisionIntentGroup`, and `intent_group_id`. It never stores expanded family members. Actual backend capability, rank-local ownership, transform, and local-plan fingerprints are deferred to Task 7 after realized binding.

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

Also add literal tests for zero-match, unknown role, incomplete advertised role coverage, overlapping conflicting scopes, full-range exclusion, atomic fused QKV conflict, allowed fixed-point expansion, expansion crossing BF16 boundary, dictionary-order-independent intent digest, invalid immutable-auxiliary evidence, and deterministic graph ordering. Verify qualified `advanced_match` and `addresses` selectors, reject ambiguous `graph`/unqualified semantic IDs, and prove built-in main roles require `GraphKind.MAIN` plus the exact semantic graph path. The auxiliary cases must prove that a mutable training-only MTP/draft receives a training intent but no rollout request; an all-frozen source-served graph emits a startup realization request; a source-served graph with any mutable owner emits an every-version graph request plus startup requests for frozen independent owners; and a checkpoint-served graph carries immutable context but no source-load request. Reject empty-domain, absent-owner, and unresolved/incompatible alias-only source-served graphs before producing an intent; accept a non-empty alias-only graph only when every alias resolves to a compatible present canonical source owner. One versioned policy governs all instances, including a different-family drafter selected through a qualified advanced/address scope. Backend unsupported-format and rank-local ownership checks belong to Tasks 7-8, after realized capabilities exist.

- [ ] **Step 2: Run the compiler tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_compiler.py`

Expected: import failure for `compile_precision_policy`.

- [ ] **Step 3: Implement compilation in explicit passes**

```python
def compile_precision_policy(...) -> CompiledPrecisionIntentGroup:
    manifests.validate_complete()
    graph_intents = tuple(
        _compile_graph_intent(policy, manifest, roles)
        for manifest in manifests.graphs_in_canonical_order()
    )
    canonical = _canonical_intent_group_payload(policy, manifests, graph_intents)
    return CompiledPrecisionIntentGroup(..., intent_group_id=sha256(canonical).hexdigest())
```

Sort graph instance IDs, semantic graph paths, semantic IDs, attributes, roles, groups, lifecycle fields, owner cadences, and components before serialization. Every declared graph instance gets an intent, but an endpoint precision assignment exists only when that graph participates in the endpoint. Built-in main roles match `GraphKind.MAIN` and the exact semantic graph path, never an instance-name spelling. Built-in roles do not select auxiliaries; a participating unselected auxiliary inherits BF16 unless a qualified scope in the same policy applies. Before cadence derivation, require each source-served graph's resolved semantic domain and canonical source-owner set to be non-empty and present, including complete compatibility checks for alias-only graphs. Emit startup realization requests for source-served frozen owners and every-version requests for source-served mutable owners. A graph containing either is summarized by the derived requirement rules from Task 2; do not store a caller-supplied cadence. Never hash object identity or dictionary insertion order.

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
- Consumes: `build_semantic_manifest_bundle(model_config: Mapping[str, object], model_revision: str, parameter_inventory: ParameterInventory, auxiliary_declarations: Sequence[AuxiliaryGraphDeclaration]) -> SemanticManifestBundle`, where each typed `AuxiliaryGraphDeclaration` supplies topology-independent graph instance ID, graph kind/provenance, rollout participation, model identity, and optional immutable-evidence attachment. It never contains PP-rank ownership.
- Produces: registered adapters selected by `model_type` and architecture capabilities; the authoritative expected-graph set; separate main/MTP/draft manifests referencing compact inventory entries; topology-derived non-empty `RoleExpectedDomain` records; exact declaration/manifest/inventory reconciliation; and `resolve_text_config()` handling nested `text_config` without assuming top-level `num_hidden_layers`.

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

Add MTP/draft fixtures for: a static checkpoint-owned MTP; independent mutable training-only and source-served MTP graphs; MTP tied aliases; a static external drafter; and mutable training-only and source-served speculative drafters using a different model-family adapter. Assert that main-model roles select none of them; every auxiliary instantiated in training has its own expected declaration and manifest even when it is not served; only participating endpoints receive default BF16 intent; checkpoint-served graphs carry complete typed evidence; and qualified tied aliases point to an explicit main-graph owner without duplicating ownership. All instances use one versioned precision policy; a different-family drafter does not carry a separate policy.

Add negative fixtures that omit an instantiated training auxiliary, declare a
mutable checkpoint-served graph, or omit any immutable evidence field. Add
literal configurations with `loss_scaling_factor=0` and `detach_heads=true` and
assert that their owners remain mutable unless the source inventory supplies
independent freeze evidence. Topology adapter tests stop at topology-independent
graph declarations; Task 7 exclusively derives owning/non-owning ranks.

- [ ] **Step 2: Run adapter tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/precision_policy/test_topology_adapters.py`

Expected: import failure for the topology registry.

- [ ] **Step 3: Implement adapter selection and semantic classification**

```python
class ModelTopologyAdapter(Protocol):
    adapter_id: str
    def supports(self, model_config: Mapping[str, object]) -> bool: ...
    def role_expected_domains(self, inventory: ParameterInventory) -> Mapping[str, RoleExpectedDomain]: ...
    def role_definitions(self, schema_version: int, expected_domains: Mapping[str, RoleExpectedDomain]) -> Mapping[str, RoleDefinition]: ...
    def build_manifest(self, model_config: Mapping[str, object], model_revision: str, inventory: ParameterInventory, declaration: ExpectedGraphDeclaration) -> SemanticGraphManifest: ...
```

Classifiers may recognize endpoint names internally, but emit canonical semantic addresses and structured families only. The bundle builder chooses an adapter independently for a different-family drafter, while retaining the single policy, and orders graph instances deterministically. Reconcile typed auxiliary declarations against the training inventory so every actually instantiated training auxiliary is present. Do not derive runtime PP ownership here. Reject ambiguous names, missing/empty expected role domains, predicate results unequal to their expected compact entry IDs, inconsistent expert counts, unnormalized one-based layer indices, unsupported model revisions, contradictory declarations, family-domain overlaps, or partial inventory coverage. Keep dense prefix layers in the correlated `LayerMember` domain even when they contain no routed expert. Emit separate fixed-attribute families for gate/up/down and Q/K/V/O and split ragged domains into multiple complete families. Adapter discovery may use lazy generators, but the resulting inventory and manifest never store an expanded family member list.

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

Invoke the materializer at the start of `Policy.__init__`, before `resolve_policy_worker_cls()` or generation-class selection, so every algorithm uses the same path and backend dispatch sees requested training/rollout formats. Repeated calls with an identical policy return the existing intents; a different policy or model revision is rejected rather than silently replacing it. `tools/config_cli.py explain-precision RECIPE` resolves inheritance and interpolation exactly as `expand` does, invokes this function once, and prints graph lifecycles, the full role predicate, compact matched domains and logical cardinalities, selected/unselected counts, layer ranges, atomic expansion, requested endpoint formats, model revisions, and intent digests without storing rendered family members. It labels transforms, physical layouts, and final plan IDs as unavailable until realized binding and never reimplements selector logic.

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
- Consumes: compiled graph intents, explicit `SourceRuntimeParallelTopology` and `DestinationRuntimeParallelTopology`, endpoint capabilities, and realized source/destination bindings. Task 4 supplies no rank ownership.
- Produces: `ComponentRole`, `ComponentBinding`, `BindingSet`, `TransformLocus`, `OwnerCadence`, `PhysicalOwner`, `BoundPhysicalOwner`, `RealizedDestinationOwnerGroup`, `ImmutableContributorCacheKey`, `MixedCadenceCompositionPlan`, `EndpointCapabilities`, derived `RankLocalEndpointOwnership`, `BoundSourcePlans`, `BoundDestinationPlans`, `BoundComponentBatch`, `DestinationCommitReady`, `DestinationPoisonReason`, `LocalExecutionPlan`, `CanonicalStartupLoadPlan`/`CanonicalStartupLoadPlanGroup`, graph-level `CanonicalRefitPlan`, alias-aware `GraphTransactionMember`, ordered `CanonicalRefitPlanGroup`, `build_canonical_plan_groups()`, validation functions, and ordered wire metadata.

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

Add tests for arbitrary future component roles, missing/duplicate components, semantic-set inequality only across endpoints required for the same owner cadence, unsupported endpoint formats, native MXFP8 direct component transfer, BF16→MXFP8 destination transform, canonical BF16→TRTLLM native loader, fused owner atomicity, source/destination TP/EP/PP ownership derivation, canonical versus rank-local digests, and deterministic plan-group assembly. Prove that all-frozen source-served graphs produce startup plans; mixed mutable/frozen graphs produce startup plans for frozen independent owners and repeated wire payloads only for mutable owners; startup-owner digests become immutable refit preconditions; and no startup owner appears in an every-version wire payload. A mutable training-only graph and checkpoint-served graph contribute no source-load plan. An alias-only member references its qualified canonical owner and adds no duplicate load, finalizer, or acknowledgement. Missing MTP/drafter binding on a derived owning rank fails, while absence on a derived non-owner rank is valid.

Add mixed-cadence realized-owner tests where frozen and mutable semantic
contributors share one destination physical owner/finalizer. An A→B→C sequence
must transfer the frozen contributors once into a verified persistent startup
cache, transfer only mutable contributors for B and C, combine cached and fresh
canonical components, and compose/finalize the physical owner exactly once per
update. Accept either advertised native partial preservation or split/repack;
fail preflight when neither is supported. Assert the cache key/capability
fingerprint enters the plan digest and that storage rebinding, evidence/layout/
topology/capability changes, explicit invalidation, or poison invalidates it.

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
    graph_instance_id: str
    semantic_graph_path: str
    semantic_id: str
    format: FormatDescriptor
    components: tuple[ComponentBinding, ...]
    physical_owners: tuple[OwnerReference, ...]
    atomic_group_ids: tuple[str, ...]

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
    mutable_contributors: tuple[OwnerReference, ...]
    mode: Literal["native_preserve", "split_repack"]
```

Validate the full plan before NCCL groups are created. Wire metadata carries graph instance ID, semantic graph path, semantic ID, component role, dtype, logical/physical shapes, axes, placement, owner, layout, transform, and plan ID; it never encodes a fixed two-field `weight/weight_scale` assumption.

`build_canonical_plan_groups()` runs only after it derives rank-local ownership
from both runtime parallel topologies and validates every binding required by a
source-served owner. It creates startup plans for source-proven frozen owners and
every-version plans for mutable owners. If a graph contains any mutable owner,
it is an every-version graph member, but its frozen independent owners stay in
the startup group and contribute only a startup-precondition digest. An
all-frozen graph has only a startup plan. An alias-only member references the
qualified canonical owner plan instead of producing a duplicate plan.

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
absence, and excludes training-only/checkpoint-served graphs from source load.
The startup group hashes ordered startup owner plans, immutable-contributor
cache keys, and alias mappings. The refit group hashes ordered graph-member
records, unique mutable-owner plan IDs, mixed-owner composition/finalizer plans,
alias mappings, the successful startup/cache precondition digest, and target
version into `transaction_group_id`. Checkpoint evidence remains serving
context, not a source transaction member.

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
- Consumes: serialized compiled rollout intents, including startup/every-version realization requests for participating graphs, and actual `vllm.__version__`; common bound-plan records from Task 7.
- Produces: generic frozen `RealizedCheckpointEvidence`, `DestinationCheckpointAttestor`, and `verify_realized_checkpoint_evidence()` in `nemo_rl.models.generation.interfaces`; frozen `VllmCapabilityProbes`; `VllmEndpointAdapter` with `capabilities()`, `configure_engine_kwargs()`, `describe_runtime_parallel_topology()`, `bind_realized_storage()`, `attest_checkpoint_realization()`, startup/update prepare/finalize methods, `load_component_batch()`, and `poison()`; `select_vllm_endpoint_adapter(version: str, probes: VllmCapabilityProbes) -> VllmEndpointAdapter`. Realized binding returns Task 7's `BoundDestinationPlans`; capability, immutable-contributor-cache, and placement fingerprints enter plan assembly only after realization. vLLM and SGLang implement the same attestor contract for checkpoint-served graphs.

- [ ] **Step 1: Write failing registry and isolation tests**

```python
@pytest.mark.parametrize(("version", "adapter_id"), [("0.25.1", "vllm-0.25.1"), ("0.28.0", "vllm-0.28.0")])
def test_exact_supported_version_selects_dedicated_adapter(version: str, adapter_id: str) -> None:
    assert select_vllm_endpoint_adapter(version, complete_probes()).adapter_id == adapter_id

def test_unknown_or_incomplete_vllm_fails_before_model_construction() -> None:
    with pytest.raises(UnsupportedVllmEndpointError, match="capability"):
        select_vllm_endpoint_adapter("0.29.0", incomplete_probes())
```

Add a test that importing the registry without vLLM installed succeeds, that selecting 0.28 never imports 0.25-only modules, and that two engines with different plans do not share process-global quantization state. Verify that a training-only graph creates no vLLM realization request, while startup-only and every-version owners both expose realized storage and placement for Task 7. A checkpoint-served graph uses its pinned native load context rather than a source-load binding, but serving remains closed until its post-load `RealizedCheckpointEvidence` exactly matches every field of `ImmutableAuxiliaryEvidence`. Add fatal stale-tag, stale-cache/path, resolved-revision, checkpoint-content, model-config, semantic-domain, and evidence-source mismatch tests. Run the same generic evidence verifier against a fake SGLang/static-drafter adapter so the contract is not vLLM-specific.

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

class DestinationCheckpointAttestor(Protocol):
    def attest_checkpoint_realization(
        self, graph_instance_id: str
    ) -> RealizedCheckpointEvidence: ...

def verify_realized_checkpoint_evidence(
    expected: ImmutableAuxiliaryEvidence,
    realized: RealizedCheckpointEvidence,
) -> None: ...

class VllmEndpointAdapter(DestinationCheckpointAttestor, Protocol):
    adapter_id: str
    def configure_engine_kwargs(self, intents: CompiledPrecisionIntentGroup, kwargs: dict[str, object]) -> None: ...
    def describe_runtime_parallel_topology(self) -> DestinationRuntimeParallelTopology: ...
    def bind_realized_storage(self, intents: CompiledPrecisionIntentGroup, model: object) -> BoundDestinationPlans: ...
    def attest_checkpoint_realization(self, graph_instance_id: str) -> RealizedCheckpointEvidence: ...
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
source field-for-field before serving. A stale tag, local cache entry, path, or
any mismatch poisons construction and fails the launcher. All destination
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
- Produces: cadence-preserving batched owner loads, canonical staging lifetime tracking, dirty-owner sets, exactly-once destination finalization, `DestinationStartupReady` for startup plans, and `DestinationCommitReady` for every-version plans only after completion fences.

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
        _quantize_bf16_to_mxfp8(owner, components[ComponentRole.LOGICAL_VALUES])
    elif owner.transform is TransformLocus.DESTINATION_NATIVE_LOADER:
        _load_logical_bf16_through_vllm(owner, components[ComponentRole.LOGICAL_VALUES])
    else:
        raise UnsupportedTransformError(owner.transform)
    self._dirty_owner_ids.add(owner.owner_id)
```

Use complete descriptors, never dtype-only dispatch. Keep logical staging alive through deferred native reload. Finalization pads/permutates/shuffles only dirty owners, records one completion event per batch, clears mutable canonical scratch after the fence, and raises if an owner is finalized twice within its cadence execution. Independent startup-finalized owners are sealed with their digest and rejected from later every-version batches. A mixed-cadence group instead retains verified immutable canonical inputs, accepts only its mutable inputs from the wire, and invokes its advertised preserve-or-repack composition/finalizer once per update.

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
- Produces: frozen `SourceParameterInventory`, source runtime parallel topology, Task 7's `BoundSourcePlans`, `bind_mxfp8_source(intents, inventory) -> BoundSourcePlans`, startup exports for source-proven frozen owners, and every-version exports for mutable owners. Native-MXFP8 owners export ordered direct `values`/`block_scales`; BF16 owners export logical BF16.

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

Add tests for grouped expert views with gradients, forged dtype metadata, mismatched scale geometry, disabled FP8 export, storage alias partitioning, synchronization before export, lifecycle-based inclusion of a served mutable MTP/drafter, exclusion of a mutable training-only auxiliary from source load, alias-owner de-duplication, and direct native component compatibility failure. Verify that a frozen source-served owner exports exactly once through the startup plan, a mutable owner exports only through the every-version plan, and a mixed graph returns both without duplicating either owner.

- [ ] **Step 2: Run source tests and observe RED**

Run: `uv run --no-sync pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py`

Expected: missing source adapter.

- [ ] **Step 3: Implement source binding from the canonical graph intents**

```python
def bind_mxfp8_source(intents: CompiledPrecisionIntentGroup, inventory: SourceParameterInventory) -> BoundSourcePlans:
    startup = _bind_semantic_sources(intents.startup_source_items, inventory)
    every_version = _bind_semantic_sources(intents.every_version_source_items, inventory)
    _validate_exact_cadence_coverage(intents, startup, every_version)
    _validate_component_geometry((*startup, *every_version))
    return BoundSourcePlans(intent_group_id=intents.intent_group_id, startup=startup, every_version=every_version)
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
- Consumes: validated `CanonicalStartupLoadPlanGroup` and `CanonicalRefitPlanGroup`, each exact rank/member/physical-owner acknowledgement set, source and destination `ray.ObjectRef` sets, plus registered abort/poison callbacks.
- Produces: `RefitPhase`, `RefitWorkerResult`, `RefitFailure`, one-shot `StartupLoadTransaction`, every-version `RefitTransaction`, `supervise_refit_futures()`, and bounded abort. Startup publishes a precondition digest before serving; every-version execution contains mutable owners only.

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

Add transaction-group tests using Task 7's main, MTP, and speculative-draft member records: every expected owning-rank FINALIZE acknowledgement is required, a served-draft-only failure prevents the main version from committing, and a stale served-draft target version is rejected. A mixed-cadence fused destination group owes exactly one finalizer acknowledgement per update even though only its mutable contributors appear in the repeated wire payload. Prove that mutable training-only and static checkpoint drafters are not transaction members; non-owning PP ranks owe no drafter acknowledgement; and alias-only MTP members reuse one physical-owner transfer/finalizer/acknowledgement. A constructed group that omits a declared owning-rank binding is rejected before PREPARE.

Add startup-load tests proving that every startup owner binds, transfers, and
finalizes exactly once before the serving gate opens; its digest is required by
later refits; it is absent from every-version payload/acknowledgement sets; and
binding, transfer, finalize, timeout, or malformed-result failure is propagated
immediately, poisons partial destination state, leaves serving closed, and exits
the launcher non-zero.

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
    STARTUP_READY = "startup_ready"
    COMMIT = "commit"
    ABORT = "abort"

@dataclass(frozen=True, slots=True)
class RefitWorkerResult:
    ok: bool
    rank: int
    phase: RefitPhase
    graph_instance_id: str
    physical_owner_id: str
    plan_id: str
    transaction_id: str
    detail: str | None = None
```

Use `ray.wait(..., num_returns=1, timeout=remaining_deadline)` across both source and destination refs and validate each ready result immediately. `StartupLoadTransaction` publishes `STARTUP_READY` only after its exact de-duplicated owning-rank set completes; it has no generation-version COMMIT and cannot run twice. `RefitTransaction` verifies the startup/cache digest and commits only after the exact acknowledgement set for every realized destination owner/finalizer group affected by a mutable contributor completes. Both sets are derived from Task 7 ownership and cadence closure, with aliases de-duplicated and non-participating graphs/ranks absent. On failure, preserve the first exception, run all abort/poison callbacks concurrently with a fixed teardown deadline, terminate owners whose abort does not acknowledge, then re-raise the first cause wrapped with structured context.

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
- Consumes: `StartupLoadTransaction`, `RefitTransaction`, and typed worker results from Task 11.
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

Add transport tests where a generation future fails while a train future remains pending, all-`None` results, malformed results, worker wrapper errors that currently return `False`, cache invalidation failure, failed `prepare_for_generation`, failed shutdown, and async refit after collector pause. Add auxiliary preflight cases for a mutable training-only MTP/drafter, a mixed frozen/mutable source-served MTP/drafter, an all-frozen source-served startup graph, a static checkpoint drafter, a missing drafter on an owning rank, and an absent drafter on a non-owning PP rank. Assert startup load runs once before collection, frozen owners never enter later payloads, and any startup/refit failure prevents resume/serve with bounded actor cleanup.

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

Replace boolean-driven refit selection with Task 7's validated cadence plans:

```python
startup_plans, refit_plans = build_canonical_plan_groups(...)
startup_digest = StartupLoadTransaction(startup_plans).run_before_serving()
RefitTransaction(refit_plans, required_startup_digest=startup_digest).run(...)
```

`trains_mtp`, `has_refit_draft_weights`, `loss_scaling_factor`, and
`detach_heads` may help discover or cross-check graph declarations, but none is
sufficient to decide owner mutability or cadence. Run a non-empty startup group
once before opening collection/generation; preserve its digest as a repeated
refit precondition. Register only graphs with mutable source-served owners as
every-version members and advance their served versions with the main COMMIT.
Reject a missing source/destination binding on a derived owning rank. Accept a
mutable training-only auxiliary with no destination, a fully evidenced
checkpoint-served auxiliary with no source plan, and absence on a derived
non-owning PP rank. Never silently disable MTP/speculative decoding because an
owning-rank drafter module was not realized.

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

Add both training modes for all five production families and exact fixture assertions for shared experts, routers, QKVO, dense layers, and output heads remaining BF16. Assert separately that built-in main-model roles never select MTP/draft addresses; BF16 defaults apply only to the training or rollout endpoint in which an auxiliary participates.

For Qwen3.5, Lightning, Ultra, Qwen3.8, and GLM fixtures that advertise MTP, exercise the full internal lifecycle matrix: mutable training-only, all-frozen `served_from_source`, mixed mutable/frozen `served_from_source`, and static `served_from_checkpoint` with complete immutable evidence. Add the same external speculative-drafter cases, including one different-family adapter. Assert that training-only and checkpoint-served owners create no source-load plan, frozen source-served owners join the one-shot startup group, only mutable source-served owners join each repeated payload, and a mixed graph's target version remains coherent with main while its startup digest is a commit precondition. Owning-rank absence fails, non-owning PP-rank absence succeeds, and tied aliases create no duplicate transfer/finalization. Include loss-scaling-zero and detached-head fixtures that remain mutable. Keep these declarations in model/runtime configuration and internal manifests; do not add lifecycle fields to the public `precision_policy` example.

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
14. MTP and speculative-drafter lifecycle diagnostics: explain graph-instance ID versus semantic graph path; training-only mutable, all-frozen and mixed mutable/frozen served-from-source, and static checkpoint-served cases; graph/model identity, pinned revision, content/configuration/semantic-domain digests, and typed evidence source; owning versus non-owning PP ranks; tied-alias de-duplication; and why zero loss scaling or detached heads do not prove freezing. State clearly that these are internal/model-runtime declarations, not additional public precision-policy selector fields. Explain the one-shot serving gate for frozen source-served owners and that only mutable source-served owners repeat inside atomic every-version refit.
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
- Produces: reproducible correctness/fault/performance artifacts tied to branch, SHA, image digest, model revision, policy/plan digests, topology, auxiliary lifecycle/evidence records, expected rank-local ownership, raw samples, and job logs.

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
                verify_realized_checkpoint_evidence(
                    graph["immutable_evidence"], graph["realized_checkpoint_evidence"]
                )
```

The scripts must support `--dry-run` and print the resolved cluster, account, branch, exact SHA, container digest, nodes/GPUs, model revision, policy/plan digest, auxiliary lifecycle/evidence summary, per-owner mutability and derived cadence, expected rank-local endpoint ownership, startup-precondition digest, command, time limit, and log directory without submitting.

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

Run each script with `--dry-run`, review its resolved submission, then submit. Cover both training modes for all five production models, mixed BF16/MXFP8 boundaries, specified TP/EP/PP/padding rows, repeated A→B→C numeric refits, fresh-load logprob comparison, injected binding/transfer/finalize/commit/silent-peer failures, and at least twenty paired steady-state performance samples where p95 is claimed. The auxiliary matrix must include mutable training-only MTP/draft success without a destination plan, all-frozen source-served one-shot startup, mixed frozen/mutable source-served startup plus atomic repeated refit, static checkpoint evidence/no-source-transfer behavior, tied-alias de-duplication, fatal missing owning-rank drafter storage, and valid absence on non-owning PP ranks. Assert frozen owners are never retransferred after startup and their startup digest remains a repeated-refit precondition. Monitor each new job for the required first five minutes and cancel/release resources immediately on a fatal failure.

- [ ] **Step 5: Evaluate hard gates and write the evidence report**

```python
assert every_numeric_case_passed
assert every_sync_and_async_fault_case_exited_nonzero_within_budget
assert refit_p50_ratio_upper_95ci <= 1.05
assert refit_p95_ratio_upper_95ci <= 1.05
assert generation_latency_ratio_upper_95ci <= 1.05
assert generation_throughput_ratio_lower_95ci >= 0.95
assert every_instantiated_training_auxiliary_was_accounted
assert every_source_served_frozen_owner_loaded_once_before_serving
assert only_mutable_served_from_source_owners_joined_each_refit_payload
assert every_refit_verified_the_startup_precondition_digest
assert no_frozen_contributor_was_retransferred_after_startup
assert every_expected_owning_rank_bound_and_acknowledged
assert no_alias_owner_was_transferred_or_finalized_twice
```

The report maps retained code to PRs #3477/#3630/#3659/#3669/#3907/#3908/#3909/#3294 and lists minimal restack ranges. Correctness/transaction commits remain separate from independently measured performance commits. Do not update an existing PR until every hard gate passes on both clusters.

- [ ] **Step 6: Commit only durable validation scripts and completed evidence**

```bash
git add tests/functional/precision_policy_matrix.sh tests/functional/refit_transaction_fault_matrix.sh tests/functional/refit_performance_matrix.sh tests/fixtures/precision_policy/cluster_matrix.yaml tests/unit/precision_policy/test_cluster_matrix.py tests/functional/L1_Functional_Tests_GB200_MXFP8.sh tests/test_suites/performance_gb200.txt docs/performance/semantic-precision-refit-validation.md
git commit -s -m "test(refit): validate semantic precision matrix"
```
