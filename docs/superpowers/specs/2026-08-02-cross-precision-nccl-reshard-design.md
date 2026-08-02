# Cross-Precision NCCL-Reshard Design

## Summary

Preserve the validated Megatron BF16-storage to vLLM MXFP8-rollout path as an
immutable baseline, then replace its MXFP8-specific NCCL-Reshard metadata and
transfer branches with a small component-based transform contract.

The first implementation supports only the already validated BF16-to-MXFP8
runtime path. Its interfaces must represent arbitrary ordered tensor
components so later NVFP4 and other storage conversions can be added without
changing the synchronizer or transfer loop.

## Goals

- Preserve the validated BF16-storage to MXFP8-rollout implementation and
  performance result.
- Remove precision-specific behavior from the NCCL-Reshard synchronizer and
  transfer loop.
- Represent one logical parameter as any number of transferred components.
- Separate source conversion, transport, and destination loading.
- Reject unsupported storage-format pairs before entering collectives.
- Allow a future format pair to be added through a codec and backend adapter,
  without changing the core synchronizer.
- Keep the production-code increase small by replacing existing MXFP8 branches
  instead of adding a parallel implementation.

## Non-Goals For The First PR

- Runtime support for NVFP4 W4A4 or W4A16.
- Support for every training and generation backend.
- Native `nccl.m2n` performance claims. The current validated environment uses
  the Python exact-transfer implementation over NCCL communicators.
- Changes to quantization math, MXFP8 layout rules, or model accuracy.
- A general graph compiler or user-extensible plugin framework.

## Stable Baseline

Create the following references at commit
`a854dbdbe33eccee80c32c8e2025fb0ac59d26d5`:

- Branch: `sna/nccl-reshard-bf16-mxfp8-stable-v1`
- Annotated tag: `nccl-reshard-bf16-mxfp8-stable-v1`

The stable reference remains unchanged. Development starts in a separate
worktree on:

- Branch: `sna/nccl-reshard-cross-precision`

## Design Principles

1. The transport does not know precision names.
2. A codec does not know Ray, NCCL, or worker topology.
3. A backend adapter does not know the other backend's internal layout.
4. Training compute precision is distinct from parameter storage format.
5. A weight version is committed only after every required component arrives.
6. Unknown format pairs and mismatched plans fail before data transfer.

For example, MXFP8 training with `fp8_param=false` still advertises BF16
parameter storage. Only `fp8_param=true` changes the source storage contract.

## Minimal Data Model

Add one small module,
`nemo_rl/weight_sync/refit_transforms.py`. It contains serializable data only;
backend callables remain local to their workers.

```python
@dataclass(frozen=True)
class TransformComponentSpec:
    role: str
    global_shape: tuple[int, ...]
    dtype_name: str


@dataclass(frozen=True)
class RefitTransformPlan:
    transform_id: str
    components: tuple[TransformComponentSpec, ...]
    finalize_scope: Literal["parameter", "layer", "model"]


class RefitTransformCodec(Protocol):
    def describe_outputs(...) -> tuple[TransformComponentSpec, ...]: ...
    def encode(...) -> tuple[torch.Tensor, ...]: ...
```

The existing NCCL-Reshard metadata builder combines these topology-independent
component descriptions with source and destination meshes and placements. This
keeps topology out of the codec without introducing another production class.
Canonical string dtype names and plain placement metadata keep serialization
and plan hashing deterministic across Ray workers.

The first registry remains a module-level mapping rather than a class-based
plugin manager:

```python
TRANSFORM_CODECS = {
    ("bf16", "mxfp8_e4m3_e8m0"): BF16ToMXFP8Codec(),
}
```

The registry key describes parameter storage, not training compute mode.

## Initialization And Negotiation

The current two-stage metadata handshake is retained but generalized:

1. The policy reports logical HF parameter metadata and source storage format.
2. The generation backend reports target storage requirements per parameter.
3. The shared registry resolves each `(source_format, target_format)` pair.
4. The policy installs the selected source codecs and returns component
   metadata after conversion.
5. Both sides independently construct a canonical transfer plan.
6. The synchronizer compares plan version, component count, and plan hash.
7. NCCL communicators transfer data only after the plans match.

The existing `list[str]` prequantization response becomes a structured request:

```python
RefitTransformRequest(
    parameter_name=name,
    source_format="bf16",
    target_format="mxfp8_e4m3_e8m0",
)
```

The existing `initialize_refit_metadata` helper owns this negotiation for all
transports. The NCCL synchronizer must call the helper rather than duplicating
its own Megatron-specific handshake.

## Per-Refit Data Flow

For every logical parameter:

1. The source adapter obtains the local Megatron tensor shard.
2. The selected codec produces components in the plan's fixed order.
3. The transport reshares each component from source placements to destination
   placements.
4. The vLLM destination adapter validates each received component and records
   it as ready for the current weight version.
5. The adapter performs parameter-, layer-, or model-level finalization only
   after every required component is ready.
6. All generation workers acknowledge the same completed weight version.

For BF16-to-MXFP8, the component order is:

1. E4M3 weight values.
2. E8M0 block scales.

The quantization math and destination layout conversion remain unchanged.

## Backend Boundaries

### Megatron Source Adapter

- Advertises actual parameter storage.
- Maps HF names to local source tensors and placements.
- Runs codec `encode` on the trainer GPU.
- Produces component tensors in the negotiated order.

### vLLM Destination Adapter

- Advertises required target storage and component family.
- Maps logical HF components to vLLM local buffers.
- Performs existing grouped-expert fusion and backend layout conversion.
- Commits only after the component set is complete.

### NCCL-Reshard Transport

- Iterates `plan.components`.
- Moves tensors according to source and destination placements.
- Handles the existing misc fallback for parameters outside the bulk path.
- Does not branch on MXFP8, NVFP4, scale names, or block sizes.

Future Megatron, DTensor, vLLM, TRT-LLM, or SGLang combinations add source or
destination adapters while reusing the plan and transport contract.

## Error Handling

Validation is fail-closed and runs before communicator data transfer:

- Unknown source or destination storage format.
- Missing codec for a requested pair.
- Incomplete or duplicate component roles.
- Shape, dtype, placement, or block-alignment mismatch.
- Source and destination plan-hash mismatch.
- Backend does not support the requested finalization scope.
- Partial component arrival or failed finalization.

The current vLLM loader updates buffers in place, so rollback to the old weight
version is not promised. If transfer or finalization fails after any component
is loaded, the synchronizer raises, keeps the update stale, and the affected
generation actors must not serve another rollout. No worker may acknowledge
success after a partial load. Transactional double-buffered rollback is outside
the first PR because it would materially increase GPU memory and code size.

The first PR also closes current validator holes, including accidental
acceptance of blockwise-FP8 storage for an MXFP8 destination without producing
the required MXFP8 scales.

## Code-Size Control

- Add one small transform-contract module.
- Reuse existing metadata, placement, and `xferdtensor` helpers.
- Move MXFP8 validation and component construction into the first codec.
- Replace the existing value/scale special cases with one component loop.
- Do not retain a second legacy MXFP8 implementation in the development branch.
- Avoid abstract base classes when a `Protocol`, dataclass, and dictionary are
  sufficient.
- Keep NVFP4 runtime implementation out of the first PR.

The expected result is a modest net increase: reusable contract and tests are
added while the existing hard-coded MXFP8 branches are removed.

## Test Strategy

### Unit Tests

- Registry resolves BF16-to-MXFP8 and rejects unknown pairs.
- Storage detection distinguishes FP8 compute from FP8 parameter storage.
- MXFP8 plan contains ordered value and scale components.
- Plan serialization and hashing are deterministic.
- Component shape, dtype, placement, and alignment mismatches fail early.
- Partial component sets never finalize.
- A mock four-component codec executes through the generic loop without core
  synchronizer changes. This demonstrates NVFP4 extensibility without claiming
  NVFP4 runtime support.
- Existing BF16 and matching blockwise-FP8 paths remain valid.

### Parity Tests

- Stable and development branches export bitwise-identical MXFP8 values and
  scales for fixed BF16 inputs.
- Stable and development branches produce identical destination weights.
- Existing NCCL-Reshard and refit unit suites pass.

### GCP-NRT Functional And Performance Gates

- Two-step correctness smoke on B200.
- Twenty-step matched run using the validated Qwen3-30B-A3B recipe.
- Compare steps 3-20 for refit, E2E, throughput, reward, and generation KL.
- Require no statistically measurable reward or KL regression.
- Require no material refit or E2E performance regression relative to the
  stable branch.

## Follow-Up PRs

After the first PR establishes the contract:

1. Add BF16-to-NVFP4 W4A16 and W4A4 codecs plus ModelOpt atomic finalization.
2. Add supported FP8-parameter-storage to NVFP4 codecs.
3. Extend backend adapters beyond Megatron-to-vLLM.
4. Package and benchmark native `nccl.m2n` independently from transform logic.

Each follow-up adds a codec or adapter and tests. It must not add precision
branches to the synchronizer.

## Acceptance Criteria

- The stable branch and tag point to the validated snapshot.
- The development work is isolated in its own worktree.
- BF16-storage to MXFP8 rollout remains correct and performance-neutral.
- The NCCL transfer loop contains no MXFP8-specific value/scale handling.
- Unsupported precision pairs fail before collectives start.
- A mock multi-component codec works without modifying the synchronizer.
- The design can add NVFP4 through a codec and destination finalizer rather
  than another transport implementation.
