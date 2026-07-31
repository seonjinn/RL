# Nemotron THD Transformer Engine CUDA Graph Correctness Design

Date: 2026-07-31

## Objective

Implement correctness-preserving Transformer Engine partial CUDA Graph training
for packed THD workloads across Nemotron 3 Nano, Super, and Ultra architecture
paths.

The implementation must support every model-compatible combination of the
existing `CudaGraphModule` regions:

- `attn`;
- `mlp`;
- `mamba`;
- `moe`;
- `moe_router`;
- `moe_router,moe_preprocess`;
- combinations of the independent attention, dense-MLP, Mamba, and MoE axes;
- empty `cuda_graph_modules`, which means whole-layer TE capture.

`shared_expert` and `moe_act` are configuration dimensions, not graph-scope
names. They are tested inside the relevant MoE boundary.

Correctness is a merge gate. Performance is measured only after forward,
backward, optimizer, router, padding, and NeMo-RL accuracy parity pass.

## Source Baseline and Provenance

Use the isolated stack:

- NeMo-RL worktree:
  `/Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731`;
- NeMo-RL branch:
  `experiment/thd-cg-hybrid-nemotron-20260731`;
- Megatron-Bridge branch:
  `sna/thd-cg-hybrid-nemotron-20260731`;
- Megatron-LM branch:
  `sj/thd-cg-hybrid-nemotron-20260731`.

Discovery anchors recorded on 2026-07-31:

| Repository | Reference | Commit |
|---|---|---|
| NeMo-RL | `origin/main` | `d152853a91fc5e1e1f66fc06e3a7e5ff5fb6ef7e` |
| Megatron-Bridge | `upstream/main` | `3bc95ef5fa4a76b7155fb090bdcfa1bf643bd56f` |
| Megatron-LM | `upstream/main` | `b19b1f47cf7e289607f3be480c5f06c6ada25b16` |
| Megatron-LM | `upstream/dev` | `95e4bafebaa799d166975ef82066a3c46648e004` |
| Megatron-LM PR 5672 | head | `6ff66f0a000ee65efa4f322c17871a3938f33427` |
| Transformer Engine | `main` | `869f99c47d5773e3dbf4a85d4cc8679c4e050089` |

Start from latest main branches. Do not merge Megatron-LM `dev` wholesale.
Port only the reviewed invariants listed below, preserving main's newer CUDA
Graph lifetime fixes.

### Required upstream invariants

| Source | Commit | Use |
|---|---|---|
| Megatron-LM PR 5672 | `6ff66f0a0` | Attention `PackedSeqParams` to TE graph adapter and tests |
| Megatron-LM PR 5975 | `4b18b260f` | Already on main; graph-pool aliases, `seq_idx`, saved tensors, explicit dispatcher outputs |
| Megatron-LM PR 5724 | `3ae5e6d9a` | Port fixed THD cumulative-sequence capacity for eager and graph paths |
| Megatron-LM PR 5541 | `1392f28a5` | Port real-versus-physical padded THD sequence-length semantics |
| Megatron-LM PR 5542 | `d1384c2d9` | Port removal of padding rows from routing/capacity accounting |
| Megatron-LM PR 5668 | `904ef6d86` | Port the invariant that HybridEP graph inputs are statically padded before graph entry |
| Megatron-LM PR 5401 | `3fae48f4a` | Port when router z-loss is enabled; avoid capture-time host scalar construction |

Megatron-LM PR 4359 remains the architectural reference for fixed-capacity THD
and padding-mask propagation. PR 5258 and PR 5618 are test/reference material;
they are not implementation bases.

PR 5635, merge `bf32f4415`, is already in PR 5672's ancestry. Treat its
TP-broadcast preservation of per-sample packed metadata as a baseline
regression gate, not a post-5672 port.

The following open work is reference-only:

- PR 6022, head `20c9a4133`, for the proposed whole-MoE HybridEP graph
  contract; it is blocked and must not be used as the implementation base;
- PR 6167, for dense-MLP local graph-manager initialization;
- PR 6168, for local last-layer pseudo-deallocation and viewless outputs.

Cherry-pick none of these by default. Add focused regression tests only when
the selected topology exercises their exact failure mode.

### Transformer Engine runtime

The runtime must contain:

- PR 2898 for graph-safe THD attention and MoE auxiliary token counting;
- PR 2937, merge `a6c70f4dc`, for parameter-gradient lifetime correctness;
- PR 3268, merge `8c606cadf`, for device-side attention padding-mask construction
  when unfused attention can be selected.

Megatron-LM main's pinned TE contains PR 2898 but predates PR 2937 and PR 3268.
Use one immutable staged TE artifact from `869f99c47` or a later verified
commit. Do not rebuild Transformer Engine inside every NeMo-RL job.
`NRL_FORCE_REBUILD_VENVS=true` may rebuild the Python environment, but the
native TE artifact remains provenance-pinned and reusable.

## Confirmed Root Cause

Packed NeMo-RL creates a fixed physical graph capacity. In the failing Nano
run, each graphable rank had capacity 2048 while the runtime logical token
counts were 1746 or 1776.

`moe_router,moe_preprocess` capture replays Tensor dispatcher attributes, but
does not replay Python structural state such as `hidden_shape`. The eager
dispatch, expert, and combine tail then returns the logical extent. The next
Mamba or TE graph still has a capacity-sized static input and capacity-sized
`seq_idx`.

This produces either:

- Mamba rejecting a `seq_idx` whose shape no longer matches hidden states; or
- TE attempting to copy a logical tensor into a capacity-sized static surface.

Appending zeros after the MoE output is not a valid fix. Packed-sequence
alignment padding can occur inside the physical token layout, and CP zigzag
changes row order. Once permutation/combine returns an undersized tensor, there
is no safe prefix assumption for reconstructing token positions.

## Core Invariants

### Physical and logical geometry

Every graphable layer boundary uses one fixed physical token capacity:

```text
NeMo packed capacity
  -> CP/SP physical layout
  -> attention, Mamba, and partial-MoE graph boundaries
  -> eager dispatcher/expert/combine continuation
  -> residual/BDA
```

No stage may narrow hidden states from physical capacity to logical occupancy.

Logical validity is represented only by:

- real `cu_seqlens`;
- physical `cu_seqlens_padded`;
- Mamba `seq_idx`;
- a boolean structural padding mask;
- loss-specific token and sample masks outside the model-layer contract.

The structural padding mask is distinct from NeMo-RL's loss `token_mask`.
Prompt tokens can be valid model tokens even when excluded from the policy
loss.

### Routing correctness

Physical padding rows must not:

- contribute to router auxiliary or z loss;
- update expert-bias counts;
- consume expert capacity;
- cause a valid token to be dropped;
- alter valid-token gradients.

The router and dispatcher may retain capacity-sized tensors for graph shape,
but validity must be applied before logical capacity accounting, following the
PR 5541 and PR 5542 semantics.

### Replay signatures

Before entering TE replay, validate every graph input's:

- field presence;
- shape;
- dtype;
- device;
- layout;
- stride;
- static packed metadata.

A mismatch fails before any rank launches a graph or collective. There is no
rank-local mid-forward eager fallback.

An unseen but in-bounds schedule geometry can capture a new graph bank only at
a globally drained pre-forward boundary. Token or packed-sequence overflow is
repacked or rejected before model execution.

## Selected Architecture

### Considered approaches

1. Dispatcher-owned replay state
   - Each dispatcher snapshots and restores the continuation state it owns.
   - Graph banks own the corresponding immutable structural state.
   - This is selected because alltoall, allgather, and HybridEP have different
     continuation contracts.

2. `TransformerLayer` dispatcher-type branches
   - Directly assigning dispatcher fields is smaller initially.
   - It couples the layer to private dispatcher fields and already fails to
     cover Flex, capacity, and shared-expert state.
   - This approach is rejected.

3. Merge Megatron-LM `dev` or PR 5258 wholesale
   - This imports unrelated divergence and an older/diverged TE pin.
   - It makes regression attribution and NeMo-RL compatibility harder.
   - This approach is rejected.

### Dispatcher-owned replay contract

Add a typed dispatcher replay-state interface to `MoETokenDispatcher`.
Production code must not switch on dispatcher class names inside
`TransformerLayer`.

The interface provides:

```python
def snapshot_cudagraph_replay_state(
    self,
    *,
    capacity_hidden_states: torch.Tensor,
    preprocessed_hidden_states: torch.Tensor,
) -> MoECudaGraphReplayState:
    ...

def restore_cudagraph_replay_state(
    self,
    state: MoECudaGraphReplayState,
    *,
    capacity_hidden_states: torch.Tensor,
    preprocessed_hidden_states: torch.Tensor,
) -> None:
    ...

def validate_cudagraph_continuation(
    self,
    *,
    capacity_hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    ...
```

The exact type names may follow an existing MCore pattern, but the interface
and ownership boundary are fixed by this design.

State requirements:

| Dispatcher | Banked/restored structural state |
|---|---|
| AllGather | local `hidden_shape`; eager `dispatch_postprocess` remains the owner of global `hidden_shape_before_permute` |
| AlltoAll | `hidden_shape`, `hidden_shape_before_permute`, Python-valued capacity and output-count invariants |
| Flex/HybridEP | top-level `hidden_shape`, original/padded token counts, `num_permuted_tokens`, capacity, and drop/pad `tokens_per_expert` when applicable |
| Flex/DeepEP | `hidden_shape`; Tensor token probabilities/indices remain explicit graph outputs |
| Flex/NCCL-EP | `hidden_shape`, local token count, and a pre-bootstrap maximum-capacity compatibility check |

Alltoall, allgather, and HybridEP are required production paths. DeepEP and
NCCL-EP receive state-contract unit tests and dependency-gated GPU smokes.
They remain fail-closed for packed partial TE graphs until their external
runtime and static-buffer tests pass.

After eager expert/combine, output shape must exactly equal residual capacity.
The output is never padded, truncated, or narrowed at this point.

### Shared expert and delayed weight gradients

Non-overlapped shared expert:

- runs inside `moe_router`;
- crosses the graph boundary as an explicit Tensor output;
- requires exact shape and gradient parity.

Alltoall overlapped shared expert:

- graph replay restores `gate_score` and `cached_fc1_input`;
- dispatcher replay also restores the shared-expert state-machine transition
  expected by the eager continuation;
- delayed shared-expert weight-gradient suppression occurs only if that work
  was actually captured.

Flex shared-expert overlap starts in eager `token_dispatch`, so it must not
receive the alltoall capture-time state transition.

### Attention and Mamba packed adapters

Attention retains PR 5672's four Tensor fields:

- `cu_seqlens_q`;
- `cu_seqlens_kv`;
- `cu_seqlens_q_padded`;
- `cu_seqlens_kv_padded`.

Mamba uses:

- `seq_idx` for packed execution;
- query real/padded cumulative lengths only when the Mamba/CP kernel consumes
  them.

Mamba does not receive unused KV fields. `total_tokens` stays outside replay
reconstruction so a graph cannot allocate a replacement `seq_idx`.

The structural padding mask is included as a fixed-shape graph input and copied
into the static surface on replay. Its values may change while its signature
does not.

### Graph-bank ownership

Keep the existing two-bank bounded schedule design, but add dispatcher state
and structural-mask signatures to each bank fingerprint.

A bank owns:

- graph callable identities;
- exact layer topology and scope;
- packed Tensor/static signatures;
- structural padding-mask signature;
- dispatcher Tensor attribute schema;
- dispatcher replay structural state;
- normalized pipeline microbatch count.

Switching, capture, eviction, and reset occur only between optimizer steps
after forward, backward, delayed wgrad, shared-expert, and communication state
is drained. A capture failure restores the previously active bank.

## Architecture Coverage

### Layer discovery

The existing graph scope enum remains unchanged:

```text
attn, mlp, moe, moe_router, moe_preprocess, mamba
```

Hybrid allocation symbols map to:

- `M`: `MambaLayer`;
- `*`: attention `TransformerLayer`;
- `-`: dense MLP layer;
- `E`: MoE layer.

Scopes apply only to matching physical layer types. A model-incompatible
request fails during topology discovery rather than silently reporting a
zero-coverage success.

Super's repeated hybrid MTP stack must be included in TE discovery by
descending into its nested `HybridStack`. The graph fingerprint includes these
inner layer identities. If a supported MTP layout cannot expose graphable
children safely, setup fails explicitly; it does not silently leave the MTP
loss path eager while claiming full scope coverage.

### Model/backend matrix

| Architecture | Required path |
|---|---|
| Nano | attention + Mamba + MoE hybrid; packed TP2/PP2/CP2/EP8; validate the current NeMo-RL recipe backend and a Bridge Flex/HybridEP override as separate rows |
| Super | attention + Mamba + latent MoE + MTP; AlltoAll router/preprocess; non-overlapped shared expert |
| Ultra | same `HybridModel` provider family; Flex/HybridEP MoE and large CP/EP topology; first validate with Bridge standalone, then NeMo-RL when a concrete model path and launch profile are available |

Do not invent a fake Ultra GRPO recipe. The repository's Ultra launch configs
receive their model path externally. Until a real checkpoint/profile is
resolved, Ultra is a Bridge/MCore correctness fixture and its NeMo-RL row is
reported as blocked, not passed.

### Scope matrix

The theoretical TE matrix is:

```text
dense axes: any subset of [attn, mlp, mamba]
MoE axis: one of [none, moe, moe_router, moe_router+moe_preprocess]
```

This yields 32 TE rows, including empty-scope whole-layer capture. A no-CG
baseline is a separate row.

Preflight classifies every theoretical row:

- runnable for the selected topology;
- unsupported because the model has no matching layer;
- unsupported because full `moe` lacks static drop-and-pad capacity;
- dependency-blocked;
- submitted.

Full `moe` is tested only when expert input is statically capacity-padded and
overflow monitoring proves no valid-token drop. Production dropless recipes
are not silently changed to make the scope run.

## NeMo-RL Changes

Forward-port only reviewed CUDA Graph work from the prior experiment. Do not
merge the old branch wholesale.

Required NeMo-RL responsibilities:

1. Validate scope names and incompatible combinations.
2. Pad physical packed tokens and cumulative metadata to configured bounds.
3. Build a structural validity mask in the exact pre-CP token layout.
4. Transform the mask through the same CP/SP layout as hidden states.
5. Carry it through `ProcessedInputs` and `ProcessedMicrobatch`.
6. Pass it into `HybridModel.forward(..., padding_mask=...)`.
7. Reject token and sequence-count overflow before graph entry.
8. Count three globally successful optimizer updates before capture.
9. Activate/capture a graph bank before the forward/backward schedule.
10. Expose capture, replay, hit, eviction, fallback, and capacity-utilization
    telemetry in worker results and W&B/TensorBoard.

`fallback_count` remains zero by construction. Failures raise; they do not
increment a counter and continue eagerly.

## Error Handling

Fail at setup when:

- a requested scope has no matching graphable layer across the model;
- `moe_preprocess` is requested without `moe_router`;
- `moe` and `moe_router` are both requested;
- packed token or sequence capacity is missing;
- the runtime TE commit lacks required correctness fixes;
- a requested dispatcher lacks a verified replay-state implementation;
- full `moe` lacks static expert capacity;
- a graph bank belongs to a different model, topology, scope, dispatcher
  schema, or packed signature;
- MTP children requested by the scope cannot be discovered.

Fail before TE replay when an input signature changes.

Fail before eager communication when restored dispatcher state disagrees with
the graph outputs.

Fail after combine and before BDA when MoE output does not exactly match
capacity. Never repair it with output padding.

## Testing Strategy

### TDD contract tests

Write and observe failures before production changes for:

- allgather state snapshot/restore;
- alltoall state snapshot/restore;
- HybridEP state snapshot/restore;
- DeepEP/NCCL-EP fail-closed gates;
- exact hidden/input/output replay signatures;
- logical 12 / capacity 16 with internal per-sequence padding;
- structural-mask CP/SP ordering;
- padding rows excluded from router loss, bias counts, and expert capacity;
- shared expert overlap state and delayed-wgrad ownership;
- MTP nested `HybridStack` discovery;
- graph-bank dispatcher-state activation and reset;
- NeMo telemetry propagation.

Use real dispatcher implementations. Small unit tests may stub collectives,
but must exercise actual permutation/unpermutation mappings rather than a
`SimpleNamespace` approximation.

### Distributed MCore gates

Run eager versus TE graph with identical weights, inputs, and seeds:

- GPT dense: `attn`, `mlp`, `attn+mlp`;
- hybrid attention/Mamba;
- allgather and alltoall `moe_router,moe_preprocess`;
- alltoall latent MoE;
- Flex/HybridEP partial MoE;
- shared expert overlap off/on where supported;
- top-1/top-2 routing;
- dropless and static drop-and-pad;
- TP2/CP2/EP and PP2 configurations;
- MTP enabled/disabled.

Compare:

- valid-token forward outputs;
- total and router auxiliary losses;
- router top-k indices exactly;
- all parameter gradients;
- optimizer-updated parameters;
- padding-row gradients and expert-capacity accounting.

Use three warmup steps, alternating fixed-signature occupancies after capture,
and at least 8-20 replay steps. Run a 100-step attention and combined hybrid
soak because upstream issue 5966 reports a possible late graph gradient
instability.

### NeMo-RL gates

Persistent scripts live under:

`experiments/cuda_graph/nemotron_thd_te_graph_20260731/`

Use W&B project `sna-cg-study` and checkpointing disabled.

For each model:

1. test-only scheduler preflight;
2. baseline plus targeted five-step correctness rows;
3. model-compatible scope smokes in parallel;
4. baseline plus selected 20-step attribution rows;
5. 100-step paired convergence/gradient soak for the best correct scope.

Report for E2E, generation, policy training, and combined policy/reference
logprob:

- step time;
- tokens/second/GPU;
- median, p95, and repeat variance;
- graph calls divided by graph-eligible module calls;
- capture/replay/cache/fallback counts;
- logical, padded, and capacity token counts.

Accuracy reporting includes:

- policy loss;
- reward;
- `gen_kl_error`;
- token/multi-logprob error;
- router indices and expert counts;
- gradient norm;
- parameter delta;
- NaN/Inf checks.

The static HTML report separates correctness, smoke, performance, accuracy,
coverage, failures, unsupported rows, and source/runtime provenance.

## Delivery Order

1. Commit this design and obtain user approval.
2. Write and commit the implementation plan.
3. Build the MCore branch from latest main plus the reviewed upstream ports.
4. Implement dispatcher replay state and padding correctness using TDD.
5. Implement Mamba/MTP discovery and graph-bank integration using TDD.
6. Push MCore to the personal fork.
7. Pin and push the Megatron-Bridge branch.
8. Forward-port NeMo-RL graph lifecycle, mask plumbing, telemetry, and harness
   using TDD.
9. Pin the Bridge branch and push NeMo-RL.
10. Stage one immutable compatible nightly/TE runtime.
11. Submit and monitor correctness gates before performance jobs.

## Completion Criteria

The work is complete only when:

- focused CPU/contract tests pass;
- distributed eager/graph forward, backward, router, and optimizer parity
  passes;
- padding rows have no model, router, capacity, or gradient effect;
- Nano and Super complete their model-compatible five-step scope matrices;
- Nano and Super complete paired 20-step performance/accuracy comparisons;
- combined `attn,mamba,moe_router,moe_preprocess` runs with zero fallback;
- Super MTP coverage is explicit and verified;
- Ultra Flex/HybridEP passes Bridge/MCore correctness, and NeMo-RL Ultra is
  either run with a real resolved profile or reported as externally blocked;
- the selected best scopes pass a 100-step stability check;
- the HTML report contains graph call coverage and provenance;
- MCore, Bridge, and NeMo-RL branches and submodule pointers are pushed in
  dependency order.
