# Graph-Safe Variable Packed Sequence Auxiliary Loss Design

Date: 2026-07-31

Status: Approved correctness amendment to
`2026-07-31-nemotron-thd-te-cuda-graph-correctness-design.md`.

## Objective

Make `seq_aux_loss` mathematically equivalent for padded eager, variable-length
packed eager, and Transformer Engine partial CUDA Graph execution. The same
fixed-shape contract must work for packed THD MoE leaves in Nemotron 3 Nano,
Super, and Ultra, including Hybrid Mamba and MTP paths.

This work is a correctness prerequisite for all later MoE scope, performance,
and convergence jobs. No packed `seq_aux_loss` result is accepted until this
design's padded-versus-packed loss and gradient gates pass.

## Confirmed Failure

NeMo-RL variable packing flattens logical samples into `[T, 1, H]`. MCore's
current router therefore observes `bsz == 1` and groups every token into one
sequence. The structural padding mask correctly removes padding routes, but it
cannot recover which logical sample owns each valid token. The current result
is wrong whenever a pack contains unequal logical samples; fixed-width
`tokens_per_sample` is the only correct existing exception.

Megatron-LM PR 6115 demonstrates the correct segmented loss semantics, but its
implementation is not a valid graph base because it:

- derives ownership from variable `cu_seqlens` inside model execution;
- uses Python/data-dependent control flow and runtime-sized allocations;
- mutates a shared `PackedSeqParams` object as a cache; and
- explicitly rejects TE CUDA Graph execution.

PR 6115 is a math and parity-test reference only. It must not be cherry-picked.

## Fixed-Shape Ownership Contract

Add these fields to `PackedSeqParams`:

```python
seq_aux_loss_sample_ids: Tensor | None = None
seq_aux_loss_num_samples: Tensor | None = None
seq_aux_loss_max_samples: int | None = None
```

| Field | Exact contract | Graph role |
|---|---|---|
| `seq_aux_loss_sample_ids` | contiguous CUDA `torch.int64`, shape `[T_router_capacity]` | Dynamic Tensor input with a fixed signature |
| `seq_aux_loss_num_samples` | contiguous CUDA `torch.int64`, scalar shape `[]` | Dynamic Tensor input; exact real sample count `N` |
| `seq_aux_loss_max_samples` | Python `int`, value `thd_max_packed_sequences - 1` | Static graph-bank capacity `S_cap` |

`N` excludes the append-dummy sequence and repeated cumulative-length capacity
entries. It satisfies `1 <= N <= S_cap`.

NeMo-RL enforces that bound from its authoritative Python sample count before
yielding a microbatch. MCore repeats the bound as a dynamic device assertion:

```python
num_samples_is_valid = (
    (seq_aux_loss_num_samples >= 1)
    & (seq_aux_loss_num_samples <= seq_aux_loss_max_samples)
)
torch._assert_async(
    num_samples_is_valid,
    "seq_aux_loss_num_samples must be in [1, seq_aux_loss_max_samples]",
)
```

The MCore check must not read the scalar on the host. In the pinned runtime,
Task 12 must prove that this device assertion captures and replays in the same
CUDA graph. If the pinned PyTorch runtime cannot capture it, the variable
packed `seq_aux_loss` graph scope remains unsupported; it must not silently
fall back or replace the assertion with `.item()`.

`T_router_capacity` is the hidden-state extent presented to the local router:

```text
T_global_capacity / context_parallel_size
    / tensor_parallel_size  when sequence_parallel is enabled
T_global_capacity / context_parallel_size
    otherwise
```

### Token ownership convention

For each real sample `s`, every physical slot in that sample's padded segment
uses ID `s`. This includes masked inter-sequence alignment padding. Append-dummy
and unused tail slots use ID `0`; the structural padding mask already makes
their routing map and auxiliary scores zero, so they cannot affect row zero.

The producer creates IDs in this exact order:

1. construct one constant-ID vector for each real physical sample segment;
2. apply the existing per-sequence CP zigzag selector to each vector;
3. concatenate the CP-local vectors and CP-local append-dummy tail; and
4. when sequence parallelism is enabled, take exactly one contiguous TP-rank
   slice of the CP-local vector.

A global contiguous CP slice is invalid. Slicing for SP more than once is also
invalid.

NeMo-RL performs this work outside graph capture, using its authoritative
Python `real_sequence_count`, physical endpoints, and fixed token capacity. It
validates shape, dtype, device, contiguity, ID bounds, and dummy ownership
before yielding the microbatch.

## Independent Graph Namespace

The O(T) MoE ownership tensor must not become an attention or Mamba input. Add
a dedicated namespace in `packed_seq_params.py`:

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

`split_moe_packed_seq_params_for_cuda_graph()` returns the two present Tensor
inputs plus static capacity. The matching merge surface is:

```python
def merge_moe_packed_seq_params_from_cuda_graph_kwargs(
    packed_seq_params: PackedSeqParams | None,
    kwargs: MutableMapping[str, object],
    static_metadata: Mapping[str, object] | None,
    prefix: str = MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    remove_from_kwargs: bool = True,
) -> PackedSeqParams | None
```

It enriches a newly rebuilt or shallow-copied `PackedSeqParams`; it never
mutates the producer-owned object.

The generic attention, Mamba, and MoE prefixes round-trip independently:

- attention-only capture excludes MoE ownership, while its eager MoE tail
  retains the original metadata;
- router, router+preprocess, whole-MoE, and whole-layer capture include the MoE
  namespace even when self-attention is not captured; and
- Mamba registers only its `seq_idx` ownership Tensor and ignores the MoE
  namespace.

TE attention defensively removes all three MoE fields before constructing its
own `PackedSeqParams` view.

The graph boundary accepts exactly the two declared Tensor keys under
`_moe_packed_seq_params_`. Any other key with that prefix raises before a
graph callable is invoked. Missing keys, changed presence, changed Tensor
signature, or changed static capacity are rejected by the same pre-launch
validation.

## Graph-Safe Router Algorithm

For:

- exact dynamic sample count `N`;
- static sample capacity `S_cap`;
- experts `E`;
- router top-k `K`;
- global valid route count `C[s,e]`; and
- local normalized score sum `P[s,e]`;

the established padded `seq_aux_loss` is:

```text
T_bar = sum(C) / (K * N)

L_seq = alpha * E / (N * K * T_bar^2)
        * sum(P * C)
```

The implementation allocates only static `[S_cap, E]` count and score buffers
and accumulates with `scatter_add_` using `seq_aux_loss_sample_ids`. Rows
`N:S_cap` stay zero. The count buffer is reduced over the existing TP/CP group.
The dynamic denominator is kept as a Tensor. Invalid dynamic `N` is reported
by the device assertion above, while safe Tensor values prevent NaN/Inf from
being produced before that assertion surfaces. A scheduler-only all-padding
sample has zero global routes, so it uses a device-side safe denominator and
selects an exact zero loss without evaluating the switch loss at zero:

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
total_num_tokens = total_routes / (topk * safe_num_samples_float)
safe_total_num_tokens = torch.where(
    has_valid_routes,
    total_num_tokens,
    torch.ones_like(total_num_tokens),
)
valid_token_count = local_counts.sum() / topk
```

After the switch helper returns, divide by `safe_num_samples_float`, then use
`torch.where(num_samples_is_valid & has_valid_routes, aux_loss,
torch.zeros_like(aux_loss))`. The all-padding forward loss and every attached
auxiliary gradient are exactly zero without host synchronization.

For valid `N`, the loss is divided by the numerically identical
`safe_num_samples_float` after the existing switch loss helper, matching
padded batch averaging. It must not introduce a per-sample `T_s**2`
normalization because that changes established loss and gradient semantics.

The same fixed-capacity implementation is used in eager and graph modes.
Dynamic derivation from `cu_seqlens`, dynamic `(N, E)` allocation, `.item()`,
`.tolist()`, and Python Tensor predicates are forbidden inside capture/replay.

## Propagation and Ownership

### MCore packed contract

- `pad_sequence_for_thd()` preserves all three ownership fields and never
  derives `N` from padded/repeated endpoints.
- Missing ownership is an error when training variable packed THD with
  `seq_aux_loss`; silently treating the pack as `bsz == 1` is forbidden.
- The existing equal-width `tokens_per_sample` path remains separate.

### Router and MoE layer

- `TransformerLayer`, `MoELayer.forward()`, `MoELayer.route()`, recompute
  closures, and partial router/preprocess callables forward IDs, count, and
  capacity explicitly.
- Existing structural padding, HybridEP zeroing, token-dropping order,
  expert-bias masking, dispatcher replay state, and shared-expert ownership
  remain unchanged.
- `mlp_chunks_for_training > 1` is not used for this variable packed path;
  independent chunk normalization would be incorrect.
- This implementation rejects `moe_router_fusion=True` for variable packed
  `seq_aux_loss`. Enabling it later requires a separate pinned-TE capability
  test for the Tensor denominator and flattened expert shape. There is no
  silent fallback to a different kernel.

### TE graph layers and banks

- Sample kwargs are derived from Task 7's canonical inner-leaf descriptors,
  never from `HybridStack` or an outer MTP owner.
- A graph-bank fingerprint freezes field presence, shape, dtype, device,
  layout, stride, and `S_cap`.
- ID and exact-count contents may vary between replays without recapture.
- Capture, activation, rollback, and reset install/restore MoE metadata with
  existing padding-mask and dispatcher state transactionally.
- Unknown or incompatible metadata raises before graph launch. Rank-local eager
  fallback is forbidden and never contributes to graph coverage.

### Hybrid, Mamba, PP/VP, and MTP

- All PP/VP stages observe the same exact `N`; each stage receives the
  router-local ID ordering required by its TP/SP rank.
- Mamba carries the shared `PackedSeqParams` through Hybrid execution but does
  not copy or register MoE ownership.
- MTP rolls token-aligned inputs within each physical sample. Sample IDs remain
  unchanged because ownership is constant over each physical sample segment.
- Ordered Hybrid MTP keeps Task 7's zero/mixed/multi-MoE-leaf fail-closed
  bounds.

## Validation and Failure Semantics

Before yielding a graph-training microbatch, NeMo-RL requires:

```text
qkv_format == "thd"
1 <= real_sequence_count <= seq_aux_loss_max_samples
seq_aux_loss_max_samples == thd_max_packed_sequences - 1
sample_ids.dtype == torch.int64
sample_ids.ndim == 1
sample_ids.numel() == T_router_capacity
sample_ids.is_contiguous()
sample_ids.device == input_ids.device
all real/alignment slots have 0 <= id < real_sequence_count
all appended dummy/unused slots have id == 0
num_samples.dtype == torch.int64
num_samples.shape == torch.Size([])
num_samples.device == input_ids.device
```

At the MCore boundary, missing fields, invalid static capacity, an incompatible
scope, an unknown MoE-prefixed key, or a changed replay signature raises before
launch. Dynamic `N == 0` or `N > S_cap` is forbidden and trips the captured
device assertion without a host scalar read. NeMo-RL also rejects either value
before yielding, so the device check protects direct MCore callers and replay
corruption. A scheduler-only all-padding call is represented as one logical
dummy sample with `N == 1`, IDs zero, and mask true.

## Correctness Gates

Every supported row compares identical weights, logical samples, and seeds
through:

1. padded eager `[L_max, N, H]` with a padding mask;
2. compact fixed-capacity packed eager `[T_capacity, 1, H]` with ownership;
3. the same packed input through TE graph after exactly three warmups, with at
   least two replays whose ID/count contents change but signatures do not.

The comparison includes valid-token outputs, router probabilities, top-k
indices, routing maps, per-sample expert counts, auxiliary loss, auxiliary
gradient scale, input gradients, router gradients, expert gradients, shared
expert/gate gradients, and zero contribution from all physical padding.

Required geometry covers logical lengths `[3, 5]`, `[1, 7, 2]`, one sample,
and `N == S_cap`; unequal/equal physical lengths; exact and tail-padded token
capacity; top-k 1/2; production score functions; TP2/SP, CP2, TP2+SP+CP2,
PP2/VP where supported; Nano/Super/Ultra Hybrid paths; and MTP depths 0/1/2.

Real CUDA capture, not a source-text assertion, is the host-synchronization
gate. The same graph must replay `N == 1`, an intermediate `N`, and
`N == S_cap` without capture error, recapture, signature drift, fallback, or
metric undercount. A separate subprocess deliberately replays `N == 0` and
`N > S_cap` and must observe the device assertion; isolation is required
because a device-side assertion poisons that CUDA context.

## Acceptance Boundary

Performance and convergence jobs may start only after:

1. padded, packed eager, and packed graph loss/gradient parity passes;
2. fixed signatures replay three distinct exact sample counts;
3. TP2+SP+CP2 matches the TP1/CP1 padded oracle;
4. attention-only, router, router+preprocess, and combined Hybrid scopes report
   exact nonzero replay counters with zero fallback;
5. Mamba never registers the MoE ownership tensors;
6. ordered Hybrid MTP and graph-bank rollback/reset regressions remain green;
7. invalid or missing static ownership and unknown MoE-prefixed keys fail
   before graph launch, while invalid dynamic `N` is rejected by the producer
   and by the captured MCore device assertion; and
8. the MCore commit is signed and DCO-compliant before Bridge and NeMo-RL
   submodule pins are committed.

After these gates, the existing experiment plan runs 20-step scope attribution
and 40/100-step accuracy/convergence comparisons with W&B project
`sna-cg-study`, checkpoints disabled, and exactly three warmup updates.
