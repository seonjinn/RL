# TE Packed Partial CUDA Graph Adapter Design

## Goal

Make the current NeMo-RL CUDA-graph branch safely support sequence-packed THD
attention through Transformer Engine (TE) partial CUDA graphs, while retaining
the branch's non-CUDA-graph changes.

The target production configuration is Qwen3-30B-A3B with sequence packing,
FP64 MoE routing left eager, and attention-only TE partial CUDA graphs.

## Scope and Non-Goals

In scope:

- Port the PR5672 packed-THD TE adapter contract into NeMo-RL.
- Pin the Megatron-LM submodule to the compatible PR5672-derived commit.
- Invalidate TE CUDA graphs whenever policy parameter storage is relocated.
- Add focused unit tests and reproducible Qwen30 benchmark recipes.

Out of scope:

- PR5783's `cuda_graph_impl=local` THD implementation.
- PR4359 integration.
- Capturing dropless all-to-all dispatch or MoE expert compute.
- Lowering `moe_router_dtype` from FP64 for a production result.
- Full-iteration CUDA graphs, generation CUDA graphs, and checkpoint saving.

## Fixed Compatibility Contract

The implementation must use:

```yaml
policy:
  sequence_packing:
    enabled: true
  megatron_cfg:
    cuda_graph_impl: transformer_engine
    cuda_graph_scope: attn
    cuda_graph_packed_seq: true
    cuda_graph_pr5672_thd: true
    cuda_graph_warmup_steps: 3
```

For Qwen3-30B-A3B, `moe_router_dtype: fp64` remains unchanged. Router,
all-to-all dispatch, and expert compute remain eager. `cuda_graph_max_packed_seqs`
is set per workload after measuring the global maximum of
`len(cu_seqlens_padded) - 1` before graph-facing padding.

## Architecture

```text
Real packed batch
  ├─ data.py: original cu_seqlens and cu_seqlens_padded
  │             └─ graph-facing PackedSeqParams padded to Nmax + 1 entries
  ├─ train.py: model receives graph-facing PackedSeqParams
  │             └─ loss receives original packed boundaries
  └─ megatron_policy_worker.py
                ├─ supplies a fixed-shape sample PackedSeqParams to TE capture
                └─ invalidates all graph state before CPU parameter offload

Megatron-LM TECudaGraphHelper
  └─ flattens dynamic PackedSeqParams tensor fields into TE graph kwargs,
     preserves static metadata, then rebuilds PackedSeqParams at replay.
```

`cuda_graph_max_packed_seqs=Nmax` fixes only the graph-facing tensor shape to
`[Nmax + 1]`. Runtime values still express the real document boundaries.
Trailing entries repeat the packed endpoint, representing zero-length trailing
sequences. The original unpadded boundaries must never be replaced in loss
calculation.

## Components

### Megatron-LM submodule

Update `3rdparty/Megatron-LM-workspace/Megatron-LM` from
`5fac56ce646e0acea6b5f7cc483db275db99e4ba` to the tested PR5672-compatible
commit `bed605f292f926090f5f43ba5e30fb024c2306dc`.

The target helper accepts `sample_packed_seq_params`, adds its dynamic tensor
fields to TE capture kwargs, validates static metadata at replay, and rebuilds
the `PackedSeqParams` object inside the capture callable.

### `nemo_rl/models/megatron/data.py`

When sequence packing and `cuda_graph_pr5672_thd` are enabled:

- Compute `cu_seqlens_pad_to_entries = cuda_graph_max_packed_seqs + 1`.
- Pad only `PackedSeqParams.cu_seqlens_{q,kv}{,_padded}` by repeating the
  endpoint.
- Preserve the original `cu_seqlens` and `cu_seqlens_padded` on the processed
  microbatch for loss computation.
- Raise a clear assertion when a packed microbatch exceeds the configured
  maximum rather than silently replaying an incorrectly shaped graph.

### `nemo_rl/models/megatron/train.py`

Pass the original packed boundaries through `model_forward` as
`loss_cu_seqlens` and `loss_cu_seqlens_padded`. The sequence-packing loss
wrapper uses those values rather than graph-facing endpoint-repeated metadata.

### `nemo_rl/models/policy/workers/megatron_policy_worker.py`

- Add `_make_cuda_graph_sample_packed_seq_params()`. Its sample has the same
  shape, dtype, device, and static metadata as the real input, with
  `[0, T, T, ...]` cumulative lengths for the capture bucket length `T`.
- Peek without consuming the training iterator and supply this sample to each
  `TECudaGraphHelper` construction when `cuda_graph_pr5672_thd` is enabled.
- Add `_invalidate_cuda_graphs_after_parameter_move()` and call it immediately
  before `move_model(self.model, "cpu")` in `offload_after_refit()`.
- Clear module graph lists, single- and multi-bucket helpers, saved graph
  references, active bucket state, and captured sequence length. Set the
  capture counter to `cuda_graph_warmup_steps`, so the next eligible training
  call captures a graph with the newly allocated GPU parameter storage.

The initial run still has exactly three eager warmup steps. Immediate capture
after a refit is required because every prior graph contains invalid parameter
pointers after CPU offload and GPU reload.

## Correctness and Failure Handling

- Packing plus an attention graph must require `cuda_graph_packed_seq=true`.
- The PR5672 path requires `cuda_graph_impl=transformer_engine`; reject an
  incompatible local implementation at config validation.
- Do not add `padding_mask` to an attention-only THD graph signature unless
  NeMo-RL supplies it at every replay. The adapter's packed boundaries are the
  attention contract for this path.
- A change to static PackedSeqParams metadata, a missing dynamic field, an
  oversized packed batch, or parameter relocation requires eager execution or
  graph recapture; it must not reuse a captured graph.
- FP64 `moe_router` graph capture is excluded until TE supports its FP64 output
  contract. Do not change router precision merely to enable a graph.

## Tests and Validation

### Unit tests

Add focused tests covering:

1. A real `cu_seqlens` tensor is retained for loss while graph-facing PSP is
   endpoint-padded to `Nmax + 1`.
2. A packed batch with `Nmax + 1` sequences fails with the configuration error.
3. TE sample PSP has matching shape and static metadata.
4. Parameter relocation clears every single- and multi-bucket graph reference
   and schedules capture for the next eligible training call.
5. Attention-only replay has no required `padding_mask` graph kwarg.

### Ptyche smoke test

Run a Qwen3-30B-A3B, 4-node × 4-GPU, packed, FP64-router job with checkpoint
saving disabled. Use `cuda_graph_warmup_steps=3` and run at least eight steps
so capture, refit/offload, reload, and replay occur twice. Use
`CUDA_LAUNCH_BLOCKING=1` only for an initial failure diagnosis; the passing
performance run must not set it.

### Performance matrix

Use identical seed, data, 4n4g topology, token budget, sequence-packing
configuration, bucket, and validation cadence for each 20-step run:

1. No CUDA graph, FP64 router.
2. PR5672 TE `attn`, FP64 router.
3. TE `moe_router,moe_preprocess`, FP32 router, diagnostic only.
4. TE `attn,moe_router,moe_preprocess`, FP32 router, diagnostic only.

Report total step time, E2E TPS/GPU, policy time/TPS, logprob time/TPS, and
generation time/TPS over steps 4–19, excluding validation steps. Do not compare
the FP32-router rows to FP64 production accuracy or present them as production
speedups.

### Accuracy sign-off

Compare only No-CG FP64 router with PR5672 TE `attn` FP64 router. Use three
identical seeds, at least 40 training steps per seed, and at least 1,024 fixed
validation samples. Record validation accuracy, reward, GenKL, policy loss,
ratio/clip diagnostics, and NaN or invalid-output counts. Approve the recipe
only if no persistent accuracy or stability regression appears across seeds.

## Expected Outcome

The immediate target is the previously observed Qwen30 attention-only result:
about 7% higher E2E throughput and 20% higher policy-training throughput than
the no-CG baseline, without a sustained accuracy regression. Further E2E gains
require separate work on FP64 MoE router capture or generation/logprob, not a
different packed-THD adapter.
