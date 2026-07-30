# NCCL-Reshard BF16-to-MXFP8 Refit Design

## Goal

Support BF16 Megatron training storage with MXFP8 vLLM rollout storage over
NCCL-Reshard, while preserving the existing MXFP8 quantization and backend
layout semantics and reducing the weight-transfer payload.

## Scope

The first implementation supports the existing NCCL-Reshard bulk FFN path:

- Dense `gate_proj`, `up_proj`, and `down_proj`.
- Grouped MoE `gate_proj`, `up_proj`, and `down_proj`.
- Megatron policy workers and vLLM generation workers.
- BF16 trainer storage with `precision=fp8`, `is_mx=true`, and
  `refit_prequantize=true`.

Attention projections, shared experts, embeddings, norms, and other parameters
continue to use the existing misc packed-broadcast path. NVFP4 is not part of
this change.

## Architecture

NCCL-Reshard continues to transfer one logical DTensor per call. The refit
layer expands each eligible BF16 weight into two ordinary, dtype-matched
transfers:

```text
BF16 local trainer shard
  -> MXFP8 quantization on the trainer GPU
  -> E4M3 value tensor       [*weight_shape]
  -> E8M0 scale tensor       [*weight_shape[:-1], K / 32]
  -> NCCL-Reshard value transfer
  -> NCCL-Reshard scale transfer
  -> canonical vLLM checkpoint-layout destinations
  -> existing vLLM post-load layout processing
```

The low-level `xferdtensor` API is unchanged. Value and scale tensors each
satisfy its same-shape and same-dtype source/destination contract.

## Metadata Handshake

During synchronizer initialization:

1. The policy publishes the original BF16 HF metadata.
2. Generation reports the exact MXFP8-eligible parameter names.
3. The policy enables trainer-side prequantization for those names.
4. NCCL-Reshard metadata describes the E4M3 value and E8M0 scale children for
   every eligible bulk parameter.
5. Generation builds and validates both destinations before the first refit.

Each transformed parameter carries:

- `refit_transform="mxfp8"`.
- Value global shape, dtype, source placements, and destination placements.
- Scale global shape, dtype, source placements, and destination placements.
- The same parent name, PP stage, and refit ordering for both transfers.

## Placement Rules

MXFP8 uses one E8M0 scale byte per 32 values along the last dimension.
Quantization before reshard is equivalent to quantization after assembly only
when every source and destination shard boundary on that dimension is aligned
to 32 values.

Initialization rejects a transformed parameter when:

- Its global last dimension is not divisible by 32.
- A source or destination placement shards the last dimension into local
  intervals that are not each divisible by 32.
- The scale shape or scale placement is inconsistent with the parent value
  tensor.

The scale placement is derived from the parent value placement. It is not
inferred from the synthetic `*_scale_from_checkpoint` name.

## Trainer Path

The trainer's cached HF-to-local mapping retains live BF16 Megatron views.
During each refit:

1. Grouped experts are stacked once when required.
2. The resulting local tensor is quantized once with
   `mxfp8_e4m3_quantize_for_refit`.
3. The value and scale tensors are transferred on the same NCCL stream.
4. Temporary tensors remain stream-safe until both transfers have been
   enqueued.

The implementation does not retain a full quantized model copy. CUDA allocator
reuse is bounded by the existing per-parameter transfer loop.

## Generation Path

Generation resolves both the canonical value target and its
`*_scale_from_checkpoint` target.

- Direct parameters receive value and scale into their corresponding targets.
- Merged gate/up and W13 parameters receive into matching temporary slices and
  copy both slices back after the paired transfers.
- Missing or shape-incompatible scale destinations fail during setup.
- Existing `process_weights_after_loading` performs padding, W13/W31
  conversion, scale swizzle, and backend-specific MoE shuffle once after all
  bulk and misc transfers finish.

Generation remains unavailable while the synchronizer is stale. Both child
transfers and post-load processing must finish before the refit reports
success.

## Correctness

The implementation must preserve:

- Bitwise E4M3 value parity with the existing receiver quantization path.
- Bitwise E8M0 scale parity, including the zero-scale clamp.
- Existing gate/up, W13, W2, TP, EP, and PP placement semantics.
- Repeated-refit correctness without stale scale tensors.
- Existing BF16-to-BF16 and matching blockwise-FP8 NCCL-Reshard behavior.

Unsupported layouts fail during initialization rather than silently falling
back to a different precision or corrupting weights.

## Performance Evaluation

Use the same model, recipe, node allocation, random seed, and measured step
window for every A/B pair. Report:

- Transfer/update time within refit.
- Total refit time.
- Generation, logprob, policy-training, and E2E step time.
- Tokens/s/GPU.
- Mean rollout reward and available logprob parity metrics.

Two baselines are retained:

1. Existing NCCL-Reshard BF16 storage transfer, to quantify transport and
   quantization overhead relative to the current NCCL implementation.
2. Existing MXFP8 legacy refit path, to quantify the end-to-end benefit of
   adding transform-aware NCCL-Reshard.

Primary steady-state summaries exclude initialization and the first two steps.
A 5-step smoke validates execution; a 20-step run supplies the reported
steady-state result.

The theoretical bulk payload changes from 2 bytes per BF16 value to
1 byte per E4M3 value plus 1 scale byte per 32 values:

```text
1 + 1 / 32 = 1.03125 bytes/value
```

The transformed payload is 51.6% of BF16, a 48.4% reduction before protocol
overhead.

