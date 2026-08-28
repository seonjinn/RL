# DeepSeek Custom PP NCCL Reshard Design

## Problem

DeepSeek-V3 uses a custom Megatron pipeline layout. NeMo-RL's NCCL reshard
refit path currently rejects every `pipeline_model_parallel_layout`, so the
48-node DeepSeek experiment falls back to packed broadcast. Each TP16
generation receiver then observes the full `1.342 TB` checkpoint stream even
though its persistent local shard is approximately `83.9 GB`.

The existing NCCL reshard implementation already supports PP stage-specific
communicators and accepts a `layer_to_pp_stage` mapping. The missing capability
is deriving that mapping from MCore's runtime custom layout without duplicating
or parsing MCore's layout syntax.

## Goal

Allow a non-interleaved MCore `PipelineParallelLayerLayout` to provide the
decoder-layer-to-PP-stage mapping used by NeMo-RL's existing NCCL reshard
metadata builder.

## Scope

- Support custom pipeline layouts with exactly one virtual pipeline stage.
- Use MCore's runtime layout API to obtain global decoder layer IDs.
- Preserve the current standard first/middle/last layer-distribution path.
- Preserve packed broadcast for embeddings, norms, attention, router, scales,
  and other parameters outside the existing FFN reshard whitelist.
- Fail before communication when the runtime layout is incomplete, duplicated,
  out of range, or inconsistent with the configured PP size.

## Non-Goals

- Virtual or interleaved pipeline parallelism.
- Generation-side pipeline parallelism greater than one.
- Changing which parameters use the bulk NCCL reshard path.
- Implementing FlashInfer TRTLLM destination-layout conversion. That remains a
  separate follow-up after the stage mapping and transport are validated.
- Changing model weights, numerical precision, training topology, or rollout
  behavior.

## Design

Add a pure helper in `nemo_rl/weight_sync/nccl_reshard_utils.py`:

```python
def build_layer_to_pp_stage_from_custom_layout(
    layout: PipelineLayerLayout,
    *,
    pp_size: int,
    layer_prefix: str,
    num_layers: int,
) -> dict[str, int]:
    ...
```

`PipelineLayerLayout` is a local `Protocol` containing only the runtime fields
and method NeMo-RL consumes. This avoids a hard import of MCore's concrete
layout class into the backend-agnostic utility module.

For each PP rank, the helper calls
`layout.get_layer_id_list(vp_stage=0, pp_rank=rank)`. It validates that:

- `layout.pipeline_model_parallel_size == pp_size`;
- `layout.virtual_pipeline_model_parallel_size == 1`;
- every layer ID is an integer in `[0, num_layers)`;
- no layer ID appears on two stages; and
- the union of IDs is exactly `range(num_layers)`.

It then returns keys in the same exported-HF namespace already consumed by the
reshard builder, such as `model.layers.17 -> 2`.

`MegatronPolicyWorker._build_layer_to_pp_stage` selects this helper when the
runtime model config has a custom layout. Otherwise it retains the existing
standard uneven-PP calculation unchanged.

`check_nccl_reshard_refit_support` no longer rejects the mere presence of a
custom layout. It continues to reject configured VPP greater than one. The
runtime helper performs the authoritative validation after MCore has converted
the user input into its validated layout object.

For the current DeepSeek PP8 layout, the resulting decoder ownership is:

| PP stage | Decoder layer IDs |
| ---: | --- |
| 0 | 0-7 |
| 1 | 8-15 |
| 2 | 16-23 |
| 3 | 24-31 |
| 4 | 32-39 |
| 5 | 40-47 |
| 6 | 48-55 |
| 7 | 56-60 |

## Error Handling

All layout inconsistencies raise `ValueError` with the invalid PP size, VPP
size, duplicated IDs, out-of-range IDs, or missing IDs. These errors occur
during refit metadata preparation, before communicator initialization or any
weight transfer.

## Verification

1. Pure CPU unit tests cover the PP8 DeepSeek mapping and every rejection
   condition.
2. Existing standard-layout and reshard metadata tests must remain unchanged
   and pass.
3. A Linux container test runs the focused unit suite because the repository
   lockfile does not support macOS.
4. A 48-node, four-step Triton canary enables `refit_transport=nccl_reshard`
   and verifies initialization, complete parameter coverage, finite training
   metrics, and successful repeated refits.
5. FlashInfer TRTLLM is not enabled by this first change. Its destination
   conversion is validated separately so transport correctness is not obscured
   by backend-layout handling.

## Success Criteria

- The exact DeepSeek PP8 layer ownership above is produced without parsing a
  layout string.
- Standard non-custom layouts are behaviorally unchanged.
- Unsupported VPP and malformed layouts fail before communication.
- The Triton DeepSeek canary completes repeated NCCL reshard refits with no
  missing parameter mapping or numerical non-finiteness.
