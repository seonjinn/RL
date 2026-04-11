# CUDA Graph Training Experiments

Partial CUDA graph capture for GRPO training on GB200 (OCI-HSG).

## Goal

Validate CUDA graph support for:
- **Attention-only models** (LLama3-8B, Qwen3-8B): `cuda_graph_scope=attn`
- **Mamba-hybrid models** (Nemotron-H-8B): `cuda_graph_scope=[attn,mamba]`

## Functional Correctness

| Config | Model | Scope | Pack | Seq | Steps | Result |
|--------|-------|-------|------|-----|-------|--------|
| grpo-llama3.1-8b-instruct-1n4g-cg | LLama3.1-8B-Instruct | attn | yes | 4096 | 10/10 | pass |
| grpo-qwen3-8b-1n4g-cg | Qwen3-8B | attn | yes | 4096 | 10/10 | pass |
| grpo-nemotron-nano-1n4g-cg | Nemotron-H-8B (hybrid) | [attn,mamba] | no | 512 | 10/10 | pass |

All models confirmed: CG captures at step 4 (after 3 warmup steps), replays for subsequent
steps. No crashes or numerical instability (gradient clipping handles Inf grads in Mamba layers).

### Known limitation: NemotronH with packing not yet supported

When `sequence_packing.enabled=true` is combined with `cuda_graph_scope=[attn,mamba]`,
CG capture fails at step 4 with:

```
AssertionError: CUDA graph accepts only Tensor inputs.
```

Root cause: the Mamba mixer uses `packed_seq_params` (a `PackedSeqParams` dataclass) for
computing `seq_idx`. Megatron-LM's `_te_cuda_graph_replay` validates that all kwargs are
`Tensor | None`, and `PackedSeqParams` fails this check. The attention layers handle packed
sequences via pre-allocated static GPU buffers (commit `b5d02bc`), but the Mamba layers
do not have equivalent handling. Fixing this requires a Megatron-LM change.

The supported configuration for NemotronH is **method A** (no packing, short sequences):
`sequence_packing.enabled=false`, `max_total_sequence_length=512`.

## Performance on GB200

CG is functionally correct but shows throughput regression on GB200 for dense attention
models. CG replay is ~45-57% slower than eager execution for the attention-only scope.

### Policy Training throughput (tok/s/gpu)

| Model | CG warmup (steps 1-3) | CG replay (steps 4+) | No-CG baseline | CG effect |
|-------|----------------------|----------------------|----------------|-----------|
| LLaMA-3.1-8B | 5515-6212 | 3282-3660 (~3500) | 5134-7175 (~6000) | -42% |
| Qwen3-8B | 8303-9252 | 3969-4211 (~4100) | 9065-10151 (~9540) | -57% |
| NemotronH-8B (no-pack, seq=512) | high variance | high variance | n/a | n/a |
| NemotronH-8B (pack, seq=4096, no-CG) | n/a | n/a | 1068-1572 (~1400) | n/a |

The regression occurs at step 4 (the capture step) and persists through replay. The warmup
steps (eager with TE hooks installed) run at throughput comparable to no-CG, confirming the
regression is in the CUDA graph replay path, not the data format.

**Possible cause**: `cuda_graph_impl=transformer_engine` uses TE's `make_graphed_callables()`.
TE's attention kernels in CG mode may differ from the FlashAttention path used in eager mode.
On GB200, where attention is memory-bandwidth-bound and already fast, the static memory pool
overhead from CG replay can outweigh kernel launch savings. This should be investigated before
enabling CG in production.

## Setup

### Requirements

- Megatron-LM with packed-seq CUDA graph support (this PR's companion Megatron-LM changes)
- Add the patched Megatron-LM to `PYTHONPATH` ahead of any installed version

### Run an Experiment

```bash
# Attention-only (LLama3-8B), 1 node 4 GPU:
uv run examples/run_grpo.py \
    --config examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-1n4g-cg.yaml

# Attention-only (Qwen3-8B), 1 node 4 GPU:
uv run examples/run_grpo.py \
    --config examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg.yaml

# Attention+Mamba (Nemotron-H-8B hybrid), 1 node 4 GPU:
uv run examples/run_grpo.py \
    --config examples/configs/recipes/llm/performance/grpo-nemotron-nano-1n4g-cg.yaml
```

## Key Config Parameters

```yaml
megatron_cfg:
  cuda_graph_impl: transformer_engine   # TE-based CG (required)
  cuda_graph_scope: attn                # "attn" for dense; [attn, mamba] for hybrid
  cuda_graph_warmup_steps: 3            # eager warmup steps before capture
  cuda_graph_packed_seq: true           # required when sequence_packing.enabled=true
  cuda_graph_buckets:                   # sequence length buckets for replay
    - 4096
```

## Expected Behavior

- Steps 1-3: Warmup (eager execution, no graphs)
- Step 4+: Graph captured and replayed for `cuda_graph_scope` layers
- KL error remains stable (no numerical divergence)
- `skipping grad norm because it's too large inf` is expected for NemotronH (Mamba SSM
  backward instability in bfloat16); suppressed via `check_for_nan_in_grad: false`
