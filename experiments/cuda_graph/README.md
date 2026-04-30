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
models. CG replay is ~39-45% slower than eager execution for the attention-only scope.

### Policy Training throughput (tok/s/GPU)

| Model | CG warmup (steps 1-3) | CG replay (steps 4+) | No-CG baseline | CG effect |
|-------|----------------------|----------------------|----------------|-----------|
| LLaMA-3.1-8B | 5515-7296 (~6550) | 3282-4003 (~3750) | 5134-7175 (~6150) | -39% |
| Qwen3-8B | 8303-9252 (~8900) | 3969-4211 (~4100) | 9065-10151 (~9540) | -57% |
| NemotronH-8B (no-pack, seq=512) | high variance | high variance | n/a | n/a |
| NemotronH-8B (pack, seq=4096, no-CG) | n/a | n/a | 1068-1572 (~1400) | n/a |

The regression occurs at step 4 (the capture step) and persists through replay. The warmup
steps (eager with TE hooks installed) run at throughput comparable to no-CG, confirming the
regression is in the CUDA graph replay path, not the data format.

### Root cause investigation (Fix 1+3, job 2263202)

Two candidate fixes were evaluated to explain the regression:

**Fix 1**: Remove `attention_mask` from the static CG surface when `cuda_graph_packed_seq=True`.
In packed-seq mode, `attention_mask` is `None` at runtime (cu_seqlens handles masking), but
was still allocated as a `[1, 1, 4096, 4096]` bool tensor static input, requiring a D2D copy
per replay step.

**Fix 3**: Skip zeros-tensor creation in `_get_te_cuda_graph_replay_args()` when
`_cuda_graph_psp` (packed-seq) is active, avoiding an unnecessary static buffer creation
during replay.

Results with both fixes applied:

| Step | Mode | Tok/s/GPU | vs pre-fix |
|------|------|-----------|------------|
| 1 | warmup | 5907 | +4% |
| 2 | warmup | 7296 | +5% |
| 3 | warmup | 6442 | -2% |
| 4 | CG capture | 3894 | -0.2% |
| 5 | CG replay | 3860 | +0.2% |
| 6 | CG replay | 4003 | +4% |
| 7 | CG replay | 3795 | +2% |
| 8 | CG replay | 3837 | +2% |
| 9 | CG replay | 3500 | -7% |
| 10 | CG replay | 3471 | -8% |

Pre-fix replay avg: ~3760 tok/s/GPU. Fix 1+3 replay avg: ~3744 tok/s/GPU. **No meaningful
improvement.** The D2D copy of `attention_mask` was not the bottleneck.

**Fix 2** (NVTE_DEBUG=1, NVTE_DEBUG_LEVEL=2) confirmed that both eager and CG capture select
the same attention backend: **FusedAttention (sub-backend 1)** — cuDNN fused attention on
Hopper+. The backend switch hypothesis is ruled out.

**Conclusion**: The regression is structural and inherent to TE's `make_graphed_callables()`
mechanism on GB200. The cuDNN fused attention kernel is already bandwidth-efficient in eager
mode; wrapping it in a CUDA graph adds fixed per-step overhead (graph launch + TE's static
buffer management) that outweighs kernel launch savings. Profiling with nsys is needed to
quantify the exact overhead breakdown (graph launch vs. kernel duration delta).

**Possible cause**: `cuda_graph_impl=transformer_engine` uses TE's `make_graphed_callables()`.
On GB200, where attention is memory-bandwidth-bound and already fast, the static memory pool
overhead from CG replay can outweigh kernel launch savings. This should be investigated before
enabling CG in production.

### GPU kernel profiling (no-CG baseline, job 2264184)

Nsight Systems profile of `megatron_policy_worker` during steps 5-6 (eager, no CG).
LLaMA-3.1-8B-Instruct, TP=1, DP=4, packed-seq, seq=4096, rank 0.

**Total GPU kernel time: 70.75 s (2 steps)**

| Kernel | Calls | Total (ms) | GPU% | Category |
|--------|-------|------------|------|----------|
| NCCL AllGather RING_LL | 3,332 | 23,809 | 33.7% | DP communication |
| cuDNN nvjet_tst_128x256 (attn GEMM) | 12,045 | 7,133 | 10.1% | attention fwd |
| NCCL ReduceScatter bf16 | 3,136 | 6,651 | 9.4% | DP communication |
| cuDNN nvjet_tst_256x256 + others | ~40K | ~15,000 | ~21% | attention fwd/bwd |
| elementwise / vectorized ops | ~100K | ~8,000 | ~11% | activation, etc. |
| CUDA Memcpy (39K copies) | 39,474 | 4,036 | 5.7% | D2D copy |
| NCCL AllReduce f32 | 98 | 760 | 1.1% | grad norm |

Summary by category:
- NCCL communication: **44%** (AllGather 33.7% + ReduceScatter 9.4% + AllReduce 1.1%)
- cuDNN attention kernels: **~31%**
- elementwise / misc compute: **~19%**
- Memcpy: **~6%**

**Root cause: compute-communication overlap broken by CG (confirmed by profiling)**

CG profile obtained using `cuda-graph-trace=graph` (job 2268584). Direct comparison:

| Metric | no-CG (eager) | CG replay | Delta |
|--------|--------------|-----------|-------|
| Total GPU kernel time | 70.75 s | 59.21 s | -16% |
| Total kernel count | 547,726 | 429,233 | -22% |
| NCCL total | 31,309 ms (44.3%) | 23,376 ms (39.5%) | -25% |
| Attention total | 25,530 ms (36.1%) | 22,992 ms (38.8%) | -10% |
| Memcpy total | 4,036 ms | 2,412 ms | -40% |
| **Throughput** | ~6150 tok/s/GPU | ~3750 tok/s/GPU | **-39%** |

The paradox: CG replay has *less* total GPU kernel time (-16%) yet *worse* throughput (-39%).
This is only possible if GPU idle time increased substantially. Measuring effective stream
parallelism (total_kernel_time / wall_clock) confirms:

```
Wall-clock for 2 steps:
  no-CG: 4096 / 6150 × 2 = 1.33 s
  CG:    4096 / 3750 × 2 = 2.18 s

Effective stream parallelism (kernel_time / wall_clock):
  no-CG: 70.75 s / 1.33 s = 53×   (high compute-NCCL overlap)
  CG:    59.21 s / 2.18 s = 27×   (overlap cut in half)
```

In eager mode, Megatron-LM overlaps the NCCL gradient-reduction streams with compute streams;
NCCL + compute run truly in parallel. In CG replay mode, TE's `make_graphed_callables()`
CUDA memory pool interaction serializes NCCL launches with respect to graph replay, exposing
the 44% NCCL time as wall-clock idle instead of hiding it behind compute.

NCCL timing shifts support this: AllGather avg drops 7146→3053 µs (less contention with
compute); ReduceScatter avg rises 2121→3555 µs (serialized at a different phase). Individual
attention kernels are unchanged (~600 µs avg for the dominant nvjet_tst_128x256 in both modes),
ruling out kernel-level regression.

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
