# CUDA Graph Training Experiments

Partial CUDA graph capture for GRPO training on GB200 (OCI-HSG).

## Goal

Validate CUDA graph support for:
- **Attention-only models** (LLama3-8B, Qwen3-8B): `cuda_graph_scope=attn`
- **Mamba-hybrid models** (Nemotron-H-8B): `cuda_graph_scope=[attn,mamba]`

## Results

| Config | Model | Scope | Steps | Result |
|--------|-------|-------|-------|--------|
| grpo-llama3.1-8b-instruct-1n4g-cg | LLama3.1-8B-Instruct | attn | 10/10 | pass |
| grpo-qwen3-8b-1n4g-cg | Qwen3-8B | attn | 10/10 | pass |
| grpo-nemotron-nano-1n4g-cg | Nemotron-H-8B (hybrid) | [attn,mamba] | 10/10 | pending |

LLaMA and Qwen3 confirmed: CG captures at step 4 (after 3 warmup steps), replays for subsequent steps. No numerical divergence vs no-CG baseline.

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
  cuda_graph_packed_seq: true           # required for sequence packing
  cuda_graph_buckets:                   # sequence length buckets for replay
    - 4096
```

## Expected Behavior

- Steps 1-3: Warmup (eager execution, no graphs)
- Step 4+: Graph captured and replayed for `cuda_graph_scope` layers
- Step timing decreases after capture (visible in `timing/train/` metrics)
- KL error remains stable (no numerical divergence)
