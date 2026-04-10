# CUDA Graph Training Experiments

Testing partial CUDA graph capture for GRPO training on GB200 (OCI-HSG).

## Goal

Validate CUDA graph support for:
- **Attention-only models** (LLama3-8B, Qwen3-8B): `cuda_graph_scope=attn`
- **Mamba-hybrid models** (Nemotron Nano 8B): `cuda_graph_scope=[attn,mamba]`

## Experiment Plan

| Experiment | Model | Scope | Status |
|-----------|-------|-------|--------|
| [llama3_attn](llama3_attn/) | LLama3.1-8B-Instruct | attn | pending |
| [qwen3_attn](qwen3_attn/) | Qwen3-8B | attn | pending |
| [nemotron_nano_mamba](nemotron_nano_mamba/) | Nemotron-Nano-8B-v1 | attn | pending |
| [nemotron_nano_full](nemotron_nano_full/) | Nemotron-Nano-8B-v1 | [attn,mamba] | pending |

## Setup

### Requirements

- Megatron-LM with packed-seq CUDA graph support (`sj/cudagraph-packedseq-port` branch)
- Set `NRL_MEGATRON_LM_DIR` to the patched Megatron-LM checkout

### OCI-HSG Interactive Node

```bash
# On OCI-HSG login node:
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/HybridEP_test
bash launch_interactive.sh

# After allocation, attach to the job:
bash <jobid>-attach.sh
```

### Pull Latest Code Inside Container

```bash
cd /path/to/nemo-rl  # wherever the repo is mounted
git pull origin feature/cuda-graph-training
```

### Run an Experiment

```bash
# Inside the container on the interactive node:
cd /path/to/nemo-rl

# Attention-only (LLama3-8B), 1 node 4 GPU GB200:
python -m nemo_rl.algorithms.grpo \
    --config-path examples/configs/recipes/llm/performance \
    --config-name grpo-llama3.1-8b-instruct-1n4g-cg

# Attention+Mamba (Nemotron Nano), 1 node 4 GPU GB200:
python -m nemo_rl.algorithms.grpo \
    --config-path examples/configs/recipes/llm/performance \
    --config-name grpo-nemotron-nano-1n4g-cg
```

## Key Config Parameters

```yaml
megatron_cfg:
  cuda_graph_impl: transformer_engine   # enable TE-based CUDA graphs
  cuda_graph_scope: attn                # "attn", "mamba", or "[attn,mamba]"
  cuda_graph_warmup_steps: 3            # warmup before graph capture
  cuda_graph_packed_seq: true           # required for sequence packing
```

## Expected Outcomes

- Steps 1-3: Warmup (no graphs, regular execution)
- Step 4+: Graph captured and replayed
- Speedup from graph replay visible in step timing
- No numerical divergence vs baseline (check KL error)
