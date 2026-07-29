# Qwen3-30B-A3B Partial CUDA Graph Performance Design

## Objective

Measure 20-step NeMo-RL GRPO performance and correctness for Qwen3-30B-A3B with Transformer Engine partial CUDA Graphs on Ptyche GB200.

## Workload

- Config: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`
- Model and tokenizer: the same verified local `Qwen/Qwen3-30B-A3B` snapshot
- Policy topology: 4 nodes × 4 GPUs, TP1, PP1, CP1, EP16
- Generation topology: 4 additional nodes × 4 GPUs, non-colocated vLLM
- Total allocation per job: 8 nodes × 4 GPUs
- Sequence length: 4096
- Training/logprob packing capacities: 4096/8192 tokens
- Steps: 20
- CUDA Graph warmups: 3 optimizer steps
- Checkpointing: disabled
- Partition: `batch`

Non-colocated generation is required because the Transformer Engine training graph keeps captured parameter addresses stable, while colocated refit offloads policy parameters.

## Matrix

Each row is an independent job with no Slurm dependency:

1. No-CG baseline
2. `attn`
3. `moe_router + moe_preprocess`
4. `attn + moe_router + moe_preprocess`

Qwen3-30B-A3B has no Mamba layers. Dynamic expert execution is not included in the initial partial-graph matrix because the official GB200 Megatron-Bridge recipe recommends attention, router, and MoE preprocessing scopes.

## Packing Contract

Set `policy.megatron_cfg.cuda_graph_max_packed_seqs=16` for every row, including baseline. NeMo-RL uses this value to cap sequences per packed microbatch, making baseline and graph jobs process the same packing geometry. Graph jobs additionally use fixed 4096-token training buffers.

## Launcher Layout

- One common launcher owns config composition, validation, environment propagation, and `sbatch`.
- Four small scope scripts pass only the scope name and graph settings.
- One matrix driver submits the four scripts independently.
- Secrets are read from the environment and never written to scripts or logs.
- Each job writes to a unique `exp_logs/qwen3-30ba3b-*` directory.

## Validation

Before submission:

- Static launcher tests verify the exact config, 20-step default, disabled checkpoints, non-colocated 4+4 node split, three warmups, packing bound parity, and scope list.
- `bash -n` validates every launcher.
- Each scope runs with `TEST_ONLY=1` before any real `sbatch`.
- Source is committed and pushed before the remote worktree pulls it.

After submission:

- Monitor all jobs for at least five minutes.
- Fail early on traceback, CUDA error, illegal memory access, NCCL failure, or packing-bound failure.
- Record canonical E2E, generation, policy-training, and policy/reference-logprob times and tokens/s/GPU.
- Record `token_mult_prob_error`, sequence error, generation/policy KL, JS divergence, probability ratios, reward, loss, and gradient norm.

## Interpretation

Performance aggregation excludes cold initialization and capture. The target comparison window is graph replay steps after warmup/capture, using the same included steps for every surviving run. Generation-dominated E2E deltas are not attributed to training CUDA Graphs unless generated-token work is paired or normalized by the logged throughput metrics.
