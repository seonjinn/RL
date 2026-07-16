# Qwen3-30B-A3B MoE CUDA Graph Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen3-30B-A3B CUDA Graph workload honor the configured training scope, then compare the current implementation, Megatron-LM PR #5672, and PR #4359 with performance and validation-quality checks.

**Architecture:** Preserve `args.cuda_graph_modules` when Megatron RL exits rollout inference instead of replacing the user-selected scope for all MoE models. Keep MoE layers in partial-capture mode, which permits router and preprocess capture while expert dispatch remains eager. Use branch-local no-CG baselines so different Megatron revisions are not conflated with CUDA Graph effects.

**Tech Stack:** NeMo-RL GRPO, Megatron-LM RL integration, Transformer Engine CUDA Graphs, Qwen/Qwen3-30B-A3B, Slurm/Pyxis on Ptyche.

## Global Constraints

- Use Qwen3-30B-A3B on 4 nodes × 4 B200 GPUs, EP=16, TP=PP=1, sequence packing enabled, and the existing container.
- Set `cuda_graph_warmup_steps: 3`, `cuda_graph_buckets: [4096]`, and `cuda_graph_max_packed_seqs: 64` for all packed CUDA Graph conditions.
- Use `moe_router` together with `moe_preprocess`; do not use `mlp` because Qwen3-30B-A3B has only MoE layers.
- Do not use the full `moe` scope: it requires drop-and-pad capacity settings that alter routing and invalidate a direct accuracy comparison.
- Keep current PR#5783, PR#5672, and PR#4359 in independent worktrees, each with its own no-CG baseline.
- Run 20-step performance matrices first, then no-CG versus the fastest correct scope for 40-step accuracy validation in each implementation.

### Task 1: Preserve the requested MoE training scope

**Files:**
- Modify: `3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/rl/rl_utils.py`
- Modify: `3rdparty/Megatron-LM-workspace/Megatron-LM/tests/unit_tests/rl/test_rl_utils.py`

- [ ] Write a failing unit test where an MoE model requests `[moe_router, moe_preprocess]` and verify that the list is restored after `megatron_rl_inference_mode` exits.
- [ ] Replace the unconditional MoE scope override with a copy of `args.cuda_graph_modules`; retain `transition_moe_cudagraphs(..., "partial")` so expert dispatch stays eager.
- [ ] Run the focused unit tests and the existing non-MoE restoration test.
- [ ] Commit the MCore patch to a dedicated seonjinn fork branch.

### Task 2: Create branch-local Qwen3-30B-A3B recipes and launch validation

**Files:**
- Create: current/PR#5672/PR#4359 branch-local `grpo-qwen3-30ba3b-4n4g-*-w3.yaml` recipes
- Modify: Ptyche launcher and model-prefetch helper

- [ ] Create no-CG, ATTN, router+preprocess, and ATTN+router+preprocess variants with unique result paths.
- [ ] Resolve defaults and verify the final configuration has the requested module list, packed-sequence invariants, EP=16, and warmup=3.
- [ ] Stage Qwen/Qwen3-30B-A3B into a branch-isolated Megatron conversion cache.
- [ ] Submit each 20-step matrix after `sbatch --test-only`, commit/push, and remote fast-forward.

### Task 3: Compare quality and performance across implementations

**Files:**
- Read: Slurm driver logs and validation outputs

- [ ] Select the fastest correct condition within each implementation using a common steady-state window excluding validation steps.
- [ ] Submit a 40-step no-CG versus selected-condition pair per implementation.
- [ ] Compare validation accuracy at steps 10, 20, 30, and 40, plus loss, reward, generation KL, response length, and reward distributions.
- [ ] Report E2E, generation, logprob, and policy-training time and throughput with sample count, step window, and implementation SHA.
