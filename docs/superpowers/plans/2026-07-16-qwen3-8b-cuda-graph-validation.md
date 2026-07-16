# Qwen3-8B CUDA Graph Validation Implementation Plan

For agentic workers: execute the tasks inline in this session because the user explicitly requested the experiments.

**Goal:** Measure packed-sequence CUDA Graph performance on Qwen3-8B and compare a fresh 40-step Llama current-ATTN run against no-CG for training-quality impact.

**Architecture:** Keep the existing isolated baseline and PR #5672 worktrees. Extend the shared Ptyche launcher with a model selector, add PR #5672 Qwen recipes that enable THD graph inputs, and stage Qwen once in the shared HF cache. Submit independent jobs only after each source branch is committed, pushed, and fast-forwarded remotely.

**Tech Stack:** NeMo-RL GRPO, Megatron-LM, Transformer Engine CUDA graphs, vLLM rollout, Slurm/Pyxis on Ptyche.

## Global Constraints

- Use 1 node × 4 B200 GPUs, sequence packing enabled, 4096-token graph bucket, and the existing nightly NeMo-RL container.
- Use `cuda_graph_warmup_steps: 3` for every CUDA Graph recipe.
- Keep no-CG, current, and PR #5672 conditions on identical model/data/seed/container settings.
- Do not use Slurm dependencies; prefetch completion is checked before submissions.
- Compare steady-state phase time and throughput separately from rollout-length-dependent E2E throughput.
- Treat loss, reward, generation KL, and reward-distribution metrics as 40-step training-quality checks; they are not a standalone held-out benchmark accuracy measurement.

### Task 1: Define Qwen3-8B comparison recipes

**Files:**
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-pr5672-attn.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-8b-1n4g-cg-pr5672-attn-mlp.yaml`
- Modify: `experiments/cuda_graph/launch_llama8b_cg_comparison_ptyche.sh`

- [ ] Select `MODEL=qwen3` recipes for no-CG, current ATTN, current ATTN+MLP, PR #5672 ATTN, and PR #5672 ATTN+MLP.
- [ ] Set the PR #5672 recipes to `cuda_graph_packed_seq: true`, `cuda_graph_pr5672_thd: true`, `cuda_graph_max_packed_seqs: 64`, bucket 4096, and warmup 3.
- [ ] Validate all modified YAML with `ruby -ryaml -e 'ARGV.each { |f| YAML.load_file(f) }'` and validate shell syntax with `bash -n`.

### Task 2: Publish and synchronize source

**Files:**
- Modify: files from Task 1

- [ ] Commit only the Qwen recipes, launcher, and this plan with `git commit -s`.
- [ ] Push `experiment/pr5672-vs-pr5783-ptyche-runtime-20260716` to the `seonjinn` fork.
- [ ] Fast-forward the remote PR #5672 worktree from `origin/experiment/pr5672-vs-pr5783-ptyche-runtime-20260716` and verify its commit SHA.

### Task 3: Stage Qwen3-8B and submit runs

**Files:**
- Modify: `experiments/cuda_graph/prefetch_llama31_8b_ptyche.sh`

- [ ] Submit one Qwen3-8B cache prefetch job and verify the expected cache snapshot exists.
- [ ] Use `sbatch --test-only` before all GPU jobs.
- [ ] Submit independent Qwen3-8B 20-step jobs for no-CG, current ATTN, current ATTN+MLP, PR #5672 ATTN, and PR #5672 ATTN+MLP.
- [ ] Submit independent Llama 3.1 8B 40-step jobs for no-CG and current ATTN.

### Task 4: Monitor and analyze

**Files:**
- Read: `experiments/cuda_graph/logs/*`

- [ ] Monitor each submitted job for five minutes and triage only its final log segment if it fails.
- [ ] Report phase times and tok/s/GPU for E2E, generation, policy+reference logprob, and policy training.
- [ ] Compare 40-step Llama windows for loss, average reward, generation KL, reward distribution, and generation length; distinguish those training-quality proxies from held-out accuracy.
