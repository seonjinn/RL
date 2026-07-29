# Qwen3-30B-A3B Partial CUDA Graph Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit and monitor a reproducible four-row, 20-step Qwen3-30B-A3B partial CUDA Graph performance matrix on Ptyche.

**Architecture:** A common Bash launcher composes the official NeMo-RL performance recipe with the required non-colocated topology and graph settings. Thin scope wrappers make each experiment independently reusable, while a matrix driver submits all rows without dependencies.

**Tech Stack:** Bash, NeMo-RL, Hydra/OmegaConf, Ray, Slurm, Transformer Engine, TensorBoard/W&B.

## Global Constraints

- Use the official `grpo-qwen3-30ba3b-4n4g.yaml` performance config.
- Preserve TP1/PP1/CP1/EP16 and allocate four policy plus four generation nodes.
- Run 20 steps with three CUDA Graph warmups and checkpointing disabled.
- Use `cuda_graph_max_packed_seqs=16` for baseline and graph rows.
- Submit to Ptyche `batch` only after commit, push, remote pull, and successful `--test-only`.
- Do not write credentials into files or command output.

---

### Task 1: Launcher Contract Tests

**Files:**
- Create: `tests/unit/experiments/test_qwen3_30ba3b_cg_launchers.py`

**Interfaces:**
- Consumes: the intended launcher paths and static shell contents.
- Produces: a failing contract that requires four valid scopes and common performance settings.

- [ ] **Step 1: Write tests for config, topology, packing parity, graph scope, and submission independence**
- [ ] **Step 2: Run the test and verify it fails because the Qwen launcher directory is absent**
- [ ] **Step 3: Keep the failing output as the RED evidence for Task 2**

### Task 2: Reusable Scope Launchers

**Files:**
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/profiles/ptyche.env`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/run_scope.sh`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/scopes/00_nocg.sh`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/scopes/01_attn.sh`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/scopes/02_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/scopes/03_attn_moe_router_preprocess.sh`
- Create: `experiments/cuda_graph/qwen3_30ba3b_4n4g/submit_performance_matrix.sh`

**Interfaces:**
- Consumes: `CLUSTER=ptyche`, the verified model snapshot, container profile, and optional `TEST_ONLY`.
- Produces: one `sbatch` invocation per wrapper and four independent invocations from the matrix driver.

- [ ] **Step 1: Implement the minimal common launcher and four wrappers**
- [ ] **Step 2: Run the focused unit test and verify it passes**
- [ ] **Step 3: Run `bash -n` on every new shell script**
- [ ] **Step 4: Run each wrapper with `TEST_ONLY=1` against a local fake `sbatch` to verify command composition without submission**

### Task 3: Source Publication

**Files:**
- Modify: `session/20260729_010733/session_state.md`
- Modify: `session/20260729_010733/timeline.md`
- Modify: `session/20260729_010733/files.md`
- Modify: `session/20260729_010733/handoff.md`

**Interfaces:**
- Consumes: verified launcher changes.
- Produces: a signed commit available to the existing remote Ptyche worktree.

- [ ] **Step 1: Review `git diff --check`, focused tests, and worktree status**
- [ ] **Step 2: Stage only spec, plan, tests, and launcher files**
- [ ] **Step 3: Commit with signoff and push the experiment branch**
- [ ] **Step 4: Pull the exact branch in the tracked-clean remote worktree without removing untracked artifacts**

### Task 4: Ptyche Submission and Monitoring

**Files:**
- Generate remotely: unique `exp_logs/qwen3-30ba3b-*/` directories
- Update locally after submission: session state files

**Interfaces:**
- Consumes: committed remote launcher scripts and valid Kerberos credentials.
- Produces: four independent Slurm job IDs with audit paths.

- [ ] **Step 1: Run all four wrappers with `TEST_ONLY=1` on Ptyche**
- [ ] **Step 2: Submit the matrix to `batch` with 20 steps**
- [ ] **Step 3: Record job IDs and confirm that no dependency is attached**
- [ ] **Step 4: Monitor status and early logs for at least five minutes**
- [ ] **Step 5: Update the durable session state with job IDs, paths, and observed status**
