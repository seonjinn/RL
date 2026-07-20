# Real SWE SpecDec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add matched full SWE-RL launch lanes for Eagle-3 K3 and DFlash K7 and submit validated smoke runs.

**Architecture:** Extend the existing typed variant table so each variant owns its method, runner, checkpoint, K, and CUDA Graph policy. Keep the full GRPO command and topology shared, and compare each SpecDec method only with a baseline using the same vLLM model runner.

**Tech Stack:** Python 3.13, NeMo-RL, vLLM 0.25.1, pytest, SLURM/Lyris, W&B

## Global Constraints

- Full real SWE GRPO only; do not use `run_grpo_rollout_benchmark.py`.
- Preserve the inherited 131072-token recipe limits.
- Keep CUDA Graphs enabled and temperature/top-p at 1.0.
- Use Model Runner V2 for Eagle-3 and Model Runner V1 for DFlash.
- Commit and push only to `github-seonjinn:seonjinn/RL.git` before submission.

---

### Task 1: Add Variant Contracts

**Files:**
- Modify: `tests/test_nemogym_swe_full_rl_launcher.py`
- Modify: `experiments/nemogym_swe_full_rl/launch_lyris.py`
- Modify: `experiments/nemogym_swe_full_rl/README.md`

**Interfaces:**
- Consumes: existing `Variant`, `RunPlan`, and `_dry_run()` contracts.
- Produces: `baseline_v1`, `eagle3_k3`, and corrected `dflash_k7` plans.

- [ ] Write dry-run tests for runner, checkpoint, K, draft limit, TP, and graphs.
- [ ] Run the focused tests and confirm they fail against the old launcher.
- [ ] Extend `Variant` and render method-specific overrides and required paths.
- [ ] Run all launcher tests, Ruff, `py_compile`, and `git diff --check`.
- [ ] Commit and push the launcher changes to the personal fork.

### Task 2: Submit Functional Smokes

**Files:**
- Read: `experiments/nemogym_swe_full_rl/launch_lyris.py`
- Write remotely: run provenance and SLURM logs under the experiment run root.

**Interfaces:**
- Consumes: committed `eagle3_k3`, `baseline_v1`, and `dflash_k7` variants.
- Produces: completed two-step W&B runs or exact failure diagnoses.

- [ ] Create one fresh Lyris worktree per submitted variant.
- [ ] Pull the personal branch and initialize recursive submodules exactly.
- [ ] Run `--mode test-only` for each variant.
- [ ] Submit Eagle-3 K3, baseline V1, and DFlash K7 smokes.
- [ ] Monitor each job for at least five minutes and capture W&B links.

### Task 3: Promote And Report

**Files:**
- Update: `experiments/nemogym_swe_full_rl/README.md`
- Update: the canonical NeMo-RL performance report after completed data exists.

**Interfaces:**
- Consumes: passing smoke provenance and metrics.
- Produces: matched 20-step runs and step 2-20 comparison tables.

- [ ] Promote only passing variants to 20 steps.
- [ ] Extract timing, throughput, generation ratio, acceptance, and mean length.
- [ ] Match Eagle-3 to V2 baseline and DFlash to V1 baseline.
- [ ] Mark partial or missing metrics explicitly and link each W&B run.

