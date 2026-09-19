# Qwen3-30B-A3B and Qwen3-235B-A22B Async-1off SpecDec Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce matched 20-step Async-1off no-SpecDec, DFlash, and DSpark measurements for Qwen3-30B-A3B and Qwen3-235B-A22B with complete CUDA Graph coverage.

**Architecture:** Add one experiment-only launcher that inherits the official Async-1off recipes and changes only immutable paths, logging, CUDA Graph controls, and the selected speculative decoder. All primary arms use `max_num_seqs=64`, `enforce_eager=false`, and `FULL_AND_PIECEWISE`; method-aware capture shapes cover request buckets through S64. Each SpecDec arm uses the deep-refit lifecycle fix and must pass a one-step gate before its independent 20-step submission.

**Tech Stack:** Python 3.13, pytest, Hydra CLI overrides, NeMo-RL, vLLM 0.25.1, SLURM, W&B.

**Spec:** User-approved matched Async-1off cohort in this session on 2026-09-18.

## Global Constraints

- Use the official `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` and `grpo-qwen3-235b-32n4g-async-1off.yaml` recipes.
- Use `flashinfer_trtllm`, `enforce_eager=false`, and `FULL_AND_PIECEWISE` for baseline and SpecDec.
- Primary comparisons use S64 for every arm; an uncapped baseline is supplemental and never the speedup denominator.
- Capture sizes include every geometric request bucket through 64 and every method-specific verification width.
- Average completed Steps 3–20 and report exposed-generation and E2E time speedups.
- Keep source under `/home`, transient caches/builds under `/raid/scratch`, and checkpoints/results under `/lustre`.
- Run `sbatch --test-only`, commit and push, then update the remote checkout before submitting.

---

### Task 1: Launcher contract

**Files:**
- Create: `experiments/q30_q235_async1off_specdec_20260918/tests/test_launch.py`
- Create: `experiments/q30_q235_async1off_specdec_20260918/launch.py`

**Interfaces:**
- Consumes: official Async-1off recipe paths and immutable model/drafter paths.
- Produces: `configuration(model, arm, steps)`, `capture_sizes(arm)`, `render(...)`, and `sbatch_arguments(...)`.

- [ ] **Step 1: Write failing contract tests**

Cover recipe selection, S64 on all primary arms, FAP, `enforce_eager=false`, `flashinfer_trtllm`, method-aware capture sizes, deep-refit on SpecDec only, independent submission, and model-specific node counts.

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest -q experiments/q30_q235_async1off_specdec_20260918/tests/test_launch.py`

Expected: collection failure because the launcher module does not exist.

- [ ] **Step 3: Implement the minimal launcher**

Render immutable run artifacts and require every checkpoint/config/container input before submission. Run `sbatch --test-only` before `sbatch --parsable` and do not create job dependencies.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest -q experiments/q30_q235_async1off_specdec_20260918/tests/test_launch.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit the launcher, tests, plan, and experiment README with sign-off.

### Task 2: Remote preflight and one-step gates

**Files:**
- Create: `experiments/q30_q235_async1off_specdec_20260918/SUBMISSIONS.md`

**Interfaces:**
- Consumes: committed launcher SHA and OCI-HSG immutable inputs.
- Produces: scheduler receipts and W&B run identifiers for six one-step jobs.

- [ ] **Step 1: Check FairShare and immutable inputs**

Use one filtered scheduler/account query and bounded `test -f` checks for the two targets, four drafters, container, and remote source SHA.

- [ ] **Step 2: Run scheduler validation**

Run the launcher with `--test-only` for Q30/Q235 baseline, DFlash, and DSpark.

- [ ] **Step 3: Submit independent one-step gates**

Submit all six gates without `afterok` dependencies and record job IDs.

- [ ] **Step 4: Monitor the cohort**

Use one `squeue --me` query per pass at intervals of at least 60 seconds for five minutes; inspect only bounded log tails on failure.

- [ ] **Step 5: Record evidence**

Record source SHA, container, recipe, checkpoint lineage, job ID, scheduler state, and W&B URL.

### Task 3: Twenty-step cohort and report

**Files:**
- Create: `experiments/q30_q235_async1off_specdec_20260918/RESULTS.md`
- Modify: `scripts/plot_specdec_q30_q235_actual_e2e.py`

**Interfaces:**
- Consumes: successful one-step gates and W&B Steps 3–20 histories.
- Produces: a compact Async-1off figure and auditable metric receipts.

- [ ] **Step 1: Submit passing arms for 20 steps**

Submit each passing arm independently with the same recipe, S64, FAP capture sizes, checkpoint, and container as its gate.

- [ ] **Step 2: Extract matched metrics**

Calculate Steps 3–20 means for `timing/train/exposed_generation`, `timing/train/total_step_time`, generation TPS/GPU, E2E TPS/GPU, reward, mean generation length, policy KL, and generation KL.

- [ ] **Step 3: Validate comparability**

Reject a speedup comparison when the target, recipe, topology, S64 cap, CUDA Graph mode, step window, checkpoint lineage, or output-length distribution is mismatched.

- [ ] **Step 4: Render the figure**

Create the existing compact two-panel NVIDIA-green graph with `Exposed generation` and `E2E` bars and `Baseline: No-SpecDec` as the centered subtitle.

- [ ] **Step 5: Verify artifacts**

Open the PNG at original resolution and confirm labels, values, spacing, and provenance before updating the HTML/W&B report.
