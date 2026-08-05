# PR 2964 Upstream-Only H100 Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify PR #2964 HybridEP plus THD sequence packing on H100 without any `seonjinn` Megatron-Bridge or Megatron-LM fork.

**Architecture:** Build an isolated validation branch from PR #2964, merge current NeMo-RL `main`, and restore the exact upstream Bridge URL and gitlink pinned by that `main`. Initialize all submodules recursively, record immutable dependency SHAs, run targeted unit tests, then submit the existing H100 performance recipe through SLURM and inspect the bounded job logs.

**Tech Stack:** Git worktrees and submodules, NeMo-RL, Megatron-Bridge, Megatron-LM, `uv`, Ray, SLURM, CW H100.

## Global Constraints

- Do not modify the user's dirty root checkout.
- No `seonjinn/Megatron-Bridge` or `seonjinn/Megatron-LM` URL may remain in the validation worktree.
- Use the upstream Bridge gitlink pinned by the latest NeMo-RL main, not an arbitrary latest Bridge commit.
- Run GPU work only through SLURM and store logs under a dedicated Lustre experiment directory.
- Commit and push the validation branch before job submission.
- Monitor the submitted job for at least five minutes and inspect only bounded log excerpts.

---

### Task 1: Construct the upstream-only validation branch

**Files:**
- Modify: `.gitmodules`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` gitlink

**Interfaces:**
- Consumes: `origin/pr/2964` and latest `origin/main`
- Produces: a clean validation commit whose Bridge and nested Megatron-LM URLs and SHAs are upstream-only

- [ ] Merge current `origin/main` into the validation branch.
- [ ] Resolve `.gitmodules` and the Bridge gitlink to the exact versions from `origin/main`.
- [ ] Run `git submodule sync --recursive` and `git submodule update --init --recursive`.
- [ ] Record NeMo-RL, Bridge, and Megatron-LM commit SHAs and verify both remote URLs are NVIDIA upstream repositories.
- [ ] Commit and push only the validation changes.

### Task 2: Verify the PR data path locally

**Files:**
- Test: `tests/unit/models/megatron/test_megatron_data.py`

**Interfaces:**
- Consumes: the upstream-only recursive checkout from Task 1
- Produces: targeted unit-test evidence for sequence-packing padding and diagnostics

- [ ] Run the PR's targeted Megatron data unit tests with `uv`.
- [ ] Run formatting or static checks limited to changed Python files if the merge changes Python content.
- [ ] Capture the exact command, exit status, and test count.

### Task 3: Run the H100 integration validation

**Files:**
- Reuse: the existing H100 HybridEP performance recipe and launcher identified from prior experiment artifacts
- Create remotely: a dedicated Lustre experiment directory containing launcher, provenance, and SLURM logs

**Interfaces:**
- Consumes: pushed validation commit and existing H100 recipe settings
- Produces: job ID, bounded logs, completion state, runtime provenance, and HybridEP/THD success or failure evidence

- [ ] Check CW SSH connectivity, Slurm account/partition, available H100 scheduling, and launcher syntax with a scheduling dry run.
- [ ] Clone or fetch the pushed validation branch under Lustre and initialize submodules recursively.
- [ ] Submit the unchanged workload topology with a short validation step count.
- [ ] Monitor scheduling and the first five minutes of execution.
- [ ] On completion, record `sacct`, key step logs, dependency SHAs, and any CUDA/Ray/HybridEP errors.
- [ ] If the smoke run succeeds, report whether a longer performance run is still needed before claiming performance equivalence.

### Task 4: Report the result

**Files:**
- Update: the existing HybridEP HTML experiment report if present

**Interfaces:**
- Consumes: Tasks 1-3 provenance and test evidence
- Produces: a concise upstream-only verdict with limitations

- [ ] State whether custom forks were fully absent.
- [ ] State whether HybridEP plus THD sequence packing initialized and completed steps on H100.
- [ ] Separate correctness validation from performance comparison.
- [ ] Link the local report artifact and identify the remote log directory and job ID.
