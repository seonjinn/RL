# vLLM 0.28 Q30 Synchronous DynamicSD OSL1K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, submit, and summarize a reproducible vLLM 0.28 standalone benchmark for Qwen3-30B-A3B DFlash and DSpark DynamicSD under a NeMo-RL-style synchronous OSL1K rollout barrier.

**Architecture:** Extend the already validated vLLM 0.28 MRV2 Dynamic-K infrastructure with a focused Q30 package. Keep each vLLM engine TP1/DP1, assign 128 requests to each of 16 external workers, and aggregate the maximum worker duration as the synchronous global barrier.

**Tech Stack:** Python 3.13, pytest, vLLM 0.28.0, PyTorch, SLURM, Pyxis/Enroot, GB200.

**Spec:** `docs/superpowers/specs/2026-09-01-vllm028-q30-sync-dynamicsd-osl1k.md`

## Global Constraints

- Pin vLLM `0.28.0` and base commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Use `max_tokens=1024`, `temperature=1.0`, `top_p=1.0`, TP1, DP1, and `FULL_AND_PIECEWISE`.
- Never modify the existing v0.25.1 or v0.27.1 runtime paths.
- Submit canaries only after local tests, commit, push, and `sbatch --test-only` pass.
- Do not submit a dependent production matrix until the canary result validator succeeds.

---

### Task 1: Import the validated v0.28 runtime foundation

**Files:**
- Reuse: `experiments/vllm_028_nemotron_bf16_matrix/`
- Reuse: `tests/test_vllm028_nemotron_bf16_matrix.py`

**Interfaces:**
- Consumes: branch `exp/vllm028-nemotron-bf16-dynamicsd`.
- Produces: authenticated v0.28 patched-container manifest and Lyris launch patterns.

- [ ] Merge the validated v0.28 foundation branch into this isolated worktree.
- [ ] Run `pytest -q tests/test_vllm028_nemotron_bf16_matrix.py tests/test_vllm028_mrv2_patch_canary.py tests/test_vllm028_mrv2_patch_matrix.py`.
- [ ] Confirm zero failures before adding Q30 behavior.

### Task 2: Define the Q30 workload and method contract with TDD

**Files:**
- Create: `experiments/vllm_028_q30_sync_dynamicsd/__init__.py`
- Create: `experiments/vllm_028_q30_sync_dynamicsd/contract.py`
- Create: `tests/test_vllm028_q30_sync_dynamicsd.py`

**Interfaces:**
- Produces: `ExperimentContract`, `MethodPlan`, `build_calibration_rows()`, and
  `build_barrier_rows()`.

- [ ] Write a failing test asserting literal target/drafter paths, OSL 1024,
  64x32 global work, 16 engines, 128 requests per engine, TP1/DP1, and v0.28
  identity.
- [ ] Run the focused test and verify it fails because the package is absent.
- [ ] Implement frozen typed contracts and strict validation.
- [ ] Run the focused test and verify it passes.
- [ ] Write a failing test requiring fixed K `{0,1,2,3,5,7}` across batch sizes
  `{1,2,4,8,16,32,64,96,128}` for both drafters.
- [ ] Implement the minimal matrix builders and verify the complete focused test.

### Task 3: Implement one-engine execution and result validation with TDD

**Files:**
- Create: `experiments/vllm_028_q30_sync_dynamicsd/benchmark.py`
- Create: `experiments/vllm_028_q30_sync_dynamicsd/results.py`
- Modify: `tests/test_vllm028_q30_sync_dynamicsd.py`

**Interfaces:**
- Consumes: `MethodPlan` and a sealed prompt manifest.
- Produces: one immutable worker result containing elapsed seconds, token counts,
  per-request finish times, SpecDec counters, selected-K evidence, and runtime
  provenance.

- [ ] Write a failing test in which a fake local engine returns two concrete
  completions and assert the literal barrier/token summary.
- [ ] Run it and verify the missing runner behavior is the failure.
- [ ] Implement prompt partitioning, generation timing, metric extraction, and
  atomic no-clobber JSON publication.
- [ ] Verify the test passes.
- [ ] Write failing tests rejecting OSL above 1024, DP above one, missing graph
  evidence, mismatched model identity, and incomplete output rows.
- [ ] Add a failing K0 diagnostic test requiring separate fields for selected
  verifier K, physical draft width, and observed drafter execution.
- [ ] Implement strict result validation and verify all focused tests pass.

### Task 4: Implement calibration and schedule selection with TDD

**Files:**
- Create: `experiments/vllm_028_q30_sync_dynamicsd/calibrate.py`
- Modify: `tests/test_vllm028_q30_sync_dynamicsd.py`

**Interfaces:**
- Consumes: validated per-BS, per-K calibration results.
- Produces: a contiguous monotone `[[start_bs,end_bs,k], ...]` schedule covering
  BS 1 through 128 and a best fixed K.

- [ ] Write a failing table-driven test with hand-calculated throughput values
  whose expected result is `[[1,8,5],[9,32,3],[33,64,2],[65,128,0]]`.
- [ ] Verify the expected failure.
- [ ] Implement selection, monotonicity, contiguous-range merging, and tie
  breaking toward smaller K.
- [ ] Verify the test and all focused tests pass.

### Task 5: Render safe Lyris canary and matrix jobs with TDD

**Files:**
- Create: `experiments/vllm_028_q30_sync_dynamicsd/submit.py`
- Create: `experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml`
- Modify: `tests/test_vllm028_q30_sync_dynamicsd.py`

**Interfaces:**
- Produces: one-GPU canary scripts, independent calibration scripts, and a
  4n4g/16-worker global-barrier script.

- [ ] Write failing tests asserting account `coreai_dlalgo_llm`, partition
  `gb200`, node-local caches, exact v0.28 guards, `max_tokens=1024`,
  `FULL_AND_PIECEWISE`, DP1, no overwrite, and source/result provenance.
- [ ] Verify the expected failure.
- [ ] Implement rendering and `--test-only`/`--submit` modes without scheduler
  side effects in the renderer.
- [ ] Verify focused tests, Ruff, Pyright, and Bash syntax.

### Task 6: Commit, push, preflight, and submit the canaries

**Files:**
- Create after submission: `experiments/vllm_028_q30_sync_dynamicsd/submissions/canary.json`

**Interfaces:**
- Produces: durable job IDs and immutable submission receipts.

- [ ] Commit only the new Q30 package, tests, design, and plan with sign-off.
- [ ] Push the isolated branch.
- [ ] Pull the branch into `/home/sna/Nemo-RL_Qwen3_Roadmap-vllm028-q30-sync`.
- [ ] Verify target, drafter, container, prompt, and output paths.
- [ ] Run `sbatch --test-only` for every canary script.
- [ ] Submit baseline, DFlash DynamicSD, and DSpark DynamicSD canaries.
- [ ] Include K0 diagnostics for both drafters and a separate DSpark adaptive
  verification compatibility canary; do not combine the two controllers.
- [ ] Use one filtered scheduler query, then inspect bounded log tails for five
  minutes at intervals of at least 60 seconds.

### Task 7: Gate and expand to the calibration/full barrier matrix

**Files:**
- Create after validation: `experiments/vllm_028_q30_sync_dynamicsd/submissions/calibration.json`
- Create after validation: `experiments/vllm_028_q30_sync_dynamicsd/submissions/barrier.json`

**Interfaces:**
- Consumes: successful canary results.
- Produces: calibrated schedules and matched 16-engine speedup measurements.

- [ ] Validate v0.28 identity, graph capture, selected-K behavior, exact work,
  speculative counters, and K0 physical-draft execution for both drafters.
- [ ] Submit all independent calibration cells without dependencies.
- [ ] Derive and freeze one schedule and best fixed K per drafter.
- [ ] Submit target-only, best-fixed, and DynamicSD barrier arms for three
  repetitions.
- [ ] If the DSpark checkpoint exposes the required confidence head, submit a
  separate adaptive-verification arm against fixed DSpark; otherwise publish
  the incompatibility receipt without a performance row.
- [ ] Aggregate global barrier speedup, TPS/GPU speedup, tail latency,
  acceptance, and K histograms into CSV and HTML.
