# Draft Cadence Step-2 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen3-8B DFlash and DSpark always/fixed cadence runs survive the deterministic Step-2 failures while retaining the existing learning semantics.

**Architecture:** Persist cadence terminal evidence at the same mutation boundary that prepares a decision. For sparse-update skip steps, force one synchronous Megatron DDP lifecycle: finish parameter publication through the supported forward-hook API and temporarily disable overlapped gradient reduction while the skipped forward/backward is active. Do not access bucket internals or mutate communication handles. This recovery is deliberately separate from the later target/draft DDP ownership split.

**Tech Stack:** Python 3.13, PyTorch, Megatron-Core DDP, NeMo-RL synchronous GRPO, pytest, uv

**Spec:** User-approved independent target/draft DDP design recorded in the 2026-08-24 conversation; this plan implements its bounded recovery gate first.

## Global Constraints

- Keep Megatron-Core pinned at `14346b65a2d0790e451919858f7771078105c5f0`.
- Do not access `_ParamAndGradBucketGroup`, `param_gather_handle`, or `param_gather_dispatched` from NeMo-RL production code.
- Preserve policy optimizer and scheduler advancement on every policy step.
- Preserve the current draft LR-clock behavior while suppressing draft parameter updates on skipped steps.
- Apply the gather barrier from the actual `update_requested=False` decision, not from a schedule-name string.
- Keep always-online and fixed `refit_only` forward overlap unchanged.
- Use `uv run` for Python tooling and signed conventional commits.

---

### Task 1: Persist Prepared Cadence Evidence

**Files:**
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Test: `tests/unit/algorithms/test_grpo_sync_draft_schedule.py`

**Interfaces:**
- Consumes: `PreparedSyncDraftDecision.terminal_evidence` and `GRPOSaveState.draft_terminal_evidence`.
- Produces: `persist_prepared_terminal_evidence(save_state, evidence) -> None`, called before the cadence transaction and worker training begin.

- [ ] **Step 1: Write a failing two-step regression test**

Construct a real `CadenceRuntimeWriter`, prepare the second always-update decision after a recorded refit observation, invoke `apply_scheduled_refit`, and assert that live evidence and save-state evidence remain equal.

- [ ] **Step 2: Run the focused test and verify the production lifecycle raises**

Run:

```bash
uv run pytest tests/unit/algorithms/test_grpo_sync_draft_schedule.py -k terminal_evidence -vv
```

Expected: failure containing `checkpointed terminal evidence diverged before update`.

- [ ] **Step 3: Add the minimal persistence helper and call it at the mutation seam**

The helper accepts `CadenceTerminalEvidence | None`, rejects missing evidence when runtime persistence is enabled, and assigns `evidence.state_dict()` before the transaction begins.

- [ ] **Step 4: Run the focused test and related cadence tests**

Run:

```bash
uv run pytest tests/unit/algorithms/test_grpo_sync_draft_schedule.py tests/unit/algorithms/test_draft_schedule_checkpoint.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add nemo_rl/algorithms/grpo_sync.py tests/unit/algorithms/test_grpo_sync_draft_schedule.py
git commit -s -m "fix(grpo): persist prepared draft cadence evidence"
```

### Task 2: Add a Public DDP Lifecycle Guard for Sparse Draft Skips

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Test: `tests/unit/models/megatron/test_megatron_worker.py`

**Interfaces:**
- Consumes: `draft_enabled`, `run_draft`, the configured `overlap_param_gather`, and the current DDP forward-hook state.
- Produces: `_conditional_draft_skip_ddp_sync()` context manager that restores exactly the forward-hook, `model_config.param_sync_func`, `model_config.grad_sync_func`, and `ddp_config.overlap_grad_reduce` state present at entry.

- [ ] **Step 1: Write failing lifecycle tests**

Cover entry with hooks enabled, entry with hooks already disabled, body failure, `run_draft=True`, parameter overlap disabled, and gradient overlap disabled. Assert that `disable_forward_pre_hook(param_sync=True)` is called only when a real skip has an active parameter hook, gradient overlap becomes synchronous for every real skip, and all entry state is restored.

- [ ] **Step 2: Run the focused tests and verify the helper is missing**

Run:

```bash
uv run pytest tests/unit/models/megatron/test_megatron_worker.py -k conditional_draft_skip -vv
```

Expected: failure because `_conditional_draft_skip_ddp_sync` does not exist.

- [ ] **Step 3: Implement the minimal context manager**

Use `disable_forward_pre_hook(param_sync=True)`, `enable_forward_pre_hook()`, the public model-config sync callbacks, and the public `ddp_config.overlap_grad_reduce` setting. Null `grad_sync_func` together with gradient overlap so pipeline schedules cannot perform an early synchronous reduction and then repeat it during gradient finalization. Do not call `start_param_sync()` directly and do not inspect bucket state. Parameter and gradient overlap are independent: an already-disabled parameter hook must not prevent the gradient lifecycle barrier.

- [ ] **Step 4: Wrap monolithic forward/backward and optimizer finalization**

Enter the guard immediately before `megatron_forward_backward` and restore it after that call has finalized model gradients. The optimizer then runs with the original DDP configuration. A raised forward/backward exception must still restore entry state.

- [ ] **Step 5: Run worker and optimizer-suspension tests**

Run:

```bash
uv run pytest tests/unit/models/megatron/test_megatron_worker.py tests/unit/models/megatron/test_draft_optimizer_suspension.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_megatron_worker.py
git commit -s -m "fix(megatron): guard sparse draft skip parameter sync"
```

### Task 3: Validate Configs and Recovery Boundaries

**Files:**
- Modify: experiment config generator or verifier under `research/` selected by the existing Q8 cadence experiment
- Test: corresponding experiment verification test

**Interfaces:**
- Consumes: always, fixed-5, fixed-10, fixed-20 arms for DFlash and DSpark.
- Produces: two-step and boundary canary configurations with explicit source revision and recovery-mode metadata.

- [ ] **Step 1: Add a failing config-verification assertion**

Require every cadence arm to record `source_revision`, schedule mode, update interval, and whether the recovery DDP barrier is active.

- [ ] **Step 2: Run the experiment verifier and observe the missing metadata failure**

Run the verifier command documented in the experiment README with `--test-only` or its local validation equivalent.

- [ ] **Step 3: Add the minimal metadata/config changes**

Do not disable `overlap_param_gather` in production configs. Create a separate control canary override with it disabled for diagnosis only.

- [ ] **Step 4: Run config tests and static validation**

Expected: all cadence arms resolve to the same model/data topology, differing only by drafter family and cadence schedule.

- [ ] **Step 5: Commit Task 3**

```bash
git add research tests
git commit -s -m "test(specdec): add cadence recovery canary gates"
```

### Task 4: Distributed Canary and Production Submission

**Files:**
- Modify: experiment README/report with submitted job IDs and immutable source revision
- No production source modifications

**Interfaces:**
- Consumes: committed and pushed recovery branch.
- Produces: 2-step recovery evidence, fixed-5 update-boundary evidence, and production job IDs.

- [ ] **Step 1: Run SLURM `--test-only` for the complete array**

Expected: all arms resolve resources, container, checkpoint, and output directories without submission.

- [ ] **Step 2: Commit and push the exact source revision**

Use signed commits and push only the recovery branch.

- [ ] **Step 3: Submit 2-step DFlash and DSpark always/fixed-5 canaries**

Monitor one scheduler query covering all submitted jobs for at least five minutes.

- [ ] **Step 4: Validate the first fixed-5 update boundary**

Run at least six steps and verify skip-to-update-to-skip behavior, target/draft parameter receipts, and absence of Step-2 DDP/evidence failures.

- [ ] **Step 5: Submit always/fixed-5/fixed-10/fixed-20 production arms**

Record job IDs, source revision, configs, logs, and W&B links in the experiment report.

- [ ] **Step 6: Start the independent Draft DDP implementation plan**

Create a separate plan covering model ownership, two-phase optimizer commit, checkpoint migration, CP split lifecycle, and overlap performance validation. Do not represent the recovery barrier as the final performance architecture.
