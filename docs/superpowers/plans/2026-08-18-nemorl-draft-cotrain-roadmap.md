# NeMo RL Speculative Draft Co-Training PR Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver reviewable Megatron co-training support for DFlash, DSpark, and multi-pass EAGLE-3 as eleven independently tested pull requests plus one non-PR integration branch.

**Architecture:** Each pull request introduces only contracts used by that pull request. Training methods produce immutable plans and explicit loss statistics; serving uses a later capability-driven adapter and target/draft component manifests. A separate integration branch combines all heads for Qwen3-8B DFlash/DSpark end-to-end validation without becoming an upstream pull request.

**Tech Stack:** Python 3.12, PyTorch, Megatron Core, Pydantic v2, vLLM 0.25.1 and 0.27.1, NCCL reshard, pytest, SLURM, self-contained HTML reporting.

**Spec:** `docs/superpowers/specs/2026-08-18-nemorl-speculative-draft-co-training.md`

## Global Constraints

- Implementation scope is the Megatron policy backend; Automodel draft training is out of scope.
- Production remains pinned to vLLM 0.25.1 until PR 11; compatibility code supports both 0.25.1 and 0.27.1.
- DSpark serving is Markov-only; do not advertise confidence-head scheduling.
- No production file may call private positional FlashAttention forward/backward APIs.
- No draft hot path may call `.cpu()`, `.item()`, or loop over batch rows for anchor or bucket planning.
- Each method PR must support both synchronous and split training before it is mergeable.
- Each PR contains only code consumed by that PR and its own acceptance tests.
- Each upstream PR is opened from `seonjinn/RL`, remains draft until its `review-pr-team` gate runs on OCI-Hsg (primary GPU-capable host), posts output/findings/dispositions to the PR, resolves high-confidence findings with regression tests where applicable, and only then requests human review; use Summary, Why, optional Performance, and Validation sections.
- Qwen3-8B is the required DFlash/DSpark integration model; BF16-to-MXFP8 refit retains a separate targeted regression gate.
- GPU runs use pinned containers, record raw artifacts, and compare vLLM versions in separate environments.
- Primary GPU execution uses OCI-Hsg host `oci-hsg-cs-001-vscode-02`, account `nemotron_n3_post`, and partition `batch`; Lyris GB200 and Pre-Tyche GB200 are preflighted fallbacks.

---

### Task 1: Maintain the Review Dashboard

**Files:**
- Create: `docs/draft_cotraining_pr_review/context.json`
- Create: `scripts/build_draft_cotraining_pr_review.py`
- Create: `tests/test_build_draft_cotraining_pr_review.py`
- Generate: `docs/draft_cotraining_pr_review/index.html`
- Plan: `docs/superpowers/plans/2026-08-18-nemorl-draft-cotrain-review-dashboard.md`

**Interfaces:**
- Consumes: this roadmap, the approved spec, branch SHAs, PR URLs, focused test records, self-review results, and benchmark artifacts.
- Produces: `python3 scripts/build_draft_cotraining_pr_review.py --context ... --output ...` and one offline HTML page with overview sections and eleven button-selected PR subpages.

- [ ] **Step 1: Execute the dashboard-specific TDD plan.**

Run each red/green cycle in `2026-08-18-nemorl-draft-cotrain-review-dashboard.md`.

- [ ] **Step 2: Validate the generated HTML contract.**

Run:

```bash
python3 /Users/sna/.codex/skills/explain-diff-html/scripts/validate_explainer.py \
  docs/draft_cotraining_pr_review/index.html
```

Expected: exit code 0 with every required section and exactly five quiz questions.

- [ ] **Step 3: Refresh after every PR state change.**

Update `context.json` first, regenerate, rerun renderer tests and the explainer validator, then commit the context, renderer, test, and generated HTML together in the roadmap repository.

### Task 2: PR 1 — Typed Training Contracts

**Files:**
- Plan: `docs/superpowers/plans/2026-08-18-nemorl-draft-cotrain-pr01-contracts.md`
- Branch: `seonjinn/draft-cotrain-contracts-20260818`

**Interfaces:**
- Consumes: current single-pass EAGLE configuration and build/forward/export behavior at upstream `7fa6e55192530ff1346d670ce74f9c70cab8f75b`.
- Produces: a typed training configuration boundary and method lookup used immediately by the existing EAGLE path, with exact legacy YAML/default parity and no serving runtime types.

- [ ] **Step 1: Execute the PR 1 TDD plan.**

Use `2026-08-18-nemorl-draft-cotrain-pr01-contracts.md`; do not add DFlash, DSpark, TTT, loss, optimizer, or vLLM fields.

- [ ] **Step 2: Run focused and no-draft parity tests.**

Expected: existing EAGLE recipe resolves identically, disabled/omitted draft configs remain unchanged, and invalid method/config combinations fail at the typed boundary.

- [ ] **Step 3: Self-review and publish the draft PR.**

Run repository pre-commit plus `review-pr-team`, resolve high-confidence findings, push only explicit PR 1 files to `seonjinn/RL`, and open a concise draft PR against NVIDIA-NeMo/RL `main`.

### Task 3: PR 2 — Streaming Vocab-Parallel Soft CE

**Files:**
- Create: `nemo_rl/models/megatron/draft/loss.py`
- Modify: the existing EAGLE loss wrapper at the then-current upstream locations.
- Test: `tests/unit/models/megatron/test_draft_loss.py`
- Test: a TP=2 distributed draft-loss test selected after rebasing PR 1.

**Interfaces:**
- Consumes: the PR 1 EAGLE training method and target/student vocab-parallel logits.
- Produces: `DraftLossStats(numerators, counts, weights, metrics)` with equal-shaped bins and `StreamingVocabParallelSoftCE` used by existing EAGLE in the same PR.

- [ ] **Step 1: Write the separate PR 2 implementation plan against merged PR 1.**

The plan must pin exact files and public signatures after PR 1 lands; it must include unequal-weight and uneven-tail red tests.

- [ ] **Step 2: Implement with red/green cycles.**

Verify FP32 dense equivalence, local-vocab gradients, TP reduction semantics, bounded teacher tiles, and no full block-teacher temporary.

- [ ] **Step 3: Self-review and publish.**

Post focused unit/distributed results and keep performance claims out unless a matched measurement exists.

### Task 4: PR 3 — DraftStepState and Existing EAGLE Split Parity

**Files:**
- Create: `nemo_rl/models/megatron/draft/step_state.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: current EAGLE train/loss wiring selected after PR 2.
- Test: focused sync/split parity tests in `tests/unit/models/policy/` and distributed DP=2 coverage.

**Interfaces:**
- Consumes: PR 2 `DraftLossStats`.
- Produces: `DraftStepState` that accumulates per-bin raw numerators/counts and applies independent draft normalization for the current single-pass EAGLE path.

- [ ] **Step 1: Write the PR 3 plan from the post-PR 2 worker APIs.**

Pin the exact `begin_train_step`, microbatch, and finish-step seams and define sample-identity behavior.

- [ ] **Step 2: Prove the current bug with a failing split-vs-sync test.**

Expected red failure: split draft gradients or metrics differ when policy-token and draft-token counts differ.

- [ ] **Step 3: Implement, verify DP parity, self-review, and publish.**

Do not add future DFlash anchor fields; the state carries method-owned opaque plan slices only when a current consumer exists.

### Task 5: PR 4 — Optional Draft Optimizer Overrides

**Files:**
- Modify: the then-current Megatron optimizer-group builder only.
- Test: focused optimizer-group and actual-step tests.

**Interfaces:**
- Consumes: existing `DRAFT_GRAD_NORM_GROUP` tagging and separate clipping behavior.
- Produces: optional draft learning-rate and weight-decay overrides; no new clipping implementation.

- [ ] **Step 1: Plan exact config and optimizer seams after PR 3.**

The red tests must prove omitted overrides preserve no-draft and existing EAGLE parameter groups exactly.

- [ ] **Step 2: Implement the minimum override group and run a real optimizer step.**

Expected: policy and draft parameters receive configured learning rates/weight decay while clipping groups remain unchanged.

- [ ] **Step 3: Self-review and publish without unrelated optimizer refactors.**

### Task 6: PR 5 — Internal DFlash Core

**Files:**
- Create: focused DFlash plan, attention, and model modules chosen by the PR-specific file map.
- Test: dense attention oracle, vectorized-plan, model forward/backward, and checkpoint round-trip tests.

**Interfaces:**
- Consumes: target hidden-state and embedding tensors supplied directly by tests.
- Produces: internal `DFlashBatchPlan` and DFlash model forward; no policy-worker, loss, config-recipe, or vLLM wiring.

- [ ] **Step 1: Write the DFlash-core spec/plan before code.**

Define the `1 + gamma` query contract, anchor-conditioning loss exclusion, deterministic per-sample anchor function, and public structured-attention backend.

- [ ] **Step 2: Implement plan and attention through dense-oracle red/green cycles.**

Cover MHA/GQA, duplicate anchors, empty/remainder/full trunks, masked rows, Q/K/V gradients, and capture-hostile operation checks.

- [ ] **Step 3: Add model and checkpoint round trip, benchmark the kernel, self-review, and publish.**

Keep the core internal until PR 6 supplies the production consumer.

### Task 7: PR 6 — DFlash Co-Training Integration

**Files:**
- Modify: training-method registry and Megatron draft setup/wiring from PRs 1-3.
- Modify: checkpoint export utilities.
- Add: one Qwen3-8B DFlash recipe or test fixture selected after rebasing.
- Test: synchronous/split one-step and export coverage.

**Interfaces:**
- Consumes: PR 5 DFlash core, PR 2 loss, PR 3 step state, PR 4 optimizer overrides.
- Produces: user-selectable `method=dflash` training and logical draft-weight export for later serving adapters.

- [ ] **Step 1: Write failing setup, sync, split, and export tests.**

The split test must prove stable sample-ID anchors and exact gradient/metric parity; invalid gamma/mask/tap settings must fail before model construction.

- [ ] **Step 2: Add the thinnest registry/worker wiring needed to pass.**

Do not add vLLM runner accessors or transport logic.

- [ ] **Step 3: Run Qwen3-8B one-step training, self-review, and publish.**

Record training time and peak memory as evidence, not as serving performance.

### Task 8: PR 7 — DSpark Markov Extension

**Files:**
- Create: a small DSpark method/model extension module.
- Modify: PR 6 registry/export wiring only where DSpark differs.
- Test: slot/label/Markov-bias tests and Qwen3-8B sync/split tests.

**Interfaces:**
- Consumes: DFlash visibility/attention core and shared training contracts.
- Produces: Markov-only DSpark with `sample_from_anchor=true`, `gamma` queries and outputs, slot-zero next-anchor label, and teacher-forced prior-token Markov inputs.

- [ ] **Step 1: Write a PR-specific plan from official 0.25.1 and 0.27.1 DSpark contracts.**

The first red test must distinguish DSpark's `gamma` queries from DFlash's `1 + gamma` queries and catch an off-by-one label.

- [ ] **Step 2: Implement only the Markov delta and export mapping.**

Do not add confidence-head serving or duplicate DFlash attention code.

- [ ] **Step 3: Run Qwen3-8B one-step training, self-review, and publish.**

### Task 9: PR 8 — Bounded Multi-Pass EAGLE-3 TTT

**Files:**
- Create: explicit EAGLE pass-plan/cache and structured attention backend modules.
- Modify: existing EAGLE method/model only.
- Test: full-model pass, RoPE, recursive-gradient, memory, and sync/split parity tests.

**Interfaces:**
- Consumes: shared loss/step/optimizer contracts.
- Produces: `ttt_steps` with K=1 exact existing parity and bounded K>1 explicit pass state.

- [ ] **Step 1: Write the PR-specific plan and K=1 failing compatibility harness.**

Define exact label indices, RoPE positions, supported maximum, and prior-pass gradient requirements.

- [ ] **Step 2: Implement K=1 parity before enabling K>1.**

No core-attention mutation or persistent hooks may survive an exception.

- [ ] **Step 3: Enable K=2/4/max, profile retained KV memory, self-review, and publish.**

Do not claim O(K) memory without peak-allocation evidence.

### Task 10: PR 9 — Dual-Version vLLM Runtime and Collective/IPC Refit

**Files:**
- Create: a focused draft runtime-adapter module.
- Modify: vLLM backend weight-loading lifecycle only after the native BF16 reload lifecycle is merged on `main`.
- Test: fake-runner capability tests plus real 0.25.1/0.27.1 install/refit smokes.

**Interfaces:**
- Consumes: logical EAGLE/DFlash/DSpark exports and the merged reload lifecycle.
- Produces: capability-driven draft access, target/draft component manifest, owner/support validation, exact load coverage, and collective/IPC finalization.

- [ ] **Step 1: Rebase on the merged #3659-equivalent lifecycle and write the adapter plan.**

Version parsing is diagnostic only; behavior tests exercise `get_draft_model`, `drafter.model`, and `speculator.model` capabilities.

- [ ] **Step 2: Write failing owner/non-owner and unsupported-PP tests.**

Expected: supported non-owner ranks participate without loading; unsupported runner/method/PP combinations fail during setup.

- [ ] **Step 3: Implement, run Qwen3-8B DFlash/DSpark collective and IPC E2E in both vLLM environments, self-review, and publish.**

### Task 11: PR 10 — Component-Aware NCCL Draft Reshard

**Files:**
- Modify: NCCL reshard classification/manifest code after PR 9.
- Modify: Megatron and vLLM local-map builders only at component boundaries.
- Test: target/draft routing, coverage, finalization, and BF16-to-MXFP8 regressions.

**Interfaces:**
- Consumes: PR 9 component manifest and merged reload lifecycle.
- Produces: draft-aware misc/direct routing with direct transfer permitted only for stable exact-layout live storage.

- [ ] **Step 1: Write the PR plan after rebasing open/merged #3477 and #3659 changes.**

Pin base SHAs and classify every overlap; do not modify `quantization/fp8.py` for draft support.

- [ ] **Step 2: Reproduce the current `draft.*` suffix misclassification in a red test.**

Expected: draft FFN names never enter a target-only direct map.

- [ ] **Step 3: Implement, run Qwen3-8B NCCL E2E plus BF16-to-MXFP8 targeted smoke, self-review, and publish.**

### Task 12: PR 11 — vLLM 0.27.1 Default Bump

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, container pins, and version documentation only.
- Test: install/import, focused refit, full Qwen3-8B 0.27.1 primary matrix, and 0.25.1 fallback adapter matrix.

**Interfaces:**
- Consumes: PR 9 semantic compatibility and PR 10 refit integration.
- Produces: vLLM 0.27.1 as the default environment while retaining explicit 0.25.1 compatibility coverage.

- [ ] **Step 1: Write the dependency-only plan and snapshot the pre-bump 0.25.1 matrix.**

If semantic code changes are required, stop and add them to PR 9 rather than this PR.

- [ ] **Step 2: Update pins/locks/containers and run dependency validation.**

Expected diff contains no NeMo RL Python behavior changes.

- [ ] **Step 3: Run both Qwen3-8B environments, self-review, and publish.**

### Task 13: Integration Worktree and Final End-to-End Gate

**Files:**
- Worktree: `.worktrees/nemorl-draft-cotrain-integration-20260818`
- Branch: `seonjinn/draft-cotrain-integration-20260818`
- Artifacts: roadmap experiment/report directories selected before GPU submission.

**Interfaces:**
- Consumes: the eleven reviewed branch heads without changing their PR diffs.
- Produces: one canary commit set and a reproducible Qwen3-8B DFlash/DSpark result matrix; it is never proposed as a large upstream PR.

- [ ] **Step 1: Create or refresh the integration branch from the latest common upstream base.**

Merge the reviewed heads in dependency order and record every source SHA in dashboard context.

- [ ] **Step 2: Run local and distributed preflight tests.**

Expected: config, unit, sync/split, export, adapter, and refit tests pass before any cluster submission.

- [ ] **Step 3: Run the Qwen3-8B matrix in separate 0.25.1 and 0.27.1 containers.**

Compare no speculation, static DFlash/DSpark, and co-trained DFlash/DSpark over training, refit, and rollout. Run at least five matched repetitions and retain raw per-request/per-refit records.

Before submission, run the local launcher `--test-only` contract, query current
accounts/FairShare, commit and push the exact integration SHA, and submit first
to OCI-Hsg `batch` under `nemotron_n3_post`. Monitor the allocation and driver
logs for the first five minutes. Use Lyris or Pre-Tyche only after recording the
fallback reason and creating a cluster-specific immutable config; do not combine
measurements across clusters.

- [ ] **Step 4: Route every failure back to its owning PR.**

Add the regression test and fix to the smallest owning branch; never leave an integration-only workaround.

- [ ] **Step 5: Regenerate and validate the review dashboard.**

The overview must show the final branch SHAs, test matrix, self-review disposition, benchmark provenance, and any unsupported combinations.
