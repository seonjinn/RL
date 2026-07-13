# SpecDec RL Framework Lessons And NeMo-RL Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an evidence-linked cross-framework SpecDec/RL report and make NeMo-RL fail loudly when speculative decoding would run with unsupported methods, an invalid draft lifecycle, or missing draft weights.

**Architecture:** The Pages repository owns a static evidence report organized by failure class rather than framework. The isolated `sna/nemorl-vllm024-upgrade` worktree owns typed configuration resolution and runtime draft-weight checks. Performance changes inside vLLM, including generic PARD CUDA-graph support, remain a separately gated experiment until correctness and parity tests pass.

**Tech Stack:** Static HTML/CSS, Python 3.13, pytest, NeMo-RL typed configuration, vLLM 0.24.0 worker extensions.

## Global Constraints

- Keep all existing user changes in both worktrees; do not revert or reformat unrelated files.
- Every framework claim must link directly to a primary GitHub PR, issue, source permalink, or local benchmark artifact.
- Distinguish merged fixes, open proposals, and unresolved issues visually and textually.
- Do not expose credentials, W&B API keys, internal tokens, or raw private logs in the Pages report.
- CUDA Graph remains enabled for performance conclusions; eager-only results must be labeled and must not prove production speedup.
- `policy.draft.enabled` means online Eagle draft training only; it must not silently imply that every vLLM speculative method receives refit draft weights.
- Unsupported methods and missing draft weights must fail at setup/update time rather than continue with dummy, stale, or partial weights.
- Sampling defaults must match vLLM 0.24 semantics: `rejection_sample_method=standard` and `draft_sample_method=greedy` unless explicitly configured and logged.
- No PARD/PARD-2 performance claim is complete without matched TP, target/draft CUDA-graph mode, acceptance by position, and matched baseline evidence.

---

### Task 1: Cross-Framework Evidence Report

**Files:**
- Create: `scripts/build_specdec_rl_framework_lessons.py`
- Create: `docs/specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html`
- Modify: `scripts/build_pages_index.py`
- Modify: `tests/test_build_pages_index.py`
- Generate: `public/reports/specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html`
- Generate: `public/index.html`
- Modify: `.gitlab-ci.yml`

**Interfaces:**
- Consumes: primary-source findings for veRL, slime, Miles, SGLang, vLLM 0.24, and current NeMo-RL code.
- Produces: canonical report URL and a dashboard navigation entry.

- [ ] **Step 1: Build the evidence matrix**

Create a dedicated builder with structured evidence records and generate a self-contained HTML page whose primary matrix rows are `Draft weight lifecycle`, `Online drafter training`, `Sampling correctness`, `CUDA graph capture`, `Buffer sizing`, `Metrics`, and `Failure recovery`. Columns cover veRL, slime, Miles, SGLang/vLLM, and the corresponding NeMo-RL gap. Each framework cell contains concise findings and direct links to the exact PR or issue.

- [ ] **Step 2: Add the NeMo-RL gap and action tables**

Add columns for severity, current code evidence, user-visible impact, upstream lesson, implementation status, and validation gate. Separate confirmed defects from performance hypotheses and already-resolved items.

- [ ] **Step 3: Add report navigation and CI validation**

Insert the report as the first NeMo-RL item and as a primary link in `scripts/build_pages_index.py`. Extend `tests/test_build_pages_index.py` to prove the link is visible outside the archive disclosure. Run the generator to create `public/index.html` and copy the report into `public/reports/`. Add a `test -f` entry to `.gitlab-ci.yml` for the canonical report.

- [ ] **Step 4: Validate the document**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

class Parser(HTMLParser):
    pass

for path in (
    Path("public/index.html"),
    Path("public/reports/specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html"),
):
    Parser().feed(path.read_text())
    print(f"parsed {path}")
PY
```

Expected: both files print as parsed with no exception.

### Task 2: Resolve And Validate The SpecDec Lifecycle

**Files:**
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/__init__.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/examples/run_grpo.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**
- Consumes: policy draft-training state and `generation.vllm_kwargs.speculative_config`.
- Produces: a resolved startup load format and explicit lifecycle validation.

- [ ] **Step 1: Add failing lifecycle tests**

Cover these cases independently:

```python
def test_online_draft_refit_rejects_generic_draft_model_method(): ...
def test_pard2_method_fails_before_vllm_startup(): ...
def test_generic_pard_uses_separate_auto_draft_load_without_online_refit(): ...
def test_online_eagle_refit_keeps_dummy_load_format(): ...
def test_generic_draft_tp_must_match_target_tp(): ...
def test_suffix_does_not_force_target_checkpoint_load(): ...
```

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_generation.py -k 'draft or pard' -q
```

Expected before implementation: the new invalid-config tests fail.

- [ ] **Step 2: Implement one lifecycle resolver**

Add a focused helper used by `configure_generation_config` that:

```python
def validate_vllm_speculative_config(
    config: VllmConfig,
    *,
    has_refit_draft_weights: bool,
) -> None:
    ...
```

It rejects `method=pard2`, rejects online draft refit for methods other than `eagle`/`eagle3`, requires a model for neural external drafters, and requires generic `draft_model` TP to equal target TP. For static neural external drafters, keep the NeMo-refit target at `load_format=dummy` and set `speculative_config.draft_load_config={"load_format": "auto"}` so only the drafter is read from disk. Preserve an explicit user-provided draft load config. It leaves Suffix, ngram, and MTP lifecycle handling distinct and does not force target checkpoint loading for model-free proposers.

- [ ] **Step 3: Derive online-refit state from both policy and method**

Change `run_grpo.py` so a bare `policy.draft.enabled=true` cannot authorize generic PARD/PARD-2 refit behavior. Preserve online Eagle behavior.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_generation.py -k 'draft or pard or speculative' -q
```

Expected: all selected tests pass.

### Task 3: Make Draft Weight Updates Complete And Observable

**Files:**
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/tests/unit/models/generation/test_vllm_backend.py`

**Interfaces:**
- Consumes: `draft.*` tensors from NeMo-RL refit and the vLLM drafter model.
- Produces: verified loaded parameter names or a startup/update exception.

- [ ] **Step 1: Add failing draft-weight tests**

Add tests that assert:

```python
def test_load_draft_weights_raises_without_vllm_drafter(): ...
def test_load_draft_weights_raises_when_loader_reports_no_loaded_names(): ...
def test_load_draft_weights_accepts_nonempty_loader_result(): ...
def test_load_draft_weights_accepts_eagle_loader_without_name_result(): ...
```

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_backend.py -k load_draft_weights -q
```

Expected before implementation: the missing-drafter and partial-load tests fail.

- [ ] **Step 2: Harden `_load_draft_weights`**

Replace the current print-and-return behavior with `RuntimeError`. Collect the names returned by `draft_model.load_weights` when that model exposes them and reject an empty returned set for a non-empty update. Preserve vLLM Eagle loaders that currently return `None`, but emit one explicit unverified-loader diagnostic instead of claiming complete name coverage. Do not scan all tensor values or add GPU synchronization to the refit hot path. Return the loaded-name set or `None` for testing and diagnostics.

- [ ] **Step 3: Preserve the no-draft update path**

Keep an empty incoming draft list as a no-op. Only raise when draft tensors actually arrive and cannot be applied completely.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_backend.py -k 'draft or mtp' -q
```

Expected: all selected tests pass.

### Task 4: Reject Missing Or Corrupt Generation Logprobs

**Files:**
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/utils.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/tests/unit/models/generation/test_vllm_utils.py`

**Interfaces:**
- Consumes: generated token IDs and vLLM per-token processed-logprob dictionaries.
- Produces: one finite logprob for every generated token or a setup/output error.

- [ ] **Step 1: Add failing extraction tests**

Cover a complete response, missing logprob list, short logprob list, missing chosen-token entry, and a non-finite chosen-token logprob. The empty-generation case remains valid.

- [ ] **Step 2: Add one pure extraction helper**

Create a typed helper that looks up each generated token ID explicitly. It must raise `RuntimeError` instead of substituting zero when vLLM omits or corrupts a generated-token logprob.

- [ ] **Step 3: Use the helper in sync and async workers**

Remove the sync broad `except Exception` and the async silent zero fallback. Keep prompt/padding logprob positions zero; generated-token positions must be complete.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_utils.py -k logprob -q
```

Expected: all selected tests pass.

### Task 5: Log The Resolved Runtime Contract

**Files:**
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `.worktrees/nemorl-vllm024-upgrade/tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**
- Consumes: final vLLM keyword arguments after NeMo-RL resolution.
- Produces: one structured startup record for model, method, K, target TP, draft TP, load format, target CUDA-graph mode, rejection method, and draft sampling method, plus checked MTP disk-load results on every worker.

- [ ] **Step 1: Add a pure summary helper and failing tests**

Add a pure function returning a typed `dict[str, object]` from the final worker configuration. Tests must prove that absent sampling keys are reported as vLLM 0.24 defaults (`standard`, `greedy`) rather than omitted.

- [ ] **Step 2: Log once before engine creation**

Use the same helper for sync and async workers. Do not add per-token logging or new hot-path synchronization.

- [ ] **Step 3: Reject failed MTP disk loads**

Update sync and async `post_init` paths to inspect every `collective_rpc("load_mtp_weights_from_disk", ...)` result. Raise `RuntimeError` if any worker returns a value other than `True`; do not continue with an absent or partially initialized MTP drafter.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --group test pytest tests/unit/models/generation/test_vllm_generation.py -k 'runtime_contract or speculative' -q
```

Expected: all selected tests pass.

### Task 6: Correctness And Performance Gates

**Files:**
- Create: `.worktrees/nemorl-vllm024-upgrade/docs/design-docs/specdec-rl-validation-gates.md`
- Modify: `public/reports/specdec_rl_framework_lessons_and_nemorl_gaps_20260709.html`

**Interfaces:**
- Consumes: Tasks 1–5 and existing benchmark artifacts.
- Produces: explicit gates for future vLLM CUDA-graph and dynamic-K patches.

- [ ] **Step 1: Document parity gates**

Specify matched seeds/prompts and tests for target-token distribution, processed logprobs, reward distribution, output termination, and SpecDec on/off equivalence. Sampling tests cover temperature 0 and temperature 1/top-p 1.

- [ ] **Step 2: Document performance gates**

Require target and draft CUDA-graph hit/miss/fallback, `B*(K+1)` capture coverage, accepted tokens by position, draft/verify padded-token counts, local-argmax collective bytes for TP>1, generation throughput, E2E throughput, and matched-baseline step times.

- [ ] **Step 3: Record current evidence without overclaiming**

Label generic PARD CUDA-graph wiring and host-allocation costs as confirmed source gaps or hypotheses according to available evidence. State that PARD K12/K16 tail acceptance saturation makes high K inefficient even after graph coverage.

### Task 7: Final Verification And Publication

**Files:**
- Verify all files from Tasks 1–6.

**Interfaces:**
- Consumes: complete report and NeMo-RL worktree diff.
- Produces: reviewed commits and a published Pages pipeline.

- [ ] **Step 1: Merge the current upstream main**

Fetch `origin/main` and merge it into `sna/nemorl-vllm024-upgrade` without discarding the branch's vLLM 0.24 commits. Resolve conflicts by preserving upstream behavior plus the tested SpecDec lifecycle contract. Record the exact post-merge commit and vLLM dependency version.

- [ ] **Step 2: Run NeMo-RL tests and static checks**

Run focused pytest commands from Tasks 2–5, then run Ruff on changed Python files.

- [ ] **Step 3: Render the report locally**

Serve `public/` with `python3 -m http.server 8765 --directory public`, inspect desktop and mobile layouts, and verify external PR links and internal navigation.

- [ ] **Step 4: Request a whole-change code review**

Review lifecycle correctness, weight completeness, hot-path overhead, test coverage, and report claim provenance. Resolve all Critical and Important findings.

- [ ] **Step 5: Commit and push by repository boundary**

Commit only the explicit NeMo-RL files on `sna/nemorl-vllm024-upgrade`. Commit only the explicit Pages/report files in the dashboard repo. Push both branches before any new cluster submission.

- [ ] **Step 6: Verify Pages and prepare experiments**

Confirm the Pages pipeline contains the new report. Only then prepare matched CUDA-graph-on smoke jobs for lifecycle and parity validation; monitor each submitted job for at least five minutes.
