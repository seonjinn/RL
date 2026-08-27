# Qwen3-30B-A3B Cadence Parallel Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Q30 cadence result pipeline, prepare balanced Adaptive-v2 inputs, and document the remaining PR1-11 runtime validation surface.

**Architecture:** Three file-disjoint deliverables live beside the existing 200-step experiment. Reporting separates pure history aggregation from W&B I/O and HTML rendering; Adaptive-v2 reuses the matched fixed-10 workload and changes only its typed schedule; the validation matrix is a source-grounded human document with explicit evidence grades.

**Tech Stack:** Python 3.11, unittest/pytest, Pydantic/OmegaConf config composition, W&B public API, JSON, self-contained HTML, Markdown.

**Spec:** `docs/superpowers/specs/2026-08-26-q30-parallel-followups-design.md`

## Global Constraints

- Do not modify or resubmit jobs `6575002`, `6575004`, `6575005`, `6575010`, `6575013`, or `6575023`.
- Keep submitted product SHA `1be8237816bfd78dad752dd5c1e0149ae2420301` and remote harness SHA `6c51b26dc531a7b0b1ca88b9d0f02c882d2c8664` unchanged.
- Aggregate W&B steps 3-200 as a closed interval and expose included steps, missing steps, and per-metric valid counts.
- Compare always/fixed-10 only with the matched static drafter of the same DFlash or DSpark family.
- Never place `WANDB_API_KEY` in source, commands, fixtures, output JSON, HTML, or logs.
- Adaptive-v2 changes only `policy.draft.update_schedule`; all matched workload fields remain identical to fixed-10.
- Documentation must distinguish code/config support from GPU runtime evidence.
- Use signed commits and modify only the files assigned to each task.

---

### Task 0: OCI-HSG CPU Detection Regression

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`

**Interfaces:**
- Consumes: `ray.sub`'s documented `CPUS_PER_WORKER` override.
- Produces: rendered sbatch jobs that export `CPUS_PER_WORKER=64` before invoking `ray.sub`, avoiding scheduler-client auto-detection on OCI-HSG compute nodes.

- [ ] **Step 1: Write the failing rendered-job regression test**

  Extend the existing rendered-job contract to assert the exact line
  `export CPUS_PER_WORKER=64`. The production mutation caught is omission of
  the override, which re-enters unavailable `scontrol` CPU discovery.

- [ ] **Step 2: Verify RED**

  Run the focused rendered-job test and confirm it fails because the export is
  absent from the generated sbatch, not because of fixture or dependency errors.

- [ ] **Step 3: Implement the minimal override**

  Add only `export CPUS_PER_WORKER=64` beside `GPUS_PER_NODE=4` in the rendered
  sbatch environment. Do not modify `ray.sub` or add fallback logic.

- [ ] **Step 4: Verify GREEN**

  Run the existing experiment contract tests, Bash syntax, Ruff, and
  `git diff --check`.

- [ ] **Step 5: Commit**

  ```bash
  git add experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py
  git commit -s -m "fix(experiment): pin OCI-HSG Ray worker CPUs"
  ```

### Task 0b: Harness-Revision Retry Identity

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`

**Interfaces:**
- Consumes: product SHA, harness SHA, and cadence variant.
- Produces: a manifest-visible submission-record path keyed by all three values, preserving old failed-attempt receipts while allowing one fail-closed retry for a corrected harness revision.

- [ ] **Step 1: Write a failing manifest contract test**
- [ ] **Step 2: Verify RED because `submission_record` is absent**
- [ ] **Step 3: Add one shared record-path helper and use it in manifest and submit modes**
- [ ] **Step 4: Run full experiment contracts, Bash syntax, Ruff, and `git diff --check`**
- [ ] **Step 5: Commit with sign-off without deleting prior receipts**

### Task 1: W&B Result Collector and HTML Report

**Files:**
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/cadence_report.py`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/test_cadence_report.py`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/fixtures/history.json`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/README.md`

**Interfaces:**
- Consumes: W&B run metadata and history for project/group named in the spec, or an equivalent JSON fixture for offline verification.
- Produces: `aggregate_history(rows: Sequence[Mapping[str, object]], start_step: int, end_step: int) -> dict[str, object]`, `build_comparisons(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]`, and `render_html(report: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write failing aggregation tests**

  Add literal fixtures proving the 3-200 closed interval, per-metric null omission, missing-step disclosure, canonical logged throughput use, and acceptance-key aliases. Name the production mutation each test catches.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  python3 -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests/test_cadence_report.py
  ```

  Expected: import failure because `cadence_report.py` does not exist.

- [ ] **Step 3: Implement minimal pure aggregation and comparison functions**

  Use arithmetic means over finite/non-null observations. Throughput comes only
  from W&B throughput keys. Match `dflash-always` and `dflash-fixed10` to
  `dflash-static`, and the corresponding DSpark rows to `dspark-static`.

- [ ] **Step 4: Add failing HTML and CLI tests**

  Assert that incomplete runs are labelled `preliminary`, unmatched rows say
  `waiting static baseline`, output includes both generation and E2E metrics,
  and a supplied sentinel API key never appears in output.

- [ ] **Step 5: Implement W&B collection and self-contained HTML rendering**

  The CLI accepts `--entity`, `--project`, `--group`, `--json-output`, and
  `--html-output`. Import W&B only inside the online collection path. Use
  `scan_history(min_step=3, max_step=201)` and atomic output replacement.

- [ ] **Step 6: Verify GREEN and document exact commands**

  Run the focused pytest file, Ruff, and `python3 -m py_compile`. Document
  offline fixture and online W&B commands without embedding credentials.

- [ ] **Step 7: Commit**

  ```bash
  git add experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting
  git commit -s -m "feat(experiment): add q30 cadence result report"
  ```

### Task 2: Adaptive-v2 Matched Inputs

**Files:**
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dflash-adaptive-v2.yaml`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs/dspark-adaptive-v2.yaml`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_adaptive_v2_contract.py`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/verify_composed_configs.py`

**Interfaces:**
- Consumes: matched fixed-10 configs and `AdaptiveDraftUpdateScheduleConfig`.
- Produces: two Linux-composable 200-step configs whose only semantic difference from their fixed-10 counterparts is the exact Adaptive-v2 schedule in the spec.

- [ ] **Step 1: Write failing matched-config tests**

  Load each fixed-10 config and its expected Adaptive-v2 path, normalize only
  `policy.draft.update_schedule`, and assert the remaining structures are equal.
  Assert the exact schedule literal and successful Pydantic validation.

- [ ] **Step 2: Verify RED**

  Run the new focused pytest file. Expected: failure because both Adaptive-v2
  files are absent.

- [ ] **Step 3: Create the two minimal config files**

  Copy the matched family config and replace only `update_schedule` with:

  ```python
  {
      "mode": "adaptive",
      "action": "sparse_update",
      "min_interval": 10,
      "max_interval": 40,
      "ewma_alpha": 0.2,
      "degradation_threshold": 0.03,
      "recovery_threshold": 0.01,
      "min_observations": 10,
      "max_burst_updates": 2,
  }
  ```

- [ ] **Step 4: Extend composition verification without hidden defaults**

  Make the verifier accept `adaptive-v2` as an expected schedule while reading
  every value directly from the composed typed config.

- [ ] **Step 5: Verify GREEN**

  Run the new and existing experiment contract tests, Ruff, YAML parsing, and
  `verify_composed_configs.py` in the pinned Linux/container environment when
  available. Record local dependency limitations rather than weakening checks.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/qwen3_30ba3b_draft_cadence_200step_20260826/configs experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests experiments/qwen3_30ba3b_draft_cadence_200step_20260826/verify_composed_configs.py
  git commit -s -m "feat(experiment): add q30 adaptive v2 inputs"
  ```

### Task 3: PR1-11 Validation Matrix

**Files:**
- Create: `docs/validation/pr1_11_draft_online_training_matrix.md`

**Interfaces:**
- Consumes: current code, focused tests, existing experiment receipts, job IDs, W&B links, and prior audit evidence.
- Produces: one reviewable table with rows for DFlash and DSpark and columns for CP1 packed, CP>1 unpacked, CP>1 packed, repeated update/refit, multi-node, evidence grade, evidence pointer, and next validation action.

- [ ] **Step 1: Audit concrete evidence**

  Inspect current implementation and bounded experiment artifacts. Do not query
  broad scheduler state, submit jobs, or infer runtime support from class flags.

- [ ] **Step 2: Write the matrix**

  Grade every cell as `code`, `unit`, `composed`, `scheduled`, or `runtime` and
  attach a commit, test/file path, job ID, or W&B link. Distinguish PR ownership
  from cross-PR validation tracked by issues such as #3750/#3757.

- [ ] **Step 3: Self-review claims**

  For every `runtime` cell, confirm a concrete job crossed the stated gate. Any
  unsupported or ambiguous claim is downgraded and listed in `Remaining gates`.

- [ ] **Step 4: Commit**

  ```bash
  git add docs/validation/pr1_11_draft_online_training_matrix.md
  git commit -s -m "docs: add draft online training validation matrix"
  ```

### Task 4: Integration Verification

**Files:**
- Modify only if a review finding requires it: files created by Tasks 1-3.

**Interfaces:**
- Consumes: the three independently reviewed commits.
- Produces: a clean branch with all focused tests passing and no changes to submitted job inputs.

- [ ] **Step 1: Review each task diff for spec compliance and code quality**
- [ ] **Step 2: Run all experiment-focused tests, Ruff, format checks, YAML parsing, and `git diff --check`**
- [ ] **Step 3: Confirm submitted remote product/harness SHAs and six job records remain unchanged**
- [ ] **Step 4: Commit only review fixes, with sign-off, then push the isolated branch**
