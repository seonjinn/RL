# Qwen3-30B-A3B Cadence Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the DFlash/DSpark 200-step fixed-cadence study without the false fixed-20 gate failure or DSpark's Blackwell FAP illegal-memory access.

**Architecture:** Keep the official NeMo-RL performance configuration and stock vLLM 0.25.1 control intact. Move long measurements to `batch_long`, make the first-refit gate process-liveness based, and inject the complete upstream #48167 runtime correction through a source-verified node-local overlay.

**Tech Stack:** Bash, Python 3.12, pytest, SLURM/Pyxis, NeMo-RL, vLLM 0.25.1, Ray, GB200 CUDA Graphs

**Spec:** `docs/superpowers/specs/2026-08-28-q30-cadence-recovery.md`

## Global Constraints

- Preserve the official Qwen3-30B-A3B performance recipe and FAP CUDA Graph mode.
- Keep checkpointing disabled in performance measurements.
- Apply the vLLM correction only to DSpark and only through a node-local overlay.
- Commit and push before submission; pull the exact commit remotely before `sbatch`.
- Use `sbatch --test-only` and monitor a running canary for at least five minutes.

---

### Task 1: Long-run and gate contract

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/README.md`

**Interfaces:**
- Consumes: `wait_for_gate(pattern, marker, timeout_seconds)` in rendered `driver.sh`.
- Produces: an 18-hour `batch_long` job and an unbounded-by-clock first-refit liveness gate.

- [ ] **Step 1: Write the failing contract assertions**

Assert that manifests and rendered jobs use `batch_long`/`18:00:00`, that
startup gates pass `2700`, and that `DRAFT_REFIT_GATE_PASS` passes `0`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`

Expected: failure because the current harness hard-codes `batch`, four hours,
and one 2700-second timeout for every gate.

- [ ] **Step 3: Implement the minimal harness change**

Give `wait_for_gate` a third timeout argument. Enforce the deadline only when
the argument is greater than zero, pass `2700` to startup/step gates, and pass
`0` to the first-refit gate. Change both manifest and SBATCH resource identity
to `batch_long` and `18:00:00`.

- [ ] **Step 4: Run the focused test and shell parser**

Run: `python -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_contract.py`

Run: `bash -n experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`

Expected: PASS.

### Task 2: Source-verified complete vLLM #48167 runtime overlay

**Files:**
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/prepare_vllm_dspark_fap_overlay.py`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/patches/vllm-0.25.1-pr48167-runtime.patch`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_prepare_vllm_dspark_fap_overlay.py`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh`
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/README.md`

**Interfaces:**
- Consumes: installed `vllm/v1/attention/backends/flashinfer.py` and a writable overlay root.
- Produces: `prepare_overlay(source_package: Path, overlay_root: Path, patch_path: Path, *, expected_patch_sha256: str) -> Path` containing an exact copy with all ten runtime-file changes and a JSON provenance receipt.

- [ ] **Step 1: Write failing real-filesystem tests**

Cover complete forward application, idempotent already-patched input, source
drift, and patch-digest drift. Assert that the source tree is unchanged and the
receipt includes the upstream PR, patch digest, and every patched-file digest.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests/test_prepare_vllm_dspark_fap_overlay.py`

Expected: import failure because the overlay module does not exist.

- [ ] **Step 3: Implement the exact fail-closed overlay**

Copy the installed `vllm` package with `shutil.copytree`, validate the pinned
runtime patch digest, require `git apply --check` in the forward or reverse
direction, apply the complete patch, and atomically write provenance JSON.

- [ ] **Step 4: Wire the overlay into DSpark node setup only**

Copy the helper into each rendered artifact. Use the targeted venv post-sync
hook only for the synchronous DSpark generation worker, after its vLLM venv is
materialized. Create `/raid/scratch/.../vllm-overlay`, prepend it to
`PYTHONPATH`, and verify the digest-bound receipt after CUDA Graph capture.
Leave DFlash without the hook or any base-environment vLLM import.

- [ ] **Step 5: Run focused and full experiment tests**

Run: `python -m pytest -q experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests`

Expected: PASS.

### Task 3: Reproducible canary and long-run submission

**Files:**
- Modify: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/README.md`
- Create: `experiments/qwen3_30ba3b_draft_cadence_200step_20260826/report/README.md`

**Interfaces:**
- Consumes: committed harness and exact remote product SHA.
- Produces: scheduler receipts, DSpark canary evidence, and accepted 200-step job IDs.

- [ ] **Step 1: Run local validation**

Run the experiment pytest suites, `ruff check`, `bash -n`, and `git diff --check`.

- [ ] **Step 2: Commit and push**

Stage only the plan, spec, experiment helper, tests, harness, and documentation;
commit with `git commit -s`, then push the feature branch.

- [ ] **Step 3: Synchronize and preflight remotely**

Pull with `git pull --ff-only`, verify the exact HEAD, run checkpoint state-dict
preflight, and execute `sbatch --test-only` for DFlash fixed-20 and DSpark fixed-5.

- [ ] **Step 4: Submit and monitor the DSpark canary**

Submit DSpark fixed-5, wait for RUNNING, then inspect one filtered scheduler
query and its artifact markers after at least five minutes. Require CUDA Graph,
step 1, and step 2 gates with no CUDA illegal-memory access.

- [ ] **Step 5: Submit the 200-step matrix**

After the canary passes, submit DFlash fixed-20 and the corrected DSpark
fixed-5/10/20 arms. Record job IDs, branch SHA, source SHA, container path, and
artifact paths in the experiment report.
