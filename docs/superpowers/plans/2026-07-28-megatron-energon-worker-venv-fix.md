# Megatron Energon Worker-Venv Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NeMo-RL's isolated MCore policy-worker environment install the Bridge dependencies required to import `MegatronPolicyWorker`, then rerun the actual NanoV3 baseline.

**Architecture:** Preserve NeMo-RL's workspace-dependency omission contract and add the missing eager Bridge import dependency directly to the `mcore` extra. Regenerate the frozen lock, verify the exact worker import boundary on Ptyche, and only then submit the unchanged four-node baseline.

**Tech Stack:** Python 3.13, uv, TOML, pytest, Ray, Slurm, NeMo-RL, Megatron-Bridge, Megatron Core.

## Global Constraints

- Limit source changes to NeMo-RL dependency metadata, lockfile, regression tests, and experiment evidence.
- Do not modify Megatron-Bridge, Megatron-LM, CUDA Graph scopes, or training semantics.
- Use the verified NanoV3 snapshot revision `97ab8012882a655dc38df4fee47422aca9caca07`.
- Keep checkpointing disabled and Hugging Face Hub access offline.
- Commit with sign-off, push, use a clean recursive Ptyche checkout, run `sbatch --test-only`, and monitor the submitted job for five minutes.

---

### Task 1: Lock the MCore worker dependency contract

**Files:**
- Modify: `tests/unit/test_megatron_bridge_provenance.py`
- Read: `pyproject.toml`
- Read: `tools/check_mbridge_deps.py`

**Interfaces:**
- Consumes: NeMo-RL's `project.optional-dependencies.mcore` declaration.
- Produces: a repository test requiring the isolated worker extra to install Bridge's eager Energon import dependency.

- [ ] **Step 1: Write the failing test**

Add a test equivalent to:

```python
def test_mcore_extra_includes_bridge_import_dependencies() -> None:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "megatron-energon[av-decode]~=7.0" in (
        root["project"]["optional-dependencies"]["mcore"]
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest --confcutdir=tests/unit -q \
  tests/unit/test_megatron_bridge_provenance.py::test_mcore_extra_includes_bridge_import_dependencies
```

Expected: failure because the MCore worker extra omits Energon.

- [ ] **Step 3: Add the missing worker dependency**

Add this literal to `project.optional-dependencies.mcore` next to the Bridge requirements:

```toml
"megatron-energon[av-decode]~=7.0",
```

- [ ] **Step 4: Run the target test and verify GREEN**

Run the same target test. Expected: one pass.

### Task 2: Regenerate and verify the frozen dependency graph

**Files:**
- Modify mechanically: `uv.lock`
- Modify: `tests/unit/test_megatron_bridge_provenance.py`

**Interfaces:**
- Consumes: the corrected `mcore` extra from Task 1.
- Produces: a frozen lock in which the selected `mcore` environment contains `megatron-energon`.

- [ ] **Step 1: Extend the lock provenance test**

Require a package named `megatron-energon` to exist in `lock["package"]`.

- [ ] **Step 2: Run the extended test and verify RED**

Run:

```bash
python3 -m pytest --confcutdir=tests/unit -q tests/unit/test_megatron_bridge_provenance.py
```

Expected: failure because the existing lock omits Energon.

- [ ] **Step 3: Regenerate the lock**

Run:

```bash
uv lock
```

Inspect `git diff --stat` and the `megatron-energon` lock record. Confirm the
additional transitive packages are required by Energon.

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
python3 -m pytest --confcutdir=tests/unit -q \
  tests/unit/test_megatron_bridge_provenance.py \
  tests/unit/test_uv_build_config.py \
  tests/unit/experiments/test_latestmain_nanov3_cg_matrix_scripts.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push**

Stage only `pyproject.toml`, `uv.lock`, and
`tests/unit/test_megatron_bridge_provenance.py`, then run:

```bash
git commit -s -m "fix: install Energon in MCore worker environments"
git push seonjinn HEAD:experiment/latestmain-pr5672-nano-matrix-20260727
```

### Task 3: Verify the real isolated worker boundary on Ptyche

**Files:**
- Reuse: `experiments/cuda_graph/latestmain_nanov3/run_ray_venv_bootstrap_smoke.sh`
- Reuse: `experiments/cuda_graph/latestmain_nanov3/profiles/ptyche.env`

**Interfaces:**
- Consumes: the pushed source and frozen lock from Task 2.
- Produces: Ptyche evidence that `megatron.energon` and `MegatronPolicyWorker` import from the same MCore worker environment used by the actual job.

- [ ] **Step 1: Update the clean recursive checkout**

Run `git pull --ff-only`, recursive submodule sync/update, confirm the exact
root/Bridge/MCore SHAs, and require an empty `git status --porcelain`.

- [ ] **Step 2: Run scheduler preflight**

Run the stored worker smoke launcher with `TEST_ONLY=1`, Ptyche, and the
schedulable backfill partition. Expected: scheduler acceptance.

- [ ] **Step 3: Submit the worker import smoke**

The smoke must import:

```python
import megatron.energon
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker
```

Monitor to a terminal state and require both imports to complete without
`ModuleNotFoundError`.

### Task 4: Resubmit the actual baseline and record evidence

**Files:**
- Reuse: `experiments/cuda_graph/latestmain_nanov3/scopes/00_nocg_baked_uv_cache.sh`
- Modify: `experiments/cuda_graph/results/latestmain_nanov3_cg_matrix_smoke.csv`
- Modify: `experiments/cuda_graph/results/latestmain_nanov3_cg_matrix_provenance.md`
- Regenerate: `experiments/cuda_graph/results/latestmain_nanov3_cg_matrix_report.html`

**Interfaces:**
- Consumes: the verified worker environment and local NanoV3 snapshot.
- Produces: a submitted 20-step no-CUDA-Graph baseline and a reproducible report entry.

- [ ] **Step 1: Run actual-job scheduler preflight**

Use `CLUSTER=ptyche`, `PARTITION_OVERRIDE=backfill`, `PHASE=performance`,
`STEPS=20`, `WANDB_MODE=offline`, and `TEST_ONLY=1`.

- [ ] **Step 2: Submit the unchanged actual baseline**

Run the same command without `TEST_ONLY=1`. Confirm the printed command contains
the local model/tokenizer snapshot, 20 steps, and checkpointing disabled.

- [ ] **Step 3: Monitor for five minutes**

Poll `squeue` and `sacct`. If running, inspect only bounded log tails and require
that the previous `megatron.energon` error does not recur.

- [ ] **Step 4: Update and verify the HTML report**

Record the new job ID/state/source SHA, render the report, validate the CSV has
16 columns per row, and run:

```bash
python3 -m pytest --confcutdir=tests/unit/experiments -q \
  tests/unit/experiments/test_latestmain_nanov3_cg_matrix_report.py
```

- [ ] **Step 5: Commit and push evidence**

Commit only the report CSV, provenance Markdown, and regenerated HTML with
sign-off, then push the experiment branch.
