# Megatron Energon Worker-Venv Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NeMo-RL's isolated MCore policy-worker environment install the Bridge dependencies required to import `MegatronPolicyWorker`, then rerun the actual NanoV3 baseline.

**Architecture:** Treat the Bridge workspace proxy as the canonical dependency declaration and make the root uv static metadata identical to it. Regenerate the frozen lock, verify the exact worker import boundary on Ptyche, and only then submit the unchanged four-node baseline.

**Tech Stack:** Python 3.13, uv, TOML, pytest, Ray, Slurm, NeMo-RL, Megatron-Bridge, Megatron Core.

## Global Constraints

- Limit source changes to NeMo-RL dependency metadata, lockfile, regression tests, and experiment evidence.
- Do not modify Megatron-Bridge, Megatron-LM, CUDA Graph scopes, or training semantics.
- Use the verified NanoV3 snapshot revision `97ab8012882a655dc38df4fee47422aca9caca07`.
- Keep checkpointing disabled and Hugging Face Hub access offline.
- Commit with sign-off, push, use a clean recursive Ptyche checkout, run `sbatch --test-only`, and monitor the submitted job for five minutes.

---

### Task 1: Lock the Bridge metadata contract

**Files:**
- Modify: `tests/unit/test_megatron_bridge_provenance.py`
- Read: `pyproject.toml`
- Read: `3rdparty/Megatron-Bridge-workspace/setup.py`

**Interfaces:**
- Consumes: `CACHED_DEPENDENCIES: list[str]` assigned in the Bridge workspace proxy.
- Produces: a repository test that requires the root `tool.uv.dependency-metadata` entry for `megatron-bridge` to contain the identical dependency set.

- [ ] **Step 1: Write the failing test**

Add an AST helper that reads the literal `CACHED_DEPENDENCIES` assignment and a test equivalent to:

```python
def test_root_bridge_static_metadata_matches_workspace_proxy() -> None:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    bridge_metadata = next(
        item
        for item in root["tool"]["uv"]["dependency-metadata"]
        if item["name"] == "megatron-bridge"
    )
    assert set(bridge_metadata["requires-dist"]) == _bridge_cached_dependencies()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest --confcutdir=tests/unit -q \
  tests/unit/test_megatron_bridge_provenance.py::test_root_bridge_static_metadata_matches_workspace_proxy
```

Expected: failure showing `megatron-core[dev,mlm]` only on the workspace-proxy side.

- [ ] **Step 3: Add the missing canonical dependency**

Add this literal to the root static `megatron-bridge` `requires-dist` list next to the other Bridge requirements:

```toml
"megatron-core[dev,mlm]",
```

- [ ] **Step 4: Run the target test and verify GREEN**

Run the same target test. Expected: one pass.

### Task 2: Regenerate and verify the frozen dependency graph

**Files:**
- Modify mechanically: `uv.lock`
- Modify: `tests/unit/test_megatron_bridge_provenance.py`

**Interfaces:**
- Consumes: the corrected root static metadata from Task 1.
- Produces: a frozen lock in which the editable `megatron-bridge` depends on the `dev` and `mlm` MCore extras and the graph contains `megatron-energon`.

- [ ] **Step 1: Extend the lock provenance test**

Require the lock's `megatron-bridge` package metadata to contain:

```python
{
    "name": "megatron-core",
    "extras": ["dev", "mlm"],
}
```

and require a package named `megatron-energon` to exist in `lock["package"]`.

- [ ] **Step 2: Run the extended test and verify RED**

Run:

```bash
python3 -m pytest --confcutdir=tests/unit -q tests/unit/test_megatron_bridge_provenance.py
```

Expected: failure because the existing lock omits the Bridge-to-MCore extra edge.

- [ ] **Step 3: Regenerate the lock**

Run:

```bash
uv lock
```

Inspect `git diff --stat` and the `megatron-bridge`, `megatron-core`, and
`megatron-energon` lock records. Reject unrelated source changes.

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
git commit -s -m "fix: restore Bridge MCore worker dependencies"
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
