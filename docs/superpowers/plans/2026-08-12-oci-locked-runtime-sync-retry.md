# OCI Locked Runtime Sync Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make content-addressed OCI runtime staging survive transient GitHub fetch failures while preserving the exact locked dependency and publication contracts.

**Architecture:** Keep retry control inside `validate_oci_container_runtime.sub`. A small Bash helper invokes the already-constructed `uv sync --locked` command at most three times against the same job-local cache, with fixed five- and ten-second delays; all later validation and atomic publication remain single-shot.

**Tech Stack:** Bash, uv `0.11.28`, pytest, SLURM, content-addressed runtime staging, deterministic HTML renderer.

## Global Constraints

- Every attempt uses the same `uv sync --locked` command and job-local `UV_CACHE_DIR`.
- Maximum attempts are exactly three, with delays of five and ten seconds.
- Retry behavior is not user configurable and does not alter the runtime stage key.
- Source, lockfile, dependency, container, Python, uv, and CUDA architecture identities remain unchanged.
- All three failures return the last sync status; the existing EXIT trap removes the incomplete stage and publication markers.
- Tests, imports, provenance verification, read-only conversion, and marker publication are never retried.
- Runtime staging remains CPU-only. GPU attestation runs only after a stage is `COMPLETED|0:0`.
- CUDA Graph warmup remains exactly three successful optimizer steps and checkpointing remains disabled.

---

### Task 1: Add bounded locked-sync retries

**Files:**
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`

**Interfaces:**
- Consumes: the existing Bash `sync_command` array and job-local `UV_CACHE_DIR`.
- Produces: `run_locked_uv_sync`, which returns on the first success or returns the third attempt's status.

- [x] **Step 1: Write failing behavior tests**

Add a fixture that evaluates the production helper with a fake uv executable.
The success case records two invocations after one injected failure and verifies
that a cache sentinel survives into attempt two. The failure case injects three
distinct nonzero statuses, verifies three invocations, and expects the third
status. Replace `sleep` only on the test process `PATH` so tests do not wait;
production delays remain fixed.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'locked_uv_sync'
```

Expected: both tests fail because `run_locked_uv_sync` does not exist.

- [x] **Step 3: Implement the minimal helper**

Define `run_locked_uv_sync` before the stage payload calls it. Iterate over
fixed delays `0 5 10`, execute `"${sync_command[@]}"`, return immediately on
success, print an attempt/delay diagnostic after attempts one and two, and
return the third command status without falling through `set -e`. Replace the
single `"${sync_command[@]}"` call with `run_locked_uv_sync`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: both retry behavior tests pass.

- [x] **Step 5: Document the bounded retry contract**

In the runtime staging section, state that locked environment sync is retried
at most three times inside one stage using one job-local cache. State that a
third failure publishes no stage or attestation evidence.

- [x] **Step 6: Run full local verification**

Run:

```bash
python3 -m pytest -q --confcutdir=tests/unit/experiments \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
bash -n \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/validate_oci_container_runtime.sub
git diff --check
```

Expected: the launcher suite passes, shell syntax passes, and no whitespace
errors are reported.

### Task 2: Update the maintained implementation explainer

**Files:**
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/explainer_context.json`
- Regenerate: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/cudagraph_implementation_explainer.html`

**Interfaces:**
- Consumes: terminal evidence from jobs `6076502` and `6077198` and the Task 1 diff.
- Produces: a validated HTML page that separates staging-network failures from CUDA Graph evidence.

- [x] **Step 1: Add concise problem and resolution context**

Record both job IDs, their exact lock-pinned failed commits, their CPU-only
allocation, and the bounded retry behavior. Do not classify either failure as
capture, replay, performance, or numerical-correctness evidence.

- [x] **Step 2: Regenerate and test the explainer**

Run:

```bash
python3 experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_explainer.py
python3 -m pytest -q --confcutdir=experiments/cuda_graph/nemotron_thd_te_graph_20260731 \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/test_render_explainer.py
python3 /Users/sna/.codex/skills/explain-diff-html/scripts/validate_explainer.py \
  /Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-main-20260806/experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/cudagraph_implementation_explainer.html
```

Expected: rendering succeeds, focused tests pass, and the validator reports
the six required sections and exactly five quiz questions.

### Task 3: Publish and rerun exact OCI validation

**Files:**
- Commit: the Task 1 and Task 2 files plus this plan.
- Local-only update after success: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/oci-hsg.env`

**Interfaces:**
- Consumes: the pushed NeMo-RL commit and unchanged Bridge `db5315d4d26737d9a124320eeca4bb3476af92e9`, MCore `c6fe36e784164b95cbfd0ee9dbf56d045fd6d70a`, TE `04a76c84423d9a4eb2f2010ef6692e347326cc00`, and container SHA256 `80f33fef2eac060bc54274446e4956d753687a54155d68be2059eccfba1e423d`.
- Produces: a completed read-only stage, a four-GPU runtime attestation, and validated MCore CUDA Graph diagnostics.

- [x] **Step 1: Commit and push**

Run repository verification, stage only named files, and commit with:

```bash
git commit -s -m "fix: retry locked OCI runtime sync"
git push seonjinn experiment/thd-cg-hybrid-nemotron-main-20260806
```

- [ ] **Step 2: Refresh remote source and scheduler preflight**

On OCI, `git pull --ff-only`, verify exact recursive submodule SHAs and zero
untracked/ignored runtime source paths, inspect FairShare, and run
`SBATCH_TEST_ONLY=1` for the CPU-only runtime stage on `cpu_datamover`.

- [ ] **Step 3: Submit and monitor runtime stage**

Submit one dependency-free stage, monitor at least five minutes and through
terminal state, and require `COMPLETED|0:0`, the exact stage marker, locked test
dependency identities, passing root tests, and read-only publication.

- [ ] **Step 4: Submit and monitor GPU attestation**

Use the completed stage job ID. Run `SBATCH_TEST_ONLY=1` immediately before the
actual four-GPU `batch` submission. Require the attestation JSON to bind the
exact container and all source/runtime identities.

- [ ] **Step 5: Run correctness gates before NeMo-RL performance**

Restore the direct-child private OCI profile with the new attestation. Submit
the coordinated Nano HybridEP release-vs-zero-grad diagnostics and independent
compact-versus-fixed THD numerical parity using `sbatch --test-only` followed
by dependency-free jobs. Only after those pass run the matched five-step
NeMo-RL baseline/attention smoke, then the 20-step performance and 100-step
accuracy comparisons.
