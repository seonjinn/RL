# vLLM 0.25.1 Drafter Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, submit, and report a reproducible NeMo-RL performance-recipe matrix for every applicable vLLM 0.25.1 speculative proposer on three Qwen3 targets.

**Architecture:** A typed Python module owns recipe metadata, proposer compatibility, Hydra overrides, and SLURM command construction. A small Bash entrypoint invokes that module to preserve the cluster-facing script workflow. A separate typed collector reads completed run metadata and logs and emits exact-baseline-matched CSV/Markdown summaries.

**Tech Stack:** Python 3.11+, dataclasses, argparse, pytest, Bash, Hydra overrides, SLURM, NeMo-RL, vLLM 0.25.1, W&B.

## Global Constraints

- Use official vLLM 0.25.1 from the existing branch.
- Use the unmodified official performance recipe for each model except step count, run outputs, logging, checkpoint saving, CUDA Graph mode, and SpecDec settings.
- Set CUDA Graph on with `enforce_eager=false`, `FULL_AND_PIECEWISE`, and native capture sizing.
- Keep temperature 1.0 and top-p 1.0 from the recipe.
- Disable checkpoint saving.
- Derive `--segment` from recipe topology and never add `--gres` on Lyris.
- Submit from a clean commit pushed only to `git@github-seonjinn:seonjinn/RL.git`.
- Run recursive submodule initialization before remote submission.
- Run `sbatch --test-only` and monitor each submitted family for at least five minutes.
- Compute final comparisons from steps 2-20 against an exact matched baseline.

---

### Task 1: Typed Matrix Resolution

**Files:**
- Create: `experiments/vllm_0251_drafter_matrix/matrix.py`
- Create: `tests/experiments/test_vllm_0251_drafter_matrix.py`

**Interfaces:**
- Produces: `resolve_run(model_key: str, variant_key: str, phase: str, cluster: str) -> ResolvedRun`
- Produces: `ResolvedRun.command_parts() -> tuple[str, ...]`
- Produces: `ResolvedRun.sbatch_parts() -> tuple[str, ...]`

- [ ] **Step 1: Write failing tests for all recipe topologies**

```python
@pytest.mark.parametrize(
    ("model_key", "nodes", "segment", "osl"),
    [("qwen30", 4, 4, 4096), ("qwen32", 4, 4, 4096), ("qwen235", 16, 16, 8192)],
)
def test_recipe_topology_is_authoritative(model_key, nodes, segment, osl):
    run = resolve_run(model_key, "baseline", "smoke2", "lyris")
    assert (run.recipe.nodes, run.recipe.segment, run.recipe.max_osl) == (nodes, segment, osl)
```

- [ ] **Step 2: Run the focused test and verify import failure**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_matrix.py`

Expected: FAIL because `experiments.vllm_0251_drafter_matrix.matrix` does not exist.

- [ ] **Step 3: Add typed recipe, cluster, phase, and proposer dataclasses**

Implement immutable `RecipeSpec`, `ClusterSpec`, `PhaseSpec`, `VariantSpec`, and
`ResolvedRun` records. Define exact entries for the three controlled recipes,
Lyris, and phases `smoke2`, `smoke5`, and `final20`.

- [ ] **Step 4: Add failing compatibility and override tests**

```python
def test_pard_selects_mrv1_and_parallel_drafting():
    run = resolve_run("qwen32", "pard_k5", "smoke2", "lyris")
    assert run.variant.runner == "mrv1"
    assert "++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true" in run.hydra_overrides

def test_eagle_selects_mrv2_without_compact_capture_sizes():
    run = resolve_run("qwen30", "eagle3_k3", "smoke2", "lyris")
    assert run.variant.runner == "mrv2"
    assert all("cudagraph_capture_sizes" not in item for item in run.hydra_overrides)
```

- [ ] **Step 5: Implement proposer resolution and fail-closed validation**

Define baseline, EAGLE3 K1/K3/K5, DFlash K3/K5, draft K1/K5, PARD K5/K16,
suffix K32, ngram K5, and ngram_gpu K5. Reject unavailable model/checkpoint
combinations before constructing a job.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_matrix.py`

Expected: all tests pass.

Commit: `git commit -s -m "feat: define vllm 0.25 drafter matrix"`

### Task 2: Submission Entrypoint And Provenance

**Files:**
- Create: `experiments/vllm_0251_drafter_matrix/submit_matrix.sh`
- Modify: `experiments/vllm_0251_drafter_matrix/matrix.py`
- Modify: `tests/experiments/test_vllm_0251_drafter_matrix.py`

**Interfaces:**
- Consumes: `resolve_run(...)`
- Produces: CLI subcommands `show`, `test-only`, and `submit`
- Produces: `<run_dir>/provenance.json` and `<run_dir>/provenance.txt`

- [ ] **Step 1: Write failing CLI tests**

Test that `show` emits a deterministic command and scheduler record, that
`test-only` includes `--test-only`, and that submission rejects a dirty or
unpushed checkout.

- [ ] **Step 2: Verify the new CLI tests fail**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_matrix.py -k cli`

Expected: FAIL because the CLI and provenance writer are absent.

- [ ] **Step 3: Implement CLI and atomic provenance output**

Use `subprocess.run(..., check=True)` with argument arrays, never shell command
concatenation. Record Git/submodule SHAs, container, snapshots, recipe, runner,
Hydra overrides, environment, and scheduler arguments before calling `sbatch`.

- [ ] **Step 4: Add the Bash wrapper**

```bash
#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/matrix.py" "$@"
```

- [ ] **Step 5: Verify unit and shell checks and commit**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_matrix.py`

Run: `bash -n experiments/vllm_0251_drafter_matrix/submit_matrix.sh`

Expected: both commands pass.

Commit: `git commit -s -m "feat: add drafter matrix submission workflow"`

### Task 3: Result Collection And Exact Baseline Matching

**Files:**
- Create: `experiments/vllm_0251_drafter_matrix/collect_results.py`
- Create: `tests/experiments/test_vllm_0251_drafter_results.py`
- Create: `experiments/vllm_0251_drafter_matrix/fixtures/sample_steps.jsonl`

**Interfaces:**
- Produces: `summarize_steps(rows: Iterable[StepRow], start: int = 2, end: int = 20) -> RunSummary`
- Produces: `match_baseline(candidate: RunSummary, baselines: Sequence[RunSummary]) -> RunSummary`
- Produces: CSV and Markdown report files.

- [ ] **Step 1: Write failing step-window tests**

Create a synthetic fixture with step 1 warmup and steps 2-20. Assert step 1 is
excluded and E2E, generation, policy, logprob, throughput, generation ratio,
acceptance, and mean accepted length are averaged independently.

- [ ] **Step 2: Run tests and verify missing collector failure**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_results.py`

Expected: FAIL because `collect_results.py` does not exist.

- [ ] **Step 3: Implement typed parsing and aggregation**

Reject incomplete windows by default and mark partial summaries only when an
explicit `--allow-partial` flag is supplied.

- [ ] **Step 4: Add failing exact-match tests**

Assert that a baseline mismatch in model, recipe, vLLM version, container,
cluster, temperature, top-p, max OSL, or CUDA Graph mode is rejected.

- [ ] **Step 5: Implement speedups and report rendering**

Report E2E and generation time speedups as `baseline_time / candidate_time`
and throughput speedups as `candidate_tps / baseline_tps`. Preserve job, log,
W&B, runner, and graph provenance.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_results.py`

Expected: all tests pass.

Commit: `git commit -s -m "feat: collect drafter matrix metrics"`

### Task 4: Runbook, Push, And Cluster Preflight

**Files:**
- Create: `experiments/vllm_0251_drafter_matrix/README.md`
- Create: `experiments/vllm_0251_drafter_matrix/REPORT.md`

**Interfaces:**
- Consumes: submission and collection CLIs.
- Produces: exact smoke/promotion commands and a live result ledger.

- [ ] **Step 1: Document the matrix and promotion gates**

List every model/variant, applicable/unsupported state, checkpoint source,
runner, candidate K, and the commands for `show`, `test-only`, `submit`, and
collection.

- [ ] **Step 2: Run the full focused verification suite**

Run: `pytest -q tests/experiments/test_vllm_0251_drafter_matrix.py tests/experiments/test_vllm_0251_drafter_results.py`

Run: `bash -n experiments/vllm_0251_drafter_matrix/submit_matrix.sh`

Run: `uv lock --check`

Expected: all commands pass.

- [ ] **Step 3: Commit and push only to the user fork**

Commit: `git commit -s -m "docs: add vllm 0.25 drafter matrix runbook"`

Push: `git push -u fork sna/nemorl-vllm0251-drafter-matrix-20260716`

- [ ] **Step 4: Sync Lyris and initialize recursive submodules**

Pull the pushed branch into a clean Lustre checkout and run
`git submodule update --init --recursive`. Verify the checkout commit and
submodule status before any scheduler command.

- [ ] **Step 5: Run scheduler preflight and submit two-step smokes**

Run `test-only` for each topology, then submit baseline plus applicable
variants without dependencies. Record job IDs in `REPORT.md` after pulling
result artifacts back locally.

- [ ] **Step 6: Monitor jobs and promote passing variants**

Monitor for at least five minutes. Classify every early failure from its log.
Submit five-step and then 20-step jobs only through the documented gates.
