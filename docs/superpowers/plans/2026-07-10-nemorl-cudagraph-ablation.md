# NeMo-RL SpecDec CUDA Graph Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CUDA-graph-off launcher mode and run the six matched Qwen3-32B 20-step NeMo-RL ablations defined in the approved design.

**Architecture:** Extend the existing tail-gated launcher with one explicit `CUDA_GRAPH_MODE=on|off` boundary. Graph-on remains the current default; graph-off selects eager execution and omits every graph compilation/capture override. Extend manifest and summary provenance so within-mode SpecDec speedups and same-variant graph-on/off deltas cannot be confused.

**Tech Stack:** Bash, Python 3.13, pytest, NeMo-RL, vLLM 0.24, W&B, SLURM/pyxis on Lyris GB200.

## Global Constraints

- Preserve the six matched rows and exact Qwen3-32B performance-recipe configuration from `docs/superpowers/specs/2026-07-10-nemorl-cudagraph-ablation-design.md`.
- Keep `CUDA_GRAPH_MODE=on` behavior unchanged.
- In graph-off mode set `enforce_eager=true` and emit no graph mode, capture maximum, capture-size, or graph-dispatch metric override.
- Record graph-off provenance as `cuda_graph_enabled=false`, `enforce_eager=true`, `graph_mode=NONE`, with capture fields `not_applicable`.
- Submit from a separate Lyris checkout; do not update the checkout used by running graph-on jobs.
- Use `--segment=4`, no `--gres`, perform `sbatch --test-only`, and monitor every job for at least five minutes.

---

### Task 1: Launcher CUDA Graph Mode

**Files:**
- Modify: `tests/test_vllm_024_tail_gate_launch.py`
- Modify: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`

**Interfaces:**
- Consumes: environment variable `CUDA_GRAPH_MODE`, default `on`.
- Produces: resolved shell values `cuda_graph_enabled`, `enforce_eager`, and `effective_graph_mode` used by command construction and manifest output.

- [ ] **Step 1: Write failing launcher tests**

Add tests that dry-run `baseline_v2`, `always_on_v2_k5`, and `fastrl_threshold_v2_k5` with `CUDA_GRAPH_MODE=off` and assert:

```python
assert "policy.generation.vllm_cfg.enforce_eager=true" in output
assert "compilation_config.cudagraph_mode=" not in output
assert "compilation_config.max_cudagraph_capture_size=" not in output
assert "compilation_config.cudagraph_capture_sizes=" not in output
assert "NRL_VLLM_ENABLE_CUDAGRAPH_DISPATCH_METRICS" not in output
```

Add a parametrized rejection test for `CUDA_GRAPH_MODE=invalid`, and retain the existing graph-on assertions without passing the new variable.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_vllm_024_tail_gate_launch.py \
  -k 'cuda_graph_mode or all_cohorts_enable_cuda_graph'
```

Expected: the off-mode tests fail because the launcher still hardcodes `enforce_eager=false` and graph overrides.

- [ ] **Step 3: Implement the launcher boundary**

Validate near the existing environment defaults:

```bash
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-on}"
case "${CUDA_GRAPH_MODE}" in
  on|off) ;;
  *) echo "ERROR: CUDA_GRAPH_MODE must be on or off" >&2; exit 2 ;;
esac
```

Resolve per job:

```bash
local cuda_graph_enabled=true
local enforce_eager=false
local effective_graph_mode="${graph_mode}"
if [[ "${CUDA_GRAPH_MODE}" == "off" ]]; then
  cuda_graph_enabled=false
  enforce_eager=true
  effective_graph_mode=NONE
fi
```

Use `enforce_eager` in the common overrides. Add dispatch metrics, compilation mode, and V2 capture geometry only when `cuda_graph_enabled=true`. Keep all SpecDec and sampling overrides independent of graph mode.

- [ ] **Step 4: Verify GREEN**

Run the focused tests from Step 2 and expect all selected tests to pass.

- [ ] **Step 5: Commit launcher behavior**

```bash
git add \
  experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh \
  tests/test_vllm_024_tail_gate_launch.py
git commit -s -m "feat(vllm): add CUDA graph ablation mode"
```

### Task 2: Manifest and Summary Provenance

**Files:**
- Modify: `tests/test_vllm_024_tail_gate_launch.py`
- Modify: `tests/test_vllm_024_tail_gate_summary.py`
- Modify: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`
- Modify: `experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py`

**Interfaces:**
- Consumes: Task 1 values `cuda_graph_enabled`, `enforce_eager`, and `effective_graph_mode`.
- Produces: manifest fields with those exact names and summary cohort matching that includes CUDA graph state.

- [ ] **Step 1: Write failing manifest tests**

Extend submit test expectations so a graph-off row contains:

```python
assert row["cuda_graph_enabled"] == "false"
assert row["enforce_eager"] == "true"
assert row["graph_mode"] == "NONE"
assert row["cudagraph_max_requests"] == "not_applicable"
assert row["cudagraph_max_tokens"] == "not_applicable"
assert row["cudagraph_capture_sizes"] == "not_applicable"
```

Add one graph-on assertion proving the existing values remain `true`, `false`, and the runner-specific graph mode.

- [ ] **Step 2: Write failing comparison tests**

Create otherwise-identical summary rows with opposite `cuda_graph_enabled` values. Assert normal baseline matching rejects them, while a dedicated same-variant graph-mode key can pair them only when every non-graph cohort field and variant matches.

- [ ] **Step 3: Verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_summary.py \
  -k 'manifest or graph_mode or cuda_graph'
```

Expected: failures identify missing manifest columns and missing graph-state cohort separation.

- [ ] **Step 4: Implement provenance and matching**

Add `cuda_graph_enabled` and `enforce_eager` to the manifest header and values. Use the resolved `effective_graph_mode`, and force all capture fields to `not_applicable` when graphs are off.

Add graph state to the normal comparison cohort fields. Add a same-variant graph-ablation pairing function that removes only graph-specific fields from its equality key and requires exact variant equality. It must not relax runner, checkpoint, recipe, topology, length, sampling, or gate-policy matching.

- [ ] **Step 5: Verify GREEN**

Run the tests from Step 3 and expect all selected tests to pass.

- [ ] **Step 6: Commit provenance support**

```bash
git add \
  experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh \
  experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_summary.py
git commit -s -m "feat(report): separate CUDA graph ablation cohorts"
```

### Task 3: Verification and Lyris Submission

**Files:**
- Verify: `experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh`
- Verify: `experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py`
- Create remotely: `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-tail-gating-cgoff-<commit>`
- Create remotely: `/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/vllm024-tail-gated/qwen32b-cgoff-step20-20260710/`

**Interfaces:**
- Consumes: committed Task 1 and Task 2 behavior.
- Produces: six SLURM job IDs, six W&B URLs, and a complete immutable submissions manifest.

- [ ] **Step 1: Run full local verification**

```bash
.venv/bin/pytest -q \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_summary.py \
  tests/test_vllm_024_tail_gate_mini_sync_grpo.py \
  tests/test_vllm_024_source_contract.py
ruff check \
  experiments/vllm_024_upgrade/summarize_tail_gated_specdec.py \
  tests/test_vllm_024_tail_gate_launch.py \
  tests/test_vllm_024_tail_gate_summary.py
bash -n experiments/vllm_024_upgrade/submit_tail_gated_specdec_step20.sh
git diff --check
```

Expected: all tests pass, Ruff and shell syntax succeed, and no whitespace errors are reported.

- [ ] **Step 2: Push the implementation commit**

```bash
git push fork sna/nemorl-vllm024-tail-gating
```

- [ ] **Step 3: Create a separate remote checkout**

On Lyris, fetch the fork branch into a new checkout named with the exact commit. Initialize every submodule recursively and verify `git status --short` is empty. Do not modify `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-tail-gating-97a206ae`.

- [ ] **Step 4: Run scheduler preflight**

For each of the six variants, run the launcher's `test-only` path with:

```bash
CUDA_GRAPH_MODE=off
MAX_STEPS=20
DRAFT_SAMPLE_METHOD=probabilistic
RUN_TAG=qwen32b-cgoff-step20-20260710
WANDB_PROJECT=nemorl-vllm024-cgoff-step20-lyris
```

Use threshold 64 and three consecutive checks for the V2 gated row. Verify Lyris reports schedulable nodes, `segment=4`, and no `--gres`.

- [ ] **Step 5: Submit all six rows**

Submit V2 `baseline_v2`, `always_on_v2_k5`, and `fastrl_threshold_v2_k5`, plus V1 `baseline_v1`, `always_on_v1_k5`, and `stock_dynamic_v1`. Give every job a unique attempt ID. Confirm every manifest row records graph-off provenance and a valid W&B URL.

- [ ] **Step 6: Monitor for five minutes**

Check `squeue`, `sacct`, and bounded driver-log searches for each job. Require all six to remain RUNNING or to have progressed normally, with no OOM, NCCL timeout, traceback, configuration contradiction, missing drafter checkpoint, or absent W&B initialization.

- [ ] **Step 7: Record live status**

Report job IDs, current steps, W&B links, and the exact graph-on references. Do not publish speedups until both members of a comparison have matching completed steps; label any interim averages with their included step set.
