# HybridEP-Default MoE 8-GPU Recipes and CW A/B Implementation Plan

> **For Codex:** Execute this plan task by task with test-first changes and
> fresh verification before each commit or success claim.

**Goal:** Make HybridEP the YAML default for every MoE `*n8g*.yaml`
performance recipe, preserve exact all-to-all comparison recipes, keep
`nemo_rl/utils/venvs.py` unchanged from the pre-cache-isolation base, submit
matched CW-DFW H100 experiments, and publish per-workload Policy, LogProb, and
end-to-end time and throughput.

**Architecture:** Build a parallel all-to-all inheritance tree from the
pre-change YAMLs. Each existing canonical `8g` filename becomes a minimal
HybridEP child of its exact `-alltoall.yaml` snapshot. Rebase 4-GPU children
onto all-to-all parents so x86 NVL8 variables do not leak into GB200 recipes.
Reuse the existing immutable SM90 DeepEP wheel, prebuilt Ray environment,
launcher, and static Pages report. Store structured workload metrics in one
machine-readable JSON artifact and render HTML from it.

**Tech Stack:** OmegaConf/Hydra YAML inheritance, pytest, Bash, uv, Ray,
SLURM/pyxis, DeepEP f725, Megatron-Core, Python JSON/HTML reporting.

---

## Task 1: Add failing recipe-contract tests

**Files:**

- Create:
  `tests/unit/tools/test_hybridep_default_8g_recipes.py`
- Modify:
  `tests/unit/tools/test_hybridep_x86_contract.py`

1. Add a table of all thirteen MoE `8g` canonical recipes and their expected
   `-alltoall.yaml` peers.
2. Resolve each canonical recipe and assert:
   `flex`, `hybridep`, 32 SMs, NVLink domain 8, eight HybridEP ranks,
   combine chunk 128, and `USE_MNNVL=0`.
3. Resolve each baseline and assert `alltoall` with no HybridEP backend, SM
   count, or x86 HybridEP environment keys.
4. Strip only the dispatcher/backend/SM/x86 environment fields and assert the
   remaining resolved configs are identical within every pair.
5. Assert dense Qwen3-32B and Llama `8g` recipes do not select HybridEP.
6. Assert the four affected `4g` descendants retain their pre-change
   dispatcher contract and no x86 HybridEP environment keys.
7. Replace the obsolete single-Qwen overlay contract and actor-cache test in
   `test_hybridep_x86_contract.py` with a source assertion that
   `create_local_venv` does not assign a per-actor `UV_CACHE_DIR`.
8. Run the focused tests and confirm they fail because the new baseline files
   and defaults do not exist yet:

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_default_8g_recipes.py \
  tests/unit/tools/test_hybridep_x86_contract.py
```

## Task 2: Build the all-to-all and HybridEP recipe trees

**Files:**

- Create thirteen files under
  `examples/configs/recipes/llm/performance/`:
  - `grpo-deepseek-v3-32n8g-alltoall.yaml`
  - `grpo-deepseek-v3-64n8g-alltoall.yaml`
  - `grpo-deepseek-v3-64n8g-async-1off-alltoall.yaml`
  - `grpo-deepseek-v3-64n8g-fp8-async-1off-alltoall.yaml`
  - `grpo-nemotron3-super-120BA12B-32n8g-alltoall.yaml`
  - `grpo-nemotron3-super-120BA12B-32n8g-async-1off-alltoall.yaml`
  - `grpo-qwen3-235b-16n8g-alltoall.yaml`
  - `grpo-qwen3-235b-32n8g-alltoall.yaml`
  - `grpo-qwen3-235b-32n8g-async-1off-alltoall.yaml`
  - `grpo-qwen3-30ba3b-4n8g-alltoall.yaml`
  - `grpo-qwen3-30ba3b-4n8g-async-1off-alltoall.yaml`
  - `grpo-qwen3-30ba3b-24n8g-async-8off-alltoall.yaml`
  - `grpo-qwen3-30ba3b-4n8g-40K-alltoall.yaml`
- Modify the corresponding thirteen canonical recipes.
- Delete:
  `grpo-qwen3-30ba3b-4n8g-hybridep.yaml`
- Modify four 4-GPU descendants:
  - `grpo-deepseek-v3-32n4g.yaml`
  - `grpo-deepseek-v3-64n4g-async-1off.yaml`
  - `grpo-qwen3-235b-16n4g.yaml`
  - `grpo-qwen3-235b-32n4g-async-1off.yaml`

1. Move each pre-change canonical body into its all-to-all peer.
2. In all-to-all descendants, replace references to canonical parents with
   the matching all-to-all parent.
3. Replace each canonical recipe with a minimal child of its exact all-to-all
   peer and add only the common HybridEP x86 NVL8 block.
4. Preserve model-specific `env_vars` by OmegaConf deep merge.
5. Repoint the four 4-GPU descendants to all-to-all 8-GPU parents without
   changing any other field.
6. Run the focused tests and require both files to pass:

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_default_8g_recipes.py \
  tests/unit/tools/test_hybridep_x86_contract.py
```

## Task 3: Remove the `venvs.py` cache-isolation change

**Files:**

- Modify: `nemo_rl/utils/venvs.py`
- Modify: `tests/unit/tools/test_hybridep_x86_contract.py`

1. Remove only:

```python
env["UV_CACHE_DIR"] = os.path.join(venv_path, ".uv-cache")
```

2. Remove the old actor-cache behavior test and its now-unused imports.
3. Verify no diff remains versus the selected x86 branch base:

```bash
git diff 4c14b04266a0b3ed8ec6121fae387d77d869bf1d \
  -- nemo_rl/utils/venvs.py
```

4. Re-run the focused tests.

## Task 4: Add reusable CW model profiles

**Files:**

- Modify:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86.env`
- Modify:
  `scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-hybridep.env`
- Create matched `*-x86.env` profile pairs for Qwen3-235B, Nemotron3 Super,
  and DeepSeek-V3 under the same directory.
- Modify:
  `scripts/experiments/oci-hsg/hybridep/README.md`
- Modify:
  `scripts/experiments/x86/hybridep/README.md`
- Modify:
  `tests/unit/tools/test_hybridep_x86_contract.py`

1. Point every baseline profile to a `-alltoall.yaml` recipe.
2. Point every HybridEP profile to the canonical recipe.
3. Set `DISPATCHER_MODE=recipe`,
   `NRL_FORCE_REBUILD_VENVS=false`, identical topology/step/time settings, and
   the exact f725 DeepEP commit in both arms.
4. Document shared driver/Ray environment preparation and forbid concurrent
   actor source rebuilds for matched runs.
5. Add profile-pair contract tests for recipe selection, dispatcher mode,
   node/GPU/segment shape, steps, DeepEP commit, and `NCCL_NVLS_ENABLE=0`.
6. Run:

```bash
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
bash -n scripts/experiments/x86/hybridep/submit_driver_venv.sh
bash -n scripts/experiments/x86/hybridep/prepare_driver_venv.sbatch
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/tools/test_hybridep_default_8g_recipes.py \
  tests/unit/tools/test_hybridep_x86_contract.py \
  tests/unit/tools/test_hybridep_submit_grpo.py
```

## Task 5: Validate all source changes and push

**Files:**

- All files from Tasks 1–4.

1. Validate every changed recipe with `tools/config_cli.py expand` and
   `minimize-check`.
2. Run configuration validation:

```bash
/private/tmp/nemo-rl-hybridep-x86-tests-20260728/bin/python -m pytest -q \
  tests/unit/test_config_validation.py \
  tests/unit/tools/test_hybridep_default_8g_recipes.py \
  tests/unit/tools/test_hybridep_x86_contract.py \
  tests/unit/tools/test_hybridep_submit_grpo.py \
  tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py
uv lock --check
git diff --check
git status --short
```

3. Inspect the complete diff, commit only scoped files with `git commit -s`,
   and push `sna/hybridep-x86-b200-h100-20260728` to `fork`.
4. Record the exact pushed SHA.

## Task 6: Add structured workload metrics and HTML generation

**Files in `/Users/sna/nemo-rl_release_perf_investigator`:**

- Create:
  `public/hybridep-x86-20260728/workload-metrics.json`
- Create:
  `public/scripts/hybridep_x86/collect_workload_metrics.py`
- Create:
  `public/scripts/hybridep_x86/render_workload_report.py`
- Create:
  `tests/test_hybridep_x86_workload_metrics.py`
- Modify:
  `public/hybridep-x86-20260728/index.html`
- Modify:
  `public/hybridep-x86-20260728/run-status.json`

1. Add failing tests for schema version, paired runs, exact requested/included
   and missing steps, per-metric valid counts, Policy/LogProb/E2E seconds and
   TPS/GPU, and smoke-versus-steady-state validity.
2. Reuse parsing behavior from
   `public/scripts/hybridep_seqpack_d893/run_log_metrics.py`, retaining raw
   per-step values and window provenance.
3. Store one workload object per model/topology and use the logged canonical
   TPS/GPU metrics; do not reconstruct throughput from averaged time.
4. Render concise conclusions, tables, and grouped bar charts from the JSON so
   HTML and machine results cannot drift.
5. Validate tests, JSON parsing, HTML parsing, and local static rendering.
6. Commit and push the Pages repo after each meaningful experiment-state
   update.

## Task 7: CW-DFW preflight and submission

**Remote paths:** Use only the user's CW Lustre project area discovered by the
preflight. Do not place environments, caches, models, or logs under `/home`.

1. Verify SSH/ControlMaster and pull the exact pushed branch on CW.
2. Sync/init recursive submodules and assert a clean checkout at the pushed
   SHA.
3. Select the highest current user-level FairShare account.
4. Confirm container, SM90 DeepEP wheel and SHA256, driver/Ray environment,
   HF caches, and model checkpoints.
5. Run `sbatch --test-only` for every intended launch.
6. Submit and monitor the runtime import gate.
7. Submit matched three-step Qwen3-30B-A3B all-to-all and HybridEP jobs in
   parallel and monitor at least five minutes.
8. After both reach the intended step gate, submit matched short Qwen3-235B
   and Nemotron3 Super pairs at valid topologies. Submit DeepSeek-V3 only when
   the placeholder model path has a verified checkpoint replacement.
9. Submit 20-step performance pairs only for workloads that pass compatibility
   and fit current allocation.
10. Record job IDs, node lists, states, exact commands, commits, hashes, and
    Lustre logs in the structured report.

## Task 8: Extract, compare, and publish results

1. For 20-step completed pairs, use common completed steps 5–20. For
   three-step jobs, use all completed steps and mark `warmup_only`.
2. Extract Policy training, Policy/reference LogProb, and E2E step seconds and
   logged TPS/GPU, plus generation context and numerical smoke metrics.
3. Compute ratio-of-sums or use the logged canonical aggregate as defined by
   the metric source; never average percentage ratios.
4. Compute HybridEP throughput improvement and time reduction.
5. Update `workload-metrics.json`, regenerate HTML, run report tests, commit,
   and push.
6. Report only terminal or current evidence: job IDs, exact measurement
   windows, absolute values, deltas, validity, failures, and remaining queued
   work.
