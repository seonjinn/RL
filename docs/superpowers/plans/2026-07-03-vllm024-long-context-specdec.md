# vLLM 0.24 Long-Context SpecDec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch matched Qwen3-8B SpecDec benchmarks for OSL 64K and supported total-context 128K.

**Architecture:** A Python materializer creates symlink-backed YaRN model views for the target and every drafter. A focused shell wrapper maps the 64K and 128K profiles onto the existing extended-method matrix so the established benchmark and partial-result behavior remain unchanged.

**Tech Stack:** Python 3, Bash, pytest, Pyright, vLLM 0.24.0, SLURM, GB200.

## Global Constraints

- Use `ISL=4096`, `OSL=65536` for 64K, and `OSL=126976` for total-context 128K.
- Use YaRN factor 4 with original context 32768 and rope theta 1000000.
- Start with batch size 1, one measurement, and no benchmark-level warmup.
- Keep Math/SWE, temperature 0/1, baseline/Suffix/PARD/PARD-2/DFlash, and the existing June 19 runtime contract.
- Use Lustre output paths, `--segment=1`, and no `--gres` on Lyris.

---

### Task 1: YaRN Model Views

**Files:**
- Create: `experiments/vllm_024_dynamicsd/materialize_long_context_model_views.py`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Produces: `materialize_model_view(source: Path, destination: Path, max_position_embeddings: int, rope_factor: float) -> dict[str, object]`.

- [ ] Add tests that require symlinked checkpoint files, an owned extended `config.json`, and reproducibility metadata.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement atomic view materialization without copying checkpoint weights.
- [ ] Run the focused tests and full experiment test file.

### Task 2: Long-Context Submission Profiles

**Files:**
- Create: `experiments/vllm_024_dynamicsd/submit_qwen8_long_context_matrix.sh`
- Modify: `experiments/vllm_024_dynamicsd/submit_qwen8_extended_methods_matrix.sh`
- Modify: `tests/test_vllm024_dynamicsd.py`

**Interfaces:**
- Consumes: model views from Task 1.
- Produces: dry-run and live 64K/128K matrix submissions using the existing `submit_matrix.sh` interface.

- [ ] Add dry-run tests for exact 64K and 128K profile arguments and job counts.
- [ ] Run the focused tests and confirm the new wrapper is missing.
- [ ] Make the existing Qwen3-8B wrapper accept context, model-view, warmup, and job-label overrides.
- [ ] Implement the two-profile wrapper with BS1 and five-hour limits.
- [ ] Run pytest, `bash -n`, and Pyright.

### Task 3: Documentation And Cluster Launch

**Files:**
- Modify: `experiments/vllm_024_dynamicsd/README.md`
- Modify: `experiments/vllm_024_dynamicsd/report/README.md`

**Interfaces:**
- Consumes: submitted job manifests and partial JSON rows.
- Produces: reproducible commands and clearly separated native/YaRN status.

- [ ] Document the profile definitions, model-view rationale, and DFlare wall-time exclusion.
- [ ] Commit and push the exact files.
- [ ] Pull the branch on Lyris, materialize views, and run `--test-only` for both profiles.
- [ ] Submit 64K/128K BS1 jobs and record job IDs and result roots.
- [ ] Monitor for five minutes, inspect early failures, and collect any completed rows.
