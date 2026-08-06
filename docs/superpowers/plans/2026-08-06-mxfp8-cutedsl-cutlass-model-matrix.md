# MXFP8 CuTeDSL versus CUTLASS Model Matrix Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Run a provenance-matched CUDA Graph comparison of FlashInfer CuTeDSL and CUTLASS dense MXFP8 linear backends for Qwen3-30B-A3B, Qwen3-235B-A22B, and Nemotron3 Super on Ptyche.

**Architecture:** Keep the shipped NeMo-RL performance recipe for each model and vary only `policy.generation.vllm_kwargs.linear_backend`. Every launcher validates the NeMo-RL and custom vLLM commits, uses `moe_backend=flashinfer_trtllm`, includes Q/K/V/O projections in MXFP8, and records outputs under a model-specific experiment root. A shared parser extracts steady-state steps 3--8 into a single model/backend comparison artifact.

**Tech Stack:** Bash, SLURM, NeMo-RL Hydra recipes, Python 3.12, pytest, JSON/CSV.

---

### Task 1: Make the Qwen3-235B launcher Ptyche-safe

**Files:**
- Modify: `experiments/qwen235b_mxfp8_linear_backends/submit_cluster.sh`
- Modify: `experiments/qwen235b_mxfp8_linear_backends/submit_matrix.sh`
- Modify: `experiments/qwen235b_mxfp8_linear_backends/README.md`
- Test: `tests/experiments/test_qwen235b_mxfp8_linear_backend_launcher.py`

**Step 1: Write the failing tests**

Add assertions that the default account is `coreai_dlalgo_llm`, the two matrix arms are independently submitted without `afterok`, and both smoke and measurement runs can be selected through `MAX_STEPS` without changing any backend-independent command text.

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/experiments/test_qwen235b_mxfp8_linear_backend_launcher.py`

Expected: FAIL because the launcher still defaults to `nemotron_sw_post` and the matrix behavior is not covered.

**Step 3: Implement the minimal launcher changes**

Set the Ptyche account default to `coreai_dlalgo_llm`. Keep each backend independently schedulable. Document the exact `ACTION=test-only`, `MAX_STEPS=2`, and `MAX_STEPS=8` commands and the required custom vLLM commit.

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/experiments/test_qwen235b_mxfp8_linear_backend_launcher.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/qwen235b_mxfp8_linear_backends tests/experiments/test_qwen235b_mxfp8_linear_backend_launcher.py
git commit -s -m "test: prepare Qwen235B backend matrix for Ptyche"
```

### Task 2: Add the Nemotron3 Super CUDA Graph backend matrix

**Files:**
- Create: `experiments/nemotron3_super_mxfp8_linear_backends/README.md`
- Create: `experiments/nemotron3_super_mxfp8_linear_backends/submit_ptyche.sh`
- Create: `experiments/nemotron3_super_mxfp8_linear_backends/submit_matrix_ptyche.sh`
- Test: `tests/experiments/test_nemotron3_super_mxfp8_linear_backend_launcher.py`

**Step 1: Write the failing tests**

Test both `flashinfer_cutedsl` and `flashinfer_cutlass` dry-run commands. Require the shipped `grpo-nemotron3-super-120BA12B-32n4g.yaml` recipe, 32 nodes x 4 GPUs, segment size 8, TP4 generation, 256 rollouts per step, sequence cap 8192, CUDA Graph enabled, MXFP8 precision, Q/K/V/O included, `moe_backend=flashinfer_trtllm`, and no dependency between arms. Normalize only the backend string and require all other command text to match.

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/experiments/test_nemotron3_super_mxfp8_linear_backend_launcher.py`

Expected: FAIL because the launcher does not exist.

**Step 3: Implement the launcher**

Follow the Qwen launchers' provenance and environment checks. Use the Ptyche defaults `coreai_dlalgo_llm`, `batch`, five hours, 32 nodes, four GPUs per node, and segment size 8. Override the Super BF16 recipe to `precision=fp8`, `is_mx=true`, and `quantization_ignored_layer_kws=[lm_head,mlp.gate]`. Disable W&B and checkpoint writes; enable TensorBoard.

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/experiments/test_nemotron3_super_mxfp8_linear_backend_launcher.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/nemotron3_super_mxfp8_linear_backends tests/experiments/test_nemotron3_super_mxfp8_linear_backend_launcher.py
git commit -s -m "feat: add Nemotron3 Super MXFP8 backend matrix"
```

### Task 3: Add a shared model-matrix result summarizer

**Files:**
- Create: `experiments/mxfp8_linear_backend_model_matrix/summarize_results.py`
- Create: `experiments/mxfp8_linear_backend_model_matrix/README.md`
- Test: `tests/experiments/test_mxfp8_linear_backend_model_matrix_summary.py`

**Step 1: Write the failing tests**

Create synthetic Qwen3-30B, Qwen3-235B, and Nemotron3 Super driver logs with eight training-result blocks per backend. Verify that steps 3--8 are summarized, output-token lengths must match between arms, normalized throughput uses CUTLASS as 1.0, and a missing or mismatched backend row raises a clear error.

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/experiments/test_mxfp8_linear_backend_model_matrix_summary.py`

Expected: FAIL because the summarizer does not exist.

**Step 3: Implement the minimal summarizer**

Reuse the existing NeMo-RL `Training Results` parsing format. Accept explicit `MODEL=RUN_ROOT` inputs, locate exactly one driver log per backend, validate equal measured-step count and mean generation length, and write `step_metrics.csv` plus `summary.json` containing absolute and CUTLASS-normalized generation/E2E throughput and latency metrics.

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/experiments/test_mxfp8_linear_backend_model_matrix_summary.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/mxfp8_linear_backend_model_matrix tests/experiments/test_mxfp8_linear_backend_model_matrix_summary.py
git commit -s -m "feat: summarize MXFP8 backend model matrix"
```

### Task 4: Verify, publish, and submit smoke runs

**Files:**
- Verify all files changed in Tasks 1--3.

**Step 1: Run focused tests**

Run:

```bash
pytest -q \
  tests/experiments/test_qwen30b_mxfp8_linear_backend_launcher.py \
  tests/experiments/test_qwen235b_mxfp8_linear_backend_launcher.py \
  tests/experiments/test_nemotron3_super_mxfp8_linear_backend_launcher.py \
  tests/experiments/test_qwen30b_mxfp8_linear_backend_summary.py \
  tests/experiments/test_mxfp8_linear_backend_model_matrix_summary.py
```

Expected: PASS.

**Step 2: Check shell syntax and dry-run parity**

Run `bash -n` for every new or modified launcher, then dry-run both backends for all three models. Confirm the normalized command differs only in the dense linear backend.

**Step 3: Push the implementation branch**

Run `git push fork sna/qwen30b-mxfp8-linear-backend-perf` after confirming the worktree is clean.

**Step 4: Refresh the Ptyche checkout**

Use a clean remote checkout of the pushed branch, run `git pull --ff-only`, and verify the custom vLLM worktree is exactly commit `a76062edee3a3ac23d47a93c7ce466f06a19111f`.

**Step 5: Validate scheduling**

Run each model matrix with `ACTION=test-only MAX_STEPS=2`. Expected: SLURM accepts all six independent jobs under account `coreai_dlalgo_llm`.

**Step 6: Submit and monitor two-step smoke jobs**

Submit both backends for each model with the same `RUN_ID`, no inter-arm dependency, and `MAX_STEPS=2`. Monitor queue and logs for at least five minutes. Cancel and fix immediately on provenance, initialization, NCCL, OOM, CUDA Graph, or token-validity failure.

**Step 7: Submit eight-step measurements after smoke qualification**

Only submit the corresponding `MAX_STEPS=8` pair after both smoke arms for that model exit successfully. Record job IDs, physical nodes, output roots, and the exact code/container provenance.
