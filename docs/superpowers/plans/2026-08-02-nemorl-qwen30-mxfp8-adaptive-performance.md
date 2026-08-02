# NeMo-RL Qwen3-30B-A3B MXFP8 Adaptive Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the dense MXFP8 adaptive kernel policy improves Qwen3-30B-A3B NeMo-RL rollout throughput under the established performance workload.

**Architecture:** Add a trace-only execution arm to the existing vLLM 0.25.1 NeMo-RL canary so representative CUDA Graph rollout traffic records exact dense MXFP8 execution signatures. Only when eligible signatures exist, profile legal TRTLLM tactics offline, build a Qwen-specific exact-match table, and compare a matched CuTeDSL baseline with the adaptive TRTLLM arm.

**Tech Stack:** NeMo-RL, vLLM 0.25.1 custom source overlay, FlashInfer TRTLLM MXFP8 runner, OmegaConf, pytest, SLURM on Ptyche GB200.

## Global Constraints

- Use `Qwen/Qwen3-30B-A3B` with dynamic MXFP8 generation and the performance recipe's ignored `q_proj`, `k_proj`, `v_proj`, and `o_proj` layers.
- Keep CUDA Graph enabled with `enforce_eager=false`.
- Use 64 prompts and 32 generations per prompt, for 2,048 rollout samples.
- Use two Ptyche nodes with four GB200 GPUs per node and TP1/EP1 generation engines.
- Never reuse the Ultra tactic table for Qwen.
- Exact signature misses must fall back to the backend default and must not fail serving.
- Commit and push before SLURM submission; run `sbatch --test-only` first and monitor the live job for at least five minutes.

---

### Task 1: Trace Arm Contract

**Files:**
- Modify: `experiments/mxfp8_adaptive_rollout_v0251/contract.py`
- Modify: `experiments/mxfp8_adaptive_rollout_v0251/run_arm.sh`
- Modify: `experiments/mxfp8_adaptive_rollout_v0251/run_eval_canary.py`
- Test: `tests/unit/experiments/test_mxfp8_adaptive_rollout_v0251.py`

**Interfaces:**
- Produces: `TraceInputs(trace_dir: Path, trace_max: int)` and `build_arm_environment("trace", ...)`.
- Produces: a trace arm that selects `flashinfer_trtllm`, enables JSONL signature tracing, and does not require a tactic table.

- [ ] **Step 1: Write failing tests for the trace environment and launcher.**
- [ ] **Step 2: Run the focused unit tests and verify the trace assertions fail.**
- [ ] **Step 3: Implement trace arm validation and environment construction.**
- [ ] **Step 4: Extend the shell and Python launchers to accept the trace arm.**
- [ ] **Step 5: Run the focused unit tests and commit the passing contract.**

### Task 2: Qwen Performance Workload Configuration

**Files:**
- Create: `experiments/mxfp8_adaptive_rollout_v0251/configs/eval_qwen3_30ba3b_performance.yaml`
- Create: `experiments/mxfp8_adaptive_rollout_v0251/data/qwen_trace_math.jsonl`
- Test: `tests/unit/experiments/test_mxfp8_adaptive_rollout_v0251.py`

**Interfaces:**
- Produces: a generation-only evaluation configuration with 64 prompts repeated 32 times by `eval.num_tests_per_prompt`.

- [ ] **Step 1: Write a failing test for model, MXFP8 scope, workload size, TP/EP, resources, and CUDA Graph settings.**
- [ ] **Step 2: Run the test and verify the configuration is missing.**
- [ ] **Step 3: Add a deterministic 64-prompt dataset and the Qwen configuration.**
- [ ] **Step 4: Run the focused tests and validate OmegaConf resolution.**
- [ ] **Step 5: Commit the Qwen workload configuration.**

### Task 3: Shape Trace Summary and Submission

**Files:**
- Create: `experiments/mxfp8_adaptive_rollout_v0251/shape_trace.py`
- Create: `experiments/mxfp8_adaptive_rollout_v0251/submit_qwen30_ptyche.sh`
- Test: `tests/unit/experiments/test_mxfp8_adaptive_rollout_v0251.py`

**Interfaces:**
- Produces: `summarize_shape_trace(trace_dir: Path) -> dict[str, object]`.
- Produces: a Ptyche submitter with `test-only` and `submit` actions for the trace arm.

- [ ] **Step 1: Write failing tests for unique signature aggregation and a valid zero-eligible result.**
- [ ] **Step 2: Run the tests and verify the summary module is absent.**
- [ ] **Step 3: Implement deterministic JSONL aggregation and trace metadata output.**
- [ ] **Step 4: Add the submission script and static shell assertions.**
- [ ] **Step 5: Run all experiment unit tests and commit.**

### Task 4: Ptyche Trace and Conditional Offline Shmoo

**Files:**
- Create when eligible: `experiments/mxfp8_adaptive_rollout_v0251/artifacts/qwen3_30ba3b_exact_tactics.json`
- Create when eligible: `experiments/mxfp8_adaptive_rollout_v0251/artifacts/qwen3_30ba3b_layer_allowlist.txt`

**Interfaces:**
- Consumes: trace JSONL signatures from Task 3.
- Produces: either a zero-eligible evidence artifact or a fingerprinted Qwen exact-tactic table and allowlist.

- [ ] **Step 1: Push the clean branch and update the Ptyche checkout with `git pull --ff-only`.**
- [ ] **Step 2: Run the submission script in `test-only` mode.**
- [ ] **Step 3: Submit the trace job and monitor startup for five minutes.**
- [ ] **Step 4: Recover any initialization, cache, NCCL, or configuration failure and resubmit.**
- [ ] **Step 5: Summarize signatures; if nonzero, shmoo every legal tactic for each exact signature and qualify repeatable winners.**

### Task 5: Matched Baseline and Adaptive Comparison

**Files:**
- Modify: `experiments/mxfp8_adaptive_rollout_v0251/submit_qwen30_ptyche.sh`
- Modify: `experiments/mxfp8_adaptive_rollout_v0251/README.md`
- Create: `experiments/mxfp8_adaptive_rollout_v0251/artifacts/qwen3_30ba3b_performance_summary.json`

**Interfaces:**
- Consumes: the qualified table from Task 4 when eligible.
- Produces: matched generation throughput, output-token count, model-load time, and elapsed-time comparison.

- [ ] **Step 1: Add baseline/adaptive A/B actions that differ only in linear backend and qualified lookup inputs.**
- [ ] **Step 2: Run shell and unit verification locally.**
- [ ] **Step 3: Commit, push, run `test-only`, then submit the A/B job.**
- [ ] **Step 4: Monitor the job and repair failures until both arms complete.**
- [ ] **Step 5: Record throughput and explain whether the Qwen quantization scope exposes any dense adaptive opportunity.**
