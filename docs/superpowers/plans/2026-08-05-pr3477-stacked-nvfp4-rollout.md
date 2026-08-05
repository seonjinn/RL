# PR 3477 Stacked NVFP4 Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stack BF16-training plus real W4A16/W4A4 NVFP4 rollout support on the exact PR 3477 head without changing its validated BF16-to-MXFP8 behavior.

**Architecture:** Keep NCCL Reshard responsible for topology-aware BF16 shard movement, then use a destination-owned transform codec to materialize precision-specific checkpoint components. MXFP8 remains a two-component receiver transform; NVFP4 adds grouped ModelOpt serialization, checkpoint-layout scratch, layerwise reload finalization, and a static calibration artifact for W4A4.

**Tech Stack:** Python 3.13, PyTorch distributed/NCCL, Ray, Megatron-LM, vLLM 0.25.1, NVIDIA ModelOpt, pytest, SLURM on GCP-NRT B200.

## Global Constraints

- Base commit is PR 3477 head `6f57c1b79504245fc8211028e504465045315f34`.
- Preserve PR 3477 BF16-to-MXFP8 output shapes and validation behavior.
- Training remains plain BF16 `MegatronPolicyWorker`; do not enable QARL or QAT.
- Quantization and deployment-layout ownership remain on the generation receiver.
- W4A4 requires a provenance-validated activation calibration artifact; no dummy scales.
- Every behavior change starts with a failing focused test and ends with container-based verification.

---

### Task 1: Cross-Precision Transform Contract

**Files:**
- Create: `nemo_rl/weight_sync/refit_transforms.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_refit_transforms.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: PR 3477's receiver-side BF16-to-MXFP8 refit behavior.
- Produces: `RefitTransformRequest`, `RefitTransformPlan`, component specifications, deterministic plan agreement, and codec lookup for BF16-to-MXFP8/W4A16/W4A4.

- [ ] Add failing tests for multi-component plan serialization, plan agreement, unsupported precision pairs, and PR 3477 MXFP8 compatibility.
- [ ] Run the focused tests in the Linux NeMo-RL container and confirm the expected failures.
- [ ] Add the generic component and codec contract with no transport-side quantization.
- [ ] Route NCCL plan construction through the contract while retaining the PR 3477 two-component MXFP8 result.
- [ ] Run the focused tests and commit the generic core.

### Task 2: NVFP4 Serializer and Calibration State

**Files:**
- Create: `nemo_rl/modelopt/models/generation/nvfp4_refit.py`
- Create: `nemo_rl/modelopt/calibration_artifact.py`
- Create: `examples/modelopt/export_nvfp4_calibration.py`
- Test: `tests/unit/models/generation/test_nvfp4_refit.py`
- Test: `tests/unit/modelopt/test_calibration_artifact.py`

**Interfaces:**
- Consumes: destination component families from Task 1.
- Produces: canonical grouped W4A16/W4A4 ModelOpt export and immutable W4A4 input-scale state.

- [ ] Add failing tests for packed weight, block scale, global scale, fused group scale sharing, and missing W4A4 calibration.
- [ ] Verify failures in the Linux container.
- [ ] Port the dependency-light NVFP4 serializer and calibration artifact validation.
- [ ] Verify bitwise parity against the canonical Megatron-Bridge/ModelOpt exporter.
- [ ] Run focused tests and commit serializer support.

### Task 3: vLLM Real-Quant Receiver

**Files:**
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- Modify: `nemo_rl/modelopt/models/generation/vllm_quant_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Test: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`

**Interfaces:**
- Consumes: BF16 NCCL destination shards and NVFP4 serializer output.
- Produces: complete W13/W2 reload groups, checkpoint-layout scratch, one layerwise reload lifecycle, and a final completion fence.

- [ ] Add failing tests for destination scratch, incomplete groups, duplicate components, grouped experts, and finalization order.
- [ ] Port receiver conversion and ModelOpt reload integration.
- [ ] Confirm PR 3477 MXFP8 tests remain unchanged and passing.
- [ ] Run focused tests and commit receiver support.

### Task 4: Recipes and Runtime Contracts

**Files:**
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.yaml`
- Create: `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh`
- Create: `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh`
- Test: `tests/test_nvfp4_rollout_recipes.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: reproducible BF16-training plus real-NVFP4 rollout jobs with real importance sampling and fail-loud post-run checks.

- [ ] Add failing recipe and worker-selection contract tests.
- [ ] Add W4A16 and calibrated W4A4 recipes using plain BF16 policy storage.
- [ ] Add two-step legacy and NCCL smoke scripts with finite metric and reload-completeness gates.
- [ ] Run recipe tests and commit runtime definitions.

### Task 5: GCP-NRT Verification

**Files:**
- Create: `experiments/bf16-nvfp4-rollout/README.md`
- Create: `experiments/bf16-nvfp4-rollout/REPORT.md`
- Create: `experiments/bf16-nvfp4-rollout/submit_gcp_nrt.sh`

**Interfaces:**
- Consumes: committed stacked branch and immutable code snapshots.
- Produces: job IDs, W&B links, refit/component timing, correctness gates, and branch provenance.

- [ ] Push the stacked branch before submission.
- [ ] Run scheduler `--test-only` for W4A16 and W4A4.
- [ ] Run focused tests in the exact nightly container.
- [ ] Submit two-step legacy and NCCL smoke jobs and monitor each for at least five minutes.
- [ ] Record completed outcomes and compare with the previously validated NVFP4 branch.
