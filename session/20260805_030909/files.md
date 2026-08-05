# Files

## Inspected

- `experiments/bf16-nvfp4-rollout/README.md` - launch matrix and pass criteria.
- `experiments/bf16-nvfp4-rollout/PLAN.md` - experiment invariants.
- `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a16-rollout.sh` - W4A16 smoke.
- `tests/test_suites/llm/performance/grpo-qwen3-30ba3b-4n4g-nvfp4-w4a4-rollout.sh` - W4A4 smoke.

## Changed

- `experiments/bf16-nvfp4-rollout/submit_gcp_nrt.sh` - reproducible GCP-NRT
  preflight and submission wrapper with focused actor-path regression gates.
- `experiments/bf16-nvfp4-rollout/README.md` - documents the GCP wrapper.
- `nemo_rl/modelopt/calibration_artifact.py` - resolves repository-relative
  quantization recipes independently of actor working directory.
- `nemo_rl/modelopt/models/generation/vllm_quant_worker.py` - normalizes the
  real-NVFP4 recipe before mode resolution.
- `tests/unit/modelopt/test_calibration_artifact.py` - project-relative path
  regression test.
- `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py` -
  verifies normalized path reaches real-NVFP4 mode resolution.

## Generated

- This session record.
