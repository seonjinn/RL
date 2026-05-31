#!/usr/bin/env bash
set -euo pipefail

# SWE-Gym one-step rollout smoke wrapper for a source-built vLLM 0.13.0 site.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"

export SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_13_0_cu129_torch28nv_source_py312}"
export SOURCE_BUILD_JSON="${SOURCE_BUILD_JSON:-$ARTIFACT_ROOT/reports/vllm_native_source_build_0_13_0.json}"
export VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-https://files.pythonhosted.org/packages/11/12/b922f96778d07df1c28dfa9a81fbc9706c13c5d0a4e8d154060818a79705/vllm-0.13.0.tar.gz}"
export WANDB_NAME="${WANDB_NAME:-qwen3-235b-swe-rollout-vllm0130src-swegym-example-smoke1step}"
export ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$ARTIFACT_ROOT/rl_rollout_capture_logs/qwen3_235b_swe_capture_vllm0130src_swegym_example_smoke1step}"
export OUTPUT_CONVERSATIONS="${OUTPUT_CONVERSATIONS:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations_swegym_example_vllm0130src_smoke.jsonl}"
export ROLLOUT_REPORT_PREFIX_TAG="${ROLLOUT_REPORT_PREFIX_TAG:-vllm0130src}"

exec "$ROOT_DIR/experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke.sh"
