#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for testing a higher vLLM source-build candidate without
# overwriting the canonical 0.10.2 watcher files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"

export ARTIFACT_ROOT
export VLLM_SOURCE_SPEC="${VLLM_SOURCE_SPEC:-https://files.pythonhosted.org/packages/11/12/b922f96778d07df1c28dfa9a81fbc9706c13c5d0a4e8d154060818a79705/vllm-0.13.0.tar.gz}"
export OUTPUT_SITE="${OUTPUT_SITE:-$ARTIFACT_ROOT/python_site/vllm_0_13_0_cu129_torch28nv_source_py312}"
export JSON_OUT="${JSON_OUT:-$ARTIFACT_ROOT/reports/vllm_native_source_build_0_13_0.json}"
export MARKDOWN_OUT="${MARKDOWN_OUT:-$ARTIFACT_ROOT/reports/vllm_native_source_build_0_13_0.md}"
export JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_native_source_build_0_13_0_job.txt}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.56.0,<5}"

exec "$ROOT_DIR/experiments/eagle3_qwen3_235b/submit_vllm_native_source_build.sh"
