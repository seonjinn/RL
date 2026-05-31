#!/usr/bin/env bash
set -euo pipefail

# ABI probe wrapper for the vLLM 0.13.0 source-build candidate.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"

export VLLM_SITE_CANDIDATES="${VLLM_SITE_CANDIDATES:-$ARTIFACT_ROOT/python_site/vllm_0_13_0_cu129_torch28nv_source_py312}"
export JSON_OUT="${JSON_OUT:-$ARTIFACT_ROOT/reports/vllm_native_abi_probe_0_13_0.json}"
export MARKDOWN_OUT="${MARKDOWN_OUT:-$ARTIFACT_ROOT/reports/vllm_native_abi_probe_0_13_0.md}"
export JOB_FILE="${JOB_FILE:-$ROOT_DIR/latest_vllm_native_abi_probe_0_13_0_job.txt}"

exec "$ROOT_DIR/experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe.sh"
