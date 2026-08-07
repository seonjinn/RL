#!/usr/bin/env bash

# Shared, fail-closed provenance helpers for the Ptyche audit launchers.

audit_sha256_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

audit_sha256_path() {
    local path=$1
    [[ -e "${path}" ]] || {
        echo "Missing fingerprinted path: ${path}" >&2
        return 1
    }
    if [[ -f "${path}" ]]; then
        audit_sha256_stream < "${path}"
        return
    fi
    (
        cd "${path}"
        find . -type f -print0 | LC_ALL=C sort -z | while IFS= read -r -d '' file; do
            printf '%s\0' "${file}"
            audit_sha256_stream < "${file}"
            printf '\0'
        done
    ) | audit_sha256_stream
}

audit_sha256_path_or_absent() {
    local path=$1
    if [[ -e "${path}" ]]; then
        audit_sha256_path "${path}"
    else
        printf 'absent:%s\0' "${path}" | audit_sha256_stream
    fi
}

audit_scripts_sha256() {
    local scripts_root=$1
    find "${scripts_root}" -maxdepth 1 -type f -name '*.sh' -print0 |
        LC_ALL=C sort -z |
        while IFS= read -r -d '' script; do
            printf '%s\0' "$(basename "${script}")"
            audit_sha256_stream < "${script}"
            printf '\0'
        done | audit_sha256_stream
}

audit_execution_inputs_sha256() {
    local path
    for path in "$@"; do
        printf '%s\0' "${path}"
        audit_sha256_path "${path}"
        printf '\0'
    done | audit_sha256_stream
}

audit_assert_clean_tracked() {
    local checkout=$1
    [[ "$(git -C "${checkout}" rev-parse --is-inside-work-tree 2>/dev/null)" == true ]] || {
        echo "Git worktree is missing: ${checkout}" >&2
        return 1
    }
    [[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=no)" ]] || {
        echo "Tracked source is dirty: ${checkout}" >&2
        return 1
    }
}

audit_require_nonempty_dir() {
    local path=$1
    [[ -d "${path}" ]] && find "${path}" -type f -size +0c -print -quit | grep -q . || {
        echo "Missing or empty required cache: ${path}" >&2
        return 1
    }
}

audit_resolve_model_snapshot() {
    local cache_root=$1 expected_shards=$2 revision snapshot shard_count
    [[ -s "${cache_root}/refs/main" ]] || { echo "Missing local model revision: ${cache_root}/refs/main" >&2; return 1; }
    revision=$(tr -d '[:space:]' < "${cache_root}/refs/main")
    snapshot=${cache_root}/snapshots/${revision}
    shard_count=$(find -L "${snapshot}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d '[:space:]')
    [[ -f "${snapshot}/model.safetensors.index.json" && "${shard_count}" == "${expected_shards}" ]] || {
        echo "Incomplete local model snapshot: ${snapshot}" >&2
        return 1
    }
    printf '%s\t%s\n' "${snapshot}" "${revision}"
}

audit_prepare_submit() {
    local repo_dir=$1
    local vllm_root=$2
    local expected_vllm_commit=$3
    local actual_vllm_commit

    audit_assert_clean_tracked "${repo_dir}"
    audit_assert_clean_tracked "${vllm_root}"
    git -C "${repo_dir}" pull --ff-only
    audit_assert_clean_tracked "${repo_dir}"
    actual_vllm_commit=$(git -C "${vllm_root}" rev-parse HEAD)
    [[ "${actual_vllm_commit}" == "${expected_vllm_commit}" ]] || {
        echo "Unexpected vLLM commit: ${actual_vllm_commit}" >&2
        return 1
    }
}

audit_write_manifest() {
    local output_root=$1
    local run_kind=$2
    local repo_dir=$3
    local vllm_root=$4
    local expected_vllm_commit=$5
    local container=$6
    local recipe=$7
    local model_snapshot=$8
    local cache_root=$9
    local scripts_root=${10}
    local nemo_rl_commit
    local vllm_commit

    shift 10
    mkdir -p "${output_root}"
    nemo_rl_commit=$(git -C "${repo_dir}" rev-parse HEAD)
    vllm_commit=$(git -C "${vllm_root}" rev-parse HEAD)
    [[ "${vllm_commit}" == "${expected_vllm_commit}" ]] || {
        echo "Unexpected vLLM commit while writing manifest: ${vllm_commit}" >&2
        return 1
    }
    python3 - "${output_root}/run_manifest.json" "${run_kind}" "${nemo_rl_commit}" \
        "${vllm_commit}" "$(audit_sha256_path "${container}")" \
        "$(audit_sha256_path "${repo_dir}/${recipe}")" \
        "$(audit_sha256_path "${model_snapshot}")" \
        "$(audit_sha256_path "${cache_root}")" \
        "$(audit_scripts_sha256 "${scripts_root}")" \
        "$(audit_execution_inputs_sha256 "${scripts_root}" "$@")" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    run_kind,
    nemo_rl_commit,
    vllm_commit,
    container_sha256,
    recipe_sha256,
    model_snapshot_sha256,
    cache_sha256,
    scripts_sha256,
    execution_inputs_sha256,
) = sys.argv[1:]
manifest = {
    "cache_sha256": cache_sha256,
    "container_sha256": container_sha256,
    "model_snapshot_sha256": model_snapshot_sha256,
    "nemo_rl_commit": nemo_rl_commit,
    "recipe_sha256": recipe_sha256,
    "execution_inputs_sha256": execution_inputs_sha256,
    "run_kind": run_kind,
    "scripts_sha256": scripts_sha256,
    "vllm_commit": vllm_commit,
}
Path(manifest_path).write_text(
    json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
}
