#!/usr/bin/env bash

# Shared, fail-closed provenance helpers for the Ptyche audit launchers.

audit_sha256_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

audit_assert_no_broken_symlinks() {
    local path=$1
    [[ -e "${path}" || -L "${path}" ]] || {
        echo "Missing fingerprinted path: ${path}" >&2
        return 1
    }
    if [[ -L "${path}" && ! -e "${path}" ]]; then
        echo "Broken symlink in fingerprinted path: ${path}" >&2
        return 1
    fi
    local broken
    broken=$(find -P "${path}" -type l ! -exec test -e {} \; -print -quit)
    [[ -z "${broken}" ]] || {
        echo "Broken symlink in fingerprinted path: ${broken}" >&2
        return 1
    }
}

audit_sha256_path() {
    local path=$1
    audit_assert_no_broken_symlinks "${path}" || return
    if [[ -f "${path}" ]]; then
        audit_sha256_stream < "${path}"
        return
    fi
    (
        cd "${path}"
        find -L . -type f -print0 | LC_ALL=C sort -z | while IFS= read -r -d '' file; do
            printf '%s\0' "${file}"
            audit_sha256_stream < "${file}"
            printf '\0'
        done
    ) | audit_sha256_stream
}

audit_scripts_sha256() {
    local scripts_root=$1
    audit_assert_no_broken_symlinks "${scripts_root}" || return
    find -L "${scripts_root}" -type f \( -name '*.sh' -o -name '*.py' \) -print0 |
        LC_ALL=C sort -z |
        while IFS= read -r -d '' script; do
            printf '%s\0' "${script#"${scripts_root}"/}"
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
    [[ -d "${path}" ]] || {
        echo "Missing or empty required cache: ${path}" >&2
        return 1
    }
    audit_assert_no_broken_symlinks "${path}" || return
    find -L "${path}" -type f -size +0c -print -quit | grep -q . || {
        echo "Missing or empty required cache: ${path}" >&2
        return 1
    }
}

audit_resolve_model_snapshot() {
    local cache_root=$1 expected_shards=$2 revision snapshot shard_count
    [[ -s "${cache_root}/refs/main" ]] || { echo "Missing local model revision: ${cache_root}/refs/main" >&2; return 1; }
    revision=$(tr -d '[:space:]' < "${cache_root}/refs/main")
    snapshot=${cache_root}/snapshots/${revision}
    audit_assert_no_broken_symlinks "${snapshot}" || return
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
    local container_sha256
    local recipe_sha256
    local model_snapshot_sha256
    local scripts_sha256
    local execution_inputs_sha256

    shift 10
    mkdir -p "${output_root}"
    nemo_rl_commit=$(git -C "${repo_dir}" rev-parse HEAD)
    vllm_commit=$(git -C "${vllm_root}" rev-parse HEAD)
    [[ "${vllm_commit}" == "${expected_vllm_commit}" ]] || {
        echo "Unexpected vLLM commit while writing manifest: ${vllm_commit}" >&2
        return 1
    }
    local cache_sha256=null
    if [[ "${cache_root}" != '-' ]]; then
        cache_sha256=$(audit_sha256_path "${cache_root}") || return
    fi
    container_sha256=$(audit_sha256_path "${container}") || return
    recipe_sha256=$(audit_sha256_path "${repo_dir}/${recipe}") || return
    model_snapshot_sha256=$(audit_sha256_path "${model_snapshot}") || return
    scripts_sha256=$(audit_scripts_sha256 "${scripts_root}") || return
    execution_inputs_sha256=$(audit_execution_inputs_sha256 "$@") || return
    python3 - "${output_root}/run_manifest.json" "${run_kind}" "${nemo_rl_commit}" \
        "${vllm_commit}" "${container_sha256}" "${recipe_sha256}" \
        "${model_snapshot_sha256}" \
        "${cache_sha256}" \
        "${scripts_sha256}" "${execution_inputs_sha256}" <<'PY'
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
    "cache_sha256": None if cache_sha256 == "null" else cache_sha256,
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

audit_assert_smoke_manifest_matches() {
    local manifest=$1 nemo_rl_commit=$2 vllm_commit=$3 recipe_sha256=$4
    local model_sha256=$5 cache_sha256=$6 scripts_sha256=$7 execution_inputs_sha256=$8
    python3 - "${manifest}" "${nemo_rl_commit}" "${vllm_commit}" "${recipe_sha256}" \
        "${model_sha256}" "${cache_sha256}" "${scripts_sha256}" \
        "${execution_inputs_sha256}" <<'PY'
import json
import sys

path, *expected = sys.argv[1:]
keys = (
    "nemo_rl_commit",
    "vllm_commit",
    "recipe_sha256",
    "model_snapshot_sha256",
    "cache_sha256",
    "scripts_sha256",
    "execution_inputs_sha256",
)
with open(path, encoding="ascii") as handle:
    manifest = json.load(handle)
for key, value in zip(keys, expected, strict=True):
    if manifest.get(key) != value:
        raise SystemExit(f"Stale smoke manifest {key}")
PY
}
