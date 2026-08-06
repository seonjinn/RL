#!/usr/bin/env bash

mxfp8_sha256_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

mxfp8_dependency_state_sha256() {
    local repo_dir=$1
    local filename
    for filename in pyproject.toml uv.lock; do
        [[ -f "${repo_dir}/${filename}" ]] || {
            echo "Missing dependency state file: ${repo_dir}/${filename}" >&2
            return 1
        }
    done
    {
        for filename in pyproject.toml uv.lock; do
            printf '%s\0' "${filename}"
            cat "${repo_dir}/${filename}"
            printf '\0'
        done
    } | mxfp8_sha256_stream
}

mxfp8_file_sha256() {
    local path=$1
    [[ -f "${path}" ]] || {
        echo "Missing fingerprinted file: ${path}" >&2
        return 1
    }
    mxfp8_sha256_stream < "${path}"
}

mxfp8_vllm_source_sha256() {
    local vllm_root=$1
    git -C "${vllm_root}" archive --format=tar HEAD | mxfp8_sha256_stream
}

mxfp8_assert_vllm_tracked_state() {
    local vllm_root=$1
    local changed_path
    while IFS= read -r changed_path; do
        [[ -z "${changed_path}" ]] && continue
        case "${changed_path}" in
            pyproject.toml|requirements/*) ;;
            *)
                echo "Disallowed tracked vLLM change: ${changed_path}" >&2
                return 1
                ;;
        esac
    done < <(
        {
            git -C "${vllm_root}" diff --name-only --no-ext-diff --
            git -C "${vllm_root}" diff --cached --name-only --no-ext-diff --
        } | sort -u
    )
}

mxfp8_vllm_dependency_state_sha256() {
    local vllm_root=$1
    git -C "${vllm_root}" diff --binary --full-index --no-ext-diff \
        --no-renames --diff-algorithm=myers --src-prefix=a/ --dst-prefix=b/ \
        HEAD -- pyproject.toml requirements/ | mxfp8_sha256_stream
}

mxfp8_vllm_build_state_matches() {
    local vllm_root=$1
    local marker_path=${vllm_root}/nemo-rl-build-state.sha256
    local recorded_sha256
    local current_sha256
    [[ -f "${marker_path}" ]] || {
        echo "Missing vLLM build-state marker: ${marker_path}" >&2
        return 1
    }
    recorded_sha256=$(cat "${marker_path}")
    current_sha256=$(mxfp8_vllm_dependency_state_sha256 "${vllm_root}")
    [[ "${recorded_sha256}" == "${current_sha256}" ]] || {
        echo "Stale vLLM build-state marker: ${marker_path}" >&2
        return 1
    }
}

mxfp8_vllm_reuse_state_valid() {
    local vllm_root=$1
    local expected_commit=$2
    local expected_wheel=$3
    local actual_commit
    [[ -d "${vllm_root}/.git" ]] || return 1
    actual_commit=$(git -C "${vllm_root}" rev-parse HEAD)
    [[ "${actual_commit}" == "${expected_commit}" ]] || return 1
    mxfp8_assert_vllm_tracked_state "${vllm_root}" || return 1
    mxfp8_vllm_build_state_matches "${vllm_root}" || return 1
    [[ -f "${vllm_root}/nemo-rl.env" ]] || {
        echo "Missing custom vLLM environment: ${vllm_root}/nemo-rl.env" >&2
        return 1
    }
    (
        unset VLLM_GIT_REF VLLM_PRECOMPILED_WHEEL_LOCATION
        source "${vllm_root}/nemo-rl.env"
        [[ "${VLLM_GIT_REF:-}" == "${expected_commit}" ]]
        [[ "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" == "${expected_wheel}" ]]
    ) || {
        echo "Stale custom vLLM environment: ${vllm_root}/nemo-rl.env" >&2
        return 1
    }
    [[ -x "${vllm_root}/.venv/bin/python" ]] &&
        "${vllm_root}/.venv/bin/python" -c 'import vllm' || {
        echo "Custom vLLM import check failed: ${vllm_root}" >&2
        return 1
    }
}
