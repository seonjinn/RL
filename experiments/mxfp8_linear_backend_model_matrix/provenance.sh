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
            requirements/*.txt) ;;
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
        HEAD -- requirements/ | mxfp8_sha256_stream
}
