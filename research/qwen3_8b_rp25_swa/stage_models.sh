#!/usr/bin/env bash
# Sourced by the container launcher; expose only applicable model overrides.
[[ "${method}" == none || "${method}" == dflash || "${method}" == dspark ]] || return 64
cp -aL "${target_source}" "${scratch_root}/target"
model_overrides=(
    "++policy.model_name=${scratch_root}/target"
    "++policy.tokenizer.name=${scratch_root}/target"
)
sha256sum "${scratch_root}/target/config.json" > "${output_root}/input-sha256.txt"
if [[ "${method}" != none ]]; then
    cp -aL "${draft_source}" "${scratch_root}/draft"
    model_overrides+=(
        "++policy.draft.model_name=${scratch_root}/draft"
        "++policy.generation.vllm_kwargs.speculative_config.model=${scratch_root}/draft"
    )
    sha256sum "${scratch_root}/draft/config.json" "${scratch_root}/draft/model.safetensors" >> "${output_root}/input-sha256.txt"
else
    draft_source=none
fi
printf 'target_source=%s\ndraft_source=%s\n' "${target_source}" "${draft_source}" > "${output_root}/input-paths.txt"
