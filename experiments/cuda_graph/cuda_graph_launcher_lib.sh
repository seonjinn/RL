#!/bin/bash

pr5672_qwen_checkpoint_dir() {
    local model=$1
    local condition=$2

    case "${model}:${condition}" in
        qwen3:pr5672-attn|qwen3:pr5672-attn-mlp)
            printf '%s\n' "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints/qwen3-8b-pr5672-20260716/${condition}"
            ;;
        *)
            return 1
            ;;
    esac
}
