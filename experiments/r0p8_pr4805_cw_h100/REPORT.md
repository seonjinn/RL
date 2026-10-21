# CW H100 result: PR #4805 Qwen3-30B-A3B

Status: CW job `20642814` ran from 16:50:08 to 16:53:11 PDT on 2026-10-21 and Slurm recorded `FAILED` with exit code `1:0`. It failed while loading the OpenMathInstruct-2 dataset because the shared `HF_DATASETS_CACHE` directory denied creation of a `.lock` file. The run never initialized vLLM, generated tokens, or tested the MoE kernel.

The job uses source commit `24e57529` (PR #4805 merge commit `dbad3b42` plus the experiment launcher), the pinned `r0.8.0-19666f86c` container, BF16 Qwen3-30B-A3B, four nodes × eight H100 GPUs, policy HybridEP EP8, generation TP2 and `moe_backend: flashinfer_trtllm`. The run manifest and logs are under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_pr4805_cw_h100/results/qwen3-30ba3b-bf16-flashinfer-trtllm-4n8g-2step-20261021T223226Z/`.

## Hardware compatibility

The pinned container uses vLLM 0.29.0. Its [unquantized BF16 TRTLLM MoE kernel](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py#L87-L95) requires CUDA device capability family 100 (Blackwell). [vLLM 0.32.1rc0 retains the same condition](https://github.com/vllm-project/vllm/blob/v0.32.1rc0/vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py#L95-L102). CW H100 GPUs are SM90, so this BF16 kernel is unsupported there. The [vLLM 0.29.0 backend selector](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/layers/fused_moe/oracle/unquantized.py#L271-L297) raises when an explicitly requested backend is unsupported; it does not silently fall back to Triton. This is a source-level conclusion because the dataset permission error prevented reaching kernel selection in this job.

Do not resubmit this BF16 FlashInfer TRTLLM recipe on CW H100. Use the original `moe_backend: triton` for H100. A Blackwell cluster is needed to test the PR #4805 BF16 FlashInfer TRTLLM path. A future Triton test also needs a writable Hugging Face datasets cache rather than `$HF_HOME/datasets` in the current shared cache.
