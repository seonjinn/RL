# CW H100 result: PR #4805 Qwen3-30B-A3B

Status: Two-step CW `batch` job `20642814` submitted on 2026-10-21 after `sbatch --test-only` succeeded. The first scheduler check showed `PENDING`, with an estimated start of 2026-10-21 18:30 PDT. This is not yet a runtime result.

The job uses source commit `24e57529` (PR #4805 merge commit `dbad3b42` plus the experiment launcher), the pinned `r0.8.0-19666f86c` container, BF16 Qwen3-30B-A3B, four nodes × eight H100 GPUs, policy HybridEP EP8, generation TP2 and `moe_backend: flashinfer_trtllm`. The run manifest and logs are under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_pr4805_cw_h100/results/qwen3-30ba3b-bf16-flashinfer-trtllm-4n8g-2step-20261021T223226Z/`.

After the run, record the resolved vLLM MoE backend, generation and policy steps, refit outcome, Slurm exit code, and any error here.
