# PR #4805: Qwen3-30B-A3B BF16 FlashInfer TRTLLM on CW H100

This historical launcher should not be resubmitted on CW H100. vLLM 0.29.0's BF16 FlashInfer TRTLLM MoE kernel requires Blackwell SM100, while H100 is SM90. Job `20642814` also stopped earlier at a dataset-cache permission error, so it did not exercise the kernel. See [REPORT.md](REPORT.md).

PR #4805 changes the `grpo-qwen3-30ba3b-4n8g.yaml` vLLM MoE kernel from `triton` to `flashinfer_trtllm`. The submitted experiment used that merged r0.8 recipe on four CW nodes with eight H100 GPUs each. It retained the recipe's HybridEP H100 topology, generation TP2, BF16 precision, 4,096-token total sequence limit, and batch sizes. The requested two training steps would have exercised generation, policy training, and at least one post-training weight refit if startup had succeeded.

The runtime image is the previously verified `r0.8.0-19666f86c` squashfs. The vLLM refit and worker source files are identical between that image's source commit and the PR merge commit; the PR changes recipes only. The checkout's merged recipe is read through the `/lustre` bind mount. The shared HF cache contains the model checkpoint.

On the CW login node, from a clean checkout of this branch under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL-r0p8-pr4805-cw-h100-20261021`:

```bash
bash experiments/r0p8_pr4805_cw_h100/submit.sh
```

The launcher runs `git pull --ff-only` and `sbatch --test-only` before submission. It records a manifest and driver log under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_pr4805_cw_h100/results/`, with Slurm output under `slurm/` and Ray logs under `ray_logs/`.
