# PR #4805: Qwen3-30B-A3B BF16 FlashInfer TRTLLM on CW H100

PR #4805 changes the `grpo-qwen3-30ba3b-4n8g.yaml` vLLM MoE kernel from `triton` to `flashinfer_trtllm`. This experiment runs that merged r0.8 recipe on four CW nodes with eight H100 GPUs each. It retains the recipe's HybridEP H100 topology, generation TP2, BF16 precision, 4,096-token total sequence limit, and batch sizes. Two training steps exercise generation, policy training, and at least one post-training weight refit.

The runtime image is the previously verified `r0.8.0-19666f86c` squashfs. The vLLM refit and worker source files are identical between that image's source commit and the PR merge commit; the PR changes recipes only. The checkout's merged recipe is read through the `/home` bind mount. The shared HF cache contains the model checkpoint.

On the CW login node, from a clean checkout of this branch:

```bash
bash experiments/r0p8_pr4805_cw_h100/submit.sh
```

The launcher runs `git pull --ff-only` and `sbatch --test-only` before submission. It records a manifest and driver log under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_pr4805_cw_h100/results/`, with Slurm output under `slurm/` and Ray logs under `ray_logs/`.
