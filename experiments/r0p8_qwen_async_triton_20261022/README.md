# r0.8 CW Qwen3 async Triton check

Run the Qwen3-32B 8n8g and Qwen3-235B-A22B 32n8g async 1-off release recipes on CW H100 with generation MoE backend `triton`. Each job requests 10 steps, preserves the recipe's training and generation topology, and disables checkpoints and W&B for this diagnostic. The 235B job keeps the nightly suite's `NCCL_NVLS_ENABLE=0`; the 32B job keeps its `NRL_IGNORE_TP_ACCURACY_CHECK=1`.

The runtime image is `r0.8.0-b0617dab1`; the mounted checkout and `PYTHONPATH` select this branch's current r0.8 source. The driver checks the imported `nemo_rl` path before training. Ray actor environments reuse the image's prebuilt `/opt/ray_venvs`; dependencies did not change between the image source and the current release tip. A user-owned Hugging Face datasets cache avoids the permission error seen in CW job `20642814`.

From the clean checkout on the CW login node:

```bash
bash experiments/r0p8_qwen_async_triton_20261022/submit.sh 32b
bash experiments/r0p8_qwen_async_triton_20261022/submit.sh 235b
```

The launcher pulls the branch, checks the model cache and container, runs `sbatch --test-only`, then submits in the `batch` partition under `coreai_dlalgo_llm`. Manifests and run logs are under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_qwen_async_triton_20261022/results/`.
