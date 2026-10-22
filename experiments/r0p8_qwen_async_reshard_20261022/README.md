# r0.8 Qwen async 1-off NCCL reshard comparison on CW H100

This experiment compares the default refit transport with
`policy.generation.refit_transport=nccl_reshard` for the Qwen3-32B 8n8g and
Qwen3-235B-A22B 32n8g async 1-off recipes. Both arms use the same r0.8.0
source, container, recipe, 10-step limit, and Triton generation override.

The default-transport jobs were submitted separately as `20697714` (32B) and
`20697737` (235B) from source commit `ae63c3183484f24d4788543333a1cb52d9ac1286`.
Their Slurm accounts were changed while pending to `coreai_dlalgo_llm`.
Their manifests remain under
`/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_qwen_async_triton_20261022/results/`.

From this branch's checkout on CW, submit the reshard arms with:

```bash
bash experiments/r0p8_qwen_async_reshard_20261022/submit.sh 32b
bash experiments/r0p8_qwen_async_reshard_20261022/submit.sh 235b
```

`submit.sh` checks the image and model cache, pulls the pushed branch, runs
`sbatch --test-only`, then submits to `batch` under `coreai_dlalgo_llm`.
Results and manifests are written under the corresponding
`r0p8_qwen_async_reshard_20261022` Lustre directory.

Compare completed steps using the console's average reward, generation KL
error, loss, total step time, and refit timing. Exclude startup and the first
training step from steady-state timing. Async rollout sampling can vary between
runs, so a 10-step result can flag a large accuracy regression but cannot
establish equivalent long-run convergence.
