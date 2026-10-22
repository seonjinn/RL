# Llama3.1 FP8 NCCL reshard smoke on CW H100

This 10-step run checks the r0.8.0 release PR's Llama3.1-8B FP8 async 1-off
recipe after enabling blockwise FP8 parameter storage for NCCL reshard.
Submit from this branch's CW checkout with:

```bash
bash experiments/r0p8_llama_fp8_reshard_20261022/submit.sh
```

The launcher checks the model cache and image, pulls the pushed branch, runs
`sbatch --test-only`, and submits to `batch` under `coreai_dlalgo_llm`.
