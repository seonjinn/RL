# Qwen3-30B-A3B refit NSys analysis

This experiment isolates the remaining BF16-to-MXFP8 refit overhead before any
optimization is selected. The BF16 and MXFP8 arms use the same Qwen3-30B-A3B
recipe, four-node allocation, NCCL Reshard transport, CUDA Graph mode, and
FlashInfer TRTLLM MoE backend. The only intended difference is the vLLM
destination precision.

The profile labels these refit phases separately:

- trainer expert stacking;
- NCCL Reshard send and receive;
- receiver BF16-to-MXFP8 quantization;
- MXFP8 value and scale copies;
- miscellaneous parameter transfer;
- synchronization, cache release, and vLLM finalization.

Detailed ranges are opt-in through `NRL_REFIT_NVTX_DETAIL=1`. Normal runs do
not emit them. NSys captures step 2 only, after one warm-up step.

## Correctness gates

Profiling does not change tensor values or the refit algorithm. Before using a
future optimization, require all of the following:

1. MXFP8 value and scale parity against the current receiver conversion.
2. Finite generation, reward, and KL metrics across a matched 20-step run.
3. The expected quantization scope: routed expert FC1/FC2 only.
4. Stable repeated refits with CUDA Graph enabled.
5. A matched end-to-end speedup outside the profiler.

## Launch

Set the standard `tools/launch` environment (`CONTAINER`, `ACCOUNT`,
`PARTITION`, `HF_HOME`, and `HF_DATASETS_CACHE`) and a shared result path, then
launch both tracked wrappers:

```bash
export REFIT_NSYS_RESULTS_ROOT=/shared/path/to/refit-nsys-results
tools/launch \
  experiments/qwen3_30ba3b_refit_nsys/run_bf16.sh \
  experiments/qwen3_30ba3b_refit_nsys/run_mxfp8.sh
```

The `.nsys-rep` files are copied into the SLURM job log tree by Ray log sync.
Use the labeled ranges to choose one optimization only after the BF16 and MXFP8
profiles have been compared.
