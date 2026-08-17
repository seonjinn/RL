# Plan

1. Validate the launcher contract locally.
2. Commit and push the exact experiment source.
3. Run `sbatch --test-only` on Lyris and Ptyche and select the earlier start.
4. Run sequential one-step Qwen BF16 and MXFP8 smoke tests.
5. Confirm the logs select `IPCWeightSynchronizer`, CUDA Graph capture succeeds,
   the MXFP8 scope contains only routed experts, and all metrics are finite.
6. Run sequential 20-step Qwen BF16 and MXFP8 measurements.
7. Repeat the validated contract for Nemotron3 Nano.
8. Compare steady-state steps using E2E, generation, policy, logprob, refit,
   reward, and generation KL metrics.

