# Timeline

## 2026-08-01 23:34 PDT

- Verified PR 3294's primary GCP Qwen3-30B 20-step performance and seven-arm ablation are complete.
- Confirmed some wider PR 3294 coverage remains incomplete: matched post-NCCL 235B legacy A/B and Async 235B baseline.
- Verified separate worktree commit `45cfb8916` and remote GCP checkout match.
- Confirmed unit job `486916` and correctness job `486926` completed successfully.
- Monitored performance jobs `486954` and `486955`; no fatal errors through step 11.
- Verified the runtime selects Python exact-transfer because `nccl.m2n` and `libnccl_m2n.so` are absent.

## 2026-08-02 00:02 PDT

- Jobs `486954` and `486955` completed 20/20 steps with exit code `0:0`.
- Step 3-20 refit improved from `4.7956 s` to `0.7867 s`: `-83.6%`, `6.10x` faster.
- E2E improved from `175.61 s` to `168.23 s`; throughput improved from `1179.61` to `1231.90 tok/s/GPU`.
- Reward and generation-KL paired confidence intervals included zero.
- Wrote `experiments/nccl_reshard_pr3294/RESULTS.md`.
