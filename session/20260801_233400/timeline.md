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

## 2026-08-02 Post-NCCL Receiver A/B

- Jobs `487298` and `487299` completed 20/20 steps with exit code `0:0`.
- Held trainer prequantization and NCCL exact-transfer constant; changed only batched MoE shuffle and loader-route caching.
- Steps 3-20 refit improved from `4.138 s` to `0.887 s`: `-78.6%`, `4.67x` faster.
- E2E improved from `175.72 s` to `172.10 s`; throughput improved from `1178.97` to `1205.04 tok/s/GPU`.
- Reward and generation-KL paired confidence intervals included zero.
- Initial jobs `487293` and `487294` failed because the container actor venv lacked vLLM 0.25 `routed_experts`; source-managed actor environments fixed the mismatch.
- Reviewed generic cross-precision support. The current implementation is MXFP8-specific and needs fail-closed validation plus a typed transform-plan/codec abstraction before adding NVFP4 or other storage pairs.
