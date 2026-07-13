# NeMo-RL Performance-Config Resubmit Status 2026-06-17

Scope: Agent B owned only this status pair and remote SLURM submissions. I did not touch vLLM standalone jobs, SWE-RL suffix jobs, or completed Qwen3-30B-A3B/Qwen3-32B rows.

## Submitted Reruns

| Job ID | Run label | Method | Status at submit | Source failed job | Log dir |
|---|---|---|---|---|---|
| 3365679 | qwen235b-perfcfg-baseline-step20-rerun-20260617 | baseline | PENDING, Priority | 3334220 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_performance_config_resubmit_logs/20260617_mathrl_qwen235b_perfcfg_step20_rerun_r1/baseline` |
| 3365680 | qwen235b-perfcfg-eagle3-step20-rerun-20260617 | Eagle-3 | PENDING, Priority | 3333537 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_performance_config_resubmit_logs/20260617_mathrl_qwen235b_perfcfg_step20_rerun_r1/eagle3` |
| 3365681 | qwen235b-perfcfg-suffix-step20-rerun-20260617 | suffix K32 | PENDING, Priority | 3333717 | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/mathrl_performance_config_resubmit_logs/20260617_mathrl_qwen235b_perfcfg_step20_rerun_r1/suffix` |

## Basis

Base config for all three submissions:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613/examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml`

Remote repo HEAD verified before submission:

`231462c16f306ec5429d1841b353720a511064ed`

Common command basis: official Qwen3-235B performance YAML, local `/lustre` Qwen3-235B snapshot, step20 OSL1024/temp1 measurement overrides, W&B disabled, checkpointing disabled, and `NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=3600`.

SpecDec command basis:

- Eagle-3 added only `policy.generation.vllm_kwargs.speculative_config.method=eagle3`, `num_speculative_tokens=3`, `model=<Qwen3-235B-A22B-Eagle3 snapshot>`, and `draft_tensor_parallel_size=4`.
- Suffix added only `policy.generation.vllm_kwargs.speculative_config.method=suffix`, `num_speculative_tokens=32`, and suffix decoding tree/cache/probability keys. The arctic site was added to `PYTHONPATH` only so the suffix implementation can import.

## Source Evidence

- Baseline `3334220` failed after `01:14:56`, with `8/20` completed steps. Failure was Megatron NCCL watchdog timeout during Step 9 policy training.
- Eagle-3 `3333537` failed after `01:21:59`, with `14/20` completed steps. Failure was Megatron NCCL watchdog timeout during Step 15 policy training.
- Suffix `3333717` failed after `01:24:00`, with `14/20` completed steps. Failure was Megatron NCCL watchdog timeout during Step 15 policy training.
- Qwen3-30B-A3B and Qwen3-32B baseline/Eagle-3/suffix rows were verified with `sacct` as `COMPLETED 0:0`, so I did not duplicate them.

## Blockers And Caveats

- Literal unmodified `16n8g` scheduling is blocked on OCI-HSG: `batch` nodes expose `gpu:4`, so these submissions use the necessary OCI-HSG scheduler compatibility shape `32` nodes x `4` GPUs. This is called out in the CSV.
- Jobs are pending for priority and have not started GPU execution yet.
- No `WANDB_API_KEY` was needed or printed.
