# Qwen3-235B-A22B Performance SpecDec Submission Status 2026-06-17

Scope: Qwen3-235B-A22B only. I checked local `latest_*` trackers, docs CSV/MD, relevant local launch scripts, the remote OCI-HSG NeMo-RL checkout, and live/accounting Slurm state on `oci-hsg-cs-001-vscode-02` and `login-lyris`. I did not submit jobs or modify launcher files.

## Existing Sync Jobs

These jobs already exist on OCI-HSG and are based on the official NeMo-RL performance recipe:

`examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml`

The local status source is `docs/nemorl_performance_config_resubmit_status_20260617.{md,csv}`. Live `squeue`/`sacct` on `oci-hsg-cs-001-vscode-02` showed:

| Job ID | Job name | Method | State | Reason | Basis |
|---|---|---|---|---|---|
| 3365679 | `online-qwen235b-perfcfg-baseline-step20-rerun-20260617-N32xG4` | baseline | PENDING | Priority | `grpo-qwen3-235b-16n8g.yaml`, OCI-HSG 32x4 scheduler compatibility |
| 3365680 | `online-qwen235b-perfcfg-eagle3-step20-rerun-20260617-N32xG4` | Eagle-3 | PENDING | Priority | same base config plus Eagle-3 `policy.generation.vllm_kwargs.speculative_config.*` |
| 3365681 | `online-qwen235b-perfcfg-suffix-step20-rerun-20260617-N32xG4` | suffix K32 | PENDING | Priority | same base config plus suffix `policy.generation.vllm_kwargs.speculative_config.*` |

These are sync runs. The local helper used by the multimodel launcher appends `policy.generation.vllm_cfg.async_engine=false` and `grpo.async_grpo.enabled=false`.

## Async-1off Evidence

The remote OCI-HSG checkout at `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613` is at HEAD `231462c16f306ec5429d1841b353720a511064ed` and contains:

- `examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n8g-async-1off.yaml`
- `examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml`
- `tests/test_suites/llm/performance/grpo-qwen3-235b-32n8g-async-1off.sh`

I found remote `slurm-2871589.out`, `slurm-2871656.out`, and related rollout-capture smoke artifacts using `grpo-qwen3-235b-32n8g-async-1off.yaml`, but those commands do not add `policy.generation.vllm_kwargs.speculative_config.*`; they are rollout-capture smokes, not async-1off SpecDec performance submissions.

Live/accounting checks on `login-lyris` showed no active or accounting records for Qwen3-235B `perfcfg` async-1off SpecDec jobs. Lyris has many older standalone vLLM and SWE-RL Qwen3-235B SpecDec trackers, but those do not inherit the official NeMo-RL performance configs and are out of scope for this check.

## Missing Variants

- Sync: baseline, Eagle-3, and suffix already have pending OCI-HSG jobs. PARD/PARD2 are not present in the performance-config rerun status pair if those are required as additional sync SpecDec cells.
- Async-1off: no submitted Qwen3-235B-A22B SpecDec performance jobs found for the official `grpo-qwen3-235b-32n8g-async-1off.yaml` basis. Baseline comparator, Eagle-3, and suffix are missing for that basis; PARD/PARD2 are also missing if included in the async SpecDec matrix.

## Safest Next Action

Do not submit with `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` for async-1off as-is, because it forces sync mode. The safest next step is a model-specific dry run using the upstream async performance test entrypoint plus explicit Hydra SpecDec overrides, or a copied model-specific wrapper that does the same. Keep shared launchers untouched.

Recommended preflight shape before any `sbatch`:

```bash
ssh oci-hsg-cs-001-vscode-02 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-mathrl-20260613 && \
TEST_DRYRUN=1 bash tests/test_suites/llm/performance/grpo-qwen3-235b-32n8g-async-1off.sh \
  cluster.num_nodes=32 cluster.gpus_per_node=4 \
  logger.wandb_enabled=False checkpointing.enabled=False \
  ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \
  ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=3 \
  ++policy.generation.vllm_kwargs.speculative_config.model=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba \
  ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=4'
```

After that dry run, submit through the cluster's normal test-suite Slurm wrapper or a model-specific copied wrapper, not through the sync online SpecDec launcher.
