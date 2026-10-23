# CW r0.8 Qwen3 async Triton run

## Submission

Both jobs were submitted on 2026-10-22 from source commit `ae63c3183484f24d4788543333a1cb52d9ac1286`, based on r0.8 commit `99ddc5ad711a0bbea6b145c3a8b4f062ef2b410c`. `sbatch --test-only` succeeded before each submission. Runtime image: `r0.8.0-b0617dab1`; the mounted r0.8 checkout is selected through `PYTHONPATH` and checked at driver startup. Both jobs use `coreai_dlalgo_nemorl`, `batch`, eight H100 GPUs per node, a four-hour time limit, 10 steps, and `++policy.generation.vllm_kwargs.moe_backend=triton`.

| Workload | Job ID | Nodes | Initial scheduler state | Run directory |
| --- | ---: | ---: | --- | --- |
| Qwen3-32B async 1-off | `20697714` | 8 | PENDING (Priority) | `results/qwen3-32b-8n8g-async-1off-triton-10step-20261022T215253Z` |
| Qwen3-235B-A22B async 1-off | `20697737` | 32 | PENDING (Priority) | `results/qwen3-235b-32n8g-async-1off-triton-10step-20261022T215306Z` |

Run directories above are under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_qwen_async_triton_20261022/`. They contain `manifest.txt` and, after startup, `run.log`; Slurm and Ray logs are under the same experiment root.

## Checks and interpretation

Local recipe composition confirmed the override resolves to `triton`, with 8 and 32 nodes and async GRPO enabled. `shellcheck`, `bash -n`, YAML parsing, and `git diff --check` passed. The user-owned datasets cache is writable and already contains OpenMathInstruct-2; both model snapshot configs and the image are readable on CW.

Qwen3-32B is a dense model, so its `moe_backend` setting does not select an expert kernel; its attention backend remains at the recipe/vLLM default. For Qwen3-235B-A22B, the override selects Triton MoE experts. Runtime success is not yet established: both jobs were pending at the first scheduler check. Check for training steps, async rollout/refit progress, and absence of the reported vLLM `Worker_TP3` illegal memory access and HybridEP `executor.cu:188` `cudaMemcpyAsync` error after the jobs start.

## First attempt

Both jobs were moved while pending to `coreai_dlalgo_llm`. Job `20697714`
started but failed before the first training step with Ray serialization error
`ModuleNotFoundError: No module named 'megatron'`. The launcher had replaced
`PYTHONPATH` with the mounted checkout path, hiding the container's Megatron-LM
source path. Job `20697737` was canceled while initializing to avoid GPU waste.
The launcher now preserves the Megatron-LM path and checks `import megatron`
before starting GRPO. Replacement job IDs will be recorded below.

## Replacement jobs

Jobs `20706885` (32B) and `20706912` (235B) were submitted after the path
fix. The launcher still contained the old Slurm account constant at submission;
both jobs were changed while pending to `coreai_dlalgo_llm`. The launcher and
cluster configuration now use that account for future submissions. Runtime
results are pending.

## Hub API failure and offline-cache retry

Jobs `20706885` (32B) and `20706912` (235B) passed the Megatron import
preflight, then failed before their first training step when vLLM requested a
Hugging Face repository file list and received HTTP 429. Both snapshot
`config.json` files were already present in the shared cache. The launcher now
sets `HF_HUB_OFFLINE=1` so vLLM uses cached files. For 235B it also sets
`NRL_REFIT_BUFFER_MEMORY_RATIO=0.005`, matching the NCCL comparison arm.

Replacement jobs `20707493` (32B, source `785e9b9de`) and `20707495` (235B,
same source) were submitted after `sbatch --test-only` succeeded. Both have
four-hour limits and use the `coreai_dlalgo_llm` account. Job `20707493` was
unexpectedly shown under `coreai_dlalgo_nemorl` immediately after submission
and was changed to `coreai_dlalgo_llm` while pending. Their run directories
are `results/qwen3-32b-8n8g-async-1off-triton-10step-20261023T001705Z` and
`results/qwen3-235b-32n8g-async-1off-triton-10step-20261023T001711Z`.
Job `20707493` reached its first generation refit. Job `20707495` also
completed the initial refit, then was canceled before a comparable training
result because its NCCL comparison arm still used the default 2% refit buffer
inside vLLM despite the exported launcher variable. The two 235B arms now
pass the 0.5% value explicitly through both
`policy.megatron_cfg.env_vars` and `policy.generation.vllm_cfg.env_vars`.

Replacement 235B job `20708178` was submitted from source `4acbc4bdd` after
`sbatch --test-only` succeeded. Its run directory is
`results/qwen3-235b-32n8g-async-1off-triton-10step-20261023T003301Z`.
It failed before startup because Hydra parsed the unquoted CLI value `0.005`
as a float, while `MasterConfig` requires string environment values. The
driver now passes the quoted Hydra value `"0.005"`; local composition
confirmed a string on both policy and vLLM sides.

Replacement 235B job `20708681` was submitted from source `27382854d`
under `coreai_dlalgo_llm` after `sbatch --test-only` succeeded. Its run
directory is
`results/qwen3-235b-32n8g-async-1off-triton-10step-20261023T004135Z`.
Runtime results remain pending.
