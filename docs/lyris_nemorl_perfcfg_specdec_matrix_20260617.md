# Lyris NeMo-RL Performance Config SpecDec Matrix 2026-06-17

## Status

Submitted 24 NeMo-RL jobs on Lyris for performance-config based Math RL measurement.
The first submission (`2149407-2149430`) reached Ray startup but failed at the Python driver import stage because the container default venv lacked `tensordict`.
Those remaining jobs were cancelled, and the matrix was resubmitted as `20260617_lyris_nemorl_perfcfg_specdec_uvdriver_r1` with `USE_SYSTEM_ENV=false` so the driver runs through the repo `uv` environment.

Current retry job range: `2149644-2149667`.

Latest retry refresh: all retry jobs were still `PENDING` immediately after submission. Representative `scontrol show job -dd` checks confirmed no GPU/GRES TRES request and the intended `SegmentSize` values:

| Job | Shape | SegmentSize | ReqTRES |
|---:|---|---:|---|
| 2149644 | qwen30ba3b sync, 4 nodes | 4 | CPU/mem/node only |
| 2149656 | qwen32 async-1off, 8 nodes | 8 | CPU/mem/node only |
| 2149660 | qwen235b sync, 16 nodes | 16 | CPU/mem/node only |
| 2149664 | qwen235b async-1off, 32 nodes | 16 | CPU/mem/node only |

| Model | Mode | Methods | Job IDs | First status |
|---|---|---|---|---|
| qwen30ba3b | sync | baseline, pard, eagle3, suffix | 2149407-2149410 | PENDING, Resources |
| qwen30ba3b | async1off | baseline, pard, eagle3, suffix | 2149411-2149414 | PENDING, Resources |
| qwen32 | sync | baseline, pard, eagle3, suffix | 2149415-2149418 | PENDING, Resources |
| qwen32 | async1off | baseline, pard, eagle3, suffix | 2149419-2149422 | PENDING, Resources |
| qwen235b | sync | baseline, pard, eagle3, suffix | 2149423-2149426 | PENDING, Priority/Resources |
| qwen235b | async1off | baseline, pard, eagle3, suffix | 2149427-2149430 | PENDING, Priority/Resources |

Full manifest: `latest_lyris_nemorl_perfcfg_specdec_matrix_20260617_jobs.csv`.

## Run Setup

Remote host: `login-lyris`

Remote repo: `/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-main-mathrl-20260613`

Remote HEAD at preflight: `231462c16`

Container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-nightly-ultra.sqsh`

Slurm: `account=coreai_dlalgo_llm`, `partition=gb200`, no GRES flag, `--comment=metrics`

Runtime env: `USE_SYSTEM_ENV=false`, `RAY_USE_EXISTING_ENV=true`, `NEMO_RL_VENV_DIR=/opt/ray_venvs`, driver venvs under `/project`.

GB200 segment policy:

| Model | Mode | Nodes | Segment |
|---|---:|---:|---:|
| qwen30ba3b | sync | 4 | 4 |
| qwen30ba3b | async1off | 4 | 4 |
| qwen32 | sync | 4 | 4 |
| qwen32 | async1off | 8 | 8 |
| qwen235b | sync | 16 | 16 |
| qwen235b | async1off | 32 | 16 |

The 32-node qwen235b async-1off rows use two 16-node segments rather than `--segment=32`, because Lyris segment sizes must stay within the GB200 placement limit and divide the requested node count.

Output/cache root: `/project/coreai_dlalgo_llm/users/sna/nemorl_perfcfg_specdec/20260617_lyris_nemorl_perfcfg_specdec_uvdriver_r1`

Sampling/measurement overrides:

- `temperature=1.0`
- `top_p=1.0`
- `top_k=-1`
- `max_new_tokens=1024`
- `min_tokens=1024`, `ignore_eos=true`, stop strings/token IDs disabled
- `max_num_steps=20`

Recipe batch settings preserved:

- qwen30ba3b/qwen32: `num_prompts_per_step=64`, `num_generations_per_prompt=32`, `train_global_batch_size=512`
- qwen235b: `num_prompts_per_step=16`, `num_generations_per_prompt=32`, `train_global_batch_size=512`

## Config Basis

| Model | Sync config | Async-1off config |
|---|---|---|
| qwen30ba3b | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml` |
| qwen32 | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml` | `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml` |
| qwen235b | `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml` | `examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off.yaml` |

## Notes

- The helper `experiments/eagle3_online/submit_nemorl_online_draft_specdec.sh` now supports preserving recipe async and sequence-packing settings through `PRESERVE_RECIPE_ASYNC=true` and `PRESERVE_RECIPE_SEQUENCE_PACKING=true`.
- Lyris `/lustre/fsw` user inode quota is at the hard limit, so new log/checkpoint/driver-venv outputs are placed under `/project`. The repo, model snapshots, arctic suffix site, and container are still read from `/lustre`.
- PARD-2 is intentionally not included in this default matrix because the cached qwen30ba3b PARD-2 drafter is not shape-compatible. The submitted matrix covers baseline, PARD K3, Eagle-3 K3, and suffix K32.
