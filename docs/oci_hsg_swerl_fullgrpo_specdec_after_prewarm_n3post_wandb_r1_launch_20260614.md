# OCI-HSG SWE-RL Full-GRPO SpecDec After-Prewarm N3Post W&B Retry Launch - 2026-06-14

Submitted Qwen3-235B SWE-RL Full-GRPO SpecDec matrix on OCI-HSG with account `nemotron_n3_post`, after the previous N3Post matrix failed before step 1 because W&B had no API key.

Run ID: `20260614_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1`

Tracker: `latest_oci_hsg_swerl_qwen235b_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_20260614_jobs.csv`

Status: `docs/oci_hsg_swerl_fullgrpo_specdec_after_prewarm_n3post_wandb_r1_status_20260614.md`

Launch settings:

- Host: `oci-hsg-cs-001-vscode-02`
- Account: `nemotron_n3_post`
- Partition: `batch`
- Walltime: `04:00:00`
- Shape: 16 nodes, 4 GPUs per node, `--segment=16`
- Max steps: `10`
- Methods: baseline, suffix K32, PARD K5, PARD-2 K1, Eagle-3 K3
- Target model: `Qwen/Qwen3-235B-A22B-Thinking-2507`
- Prewarm cache: `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/nemo_rl/Qwen/Qwen3-235B-A22B-Thinking-2507/iter_0000000/run_config.yaml`

Jobs:

| job_id | method | K | first state | latest observed state |
| --- | --- | ---: | --- | --- |
| 3299487 | baseline | 0 | RUNNING | COMPLETED after `03:39:53` |
| 3299488 | suffix | 32 | RUNNING | FAILED after `00:04:59` |
| 3299489 | PARD | 5 | RUNNING | COMPLETED after `03:41:47` |
| 3299490 | PARD-2 | 1 | PENDING | FAILED after `00:04:35` |
| 3299491 | Eagle-3 | 3 | PENDING | COMPLETED after `03:49:50` |

Latest refresh:

- Refreshed at `2026-06-14T04:20-07:00` from `squeue`/`sacct` and local/remote logs.
- Baseline `3299487`, PARD K5 `3299489`, and Eagle-3 K3 `3299491` completed under account `nemotron_n3_post`.
- Baseline `3299487` logged `Async GRPO training complete!` after checkpointing step 9. Parsed step 2-9 mean total step time is `1243.29s`, E2E throughput `103.75` tok/s/GPU, generation-worker throughput `213.06` tok/s/GPU.
- PARD K5 `3299489` logged `Async GRPO training complete!` after checkpointing step 6. Parsed step 2-6 mean total step time is `1952.39s`, E2E throughput `27.38` tok/s/GPU, generation-worker throughput `57.07` tok/s/GPU.
- Eagle-3 K3 `3299491` completed setup (`Total setup time: 775.5s`) and logged `Async GRPO training complete!` after checkpointing step 8. Parsed step 2-8 mean total step time is `1389.83s`, E2E throughput `50.08` tok/s/GPU, generation-worker throughput `103.47` tok/s/GPU.
- Suffix K32 `3299488` failed in vLLM worker initialization with `ModuleNotFoundError: No module named 'arctic_inference.suffix_decoding._C'`, then `RuntimeError: Engine core initialization failed`.
- PARD-2 K1 `3299490` failed in vLLM worker initialization because the staged official PARD-2 vLLM extension `_C.abi3.so` has a Torch/C10 ABI mismatch: `undefined symbol: _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib`.
- W&B auth passed in the retry; these are no longer the earlier `No API key configured` failures.
- Live metrics are recorded in `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_live_metrics_20260614.md` and `docs/oci_hsg_swerl_fullgrpo_n3post_wandb_r1_live_summary_20260614.csv`.

Notes:

- `WANDB_API_KEY` was passed only as a transient submission environment variable. It is not persisted in tracker files, docs, or scripts.
- Downloaded local logs are redacted by `scripts/fetch_swerl_fullgrpo_logs_20260613.sh` immediately after `rsync`; the local secret-pattern scan is clean after the latest fetch.
- OCI remote inspection found the suffix Arctic site has `_C.cpython-312-aarch64-linux-gnu.so`, but the vLLM actor env is Python 3.13. A py3.13 Arctic site is needed before retrying suffix.
- The PARD-2 retry needs a vLLM native build compatible with the Rui SWE-RL container's Torch/C10 ABI; the existing staged official site is not compatible.
- PARD-2 overlay follow-ups were submitted under account `nemotron_n3_post`: r1 `3299678` failed on missing `vllm._C_stable_libtorch`, r2 `3299698` failed on missing `_C::scaled_fp4_quant.out`, r3 `3299797` failed because overlay `vllm.vllm_flash_attn` shadowed the base compiled extensions, and r4 `3299863` progressed past that point but failed on a remaining direct `torch.ops._C.scaled_fp4_quant.out` reference in `vllm/compilation/passes/fusion/matcher_utils.py`.
- PARD-2 r5 `3299946` failed under account `nemotron_n3_post` after `00:20:46`. Its Python overlay removed stale native objects, used the base container flash-attn package, made `_C_stable_libtorch` optional, and guarded the remaining FP4 fusion `.out` lookups. It progressed through PARD-2 vLLM V1 engine init, 235B target checkpoint loading, PARD-2 drafter model loading, target layer selection `(94, 87, 79, 71)`, torch.compile, and KV cache profiling, then failed at OpenAI serving setup with `TypeError: OpenAIServingChat.__init__() missing 1 required keyword-only argument: 'openai_serving_render'`.
- PARD-2 r6 `3300101` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_openairender_n3post_r6`. It patched chat serving and progressed through target/drafter loading, target layer selection `(94, 87, 79, 71)`, torch.compile, and KV cache profiling, then failed after `00:19:59` with `TypeError: OpenAIServingTokenization.__init__() missing 1 required positional argument: 'openai_serving_render'`.
- PARD-2 r7 `3300266` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_openairender_tokenization_n3post_r7`. It keeps the r5/r6 Python overlay and additionally passes `openai_serving_render` to `OpenAIServingTokenization` when required by the newer vLLM constructor. It passed the r6 failure point, completed setup in `1141.9s`, and started rollout collection, but then stayed at `0/32` while OpenAI endpoint requests returned `500`; root stack was vLLM `EngineCore` dying in `take_draft_token_ids()` after Ray collective RPC timed out waiting for a worker actor. It was cancelled after `00:31:39` to free the allocation.
- PARD-2 r8 `3300390` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_raycgraph_shm_lowconc_n3post_r8`. It kept the r7 compatibility patch and added Ray/vLLM runtime env pins, `RAY_CGRAPH_get_timeout=7200`, `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm`, and lower SWE agent concurrency (`128`). It reached setup completion and rollout start, then failed with repeated `EngineDeadError`/OpenAI `500`s; the root marker was Ray cluster `2.54.0` versus vLLM EngineCore worker process `2.55.1`. It was cancelled after `00:25:54`.
- PARD-2 r9 `3300540` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_ray254_lowconc_n3post_r9`. It kept the r8 SHM/compiled-DAG and SWE-agent concurrency settings and pinned `RAY_VERSION=2.54.0`, but did not pass the r8 patched PARD-2 vLLM overlay path. It therefore reverted to the default official PARD-2 vLLM site and failed after `00:04:09` on the old `_C.abi3.so` Torch/C10 ABI mismatch.
- PARD-2 r10 `3300565` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_ray254_lowconc_n3post_r10`. It combined the r8 patched PARD-2 vLLM overlay with `RAY_VERSION=2.54.0`, SHM compiled-DAG settings, and lower SWE-agent concurrency (`128`). It passed r9's immediate ABI-failure point and loaded checkpoints, but repeated the Ray mismatch (`cluster Ray 2.54.0`, worker process Ray 2.55.1`) because the command still retained `NRL_FORCE_REBUILD_VENVS=false`; it was cancelled after `00:15:29` to free 16 nodes.
- PARD-2 r11 `3300648` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_ray254_freshvenv_lowconc_n3post_r11`. It kept the patched overlay and Ray/SHM/low-concurrency settings, added `NRL_FORCE_REBUILD_VENVS=true`, `NEMO_RL_VENV_DIR=/opt/ray_venvs_swerl_ray254_r11`, and forwarded Ray/fresh-env controls through `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`. It reached PARD-2 checkpoint loading but repeated the stale `.actor_venvs/qwen17b_pard_static_step20_retry1` Ray `2.54.0`/`2.55.1` mismatch and was cancelled before rollout or step 1.
- PARD-2 r12 `3302209` was submitted under account `nemotron_n3_post` with run id `20260614_oci_hsg_swerl_qwen235b_fullgrpo_pard2_pyoverlay_envroot_ray254_freshvenv_lowconc_n3post_r12`. It adds OCI-HSG remote NeMo-RL env-root fixes for actor runtime env and vLLM internal Ray workers, uses `/opt/ray_venvs_swerl_ray254_r12`, and is currently pending.
- The OCI after-prewarm wrapper now fails locally when `SUBMIT=true` and `WANDB_API_KEY` is missing, because this Rui SWE-RL config has `logger.wandb_enabled=True`.
- `SBATCH_DEPENDENCY` was explicitly empty. Prewarm job `3291097` had already completed and Slurm rejected a stale `afterok:3291097` dependency on the previous submit attempt.
- Dry-run diagnostics are redacted in `tmp/oci_hsg_swerl_after_prewarm_n3post_wandb_r1_dryrun_20260614.redacted.log`.
