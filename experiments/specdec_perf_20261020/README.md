# SpecDec perf on the draft co-training stack (2026-10-20)

Measures how much DFlash and DSpark speculative decoding speed up rollout and the full GRPO step, on latest main plus all 11 draft co-training PRs (#3701 through #3757). Baseline is the shipped performance recipe, unchanged. SpecDec arms add the speculative settings plus the `max_num_seqs` and CUDA graph capture sizes that SpecDec needs. Online arms also switch training to TP2.

## Arms (Qwen3-30B-A3B, OCI-HSG 4 nodes x 4 GB200)

Sync arms start from `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` (colocated, 16 vLLM engines). Async arms start from `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` (2 train nodes, 2 generation nodes, 8 engines). Generation uses `moe_backend: flashinfer_trtllm` and CUDA graphs on (`enforce_eager: false`) in every arm.

Two SpecDec numbers are easy to confuse. **block_size** (8) is fixed when the drafter is trained and lives in its checkpoint. **K** (`num_speculative_tokens`) is chosen at SpecDec time, and every SpecDec config name ends in `_k<K>`. DFlash arms use K=7 = block_size - 1, because the anchor slot holds the last accepted token and proposes nothing. K=7 was also the earlier 20-step Q30 winner over K5. DSpark arms use K=8 = block_size, because the DSpark anchor slot also predicts the next token. The Megatron `DSparkBatchPlan`, NeMo-RL's draft-runtime K check and the upstream DSpark recipe all use K = block_size. The drafter checkpoint has no `sample_from_anchor` key, which vLLM reads as true.

| Mode | Config | Training layout | Drafter | seqs/engine | Graphs up to | What it isolates |
|---|---|---|---|---|---|---|
| sync | perf recipe as-is | TP1 EP16 | none | 1024 (vLLM default) | 1024 tok (default) | baseline |
| sync | `frozen_dflash_k7` | TP1 EP16 | DFlash, frozen, K=7 | 128 | 1024 tok | rollout speedup from SpecDec alone |
| sync | `frozen_dspark_k8` | TP1 EP16 | DSpark, frozen, K=8 | 128 | 1152 tok | same, DSpark |
| sync | `tp2_baseline` | TP2 SP ETP1 EP16 | none | 1024 (vLLM default) | 1024 tok (default) | baseline for the online layout (online drafting requires target TP2) |
| sync | `online_dflash_k7` | TP2 SP ETP1 EP16 | DFlash, trained and refit every step, K=7 | 128 | 1024 tok | co-training and refit cost on top of SpecDec |
| sync | `online_dspark_k8` | TP2 SP ETP1 EP16 | DSpark, trained and refit every step, K=8 | 128 | 1152 tok | same, DSpark |
| async | async recipe as-is | TP1 EP8 | none | 128 (vLLM async default) | 256 tok (default) | async baseline |
| async | `async_baseline_mbt16k` | TP1 EP8 | none | 128 (vLLM async default) | 256 tok (default) | control for the s128 arms: only the 16384-token scheduler budget |
| async | `async_baseline_s256_mbt16k` | TP1 EP8 | none | 256 | 512 tok (default) | control for the s256 arms: same budget and concurrency |
| async | `async_frozen_dflash_k7_s128`, `async_frozen_dspark_k8_s128` | TP1 EP8 | frozen, K=7 / K=8 | 128 | 1024 / 1152 tok | SpecDec at the baseline's concurrency |
| async | `async_frozen_dflash_k7_s256`, `async_frozen_dspark_k8_s256` | TP1 EP8 | frozen, K=7 / K=8 | 256 | 2048 / 2304 tok | SpecDec with every rollout of a step in flight |
| async | `async_tp2_baseline` | TP2 SP ETP1 EP8, data plane on | none | 128 (vLLM async default) | 256 tok (default) | async baseline for the online layout |
| async | `async_tp2_baseline_mbt16k` | TP2 SP ETP1 EP8, data plane on | none | 128 (vLLM async default) | 256 tok (default) | control for the async online arms |
| async | `async_online_dflash_k7`, `async_online_dspark_k8` | TP2 SP ETP1 EP8, data plane on | trained and refit every step, K=7 / K=8 | 128 | 1024 / 1152 tok | async co-training cost |

Arm files compose fragments through a `defaults:` list: `spec_frozen_*.yaml` or `draft_online_*.yaml` for the drafter, and `capture_k{7,8}_s{128,256}.yaml` for `max_num_seqs` plus `compilation_config.cudagraph_capture_sizes`.

Why the capture fragment matters: a verify step feeds `seqs x (K+1)` tokens to the target. On GB200, vLLM's default capture ceiling is `min(2 x max_num_seqs x (K+1), 1024)` tokens, and its default grid only covers 1, 2, 4, 8, 16, ... requests. A 128-seq K=8 verify step (1152 tokens) and any 256-seq step past 1024 tokens would run without graphs. The fragments capture up to `max_num_seqs x (K+1)`, in multiples of K+1 for every request count up to 32 and every even count above it. Sync SpecDec arms set `max_num_seqs` to 128, the rollouts per engine per step (2048 / 16). The async engine defaults to 128 seqs, so the s128 arms match the async baseline, and the s256 arms get `async_baseline_s256_mbt16k` as their control.

Async SpecDec arms also add `async_mbt16k.yaml` (`max_num_batched_tokens: 16384`). The async engine starts `AsyncLLM` with vLLM's engine usage context, which falls back to a 2048-token scheduler budget. The sync `vllm.LLM` engine gets 16384 on GB200. vLLM 0.29 charges each decode row 2K+1 = 15 budget tokens for DFlash K=7 and 2K = 16 for DSpark K=8 (upstream fix #59468, not in 0.29). At 2048, 256 seqs would decode about 136 rows per step and hit the #56448 startup assert. The `*_mbt16k` controls separate the budget change from SpecDec.

## Long-output variant (32K)

The `seq32k_*` arms rerun the sync and async baselines and the frozen arms with `max_total_sequence_length: 32768`, so responses can reach about 32K tokens. Decode takes a larger share of the step at this length, which is where SpecDec should pay off most. `seq32k.yaml` also turns on full activation recompute and 2048-token logprob chunks, because a 32K packed micro-batch on TP1 needs both (as in `grpo-math-qwen3-30ba3b-megatron-tp4-32k.yaml`). Compare 32K arms only with each other. The async 32K arms use 128 seqs per engine.

| Mode | Config | Drafter | seqs/engine |
|---|---|---|---|
| sync | `seq32k_baseline` | none | 1024 (vLLM default) |
| sync | `seq32k_frozen_dflash_k7`, `seq32k_frozen_dspark_k8` | frozen, K=7 / K=8 | 128 |
| async | `seq32k_async_baseline_mbt16k` | none | 128 (vLLM async default) |
| async | `seq32k_async_frozen_dflash_k7_s128`, `seq32k_async_frozen_dspark_k8_s128` | frozen, K=7 / K=8 | 128 |

KV cache is the main risk. By estimate, a sync engine at `gpu_memory_utilization: 0.6` holds about 0.45M tokens, and an async engine at 0.8 about 0.85M. Long tails can force preemption. Check the logged KV cache size and preemption counts on `seq32k_baseline` before trusting the other 32K arms. Twenty steps may not fit in the 4-hour limit. With checkpointing off, a timed-out arm keeps the steps it logged, and arms compare on their shared closed step window.

Drafters live under `$U/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-{dflash,dspark}/exported-checkpoint-25391`. Both use full attention, target layers `[1, 12, 23, 34, 45]`, and mask token 151669.

## K sweep

Block 8 sets the largest K (DFlash K7, DSpark K8), not the best K. Per-token acceptance is 0.25-0.41 at temperature 1.0, so the last draft slots rarely land but still cost verify tokens. On Qwen3.5 the smaller K won (DSpark K4 over K8). The sweep runs smaller K on Q30 and Nemotron 3.5 Lightning, sync and async, with fresh baselines from the same commit.

| Model | Mode | DFlash K | DSpark K | Control |
|---|---|---|---|---|
| Q30 | sync | 3, 5 (`frozen_dflash_k{3,5}`) | 4, 6 (`frozen_dspark_k{4,6}`) | perf recipe |
| Q30 | async | 3, 5 (`async_frozen_dflash_k{3,5}_s128`) | 4, 6 (`async_frozen_dspark_k{4,6}_s128`) | `async_baseline_mbt16k` |
| N35 | sync | 3 (`frozen_dflash_k3`) | 4 (`frozen_dspark_k4`) | `baseline_sync_nopc` |
| N35 | async | 3 (`async_frozen_dflash_k3_s128`) | 4 (`async_frozen_dspark_k4_s128`) | `async_baseline_mbt16k_nopc` |

Each arm reuses the K7/K8 drafter fragment and overrides only `num_speculative_tokens`, with the capture list for its own K (`capture_k{3,4,5,6}_s128.yaml`). Run names carry the `ksweep` tag. The K7/K8 arms above are the top of each curve.

## DAPO-Math-17k variant

The earlier Q30 runs that reached about 2.2x generation speedup used DAPO prompts and long outputs. The `dapo_*` arms swap OpenMathInstruct-2 for DAPO-Math-17k (`dapo.yaml`: no prompt file, `dapo_math_verify`) and keep everything else. `dapo_{baseline,frozen_dflash_k7,frozen_dspark_k8,async_*}` run at 4K, and `dapo_seq32k_*` at 32K. Compare each DAPO arm only with the DAPO control of the same length and mode. Validation stays a 5% split of the training set, because the runs are HF-offline and the AIME-2024 set is not cached.

## Usage

```bash
# once: stage the nightly image (about 15 min), then clone on lustre
sbatch experiments/specdec_perf_20261020/stage_nightly.sbatch

experiments/specdec_perf_20261020/submit.sh \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml q30-4n4g-baseline-gate 2

experiments/specdec_perf_20261020/submit.sh \
  experiments/specdec_perf_20261020/configs/qwen3_30ba3b/frozen_dflash_k7.yaml q30-4n4g-frozen-dflash-k7 20
```

Every run passes `checkpointing.enabled=false`. Comparisons use 20 steps.

## Non-obvious details

- Jobs run with `HF_HUB_OFFLINE=1` and no `HF_TOKEN`. The login-shell token is revoked and returns 401 even for public repos. Model and dataset come from the `$U/hf_home` cache.
- Online arms fail `MasterConfig` validation unless five Eagle3-only keys are removed from the base `policy.draft` block (`token_chunk_size`, `learning_rate`, `ttt_steps`, `ttt_step_loss_decay`, `train_embed_and_head`). The Megatron DFlash/DSpark draft configs forbid extra keys. `submit.sh` adds the `~policy.draft.*` deletes for any config that uses a `draft_online_*` fragment.
- Leave the target attention backend on auto (FlashInfer with TRTLLM attention). FULL verify graphs need its UNIFORM_BATCH support. Otherwise vLLM drops to PIECEWISE with a "not supported with spec-decode" warning, so grep the startup log for it.
- DSpark runs only on vLLM's Model Runner V2. A custom logits processor or another V2-unsupported feature silently selects V1, which breaks DSpark and moves DFlash to the V1 proposer.
- Every job rebuilds its worker venvs. The nightly image sets `NEMO_RL_VENV_DIR=/opt/ray_venvs`, so venvs are container-local and jobs do not share them. The image's prebuilt venvs pin its own Megatron-Bridge and vLLM. That Bridge imports `te_checkpoint`, which this checkout's Megatron-LM lacks, so a job that skips the rebuild fails at policy-worker init.
- `--segment` equals the node count, so all 16 ranks sit in one NVL72 domain. The HybridEP env vars in the recipe assume that. Set `SEGMENT` for jobs larger than one domain.
- A job writes to lustre only under `exp_logs/<run>` (metrics, wandb files) and `logs/<run>` (Slurm and Ray logs). ray.sub runs the container with `--no-container-mount-home`, so in-container caches (`~/.cache` for vLLM, FlashInfer and Triton, the uv cache at `/opt/nemo_rl_cache/uv`, worker venvs at `/opt/ray_venvs`) sit on node-local disk and go away with the job. The large per-job lustre write was `train_data_step*.jsonl`, about 340 MB per step at 4K. Both submit scripts pass `++env.should_log_nemo_gym_responses=true`, the only switch that turns it off.
- Both submit scripts run `git submodule update --init --recursive` before `sbatch`. Run it yourself after any branch switch or merge, before you read submodule code.
- Compare arms on one shared closed step window (steps 3 to 20), using arithmetic means of the logged per-step values. Synchronous keys: `timing/train/generation`, `timing/train/prepare_for_generation/total` (refit), `timing/train/policy_training`, `timing/train/policy_and_reference_logprobs`, `timing/train/total_step_time`, `performance/generation_tokens_per_sec_per_gpu`, `performance/tokens_per_sec_per_gpu`.
- Before trusting a speedup, check reward, entropy, KL, acceptance and mean generated length per arm.