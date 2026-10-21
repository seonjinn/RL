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

## Usage

```bash
# once: stage the nightly image (about 15 min), then clone on lustre
sbatch experiments/specdec_perf_20261020/stage_nightly.sbatch

# first job on a fresh checkout builds the worker venvs
FORCE_REBUILD_VENVS=true experiments/specdec_perf_20261020/submit.sh \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml q30-4n4g-baseline-gate 2

# later jobs reuse venvs/
experiments/specdec_perf_20261020/submit.sh \
  experiments/specdec_perf_20261020/configs/qwen3_30ba3b/frozen_dflash_k7.yaml q30-4n4g-frozen-dflash-k7 20
```

Every run passes `checkpointing.enabled=false`. Comparisons use 20 steps.

## Non-obvious details

- Jobs run with `HF_HUB_OFFLINE=1` and no `HF_TOKEN`. The login-shell token is revoked and returns 401 even for public repos. Model and dataset come from the `$U/hf_home` cache.
- Online arms fail `MasterConfig` validation unless five Eagle3-only keys are removed from the base `policy.draft` block (`token_chunk_size`, `learning_rate`, `ttt_steps`, `ttt_step_loss_decay`, `train_embed_and_head`). The Megatron DFlash/DSpark draft configs forbid extra keys. `submit.sh` adds the `~policy.draft.*` deletes for any config that uses a `draft_online_*` fragment.
- Leave the target attention backend on auto (FlashInfer with TRTLLM attention). FULL verify graphs need its UNIFORM_BATCH support. Otherwise vLLM drops to PIECEWISE with a "not supported with spec-decode" warning, so grep the startup log for it.
- DSpark runs only on vLLM's Model Runner V2. A custom logits processor or another V2-unsupported feature silently selects V1, which breaks DSpark and moves DFlash to the V1 proposer.
- Only one job may run with `FORCE_REBUILD_VENVS=true` per checkout. Each node's builder deletes the shared `venvs/` tree first, so concurrent rebuilds corrupt each other.
- `--segment` equals the node count, so all 16 ranks sit in one NVL72 domain. The HybridEP env vars in the recipe assume that.
- Compare arms on one shared closed step window (steps 3 to 20), using arithmetic means of the logged per-step values. Synchronous keys: `timing/train/generation`, `timing/train/prepare_for_generation/total` (refit), `timing/train/policy_training`, `timing/train/policy_and_reference_logprobs`, `timing/train/total_step_time`, `performance/generation_tokens_per_sec_per_gpu`, `performance/tokens_per_sec_per_gpu`.
- Before trusting a speedup, check reward, entropy, KL, acceptance and mean generated length per arm.