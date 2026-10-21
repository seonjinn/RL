# SpecDec perf on the draft co-training stack (2026-10-20)

Measures how much DFlash and DSpark speculative decoding speed up rollout and the full GRPO step, on latest main plus all 11 draft co-training PRs (#3701 through #3757). Baseline is the shipped performance recipe, unchanged. SpecDec arms add only the speculative settings.

## Arms (Qwen3-30B-A3B, OCI-HSG 4 nodes x 4 GB200)

Sync arms start from `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` (colocated, 16 vLLM engines). Async arms start from `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` (2 train nodes, 2 generation nodes, 8 engines). Generation uses `moe_backend: flashinfer_trtllm` and CUDA graphs on (`enforce_eager: false`) in every arm.

Two SpecDec numbers are easy to confuse. **block_size** (8) is fixed when the drafter is trained and lives in its checkpoint. **K** (`num_speculative_tokens`) is chosen at SpecDec time. Every arm uses K=7, the earlier 20-step Q30 winner over K5, and every SpecDec config name ends in `_k7`.

| Mode | Config | Training layout | Drafter | seqs/engine | Graphs up to | What it isolates |
|---|---|---|---|---|---|---|
| sync | perf recipe as-is | TP1 EP16 | none | vLLM default | 512 tok (default) | baseline |
| sync | `frozen_dflash_k7` | TP1 EP16 | DFlash, frozen, K=7 | 128 | 1024 tok | rollout speedup from SpecDec alone |
| sync | `frozen_dspark_k7` | TP1 EP16 | DSpark, frozen, K=7 | 128 | 1024 tok | same, DSpark |
| sync | `tp2_baseline` | TP2 SP ETP1 EP16 | none | vLLM default | 512 tok | baseline for the online layout (online drafting requires target TP2) |
| sync | `online_dflash_k7` | TP2 SP ETP1 EP16 | DFlash, trained and refit every step, K=7 | 128 | 1024 tok | co-training and refit cost on top of SpecDec |
| sync | `online_dspark_k7` | TP2 SP ETP1 EP16 | DSpark, trained and refit every step, K=7 | 128 | 1024 tok | same, DSpark |
| async | async recipe as-is | TP1 EP8 | none | vLLM default | 512 tok | async baseline |
| async | `async_frozen_{dflash,dspark}_k7_s128` | TP1 EP8 | frozen, K=7 | 128 | 1024 tok | SpecDec at half the per-step load per engine |
| async | `async_frozen_{dflash,dspark}_k7_s256` | TP1 EP8 | frozen, K=7 | 256 | 2048 tok | SpecDec at the full per-step load per engine |
| async | `async_tp2_baseline` | TP2 SP ETP1 EP8, data plane on | none | vLLM default | 512 tok | async baseline for the online layout |
| async | `async_online_{dflash,dspark}_k7` | TP2 SP ETP1 EP8, data plane on | trained and refit every step, K=7 | 256 | 2048 tok | async co-training cost |

Arm files compose fragments through a `defaults:` list: `spec_frozen_*_k7.yaml` or `draft_online_*_k7.yaml` for the drafter, and `capture_k7_s{128,256}.yaml` for `max_num_seqs` plus `compilation_config.cudagraph_capture_sizes`.

Why the capture fragment matters: a K=7 verify step feeds `seqs x (K+1)` tokens to the target. vLLM only captures CUDA graphs up to 512 tokens by default, so 128 seqs x 8 = 1024 verify tokens would run without graphs. The fragments capture up to `max_num_seqs x 8`, using multiples of 8 so uniform verify batches land on a captured size. `max_num_seqs` equals the rollouts per engine per step (2048 / 16 sync, 2048 / 8 async); async also tries 128 to test whether a smaller verify batch pays off.

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
- Only one job may run with `FORCE_REBUILD_VENVS=true` per checkout. Each node's builder deletes the shared `venvs/` tree first, so concurrent rebuilds corrupt each other.
- `--segment` equals the node count, so all 16 ranks sit in one NVL72 domain. The HybridEP env vars in the recipe assume that.
- Compare arms on one shared closed step window (steps 3 to 20), using arithmetic means of the logged per-step values. Synchronous keys: `timing/train/generation`, `timing/train/prepare_for_generation/total` (refit), `timing/train/policy_training`, `timing/train/policy_and_reference_logprobs`, `timing/train/total_step_time`, `performance/generation_tokens_per_sec_per_gpu`, `performance/tokens_per_sec_per_gpu`.
- Before trusting a speedup, check reward, entropy, KL, acceptance and mean generated length per arm.