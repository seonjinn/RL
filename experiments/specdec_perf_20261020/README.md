# SpecDec perf on the draft co-training stack (2026-10-20)

Measures how much DFlash and DSpark speculative decoding speed up rollout and the full GRPO step, on latest main plus all 11 draft co-training PRs (#3701 through #3757). Baseline is the shipped performance recipe, unchanged. SpecDec arms add only the speculative settings.

## Arms (Qwen3-30B-A3B, OCI-HSG 4 nodes x 4 GB200)

All arms start from `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`. Generation uses `moe_backend: flashinfer_trtllm`. K is 7 everywhere: the drafters are trained with block 8, and the earlier 20-step Q30 study found K7 marginally ahead of K5.

| Config | Training layout | Drafter | What it isolates |
|---|---|---|---|
| perf recipe as-is | TP1 EP16 | none | baseline |
| `frozen_dflash_k7` | TP1 EP16 | DFlash b8, step 25391, frozen | rollout speedup from SpecDec alone |
| `frozen_dspark_k7` | TP1 EP16 | DSpark b8, step 25391, frozen | same, DSpark |
| `tp2_baseline` | TP2 SP ETP1 EP16 | none | baseline for the online layout (online drafting requires target TP2) |
| `online_dflash_g7` | TP2 SP ETP1 EP16 | DFlash, trained every step, refit every step | co-training and refit cost on top of SpecDec |
| `online_dspark_b8` | TP2 SP ETP1 EP16 | DSpark, trained every step, refit every step | same, DSpark |

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

- Only one job may run with `FORCE_REBUILD_VENVS=true` per checkout. Each node's builder deletes the shared `venvs/` tree first, so concurrent rebuilds corrupt each other.
- `--segment` equals the node count, so all 16 ranks sit in one NVL72 domain. The HybridEP env vars in the recipe assume that.
- Compare arms on one shared closed step window (steps 3 to 20), using arithmetic means of the logged per-step values. Synchronous keys: `timing/train/generation`, `timing/train/prepare_for_generation/total` (refit), `timing/train/policy_training`, `timing/train/policy_and_reference_logprobs`, `timing/train/total_step_time`, `performance/generation_tokens_per_sec_per_gpu`, `performance/tokens_per_sec_per_gpu`.
- Before trusting a speedup, check reward, entropy, KL, acceptance and mean generated length per arm.