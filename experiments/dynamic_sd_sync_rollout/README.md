# DynamicSD under Synchronous RL Rollout

## Goal

vLLM 0.24 ships Dynamic Speculative Decoding (PR #32374): the draft length K is
selected at runtime from a user-supplied batch-size -> K lookup table
(`speculative_config.num_speculative_tokens_per_batch_size`). A synchronous GRPO
rollout is the ideal stress case: each step launches
`num_prompts_per_step x num_generations_per_prompt` sequences at once, then the
in-flight batch drains to a long tail, so the engine traverses the whole
batch-size axis every step. Fixed-K SpecDec must pick one point on that axis;
DynamicSD can ride the curve.

Question: how much rollout-batch wall time does DynamicSD recover vs baseline
and vs fixed-K EAGLE3 for Qwen3-30B-A3B / Qwen3-32B / Qwen3-235B-A22B
(Thinking drafters, temperature 1.0) on Math and SWE prompts?

## Method

1. `profile` mode: per model x benchmark, sweep BS {1..128} x K {0,1,2,3,5}
   with fixed OSL (ignore_eos) and measure output tok/s, ITL, acceptance length.
2. `derive_dynamic_k_table.py`: argmax tok/s per BS -> merged
   `[[bs_lo, bs_hi, K], ...]` ranges (K=0 disables speculation in that range).
3. `rollout` mode: N prompts x G generations per step with barrier semantics,
   natural stopping (`max_tokens` cap), temperature 1.0. Variants: baseline /
   fixed-K eagle3 / dynamic. Metrics: per-step wall, tok/s, output length tail
   (p50/p90/max), acceptance length, per-request finish times (drain curve).

## Setup

- Cluster: Lyris GB200 (driver 580.173), venv
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/venvs/vllm024` (vLLM 0.24.0).
- Rollout shapes mirror one vLLM DP-worker shard of the NeMo-RL GB200 SyncRL
  performance recipes (`examples/configs/recipes/llm/performance/*4g.yaml`),
  sampling from `grpo_math_1B.yaml` (temperature 1.0, top_p 1.0, seed 42):
  Targets are the exact NeMo-RL recipe models (base, hybrid-thinking); the
  Thinking speculators list these base models as their verifier on HF.
  | Target (recipe) | TP | N x G per engine | max_tokens | Drafter |
  |--------|----|-----------------|-----------|---------|
  | Qwen/Qwen3-30B-A3B (`30ba3b-4n4g`: 64x32 / 16 engines) | 1 | 4 x 32 = 128 | 4096 | RedHatAI/Qwen3-30B-A3B-Thinking-2507-speculator.eagle3 |
  | Qwen/Qwen3-32B (`32b-4n4g`: 64x32 / 8 engines) | 2 | 8 x 32 = 256 | 4096 | RedHatAI/Qwen3-32B-Thinking-speculator.eagle3 |
  | Qwen/Qwen3-235B-A22B (`235b-16n4g`: 16x32 / 8 engines) | 4 (recipe TP8, single-node cap) | 2 x 32 = 64 | 8192 | RedHatAI/Qwen3-235B-A22B-Thinking-2507-speculator.eagle3 |
- Prompts: Math = `data/math_500.jsonl`, SWE = `data/swebench_verified_prompts_all.jsonl`
  in the remote `vllm-benchmark` dir.

## Usage

```bash
# Phase 1: profiling grid (one job per K in {0,1,2,3,5})
MODE=profile bash experiments/dynamic_sd_sync_rollout/submit_matrix_lyris.sh qwen3_30ba3b math

# Phase 2: derive the table from harvested profile JSONs
# (--extend-to optional; must be >= the preset's max profiled batch size)
python experiments/dynamic_sd_sync_rollout/derive_dynamic_k_table.py \
  results/.../profile_k*.json --output dynamic_spec.json

# Phase 3: rollout comparison (baseline / fixed-K / dynamic)
MODE=rollout DYNAMIC_SPEC_JSON=dynamic_spec.json \
  bash experiments/dynamic_sd_sync_rollout/submit_matrix_lyris.sh qwen3_30ba3b math
```

## Key questions

- Does the fixed-K optimum differ between the batch-launch phase (BS=128) and
  the drain tail (BS<8)? If yes, DynamicSD has headroom by construction.
- Is the MoE (30B-A3B, 235B) optimal-K profile non-monotonic in BS, as the
  Cohere blog reports?
- Does temperature-1.0 acceptance (known to collapse vs greedy) shrink the
  usable K range enough that DynamicSD mostly selects K<=1 at high BS?
