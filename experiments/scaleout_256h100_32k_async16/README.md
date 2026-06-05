# 256 H100 scale-out @ 32K seq, async-16, GBS=512

First scale beyond 128 H100 for Qwen3-235B SWE async GRPO. Two variants in run order:

1. **BF16 KV** (job 12550182) — `submit_hybridep_pr2514_256h100_32k_async16_bf16kv.sh`, gmu=0.85
2. **FP8 KV**  (job 12550221) — `submit_hybridep_pr2514_256h100_32k_async16_fp8kv.sh`, gmu=0.80, kv_cache_dtype=fp8_e4m3

## Topology

- 32 nodes × 8 H100 = 256 GPU
- 24 nodes vLLM (192 generators, 24 DP groups TP=8) + 8 nodes training (64 train GPU)
- Train parallelism TP=4 × EP=8 × CP=1 × PP=8 × DP=2 = 64 ✓
- `recompute_granularity=full` explicit override (PR #2280 default broken selective)

## Key questions

1. Does GBS=512 + 32K seq fit 8-node training (TP=4 EP=8 PP=8 DP=2)?
2. Does 24-DP-group vLLM pool with timeout60 produce a meaningful net E2E improvement vs 12-DP-group baseline at GBS=256?
3. Does FP8 KV's long-tail-BW advantage resurface at max_model_len=32768 where timeout60 still permits longer tails than 16384?

## Branch / commit

`sj/super-v3-perf-patch+pr2280+pr2514` @ `00cf6b43d` (prefill/decode breakdown patch in vLLM async logger).

## Baseline for comparison

- 11912255: HybridEP+FP8-Tr v2 @ 16n×8, 20/20 steps, 414.88s/step
- 11919621: slot C (4-way apples-to-apples), 411.40s/step
- 11819947: 16n full rollout BF16 KV baseline
- 11835558: 16n FP8 KV rollout (-12.3% exposed gen vs 11819947, no timeout60)
