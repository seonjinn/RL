# NeMo-RL vLLM 0.25.1 Eagle-3 Full CUDA Graph Plan

**Goal:** Validate NeMo-RL on vLLM 0.25.1 and measure Eagle-3 with the same performance recipe and CUDA Graph coverage as the baseline.

**Baseline:** NVIDIA NeMo-RL `origin/main` at `2ba2f0a73`. The existing v0.24 and experimental NUMA/offload worktrees remain unchanged.

## Implementation

1. Port the tested vLLM 0.25 rendering, Ray executor, and worker API adapters onto current main.
2. Pin the official vLLM 0.25.1 CUDA 13 wheels and matching FlashInfer/OpenAI/grammar dependencies.
3. Add dependency and source-contract tests before changing each contract.
4. Add an isolated Qwen3-30B-A3B performance-recipe launcher with matched baseline and Eagle-3 variants.
5. Force Model Runner V2 and `FULL_AND_PIECEWISE` CUDA Graph mode for both baseline and Eagle-3.
6. Derive speculative verification capture sizes from `(K + 1) * batch_size` and record the resolved values.
7. Keep DynamicSD opt-in. Apply any source patch only when the installed vLLM source still matches the affected 0.25.1 code.
8. Disable checkpointing, preserve the performance recipe's remaining defaults, and log the resolved runtime configuration to W&B and run metadata.

## Verification

1. Run focused dependency, source-contract, generation-worker, and launcher tests.
2. Run formatting and static checks on changed files.
3. Push the isolated branch only to `github-seonjinn:seonjinn/RL.git`.
4. On a GB200 cluster, pull the branch, initialize recursive submodules, and run an import/version smoke test.
5. Submit matched baseline and Eagle-3 K3/K5 smoke runs, monitor startup for five minutes, then promote passing variants to 20 steps.
6. Compare steps 2-20 for generation/E2E time and throughput, acceptance rate, mean accepted length, and CUDA Graph coverage/fallback.

