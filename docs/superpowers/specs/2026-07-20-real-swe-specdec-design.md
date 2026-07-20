# Real SWE SpecDec Design

## Goal

Measure full Qwen3-30B-A3B SWE GRPO performance with Eagle-3 K3 and DFlash
K7. The run must execute rollout, reward, logprob, policy training, and vLLM
refit rather than the rollout-only benchmark.

## Matched Comparisons

- Eagle-3 K3 uses vLLM Model Runner V2 and is compared with `baseline`.
- DFlash K7 uses vLLM Model Runner V1 and is compared with `baseline_v1`.
- All variants inherit the SWE2 recipe's 131072-token limits, use temperature
  1.0, top-p 1.0, CUDA Graphs, the same three-example dataset, and the same
  nine-node topology.
- Each SpecDec variant must pass a two-step smoke before promotion to 20 steps.
- Report step 2-20 E2E and generation time/throughput, policy and logprob time,
  generation ratio, acceptance rate, and mean accepted length.

## Checkpoints And Runtime

- Eagle-3 uses the pinned RedHatAI Qwen3-30B-A3B Thinking-2507 snapshot and
  draft TP1. It preserves the V2 baseline's CUDA Graph sizing.
- DFlash uses the pinned RedHatAI Qwen3-30B-A3B DFlash snapshot, K7, draft TP1,
  and draft `max_model_len=40960`.
- DFlash uses Model Runner V1 because vLLM 0.25.1 Model Runner V2 does not
  enforce the draft model's context limit. V1 safely disables drafting after
  the DFlash K7 boundary while target-only decoding can continue to 131072.
- DFlash CUDA Graph capture sizes are `[8,16,32,64,128]`.
- The matched V1 baseline uses FULL graphs with request capture sizes
  `[1,2,4,8,16]`, corresponding to the DFlash K7 query-token shapes.

## Safety Gates

- The launcher validates the checkpoint required by each variant.
- Dry-run tests assert runner selection, exact checkpoint, context limit,
  draft TP, method, K, and CUDA Graph settings.
- Submission uses a clean fresh worktree with recursive submodules at pinned
  commits, `sbatch --test-only`, five-hour Lyris allocation, and five-minute
  startup monitoring.
