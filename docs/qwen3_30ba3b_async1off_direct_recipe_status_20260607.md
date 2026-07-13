# Qwen3-30B-A3B Async-1off Direct Recipe Status

Status timestamp: 2026-06-07 14:15 PDT

## Purpose

Run the official NeMo-RL async-1off performance recipe directly and compare
baseline against PARD K=3 with minimal overrides. This is intended to answer
whether speculative decoding helps when the GRPO recipe uses asynchronous
rollout/training overlap.

## Recipe

Remote worktree:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL-main-vllm020-20260606`

Config:

`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml`

Direct recipe shape confirmed in the driver config:

| Field | Value |
|---|---:|
| Model | `Qwen/Qwen3-30B-A3B` |
| NeMo-RL / vLLM runtime | latest-main nightly, vLLM `0.20.0` |
| Cluster | 4 nodes x 4 GPUs |
| Training resources | 2 nodes x 4 GPUs |
| Generation resources | 2 nodes x 4 GPUs, non-colocated |
| `grpo.async_grpo.enabled` | `true` |
| `policy.generation.vllm_cfg.async_engine` | `true` |
| `grpo.num_prompts_per_step` | 64 |
| `grpo.num_generations_per_prompt` | 32 |
| Rollout trajectories per step | 2048 |
| `policy.train_global_batch_size` | 512 |
| Generation TP | 1 |
| Training TP / PP / CP / EP | 1 / 1 / 1 / 8 |
| `policy.max_total_sequence_length` | 4096 |
| `policy.generation.max_new_tokens` | 4096 |
| Sampling | temperature 1.0, top_p 1.0, top_k null |

## Jobs

| Job | Variant | Final status | Notes |
|---:|---|---|---|
| 3207260 | Baseline, direct YAML | CANCELLED after 29m50s | No speculative overrides. Cancelled after first weight update path stopped making progress. |
| 3207261 | PARD K=3, direct YAML | CANCELLED after 29m50s | Only added vLLM `speculative_config`; generation logprobs omitted. Cancelled after the same first weight update path stopped making progress. |

PARD drafter snapshot:

`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--amd--PARD-Qwen3-0.6B/snapshots/34e73caf94c021c3d4b8bad86ebd3616850ab4fd`

## Direct YAML Confirmation

Both jobs used the official YAML directly:

`/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml ...`

The PARD run only added CLI overrides for:

| Override group | Value |
|---|---|
| `policy.generation.vllm_kwargs.speculative_config.method` | `draft_model` |
| `policy.generation.vllm_kwargs.speculative_config.model` | public PARD drafter snapshot |
| `policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens` | 3 |
| `policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size` | 1 |
| `policy.generation.vllm_kwargs.speculative_config.parallel_drafting` | `true` |

So this is not evidence that a custom generated config caused the failure.

## Important Observation

The PARD job confirms NeMo-RL async GRPO and async engine are enabled, but vLLM
prints:

`Async scheduling not supported with draft_model-based speculative decoding and will be disabled.`

So the comparison is still a valid direct-recipe GRPO comparison, but it is not
evidence that vLLM internal async scheduling is preserved under draft-model
speculative decoding. This may reduce the expected async-1off benefit even if
PARD accelerates generation.

## Current Failure Mode

Both direct-YAML jobs reached vLLM worker initialization and the first refit /
weight-update path. The driver logs then showed:

`Error: Worker failed to update weights. Result: False`

The logs did not emit full `Training Results`, timing metrics, or token metrics
before cancellation. This points to the latest-main/vLLM `0.20.0` async-engine
non-colocated collective refit path, not to the YAML recipe shape itself.

## Prior Sync GBS2048 Attempt

The earlier sync-launcher GBS2048 jobs are not final performance evidence:

| Job | Variant | Outcome |
|---:|---|---|
| 3207093 | sync baseline, GBS2048 | FAILED at Step 3 vLLM `wake_up()` CUDA OOM |
| 3207094 | sync PARD K=3, GBS2048 | FAILED at Step 3 vLLM `wake_up()` CUDA OOM |

Both baseline and PARD failed in the same vLLM wake-up path, so this points to a
memory-shape issue in that sync launcher path rather than a PARD-only failure.

## Next Evidence Needed

1. Keep using official recipe YAMLs as the base and layer only specdec overrides.
2. Avoid treating the async-1off direct-YAML result as a performance comparison
   until the refit/weight-update failure is fixed or isolated.
3. Use the sync full-GRPO path for PARD functionality checks while debugging the
   async-engine collective update issue.
