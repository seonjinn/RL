# Lyris NeMo-RL Nightly 20260620 Smoke

Date: 2026-06-20 PDT

## Container

- Lyris container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260620.sqsh`
- Active symlink: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly.sqsh`
- Symlink target verified after download: `nemo_rl_nightly_20260620.sqsh`

## Smoke Run

| Field | Value |
|---|---|
| Job ID | `2166227` |
| Status | `COMPLETED`, exit code `0:0` |
| Worktree | `/project/coreai_dlalgo_llm/users/sna/RL-latest-main-canary-20260618` |
| Commit | `bab69c2` |
| Model | Qwen3-32B |
| Mode | sync GRPO |
| SpecDec | Eagle-3, `num_speculative_tokens=3`, draft TP=2 |
| Steps | 1 |
| Max new tokens | 1024 |
| Temperature/top_p/top_k | 1.0 / 1.0 / -1 |
| Nodes/GPUs | 4 nodes, 16 GPUs |
| vLLM runtime | v0.20.0 inside `VllmGenerationWorker` |
| Log root | `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs/20260620_lyris_nightly20260620_qwen32_sync_eagle3_step1_r1` |

## Result

The run completed generation, reward processing, logprob inference, and policy training for step 1.

| Metric | Value |
|---|---:|
| Loss | 0.0025 |
| Generation KL error | 0.0011 |
| Avg reward | 0.2935 |
| Mean generation length | 1014.4653 |
| Total step time | 127.60 s |
| Generation time | 36.02 s |
| Policy training time | 58.24 s |
| Policy/reference logprobs time | 28.66 s |
| E2E throughput | 1120.56 tok/s/GPU |
| Generation worker throughput | 3969.53 tok/s/GPU |
| Mean accepted length | 2.31-2.41 |
| Avg draft acceptance rate | 43.7-46.9% |

## Notes

- The new nightly container successfully starts the Ray cluster, initializes vLLM workers, loads the Eagle-3 drafter, runs SpecDec generation, computes logprobs, and completes one policy training step.
- vLLM reported `Initializing a V1 LLM engine (v0.20.0)`, so this NeMo-RL nightly path is still the vLLM 0.20.0 worker runtime rather than standalone vLLM 0.20.2.
- Expected non-fatal warnings appeared: Hermes tool parser patch snippet not found, `VLLM_ATTENTION_BACKEND`/`VLLM_USE_V1` unknown env var warnings, and Megatron sequence-parallel performance warnings.
