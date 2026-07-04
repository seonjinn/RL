# NeMo-RL vLLM 0.20 Best-Config Math Continuation Plan

**Goal:** Continue NeMo-RL Math performance-recipe measurements with CUDA graphs enabled and the strongest compatible vLLM 0.20 standalone SpecDec configurations.

**Global constraints:**

- Lyris GB200 only; no `--gres`.
- Preserve each checked-in `examples/configs/recipes/llm/performance` recipe and add only explicit runtime, logging, and SpecDec overrides.
- Use `temperature=1.0`, `top_p=1.0`, recipe OSL, and `max_steps=20` for final measurements.
- Keep CUDA graphs enabled with `policy.generation.vllm_cfg.enforce_eager=false`.
- Use `TRITON_ATTN` and Triton MoE for the new matched cohorts.
- Use Q30/Q32 segment equal to requested nodes. Use Q235 `--segment=16`, because Lyris limits a segment to at most 18 nodes.
- Record the NeMo-RL SHA, container path and checksum, rendered command, topology, W&B identity, test-only job, and submitted job in a CSV manifest.
- Run `sbatch --test-only`, commit and push before submission, and monitor submitted jobs for at least five minutes.

## Task 1: Declarative Launcher and Tests

Add one launcher under `experiments/eagle3_online/` and one focused test module under `tests/`.

The launcher must render complete commands locally and must not recover or `eval` commands from historical SLURM logs. It must validate only selected assets and pin the NeMo-RL source SHA.

Supported contracts:

| Model | Mode | Recipe | Nodes | Segment | Final methods |
|---|---|---|---:|---:|---|
| Qwen3-30B-A3B | sync | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 | 4 | suffix K32 |
| Qwen3-30B-A3B | async-1off | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` | 4 | 4 | suffix K32 |
| Qwen3-32B | sync | `grpo-qwen3-32b-4n4g.yaml` | 4 | 4 | suffix K32, Eagle-3 K3 |
| Qwen3-32B | async-1off | `grpo-qwen3-32b-8n4g-async-1off.yaml` | 8 | 8 | suffix K32, Eagle-3 K3 |
| Qwen3-235B-A22B | sync | `grpo-qwen3-235b-32n4g.yaml` | 32 | 16 | baseline, suffix K32, Eagle-3 K3 |
| Qwen3-235B-A22B | async-1off | `grpo-qwen3-235b-32n4g-async-1off.yaml` | 32 | 16 | baseline, suffix K32, Eagle-3 K3 |

Q30 PARD K16 is already present in the exact July cohort. Do not use the available Q30 Eagle-3 Thinking-2507 checkpoint against the base target. Q32 PARD K5 and Q235 PARD K16 remain a separate compatibility follow-up because their target/draft TP and no-AR-RMS requirements differ from the Suffix/Eagle cohort.

## Task 2: Verify and Submit

1. Run dry-run tests and shell syntax checks.
2. Push the branch and fast-forward the Lyris reporting checkout.
3. Run launcher preflight and `sbatch --test-only` for every selected contract.
4. Submit Q30/Q32 step-20 variants.
5. Submit Q235 baseline/Suffix/Eagle smokes with `max_steps=2`; promote only cohorts that initialize and complete the baseline smoke.
6. Monitor for five minutes and record early failures without blind retries.

## Task 3: Results and Reporting

Collect completed step metrics into the strict NeMo-RL schema. Match baselines by model, mode, OSL, sampling, CUDA graph state, cluster, topology, attention backend, MoE backend, and source cohort. Report generation/E2E time and throughput speedups plus acceptance rate and accepted length.
