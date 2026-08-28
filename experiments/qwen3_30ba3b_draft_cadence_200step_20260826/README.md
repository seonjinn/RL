# Qwen3-30B-A3B fixed-interval drafter study

This experiment compares DFlash and DSpark online drafter updates every 5, 10,
or 20 Math GRPO policy steps. All six arms inherit the official
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` recipe.

## Matrix

| Drafter | Update interval | Variant |
|---|---:|---|
| DFlash K5 | 5 steps | `dflash-fixed5` |
| DFlash K5 | 10 steps | `dflash-fixed10` |
| DFlash K5 | 20 steps | `dflash-fixed20` |
| DSpark K5 | 5 steps | `dspark-fixed5` |
| DSpark K5 | 10 steps | `dspark-fixed10` |
| DSpark K5 | 20 steps | `dspark-fixed20` |

Here, fixed interval N means that the drafter is trained and refit every N
policy steps. It is different from the separate `fixed` arms in the
fixed-versus-always study, where the drafter remains frozen and receives no
online update.

## Preserved performance recipe

The inherited workload remains 4 nodes × 4 GPUs, 64 prompts × 32 generations,
global batch 2048, a 4096-token policy/generation limit, TP1/EP16/PP1/CP1,
sequence packing with fused loss, validation every 10 steps, shuffled
OpenMathInstruct-2 with a 5% validation split, disabled checkpoint writes, and
the Triton MoE backend. The overlays add only:

- 200 total steps;
- local target and tokenizer paths;
- W&B cadence metrics without durable checkpoint evidence;
- one K5 DFlash or DSpark drafter, optimizer, and fixed update schedule.

The launcher does not override `data_plane.enabled`, vLLM `max_num_seqs`, the
compilation backend, CUDA Graph mode, or capture sizes. Runtime defaults decide
the CUDA Graph coverage, matching the official performance flow.

## Immutable inputs

- Product source: `/home/sna/nemorl-q30-cadence-product-20260826`
- Product SHA: `1ce79c48334496fe4d86cf99fb3d27208b9f9b51`
- Target: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B`
- DFlash: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391`
- DSpark: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391`
- W&B project: `sna-specdec`
- W&B group: `q30ba3b-draft-cadence-200step-20260826`

## Validation and submission

Each variant must pass the exact state-dict check, composed-config verifier,
W&B authentication, scheduler dry-run, CUDA Graph gate, Step 1 gate, and Step 2
gate. `dflash-fixed5` is the canary; the remaining five jobs are submitted only
after it reaches Step 2 without host or GPU OOM.

```bash
uv run --no-project --with pytest --with pydantic python -m pytest -q \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests
uv run --no-project --with ruff ruff check \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826
bash -n \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh
```
