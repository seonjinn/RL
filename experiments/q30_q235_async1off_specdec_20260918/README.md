# Qwen3 Async-1off Speculative-Decoding Study

This experiment measures no-SpecDec, DFlash, and DSpark under the official
NeMo-RL Async-1off performance recipes for Qwen3-30B-A3B and
Qwen3-235B-A22B.

## Controlled comparison

| Model | Official recipe | Nodes | SpecDec arms |
|---|---|---:|---|
| Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` | 4 × 4 GPUs | DFlash K5, DSpark K5 |
| Qwen3-235B-A22B | `grpo-qwen3-235b-32n4g-async-1off.yaml` | 32 × 4 GPUs | DFlash K7, DSpark K7 |

Every primary arm uses:

- `max_num_seqs=64`
- `enforce_eager=false`
- `FULL_AND_PIECEWISE` CUDA Graphs
- `flashinfer_trtllm` MoE backend
- geometric request buckets through S64
- method-aware verification-width capture sizes for SpecDec

SpecDec arms additionally use `specdec_deep_refit`. The no-SpecDec baseline
does not enter the drafter lifecycle path. Jobs are independent; no scheduler
dependency serializes the matrix.

## Checkpoints

| Arm | Drafter lineage |
|---|---|
| Q30 DFlash/DSpark | PTV3-SWE SWA, step 44000 |
| Q235 DFlash | PTV2-en B8, step 25391 |
| Q235 DSpark | PTV3-RP25 B8, step 44000 |

The Q235 arms use the newest base-target checkpoint available for each method.
This is a best-available performance comparison, not a controlled
checkpoint-lineage ablation.

## Measurement

One-step gates must finish before 20-step jobs are submitted. Final averages
use Steps 3–20 inclusive. Async-1off overlaps rollout and policy work, so the
primary timing figure reports:

- `timing/train/exposed_generation`
- `timing/train/total_step_time`

Raw generation TPS/GPU, E2E TPS/GPU, reward, mean generated tokens, policy KL,
and generation KL remain in the result receipt.

## Launcher examples

```bash
python3 experiments/q30_q235_async1off_specdec_20260918/launch.py \
  --model q30 --arm baseline --steps 1 --test-only

python3 experiments/q30_q235_async1off_specdec_20260918/launch.py \
  --model q30 --arm dflash_k5 --steps 1 --submit

python3 experiments/q30_q235_async1off_specdec_20260918/launch.py \
  --model q235 --arm dspark_k7 --steps 20 --submit
```
