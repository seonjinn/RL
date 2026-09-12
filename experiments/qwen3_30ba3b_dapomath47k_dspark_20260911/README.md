# Qwen3-30B-A3B DAPOMath17K 47K DSpark study

This matched NeMo-RL vLLM 0.25.1 study measures frozen DSpark on the long-context
DAPOMath17K workload represented by the upstream Nemotron Nano DAPO recipe.

## Workload contract

- DAPOMath17K with `dapo_math_verify`
- 128 prompts x 16 generations = train GBS 2048
- 2,048-token input cap, 47,104-token response cap, 49,152-token total cap
- DAPO reward scaling, overlong shaping, and loss settings
- Qwen3-30B-A3B Base, BF16, Megatron TP2/EP8/CP4
- 8 nodes x 4 GB200 GPUs, sequence packing enabled
- vLLM 0.25.1, FlashInfer TRTLLM MoE, FULL_AND_PIECEWISE CUDA Graph
- No-SpecDec baseline versus frozen PTV3-SWA step-44000 DSpark K3 and K5

This is a Qwen/vLLM adaptation of the 49K DAPO recipe, not a claim that the
upstream Nano/TRTLLM recipe runs unchanged. Checkpointing and validation are
disabled so the timing comparison isolates the training-step workload.

## Launch protocol

```bash
bash submit_matrix.sh --test-only
bash submit_matrix.sh --submit-gates
# Submit only after all 1-step gates finish successfully:
bash submit_matrix.sh --submit-20
```

Jobs are independent and have no artificial SLURM dependencies.

## Submission receipt

Submitted on 2026-09-11 PDT from source commit `0bc95ecf9` using
`coreai_dlalgo_nemorl`, selected after FairShare and `sbatch --test-only`
comparison.

| Arm | Gate job | Steps | State after five minutes |
|---|---:|---:|---|
| Baseline | 7092399 | 1 | PENDING, reason `None` |
| DSpark K3 | 7092401 | 1 | PENDING, reason `None` |
| DSpark K5 | 7092403 | 1 | PENDING, reason `None` |

All three arms passed the launcher contract tests and SLURM test-only check.
The 20-step matrix remains intentionally unsubmitted until every gate completes
one rollout, reward, logprob, policy-training, and refit-free baseline/SpecDec
step without OOM or distributed failure.
