# Nemotron3 Super 200-step HybridEP stability study

## Objective

Run the canonical synchronous Nemotron3 Super performance recipe to Step 200
with HybridEP and full checkpoint/resume continuity. This is a long-horizon
accuracy and runtime-stability study, not a new dispatcher performance A/B.

## Fixed inputs

- Hardware: 32 GCP-NRT B200 nodes, 8 GPUs per node
- Recipe: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n8g.yaml`
- NeMo RL: `541413bd2912561950413b39809db40590a652bb`
- Megatron-Bridge: `fcbabe7845bce2a3281318111d0c86159fc19890`
- Megatron-Core: `34b55f24f0826c9aebd6693ecb60648cd934737d`
- DeepEP HybridEP: `17cfb817bccec3a9c247013360cc550c2bac441e`
- Maximum training steps: 200
- Dispatcher: HybridEP with one-time THD input alignment

The launcher changes only the maximum step count, checkpoint policy, run name,
and four-hour SLURM walltime. Model, batch sizes, sequence lengths, generation
settings, and Megatron parallelism remain those of the canonical recipe.

## Checkpoint and resume policy

- Submit 20 sequential `afterok` rounds.
- Stop each round before the four-hour allocation limit and save by 3:15.
- Save full model and optimizer state.
- Keep only the latest fine-tuning checkpoint.
- Resume each round from the dispatcher-specific checkpoint root.
- Extend the chain only if 20 rounds do not reach Step 200.

The 20-round reservation ceiling is 20,480 B200 GPU-hours. Jobs normally exit
after the internal checkpoint deadline, so expected usage is lower. The
existing 20-step runtime predicts approximately 13,500 B200 GPU-hours of model
work, plus repeated setup and checkpoint overhead.

## Submission protocol

```bash
bash experiment_logs/pr2964-super-200step-20260809/submit_super_chain.sh plan 20
bash experiment_logs/pr2964-super-200step-20260809/submit_super_4hour.sh test-only 1
bash experiment_logs/pr2964-super-200step-20260809/submit_super_chain.sh submit 20
```

The chain launcher repeats the scheduler preflight before submitting Round 1
and records every job ID in `job-chain.env` under the remote experiment root.

## Accuracy audit

Merge TensorBoard histories across rounds by global training step and retain the
last observation for duplicate step records. Audit Steps 2-200 for missing,
duplicate, null, NaN, and Inf values. Report validation accuracy every ten
steps, reward, loss, `train/gen_kl_error`, JS divergence, KL penalty, entropy,
gradient norm, importance ratios, valid samples/tokens, response length, and
truncation. Inspect every resume boundary separately and retain the existing
rare `token_mult_prob_error` warning rather than hiding exponential outliers in
an arithmetic mean.
