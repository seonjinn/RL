# PR #2964 Qwen3-30B-A3B 200-Step Resumed A/B

## Objective

Run the canonical Qwen3-30B-A3B synchronous GRPO performance recipe to 200
steps with the all-to-all and HybridEP dispatchers. Each arm resumes from its
own checkpoint across dependent four-hour GCP-NRT B200 allocations.

## Fixed Inputs

- Recipe: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml`
- Topology: four nodes, eight GPUs per node, segment size four
- NeMo-RL validation source: `541413bd2912561950413b39809db40590a652bb`
- PR #2964 production tree: `eecfeeb08958e7211421231a84b603631f151f45`
- Megatron-Bridge: `fcbabe7845bce2a3281318111d0c86159fc19890`
- MCore routing fix: `34b55f24f0826c9aebd6693ecb60648cd934737d`
- DeepEP HybridEP: `17cfb817bccec3a9c247013360cc550c2bac441e`
- Maximum configured steps: 200
- SLURM walltime: four hours per round
- Internal checkpoint deadline: three hours and 15 minutes
- Checkpointing: full model and optimizer state; latest checkpoint retained
- Resume rounds: up to three per dispatcher

The only intentional workload difference is the dispatcher and its required
HybridEP configuration.

## Submission

Run both scheduler probes before either real submission:

```bash
bash experiment_logs/pr2964-q30-4hour-20260809/submit_q30_4hour.sh baseline test-only 1
bash experiment_logs/pr2964-q30-4hour-20260809/submit_q30_4hour.sh hybridep test-only 1
```

Then submit Round 1 for both arms independently. Submit Rounds 2 and 3 with
`JOB_DEPENDENCY=afterok:<prior-job-id>`:

```bash
bash experiment_logs/pr2964-q30-4hour-20260809/submit_q30_4hour.sh baseline submit 1
bash experiment_logs/pr2964-q30-4hour-20260809/submit_q30_4hour.sh hybridep submit 1
```

## Checkpoint and Terminal-State Policy

NeMo-RL must checkpoint and exit cleanly before the SLURM walltime. A dependent
round automatically resumes from the latest checkpoint. `TIMEOUT`, an
application fatal, or a resume from the wrong step is a failed round and must
be diagnosed before retrying.

## Analysis

Merge each arm's round histories by training step and require Steps 2–200 once
after deterministic deduplication. Report missing, duplicate, null, NaN, and
Inf counts for every metric. Compare accuracy at common validation checkpoints,
and report reward, loss, `train/gen_kl_error`, valid samples, token work,
E2E/generation/policy/LogProb time, and the corresponding logged throughput.
Exclude checkpoint-save steps from pure dispatcher means and report checkpoint
size, file count, save duration, and resume continuity separately.
