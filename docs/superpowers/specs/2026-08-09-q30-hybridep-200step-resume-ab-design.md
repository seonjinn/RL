# Qwen3-30B-A3B HybridEP 200-Step Resumed A/B Design

## Objective

Measure whether HybridEP sequence-packed training remains numerically stable for
200 synchronous GRPO steps across four-hour allocations and whether its
previously observed policy-training and LogProb speedups persist relative to the
all-to-all baseline.

## Workload and Controls

- Run the canonical `grpo-qwen3-30ba3b-4n8g.yaml` performance recipe on
  GCP-NRT B200 with four nodes and eight GPUs per node.
- Submit one all-to-all baseline arm and one HybridEP arm. The dispatcher is the
  only intentional workload difference.
- Use NeMo-RL validation commit
  `541413bd2912561950413b39809db40590a652bb`, whose `nemo_rl/` production tree
  matches PR #2964 head `eecfeeb08958e7211421231a84b603631f151f45`.
- Use Megatron-Bridge `fcbabe7845bce2a3281318111d0c86159fc19890`,
  routing-fixed MCore `34b55f24f0826c9aebd6693ecb60648cd934737d`, and
  DeepEP HybridEP `17cfb817bccec3a9c247013360cc550c2bac441e`.
- Preserve the validated container, model, data, batch sizes, and topology. Set
  `grpo.max_num_steps=200` and enable checkpointing only for this experiment.
- Use a separate checkpoint directory for each dispatcher. Save the complete
  model, optimizer, dataloader, and training state so each later allocation
  resumes automatically from the latest completed step.
- Exclude known unhealthy nodes `pool0-0167`, `pool0-0272`, and `pool0-0337`.

## Launcher Design

Keep the existing 20-step launcher backward compatible. Add optional
environment overrides for maximum steps, experiment root, walltime, and
checkpoint settings; all defaults retain the prior 20-step behavior. A
dedicated wrapper sets `MAX_NUM_STEPS_OVERRIDE=200` and
`TIME_LIMIT_OVERRIDE=04:00:00`, then records the resolved command and immutable
source metadata in each round-specific run directory.

Set `checkpointing.checkpoint_must_save_by=00:03:15:00`. NeMo-RL uses its
measured iteration time to checkpoint after the last safe step, flush the
asynchronous distributed save, and stop normally before the scheduler limit.
Set `checkpointing.save_period=200` so intermediate periodic checkpoints do not
distort performance. Retain only the latest checkpoint, set
`checkpointing.metric_name=null`, and save optimizer state. Submit three
sequential rounds per dispatcher with `afterok` dependencies; baseline and
HybridEP chains may run concurrently.

Before submission, the launcher must verify the NeMo-RL, MCore, DeepEP wheel,
container, and submodule provenance. Run `sbatch --test-only` for both arms,
then submit both jobs without a dependency so they may schedule concurrently.
Each job requests all eight GPUs on every allocated node.

## Measurements

Use W&B's exact history records from every round and compare the closed Steps
2–200 window after both arms finish. Deduplicate by training step and record
included and missing steps independently for every metric. Compare validation
accuracy only at checkpoints emitted by both arms. Exclude timeout/final
checkpoint-save steps from the pure dispatcher performance average, but retain
them in the correctness and E2E operational record.

- Validation accuracy at every emitted validation checkpoint.
- Reward, policy loss, `train/gen_kl_error`, valid sample count, and token work.
- E2E, generation, policy-training, and LogProb time.
- Logged E2E, generation, policy-training, and LogProb throughput in
  tokens/second/GPU.
- Runtime failures, hangs, illegal memory access, collective errors, NaN, and
  Inf signals.

## Interpretation

The experiment establishes no observed numerical regression only if both arms
complete 200/200, every resume starts from the prior checkpoint without a step
gap, core metrics are finite and complete, and HybridEP trends do not show a
sustained degradation relative to baseline. One logical run per arm does not
prove statistical equivalence or long-horizon convergence; observed differences
remain descriptive and must be reported with workload/token-volume differences.

## Failure Handling

Monitor each active round for at least five minutes after it starts. Preserve
logs and the last complete checkpoint before retrying. Infrastructure failures
before a new step may be retried from the latest checkpoint; application
failures after training resumes are reported and are not silently replaced.
Measure the first checkpoint's byte and file-count overhead before deciding
whether retention or cleanup settings need adjustment.
