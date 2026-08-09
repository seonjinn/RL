# Qwen3-30B-A3B HybridEP 100-Step A/B Design

## Objective

Measure whether HybridEP sequence-packed training remains numerically stable for
100 synchronous GRPO steps and whether its previously observed policy-training
and LogProb speedups persist relative to the all-to-all baseline.

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
- Preserve the validated container, model, data, batch sizes, topology, and
  checkpoint-disabled behavior. Change only `grpo.max_num_steps` from 20 to
  100 and use distinct run names and output directories.
- Exclude known unhealthy nodes `pool0-0167`, `pool0-0272`, and `pool0-0337`.

## Launcher Design

Keep the existing 20-step launcher backward compatible. Add optional
environment overrides for maximum steps, experiment root, and walltime; all
defaults retain the prior 20-step behavior. A dedicated 100-step wrapper sets
the overrides and records the resolved command and immutable source metadata in
each run directory.

Before submission, the launcher must verify the NeMo-RL, MCore, DeepEP wheel,
container, and submodule provenance. Run `sbatch --test-only` for both arms,
then submit both jobs without a dependency so they may schedule concurrently.
Each job requests all eight GPUs on every allocated node.

## Measurements

Use W&B's exact history records and a closed Steps 2–100 window. Record included
and missing steps independently for every metric.

- Validation accuracy at every emitted validation checkpoint.
- Reward, policy loss, `train/gen_kl_error`, valid sample count, and token work.
- E2E, generation, policy-training, and LogProb time.
- Logged E2E, generation, policy-training, and LogProb throughput in
  tokens/second/GPU.
- Runtime failures, hangs, illegal memory access, collective errors, NaN, and
  Inf signals.

## Interpretation

The experiment establishes no observed numerical regression only if both jobs
complete 100/100, core metrics are finite and complete, and HybridEP trends do
not show a sustained degradation relative to baseline. One run per arm does
not prove statistical equivalence or long-horizon convergence; observed
differences remain descriptive and must be reported with workload/token-volume
differences.

## Failure Handling

Monitor each job for at least five minutes after it starts. Preserve logs before
retrying. Infrastructure failures before Step 1 may be retried on clean nodes;
application failures after training begins are reported as experiment results
and are not silently replaced.
