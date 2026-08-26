# Qwen3-30B-A3B Async 8-Off Accuracy A/B

## Goal

Measure the short-horizon accuracy impact of matching the training global
batch size to the rollout batch size and enabling `force_on_policy_ratio` in
the existing Qwen3-30B-A3B async 8-off performance recipe.

## Controlled Pair

All cases use the same NeMo-RL commit, container, model, dataset, seed,
24-node topology, async trajectory-age limit, validation cadence, optimizer,
learning-rate schedule, generation settings, and 200-step limit.

| Case | Train GBS | Rollout batch | `force_on_policy_ratio` | Optimizer batches per rollout set |
| --- | ---: | ---: | --- | ---: |
| Baseline | 512 | 2048 | false | 4 |
| GBS-only | 2048 | 2048 | false | 1 |
| GBS + logprob skip | 2048 | 2048 | true | 1 |

The reference-policy KL penalty remains `0.01` in all three cases. This avoids
mixing reference-logprob skipping or an objective change into this A/B.

## Evaluation

Compare the following W&B trajectories over equal step windows:

- validation accuracy (the mean validation reward);
- training reward and loss;
- generation KL error, policy KL error, and JS-divergence error;
- token multiplicative probability error and sampling importance ratio;
- approximate entropy and gradient norm when present; and
- end-to-end, generation, logprob, and policy-training time.

This single-seed, 200-step ablation is intended to detect a gross regression
and characterize short-horizon directionality. It does not establish
convergence equivalence. Exact step-wise equality is not expected because
async scheduling is not deterministic. If the result is close or noisy, the
next gate is repeated seeds or a longer run with a predefined non-inferiority
margin.

## Execution

1. Validate scheduler feasibility with `MODE=test` (`sbatch --test-only`).
2. Submit all three 200-step cases with `MODE=submit`.
3. Monitor the jobs for at least five minutes and inspect their driver logs.
4. Export the common W&B metrics and write `REPORT.md` with the measured result.
