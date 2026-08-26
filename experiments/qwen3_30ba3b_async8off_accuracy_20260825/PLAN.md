# Qwen3-30B-A3B Async 8-Off Accuracy A/B

## Goal

Measure the convergence impact of matching the training global batch size to
the rollout batch size and enabling `force_on_policy_ratio` in the existing
Qwen3-30B-A3B async 8-off performance recipe.

## Controlled Pair

Both cases use the same NeMo-RL commit, container, model, dataset, seed,
24-node topology, async trajectory-age limit, validation cadence, optimizer,
learning-rate schedule, generation settings, and 200-step limit.

| Case | Train GBS | Rollout batch | `force_on_policy_ratio` | Optimizer batches per rollout set |
| --- | ---: | ---: | --- | ---: |
| Baseline | 512 | 2048 | false | 4 |
| Variant | 2048 | 2048 | true | 1 |

The reference-policy KL penalty remains `0.01` in both cases. This avoids
mixing reference-logprob skipping or an objective change into this A/B.

## Evaluation

Compare the following W&B trajectories over equal step windows:

- validation accuracy and validation reward;
- training reward and loss;
- generation KL error, policy KL error, and JS-divergence error;
- token multiplicative probability error and sampling importance ratio;
- approximate entropy and gradient norm when present; and
- end-to-end, generation, logprob, and policy-training time.

The variant is acceptable only if its validation trajectory is statistically
consistent with the baseline and it does not introduce unstable KL/error
metrics. Exact step-wise equality is not expected because async scheduling is
not deterministic.

## Execution

1. Validate the launcher with `MODE=test` (`sbatch --test-only`).
2. Submit both 200-step cases with `MODE=submit`.
3. Monitor the two jobs for at least five minutes and inspect their driver logs.
4. Export the common W&B metrics and write `REPORT.md` with the measured result.

