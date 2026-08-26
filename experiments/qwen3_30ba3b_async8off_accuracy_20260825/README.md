# Qwen3-30B-A3B Async 8-Off Accuracy A/B

This three-arm ablation separates the impact of matching train GBS to the
2048-sample rollout batch from the additional impact of enabling
`force_on_policy_ratio` and skipping previous-policy logprobs.

Run a scheduler validation first:

```bash
MODE=test EXPECTED_COMMIT=$(git rev-parse HEAD) \
  bash experiments/qwen3_30ba3b_async8off_accuracy_20260825/submit_accuracy_ab.sh
```

Submit the three-arm ablation:

```bash
MODE=submit EXPECTED_COMMIT=$(git rev-parse HEAD) \
  bash experiments/qwen3_30ba3b_async8off_accuracy_20260825/submit_accuracy_ab.sh
```

The script is intended to run from the experiment worktree on CW. It records
the source commit, image path, recipe hash, fixed overrides, and submitted job
IDs under the Lustre experiment directory.
