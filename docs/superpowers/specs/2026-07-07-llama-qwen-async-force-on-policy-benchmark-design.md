# Llama and Qwen Async Force-On-Policy Benchmark Design

## Goal

Measure the isolated performance and correctness effect of
`loss_fn.force_on_policy_ratio=true` on the native four-GPU-per-node GB200
performance recipes that have not already enabled the flag.

## Fixed software and cluster

- NeMo-RL source commit:
  `d4cfecf90db41cdf142629963b54b67ab479ab02`
- Source branch: `sna/nemorl-main-pr3030-q235-20260701`
- Immutable container:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260630_0215.sqsh`
- Container SHA-256:
  `bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510`
- Cluster: Pre-Tyche GB200-NVL36
- Account: `coreai_dlalgo_llm`
- Partition: `36x2-a01r`
- All nodes expose four GPUs; do not use `--gres`.
- Use exclusive allocations with `--comment=metrics`.

No NeMo-RL, vLLM, Megatron-Bridge, or Megatron-LM source changes are allowed.

## Experiment matrix

Each recipe has a control and a treatment, for eight 20-step jobs total.

| Family | Recipe | Mode | Nodes | GPUs/node | Segment | Pair |
|---|---|---|---:|---:|---:|---|
| Llama 3.1 8B | `grpo-llama3.1-8b-instruct-2n4g.yaml` | sync | 2 | 4 | 2 | control/force |
| Llama 3.1 8B | `grpo-llama3.1-8b-instruct-2n4g-async-1off.yaml` | async-1off | 2 | 4 | 2 | control/force |
| Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` | async-1off | 4 | 4 | 4 | control/force |
| Qwen3-32B | `grpo-qwen3-32b-8n4g-async-1off.yaml` | async-1off | 8 | 4 | 8 | control/force |

The following are intentionally excluded:

- Every recipe whose filename uses an `8g` topology.
- Qwen3-235B, because its relevant performance recipes already enable
  `force_on_policy_ratio`.
- Qwen3-30B-A3B async-8off, because the requested scope is async-1off.
- The Qwen sync benchmark already running in
  `pretyche_force_on_policy_ratio_q30_q32_20260707`.

## Paired configuration contract

Both sides of every pair apply the same overrides:

```text
grpo.max_num_steps=20
checkpointing.enabled=false
policy.train_global_batch_size=2048
cluster.segment_size=<native node count>
```

The only paired difference is:

```text
loss_fn.force_on_policy_ratio=false  # control
loss_fn.force_on_policy_ratio=true   # treatment
```

Every resolved recipe must satisfy:

- `num_prompts_per_step=64`
- `num_generations_per_prompt=32`
- rollout batch `64 * 32 = 2048`
- `policy.train_global_batch_size=2048`
- `loss_fn.reference_policy_kl_penalty=0.01`
- `grpo.seq_logprob_error_threshold=null`
- recipe mode and `in_flight_weight_updates` match the original YAML
- native four-GPU-per-node topology matches the manifest

The sync Llama pair must resolve with async disabled. The three async pairs
must resolve with async enabled, maximum trajectory age one, and in-flight
weight updates enabled.

## Submission and safety gates

The user requested direct 20-step runs, so no two-step model smoke precedes this
matrix. Direct submission is still gated by lightweight checks:

1. Resolve all eight configurations and assert paired equality after
   normalizing `force_on_policy_ratio` and run-specific logger fields.
2. Run the existing unit tests for force-on-policy skip behavior and clipped
   policy-gradient behavior in the pinned container.
3. Verify source HEAD/upstream, recursive submodules, clean source status,
   immutable container SHA-256, and runner SHA-256.
4. Run `sbatch --test-only` for all eight jobs before any real submission.
5. Refuse an existing non-empty job manifest to prevent duplicate submission.
6. Record job ID, recipe, mode, force value, topology, source SHA, container
   SHA, runner SHA, and W&B run name immediately after submission.
7. Monitor every submitted job for at least five minutes and inspect any early
   exit before proceeding.

Use paired walltimes that are identical within each pair:

- Llama 3.1 8B: two hours
- Qwen3-30B-A3B: three hours
- Qwen3-32B: four hours

## Correctness analysis

Performance is not considered valid unless both sides of a pair complete all
20 steps with exit code `0:0`.

For each matched step, collect:

- reward mean and reward-distribution buckets
- loss
- generation KL error
- reference KL penalty
- sampling importance ratio
- token probability-error metrics
- force skip marker count
- reference-policy logprob timing presence
- NaN, Inf, OOM, CUDA, NCCL, Ray, and Python fatal signatures

Force runs intentionally report PPO probability ratio as exactly one. Their
sequence-level probability-error metrics are zero placeholders when previous
policy logprobs are skipped, so those fields are not correctness evidence.
Token-level error, generation KL, reference KL, reward, and loss must be used
instead.

Async-1off trajectories may be one policy version old while the treatment
forces the PPO ratio to one. Therefore async reward, KL, and loss curves must be
reported separately from sync and must not be pooled with the sync Llama pair.

## Performance analysis

Exclude warm-up step 1 and validation-bearing steps 10 and 20. For every pair,
report the mean, median, sample count, and treatment-versus-control delta for:

- E2E step time
- E2E tokens/s/GPU and samples/s
- generation time and generation tokens/s
- policy-and-reference-logprob time and tokens/s/GPU
- policy-training time and tokens/s/GPU
- preparation and weight-transfer time

The final report must include Pre-Tyche job IDs, W&B URLs, exact software
identities, resolved overrides, terminal states, correctness findings, and
limitations. No speedup is reported for an incomplete pair.

## Failure handling

- A configuration or test-only failure blocks the complete matrix until fixed.
- An early runtime failure is classified from the first fatal signature; do not
  hide it with a timeout increase or backend override.
- A failed pair is not silently retried with a different recipe, backend,
  topology, or container.
- AWS-DFW is a fallback only if Pre-Tyche cannot schedule the native topology;
  cross-cluster results are labeled separately and are never pooled.
