# NeMo-RL Eagle-3 DynamicSD Performance Design

## Objective

Measure whether vLLM 0.24 DynamicSD improves full NeMo-RL GRPO performance
relative to both target-only decoding and fixed-K Eagle-3. The primary gate is
the synchronous Math performance recipe because it exposes rollout latency in
the end-to-end step time and avoids async overlap obscuring the generation
effect.

The first phase covers Qwen3-30B-A3B and Qwen3-32B. Qwen3-235B follows only
after its independent vLLM EngineCore rendezvous-port failure is fixed. Async
1-off follows after the synchronous comparison is valid.

AWS-DFW is the primary execution environment because the same vLLM 0.24 branch
has already completed ten-step synchronous and async 1-off NeMo-RL runs there
for both first-phase models. Cluster replicas are separate validation cohorts,
not interchangeable rows in the AWS-DFW comparison.

## Non-Goals

- Changing GRPO, reward computation, policy training, or weight refit logic.
- Comparing against historical full-CUDA-graph or eager-mode measurements.
- Treating a generation-only benchmark as NeMo-RL end-to-end evidence.
- Claiming a Qwen3-235B result before that model reaches training steps with
  the vLLM 0.24 runtime.

## Experiment Matrix

Use the upstream performance recipes without changing their model, topology,
dataset, rollout shape, training batch sizes, sampling, or sequence limits:

| Model | Recipe | Nodes | Segment | Variants |
|---|---|---:|---:|---|
| Qwen3-30B-A3B | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 | 4 | baseline, Eagle-3 K5, DynamicSD |
| Qwen3-32B | `grpo-qwen3-32b-4n4g.yaml` | 4 | 4 | baseline, Eagle-3 K5, DynamicSD |

Every run uses:

- `grpo.max_num_steps=20`
- `temperature=1.0` and `top_p=1.0` from the performance recipe
- `checkpointing.enabled=false`
- `policy.generation.vllm_cfg.enforce_eager=false`
- vLLM 0.24 Model Runner V1
- `compilation_config.cudagraph_mode=PIECEWISE`
- W&B logging enabled with model, variant, graph mode, and vLLM version in the
  run name

PIECEWISE is mandatory for DynamicSD in vLLM 0.24. Baseline and fixed-K runs
must use the same graph mode, even if another graph mode is faster, because the
comparison must isolate the scheduling policy rather than graph coverage.

## Speculative-Decoding Configuration

The fixed variant uses Eagle-3 with `num_speculative_tokens=5`. The DynamicSD
variant uses the same target model, drafter checkpoint, draft tensor
parallelism, and global maximum K, plus this scheduler-batch-size policy:

| Active scheduler batch size | K |
|---:|---:|
| 1-16 | 5 |
| 17-32 | 4 |
| 33-64 | 3 |
| 65-128 | 1 |
| 129-512 | 0 |

The vLLM `speculative_config` is passed through
`policy.generation.vllm_kwargs`. The experiment launcher owns the explicit
values; no DynamicSD defaults are added to NeMo-RL's shared configuration
schema.

## Components

### Experiment Launcher

Add a launcher under `experiments/vllm_024_upgrade/` that renders one SLURM
job per model and variant. It reuses `ray.sub`, derives `--segment` from the
recipe node count, stores logs under a run-specific Lustre directory, and
records each job ID, commit, recipe, drafter path, container, and override set
in a TSV manifest.

The launcher supports `dry-run`, `test-only`, and `submit`. Submission is
allowed only from a clean, pushed commit and after scheduler validation.

### Configuration Overlay

Use a small experiment-owned YAML overlay or structured launcher arguments to
represent the Eagle-3 and DynamicSD dictionaries. Prefer an overlay when it
avoids fragile shell quoting of the nested DynamicSD schedule. The overlay
must not duplicate defaults already supplied by the upstream performance
recipe.

### Runtime Metrics

Reuse NeMo-RL's vLLM metrics logger and existing W&B logging path. Add only the
minimum instrumentation needed to expose, per training step:

- accepted draft tokens and proposed draft tokens
- acceptance rate
- weighted mean accepted length
- active DynamicSD K distribution, if vLLM exposes it through metrics

If vLLM does not expose the selected-K distribution, log the configured
schedule and the observed scheduler batch-size distribution. Do not infer an
exact K histogram without runtime evidence.

### Result Collector

Collect SLURM, driver, TensorBoard, and W&B-derived metrics into a CSV with one
row per model and variant. Preserve the source job ID and W&B URL. A report row
is valid only when the run reaches Step 20 and the expected configuration is
present in the driver log.

## Data Flow

1. The launcher reads the selected upstream performance recipe.
2. It applies only run length, checkpoint, logging, graph-mode, and SpecDec
   overrides.
3. `ray.sub` creates the NeMo-RL cluster and starts full GRPO.
4. NeMo-RL passes the vLLM 0.24 `speculative_config` to each generation
   engine.
5. vLLM selects K from the current scheduler batch size for DynamicSD.
6. NeMo-RL records rollout, training, and acceptance metrics by step.
7. The collector validates provenance, drops Step 1 from steady-state means,
   and matches each SpecDec row to its same-model baseline.

## Metrics and Comparison Rules

Compute steady-state means over Steps 2-20:

- generation time
- E2E step time
- generation throughput in tok/s/GPU
- E2E throughput in tok/s/GPU
- acceptance rate
- mean accepted length

For each SpecDec variant, report:

- generation-time speedup: baseline generation time / variant generation time
- E2E step-time speedup: baseline E2E time / variant E2E time
- generation-throughput speedup: variant generation throughput / baseline
- E2E-throughput speedup: variant E2E throughput / baseline
- DynamicSD improvement over fixed K using the same four ratios

Baselines must match model, recipe, mode, max OSL, temperature, top-p, graph
mode, vLLM version, container, node/GPU shape, and commit. Historical rows that
do not match all fields remain separate.

## Validation Gates

1. Unit tests confirm that the nested DynamicSD schedule reaches vLLM without
   string coercion or key loss.
2. Launcher tests confirm recipe, node count, segment, graph mode, checkpoint
   setting, W&B name, and all three variant configurations.
3. A two-step smoke for each model confirms model load, CUDA Graph capture,
   rollout generation, policy training, acceptance counters, and clean exit.
4. The six 20-step jobs are submitted only after all smoke gates pass.
5. A performance row is final only after Step 20 and log/config provenance
   validation.

Accuracy-sensitive metrics such as reward, response length, KL, and policy
loss are retained in W&B. Stochastic sampling means exact token identity is not
required, and a 20-step performance run does not establish accuracy parity.
Any NaN or invalid reward blocks the performance conclusion. A greater than
10% relative change from the matched baseline in mean reward, mean response
length, or approximate KL over Steps 2-20 triggers investigation and prevents
an unqualified performance recommendation.

## Failure Handling

- Engine initialization or CUDA Graph failure: stop that variant and preserve
  the full driver/Ray logs; do not fall back to eager execution.
- Missing acceptance metrics: mark the SpecDec row incomplete rather than
  reporting speedup without verification that drafting was active.
- OOM: reduce no recipe-owned batch or sequence setting in place. Record a
  separate resource-adjusted cohort if a change is required.
- Timeout: retain completed steps as partial evidence, but do not label the row
  final. Retry with a longer walltime without changing runtime settings.
- Qwen3-235B rendezvous-port failure: resolve and validate it independently
  before adding that model to this matrix.

## Rollout Sequence

1. Implement and test the launcher, overlay, and metric extraction locally.
2. Push the branch and pull it into the AWS-DFW GB200 cluster worktree.
3. Run `--test-only` for Qwen3-30B-A3B and Qwen3-32B.
4. Submit two-step baseline, fixed-K, and DynamicSD smokes.
5. Monitor each smoke for at least five minutes and through policy training.
6. Submit the six 20-step jobs after all smoke gates pass.
7. Publish the matched result table and W&B links.
8. Extend the validated setup to async 1-off, then Qwen3-235B after its port
   blocker is cleared.

## Success Criteria

The integration is operational when both models complete 20 GRPO steps for all
three variants, DynamicSD produces positive draft/accept counters, and its K
policy is evidenced by runtime metrics. A performance benefit is claimed only
when DynamicSD improves either E2E throughput or E2E step time over fixed K
without a material regression in reward, KL, loss health, or response-length
distribution.
