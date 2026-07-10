# Mini Sync-GRPO Tail-Gate Validation Design

## Objective

Validate the vLLM 0.24 external-Eagle tail gate in a short, real NeMo-RL
synchronous GRPO run before launching the full performance matrix. The smoke
must exercise rollout generation, reward, logprob, and policy training while
showing exactly when speculative decoding changes from off to on.

## Workload

Use the upstream Qwen3-32B 4-node, 4-GPU performance recipe and preserve its
model, policy parallelism, vLLM target TP, Triton MoE backend, CUDA graphs, and
sampling contract. Only smoke-sized workload dimensions change:

- 16 prompts per step;
- 4 generations per prompt, for 64 initial rollouts;
- train global batch size 64;
- maximum total/output length 1024;
- vLLM maximum model length 1056, including 32 speculative-token headroom;
- 2 GRPO steps;
- checkpointing disabled; and
- temperature 1.0, top-p 1.0, standard rejection sampling, and probabilistic
  Eagle draft sampling.

The matrix contains matched V2 arms only:

| Arm | SpecDec behavior |
|---|---|
| `baseline_v2` | No drafter |
| `always_on_v2_k5` | Eagle-3 K5 from the first decode |
| `fastrl_threshold_v2_k5` | K0 advance-only until the FastRL tail gate activates, then K5 |

The threshold arm uses threshold 32 and 10 consecutive qualifying decode
checks. It must first observe more than 32 active requests, preventing an early
prefill-completion false activation.

## Activation Observability

The scheduler already records the exact activation batch and sequence length.
Add an activation-only scheduler tick counter and derive these W&B scalars for
each GRPO step:

- `train/vllm/tail_gate_activation_tick`;
- `train/vllm/tail_gate_activation_batch`;
- `train/vllm/tail_gate_activation_sequence_length`;
- `train/vllm/tail_gate_enabled_step_ratio`;
- `train/vllm/tail_gate_advance_only_step_ratio`; and
- `train/vllm/tail_gate_activation_predicted_speedup` for roofline runs.

Activation counters update only when `tail_gate_just_activated` is true. The
HTML report renders activation events with scheduler tick on the x-axis and
inflight batch on the y-axis, includes a horizontal threshold line at 32, and
labels the OFF-to-ON point. W&B retains activation tick and batch as separate
step-wise graphs so the event is inspectable directly from the run.

## Functional Gates

The two-step threshold smoke is valid only when:

- both GRPO steps complete through policy training;
- the target and Eagle checkpoints load;
- V2 and `FULL_AND_PIECEWISE` CUDA graphs are selected;
- the gate observes both K0 and K5 scheduler steps;
- exactly one activation is recorded per training rollout;
- activation batch is positive and no larger than 32;
- activation tick is positive;
- enabled and advance-only ratios are each strictly between 0 and 1;
- proposals and accepted tokens are positive after activation;
- reward, policy loss, logprob, and generated lengths are finite; and
- there is no stale draft ID, invalid token, NaN, OOM, NCCL, or q-cache error.

Baseline, always-on, and threshold arms must match every immutable cohort field
except the declared SpecDec variant. The smoke reports timing and throughput but
does not claim stable speedup from two steps.

## Cluster Execution

Run on Pre-Tyche first using account `coreai_dlalgo_llm`, partition `batch`,
four nodes, `--segment=4`, no GPU GRES, and the staged 20260705 NeMo-RL nightly
image. Submit only after the branch is pushed to the personal fork, the remote
checkout is updated, recursive submodules are initialized, and launcher
`test-only` succeeds. Monitor for at least five minutes and through the first
policy-training step.

## Non-Goals

- This smoke does not replace the matched 20-step performance run.
- It does not calibrate or test the roofline arm.
- It does not alter GRPO, rewards, sampling, target or draft checkpoints, or
  policy-training math.
- It does not reduce the Qwen3-32B model topology or use a synthetic rollout
  backend.
