# DFlash Online Policy-Training Profile Plan

## Goal

Explain the policy-training slowdown with a matched, correctness-preserving GPU profile before changing the training implementation.

## Scope

- Use one exact source head for both arms.
- Keep Qwen3-8B, DAPOMath17K, TP2/DP2/PP1/CP1, GBS32, MBS1, sequence packing disabled, sequence parallel disabled, seed 42, and the public DFlash K7 generation runtime identical.
- Compare fixed-drafter control against true-online draft training.
- Profile only `megatron_policy_worker` for steady-state steps 3 through 5.
- Preserve CUDA Graphs for generation; generation workers are not profiled.
- Keep W&B disabled for the profiling arms and write profiles plus compact provenance to the experiment result directory.

## Profiler Choice

Use NeMo-RL's supported Nsight Systems worker integration. The current tree has no `NtraceCallback` integration or ntrace Parquet writer, so adding ntrace to the science path would introduce a new dependency and callback code before measuring the existing implementation. Nsight already supports policy-only capture with `NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker` and a bounded step range.

After capture, export the reports to SQLite on OCI-HSG and run the `llm-analyzer` silicon breakdown. If the Nsight hierarchy cannot attribute the online-only cost below the draft forward/backward boundary, add bounded NVTX ranges in a separate TDD change and repeat the same two-arm profile.

## Experiment Matrix

| Arm | Target training | Draft training | Draft refit | Update probe |
|---|---:|---:|---:|---:|
| fixed-control | on | off | off | off |
| online-current | on | on | every optimizer step | on |
| online-probe-off | on | on | every optimizer step | off |

The third arm isolates the diagnostic checksum cost without changing model updates or refit cadence.

## Gates

1. Static config-parity test permits only the four intended draft-training deltas.
2. One warmup plus three captured steps complete with finite target loss and gradients.
3. Online arms require nonzero draft loss/gradient/update and live refit evidence.
4. Fixed control requires no draft optimizer/update/refit markers and positive K7 acceptance.
5. All arms must use the same prompt IDs, realized token counts, checkpoint, seed, and CUDA Graph capture list.
6. Profiles are valid only if the parent SLURM job exits zero and the report files are copied from node-local Ray logs to the durable result directory.

## Analysis Order

1. Compare fixed-control with online-current to quantify total online-only GPU and CPU-launch overhead.
2. Compare online-current with online-probe-off to isolate update-probe synchronization.
3. Classify remaining online-only time into draft body forward/backward, projected CE, hidden capture/copies, metadata collectives, and optimizer.
4. Optimize one component at a time in that order, requiring loss/gradient/update/refit/acceptance parity after every change.

## Stop Conditions

- Do not reduce refit cadence; that changes online-training semantics.
- Do not skip target/reference log-probabilities under the current GRPO correction and KL settings.
- Do not compact projected CE rows until active-row loss and gradient equivalence are proven.
- Do not claim an optimization from W&B step timers alone; require matched profile evidence and at least three steady-state iterations.
