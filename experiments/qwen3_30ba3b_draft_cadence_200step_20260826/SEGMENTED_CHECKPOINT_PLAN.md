# OCI-HSG four-hour segmented checkpoint plan

This document defines the recovery contract for the 200-step Qwen3-30B-A3B
DFlash and DSpark cadence matrix. The checked-in launcher implements this
contract, but the matrix must not be described as resume-validated until the
save-and-resume canaries pass on OCI-HSG.

## Scheduler and continuation contract

Each arm owns one logical run ID and one durable result root. Attempts are
separate scheduler jobs, but they are continuations of the same experiment.

- Use OCI-HSG `batch` with `--time=04:00:00`, account `nemotron_n4_post`, four
  nodes, four GPUs per node, and the same topology, container, source SHA,
  workload, and drafter checkpoint as the matched arm.

- Permit only the matched performance cohort: baseline and the official
  `*-cg2048` variants. Reject legacy and retry variants before creating durable
  artifacts. Preserve the official legacy-GRPO path with
  `data_plane.enabled=false` and `cadence_runtime.enabled=false`. The online
  schedule remains active through `policy.draft.update_schedule`.

- Set `checkpoint_must_save_by: "00:02:45:00"`. This timeout starts inside the
  training loop, so the remaining scheduler budget is reserved for Ray/model
  startup, checkpoint finalization, W&B flush, and teardown.

- Chain five total segments with arm-local `afterok` dependencies. Each
  continuation first checks for the validated Step 200 completion receipt. It
  exits without starting Ray when the run is already complete; otherwise it
  resumes the highest complete `step_N` checkpoint.

- Do not use `afterany` for unattended recovery. A model, state, or receipt
  failure must stop the chain instead of repeatedly consuming nodes.

- Do not rely on `--requeue`, an unhandled SLURM pre-timeout signal, or
  compute-node self-resubmission. The pinned driver has no signal-to-checkpoint
  contract, and self-submission complicates the launcher's fail-closed receipt
  identity. NeMo-RL's own timeout-based checkpoint and clean exit is the
  recovery boundary.

- The pinned source deliberately treats a timeout checkpoint as a resumable
  boundary rather than terminal completion. Only a validated Step 200 standard
  checkpoint can produce the experiment completion receipt.

## Checkpoint contract

Use the following logical settings for each arm:

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: <result-root>/checkpoints
  metric_name: null
  higher_is_better: true
  save_period: 200
  keep_top_k: 1
  ft_save_period: 20
  ft_keep_latest_k: 2
  save_optimizer: true
  checkpoint_must_save_by: "00:02:45:00"
```

`save_period: 200` marks the official terminal checkpoint. The 20-step fault
tolerance cadence aligns with fixed intervals 5, 10, and 20. Retention keeps
the Step 200 checkpoint plus the two most recent recovery checkpoints; metric
ranking is deliberately disabled. Optimizer state is mandatory because online
drafter training cannot resume exactly from model weights alone.

For this performance cohort, `checkpointing.checkpoint_dir` is the logical
result root's `checkpoints/` child. Model, drafter, optimizer, dataloader, and
draft-scheduler state close as one standard checkpoint identity. The resume
startup refit uses the default target-plus-draft selection, restoring the
checkpointed trainable draft into vLLM before the next generation step. A
temporary or partial checkpoint is never a resume candidate. Cadence-runtime
decision ledgers and terminal-evidence receipts are intentionally outside this
legacy-GRPO comparison.

The primary performance report must omit checkpoint-bearing steps from its
steady-state window or show checkpoint time separately. Otherwise a segmented
arm is not directly comparable with the original checkpoint-disabled cohort.

## `master_param` patch rationale

Megatron's precision-aware distributed optimizer initializes a parameter's
master and moment state lazily. Distributed checkpoint serialization traverses
every parameter and calls `_get_main_param_and_optimizer_states()`, which
previously assumed that `master_param` already existed. An unupdated MoE expert
or suspended drafter parameter therefore surfaced as `KeyError: master_param`.

The node-local compatibility overlay invokes MCore's existing `init_state_fn`
immediately before serializing a precision-aware optimizer. That callback only
initializes empty entries, so existing master weights and moments remain
unchanged. The overlay is bound to the exact source, patch, and patched-file
SHA256 digests and fails closed on drift; the pinned source checkout is never
modified.

## W&B continuity

Derive one logical run ID from the variant, pinned source SHA, and committed
harness SHA. The no-clobber submission record prevents a second submission of
that identity. Every attempt exports the same `WANDB_RUN_ID`, project, group,
and run name. A deliberate independent replicate must use a new committed
harness identity instead of deleting or reusing the old record.

- First attempt: `WANDB_RESUME=allow`.

- Continuations: `WANDB_RESUME=must`.

- Store the scheduler job ID and segment index as attempt metadata, not as a
  new W&B run identity.

- Continue only after a clean, validated checkpoint exit. A hard crash may
  have uploaded steps newer than the latest checkpoint; blindly resuming that
  W&B run would then duplicate or reorder metrics for replayed steps.

## Canary gates

Retain the existing source-clean, immutable-input, W&B authentication,
state-dict, CUDA Graph, Step 1, Step 2, and first scheduled refit gates. Add the
following segmented-recovery canary for both DFlash and DSpark:

1. Deliberately leave at least one precision-aware optimizer entry lazy, save a
   full optimizer checkpoint, and confirm that the compatibility overlay
   initializes only that empty entry without raising `KeyError: master_param`.

2. Close a full durable intermediate checkpoint, including target and drafter
   weights, optimizer, dataloader payload, and serialized draft-scheduler
   state.

3. End the first process cleanly, start a new scheduler attempt, and confirm it
   selects the highest completed checkpoint rather than any temporary
   directory.

4. Complete at least two further policy steps. Confirm the draft schedule state,
   trainable and serving drafter identity, optimizer state digest, consumed
   samples, and learning-rate schedule advance from the saved boundary.

5. Confirm W&B shows one run with monotonic global steps and no second run ID.

6. Compare an uninterrupted 22-step control with the resumed canary. Require
   equivalent checkpoint/schedule state and science metrics. Do not require
   bitwise-identical generated tokens unless every generation, sampler, worker,
   and CUDA RNG state is explicitly checkpointed and verified.

Static/frozen canaries do not exercise online optimizer or applied-draft
recovery, so they cannot substitute for the two online drafter canaries.

## Exact-trajectory and RNG limitation

Model, optimizer, dataloader, and cadence state are necessary for correct
continuation, but they do not automatically prove bitwise trajectory identity.
Ray actor reconstruction, vLLM scheduling, sampling RNG, CUDA RNG, and request
ordering may change across a process boundary. Consequently:

- performance and acceptance comparisons should use matched post-warmup windows;

- a segmented run should be labelled state-continuous, not exact-trajectory,
  until the uninterrupted/resumed trajectory-hash canary passes; and

- a hard-crash recovery must not be silently merged into the primary science
  cohort when it replays work after the last checkpoint.

## Storage and retention

Keep source and scripts under `/home`, build/venv/cache/lock activity under
`/raid/scratch`, and only durable checkpoints, run contracts, receipts, and
large logs under the experiment's `/lustre/fs1/.../experiments/` root. Never
place the resumable checkpoint root under `/tmp` or `/raid/scratch`.

Use one checkpoint directory per logical arm so two attempts cannot write the
same temporary checkpoint concurrently. The continuation preflight must reject
an active predecessor, source/config digest drift, a changed W&B run contract,
or a cadence result root that does not own the selected checkpoint. Retain only
the terminal checkpoint and two newest recovery checkpoints by default; keep
additional checkpoints only for a diagnosed failure or explicit scientific
comparison.
