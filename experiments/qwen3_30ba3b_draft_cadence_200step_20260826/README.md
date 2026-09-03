# Qwen3-30B-A3B paired CUDA Graph drafter-cadence study

This experiment compares DFlash and DSpark K5 drafter cadences. The primary
`baseline` and `-cg2048` matrix is one matched official-performance-recipe
cohort: GBS 2048, OSL 4096, TP1/EP16/PP1/CP1, sequence packing with fused loss,
the current 25,391-step drafters, and a dense FAP CUDA Graph ladder through
2048. Historical non-`cg2048` static/always configs remain available only to
reproduce the earlier GBS-512 study.

## Paired CUDA Graph matrix

| Cohort | Drafter | Schedule | Matched FAP variant |
|---|---|---|---|
| Official performance recipe | None | No SpecDec baseline | `baseline` |
| Official performance recipe | DFlash K5 | Frozen/static | `dflash-static-cg2048` |
| Official performance recipe | DFlash K5 | Refit every step | `dflash-always-cg2048` |
| Official performance recipe | DFlash K5 | Refit every 5 steps | `dflash-fixed5-cg2048` |
| Official performance recipe | DFlash K5 | Refit every 10 steps | `dflash-fixed10-cg2048` |
| Official performance recipe | DFlash K5 | Refit every 20 steps | `dflash-fixed20-cg2048` |
| Official performance recipe | DFlash K5 | Adaptive v2 | `dflash-adaptive-v2-cg2048` |
| Official performance recipe | DSpark K5 | Frozen/static | `dspark-static-cg2048` |
| Official performance recipe | DSpark K5 | Refit every step | `dspark-always-cg2048` |
| Official performance recipe | DSpark K5 | Refit every 5 steps | `dspark-fixed5-cg2048` |
| Official performance recipe | DSpark K5 | Refit every 10 steps | `dspark-fixed10-cg2048` |
| Official performance recipe | DSpark K5 | Refit every 20 steps | `dspark-fixed20-cg2048` |
| Official performance recipe | DSpark K5 | Adaptive v2 | `dspark-adaptive-v2-cg2048` |

The non-`cg2048` `dflash-static`, `dflash-always`, `dspark-static`, and
`dspark-always` variants retain their original 16-prompt × 32-generation,
global-batch-512, 8192-token, TP2/EP8 workload and checkpoint/runtime settings.
Do not compare the legacy fixed-vs-always cohort directly with the official performance-recipe cohort.
All baseline speedups reported for the primary study must use `baseline` versus
the `-cg2048` variants over the same completed step window.

The explicit ladder preserves every default bucket through 512, adds
576/640/704/768 to cover the high-concurrency K5 verification range, and then
adds 832/896/960/1024/1280/1536/1792/2048 for mixed prefill-decode coverage.
DFlash also includes 2046 because its MRV1 FULL-decode path rounds capture
sizes to multiples of `K + 1 = 6`; without that entry, a nominal 2048 ceiling
would round above the limit and silently fall back to 1794. The 768 bucket is
the important one for the current 128-request generation worker
(`128 * (K + 1) = 768`). Buckets above 768 are deliberately retained as an
experimental request, but they are not expected to improve pure K5 decode at
the current concurrency and may increase graph memory and startup time.

Here, fixed interval N means that the drafter is trained and refit every N
policy steps. It is different from the separate `fixed` arms in the
fixed-versus-always study, where the drafter remains frozen and receives no
online update.

## Preserved pair contracts

Within the matched official performance-recipe cohort, including baseline,
frozen/static, always, fixed-interval, and adaptive-v2, the inherited workload remains
4 nodes × 4 GPUs, 64 prompts × 32 generations, global batch 2048, a 4096-token
policy/generation limit, TP1/EP16/PP1/CP1, sequence packing with fused loss,
validation every 10 steps, shuffled OpenMathInstruct-2 with a 5% validation
split, and the Triton MoE backend. The original unsegmented cohort disabled
checkpoint writes; the OCI-HSG four-hour segmented launcher uses the
checkpoint contract in `SEGMENTED_CHECKPOINT_PLAN.md`. Those cadence overlays
add only:

- 200 total steps;
- local target and tokenizer paths;
- W&B cadence metrics without durable checkpoint evidence;
- one K5 DFlash or DSpark drafter, optimizer, and fixed update schedule;
- `policy.offload_optimizer_for_refit=false`, which prevents the optimizer CPU
  copy from overlapping vLLM's sleep-weight backup on the GB200 nodes.

The checked-in launcher uses OCI-HSG `batch` with a four-hour scheduler limit,
enables standard NeMo-RL checkpoints, and asks NeMo-RL to checkpoint and exit
before the scheduler deadline. Segmented execution is restricted to the
matched `*-cg2048` cohort (plus its baseline); legacy and retry variants fail
before writing submission artifacts. The official performance path remains
legacy GRPO with both `data_plane.enabled=false` and
`cadence_runtime.enabled=false`. Online updates are controlled only by
`policy.draft.update_schedule`; the regular checkpoint stores policy, drafter,
optimizer, dataloader, and draft-scheduler state. On resume, the normal initial
policy-to-generation refit transfers both target and draft weights. Checkpoint
steps must be excluded from steady-state performance windows or reported
separately, so checkpoint I/O is not mistaken for cadence overhead.

The earlier `KeyError: master_param` came from Megatron distributed-checkpoint
serialization, not from a missing trained weight. Megatron's precision-aware
distributed optimizer creates per-parameter master state lazily, while the
serializer traverses parameters that may not yet have optimizer state. The
node-local compatibility overlay invokes MCore's existing `init_state_fn`
before serialization; it initializes only empty entries and is pinned by
source, patch, and output SHA256 digests. The pinned NeMo-RL source also treats
a timeout checkpoint as a resumable boundary instead of trying to close the
run as terminal before Step 200. A save-and-resume canary must pass before the
segmented launcher is used for the full matrix.

The CUDA Graph and first two step gates retain 45-minute diagnostic deadlines;
the first scheduled refit gate waits while the training process is alive and
therefore cannot misclassify a slow fixed-20 run as a refit hang.

The launcher does not override `data_plane.enabled`, vLLM `max_num_seqs`, the
compilation backend, CUDA Graph mode, or capture sizes on the command line.
Default references retain their existing runtime coverage; each `-cg2048`
sibling carries the approved compilation block in its selected config.

DSpark jobs additionally copy the container's installed vLLM package to
node-local scratch and apply the ten runtime-file changes from vLLM #48167,
followed by the two stale-causality replacements from commit `bf372f9bb5`.
Code tracing showed that its attention guard alone is insufficient on v0.25.1:
the stock model runner creates the CUDA Graph manager before draft attention is
initialized and the stock DSpark speculator does not consult the corrected
support classification. The PR's final patch also removed `dflash_causal`
initialization while leaving two CUDA Graph call sites that still read it. The
follow-up uses the group-level causality initialized by `set_attn`. The helper
pins both patch digests, requires a clean forward or reverse application,
writes per-file digests, and fails closed on source drift. DFlash remains on
the unmodified package.

## Immutable inputs

- Product source: `/home/sna/nemorl-q30-cadence-syncfix-product-20260902`
- Product SHA: `9be09f0eb9120e37ab9e4e51ecca98f11d9814da`
- Every DSpark online cg2048 arm excludes `nvl72118-T01`, where job 6826731
  received a host-cgroup OOM during its first drafter update. The
  `dspark-always-cg2048-retry` alias reuses the matching always-update config.
- Target: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf-local/Qwen/Qwen3-30B-A3B`
- DFlash: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dflash/exported-checkpoint-25391`
- DSpark: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/assets/q30-base-nemotron-b8-full-s25391-v1/base-dspark/exported-checkpoint-25391`
- W&B project: `sna-specdec`
- W&B group: `q30ba3b-draft-cadence-200step-20260826`

## Validation and submission

Each variant must pass the exact state-dict check, composed-config verifier,
W&B authentication, scheduler dry-run, CUDA Graph gate, Step 1 gate, and Step 2
gate. The launcher waits for a deterministic first-refit gate at Step 1 for
`always` and at Step 5, 10, or 20 for fixed cadences. Frozen/static has no refit
to gate, while adaptive-v2 has no predetermined refit step; the launcher does
not claim a deterministic refit-gate pass for either schedule. `dspark-fixed5`
is the correctness canary for the vLLM runtime patch; `dflash-fixed20`
independently validates the corrected long-run gate on the unchanged DFlash
runtime.

Before any four-hour segmented matrix is submitted, one online canary per
drafter must also pass the recovery gates documented in
`SEGMENTED_CHECKPOINT_PLAN.md`: durable intermediate closure, clean restart at
the next policy step, restored optimizer and schedule identity, restored
serving-draft weights, and one continuous W&B run. Passing those gates proves
stateful recovery. It does not by itself prove bitwise-identical trajectories;
the resumed and uninterrupted canaries must compare trajectory inputs or hashes
before the study claims exact RNG equivalence.

```bash
uv run --no-project --with pytest --with pydantic python -m pytest -q \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/tests \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/reporting/tests
uv run --no-project --with ruff ruff check \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826
bash -n \
  experiments/qwen3_30ba3b_draft_cadence_200step_20260826/submit_qwen3_30ba3b_cadence_200step.sh
```
