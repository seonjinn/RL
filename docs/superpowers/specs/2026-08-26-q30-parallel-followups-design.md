# Qwen3-30B-A3B Cadence Parallel Follow-ups Design

## Goal

Prepare three independent deliverables while the six submitted Qwen3-30B-A3B
cadence jobs wait for OCI-HSG allocation:

1. a reproducible W&B-to-HTML result pipeline;
2. balanced Adaptive-v2 200-step DFlash and DSpark experiment inputs;
3. an evidence-graded PR1-11 validation matrix for sequence packing, context
   parallelism, repeated online update/refit, and multi-node execution.

The submitted jobs and their pinned remote product/harness revisions must remain
unchanged.

## Result Pipeline

The collector discovers only runs in W&B project `sna-specdec`, group
`q30ba3b-draft-cadence-200step-20260826`. It aggregates the closed step window
3-200 with `scan_history`, retaining included steps, missing steps, and the valid
count for each metric. It never reconstructs throughput from averaged time.

The report includes generation throughput, generation time, E2E throughput,
E2E step time, policy training, policy/reference logprob, refit, acceptance rate,
mean accepted length, cadence reason counts, and completed-step status. Always
and fixed-10 are compared with the static drafter of the same drafter family.
This is explicitly labelled a cadence-relative comparison; the matrix has no
matched no-SpecDec baseline and must not claim SpecDec-versus-baseline speedup.

Collection produces deterministic JSON and a self-contained HTML page. A run
with incomplete history remains visible and is labelled preliminary. A missing
matched static row reports `waiting static baseline` instead of a speedup.
`WANDB_API_KEY` is accepted only through the environment and is never rendered,
logged, or serialized.

## Adaptive-v2 Candidate

Adaptive-v2 is a balanced hypothesis, not a claimed optimum. Both DFlash and
DSpark use the same workload, topology, optimizer, sequence-packing, CUDA Graph,
checkpoint, and 200-step settings as their submitted fixed-10 arm. Only the
update schedule changes:

```yaml
mode: adaptive
action: sparse_update
min_interval: 10
max_interval: 40
ewma_alpha: 0.2
degradation_threshold: 0.03
recovery_threshold: 0.01
min_observations: 10
max_burst_updates: 2
```

The 10-step floor prevents more frequent updates than fixed-10, the 40-step cap
prevents indefinite acceptance drift, the 3-point degradation trigger filters
small noise, and the two-update burst cap limits recovery overhead. The
candidate is prepared and composition-tested but not submitted in this scope.

## PR1-11 Validation Matrix

Every matrix cell records one of five evidence grades:

- `code`: the path is implemented but not exercised;
- `unit`: deterministic automated tests pass;
- `composed`: exact Linux/container configuration composition passes;
- `scheduled`: an exact scheduler preflight or job receipt exists;
- `runtime`: a GPU run crossed its required behavioral gate.

The matrix covers DFlash and DSpark across CP1 packed sequences, CP>1 unpacked
sequences, CP>1 packed sequences, repeated update/refit, and multi-node
execution. Evidence must include a file, commit, job ID, or W&B link. Absence of
runtime evidence is written as an open validation item, never inferred from
config support flags.

## Isolation and Verification

All files are added only to the isolated branch
`codex/dflash-dspark-cadence-latest-main-20260826`. The stable source SHA
`4ee518b5dc2ed16f75e31876b477ea5ecf7d8c9b`, submitted product SHA
`1be8237816bfd78dad752dd5c1e0149ae2420301`, and remote harness SHA
`6c51b26dc531a7b0b1ca88b9d0f02c882d2c8664` remain untouched. Production code
is developed test-first; human documentation is reviewed against concrete
evidence rather than given text-change tests.
