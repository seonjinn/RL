# vLLM 0.25.1 Drafter Matrix Live Report

## Status

The matrix, strict result collector, and Lyris submission workflow are under
validation. No result below is considered reportable until its exact matched
baseline and candidate both complete steps 2-20.

| Field | Controlled value |
|---|---|
| Branch | `sna/nemorl-vllm0251-drafter-matrix-20260716` |
| vLLM | official 0.25.1 |
| Cluster | Lyris GB200, `coreai_dlalgo_llm`, `--segment=<nodes>` |
| CUDA Graph | enabled, `FULL_AND_PIECEWISE`, native sizing |
| Sampling | temperature 1.0, top-p 1.0 |
| Checkpoint saving | disabled |
| W&B project | `nemo-rl-vllm0251-drafter-matrix` |
| Final window | steps 2-20 inclusive |

## Lyris Preflight Inventory

Read-only inspection on 2026-07-16 found all three target snapshots, all three
EAGLE3 snapshots, and the shared PARD snapshot complete at their pinned
revisions. Both exact DFlash snapshots are absent and must pass the staging
job before DFlash smoke submission. The available image is
`nemo_rl_nightly_20260715.sqsh`; the older derived image referenced by the
previous experiment has been removed.

At that snapshot, partition `gb200` had no idle nodes: 244 were allocated and
the remainder were in maintenance/down/drain states. FairShare for user `sna`
under `coreai_dlalgo_llm` was 0.793651. Scheduler preflight may run, but smoke
jobs can remain pending until nodes return.

## Applicability And Run Ledger

`planned` means the exact checkpoint/config is defined but has not yet passed
the current branch's two-step runtime gate. Job IDs and W&B links are added only
after real submission.

| Model | Variant family | Candidates | Runner | State | Job/W&B | Reason or gate |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | baseline | MRv2, MRv1 | mixed | planned | pending | exact controls for both runner families |
| Qwen3-30B-A3B | EAGLE3 | K1, K3, K5 | MRv2 | planned | pending | exact base-model head |
| Qwen3-30B-A3B | DFlash | K3, K5 | MRv2 | planned | pending | exact head; draft FlashAttention |
| Qwen3-30B-A3B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-30B-A3B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |
| Qwen3-32B | baseline | MRv2, MRv1 | mixed | planned | pending | exact controls for both runner families |
| Qwen3-32B | EAGLE3 | K1, K3, K5 | MRv2 | planned | pending | exact base-model head |
| Qwen3-32B | DFlash | K3, K5 | MRv2 | planned | pending | exact head; draft FlashAttention |
| Qwen3-32B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-32B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |
| Qwen3-235B-A22B | baseline | MRv2, MRv1 | mixed | planned | pending | exact controls for both runner families |
| Qwen3-235B-A22B | EAGLE3 | K1, K3, K5 | MRv2 | planned | pending | exact NVIDIA head |
| Qwen3-235B-A22B | DFlash | K3, K5 | MRv2 | unsupported | n/a | no exact public Qwen3-235B DFlash checkpoint |
| Qwen3-235B-A22B | draft/PARD | draft K1/K5; PARD K5/K16 | MRv1 | planned | pending | shared AMD 0.6B drafter; sequential/parallel split |
| Qwen3-235B-A22B | suffix/ngram | suffix K32; ngram K5; ngram-gpu K5 | MRv1 | planned | pending | checkpoint-free proposers |

## Explicitly Out Of Scope

| Method | State | Reason |
|---|---|---|
| Native MTP | unsupported | controlled Qwen3 targets do not embed MTP heads |
| DSpark | unsupported | no exact target-specific checkpoint |
| Medusa | unsupported | no exact target-specific checkpoint |
| `mlp_speculator` | unsupported | vLLM 0.25.1 MRv1 runtime gap |
| hidden-state extraction/custom class | excluded | not standalone acceleration proposers |
| PARD-2 | separate experiment | requires non-upstream method/checkpoint support |
| DFlare | separate experiment | requires non-upstream implementation |

## Promotion Gates

1. `show` resolves a valid model/method/checkpoint and preserves recipe controls.
2. `test-only` passes for the exact topology with no dependency or `--gres`.
3. `smoke2` reaches the second completed training step and emits required metrics.
4. `smoke5` confirms stable runner, CUDA Graph mode, and acceptance telemetry.
5. `final20` completes; steps 2-20 are compared only to the exact runner-matched baseline.

## Result Table

| Model | Variant | Steps | E2E time | E2E TPS/GPU | Gen time | Gen TPS/GPU | Gen ratio | Acceptance | Mean accepted | Speedups | W&B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Validation

Run before pushing and again from the clean cluster checkout:

```bash
pytest -q \
  tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/experiments/test_vllm_0251_drafter_results.py \
  tests/experiments/test_vllm_0251_drafter_staging.py \
  tests/experiments/test_vllm_0251_suffix_dependency.py
bash -n experiments/vllm_0251_drafter_matrix/submit_matrix.sh
bash -n experiments/vllm_0251_drafter_matrix/submit_stage_drafters.sh
ruff check \
  experiments/vllm_0251_drafter_matrix \
  tests/experiments/test_vllm_0251_drafter_matrix.py \
  tests/experiments/test_vllm_0251_drafter_results.py
uv lock --check
git diff --check
```
