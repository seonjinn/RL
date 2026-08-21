# DFlash Policy and Refit Optimization Specification

## Objective

Reduce Qwen3-8B online-drafter policy-training and draft-refit overhead without changing optimizer cadence, draft freshness, loss math, tensor-parallel semantics, or generation behavior.

## Product changes

1. Hidden-state hooks retain detached source views plus tensor version snapshots. Materialization validates that the source tensors were not modified in place, concatenates embeddings and the three auxiliary states once, and returns non-overlapping views backed by that one allocation. Pipeline-parallel transfer keeps its existing communication order and validates sources before send.
2. DFlash export gathers all TP-sharded tensors of the same dtype/device with one flat-buffer collective. Reconstruction preserves parameter order, public names, logical shapes, rank order, and split axes. TP1 and already-logical tensors remain identity paths.

## Non-goals

- Do not change draft update/refit cadence.
- Do not change GBS, MBS, sequence packing, or optimizer hyperparameters in product code.
- Do not alter policy/refit transports, PR9-11 branches, or active 1000-step runs.
- Do not claim an E2E improvement from unit proxies.

## Correctness gates

- Hidden capture must match old values, shapes, dtypes, and devices for MBS1 and MBS2.
- A source in-place mutation after capture must fail before materialization or PP send.
- Hidden capture must use one concatenating allocation and no per-hook clone.
- DFlash TP2 export must match tensor-by-tensor reference values and gradients are irrelevant because export runs under refit/no-grad.
- TP2 export performs one `all_gather` per dtype/device bucket, not one per parameter.
- TP1 export is unchanged.
- Focused tests, Ruff, formatting, Pyrefly/diff checks, and a 4-GPU exact-head gate must pass before benchmarking.

## Performance validation

Use matched Qwen3-8B DFlash K7 fixed/online pairs with seed 42, TP2/PP1/CP1/DP2, four generations per prompt, identical prompt order, checkpoint, container, and CUDA Graph settings:

- GBS32/MBS1 control.
- GBS64/MBS1 to isolate fixed-per-step amortization.
- GBS64/MBS2 to test reduced microbatch and collective-launch overhead.

Exclude steps 0-4. Report time per sample and token, policy/refit/E2E time, generation TPS, acceptance rate, peak memory, draft loss, and update/refit correctness. Run paired/crossover placement where scheduling permits.
