# DFlare Result Integration and Reliability Design

## Goal

Publish every completed July 3-4 vLLM 0.24 long-context result and make the
AngelSlim DFlare runner preserve usable measurements when distributed result
collection or the SLURM wall-time fails.

## Scope

- Normalize the 20 Native 32K, 20 YaRN 64K, and 20 YaRN total-128K
  vLLM-native result files.
- Add the three newly completed DFlare results to the existing five-row
  completed table.
- Track timeout, OOM, and incomplete DFlare jobs separately from performance
  rows.
- Fix the DFlare rank-0 OOM caused by gathering CUDA-backed Python objects.
- Save rank-local compact metrics before distributed collection so completed
  shards survive a later timeout or gather failure.
- Restage the patch and rerun the missing DFlare profiles.

## Result Model

The vLLM-native table uses one row per batch size and matches a baseline by:

`runtime, model, domain, temperature, top_p, batch_size, ISL, OSL, context profile, position encoding, CUDA graph mode, and setup`.

The report shows throughput, throughput speedup, latency, latency speedup,
acceptance rate, mean accepted length, method K, job/source path, and completion
state. No speedup is computed across AngelSlim and vLLM-native runtimes.

AngelSlim DFlare remains a separate completed-results section because its
serial Transformers runner and PyTorch SDPA fallback are not comparable to the
vLLM-native engine.

## DFlare Collection Fix

The current runner gathers response objects that contain CUDA `output_ids`.
`torch.distributed.gather_object()` unpickles those tensors on rank 0 and can
initialize or allocate storage on every source CUDA device. At 128K context,
that allocation fails after generation has already completed.

Before collection, each response is reduced to a CPU-only record containing:

- output token count
- time per output token
- acceptance-length list
- block size and run mode

No generated token tensor is required after local decoding. Each rank writes
an atomic `<output>.rank<N>.partial.json` file before entering the collective.
Rank 0 gathers only these compact records and writes the existing final JSON
schema, preserving compatibility with the report parser.

## Long-Running Jobs

The six-hour process-group timeout remains necessary but does not override the
five-hour `gb200` partition wall-time. Rank-local partial files distinguish a
slow shard from a failed collective and preserve completed shards. Final
long-context retries use the patched runner; profiles that cannot fit five
hours use the eight-hour `gb200-backfill` partition and retain partial output
if preempted.

FlashAttention availability is recorded in every row. Results produced by the
current PyTorch SDPA fallback remain valid but are labeled as such. A separate
staging smoke determines whether FlashAttention 2 can be installed in the
ARM64 image before it is used for reportable comparisons.

## Error Handling

- Atomic JSON writes use a temporary sibling followed by rename.
- Final results include only complete matched rows.
- Rank-local partial results are labeled partial and never used for speedup.
- TIMEOUT, OOM, process-group timeout, and missing-result states are retained
  in a status table with job IDs and root causes.
- Duplicate retries are de-duplicated by setup key, preferring the latest
  complete result.

## Verification

- Unit tests prove CUDA-like response payloads are converted to CPU-only
  compact records before gather.
- Unit tests cover rank partial JSON and final aggregation.
- Result-parser tests cover all three context profiles and exact baseline
  matching.
- The generated HTML must parse successfully and contain all expected job IDs.
- Remote smoke jobs must remain error-free for five minutes before production
  retries are accepted.
