# Qwen3-235B Exact-Recipe Rollout Timeout

Date: 2026-07-02

## Outcome

Pretyche job `2319201` inherited the Qwen3-235B performance recipe's
`async_engine=true` setting and reached Step 6. It then failed with exit code
`1:0` after every async rollout engine lost at least one TP8 worker.

The first failing collective was a 600-second NCCL `_ALLGATHER_BASE` during
vLLM `sample_tokens()`. Workers in the same failure window reported collective
sequence numbers `32471`, `32477`, and `32478`. This is not a slow collective
that eventually completed; the watchdog terminated the workers and all engine
cores became dead.

## Failure Timeline

1. The run completed W&B Steps 2-4 and entered training Step 5.
2. At 11:07 PDT all 16 colocated engines entered vLLM level-1 sleep.
3. At 11:08 PDT they woke the `weights` and `kv_cache` tags after refit.
4. At about 11:09 PDT TP8 workers entered `_ALLGATHER_BASE` operations with
   `18,992` input and `151,936` output elements.
5. At 11:19 PDT the 600-second watchdog fired across multiple engines and
   nodes. The job entered Step 6 only to surface `EngineDeadError`, then failed.

The failures span many nodes, so the current evidence does not point to one bad
GPU or one bad host.

## Controlled Comparison

| Job | Colocated | vLLM async engine | Attention | Result |
|---|---|---|---|---|
| `2318729` | true | false | `TRITON_ATTN` | Completed 20/20 |
| `2319201` | true | true | recipe default, resolved to FlashInfer | Failed in Step 6 |
| `2319329` | false | true | recipe default | Failed in Step 1 policy/reference logprob |
| `2319427` | false | true | recipe default | Running with topology-aware segment placement |

The successful and failed sync jobs differ in two runtime variables, not one:
vLLM async-engine mode and attention backend. Therefore neither variable is an
isolated root cause for the colocated failure yet.

## Current Hypothesis

The non-colocated async-1off control `2319329` removed sleep/wake while keeping
the async engine. It completed its first rollout and reward computation, but
then failed after 26:35 in policy/reference logprob. The first observed timeout
was TP `_ALLGATHER_BASE`; EP `ALLTOALL_BASE` and pipeline collectives timed out
in the same failure window. The first collective made no progress for 600
seconds, so increasing the timeout is not supported by the evidence.

The resolved config for `2319329` had `cluster.segment_size=None`, even though
the Slurm allocation used `--segment=16`. NeMo-RL only activates its
topology-aware Ray placement constraints when `cluster.segment_size` is set.
With `None`, the 16 training nodes and 16 inference nodes can be interleaved
across two NVLink segments. This is consistent with the earlier failed smoke
job `2318035`. In contrast, control job `2318343` completed when Slurm used
`--segment=8` and Hydra used `cluster.segment_size=8`, keeping each eight-node
role inside one segment.

Job `2319427` is the minimal config-only retry. It preserves the exact
performance async-1off recipe and adds only `cluster.segment_size=16`, matching
Slurm `--segment=16`. It passed `sbatch --test-only` and started on nodes
`ptyche[0181-0196,0217-0232]`. It also passed five minutes of startup monitoring
with zero NCCL timeout, `EngineDeadError`, or distributed-backend fatal
markers. All 128 Ray actors joined, W&B run `yy7q1jcl` was created, and the
resolved vLLM 0.20.0 config confirms CUDA Graphs, prefix caching, chunked
prefill, and the Triton MoE backend. CUDA Graph capture completed and the job
entered Step 1/20. If it passes policy logprob and completes update boundaries,
topology placement is the demonstrated root cause for the non-colocated
failure. The colocated Step 6 sleep/wake failure remains a separate issue until
an exact-sync control isolates the attention backend.

## Unrelated vLLM Issue

vLLM issue 42821 and PR 42823 address repeated `model.load_weights()` corrupting
unquantized MoE weights in FlashInfer CUTLASS/TRTLLM layouts. This run used the
Triton MoE backend and failed with a TP collective timeout, not silent output
corruption. That patch is not a demonstrated fix for this failure.

## Scheduling Decision

Pretyche exact-sync Eagle jobs `2319202`-`2319205` and Lyris colocated PARD
jobs `2261382`-`2261383` are held to avoid consuming 32-node allocations on a
baseline path that is currently unstable. Lyris exact async-1off jobs remain
eligible, and Pretyche topology-aware control `2319427` is running.

No NeMo-RL or vLLM core patch has been applied for this issue.
