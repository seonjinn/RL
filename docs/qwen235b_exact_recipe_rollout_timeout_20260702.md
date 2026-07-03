# Qwen3-235B Exact-Recipe Rollout Timeout

Date: 2026-07-02

## Outcome

Pretyche job `2319201` inherited the Qwen3-235B performance recipe's
`async_engine=true` setting and reached Step 6. It then failed with exit code
`1:0` after every async rollout engine lost at least one TP8 worker. The later
Triton-attention control `2319650` passed the earlier Step 8 boundary and
preserved W&B Steps 2-11, but reproduced the rollout TP8 timeout in Step 12.

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
| `2319427` | false | true | recipe default | Passed initial topology boundary; rollout TP8 timed out in Step 8 |
| `2319650` | false | true | `TRITON_ATTN` | Passed Step 8; rollout TP8 timed out in Step 12 |

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
Slurm `--segment=16`. It passed `sbatch --test-only`, all 128 Ray actors joined,
and W&B run `yy7q1jcl` was created. The run passed the former Step 1
policy/reference-logprob failure boundary and entered Step 8, confirming that
missing topology-aware placement caused that earlier failure.

The run then exposed a separate rollout failure. At 12:49 PDT, vLLM TP8
`_ALLGATHER_BASE` sequence `59786` timed out after 600 seconds during Step 8
decode. The captured stack is `qwen3_moe.compute_logits -> LogitsProcessor ->
tensor_model_parallel_all_gather`; this is not the later weight-update call.
Each rank had `1,291,456` input elements and `10,331,648` output elements. The
watchdog aborted Ray rollout workers, the EngineCore became dead, and subsequent
generation and weight-update calls raised
`EngineDeadError`. Topology-aware placement therefore fixes initial
train/inference placement but does not make the exact async rollout TP8 decode
path stable through 20 steps.

Pretyche job `2319650` kept the exact async-1off recipe, TP8, 32x4 shape,
Slurm/Hydra segment 16, CUDA Graphs, and Triton MoE unchanged, and changed only
the attention backend to `TRITON_ATTN`. It passed scheduler preflight as
test-only job `2319649` and ran farther than the default-attention control, but
still failed in the same rollout logits-gather path. Attention backend selection
therefore changes when the failure appears but is not a demonstrated fix.

## Step 12 Rank-Level Evidence

The first fatal operation in job `2319650` was TP process group 2 sequence
`104532`, `_ALLGATHER_BASE`, with `1,595,328` input elements and `12,762,624`
output elements. All eight TP ranks reported the same sequence number, operation,
input/output sizes, and process-group progress: last enqueued `104532`, last
completed `104531`. The logged placement put ranks 0-3 on `10.52.103.44` and
ranks 4-7 on `10.52.103.45`, so this TP8 group spanned two four-GPU compute
nodes.

The stack is:

`qwen3_moe.compute_logits -> LogitsProcessor._gather_logits ->`
`tensor_model_parallel_all_gather -> torch.distributed.all_gather_into_tensor`.

This evidence argues against collective-order or tensor-shape divergence for
this occurrence: every rank reached the same collective, but it made no progress
for 600 seconds. It does not by itself distinguish a cross-node NCCL transport
stall, a device-side execution stall, or an interaction with the vLLM
`AsyncLLM` EngineCore/Ray compiled-DAG executor. The later Megatron EP and PP
timeouts occurred only after the rollout engine died and are secondary.

Cross-node TP8 is not sufficient to reproduce the problem by itself. Diagnostic
job `2318729` also used TP8 and `TRITON_ATTN` and completed 20/20, but it used
the synchronous vLLM engine rather than the performance recipe's
`async_engine=true` path. The remaining boundary is therefore the long-lived
async EngineCore/Ray execution path combined with cross-node TP8, not simply the
attention backend or topology placement.

A target TP4 exact-recipe smoke run is the smallest useful isolation control:
it would keep each rollout TP group within one four-GPU node while retaining the
async engine and other recipe settings. The checkpoint contains 437.90 GiB of
safetensor shards. Job `2319650` measured 54.92 GiB of model memory per TP8 rank,
which is consistent with the 54.74 GiB static shard estimate; TP4 therefore
requires about 109.48 GiB per rank before KV cache and CUDA Graph overhead. The
resolved recipe uses dedicated generation GPUs with
`gpu_memory_utilization=0.8`, and TP8 engine initialization consumed 154,482 MB
of the 189,471 MB visible per GPU. TP4 is therefore plausible on a 192 GB B200
because vLLM can trade KV-cache capacity for the larger weight shard, but it is
not proven until engine initialization and one complete rollout pass.

The engine layout must be controlled separately from memory fit. The exact
recipe reserves 16 generation nodes, or 64 GPUs. TP8 creates eight engines; a
TP4-only override would create 16 engines and halve requests per engine, so it
would not isolate cross-node TP communication. The controlled smoke should use
TP4 and eight generation nodes to retain eight engines. Keeping the 32-node
allocation and segment 16 for the three-step diagnostic preserves the training
shape and topology at the cost of eight idle nodes. A later performance run
would need a separately matched TP4 baseline rather than comparison against the
published TP8 baseline.

This control is also directly supported by the runtime warning: vLLM reports
that TP8 exceeds the four GPUs reserved per node, spreads each engine across two
nodes, and disables custom all-reduce for the cross-node process group. No TP4
job or core-code change has been launched from this analysis.

## Unrelated vLLM Issue

vLLM issue 42821 and PR 42823 address repeated `model.load_weights()` corrupting
unquantized MoE weights in FlashInfer CUTLASS/TRTLLM layouts. This run used the
Triton MoE backend and failed with a TP collective timeout, not silent output
corruption. That patch is not a demonstrated fix for this failure.

## Scheduling Decision

Pretyche exact-sync Eagle jobs `2319202`-`2319205`, exact async Eagle jobs
`2319487`-`2319489`, and Triton-attention async Eagle jobs `2319975`-`2319977`
remain held. On Lyris, the matched PARD smoke baseline `2261382` remains held,
while the target-TP8/draft-TP8 PARD K1 smoke `2261383` is eligible and scheduled.
The exact async baseline `2261942`, exact async baseline/Eagle pair
`2264402`-`2264403`, and exact sync baseline `2264443` are also eligible. The
target-TP8/draft-TP1 PARD jobs `2261943`-`2261946` and `2264444`-`2264447`
remain held because vLLM V1 rejects unequal target and draft TP sizes.

No NeMo-RL or vLLM core patch has been applied for this issue.
