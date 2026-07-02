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
| `2319329` | false | true | recipe default | Running; passed five-minute startup gate |

The successful and failed sync jobs differ in two runtime variables, not one:
vLLM async-engine mode and attention backend. Therefore neither variable is an
isolated root cause yet.

## Current Hypothesis

The strongest current hypothesis is an interaction between the async vLLM
engine, colocated level-1 sleep/wake after weight refit, and multi-node TP8
execution. FlashInfer attention remains a competing variable because the
successful diagnostic explicitly used Triton attention.

The non-colocated async-1off control `2319329` removes sleep/wake while keeping
the async engine. It started on 32 Pretyche nodes with `segment=16`, created W&B
run `80eouh9d`, and passed the five-minute startup gate with zero strict fatal
markers. If it runs beyond the same update boundary, that supports the
sleep/wake interaction hypothesis. A second minimal control should keep the
exact sync recipe and async engine but add only
`attention_backend=TRITON_ATTN`.

## Unrelated vLLM Issue

vLLM issue 42821 and PR 42823 address repeated `model.load_weights()` corrupting
unquantized MoE weights in FlashInfer CUTLASS/TRTLLM layouts. This run used the
Triton MoE backend and failed with a TP collective timeout, not silent output
corruption. That patch is not a demonstrated fix for this failure.

## Scheduling Decision

Pretyche exact-sync Eagle jobs `2319202`-`2319205` and Lyris colocated PARD
jobs `2261382`-`2261383` are held to avoid consuming 32-node allocations on a
baseline path that is currently unstable. Lyris exact async-1off jobs remain
eligible, and Pretyche non-colocated control `2319329` is running.

No NeMo-RL or vLLM core patch has been applied for this issue.
