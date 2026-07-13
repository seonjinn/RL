# Qwen3-32B / Qwen3-30B-A3B vLLM Step-3 OOM Root Cause

Date: 2026-06-06 PDT

Update: 2026-06-07 PDT added the Qwen3-30B-A3B GBS2048 retry evidence.

## Verdict

The Step-3 failures are not ordinary rollout KV-cache exhaustion during active
decode. They are vLLM sleep/wake allocator failures in the colocated
NeMo-RL/vLLM lifecycle.

The failed jobs pass Step 1-2, then fail at Step 3 when NeMo-RL prepares
generation after policy training/refit. vLLM is sleeping between phases and
wakes different memory groups by tag. The failure occurs in:

```text
VllmGenerationWorker.wake_up()
  -> vllm.v1.worker.gpu_worker.Worker.wake_up()
  -> vllm.device_allocator.cumem.CuMemAllocator.wake_up(tags)
  -> create_and_map(handle)
  -> CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139
```

So the immediate failing allocation is a CuMem remap during `wake_up`, not a
new prompt/KV allocation after the block manager has filled up.

## Evidence

Original failing envelope:

```text
gpu_memory_utilization=0.90
max_num_batched_tokens=32768
max_num_seqs=32
max_model_len=4096
```

Successful Qwen3-32B retry envelope:

```text
gpu_memory_utilization=0.80
max_num_batched_tokens=16384
max_num_seqs=32
max_model_len=4096
```

Qwen3-30B-A3B GBS2048 failed baseline `3207093` used the same oversized
vLLM envelope as the failed PARD run, but had SpecDec disabled:

```text
policy.draft.enabled=false
speculative_config=None
gpu_memory_utilization=0.90
max_num_batched_tokens=32768
max_num_seqs=32
max_model_len=4096
```

That matters because `3207093` proves the GBS2048 Step-3 OOM is not caused by
the drafter. The matched PARD K3 run `3207094` only adds about 1.1 GiB of model
load footprint (`56.88 GiB` baseline -> `57.99 GiB` PARD), but the baseline
already fails on the same vLLM sleep/wake path.

Qwen3-32B failed baseline `3197798`:

```text
Step 1 total 271.79s, generation 120.33s
Step 2 total 279.17s, generation 117.15s
Step 3 starts generation, then VllmGenerationWorker.wake_up() fails.
```

At engine init, `3197798` reserved a much larger vLLM KV/cache budget:

```text
GPU KV cache size: 1,071,328 tokens
Available KV cache memory: 130.78 GiB
CuMemAllocator sleep freed 161.46 GiB
```

Before the Step-3 failure, vLLM sleep/free still left increasing residual GPU
memory in use:

```text
initial sleep: still in use 3.62 GiB
after Step 1: still in use about 9.3 GiB
after Step 2: still in use about 11.4 GiB
```

The completed Qwen3-32B baseline retry `3197980` reduced the vLLM memory
footprint:

```text
GPU KV cache size: 936,352 tokens
Available KV cache memory: 114.30 GiB
CuMemAllocator sleep freed 144.96 GiB
```

The completed Qwen3-32B PARD K3 retry `3197981` reduced it further:

```text
GPU KV cache size: 647,984 tokens
Available KV cache memory: 113.71 GiB
```

The completed retry proves the integration path works when the vLLM sleep/wake
footprint is smaller. It also makes a pure algorithmic/speculator failure
unlikely.

For Qwen3-30B-A3B GBS2048, the relevant failing memory evidence is:

```text
3207093 vLLM model loading: 56.88 GiB
3207093 GPU KV cache size: 1,152,896 tokens
3207093 max 4096-token request concurrency: 281.47x
3207093 sleep freed: 162.45 GiB
3207093 sleep still in use before failure: about 15.82 GiB
3207093 failure: wake_up(tags=["kv_cache"]) -> CuMem create_and_map OOM
```

Active KV usage was tiny in the emitted vLLM stats, around `0.8%`. The OOM is
therefore from the reserved awake footprint being remapped into too little
headroom, not from live rollout KV blocks filling the cache.

## Code Path

NeMo-RL creates vLLM with sleep mode enabled in:

```text
experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_worker.py
```

The worker `sleep()` path resets prefix cache, calls `self.llm.sleep(level=1)`,
then runs Python GC and `torch.cuda.empty_cache()`.

The worker `wake_up()` path calls `self.llm.wake_up(**wake_up_args)`.

The generation wrapper calls `wake_up` from `prepare_for_generation()` in:

```text
experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/models/generation/vllm/vllm_generation.py
```

GRPO colocated refit uses:

```text
policy.offload_before_refit()
policy_generation.prepare_for_generation(tags=["weights"])
weight transfer/update
policy.offload_after_refit()
policy_generation.prepare_for_generation(tags=["kv_cache"])
```

in:

```text
experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/nemo_rl/algorithms/grpo.py
```

The failed Step-3 trace is exactly on this wake-up path.

## Likely Root Cause

The original `gpu_memory_utilization=0.90` and
`max_num_batched_tokens=32768` combination lets vLLM size a large CuMem/KV
allocation pool. In colocated Full-GRPO, that pool is repeatedly slept,
partially discarded, remapped, and overlapped with Megatron policy memory,
optimizer offload/reload, and refit buffers.

After a couple of GRPO steps, residual non-vLLM GPU usage and/or allocator
fragmentation leaves insufficient contiguous/available addressable memory for
vLLM's `wake_up(tags=["weights"])` remap. Lowering `gpu_memory_utilization` and
`max_num_batched_tokens` shrinks the pool enough that the same loop completes.

This is therefore best classified as a memory-envelope/lifecycle issue:

```text
colocated NeMo-RL refit + vLLM sleep/wake CuMem remap + oversized vLLM KV budget
```

not as:

```text
acceptance-rate issue
drafter quality issue
active decode KV cache usage issue
```

This is not proven to be a monotonic memory leak. The more precise reading is
that the original envelope is near the colocated refit/sleep/wake limit. Step-3
timing and increasing residual memory after sleep make fragmentation/headroom
pressure plausible, but the same code path completes when the vLLM envelope is
shrunk. There is also one useful counterexample: Qwen3-32B PARD K5 job
`3197802` completed five steps at the original envelope. That means
`gmu=0.90/bt32k` is marginal/config-sensitive, not categorically impossible.

## Secondary Code-Review Finding

The cleanup path can hide partial cleanup failures:

```text
VllmGeneration.finish_generation()
  -> run worker sleep/reset methods
  -> returns all(result for result in results if result is not None)
```

If a worker method returns `None`, the current aggregation does not force a
failure. The worker `reset_prefix_cache()` and `sleep()` also do not check the
return value of the underlying vLLM reset. vLLM's block pool can fail to reset
when blocks are still in use, so stricter return-value handling and explicit
block/scheduler diagnostics would help catch dirty state before the later
CuMem wake-up OOM.

This is best treated as a diagnostic and robustness issue, not the primary
root cause, because the observed fatal trace is still the CuMem `wake_up`
remap under an oversized memory envelope.

## Recommended Validation

Run the same Qwen3-32B/Qwen3-30B-A3B matrix with one axis changed at a time:

| Test | Expected result |
|---|---|
| `gmu=0.90`, `bt=16384` | Separates batched-token/KV budget from gmu. |
| `gmu=0.80`, `bt=32768` | Separates gmu headroom from batched-token shape. |
| `gmu=0.70`, `bt=8192` | Conservative fallback for Qwen3-235B while debugging. |
| Disable vLLM sleep mode only for a one-step diagnostic | Tests whether remap itself is the trigger; not expected to be viable for colocated training memory long-term. |
| Recreate vLLM engine every step as a diagnostic | Expensive, but tests whether allocator state persists across sleep/wake cycles. |
| Log `torch.cuda.mem_get_info()` before sleep, after sleep, before `wake_up(weights)`, after refit, and before `wake_up(kv_cache)` | Confirms the exact headroom at the failing remap point. |
| Add strict assertions on reset/sleep success and used/free vLLM blocks before refit | Tests whether requests/blocks remain live across `finish_generation()`. |
| Explicit `kv_cache_memory_bytes` from a known-good run while restoring `max_num_batched_tokens=32768` | Separates static KV sizing from activation/profile pressure. |

## Practical Current Fix

For current performance runs, keep the conservative envelope:

```text
gpu_memory_utilization <= 0.80
max_num_batched_tokens <= 16384
max_num_seqs = 32
```

Submitted validation retries on 2026-06-07 PDT. The first pair,
`3207472`/`3207473`, was cancelled because it was submitted from `/home/sna`;
the container run did not see the Ray startup marker path. The active Lustre
resubmissions are:

| Model | Job | Mode | K | Shape |
|---|---:|---|---:|---|
| Qwen3-30B-A3B | `3207492` | baseline | 0 | GBS2048, 4n4g, `gmu=0.80`, `bt=16384`, worker `max_num_seqs=32` |
| Qwen3-30B-A3B | `3207493` | public PARD | 3 | GBS2048, 4n4g, `gmu=0.80`, `bt=16384`, worker `max_num_seqs=32` |

These are the direct retries for `3207093` and `3207094`. They keep the same
GBS2048 workload but shrink only the vLLM reservation envelope.

For Qwen3-235B fallback, the already submitted `mem70/bt8k` jobs are the right
direction because Qwen3-235B has much less slack around refit and policy/logprob
memory.
