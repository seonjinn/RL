# Async GRPO: CUDA Illegal Memory Access After First Weight-Updating Refit

## Context

We are comparing **synchronous** and **asynchronous GRPO** (Generalized Reinforcement
Policy Optimization) for training a Vision Language Model (VLM) on a 4-node GPU cluster.

- **Sync GRPO** colocates generation and training on all 4 nodes. Each step does:
  generate rollouts → compute logprobs → train → repeat.
- **Async GRPO** separates generation and training onto different nodes. A background
  `AsyncTrajectoryCollector` generates rollouts continuously and pushes them into a
  `ReplayBuffer`. The training loop pulls batches from the buffer and trains
  independently. After each training step, updated weights are broadcast to the
  generation workers via NCCL ("refit").

The goal is to measure whether async GRPO improves GPU utilization by overlapping
generation and training.

### Codebase Layout

- **Async GRPO entry point**: `nemo_rl/algorithms/grpo.py` → `_async_grpo_train()`
  (line ~3200). Orchestrates the buffer, collector, training loop, and refit cycle.
- **Sync GRPO entry point**: `nemo_rl/algorithms/grpo.py` → `grpo_train()`
  (line ~1700). Single-threaded generate → train loop.
- **Replay buffer & collector**: `nemo_rl/algorithms/async_utils.py` →
  `ReplayBuffer` (line ~50) and `AsyncTrajectoryCollector` (line ~400).
- **Megatron policy worker**: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  → `MegatronPolicyWorker.train()` (line 271), `offload_before_refit()` (line 1188),
  `prepare_for_training()` (line 1168), `offload_after_refit()` (line 1222).
- **Weight refit logic**: `nemo_rl/algorithms/grpo.py` → `refit_policy_generation()`
  (line 1438). In non-colocated mode (async), uses NCCL broadcast (line 1513-1520).
  In colocated mode (sync), uses ZMQ IPC (line 1496-1505).
- **Crash site (surface)**: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/utils/train_utils.py`
  → `logical_and_across_model_parallel_group()` (line 318).
- **Launch scripts**: `batch_nanov3_gym_grpo.sh` (sync),
  `batch_nanov3_gym_grpo_async.sh` (async).

### Model

Nemotron-3-Nano-Omni-30B-A3B — a hybrid Transformer/Mamba Mixture-of-Experts (MoE)
model. Relevant parallelism settings on training nodes:

- **TP=8** (tensor parallel, all 8 GPUs per node)
- **EP=8** (expert parallel, one expert shard per GPU)
- **CP=1** (no context parallelism)
- **DP = total_train_GPUs / (TP × CP)** (data parallel, varies by config)

Generation uses vLLM with **TP=2** (4 vLLM instances per generation node).

### Experiment Configuration

All runs use:
- `num_prompts_per_step=15`, `num_generations_per_prompt=16` → 240 total generations
- `train_global_batch_size=240`
- `max_num_epochs=1000` (effectively unlimited; runs are time-limited to 4 hours)
- `max_trajectory_age_steps=1` (only data from the current policy version is used)
- `in_flight_weight_updates=False`
- `colocated.enabled=False` (generation and training on separate nodes)
- `use_distributed_optimizer=True`, `use_precision_aware_optimizer=True`,
  `store_param_remainders=True`
- `optimizer_cpu_offload=False`, `offload_optimizer_for_logprob=False`,
  `is_generation_colocated=False`
- `reference_policy_kl_penalty=0.0` (no reference model used)
- `enable_cuda_graph=False` (CUDA graphs disabled for Megatron training)
- Dataset: `mix_text_vision_2k_dfw_image_only_filtered.jsonl` (902 entries)
- Generation backend: NeMo Gym (agentic rollout with judge evaluation)

---

## Observed Behavior

All three async GRPO configurations crash with:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The crash follows a consistent pattern: Step 1 trains successfully, the post-Step 1
refit (the first weight-updating refit) succeeds, and then Step 2's training crashes.

| Job ID   | Config | Train Nodes | DP | Step 1 Train | Post-Step 1 Refit | Step 2 Train |
|----------|--------|-------------|----|--------------|--------------------|--------------|
| 11978373 | 1g3t   | 3           | 3  | OK           | OK                 | **CRASH**    |
| 11978375 | 2g2t   | 2           | 2  | OK           | OK                 | **CRASH**    |
| 11978429 | 3g1t   | 1           | 1  | OK           | OK                 | **CRASH**    |

The sync GRPO job (11978368, 4 colocated nodes) does **not** crash.

### Precise Lifecycle at Crash

```
Cycle 0 (setup):    refit₀ (broadcast initial weights to gen workers)
Cycle 1 (Step 1):   logprobs → train₁ → refit₁ (broadcast updated weights)
Cycle 2 (Step 2):   logprobs → train₂ → CRASH (during train₂)
```

Only step headers `Step 1/100000` and `Step 2/100000` are printed. The crash happens
during Step 2's `policy.train()` call (line 3884 in `grpo.py`), which is the first
training call after a **weight-updating** refit.

### The Central Asymmetry: Why Step 1 Succeeds and Step 2 Crashes

Refit₀ and refit₁ call the **same code path**: `offload_before_refit()` →
`broadcast_weights_for_collective()` → `prepare_for_training()` (lines 1513-1520 in
`grpo.py`). Mechanically identical. The difference is the **state** entering refit₁
vs refit₀:

| State Element | At Refit₀ (setup) | At Refit₁ (post-Step 1) |
|---|---|---|
| Optimizer state (exp_avg, exp_avg_sq) | **Empty** — no `.step()` yet | Populated from Step 1's training |
| Param remainders (precision-aware) | Not yet materialized | Populated from Step 1's `optimizer.step()` |
| Grad buffers / main_grad registrations | Never allocated on CUDA | Allocated by DDP during train₁'s backward |
| Autograd graph remnants | None | Potentially retained from train₁ |
| Mamba workspace tensors | Not yet allocated | Lazily allocated during train₁'s forward |

This is the key to the entire crash. At Refit₀, `offload_before_refit()` calls
`move_optimizer("cpu")`, but the optimizer's `state` dict is **empty** (no
`optimizer.step()` has run yet), so the move is a no-op. Everything stays on CUDA,
and Step 1 trains normally. At Refit₁, the optimizer has populated momentum states
from Step 1, and grad buffers are allocated. The offload actually moves state to CPU
and frees buffers — but the subsequent `prepare_for_training()` fails to fully restore
them. Step 2 then runs against corrupted state.

## Crash Details

The error surfaces during a trivial tensor allocation inside
`logical_and_across_model_parallel_group` (called after `optimizer.step()`):

```python
# 3rdparty/.../train_utils.py:318
input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
```

Called from `MegatronPolicyWorker.train()` (line 407 in
`megatron_policy_worker.py`). Because CUDA errors are **asynchronous**, this is not the
actual corruption site — the real illegal access happened during an earlier kernel, and
the error was only detected when this allocation forced a sync.

The NCCL watchdog then cascades the error across all process groups
(`EXPERT_DATA_PARALLEL_GROUP`, `DATA_PARALLEL_GROUP_WITH_CP`). These NCCL failures are
**downstream symptoms** — once the CUDA context is poisoned, all subsequent NCCL
operations fail.

## Root Cause Analysis

### Root Cause: Unnecessary Offloading on Dedicated Training Nodes

The crash is caused by `offload_before_refit()` and `prepare_for_training()` being
called in the non-colocated (async) refit path, where they are **completely unnecessary
and harmful**.

**Why offloading exists:** In **colocated (sync) GRPO**, the training model and
optimizer must be offloaded to CPU because the GPU is shared with the generation engine
(vLLM/SGLang). The generation engine needs maximum GPU memory for model execution and
KV caching during rollout generation.

**Why offloading is wrong in async:** In **non-colocated (async) GRPO**, the training
nodes are **fully dedicated** to training. Generation takes place on separate nodes.
During the weight refit/broadcast phase, there is **zero memory pressure** on the
training GPUs — nothing else runs on them. The broadcast operation
(`broadcast_weights_for_collective`, line 1134-1144) only reads model parameters
(via `_iter_params_with_optional_kv_scales`). It has no dependency on optimizer or grad
buffer state.

Yet the non-colocated refit path (`grpo.py:1513-1520`) calls `offload_before_refit()`
anyway, which moves grads and optimizer state to CPU, frees CUDA memory, and then calls
`prepare_for_training()` — which fails to fully restore the state due to the guard
condition bugs described below.

### Crash Mechanism: Three Converging Bugs

All three bugs are consequences of the unnecessary offload cycle. Removing the offload
eliminates all three.

#### Bug 1: Optimizer State Device Mismatch

`offload_before_refit()` **unconditionally** moves the optimizer to CPU
(`megatron_policy_worker.py:1203-1208`):

```python
if hasattr(self, "optimizer") and self.optimizer is not None and not self.optimizer_cpu_offload:
    self.move_optimizer("cpu")
```

But `prepare_for_training()` only moves it back **conditionally**
(`megatron_policy_worker.py:1177-1183`):

```python
if (
    hasattr(self, "optimizer") and self.optimizer is not None
    and not self.optimizer_cpu_offload
    and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
):
    self.move_optimizer("cuda")
```

In our config (`offload_optimizer_for_logprob=False`, `is_generation_colocated=False`),
the guard is `False`. The optimizer's momentum tensors (`exp_avg`, `exp_avg_sq`) are
stranded on CPU while model parameters and gradients are on CUDA.

Megatron's precision-aware optimizer uses custom fused CUDA kernels (multi-tensor Adam)
that access memory addresses via `.data_ptr()`. These kernels do not perform high-level
PyTorch device checks. Passing a CPU host pointer to a CUDA kernel as a global memory
address triggers an **instant illegal memory access**. Because CUDA execution is
asynchronous, this failure is not reported immediately but is caught at the next
synchronization point (`logical_and_across_model_parallel_group`).

**Why Step 1 succeeds:** At Refit₀, the optimizer `state` dict is empty — no
`optimizer.step()` has run. `move_optimizer("cpu")` iterates an empty dict and does
nothing. Step 1's `optimizer.step()` lazily initializes state directly on CUDA. At
Refit₁, the state is now populated, and the move to CPU is effective. But
`prepare_for_training()` never moves it back. Step 2 crashes.

#### Bug 2: Grad Buffer Pointer Staleness in DistributedOptimizer

`offload_before_refit()` calls `move_model("cpu", move_params=False, move_grads=True)`
(line 1199), which frees the CUDA grad buffer storage. After `torch.cuda.empty_cache()`
(line 1211), the underlying CUDA memory is released. `prepare_for_training()` calls
`move_model("cuda", move_grads=True, move_params=True)` (line 1170), which allocates
**new** CUDA grad buffers — guaranteed to be at different addresses after the cache
flush.

Megatron's `DistributedOptimizer` builds `param_to_bucket_views` and per-bucket
`main_grad` views once during initialization. These are persistent pointer views mapped
to exact CUDA memory addresses. Since `prepare_for_training()` does not rebuild or
re-register these views, the optimizer's internal pointers reference freed memory.

**Why Step 1 succeeds:** At Refit₀, grad buffers have never been CUDA-allocated
(no backward has run). `move_model("cpu", move_grads=True)` has nothing to free.
`prepare_for_training()` allocates them fresh, consistent with the optimizer's initial
views. At Refit₁, the offload frees Step 1's grad buffers and
`prepare_for_training()` re-allocates at new addresses. The optimizer's pre-existing
views still reference Step 1's addresses → Step 2's backward writes to freed memory.

#### Bug 3: Mamba Training Workspace Staleness

NemotronH's Mamba layers lazily allocate workspace tensors for `selective_scan_fn` and
`causal_conv1d` during the first forward pass and cache them as module attributes. The
`train()` method resets *inference* state but does **not** reset training workspace
tensors.

After the `move_model("cpu") → move_model("cuda")` cycle in Refit₁, these workspace
pointers may reference stale CUDA addresses. Train₂'s Mamba forward kernels then
access freed memory. At Refit₀, no workspaces exist yet (no forward has run), so
this is harmless.

### Secondary Bug: Incomplete `move_optimizer` Implementation

The custom `move_optimizer` in `megatron_policy_worker.py` (line 1290-1311) only moves
optimizer `state` dict values (momentum, variance):

```python
for _, state in optimizer_state.items():
    for k, v in state.items():
        if torch.is_tensor(v):
            if device == "cpu":
                if v.is_cuda:
                    state[k] = v.to("cpu")
            elif device == "cuda":
                if not v.is_cuda:
                    state[k] = v.to("cuda")
```

It completely ignores `param_groups['params']` — the FP32 master weight copies in
`DistributedOptimizer`. Megatron-LM's native `offload_to_cpu()` (in
`megatron/core/optimizer/optimizer.py:339-355`) handles **both**:

```python
for param_group in self.optimizer.param_groups:
    for p in param_group['params']:
        if isinstance(p, torch.Tensor) and p.is_cuda:
            p.data = p.data.cpu()

for state_dict in self.optimizer.state.values():
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            state_dict[k] = v.cpu()
```

Even if `move_optimizer` were correctly called in both directions, it would leave FP32
master weights in a split-device state. This is an independent correctness issue that
should be fixed regardless.

### Why Sync GRPO Does Not Crash

In sync (colocated) mode, the refit lifecycle is:

```
offload_before_refit → IPC weight transfer → offload_after_refit → generation → prepare_for_training
```

`offload_after_refit()` (`megatron_policy_worker.py:1222-1239`) is **more aggressive**:
it moves the **entire model** to CPU (`move_model("cpu")` with default args moves both
params and grads), sets eval mode, and then calls `offload_before_refit()` again
(double cleanup). This means that by the time `prepare_for_training()` runs in the sync
path, **everything** is on CPU — so the full `move_model("cuda")` and
`move_optimizer("cuda")` sequence runs (the `is_generation_colocated` guard is `True`),
performing a complete top-to-bottom GPU reload that rebuilds all state fresh.

In async, `offload_after_refit()` is never called (it's gated on `colocated_inference`
at `grpo.py:1532`). `prepare_for_training()` tries to do a partial reload, exposing
all three bugs.

## Recommended Fix

### Primary Fix: Bypass Offloading in Non-Colocated Mode

In `refit_policy_generation()` (`grpo.py:1506-1520`), the non-colocated branch
currently does:

```python
policy.offload_before_refit()
futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
futures_inference = policy_generation.update_weights_from_collective()
ray.get(futures_train)
results = ray.get(futures_inference)
update_success = all(result for result in results if result is not None)
policy.prepare_for_training()
```

Replace with:

```python
futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
futures_inference = policy_generation.update_weights_from_collective()
ray.get(futures_train)
results = ray.get(futures_inference)
update_success = all(result for result in results if result is not None)
```

This removes both `policy.offload_before_refit()` and `policy.prepare_for_training()`.

**Why this works:** The broadcast only reads model parameters (which remain on CUDA).
The optimizer, grad buffers, and Mamba workspaces are never touched. All CUDA memory
addresses remain stable. No pointers are invalidated.

**Performance benefit:** This also eliminates the PCIe overhead of offloading and
onloading tens of gigabytes of optimizer state (FP32 master weights, momentum, variance)
that was pure waste on dedicated training nodes.

### Secondary Fix: Repair `move_optimizer`

Regardless of the primary fix, `move_optimizer` should be updated to handle
`param_groups['params']` for correctness. This prevents future bugs if `move_optimizer`
is called from other code paths.

### Verification Plan

1. **State dump (pre-fix verification):** Add logging at the top of
   `MegatronPolicyWorker.train()` to print optimizer state device and grad buffer
   pointers at Step 1 and Step 2 entry. Run one async config (e.g., 3g1t). This
   confirms the device mismatch (Bug 1) and pointer staleness (Bug 2) before applying
   the fix.

2. **Apply primary fix:** Remove `offload_before_refit()` and
   `prepare_for_training()` from the non-colocated branch.

3. **Run all three async configs** (1g3t, 2g2t, 3g1t). Verify all train past Step 2.

## Supporting Observations

### AccumulateGrad Warning (Present in Both Sync and Async — Not Diagnostic)

Both the async crash run (11978373, line 8438) and the sync success run (11978368,
line 2796) emit the same `AccumulateGrad` stream mismatch warning on all workers. Since
the sync run trains fine with this warning, **it is decoupled from the crash** and is a
benign artifact of Megatron's DDP/pipeline-parallel setup.

### GPU Memory Growth Between Refits

| Event | Allocated | Reserved | Log Line |
|---|---|---|---|
| Before refit₀ (setup) | 17.44 GB | 19.58 GB | 7166 |
| After refit₀ (setup) | 8.73 GB | 8.73 GB | 7195 |
| Before refit₁ (post-Step 1) | 32.90 GB | 33.18 GB | 8495 |
| After refit₁ (post-Step 1) | 9.09 GB | 13.21 GB | 8496 |

The growth from 17.44 to 32.90 GB is **expected**: refit₀ happens during setup before
training materializes grad buffers, DDP state, and activation memory. After Step 1,
these persistent buffers exist.

### Consistent Across DP Sizes

The crash occurs at DP=1, 2, and 3. This rules out cross-rank DP communication bugs
as the primary cause.

## Lower-Likelihood Hypotheses (Ruled Out or Subsumed)

These hypotheses from the original investigation are no longer considered primary
candidates. They are either subsumed by the root cause above or have been ruled out.

- **Data-dependent crash (H4):** The crash occurs at the same lifecycle point across
  3 configs with different DP sizes and buffer fill patterns. Subsumed: the lifecycle
  corruption is deterministic regardless of batch content.
- **Broadcast stream interaction (H6):** Possible but secondary. Removing the offload
  cycle removes the stream ordering ambiguity. Can add `torch.cuda.synchronize()` after
  `ray.get()` calls if needed.
- **GPU memory pressure (H7):** The crash occurs at DP=1 with ~33 GB on 80 GB H100s.
  Not compelling.
- **Ray ObjectRef lifetime (H8):** No evidence of premature deallocation. The crash
  pattern (deterministic at Step 2, independent of batch) doesn't fit a refcount race.

## Related Jobs and Log Locations

| Job ID   | Config        | Status    | Log Path |
|----------|---------------|-----------|----------|
| 11978373 | async 1g3t    | CRASHED   | `11978373-logs/ray-driver.log` |
| 11978375 | async 2g2t    | CRASHED   | `11978375-logs/ray-driver.log` |
| 11978429 | async 3g1t    | CRASHED   | `11978429-logs/ray-driver.log` |
| 11978368 | sync 4n       | RUNNING   | `11978368-logs/ray-driver.log` |
| 11977729 | async 1g3t    | CRASHED   | `11977729-logs/ray-driver.log` |

All log directories are under
`/lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl-super-vllm0.20/`.

### Key Log Lines for 11978373 (Representative Crash)

- **Initial refit₀ success**: line 7196
- **Memory before refit₀**: line 7166 (`17.44GB allocated`)
- **Step 1 header**: line 8158
- **Step 1 train start**: line 8404
- **AccumulateGrad warning**: line 8438 (also fires in sync run at line 2796 — not
  diagnostic)
- **Step 1 train end**: line 8468 (`elapsed=100.13s`)
- **Post-Step 1 refit₁**: line 8489
- **Memory before refit₁**: line 8495 (`32.90GB allocated`)
- **Step 1 total_step_time end**: line 8498 (`elapsed=210.52s`)
- **Step 2 header**: line 8638
- **Step 2 train start**: line 8854
- **policy_training end**: line 9491 (timer exits via context manager before error
  propagates)
- **total_step_time end**: line 9492
- **Traceback**: line 9493 (`policy.train()` at `grpo.py:3884`)
- **First CUDA error**: line 9529
- **Error in async loop**: line 10819

### Calibration Data

- AccumulateGrad warning: fires in **both** sync (11978368, line 2796) and async
  (11978373, line 8438). Sync trains fine → warning is decoupled from crash.
- CUDA graphs: **disabled** for Megatron training (`enable_cuda_graph=False` in
  `DistributedDataParallelConfig`, line 1950).
- KL penalty: **0.0** → no reference model swap occurs.
