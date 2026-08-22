# MXFP8 Sync Refit Optimization Findings

## Scope

This investigation targets synchronous colocated CUDA IPC refit for
Qwen3-30B-A3B with BF16 training and MXFP8 rollout on GB200. The reference
comparison uses steady-state means from steps 3--20:

| Metric | BF16 rollout | MXFP8 rollout |
| --- | ---: | ---: |
| Total refit | 4.62 s | 7.55 s |
| Transfer and update | 1.74 s | 4.49 s |

The MXFP8 run already includes trainer-side prequantization, persistent IPC
buffers, slim offload, batched expert shuffle, loader-route caching, a 4 GiB
refit buffer, FlashInfer TRTLLM MoE kernels, and CUDA Graph execution.

The historical sub-second MXFP8 result used disaggregated NCCL Reshard. It is
not a directly comparable result for this colocated CUDA IPC path.

## Existing Optimization Value

An earlier controlled ablation measured the following total-refit changes:

| Change | Total refit | Incremental change |
| --- | ---: | ---: |
| Baseline | 15.47 s | -- |
| Trainer prequantization | 13.54 s | -1.93 s |
| Persistent IPC buffers and slim offload | 13.48 s | -0.06 s |
| Batched MoE shuffle | 8.46 s | -5.02 s |
| Loader-route cache | 8.39 s | -0.07 s |

The next useful work should therefore reduce layout-conversion work or IPC
synchronization. More Python-side lookup caching is unlikely to produce a
large gain.

## Candidate Optimizations

The candidates below are ordered by expected value before phase-level
profiling. Job `6455271` measures each phase with NVTX and Nsight Systems.

1. **Fuse the expert layout conversion.** The current batched path still uses
   separate weight gather, scale gather, scale interleave, W13-to-W31 reorder,
   and final destination copies. A fused kernel that writes the TRTLLM runtime
   layout directly can remove several full expert-weight memory passes.
2. **Reduce IPC synchronization frequency.** The reference run packs about 16
   groups. Each group introduces a source stream fence, receiver stream fence,
   and ACK dependency. A larger safe bucket or CUDA IPC event protocol can
   reduce host-visible serialization.
3. **Keep row permutations on the GPU.** The current cache stores CPU
   permutations on each layer and copies them to the GPU during every refit.
   A process-wide cache keyed by shape, device, tile, and gated layout can
   remove repeated allocation and host-to-device copies.
4. **Remove unconditional receiver cleanup.** The receiver runs
   `gc.collect()` and `torch.cuda.empty_cache()` after refit. This should be
   skipped on the persistent-buffer path if a memory-stability test shows that
   it is unnecessary.
5. **Limit finalization to updated modules.** The receiver currently runs a
   full-model `process_weights_after_loading` pass. Tracking dirty MXFP8
   modules could avoid unrelated traversal, although the small loader-cache
   ablation suggests a lower ceiling than the first four candidates.
6. **Emit prepared TRTLLM weights from the trainer.** This is the largest
   architectural change. It would quantize and arrange complete expert stacks
   in the final TRTLLM layout before transfer, leaving the receiver with a
   direct install. It needs a prepared-weight manifest and backend-specific
   ownership rules.

FlashInfer's MXFP8 GEMM handoff benchmark APIs consume already prepared dense
weights. They do not convert checkpoint-layout expert weights during refit, so
they do not directly solve this bottleneck.

## Measurement Gate

Implementation should follow the measured phase ranking. Every candidate must
pass a matched 20-step run, output-token and `gen_kl_error` checks, CUDA Graph
execution, and a memory-stability run before it replaces the current path.
