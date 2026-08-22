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

The next useful work should reduce the number of per-expert quantization,
packing, and receiver-load operations. More Python-side lookup caching is
unlikely to produce a large gain.

## Measured Phase Breakdown

OCI-HSG job `6456388` profiled step 2 with NVTX and Nsight Systems across all
16 policy and 16 generation ranks. The table reports the median accumulated
time per rank; the ranges across ranks were narrow enough to support the same
ranking on every rank. Nsight Systems inflates the absolute refit time, so use
these values to rank phases rather than to predict the exact unprofiled time.

| Side | Phase | Median time/rank | Calls/rank |
| --- | --- | ---: | ---: |
| Policy | Full IPC stream | 7.149 s | 1 |
| Policy | MXFP8 quantization | 2.984 s | 18,432 |
| Policy | Pack into staging buffer | 1.799 s | 37,299 |
| Policy | ACK waits and stream fences | 0.385 s | -- |
| Receiver | Full weight update | 7.164 s | 1 |
| Receiver | Load staged buckets | 4.133 s | 16 |
| Receiver | Finalize model | 0.155 s | 1 |
| Receiver | Batched TRTLLM expert shuffle | 0.057 s | 48 |
| Receiver | Open IPC handles | 0.020 s | 16 |
| Receiver | Cleanup | 0.012 s | 1 |

The existing batched TRTLLM shuffle is not the remaining bottleneck. It takes
only 57 ms per rank. The dominant cost comes earlier: the source quantizes each
expert tensor separately, packs tens of thousands of small tensors, and the
receiver routes those tensors through the normal per-parameter loader.

## Recommended Optimization Order

1. **Group experts before quantization and export.** Build each layer's expert
   matrices as grouped tensors, flatten `[E, M, K]` to `[E*M, K]`, and quantize
   once per grouped matrix. This can reduce 18,432 quantization calls toward a
   few grouped calls per layer.
2. **Load grouped payloads directly.** Transfer one grouped weight and scale
   payload per layer and matrix family, then copy directly into the stacked
   checkpoint-layout destination. This removes most of the 37,299 small pack
   and receiver-load operations.
3. **Quantize into persistent grouped staging buffers.** After the grouped path
   is correct, write quantized values and scales into reusable transfer buffers
   to remove the remaining source packing copy.
4. **Keep the existing batched TRTLLM post-load shuffle.** Its measured 57 ms
   cost is small. A fused final-layout producer is a later option, not the
   first target.
5. **Treat IPC fences and cleanup as low-ceiling work.** ACK waits and fences
   total less than 0.4 s, while cleanup takes about 12 ms. Larger buckets or
   CUDA IPC events may help after the grouped data path is complete.

FlashInfer's MXFP8 GEMM handoff benchmark APIs consume already prepared dense
weights. They do not convert checkpoint-layout expert weights during refit, so
they do not directly solve this bottleneck.

Qwen3-30B-A3B also has a matched 20-step quantization-scope comparison. Adding
QKVO projections to the MoE-only scope changes E2E throughput by only +0.33%,
reduces generation throughput by 3.13%, and increases refit time by 8.69%.
This result supports optimizing the grouped MoE path first rather than widening
the default quantization scope.

## Measurement Gate

Implementation should follow the measured phase ranking. The first patch should
group expert quantization, staging, and receiver loading without changing the
runtime TRTLLM layout contract. It must pass a matched 20-step run, output-token
and `gen_kl_error` checks, CUDA Graph execution, and a memory-stability run
before it replaces the current path.
