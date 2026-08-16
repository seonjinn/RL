# Where MoE-only MXFP8 saves rollout time

## Method

This matched ntrace experiment compares BF16 and MoE-only MXFP8 rollout for
Qwen3-30B-A3B on GB200. Both arms use eight independent TP1/PP1/EP1 vLLM
workers, CUDA Graphs, vLLM 0.25.1, FlashInfer 0.6.13, the
`flashinfer_trtllm` MoE backend, and NCCL Reshard refit. MXFP8 applies only to
routed expert FC1 and FC2. Attention, the router gate, and `lm_head` stay BF16.

The trace records GRPO steps 2--4. Step 2 includes Python stack-capture cost,
so the performance comparison uses steady steps 3 and 4. The two arms generated
1.62275M and 1.62221M output tokens per rank in those windows, a difference of
only 0.033%. Every time below is normalized by generated tokens.

## Result

| Metric, steady steps 3--4 | BF16 | MoE-only MXFP8 | Change |
|---|---:|---:|---:|
| Throughput | 8,199 tok/s/GPU | 10,502 tok/s/GPU | **1.281x** |
| Critical rollout wall time | 122.0 s/Mtoken | 95.2 s/Mtoken | **-21.9%** |
| Active GPU time | 117.0 s/Mtoken | 89.7 s/Mtoken | **-23.4%** |
| Stack-attributed MoE time | 68.6 s/Mtoken | 41.7 s/Mtoken | **-39.3%** |
| GPU idle time | 5.0 s/Mtoken | 5.5 s/Mtoken | +0.6 s/Mtoken |

The raw kernel-name view explains the MoE reduction. Expert FC1/SwiGLU BMM
($K=2048$, $N=1536$) falls from 27.77 to 15.34 s/Mtoken (-44.8%), and FC2
($K=768$, $N=2048$) falls from 14.78 to 8.67 s/Mtoken (-41.3%). The routed
$M$ varies by expert and scheduler step. Routing and finalize do not improve:
they rise from 4.18 to 4.37 s/Mtoken (+4.8%). Attention stays flat at about
21.3 s/Mtoken, which is expected because attention remains BF16. MXFP8
quantization and layout add 0.78 s/Mtoken.

The stack view is a conserved wall-time attribution. The raw kernel view is an
additive diagnostic averaged across all captured windows; the two views must
not be added together.

## What to optimize next

1. **Expert grouped BMM remains the first steady-state target.** MXFP8 cuts MoE
   time substantially, but the MoE path still owns 46.5% of MXFP8 active GPU
   time. FC1 is the larger remaining expert kernel family. Record the routed
   $M$-bucket frequency and tune the common buckets first. If the remaining MoE
   time were halved with all other work fixed, this profile bounds the
   additional throughput gain at about 1.28x.
2. **Tune or fuse routing and finalize.** This work is unchanged by weight
   precision and regresses 4.8% per token in the matched run.
3. **Remove cold-start tuning and JIT from production startup.** FlashInfer
   profiled 21 MoE inputs on every worker and took about 18 minutes before CUDA
   Graph capture. The run also reported an uncovered
   `MXFP8QuantizeLinearKernel` JIT during inference. A versioned, shared
   autotune cache plus workload-shape warmup should move both costs out of the
   rollout path.
4. **Treat attention as the next non-MoE floor.** Its 21.3 s/Mtoken does not
   change between arms and becomes a larger share after expert acceleration.

All eight ranks saved records, Python stacks, CUDA Graph nodes, and source
manifests. Native multirank analysis reports mean iteration jitter of 0.005% for
BF16 and 0.002% for MXFP8, with no single straggler in either arm. No
communication kernel appears because the eight rollout workers run as
independent TP1 workers. Four GRPO steps completed with finite generation KL
error: BF16 reported 0.0018--0.0020 and MXFP8 reported 0.0038--0.0042. This short run
verifies functional execution and comparable token volume. The 1.281x speedup
is the measurement from this run; no run-to-run uncertainty was collected. The
higher MXFP8 KL range requires a longer convergence and accuracy study before
making an accuracy claim. SLURM reported failure
only after trace save because Ray initialized a core worker twice during Python
finalization. The saved artifacts passed the rank 0--7 file gate.

Provenance: NeMo-RL experiment snapshot `3837bff80`, analysis workflow
`48f2c4057`, ntrace `92a94b8`, vLLM 0.25.1, FlashInfer 0.6.13. Canonical data:
`data/matched_comparison.json`, `data/matched_bf16/`, and
`data/matched_mxfp8/`.
