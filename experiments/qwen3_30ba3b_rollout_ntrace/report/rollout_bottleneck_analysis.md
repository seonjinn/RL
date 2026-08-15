# Where Qwen3-30B-A3B MXFP8 rollout spends GPU time

## Scope

This ntrace capture profiles Qwen3-30B-A3B generation in NeMo-RL on GB200.
The generation pool uses eight independent TP1/PP1/EP1 vLLM workers, CUDA
Graphs, vLLM 0.25.1, FlashInfer 0.6.13, and the `flashinfer_trtllm` MoE
backend. MXFP8 applies only to the routed expert FC1 and FC2 weights. Attention,
the MLP gate, and `lm_head` remain BF16. The run records one warm-up rollout and
three measured rollouts on ranks 0--7.

The optimizer learning rate is zero. This keeps the policy weights fixed across
the captured rollouts. The trace measures the synchronous GRPO rollout phase;
it does not measure async 1-off pipeline overlap.

## Validation

- All eight ranks contain records, stacks, graph-node metadata, and source
  manifests.
- Each rank contains about 17.3 million measured replay rows and 28.8K--30.2K
  unique replay nodes.
- Every replay node maps to a captured CUDA Graph clone. Uncovered nodes: zero.
- The breakdown keeps all three explicit iteration windows and reports the
  later two separately from the first-window stall.
- UTC-aligned multirank analysis detects no communication kernels, no
  straggler, 0.01% mean jitter, and 0.04% p90 jitter. The worst iteration has
  only 73 ms of rank spread.
- Each rank has 17.4K--23.6K non-graph rows without an iteration-0 stack match.
  This is about 0.1% of active events and does not affect graph replay coverage.
- The capture finished all four GRPO steps and saved the trace before a
  post-save Ray finalizer error. The finalizer error did not modify the saved
  artifacts.

## Main result

The three windows contain two different regimes. The first traced window lasts
205.06 seconds and contains 133.63 seconds of GPU idle time (65.2%). The next
two windows average 77.51 seconds, with 73.15 seconds active and 4.35 seconds
idle (5.6%). The all-window 39.5% idle average is therefore not a representative
steady-state value. It is dominated by a first-window stall that needs separate
request-orchestration and profiler-boundary analysis.

Across ranks, active time varies by only 2.7% from minimum to maximum. Worker
compute skew and NCCL communication are therefore not the source of the
first-window stall.

The stack-attributed MoE path averages 34.04 seconds in the later two windows.
This is 46.5% of steady-state active GPU time. Raw kernel-name analysis splits
the expert BMM sample into 12.29 seconds for FC1/SwiGLU kernels and 7.07 seconds
for FC2 kernels. MoE routing and finalize kernels add 3.52 seconds, and MXFP8
quantization/layout kernels add 0.63 seconds. Attention kernels account for
16.95 seconds in the same additive raw-kernel view.

| Quantity | Rank mean | Range |
|---|---:|---:|
| First traced window | 205.06 s | 133.63 s idle (65.2%) |
| Later two windows | 77.51 s | 4.35 s idle (5.6%) |
| Later-window GPU active | 73.15 s | 71.47--74.83 s by iteration mean |
| Later-window MoE | 34.04 s | 46.5% of active time |
| Raw expert FC1 BMM | 12.29 s | additive kernel time |
| Raw expert FC2 BMM | 7.07 s | additive kernel time |
| Raw attention | 16.95 s | additive kernel time |

The built-in ntrace classifier labels 86% of active time as `Other` because it
does not yet recognize the FlashInfer TRTLLM `bmm_*` and `fmhaSm100f*` names.
The report therefore uses Python stack paths for conserved wall-time
attribution and raw kernel names only for the diagnostic FC1/FC2 and attention
split. These two views must not be added together.

## Optimization limit

For the later two windows, if MoE execution becomes twice as fast and all
non-MoE and idle time stays constant, Amdahl's law gives an upper-bound speedup
of 1.28x. If MoE time could be removed entirely, the bound is 1.78x. When the
first-window stall is included, these bounds fall to 1.16x and 1.39x.

This points to two independent targets. First, tune expert FC1/FC2 BMM tactics
and routing for steady-state generation. Second, explain the 133.63-second
first-window idle interval by aligning request, scheduler, and profiler
timestamps. It may be rollout orchestration, request availability, or a capture
boundary effect; the GPU trace alone cannot choose among them.
Dense-linear adaptive selection alone does not optimize this recipe because
QKVO and other dense projections stay BF16; the quantized work runs through the
MoE grouped BMM backend.

## Control-run limitation

The matched BF16 trace did not reach generation. Qwen's sparse refit path tried
to copy a 2D expert W2 tensor into vLLM's 3D expert destination and failed with
a `[2048, 768]` to `[12, 768, 64]` shape mismatch. This is a refit layout support
gap, not a BF16 kernel or ntrace failure. This report therefore identifies the
MXFP8 rollout bottleneck but does not claim a matched BF16-versus-MXFP8 speedup.
A generation-only BF16 control or a Qwen-compatible sparse-refit conversion is
required for that comparison.

## Provenance

- NeMo-RL experiment snapshot: `4048c388`
- ntrace source: `bed58828e16abb47032d5fa3fd3d81c371dafc5a`
- Analysis workflow: `5e06de216`
- Data: `data/rollout_bottleneck_summary.json` and
  `data/rollout_bottleneck_summary.tsv`
- Native multirank report: `ntrace_multirank.html`
