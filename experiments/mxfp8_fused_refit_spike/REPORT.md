# MXFP8 Fused Refit Feasibility

## Question

Can the synchronous BF16-training to MXFP8-rollout refit path reduce its
remaining transfer/update time by combining expert quantization and TRTLLM
layout preparation without duplicating the prepared model?

## Baseline

- Code: PR 3294 head `88721ced809fddf179b73f77393cd7f7e253bc53`
- Hardware: GB200
- Model: Qwen3-30B-A3B
- Refit: synchronous colocated CUDA IPC
- Rollout: MXFP8, FlashInfer TRTLLM MoE, CUDA Graphs enabled
- Existing result: refit `15.47 s -> 8.39 s` from naive to full PR 3294
- Remaining transfer/update time in the optimized run: `4.47 s`

## Candidate Paths

The representative local expert tensors are `E=128`, `I=768`, and `K=2048`.
The combined W13 and W2 MXFP8 payload is 576 MiB per layer and generation
worker.

| Path | Weight operations after quantization | Full-tensor scratch |
|---|---|---:|
| Current | copy to live, W13 swap, gather to scratch, copy back | 576 MiB plus a 384 MiB W13 swap result |
| Composed | copy to live, composed swap+shuffle gather, copy back | 576 MiB |
| Direct | gather from the IPC source into the final live layout | 0 MiB |

The direct path keeps the existing CUDA Graph-visible parameter storage. It
does not keep a second prepared model. Scales still need a small staging path
because TRTLLM consumes the interleaved scale layout.

## Validation Gate

1. Require bitwise equality with the existing weight and scale layout.
2. Keep every runtime parameter data pointer unchanged across refits.
3. Do not add more than 4 GiB of persistent memory per generation worker.
4. Measure the full quantize-plus-layout interval, not only the row gather.
5. Run a matched 20-step end-to-end A/B only if the microbenchmark saves at
   least 10% of layout time or projects to at least 0.5 s per refit.

## Runs

| Job | Purpose | Status |
|---|---|---|
| `6472286` | Five-step steady-state NSys capture, profile step 3 | Invalid: policy workers crashed in `cudaProfilerStart` |
| `6472370` | Initial microbenchmark | Invalid: incomplete container Python environment |
| `6472474` | Isolated vLLM environment attempt | Invalid: missing detached-worktree submodules |
| `6472547` | First complete vLLM environment | Invalid: standalone vLLM circular import |
| `6472706` | Current/composed/direct comparison using the cached environment | Completed |

## Result

The weight-layout microbenchmark passed bitwise parity.

| Measured interval | Current | Composed swap+shuffle | Direct final write |
|---|---:|---:|---:|
| Layout only | 3.197 ms | 1.830 ms (`1.75x`) | 1.472 ms (`2.17x`) |
| Quantize and layout | 3.625 ms | 2.205 ms (`1.64x`) | 1.843 ms (`1.97x`) |

The composed path removes 42.7% of weight-layout latency without changing the
loader or the runtime parameter pointers. The direct path removes 53.9% and
also removes the 576 MiB full-weight scratch buffer, but it needs a new loader
contract between IPC receive and the TRTLLM runtime layout.

For a 48-layer model, the isolated result projects to about 66 ms saved by the
composed path or 83 ms by direct write per refit. That is much smaller than the
remaining 4.47 s transfer/update interval. The composed path is being carried
to a 20-step E2E test because it is low risk. The direct loader change is not
justified unless the receiver-only profile shows a larger hidden cost.
