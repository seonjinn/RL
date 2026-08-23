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
| `6472933` | First GPU test command | Invalid: vLLM-marked tests were excluded, so zero tests ran |
| `6473033` | Broad GPU test selection | Invalid: eight target tests passed, then a pre-existing base-branch error-message assertion failed |
| `6473130` | Scoped GPU value and storage validation | Completed: 9 passed |
| `6473035` | Matched 20-step end-to-end run | Completed, 20/20 |
| `6473042` | Receiver-only NSys capture after the E2E run | Running on the same nodes |
| `6473651` | Same-node, 20-step current swap-path control | Queued after `6473042` |

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

## Fused Quantize and Shuffle Assessment

A true fused producer would read each BF16 expert block once, quantize it, and
write the weight and E8M0 scale directly in the final TRTLLM row and scale
layout. The current vLLM 0.25.1 and FlashInfer interface does not accept a row
permutation or a final expert-layout destination in `mxfp8_quantize`. Supporting
this therefore needs both a new producer kernel and a prepared-payload contract
that tells vLLM to skip its normal post-load conversion.

The direct-write microbenchmark bounds the extra benefit of that larger change:
it saves about 83 ms per 48-layer refit versus 66 ms for the implemented
composed path. The additional upper-bound gain is therefore only about 17 ms
per refit for this model. A full fused producer is not the next implementation
target unless profiling shows that scale interleave or synchronization hides a
larger cost outside the measured weight kernels.

The validation run confirmed bitwise weight and scale parity for gated,
non-gated, and padded expert shapes. It also confirmed that the live vLLM
parameter objects and storage addresses remain unchanged across refit.

## End-to-End Candidate

Job `6473035` completed all 20 steps. Values below are arithmetic means over
steps 3--20. The comparison column uses the prior clean PR 3294 run; the
same-node current-path control is still pending.

| Metric | Prior PR 3294 | Composed candidate | Change |
|---|---:|---:|---:|
| E2E step | 184.77 s | 189.76 s | +2.70% |
| E2E throughput | 2,240.04 tok/s/GPU | 2,189.72 tok/s/GPU | -2.25% |
| Generation | 51.52 s | 51.59 s | +0.13% |
| Refit total, steps with transfer | 8.392 s | 8.418 s | +0.31% |
| Transfer/update | 4.469 s | 4.425 s | -0.99% |

Candidate step 11 followed a validation boundary and did not run weight
transfer. Its `1.75 s` refit timer is excluded from both refit rows above; all
18 steps remain in the E2E rows. The 44 ms transfer/update reduction agrees
with the isolated kernel projection, but total refit is unchanged and the
reduction is too small to improve E2E throughput. The E2E difference comes
from workload variation outside generation and refit. The candidate's
correctness signals remain in the prior baseline range: `gen_kl_error`
`0.003975`, reward `0.5279`, and loss `0.00103`.

The end-to-end candidate and current-path control both use
`nvl72058-T01`--`nvl72058-T04`, the same container, workload, seed, fixed
4-GiB IPC buffer, CUDA Graph setting, and 20-step aggregation contract. The
control is pinned to PR 3294 head
`88721ced809fddf179b73f77393cd7f7e253bc53`; the candidate changes only the
composed W13 row permutation and its tests. This paired run is required because
the first candidate warm-up step reported a larger refit change than the
isolated kernel timings predict.
