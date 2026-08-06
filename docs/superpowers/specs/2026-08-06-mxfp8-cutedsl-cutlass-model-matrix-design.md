# MXFP8 CuTeDSL versus CUTLASS Model Matrix

## Objective

Determine whether FlashInfer CuTeDSL is consistently faster than FlashInfer
CUTLASS for dense MXFP8 linear layers during CUDA Graph NeMo-RL rollouts.
Evaluate Qwen3-30B-A3B, Qwen3-235B-A22B, and Nemotron3 Super without changing
the MoE backend or quantization scope between arms.

## Experimental Variable

Change only `policy.generation.vllm_kwargs.linear_backend`:

- `flashinfer_cutlass`
- `flashinfer_cutedsl`

Keep `moe_backend=flashinfer_trtllm`. Enable CUDA Graph execution with
`enforce_eager=false`. Quantize dense projections, including Q/K/V/O, and MoE
weights to MXFP8; leave `lm_head` and the router gate unquantized.

## Model Matrix

| Model | Recipe/topology | Sequence cap | Rollouts per step |
|---|---|---:|---:|
| Qwen3-30B-A3B | 4 nodes x 4 GB200, rollout TP1 | 4,096 | 2,048 |
| Qwen3-235B-A22B | 16 nodes x 4 GB200, rollout TP4 | 8,192 | 512 |
| Nemotron3 Super | 32 nodes x 4 GB200, rollout TP4 | 8,192 | 256 |

Use each model's shipped NeMo-RL performance recipe. Backend comparison rows
must use the same model, seed, generation policy, node topology, container,
vLLM commit, FlashInfer version, and physical quantization scope.

## Execution Gates

1. Run scheduler validation for both arms.
2. Run a two-step smoke for initialization, refit, CUDA Graph capture/replay,
   token validity, NCCL, and memory headroom.
3. Submit an eight-step measurement only after both smoke arms pass.
4. Report steady-state steps 3 through 8.

Do not introduce a dependency between the two backend arms. They must be
independently schedulable, and the final comparison must record physical node
allocations because a one-trial node difference can be comparable to a small
backend delta.

## Metrics

Primary metrics:

- generated tokens per second per GPU;
- generation time.

Secondary metrics:

- end-to-end tokens per second per GPU;
- total step time;
- transfer and weight-update/refit time;
- realized mean generation length.

## Validity Criteria

A result is valid only when both arms:

- complete the same measured steps with exit code zero;
- successfully capture and replay CUDA Graphs;
- use identical realized generated-token counts per compared step;
- report the requested dense and MoE backends in runtime logs;
- have no OOM, NCCL, traceback, engine-death, or token-validity failure;
- record NeMo-RL, vLLM, FlashInfer, container, and checkpoint provenance.

The conclusion must be model- and workload-specific unless CuTeDSL wins every
valid model row by more than observed run-to-run variance. A small positive
mean from a single allocation is not sufficient for an "always faster" claim.

## Reporting

Create one report containing raw and normalized generation throughput, E2E
throughput, generation time, and refit time. Preserve the existing Qwen3-30B
result as historical evidence, but rerun it under the same final provenance if
the Qwen3-235B or Super environment uses a different code or container pin.
