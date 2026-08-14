# MXFP8 Linear Backend A/B on GB200

This experiment keeps BF16 training, the model, data, seed, topology, MXFP8
scope, and refit settings fixed. Only vLLM's dense MXFP8 `linear_backend`
changes between arms.

## Matrix

| Model | Backends | Allocation | MXFP8 scope |
|---|---|---|---|
| Qwen3-30B-A3B | CUTLASS, CuTeDSL, TRTLLM | 4 x 4 GB200 GPUs | MoE experts plus Q/K/V/O |
| Nemotron 3 Nano | CUTLASS, CuTeDSL | 4 x 4 GB200 GPUs | MoE experts plus Q/K/V/O |

The vocabulary head and replicated MoE router remain BF16. Nano's dense
hidden dimension is 2688, which does not satisfy the TRTLLM kernel's `K % 256`
contract. The submit script rejects that arm instead of silently benchmarking
a CuTeDSL fallback.

## Run

Validate scheduling first:

```bash
MODEL=qwen30b LINEAR_BACKEND=flashinfer_cutlass ACTION=test-only \
  experiments/mxfp8_linear_backend_ab/submit_lyris.sh
```

Use `ACTION=submit MAX_STEPS=2` for smoke tests and `MAX_STEPS=20` for the
measurement. Report steady-state means over steps 3-20 for generation, refit,
logprob, policy training, E2E step time, and tokens/s/GPU.

Pinned sources:

- NeMo-RL experiment branch: `sna/linear-backend-gb200-ab-20260813`
- vLLM: `a76062edee3a3ac23d47a93c7ce466f06a19111f`
- vLLM base: `v0.25.1`
