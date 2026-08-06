# Qwen3-30B-A3B 32K Rollout Result

## Scope

This experiment compares the stock FlashInfer CuTeDSL MXFP8 linear path with
the corrected FlashInfer TRTLLM adaptive path in a 20-step NeMo-RL run. The
workload permits up to 32,768 output tokens per rollout and keeps CUDA Graphs
enabled.

## Configuration

- Cluster: OCI-HSG GB200
- Model: `Qwen/Qwen3-30B-A3B`
- Training steps: 20
- Prompts per step: 48
- Generations per prompt: 4
- Rollouts per step: 192
- Maximum input length: 2,048 tokens
- Maximum output length: 32,768 tokens
- Maximum total sequence length: 34,816 tokens
- vLLM GPU memory utilization: 0.5
- CUDA Graphs: enabled (`enforce_eager=false`)
- MoE backend: `flashinfer_trtllm`
- Dense MXFP8 backends: `flashinfer_cutedsl` and
  `flashinfer_trtllm_adaptive`
- Jobs: CuTeDSL `5906397`; TRTLLM Adaptive `5906398`

Both jobs completed all 20 steps with exit code `0:0`. No CUDA OOM or Python
traceback was observed. Metrics below use paired steady-state Steps 3-20.

## Results

| Metric | CuTeDSL | TRTLLM Adaptive | Adaptive / CuTeDSL |
|---|---:|---:|---:|
| Mean generation length (tokens) | 26,635.29 | 26,635.29 | 1.0000x |
| Generation throughput (tokens/s/GPU) | 1,443.43 | 1,430.14 | 0.9908x |
| Generation time (s, lower is better) | 222.38 | 224.43 | 0.9909x speed |
| End-to-end throughput (tokens/s/GPU) | 903.73 | 895.75 | 0.9912x |

The TRTLLM adaptive path is about 0.9% slower than CuTeDSL in this matched
long-context workload. The result does not support enabling the adaptive path
for this Qwen3-30B-A3B configuration. CuTeDSL remains the preferred default.

## DAPO Interpretation

DAPO is an RL algorithm and recipe family, not a single fixed 32K context
configuration. This run uses a DAPO-like long-rollout operating point by
explicitly setting the output cap to 32K. The realized mean output length is
26.6K tokens, so the result represents genuinely long generations rather than
only a large unused limit.

## Reproduction

```bash
MAX_STEPS=20 \
MAX_NEW_TOKENS=32768 \
MAX_INPUT_SEQ_LENGTH=2048 \
MAX_TOTAL_SEQUENCE_LENGTH=34816 \
GPU_MEMORY_UTILIZATION=0.5 \
experiments/qwen30b_mxfp8_linear_backends/submit_long32k_ptyche.sh
```

The plotting inputs are generated with `summarize_results.py --first-step 3`.
The 600 DPI PNG and vector PDF figures are produced by `plot_results.py`.
