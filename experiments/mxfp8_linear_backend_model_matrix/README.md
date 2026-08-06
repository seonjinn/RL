# MXFP8 Linear Backend Model Matrix Summary

This tool summarizes matched FlashInfer CUTLASS and CuTeDSL MXFP8 linear
backend runs for the Qwen3-30B-A3B, Qwen3-235B-A22B, and Nemotron3 Super
model matrix. Each input is an explicit model-to-run-root mapping. A run root
must contain one `*-logs/ray-driver.log` file for each backend at
`flashinfer_cutlass/` and `flashinfer_cutedsl/`, plus the `run_manifest.json`
emitted by each launcher in the same backend directory.

Run it after both eight-step measurement arms complete:

```bash
python experiments/mxfp8_linear_backend_model_matrix/summarize_results.py \
  --model-run qwen3-30b=/lustre/.../qwen30b-mxfp8-linear-backends/RUN_ID \
  --model-run qwen3-235b=/lustre/.../qwen235b-mxfp8-linear-backends/RUN_ID \
  --model-run nemotron3-super=/lustre/.../nemotron3-super-mxfp8-linear-backends/RUN_ID \
  --output-dir /lustre/.../mxfp8-linear-backend-model-matrix/RUN_ID
```

By default, the CLI requires exactly the three model identifiers shown above
and rejects missing or unknown labels. `--allow-partial` permits a known subset
for development without allowing misspelled or unknown model names.

The output contains every parsed log block in `step_metrics.csv` and the
steady-state steps 3 through 8 in `summary.json`. The JSON preserves each
validated run manifest alongside the absolute
generation and end-to-end throughput plus generation and end-to-end latency.
Throughput ratios are normalized to CUTLASS; latency values are expressed as
CUTLASS latency divided by the backend latency, so values above one indicate a
faster backend.

The tool rejects a row before writing a result unless exactly one driver log is
present per backend, each manifest declares the expected model and backend,
and all invariant manifest fields match across the two arms. The invariant
fields are the exact NeMo-RL and vLLM commits, container, recipe, CUDA Graph
mode, quantization scope, and MoE backend. Every requested measured step must
also be complete, and paired measured steps must have identical mean generation
lengths. A malformed `Training Results` block is rejected at its original step
ordinal. These checks run before speedups are calculated.
