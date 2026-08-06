# MXFP8 Linear Backend Model Matrix Summary

This tool summarizes matched FlashInfer CUTLASS and CuTeDSL MXFP8 linear
backend runs for the Qwen3-30B-A3B, Qwen3-235B-A22B, and Nemotron3 Super
model matrix. Each input is an explicit model-to-run-root mapping. A run root
must contain one `*-logs/ray-driver.log` file for each backend at
`flashinfer_cutlass/` and `flashinfer_cutedsl/`.

Run it after both eight-step measurement arms complete:

```bash
python experiments/mxfp8_linear_backend_model_matrix/summarize_results.py \
  --model-run qwen3-30b=/lustre/.../qwen30b-mxfp8-linear-backends/RUN_ID \
  --model-run qwen3-235b=/lustre/.../qwen235b-mxfp8-linear-backends/RUN_ID \
  --model-run nemotron3-super=/lustre/.../nemotron3-super-mxfp8-linear-backends/RUN_ID \
  --output-dir /lustre/.../mxfp8-linear-backend-model-matrix/RUN_ID
```

The output contains every parsed log block in `step_metrics.csv` and the
steady-state steps 3 through 8 in `summary.json`. The JSON preserves absolute
generation and end-to-end throughput plus generation and end-to-end latency.
Throughput ratios are normalized to CUTLASS; latency values are expressed as
CUTLASS latency divided by the backend latency, so values above one indicate a
faster backend.

The tool rejects a row before writing a result unless exactly one driver log is
present per backend, every requested measured step is complete, and the paired
measured steps have identical mean generation lengths. A malformed
`Training Results` block is rejected at its original step ordinal. This
validation prevents a speedup from being reported for incomplete or unmatched
generation work.
