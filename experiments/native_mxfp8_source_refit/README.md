# Native MXFP8 Source Refit

This experiment validates native Transformer Engine MXFP8 parameter storage
(`fp8_param=true`) transferred to an MXFP8 vLLM generation model through the
non-colocated NCCL-Reshard path.

The first gate is a two-step Qwen3-30B-A3B run on four GCP-NRT B200 nodes. If
that succeeds, run the same immutable code and container for 20 steps.

A two-node Qwen3-0.6B profile reserves one trainer node and one generation
node while exercising the same disaggregated refit contract as the larger job.

```bash
ACTION=test-only MAX_STEPS=2 \
  experiments/native_mxfp8_source_refit/run_gcp_nrt.sh

ACTION=submit MAX_STEPS=2 \
  experiments/native_mxfp8_source_refit/run_gcp_nrt.sh

ACTION=submit PROFILE=qwen06b MAX_STEPS=2 \
  experiments/native_mxfp8_source_refit/run_gcp_nrt.sh
```

The launcher records the resolved code SHA, container SHA256, topology, cache,
and command inputs under the shared experiment result root. The pinned archive
supplies `uv`; each node then creates Ray 2.56.1, driver, and actor
environments on Python 3.13.14 as required by this source revision. Writable
venvs are isolated by source SHA, container SHA256, and SLURM job ID; only the
download cache is shared across jobs. The node setup redirects the container's
`ray` entry point to that job-local runtime so the head, workers, and driver use
the same Python patch version.
