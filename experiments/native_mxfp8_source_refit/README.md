# Native MXFP8 Source Refit

This experiment validates native Transformer Engine MXFP8 parameter storage
(`fp8_param=true`) transferred to an MXFP8 vLLM generation model through the
non-colocated NCCL-Reshard path.

The first gate is a two-step Qwen3-30B-A3B run on four GCP-NRT B200 nodes. If
that succeeds, run the same immutable code and container for 20 steps.

```bash
ACTION=test-only MAX_STEPS=2 \
  experiments/native_mxfp8_source_refit/run_gcp_nrt.sh

ACTION=submit MAX_STEPS=2 \
  experiments/native_mxfp8_source_refit/run_gcp_nrt.sh
```

The launcher records the resolved code SHA, container path, topology, cache,
and command inputs under the shared experiment result root.
