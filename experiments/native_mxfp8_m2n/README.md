# Native MXFP8 M2N validation

This transport-only experiment compares the NeMo-RL Python reshard fallback
with the real `nccl.m2n.reshard` operation. It sends existing E4M3 values and
compact E8M0 scale bytes, without dequantizing or requantizing either component.
It does not measure vLLM loading, TRTLLM layout conversion, or E2E training.

The test changes payloads three times and checks every received byte against
global-coordinate-based expected values. Cases include expert, row, and K-axis
sharding, replicated destinations, and dense projections. K shards must contain
complete 32-value blocks. Each backend uses the same payload and communicator.

Native mode fails if M2N is unavailable; it never silently measures the fallback.
`native-grouped` uses the upstream group boundary, which is not assumed to
improve performance. Cold calls and steady measurements are reported separately.
Latency is the maximum participating-rank time per iteration, including stream
completion. Payload throughput is aggregate destination bytes divided by that
latency, not a claim about physical link bandwidth.

Run on GB200 with the pinned NeMo-RL container. Keep source under `/home`,
temporary files under `/raid/scratch`, and only final JSON/log artifacts under
`/lustre`. Do not upgrade the container's Torch or vLLM to make M2N importable.
Library provenance and compatibility must be recorded first.

```bash
# From an allocated container with all required libraries installed.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/native_mxfp8_m2n/benchmark.py --backend python \
  --output /raid/scratch/$USER/m2n-python.json
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/native_mxfp8_m2n/benchmark.py --backend native \
  --output /raid/scratch/$USER/m2n-native.json
```

Use `submit_oci.sh` with `ACTION=test-only`, then `ACTION=submit`. Source must
be committed and pushed. The submission script pulls before freezing the SHA.
No model checkpoint or training dataset is needed.
