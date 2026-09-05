# Precision Matrix Refresh

This experiment compares three precision arms on one pinned NeMo-RL source
revision. Each model and execution mode keeps its workload, GPU count, and
parallelism fixed across the three arms.

| Arm | Policy training | Rollout |
|---|---|---|
| `bf16-bf16` | BF16 | BF16 FlashInfer TRTLLM |
| `bf16-mxfp8` | BF16 | MXFP8 FlashInfer TRTLLM |
| `mxfp8-mxfp8` | MXFP8 with `fp8_param=true` | MXFP8 FlashInfer TRTLLM |

`sync` uses colocated CUDA IPC refit. `async` uses disaggregated NCCL Reshard
refit. All runs execute 20 steps; reports use steps 2-19.

Use the cluster-specific launcher so its scheduler arguments match the target
cluster. OCI requests `batch` and four GPU GRES per node. Ptyche requests
`batch`, and Lyris requests `gb200`; both allocate whole nodes without a GRES
request. Scheduler preflight rejects both clusters when no partition is given.
Lyris Qwen3-235B jobs read the immutable model snapshot from Lustre instead of
copying hundreds of GB into each job's node-local cache. Their dataset, venv,
Ray, and compiler caches still use `/raid/scratch`.

Run one arm on OCI:

```bash
MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=test-only \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh

MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=submit \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh
```

Submit a matrix by invoking the launcher once per model, mode, and arm. Run
`ACTION=test-only` first. OCI-HSG has all four model caches. Ptyche currently
has Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Qwen3.5-35B-A3B caches. Verify
the requested model cache before using the Lyris launcher.
