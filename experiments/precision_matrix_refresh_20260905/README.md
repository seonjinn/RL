# Precision Matrix Refresh

This experiment compares three precision arms on one pinned NeMo-RL source
revision. Each model and execution mode keeps its workload, GPU count, and
parallelism fixed across the three arms.

| Arm | Policy training | Rollout |
|---|---|---|
| `bf16-bf16` | BF16 | BF16 FlashInfer TRTLLM, except Qwen3-235B TP8 uses Triton |
| `bf16-mxfp8` | BF16 | MXFP8 FlashInfer TRTLLM |
| `mxfp8-mxfp8` | MXFP8 with `fp8_param=true` | MXFP8 FlashInfer TRTLLM |

`sync` uses colocated CUDA IPC refit. `async` uses disaggregated NCCL Reshard
refit. All runs execute 20 steps; reports use steps 2-19. Qwen3-235B has a
1536-wide expert dimension. TP8 produces a 192-wide local BF16 expert shard,
which the FlashInfer TRTLLM BF16 kernel rejects because it is not a multiple of
128. The Triton baseline matches the upstream Qwen3-235B performance recipe;
both MXFP8 arms continue to use FlashInfer TRTLLM.

Use the cluster-specific launcher so its scheduler arguments match the target
cluster. OCI requests `batch` and four GPU GRES per node. Ptyche requests
`batch`, and Lyris requests `gb200`; both allocate whole nodes without a GRES
request. Scheduler preflight rejects both clusters when no partition is given.
Lyris Qwen3-235B jobs read the immutable model snapshot from Lustre instead of
copying hundreds of GB into each job's node-local cache. Their dataset, venv,
Ray, and compiler caches still use `/raid/scratch`. Every exclusive allocation
clears this experiment's old node-local root before creating the new run cache;
durable Lustre logs and results are not removed.

Before submission, the launcher packs the clean source tree and all pinned
submodules into one immutable tar file under `/home`. Each allocated node
extracts that file into its local scratch directory and builds there. Parallel
jobs therefore cannot race while building editable Megatron-Core extensions
from one shared source checkout.

The original Hugging Face weights, venvs, and compiler caches stay node-local,
but `NRL_MEGATRON_CHECKPOINT_DIR` points to the shared converted-checkpoint
cache. All policy ranks can therefore read the `run_config.yaml` and weight
shards produced by the one-time Hugging Face-to-Megatron conversion.

Run one arm on OCI:

```bash
MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=test-only \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh

MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=submit \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh
```

Qwen3.5 EP32 host-memory smoke tests use eight 4-GPU nodes. Run the all-to-all
arm first to isolate the memory effect of EP32, then enable HybridEP with the
same topology to measure dispatcher performance:

```bash
MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-alltoall MAX_STEPS=2 \
  ACTION=test-only ./experiments/precision_matrix_refresh_20260905/submit_oci.sh

MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-hybridep MAX_STEPS=2 \
  ACTION=test-only ./experiments/precision_matrix_refresh_20260905/submit_oci.sh
```

Submit a matrix by invoking the launcher once per model, mode, and arm. Run
`ACTION=test-only` first. OCI-HSG has all four model caches. Ptyche currently
has Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Qwen3.5-35B-A3B caches. Verify
the requested model cache before using the Lyris launcher.
