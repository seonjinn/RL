# MXFP8 Sync Refit Phase Profile

This experiment profiles one steady-state refit for Qwen3-30B-A3B on the
synchronous colocated CUDA IPC path. It keeps the model topology and refit
configuration from the final PR 3294 run, but reduces rollout work so the
profile focuses on refit.

The added NVTX ranges separate trainer quantization and staging, IPC bucket
fences and waits, vLLM loading and finalization, and the MXFP8 MoE layout
conversion phases. The run captures step 2 only and writes Nsight reports to
the Ray log directory synced by `ray.sub`.

```bash
sbatch experiments/mxfp8_sync_refit_profile/submit_oci_hsg.sbatch
sbatch experiments/mxfp8_sync_refit_profile/submit_lyris.sbatch
```
