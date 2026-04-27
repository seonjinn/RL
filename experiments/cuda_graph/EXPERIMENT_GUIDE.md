# CG Experiment Guide — Local-first, Cluster-only-for-runs

This is a step-by-step protocol for any agent or human picking up the CUDA Graph (CG) + GRPO performance work. It is local-first: you understand and edit code on your laptop. The cluster (OCI-HSG) is a pure compute target — no editing there, no exploratory shell sessions there.

If you have not yet read the cross-session handoff at `memory/project_cg_offload_stream_handoff_apr26.md`, read it first. This guide assumes you understand the basic problem (CG slows GRPO offload phases by 18-50 s/step) and the attempts we have made.

---

## 0. One-time orientation (do this once per session)

1. Check whether there is an active SSH control socket: `ssh -O check oci-hsg-cs-001-vscode-01`. If "Master running" you are good. If not, ask the user to run `! ssh oci-hsg-cs-001-vscode-01` so MFA happens once and the socket lives for 24 h.
2. Confirm local checkout is the right branch: `git -C /Users/sna/CudaGraph_PR/RL status` should show `feature/cuda-graph-training`.
3. Confirm cluster checkout has the same HEAD: `ssh oci-hsg-cs-001-vscode-01 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-cg-test && git log --oneline -3'` should match local.

---

## 1. Repository layout (local and remote)

| Local (read / edit) | Cluster (run only) |
|---|---|
| `/Users/sna/CudaGraph_PR/RL` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-cg-test` |
| `/Users/sna/CudaGraph_PR/Megatron-LM` | (mounted via container at runtime) |
| `/Users/sna/CudaGraph_PR/TransformerEngine` | (mounted via container at runtime) |

The cluster mirror is a clone of the local RL repo (remote `seonjinn`). The Megatron-LM and TransformerEngine sources are baked into the container image; if you change them locally you must either rebuild the image or use a mount override.

The active branch is `feature/cuda-graph-training`. The `seonjinn` remote is `git@github.com:seonjinn/RL.git`. `origin` is `git@github.com:NVIDIA-NeMo/RL.git` (upstream, do not push).

---

## 2. Files that matter

### Worker code (all CG offload optimizations live here)

`nemo_rl/models/policy/workers/megatron_policy_worker.py`

- `_offload_copy_stream` property (~line 246) — lazy `torch.cuda.Stream()` for offload copies.
- `use_reference_model` context manager (~line 870) — saves policy state, loads reference, yields, restores.
- `move_model` method (~line 1521) — DDP / FSDP / generic offload between cuda and cpu.
- `move_optimizer` method (~line 1569) — per-tensor optimizer state move.

### Setup code (one-time pinning)

`nemo_rl/models/megatron/setup.py:setup_reference_model_state` (~line 1163) — allocates pinned host buffers for `reference_state_dict` once at startup.

### Megatron-LM (read-only for now; modifying requires container rebuild)

`/Users/sna/CudaGraph_PR/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:964-994` — `_ParamAndGradBuffer.offload_to_cpu` / `reload_from_cpu`. Already uses `non_blocking=True` and pinned pages. Picks up the active stream from the calling context.

### Submission scripts (in cluster repo root)

| Script | Purpose |
|---|---|
| `run_cg_llama_synfix.sh` | Training run, 40 steps, val_period=10, seed=42. Outputs to `experiments/cuda_graph/llama_synfix/slurm-<JOBID>.out`. Use this for step-time + reward + val_acc measurements. |
| `run_cg_llama_nsys_synfix.sh` | nsys profiling, 7 steps, profiles steps 5-6. Outputs `experiments/cuda_graph/llama_nsys_synfix/megatron_policy_worker_*.sqlite`. Use for stall analysis. |

The scripts have hardcoded sbatch flags (`--account=coreai_dlalgo_nemorl`, `--partition=batch`, `--nodes=1`, `--gres=gpu:4`, `--time=03:30:00`). The Hydra overrides are inside the `cg_run_llama_*.sh` helper invoked under srun.

The standard overrides (do not change without recording why):

```
grpo.max_num_steps=40
grpo.val_period=10
grpo.seed=42
+policy.megatron_cfg.distributed_data_parallel_config.num_distributed_optimizer_instances=2
+policy.megatron_cfg.cuda_graph_te_overlap_replay=true
```

`num_distributed_optimizer_instances=2` is the dpinst=2 setting that interacts with Track 2 overlap. `cuda_graph_te_overlap_replay=true` enables TE's replay overlap stream.

---

## 3. The submission workflow

Always:

1. Edit locally.
2. Commit: `git -C /Users/sna/CudaGraph_PR/RL add <files> && git commit -s -m "..."`. Do not amend, do not skip hooks.
3. Push to seonjinn remote: `git push seonjinn feature/cuda-graph-training`.
4. Sync cluster: `ssh oci-hsg-cs-001-vscode-01 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-cg-test && git fetch seonjinn feature/cuda-graph-training && git reset --hard seonjinn/feature/cuda-graph-training && git log --oneline -3'`. Confirm HEAD matches.
5. Submit: `ssh oci-hsg-cs-001-vscode-01 'cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-cg-test && rm -f experiments/cuda_graph/llama_synfix/slurm-*.out && sbatch run_cg_llama_synfix.sh'`. Capture the JOBID.
6. Record the JOBID locally (in memory or a status file). Do not submit and forget.

Never edit code on the cluster. If you find yourself wanting to, the right move is to fix it locally and re-push.

If the cluster has a stash from a previous reset, run `git stash drop` (only after checking it does not contain uncommitted experimental edits you want).

---

## 4. Measuring step time

After submission:

```
ssh oci-hsg-cs-001-vscode-01 'squeue -u $USER -j <JOBID> -o "%.10i %.10T %.10M"'
ssh oci-hsg-cs-001-vscode-01 'grep -E "Step [0-9]+/40|Total step time" /lustre/.../experiments/cuda_graph/llama_synfix/slurm-<JOBID>.out | tail -15'
```

Step 1 always includes CG capture (typically 110-130 s). The number that matters is steady-state — step 2 onwards.

Reference numbers (Llama-3 1B, 4 GPU, OCI-HSG nvl72 nodes):

| Configuration | s/step (steady state) |
|---------------|------------------------|
| noCG baseline | 44 |
| CG + .item() patch only (a008a47b) | 62 |
| CG + sync fix (f087adc6) | 110-121 |
| CG + pinned dest (e4c01800) | 110-115 |
| CG + copy stream (4131835f) | running |

The real target is step time ≤ 44 s with CG enabled, while keeping reward / val_acc / mean_logprob bit-identical to noCG.

---

## 5. Measuring accuracy

Pull the metrics file:

```
ssh oci-hsg-cs-001-vscode-01 'grep -E "reward|val_acc|mean_logprob" /lustre/.../experiments/cuda_graph/llama_synfix/slurm-<JOBID>.out | tail -50'
```

Compare against the same fields from a noCG run with the same seed. Acceptable tolerance is `< 1e-4` divergence by step 40. Anything larger means a correctness bug — usually a missing `wait_stream` between a copy stream and a downstream consumer.

---

## 6. nsys profiling

When step time is unexpected (regressed or unchanged), profile:

```
ssh oci-hsg-cs-001-vscode-01 'cd /lustre/.../nemo-rl-cg-test && sbatch run_cg_llama_nsys_synfix.sh'
```

After it finishes, copy the analysis script and run it on the cluster:

```
scp /tmp/nsys_top100.py oci-hsg-cs-001-vscode-01:/tmp/
ssh oci-hsg-cs-001-vscode-01 'python3 /tmp/nsys_top100.py /lustre/.../experiments/cuda_graph/llama_nsys_synfix/megatron_policy_worker_*.sqlite | head -120'
```

What to look at:

- D2H + H2D long-tail (>50 ms) count and total. Pre-fix this was 154 stalls / 47 s.
- NVTX containment of each long stall. If most stalls are outside any NVTX range, they are at phase boundaries (offload/refit). If they fall inside `transformer_layer` or `attention_forward`, the source is TE / Megatron internals, not RL workers.
- Surrounding kernels. Helps map a stall to the offending phase.

If `nsys_top100.py` does not exist on the cluster, recreate it locally and `scp`. The script is small — it queries the sqlite tables `CUPTI_ACTIVITY_KIND_RUNTIME` (cudaMemcpyAsync calls), `CUPTI_ACTIVITY_KIND_MEMCPY` (the actual GPU-side copies), and `NVTX_EVENTS` (for context).

---

## 7. Cancelling and cleaning up

Cancel: `ssh oci-hsg-cs-001-vscode-01 'scancel <JOBID>'`. Confirm with `squeue -u $USER`.

Do NOT `rm -rf experiments/cuda_graph/llama_synfix/` between runs unless you know the outputs are saved elsewhere — slurm output files have run-specific findings. The submission script removes only `slurm-*.out`, not the logs subdirectory.

---

## 8. Common pitfalls (read these before debugging)

- **The 47 s of nsys stalls is not 47 s of wall-clock loss.** Most of it overlaps with vllm sleep/wake and ray actor work. Always validate end-to-end step time, not nsys aggregates alone.
- **Pinned host alone does not fix CG-induced memcpy stalls.** The dominant cost is queue back-pressure on the default stream, not pageable fallback. Pinning is necessary but not sufficient.
- **Never `torch.cuda.synchronize()` on the hot path to "fix" memcpy stalls.** It collapses all the queued cost into one barrier, breaks every overlap that was working, and regresses step time. Confirmed: f087adc6 went 62 → 110-121 s.
- **`wait_stream` is required at every copy-stream boundary.** Producer side: `copy_stream.wait_stream(default_stream)` before issuing copies. Consumer side: `default_stream.wait_stream(copy_stream)` after the copy block. Missing one of these causes silent races and accuracy drift.
- **First step always includes CG capture overhead.** Do not report step 1 as the steady-state number. Use step 2+.
- **Hybrid pretraining (mamba/attn/MoE-router) does not reproduce this regression** because pretraining has no offload phase. The bug is GRPO-specific. Do not assume CG works because it works in pretrain benchmarks.

---

## 9. Decision points

After every measurement, before doing more work, ask:

1. Did step time drop vs the prior run? If no, the change is likely wrong — revert or adjust.
2. Did accuracy stay within `1e-4`? If no, find the missing wait_stream.
3. Did the nsys mega-stall count drop? If yes but step time did not, the stalls were already overlapping with something useful.
4. Are we within striking distance of 44 s? If yes, validate with a longer run (40+ steps) and start looking at reward/val_acc trajectories.

When stuck (two attempts in a row regress or fail to move the number), stop and write a memory note. Do not loop without recording.

---

## 10. What to do when picking this up next session

1. Read `memory/project_cg_offload_stream_handoff_apr26.md` (cross-session handoff).
2. Read this file (workflow).
3. `ssh oci-hsg-cs-001-vscode-01 'squeue -u $USER -o "%i %T %M %j"'` to see what jobs are alive.
4. If a job is running, check its steady-state step time (Section 4).
5. If no job is running and you do not yet know the result of the last submission, find the most recent `slurm-*.out` in `experiments/cuda_graph/llama_synfix/` and read its tail.
6. Apply the decision tree from Section 9.
