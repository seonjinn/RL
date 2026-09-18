# Large-model BF16 host-memory investigation

Base: NeMo-RL main `8241b9f6f4caf162e35b785142922d934e108d86`.

Scope: Nemotron3 Super Sync and Qwen3-235B Sync performance recipes on GB200.
AlltoAll baseline versus HybridEP; do not change batch, precision, or algorithm
to make the memory failure disappear. Diagnostic configuration changes must be
recorded separately. No MXFP8 work is included.

Prior Super TP8 diagnostic completed step 1, then failed during step-2
`offload_before_refit`; node memory reached 895.31/908.79 GiB. The per-category
tensor/allocator breakdown remains unmeasured. No causal regression PR identified.

## Measured evidence (September 17)

### Full-run CPU failure localized to reference weight swapping

The NUMA-relaxed AlltoAll run passed generation and policy initialization but
failed during reference logprob with an explicit Ray host-memory OOM:
870.20/908.79 GiB, exceeding the 95% protection threshold. Disabling hard NUMA
memory binding therefore did not resolve total host-memory pressure.
[Run evidence](https://wandb.ai/nvidia/sna-hybridep-memory-audit/runs/pq9wdzvt).

| Policy-worker allocation | CPU tensor payload | Pinned allocator increment |
| --- | ---: | ---: |
| Policy offload backup | 20.74 GiB | 24.00 GiB |
| Persistent reference weights | 20.90 GiB | 30.91 GiB |
| Temporary policy copy for reference swap | 20.90 GiB | 30.91 GiB |

`use_reference_model()` creates the temporary copy using
`item.detach().to(device="cpu", non_blocking=True, copy=True)` while the
existing policy backup and reference tensors remain resident. Measured allocator
backing rises from 54.91 to 85.82 GiB per policy worker; four such increments
represent approximately 123.64 GiB per node. Driver records are deduplicated
across ranks, so printed before/after PSS values are not a same-PID paired trace.
The raw rank-99 initialization log confirms the pre-swap 54.91 GiB allocation,
and its after-copy driver record reports 85.82 GiB.

Independent one-second node samples corroborate this allocation: one node rose
from 759.61 to 885.11 GiB over approximately seven seconds around reference
swapping. Its shmem component rose from 656.12 to 779.76 GiB, a **123.64 GiB**
increase, matching four 30.91 GiB temporary pinned allocations. Across 32 nodes,
observed cgroup peaks were 838.91–892.28 GiB. Sampled kernel `oom_kill` counters
remained zero; this is consistent with Ray proactively killing workers before
the kernel memory limit. Shmem is part of file memory and is not added twice.

At termination, Ray listed four generation workers at 91.80–92.58 GiB each and
four policy workers at 90.83–91.64 GiB each. These process measurements are not
additive with cgroup totals or shared-memory counters. The temporary copy is
the observed peak trigger on top of generation offload residency, allocator
slack, and container memory—not evidence of an unbounded leak or a HybridEP
dispatch failure. The copy operation predates this nightly and is also present
in the parent of commit e2f48a9973 (PR #2053); line blame alone does not identify
its introduction. The specific change making the combined working set exceed the limit
is still unproven. No successful 20-step run or production fix is claimed.

Existing policy CPU backup allocations are a potential reusable destination,
but their contents cannot be assumed current after training. A safe reuse path
must refresh them from the resident policy, preserve uncovered buffers and
extra state, and verify exact policy restoration. Simply deleting the temporary
copy or restoring stale offload values is not a valid fix.

An opt-in diagnostic now implements this refresh with
`AUDIT_REUSE_REFERENCE_BACKUP=1`. It accepts ordinary BF16 DDP parameter
buffers, validates storage and view bounds, completes the current GPU-to-CPU
copy before reference installation, and separately copies uncovered state.
It does not assume the existing offload backup is current. Two real-GPU tests
failed on the original copying behavior and pass on the new helper, including
stale backups, strided views, uncovered FP32 state, repeated refresh, and
incompatible backup rejection. Full-model correctness and memory validation
remain required; this is not a production default or a performance result.

### Backend-only control

The Triton-only control completed (exit 0, 4m23s), including initialization and
two sleep/wake cycles, without the experimental accounting correction. Same
image, dummy weights, BF16, TP4, sleep enabled, and utilization 0.7:

| Per worker / GPU | Automatic backend | Explicit Triton |
| --- | ---: | ---: |
| Model allocation | 64.35 GiB | 56.85 GiB |
| Missing idle pool segments | 54.00 GiB | 0 GiB |
| Profiling non-Torch delta | -52.93 GiB | +1.06 GiB |
| CPU sleep-backup payload | 64.48 GiB | 56.98 GiB |
| CPU backup backing-map RSS | 86.39 GiB | 86.39 GiB |

Automatic-backend CPU numbers come from the accounting-corrected lifecycle probe,
because the uncorrected automatic-backend probe fails before sleep. Triton fixes
the observed initialization accounting symptom but does **not** reduce pinned
host backing: both storage-size distributions round to the same allocator size
classes. This is not real-weight correctness, throughput, or 20-step validation.
The 7.5 GiB weight difference is exactly explained by TRTLLM intermediate-dimension
padding from 672 to 768: 40 MoE layers × 2 matrices × 512 experts × 1024 latent
width × 96 extra columns/rows × 2 BF16 bytes = 8,053,063,680 bytes per TP rank.
This is generation-backend weight-layout padding, not HybridEP token padding.
The full utilization-0.6 rerun failed during policy actor startup; the first dead
worker log has no terminal Python traceback. Subsequent NCCL peer-closed errors
are not proof of a networking root cause or CPU OOM. Full attribution remains open.

Fresh image: PyTorch 2.11.0+cu130, vLLM 0.25.1. Image and main-base dependency
pins match. GPU imports and real pinned-tensor accounting smoke passed.

| Super BF16 TP4 initialization probe | Non-Torch delta | KV budget | Result |
| --- | ---: | ---: | --- |
| Sleep enabled | -52.93 GiB | 115.94 GiB | GPU OOM |
| Sleep disabled | +1.07 GiB | 61.94 GiB | Initialization passed |
| Sleep enabled + measured idle-segment accounting correction | +1.07 GiB | 61.94 GiB | Initialization passed |

Both use dummy weights, utilization 0.7, and the same 64.35 GiB model allocation.
These are isolation probes, not real-weight inference or training validation.
Do not disable sleep in training without validating its offload/refit lifecycle.

Pool inspection measured 54.001953125 GiB of idle segments absent from CuMem's
pointer registry but still present in PyTorch's reserved-pool snapshot, on all
four GPUs both before and after profiling. No missing non-idle segment was found.
The corrected probe completed with exit 0 in 4m28s and emitted INIT_PROBE_PASS.
One shared-memory resource-tracker cleanup warning occurred on shutdown; this
is not evidence of a clean repeated lifecycle or real-weight training success.
This matches the KV-budget inflation. The opt-in `AUDIT_CORRECT_CUMEM=1` probe
corrects only snapshot accounting during awake initialization; it is experimental,
not a production sleep/wake allocator fix. It rejects missing active segments
or corrections exceeding reported reservation. A CUDA-graph profiling-boundary
relocation was tested and **did not fix** the issue; do not deploy that option.

Separately, an image-only comparison found rootfs tmpfs residency increased from
72.29 to 87.81 GiB per node. `shmem` is included in `file`, not additive. Environment
overrides did not relocate the Pyxis rootfs. This reduces CPU headroom but is not
proof of the historical CPU OOM's complete cause.

The full Super utilization-0.6 diagnostic passed generation initialization. Each
TP4 vLLM worker then backed up 64.48 GiB of weights to CPU and discarded 97.75 GiB
of KV cache. Policy initialization subsequently failed while starting the async
checkpoint Manager, before model setup completed. `ActorDiedError`/`EOFError`
do not establish CPU OOM. Detailed worker logging is enabled for the next run.
Top-level `checkpointing.enabled=false` does not disable the nested
`policy.megatron_cfg.checkpoint.async_save=true` startup path.

Full CPU tensor attribution, regression-PR attribution, matched HybridEP runs,
and successful 20-step validation remain outstanding.

## CPU sleep-backup residency

### Pinned versus pageable diagnostic

Two fresh one-node probes used the automatic backend and the same initialization
accounting correction, BF16, TP4, utilization0.7, and two sleep/wake cycles.
Both emitted INIT_PROBE_PASS and passed bytewise first/last-scalar checks for
each CUDA parameter after each wake (not exhaustive weight or accuracy checks).
Both explicitly synchronized after wake. Measurements below span four workers;
they are diagnostic observations on separate nodes, not E2E performance results.

| Per worker | Pinned | Pageable |
| --- | ---: | ---: |
| Backup payload | 64.48 GiB | 64.48 GiB |
| Process PSS after sleep | 92.40–93.72 GiB | 74.23–76.26 GiB |
| Process PSS after wake | 92.40–93.72 GiB | 12.17–20.10 GiB |
| First sleep | 4.93–5.59 s | 12.45–13.65 s |
| Second sleep | 0.75–1.12 s | 31.88–35.01 s |
| Wake, both cycles | 0.74–1.10 s | 0.65–0.84 s |

Pageable backing reduces retained CPU memory but has a severe observed offload
cost. Do not adopt it as a default from these two-cycle dummy-weight probes.
The experiment changes only CuMem sleep backup pinning, not policy offload.
`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` controls a different offload path.

The corrected initialization followed by two level-1 sleep/wake cycles completed
with exit 0 in 4m50s. This was still a dummy-weight probe, not a training benchmark.

| Per TP4 vLLM worker | After sleep | After wake |
| --- | ---: | ---: |
| Live CPU backup storage | 64.48 GiB | 0 GiB |
| Process PSS | 91.97–93.06 GiB | Essentially unchanged |
| Process shared-memory PSS | 86.74 GiB | Essentially unchanged |

All backup pointers intersected `/dev/zero (deleted)` mappings. For one worker,
291 matching VMAs totaled 86.39 GiB in Size/RSS/PSS, versus 64.48 GiB of tensor
storage. VMA counters can include allocator slack, not just the live payload.
The second cycle remained stable within a few MiB; this does not demonstrate an
unbounded leak. The follow-up probe found the sum of per-storage power-of-two
rounding to be exactly **86.392578125 GiB**, matching the VMA RSS. This accounts
for **21.9140625 GiB of size-class slack per worker**, or 87.65625 GiB across
four workers. Allocator reserved/allocated backing remained 86.4086 GiB with no
new host allocation/free calls on cycle two. Its active-byte counter increased
despite stable backing and OS PSS; do not interpret that counter alone as live
physical memory or evidence of an accumulating leak.

The archived July 15 image contains the same PyTorch 2.11.0+cu130 and vLLM 0.20.0.
Its CuMem manual idle-segment unmap and reserved-memory subtraction already exist.
That rules out describing those mechanisms as newly introduced in vLLM 0.25.1;
the exact change exposing the failure still requires controlled runtime evidence.

## Validation gates

1. Stage a new immutable nightly with node-local import caches.
2. Record image checksum, baked commit, package versions, and executed source paths.
3. Pass one-node GPU/worker import checks before multi-node launch.
4. Measure reference weights, policy CPU backup, optimizer state, pinned allocator,
   process PSS/USS, cgroup anonymous/file memory around refit and reference-logprob.
5. Reproduce with matched recipes; distinguish live tensor residency, allocator
   caching, transient copies, and actual growth across repeated steps.
6. Test the evidence-supported fix in both arms through at least 20 steps;
   preserve validation and accuracy metrics. Profiling timings are not benchmark results.

Do not disable Ray's memory protection or treat a longer timeout as an OOM fix.
Source lives in /home, build/cache data on node-local scratch, and images and
durable results on Lustre. Existing jobs and known-good images are preserved.
