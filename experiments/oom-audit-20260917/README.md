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

Fresh image: PyTorch 2.11.0+cu130, vLLM 0.25.1. Image and main-base dependency
pins match. GPU imports and real pinned-tensor accounting smoke passed.

| Super BF16 TP4 initialization probe | Non-Torch delta | KV budget | Result |
| --- | ---: | ---: | --- |
| Sleep enabled | -52.93 GiB | 115.94 GiB | GPU OOM |
| Sleep disabled | +1.07 GiB | 61.94 GiB | Initialization passed |

Both use dummy weights, utilization 0.7, and the same 64.35 GiB model allocation.
These are isolation probes, not real-weight inference or training validation.
Do not disable sleep in training without validating its offload/refit lifecycle.

Pool inspection measured 54.001953125 GiB of idle segments absent from CuMem's
pointer registry but still present in PyTorch's reserved-pool snapshot, on all
four GPUs both before and after profiling. No missing non-idle segment was found.
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
