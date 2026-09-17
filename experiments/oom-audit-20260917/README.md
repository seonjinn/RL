# Large-model BF16 host-memory investigation

Base: NeMo-RL main `8241b9f6f4caf162e35b785142922d934e108d86`.

Scope: Nemotron3 Super Sync and Qwen3-235B Sync performance recipes on GB200.
AlltoAll baseline versus HybridEP; do not change batch, precision, or algorithm
to make the memory failure disappear. Diagnostic configuration changes must be
recorded separately. No MXFP8 work is included.

Prior Super TP8 diagnostic completed step 1, then failed during step-2
`offload_before_refit`; node memory reached 895.31/908.79 GiB. The per-category
tensor/allocator breakdown remains unmeasured. No causal regression PR identified.

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
