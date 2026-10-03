# Precision Matrix Refresh

This experiment compares four precision arms on one pinned NeMo-RL source
revision. Each model and execution mode keeps its workload, GPU count,
parallelism, and two policy logprob passes fixed across all arms.

The refresh uses vLLM 0.29 and records the exact source and image in its
submission ledger. On October 3, main `fb8396ada` and the refreshed PR3294
head `54ed892b2` were merged into the integration branch. Earlier completed
performance results remain pinned to `351da834c`; they are not measurements
of today's integration source. Verify the nightly runtime before submitting
new GPU comparisons.

| Arm | Policy training | Rollout |
|---|---|---|
| `bf16-bf16` | BF16 | BF16 FlashInfer TRTLLM |
| `bf16-mxfp8` | BF16 | MXFP8 FlashInfer TRTLLM |
| `mxfp8-false-mxfp8` | MXFP8 with BF16 parameters | MXFP8 FlashInfer TRTLLM |
| `mxfp8-true-mxfp8` | MXFP8 with MXFP8 parameters | MXFP8 FlashInfer TRTLLM |

`sync` uses colocated CUDA IPC refit. `async` uses
disaggregated NCCL Reshard refit and keeps the smaller training topology because
generation has separate workers. Qwen3.5 Async reserves one four-node segment
for training and one four-node segment for generation, which keeps the EP16
training group inside one NVLink domain. All runs execute 20 steps; reports use
steps 2-20. Every arm uses the same FlashInfer TRTLLM backend so precision is
the only generation change.

Qwen3.5 performance runs use 128 prompts and 16 generations per prompt, for a
training global batch size of 2048. The launcher applies all three values to
every Qwen3.5 precision arm so BF16 and MXFP8 rollout runs stay matched.
Every arm also uses router dtype `fp32` and GRPO seed `42`; this avoids
comparing the inherited BF16 `fp64` router against MXFP8 training with an
`fp32` router.

Set `QUANT_SCOPE=moe_qkvo` on a Qwen3.5 MXFP8 rollout to quantize full-attention
Q/K/V/O projections in addition to routed experts. Linear-attention modules,
shared experts, routers, vision modules, MTP, and the LM head remain BF16. The
first two and last six layers also remain BF16. The default `QUANT_SCOPE=moe`
keeps all attention projections in BF16.

Automatic KV-cache accounting can overestimate available memory
after CuMem sleep-pool reclamation. Set `GPU_MEMORY_UTILIZATION` to the same
value for every comparison arm when a model needs more wake-up headroom. This
keeps GBS, parallelism, CUDA Graphs, and all workload settings unchanged. The
launcher leaves each recipe's value intact when the variable is unset.
`SUPER_GPU_MEMORY_UTILIZATION` remains a compatibility alias for existing Super
commands.

Use the cluster-specific launcher so its scheduler arguments match the target
cluster. OCI requests `batch` and four GPU GRES per node. Ptyche requests
`batch`, and Lyris requests `gb200`; both allocate whole nodes without a GRES
request. Scheduler preflight rejects both clusters when no partition is given.
Lyris Qwen3-235B jobs read the immutable model snapshot from Lustre instead of
copying hundreds of GB into each job's node-local cache. Their dataset, venv,
Ray, and compiler caches still use `/raid/scratch`. Every exclusive allocation
clears only its own node-local run directory before creating the new cache.
Concurrent runs therefore cannot delete each other's source or environment;
durable Lustre logs and results are not removed.

The launcher keeps GPU-local CPU affinity but disables hard NUMA memory binding.
Large policy and reference workers can then use memory from the whole node
instead of exhausting one NUMA node while other host memory remains free. Set
`NRL_DISABLE_NUMA_MEMBIND=0` only for a controlled locality comparison.

Before submission, the launcher packs the clean source tree and all pinned
submodules into one immutable tar file under the durable result root. Each allocated node
extracts that file into its local scratch directory and builds there. Parallel
jobs therefore cannot race while building editable Megatron-Core extensions
from one shared source checkout.

This refresh pins Megatron-Bridge `1f8873bb` and nested Megatron-Core
`6a366090`. These revisions include construction-time module-name propagation,
so the TE precision recipe selects BF16 or MXFP8 parameter storage before each
module allocates its weights.

The launcher reuses prebuilt `/opt/ray_venvs` and places the node-local Bridge
and Megatron-LM source first on `PYTHONPATH`. The September 29 image used for
earlier measurements is no longer a full dependency match for October 3 main:
it has Ray 2.56.1, while main now requires >=2.58.0. A new nightly is being
staged separately. Do not call an older-image source overlay a full-lock
current-main validation, or reuse it for new performance results without
recording that difference and verifying the required actor dependencies.

## Lock-Aligned Refresh

The October 3 nightly also contains Ray 2.56.1. `build_aligned_runtime.sbatch`
derives one reusable image from it, synchronizing the pinned lock into the
driver and all six matrix actor interpreters at their existing `/opt` paths.
Builds and uv/enroot caches use node-local `/raid/scratch`; only the finished
squashfs, provenance and small audit logs persist on Lustre. Other backend
actor environments are not certified by this experiment.

The Ray CLI may use a direct Python shebang or uv's adjacent-Python shell
trampoline. The audit recognizes these entry points, probes the selected
interpreter against the driver, and runs `ray --version`. An arbitrary shell
wrapper or a different interpreter/Ray installation is rejected. Runtime
metadata records the selected interpreter and the actual CLI version.

The build checks exact source/submodule pins and base-image SHA256, records
the failing pre-sync runtime audit, and refuses publication unless the
post-sync audit passes. Run `smoke_nightly_image.sbatch` with
`ALIGNED_RUNTIME_REQUIRED=1` on one four-GPU GB200 node before benchmarks.
Pass the resulting immutable image explicitly as `CONTAINER`; do not use the
launcher's historical September 29 default for these runs. The base Docker
build labels remain historical; adjacent source metadata records the derived
image's authoritative source and dependency hashes.

Set `PERFORMANCE_RECIPE=1 PERFORMANCE_PROFILE=runtime-aligned` for the new
comparison. Historical YAMLs remain unchanged. This profile uses:

- Qwen3.5 Async: GBS2048, 16 nodes, eight training and eight generation nodes,
  segment8. TP/CP/EP and both logprob passes are unchanged.
- Super Sync: GBS256, 32 nodes, training TP2/EP16, rollout TP4, explicit 32 GiB
  KV cache in every arm. Host OOM remains a separate diagnosis; this budget
  addresses the observed GPU wake-up allocation failure, not proven host OOM.

All other cells retain their historical performance configuration. New run
names include `runtime-aligned`. Use a fresh, profile-specific submission
ledger and report these changed topologies/KV budgets separately from earlier
results. The configuration preflight must pass with the same profile used
for submission; composition is not a model correctness test.

`audit_runtime_lock.sbatch` inventories the driver, vLLM and Megatron actor
interpreters in a pinned nightly and runs `uv sync --frozen --dry-run` for
each role against the current source lock. Submit it on a CPU node with
`REPO`, full `SOURCE_SHA`, immutable `CONTAINER`, and durable `RESULT_DIR`.
It installs nothing, keeps temporary source/cache files on node-local scratch,
and saves three inventories and dependency plans. A passing dry run establishes
an installation plan, not runtime parity or a GPU/model correctness pass.

Lightning's existing GBS 16 YAMLs are functional smoke configurations, not
performance baselines. A proposed GBS 512 comparison uses 64 prompts x 8
generations, retains the model-specific mixer exclusions, and must keep both
logprob passes across all four arms. New wrappers and performance routing are
not yet implemented; GBS 2048 is a separate capacity/performance question.

The original Hugging Face weights, venvs, and compiler caches stay node-local,
but `NRL_MEGATRON_CHECKPOINT_DIR` points to the shared converted-checkpoint
cache. All policy ranks can therefore read the `run_config.yaml` and weight
shards produced by the one-time Hugging Face-to-Megatron conversion.

Run one arm on OCI:

```bash
MODEL=qwen30 MODE=async ARM=mxfp8-true-mxfp8 PERFORMANCE_RECIPE=1 ACTION=test-only \
  ./experiments/mxfp8_vllm029_fullperf_20260930/submit_oci.sh

MODEL=qwen30 MODE=async ARM=mxfp8-true-mxfp8 PERFORMANCE_RECIPE=1 ACTION=submit \
  ./experiments/mxfp8_vllm029_fullperf_20260930/submit_oci.sh
```

For memory diagnostics, set `NRL_LOG_LEVEL=DEBUG` and `MAX_STEPS=2` while
retaining the performance recipe. This enables per-rank CUDA counters at
offload, IPC and export boundaries, plus a storage-size sample every 256
exported tensors. Diagnostics do not change weights, buffer ownership,
sleep level or workload size; do not mix their timings with the normal
20-step performance averages.

`test_refit_memory_logging.py` checks the diagnostic wrapper on CPU using
the actual method body, with the transport stubbed. It checks payload
identity and exception propagation, not GPU numerical correctness.

Qwen3.5 EP32 host-memory smoke tests use eight 4-GPU nodes. Run the all-to-all
arm first to isolate the memory effect of EP32, then enable HybridEP with the
same topology to measure dispatcher performance:

```bash
MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-alltoall MAX_STEPS=2 \
  ACTION=test-only ./experiments/mxfp8_vllm029_fullperf_20260930/submit_oci.sh

MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-hybridep MAX_STEPS=2 \
  ACTION=test-only ./experiments/mxfp8_vllm029_fullperf_20260930/submit_oci.sh
```

Submit a matrix by invoking the launcher once per model, mode, and arm. Run
`ACTION=test-only` first. OCI-HSG has all four model caches. Ptyche currently
has Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Qwen3.5-35B-A3B caches. Verify
the requested model cache before using the Lyris launcher.
