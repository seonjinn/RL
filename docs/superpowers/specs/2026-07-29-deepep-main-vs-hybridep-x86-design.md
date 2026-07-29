# DeepEP Main Versus HybridEP on x86 Design

## Goal

Measure standard DeepEP and HybridEP on CW-DFW H100 and GCP-NRT B200 with
matched NeMo-RL GRPO workloads. Use the exact latest branch heads observed on
2026-07-29:

- standard DeepEP: `main@dd758caf451848bd150e1046af3d0a73e5fff38d`;
- HybridEP: `hybrid-ep@f725d29699f5bda9ba789456bb9579af69844685`.

Report Policy training, LogProb, and end-to-end time and throughput together
with short-run numerical evidence, persistent storage cost, and HybridEP
padding overhead. Preserve the existing all-to-all results as a separate
control rather than relabeling them as DeepEP.

## Comparison Contract

The primary comparison is a branch-to-branch deployment comparison:

| Arm | DeepEP source | Megatron dispatcher |
| --- | --- | --- |
| DeepEP | `main@dd758caf451848bd150e1046af3d0a73e5fff38d` | `flex` + `deepep` |
| HybridEP | `hybrid-ep@f725d29699f5bda9ba789456bb9579af69844685` | `flex` + `hybridep` |

This comparison answers which current upstream implementation is preferable
as deployed. It does not isolate the dispatcher algorithm because the source
branches differ.

When the `hybrid-ep` package can also execute the `deepep` backend, add a
secondary backend-isolation comparison using the same f725 wheel for both
dispatcher choices. Label that result separately. Do not mix it into the
primary branch-to-branch headline.

The existing all-to-all arm remains a third control. Its historical
all-to-all-versus-HybridEP results stay visible in the report but do not count
as DeepEP-versus-HybridEP measurements.

## Dependency and Runtime Architecture

One installed Python distribution cannot resolve two DeepEP git revisions at
the same import location simultaneously. Keep the repository's canonical
HybridEP dependency pin at f725 so canonical HybridEP recipes remain
installable. Build immutable, architecture-specific wheels for each arm and
install the selected wheel into a per-run `/tmp` overlay that precedes the
shared actor environment on `PYTHONPATH`.

The launcher must prove that the imported package and native extension come
from that overlay. Record the source branch, source SHA, wheel SHA256,
container SHA256, GPU architecture, build job, runtime package version, and
resolved module paths in each run's metadata.

Generalize the x86 wheel builder with an explicit variant:

- `DEEPEP_VARIANT=deepep` checks out `main`, builds the exact SHA, and validates
  `deep_ep`, the legacy `Buffer` interface used by Megatron-Core, and the
  main-branch native extension;
- `DEEPEP_VARIANT=hybridep` checks out `hybrid-ep`, preserves the existing
  multi-node transport settings, and validates `HybridEPBuffer` plus
  `hybrid_ep_cpp`.

The builder must reject a branch/variant mismatch and publish only an
immutable wheel, checksum, and metadata under Lustre. Source checkout and
compilation use node-local job scratch.

## Latest Main Compatibility Layer

DeepEP main still exports the legacy `Buffer` required by the current
Megatron-Core fused all-to-all path, but its Python and native-extension
layout differs from the HybridEP branch:

- `EventOverlap` is exported at the `deep_ep` package top level rather than
  through the same `deep_ep.utils` import used by the pinned Megatron-Core;
- the main wheel exposes its extension through the current package namespace,
  not the `deep_ep_cpp` and `hybrid_ep_cpp` modules assumed by the existing
  HybridEP-only probe;
- `ElasticBuffer` is the V2 interface, but the current Megatron-Core standard
  DeepEP backend continues to use legacy `Buffer`.

Make the smallest upstream Megatron-LM compatibility change on a branch of
`git@github.com:seonjinn/Megatron-LM.git`. The import path must support both
dd758 main and f725 hybrid-ep without changing dispatcher behavior. Add a unit
test that loads representative package stubs for each export layout and proves
that `HAVE_DEEP_EP` remains true.

Commit and push the Megatron-LM change first, update the Megatron-Bridge
gitlink, then update the NeMo-RL gitlinks. Do not edit Megatron-LM only through
the enclosing Bridge worktree. All commits must be signed off.

DeepEP main requires PyTorch 2.10 and NCCL 2.30.4. The current repository lock
provides PyTorch 2.11.0 but NCCL 2.28.9. dd758 unconditionally compiles its
elastic NCCL backend and validates the loaded NCCL binary during top-level
import, so 2.28.9 is a hard build and import blocker even though
Megatron-Core uses the legacy `Buffer` path.

Prefer a newly staged NeMo-RL nightly that already contains NCCL 2.30.4. If
the staged nightly does not, stage the exact `nvidia-nccl-cu13==2.30.4` wheel
as an immutable Lustre artifact and install it into the same per-run overlay
as the selected DeepEP wheel. Prepend the overlay's NCCL library directory to
`LD_LIBRARY_PATH` before Ray starts. Apply the identical NCCL runtime to both
DeepEP and HybridEP arms so the comparison does not confound the dispatcher
with an NCCL-version difference.

Before performance work, record the effective PyTorch, CUDA, NCCL, NVSHMEM,
driver, and firmware versions from the allocated runtime. Require the runtime
probe to report NCCL 2.30.4 for both arms. A version mismatch is never hidden.

## Launcher and Recipe Contract

Extend `DISPATCHER_MODE` to accept:

- `recipe`: preserve the selected YAML;
- `deepep`: apply `moe_token_dispatcher_type=flex`,
  `moe_flex_dispatcher_backend=deepep`, and
  `moe_deepep_num_sms=20`;
- `hybridep`: apply `moe_token_dispatcher_type=flex`,
  `moe_flex_dispatcher_backend=hybridep`, and
  `moe_hybridep_num_sms=32`.

The DeepEP arm starts from the matched all-to-all performance recipe and adds
only the standard DeepEP dispatcher overrides. The HybridEP arm uses the
canonical HybridEP performance recipe or the equivalent launcher override.
Resolved configs must match outside the dispatcher-specific fields and run
labels.

Every x86 run requires an explicit DeepEP wheel and exact SHA. The launcher
must reject:

- a wheel without an explicit full SHA;
- `deepep` paired with a HybridEP-only wheel;
- `hybridep` paired with a main-only wheel;
- an overlay import that resolves outside the expected `/tmp` directory;
- a source SHA, wheel checksum, or runtime API mismatch.

Do not modify `nemo_rl/utils/venvs.py`. Reuse one prepared driver/Ray
environment and one actor-environment set per hardware platform. Put source,
containers, wheels, caches, environments, checkpoints, and run artifacts
under Lustre, not `/home`.

## Workload Gates

Use the existing valid performance recipes and model revisions. Proceed in
this order on each hardware platform:

1. one-node allocated-GPU import/API smoke for each wheel;
2. matched three-step Qwen3-30B-A3B DeepEP and HybridEP runs;
3. matched short Qwen3-235B and Nemotron3 Super runs at their valid topology;
4. matched 20-step runs for workloads that pass the compatibility gate and
   fit the current allocation.

Do not submit a larger model until its checkpoint snapshot is complete and
offline verification reports zero missing referenced shards.

Each submission requires:

- a clean pushed source commit and recursive submodules at the recorded
  gitlinks;
- `git pull --ff-only` on the cluster checkout;
- current FairShare inspection;
- `sbatch --test-only`;
- at least five minutes of monitoring after submission;
- bounded log scans for import errors, actor loss, timeout, NCCL, CUDA, RDMA,
  OOM, NaN, Inf, and checkpoint failures.

If dd758 main fails because of an upstream API or runtime dependency
incompatibility, preserve and report the exact failure. Do not silently
replace it with an older main commit. Any later compatibility search is a
separately labeled fallback experiment.

## Performance Metrics

Clean timing runs disable padding diagnostics. For completed 20-step pairs,
use matched steps 5-20. For three-step compatibility runs, report common
completed steps and label the numbers as smoke-only.

For each model, hardware platform, and arm, report:

- Policy training mean, median, total seconds, and ratio-of-sums
  tokens/second/GPU;
- Policy and reference LogProb mean, median, total seconds, and
  ratio-of-sums tokens/second/GPU;
- end-to-end mean, median, total step seconds, and ratio-of-sums
  tokens/second/GPU;
- generation time and throughput as supporting context;
- completed-step count, selected step window, topology, and GPU count.

Use:

```text
throughput_change_percent =
    (HybridEP throughput / DeepEP throughput - 1) x 100

time_reduction_percent =
    (1 - HybridEP time / DeepEP time) x 100
```

Never average per-step throughput ratios. Sum the phase's tokens, divide by
the phase's summed time, then divide by the GPUs assigned to that phase.

## Numerical and Accuracy Evidence

Collect reward, generation KL, entropy, gradient norm, validation accuracy,
response length, and all non-finite signals on the same comparison window.
For 20-step runs, show per-step traces and aggregate deltas. Identical or
nearby short-run values are smoke-level evidence only; they do not prove
long-horizon convergence equivalence.

A compatibility run passes numerical smoke when:

- both arms complete the requested common steps;
- all required metrics are finite;
- no invalid-token, loss-scaling, or checkpoint error occurs;
- reward, KL, and validation differences are reported without presenting a
  stochastic three-step delta as a regression.

## Storage and Padding Overhead

Separate persistent storage from transient HybridEP padding:

- per-run storage: raw bytes under each run root after terminal completion;
- wheel storage: immutable wheel bytes per architecture and variant;
- shared environment/cache storage: before-and-after raw bytes with path,
  scope, method, and timestamp;
- checkpoint storage: reported separately and excluded from dispatcher
  overhead.

Run HybridEP packing diagnostics as a separate non-timing job. Report weighted
fake-token overhead, median, p95, and maximum per-call overhead. The weighted
metric is `sum(added tokens) / sum(raw tokens)`. Explain that fake tokens
consume transient activation, compute, and communication capacity, not
persistent disk space.

## Report Schema and HTML

Upgrade the existing Pages artifact so every completed workload has explicit
`deepep` and `hybridep` arm objects. Each arm requires:

- NeMo-RL, Megatron-Bridge, and Megatron-LM commits;
- DeepEP branch and commit;
- wheel and container SHA256;
- recipe path and resolved-config hash;
- dispatcher settings and runtime library versions;
- job ID, terminal state, node list, log path, and W&B source;
- performance, numerical, storage, and padding metrics with measurement
  windows.

Normalize storage keys to raw bytes plus human-readable display values.
Generate status and limitations from the structured artifact so stale manual
footer text cannot contradict the result rows.

The HTML page stays concise:

- one headline conclusion per model and hardware pair;
- grouped Policy, LogProb, and end-to-end throughput/time charts;
- a compact quality and overhead table;
- exact branch/SHA provenance and smoke-versus-performance validity;
- links to lightweight scripts/config snapshots and Lustre log paths.

Large logs and model artifacts remain on Lustre. The Pages repository stores
only HTML, small JSON summaries, plots, scripts, and configuration snapshots.

## Verification

Before any cluster submission:

1. add failing tests for both DeepEP export layouts;
2. add failing launcher tests for the `deepep` mode and variant mismatch;
3. add failing wheel-builder contract tests for `main` and `hybrid-ep`;
4. implement the minimal compatibility and launcher changes;
5. run focused Megatron-LM tests in the approved container;
6. run focused NeMo-RL launcher/recipe tests, shell syntax checks,
   `uv lock --check`, and `git diff --check`;
7. add failing report validator/renderer tests for provenance, quality,
   storage, and padding;
8. update the report pipeline and require its focused tests plus HTML/JSON
   parsing to pass;
9. commit with sign-off, push exact branches, and record all resulting SHAs.

No result is called successful until its exact verification command and
terminal output have been inspected.
