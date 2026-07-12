# NeMo 26.06 EP8 Staged Validation Design

- **Approved:** 2026-07-12
- **Primary cluster:** Pre-Tyche
- **Source branch:** `sna/nemo-2606-cutedsl-a2a-factorial-20260712`
- **Model:** `Qwen/Qwen3-30B-A3B`

## Purpose

Establish a functional two-node EP8 foundation before running the CuTeDSL,
full-iteration CUDA Graph, and A2A-overlap performance matrix. This stage does
not produce an accepted performance result. It proves that the exact future
benchmark topology can complete generation, refit, logprob, policy training,
and the next mature optimizer offload without host OOM.

This document supplements the 2026-07-10 integration design. Where topology or
experiment sequencing differs, this staged design is authoritative.

## Evidence Behind the Stage

Pre-Tyche job `2364903` validated the locked runtime, focused tests, Pyrefly,
four-GPU Transformer Engine smoke, CuTeDSL `GroupedGemmGluSm100`, and vLLM
level-2 discard. It completed one EP4 optimizer update, then failed at the next
`policy.offload_before_refit()` boundary:

- Slurm step `2364903.4`: `OUT_OF_MEMORY`, exit `0:125`.
- MaxRSS: `430794304K`.
- Policy-worker RSS immediately before the failed optimizer move:
  `77.110–78.106 GiB` per rank.
- vLLM discarded `107.79 GiB` per worker with no CPU weight backup.

The EP4 rank owned 8,788,850,688 parameters. With EP8, the local Qwen3 expert
shard falls to an estimated 5,164,972,032 parameters, 58.77% of the EP4 count.
The estimated Adam-state demand per node falls from about 261.9 GiB to
153.9 GiB. This supports an EP8 experiment, but only a real three-update run can
establish that it passes.

## Fixed Functional Topology

The functional gate uses the same policy topology and workload shape intended
for the timing matrix:

- two Pre-Tyche nodes, four GB200 GPUs per node;
- TP1, PP1, CP1, ETP1, EP8;
- sequence length 1024;
- GBS16 and MBS1, yielding two local microbatches;
- synchronous GRPO with eight prompts and two generations per prompt;
- colocated vLLM generation;
- MXFP8, grouped GEMM, TE op fuser, GLU interleave 32;
- CuTeDSL enabled;
- full-iteration CUDA Graph disabled;
- A2A overlap disabled;
- dynamic batching and sequence packing disabled;
- immutable image SHA
  `dd32f77a0a6fb09710e31f87402f0433413b9c71120fe893297e2f46e32ce8be`.

Both placement layers must describe the same two-node segment:

- Slurm submission: `--nodes=2 --segment=2`;
- NeMo-RL runtime config: `cluster.num_nodes=2`,
  `cluster.gpus_per_node=4`, `cluster.segment_size=2`.

The current recipe inherits `cluster.segment_size=4` from its parent. The
implementation must override it explicitly and add a runtime CLI override so a
future parent change cannot silently restore the mismatch.

## Functional-Mode Interface

The two-node launcher and common payload gain an explicit functional mode. It
is separate from accepted timing behavior.

```text
NEMO2606_FUNCTIONAL_GATE=1
NEMO2606_FACTORIAL_CONTEXT=g0a0
CUTEDSL_BENCHMARK_ORDER=on
CUTEDSL_BENCHMARK_PROFILE=0
NEMO2606_FUNCTIONAL_UPDATES=3
```

Functional mode must:

1. accept exactly one CuTeDSL-ON arm;
2. run exactly three GRPO updates;
3. reject full-CG and A2A selectors;
4. reject Nsight profiling;
5. emit `functional_gate=true` in the manifest and report;
6. exclude all functional metrics from replicate collection and performance
   acceptance.

Normal timing mode keeps the existing paired ON/OFF, warmup, measurement, and
replica contracts unchanged. A functional run cannot be relabeled as a timing
run after completion.

## Memory Evidence

Every policy offload event records:

- worker RSS and system-available memory;
- optimizer CUDA tensor bytes;
- cgroup `memory.current` and `memory.max` when exposed;
- the lifecycle action and EP rank.

The gate passes only if:

- all three updates complete;
- generation, refit, policy/reference logprob, and policy training occur on
  every update;
- the Step-2 mature optimizer offload emits `after_completion` on every rank;
- Slurm records no OOM or worker death;
- cgroup peak remains below 95% of `memory.max` when the limit is available;
- CuTeDSL activation is visible in runtime or retained kernel evidence;
- source, submodules, image, and effective topology match the manifest.

If the cgroup limit is unavailable, the run may pass the memory criterion only
with three completed updates, complete per-rank offload telemetry, and clean
Slurm accounting. No estimated memory value substitutes for the runtime gate.

## Failure and Fallback

Any failure stops performance submission and creates a bounded incident with
symptom, boundary evidence, root cause, tested change, and verification job.

If EP8 still fails from mature optimizer-state host capacity, the next design
uses synchronous non-colocated GRPO on four nodes: two EP8 policy nodes and two
vLLM inference nodes. That topology is reported as a separate experiment track
and is never compared directly with colocated E2E results.

`optimizer_cpu_offload=true`, reduced sequence length, reduced prompts, and
serialized CPU offload are not accepted fallbacks because they either alter
PolicyTraining performance or do not reduce the final optimizer-state bytes.

## Transition to Performance Measurement

Only a passing EP8 functional gate unlocks CuTeDSL and A2A timing runs. Before
the full eight-cell matrix:

- capture and replay an identical logprob/training workload for every cell;
- use complementary replicate-block ordering;
- run five warmups plus twenty measured updates and at least three replicas;
- run timing and Nsight profiling as separate Slurm jobs;
- attribute CuTeDSL kernels, CUDA Graph replay/CPU launch-gap reduction, and
  A2A compute/communication overlap independently;
- report E2E, generation, logprob, policy training, refit, and total-step
  throughput and latency.

Full-iteration CUDA Graph remains a separate integration gate because the
current implementation rejects colocated refit and logprob lifecycle changes.
The EP8 functional result does not claim full-CG support.

## Deliverables

- TDD coverage for recipe topology, functional-mode fail-closed behavior,
  manifest classification, and timing-mode non-regression.
- Signed feature-branch commit and push before submission.
- Pre-Tyche `sbatch --test-only` evidence followed by one monitored functional
  job.
- Bounded logs and a deterministic HTML incident or success record.
- No speedup claim until replicated timing and dedicated profiler evidence are
  both complete.
