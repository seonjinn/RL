# NeMo 26.06 EP8 Staged Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed two-node EP8 functional gate that completes three NeMo-RL GRPO updates with CuTeDSL enabled, captures host/optimizer memory evidence, and cannot be accepted as performance data.

**Architecture:** Extend the existing two-node factorial harness with a separate functional launcher and an explicit payload mode. Keep the timing path unchanged, classify functional evidence in the immutable manifest, and reject that evidence in the replicate collector. Add best-effort cgroup and optimizer CUDA telemetry at the existing optimizer-offload lifecycle boundaries.

**Tech Stack:** Bash/Slurm/Ray, NeMo-RL GRPO, OmegaConf/Pydantic config validation, Python 3.13, PyTorch, psutil, pytest, Ruff, Pyright.

## Global Constraints

- Functional topology is exactly two nodes × four GB200 GPUs, TP1/PP1/CP1/ETP1/EP8, sequence length 1024, GBS16, MBS1.
- Slurm uses `--nodes=2 --segment=2`; runtime config uses `cluster.num_nodes=2`, `cluster.gpus_per_node=4`, `cluster.segment_size=2`.
- Functional mode accepts only `g0a0`, CuTeDSL ON, full-iteration CUDA Graph OFF, A2A overlap OFF, profiling OFF, and exactly three updates.
- Functional manifests set `functional_gate=true` and are rejected by performance replicate collection.
- Timing mode retains paired ON/OFF, at least five warmups, at least ten measured updates in the payload, and at least three replicas/twenty measured updates in the factorial launcher.
- Telemetry is best-effort and must never change offload behavior or turn a successful training action into a failure.
- No performance claim is accepted from this functional gate.
- All commits are signed off and pushed only to `sna/nemo-2606-cutedsl-a2a-factorial-20260712` before cluster submission.

## File Structure

- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml`: authoritative EP8 runtime topology.
- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh`: existing two-node launcher with mutually exclusive functional and factorial modes.
- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`: shared runtime payload with mutually exclusive functional and timing modes.
- `nemo_rl/utils/host_memory.py`: typed, component-wise best-effort process/system/cgroup snapshot.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`: offload lifecycle fields and optimizer CUDA-byte sampling.
- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py`: performance acceptance boundary.
- `tests/test_nemo2606_multinode_factorial_harness.py`: recipe, launcher, payload, and collector contracts.
- `tests/unit/utils/test_host_memory.py`: cgroup parsing and unavailable-data behavior.
- `tests/unit/models/policy/test_megatron_worker.py`: optimizer telemetry and lifecycle ordering.

---

### Task 1: EP8 Topology and Functional Submission Interface

**Files:**
- Modify: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh`
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`

**Interfaces:**
- Consumes: existing cluster profile, `ray.sub`, and `run_cutedsl_matrix.sbatch`.
- Produces: environment contract `NEMO2606_FUNCTIONAL_GATE=1`, `NEMO2606_FUNCTIONAL_UPDATES=3`, `CUTEDSL_BENCHMARK_ORDER=on`, `CUTEDSL_BENCHMARK_PROFILE=0`, `NEMO2606_FACTORIAL_CONTEXT=g0a0`.

- [ ] **Step 1: Write failing recipe and launcher tests**

Add assertions that resolved `config["cluster"]["segment_size"] == 2` and a mocked-`sbatch` test that runs the existing launcher with `NEMO2606_FUNCTIONAL_GATE=1` and `--test-only`, records one call, and checks:

```python
assert call["functional_gate"] == "1"
assert call["functional_updates"] == "3"
assert call["context"] == "g0a0"
assert call["order"] == "on"
assert call["profile"] == "0"
assert "--nodes=2" in call["argv"]
assert "--segment=2" in call["argv"]
assert "--test-only" in call["argv"]
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/test_nemo2606_multinode_factorial_harness.py::test_multinode_recipe_has_ep8_and_two_local_microbatches \
  tests/test_nemo2606_multinode_factorial_harness.py::test_functional_submitter_exports_one_fail_closed_job
```

Expected: recipe assertion fails with inherited segment size `4`, and the functional-mode assertion fails because the launcher still submits the normal matrix.

- [ ] **Step 3: Override runtime segment size**

Add this exact recipe field:

```yaml
cluster:
  num_nodes: 2
  segment_size: 2
```

- [ ] **Step 4: Implement functional mode in the existing launcher**

Preserve the existing source pin, mounts, shared-fs canary, cluster profile, `ray.sub`, and export-file pattern. When `NEMO2606_FUNCTIONAL_GATE=1`, submit exactly one job and export these fixed values:

```bash
"CUTEDSL_BENCHMARK_EXISTING_RAY=1"
"CUTEDSL_BENCHMARK_NUM_NODES=2"
"CUTEDSL_BENCHMARK_GPUS_PER_NODE=4"
"CUTEDSL_BENCHMARK_ORDER=on"
"CUTEDSL_BENCHMARK_PROFILE=0"
"NEMO2606_FUNCTIONAL_GATE=1"
"NEMO2606_FUNCTIONAL_UPDATES=3"
"NEMO2606_FACTORIAL_CONTEXT=g0a0"
"NEMO2606_FULL_CG_ENABLED=0"
"NEMO2606_A2A_ENABLED=0"
```

The script continues to accept only zero arguments or `--test-only`, replaces the profile segment with `--segment=2`, appends `--nodes=2`, and writes a functional JSONL submission record only for a real submission. Normal mode must still emit four contexts × three replicas with alternating ON/OFF order. Export `CUTEDSL_BENCHMARK_SEGMENT_SIZE=2` in both modes so the payload records the effective scheduler segment instead of the cluster-profile default.

- [ ] **Step 5: Run focused tests and shell syntax checks**

Run:

```bash
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh
uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'recipe or functional_submitter'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-2n4g-megatron-mxfp8-factorial.yaml \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh \
  tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: add EP8 functional gate launcher"
```

### Task 2: Fail-Closed Functional Payload and Manifest

**Files:**
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Modify: `tests/test_nemo2606_multinode_factorial_harness.py`

**Interfaces:**
- Consumes: Task 1 environment contract.
- Produces: `benchmark_manifest.json` with `functional_gate: bool`, one `on` arm, `total_updates=3`, effective scheduler/runtime segment `2`, and no `timing_summary.json` or profiler output in functional mode.

- [ ] **Step 1: Write failing payload contract tests**

Add source-level contracts for all of these fail-closed checks:

```bash
[[ "${FUNCTIONAL_GATE}" == "0" || "${FUNCTIONAL_GATE}" == "1" ]]
[[ "${FUNCTIONAL_UPDATES}" == "3" ]]
[[ "${TIMING_ORDER}" == "on" ]]
[[ "${PROFILE_ENABLED}" == "0" ]]
[[ "${FEATURE_CONTEXT}" == "g0a0" ]]
[[ "${NEMO2606_FULL_CG_ENABLED}" == "0" ]]
[[ "${NEMO2606_A2A_ENABLED}" == "0" ]]
```

Also assert that timing-mode ON/OFF validation remains in the `else` branch and that `cluster.segment_size=${BENCHMARK_SEGMENT_SIZE}` is in `COMMON_OVERRIDES`. The topology and scheduler manifest must both use `BENCHMARK_SEGMENT_SIZE=2`, not raw `CUTEDSL_SEGMENT` from the cluster profile.

- [ ] **Step 2: Confirm RED**

Run:

```bash
uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py -k 'functional_payload or matrix_payload'
```

Expected: new functional payload contracts fail.

- [ ] **Step 3: Add functional controls and conditional validation**

Declare and export:

```bash
FUNCTIONAL_GATE="${NEMO2606_FUNCTIONAL_GATE:-0}"
FUNCTIONAL_UPDATES="${NEMO2606_FUNCTIONAL_UPDATES:-3}"
```

Validate `FUNCTIONAL_GATE` before branching. In functional mode, reject every value outside the fixed contract and set:

```bash
timing_arms=(on)
WARMUP_UPDATES=0
MEASURED_UPDATES=0
TOTAL_UPDATES="${FUNCTIONAL_UPDATES}"
```

In timing mode, retain the existing warmup, measured-update, paired-arm, and total-update validation without weakening thresholds.

- [ ] **Step 4: Make config validation mode-aware**

Always resolve and save the ON configuration. Resolve OFF and compute `matched_config_diff.json` only in timing mode. Validate the exact topology including `segment_size`, and emit:

```python
manifest = {
    "functional_gate": os.environ["FUNCTIONAL_GATE"] == "1",
    "performance_eligible": os.environ["FUNCTIONAL_GATE"] != "1",
    # existing immutable source/image/topology/config fields
}
```

In functional mode, set `fixed_config_evidence` and `arms` from ON only; keep the existing ON/OFF structures byte-compatible in timing mode.

- [ ] **Step 5: Run one functional arm without performance extraction**

Reuse `run_timing_arm` for the three-update GRPO process, but after a successful ON run skip the metric median/ratio extractor, timing summary construction, and Nsight section. Write `functional_gate_summary.json` containing:

```json
{
  "functional_gate": true,
  "completed_updates": 3,
  "performance_eligible": false,
  "arm": "on"
}
```

Derive `completed_updates` from the total-step TensorBoard series and fail unless it equals three. Require all component metric series for generation, refit, policy/reference logprob, and policy training to contain all three updates.

Collect bounded `megatron_policy_offload_memory` matches from both the driver `grpo.log` and `${RAY_CLUSTER_LOG_DIR}` because policy-worker stdout may live only in Ray logs. The summary must require `phase=after_completion offload_sequence=2` for all eight distinct global ranks and retain bounded CuTeDSL activation matches such as `GroupedGemmGluSm100`. An in-job summary records the evidence; final no-OOM acceptance still requires post-job Slurm accounting.

- [ ] **Step 6: Run payload tests and syntax check**

Run:

```bash
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch
uv run pytest -q tests/test_nemo2606_multinode_factorial_harness.py
```

Expected: all tests pass and timing-mode tests remain green.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch \
  tests/test_nemo2606_multinode_factorial_harness.py
git commit -s -m "feat: add fail-closed functional payload mode"
```

### Task 3: Best-Effort Cgroup and Optimizer CUDA Telemetry

**Files:**
- Modify: `nemo_rl/utils/host_memory.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Create: `tests/unit/utils/test_host_memory.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`

**Interfaces:**
- Consumes: cgroup v2 `/sys/fs/cgroup/memory.current` and `/sys/fs/cgroup/memory.max` when readable; `self.optimizer.state_dict()` tensor values.
- Produces: `HostMemorySnapshot(process_rss_gib, system_available_gib, cgroup_memory_current_gib, cgroup_memory_max_gib, cgroup_memory_peak_gib)` and event fields `global_rank`, `ep_rank`, `offload_sequence`, `lifecycle_action`, `optimizer_cuda_tensor_bytes`.

- [ ] **Step 1: Write failing cgroup snapshot tests**

Use a temporary cgroup tree and monkeypatch the module constants for `/proc/self/cgroup` and `/sys/fs/cgroup`. Cover a unified-cgroup line `0::/slurm/job`, numeric `memory.current=2147483648`, `memory.max=4294967296`, and `memory.peak=3221225472`; assert `2.0`, `4.0`, and `3.0` GiB. Add root-namespace fallback, `memory.max == "max"`, malformed input, and independent psutil failure cases. A psutil failure must preserve valid cgroup values; a cgroup failure must preserve RSS/system values.

- [ ] **Step 2: Confirm RED**

Run:

```bash
uv run pytest -q tests/unit/utils/test_host_memory.py
```

Expected: tests fail because cgroup fields are absent.

- [ ] **Step 3: Implement typed cgroup sampling**

Make every tuple field optional, make `_get_host_memory_snapshot()` always return a snapshot, and sample RSS, system memory, and cgroup independently. Parse the `0::<relative path>` cgroup-v2 entry, try the relative directory first, and fall back to the mounted root for container namespaces. Use this helper for `memory.current`, `memory.max`, and `memory.peak`:

```python
def _read_cgroup_gib(path: Path, *, allow_max: bool = False) -> float | None:
    try:
        raw = path.read_text().strip()
        if allow_max and raw == "max":
            return None
        value = int(raw)
        return value / _GIB if value >= 0 else None
    except (OSError, ValueError):
        return None
```

Append `cgroup_memory_current_gib`, `cgroup_memory_max_gib`, and `cgroup_memory_peak_gib` to every emitted line, formatting values to three decimals or `unavailable`. Delta formatting must tolerate optional RSS/system values. Keep the entire public emitter best-effort.

- [ ] **Step 4: Write failing optimizer-byte lifecycle test**

Build a fake optimizer state with one CUDA-marked tensor of 16 bytes, a duplicate reference to that tensor, and one CPU tensor. Call offload twice and assert each lifecycle line contains global rank, EP rank, lifecycle action, and monotonic `offload_sequence` values `1` then `2`; `before_optimizer_move` contains `optimizer_cuda_tensor_bytes=16` and `after_completion` contains `optimizer_cuda_tensor_bytes=0` after the fake move mutates the state. Add an inaccessible-state test expecting `unavailable` without disrupting lifecycle completion.

- [ ] **Step 5: Implement optimizer CUDA-byte sampling**

Add a typed module-level helper that inspects the optimizer's live `state` mapping, or callable `_get_state()` fallback, and recursively walks mapping/list/tuple values. Track visited container and tensor identities so shared tensors are counted once. Sum `numel() * element_size()` only when `torch.is_tensor(value)` and `value.is_cuda`; return zero for a valid empty/CPU state and `None` when state is inaccessible. Catch all inspection errors. Sample fresh fields at every event so the post-move value can become zero:

```python
fields={
    "global_rank": getattr(self, "rank", "unavailable"),
    "ep_rank": best_effort_expert_parallel_rank,
    "offload_sequence": self._nemo2606_offload_sequence,
    "lifecycle_action": "offload_before_refit",
    "optimizer_cuda_tensor_bytes": optimizer_cuda_tensor_bytes,
}
```

Render unavailable ranks and byte counts as the literal `unavailable`; do not materialize `state_dict()`, copy tensors, or call CUDA synchronization.

- [ ] **Step 6: Run telemetry tests**

Run:

```bash
uv run pytest -q \
  tests/unit/utils/test_host_memory.py \
  tests/unit/models/policy/test_megatron_worker.py -k 'host_memory or offload'
uv run ruff check nemo_rl/utils/host_memory.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/utils/test_host_memory.py \
  tests/unit/models/policy/test_megatron_worker.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add nemo_rl/utils/host_memory.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/utils/test_host_memory.py \
  tests/unit/models/policy/test_megatron_worker.py
git commit -s -m "feat: record optimizer offload memory evidence"
```

### Task 4: Performance-Acceptance Boundary

**Files:**
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py`
- Modify: `tests/test_cutedsl_replicate_collector.py`

**Interfaces:**
- Consumes: manifest boolean `functional_gate` and `performance_eligible`.
- Produces: `CollectorError("functional-gate evidence is not performance eligible")` before timing artifacts are loaded.

- [ ] **Step 1: Write the failing rejection test**

Start from the collector's valid replicate fixture, set:

```python
manifest["functional_gate"] = True
manifest["performance_eligible"] = False
```

Assert collection raises `CollectorError` matching `functional-gate evidence is not performance eligible` even if a timing summary is present.

- [ ] **Step 2: Confirm RED**

Run:

```bash
uv run pytest -q tests/test_cutedsl_replicate_collector.py -k functional_gate
```

Expected: the fixture is accepted or fails for an unrelated reason.

- [ ] **Step 3: Reject functional evidence at manifest validation**

At the start of manifest identity validation add:

```python
if manifest.get("functional_gate") is True or manifest.get("performance_eligible") is False:
    raise CollectorError(
        f"job {job_id} functional-gate evidence is not performance eligible"
    )
```

Keep legacy timing manifests eligible when both new fields are absent.

- [ ] **Step 4: Run collector regression tests**

Run:

```bash
uv run pytest -q tests/test_cutedsl_replicate_collector.py
uv run ruff check experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py \
  tests/test_cutedsl_replicate_collector.py
```

Expected: all collector tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py \
  tests/test_cutedsl_replicate_collector.py
git commit -s -m "test: exclude functional runs from performance collection"
```

### Task 5: Integrated Verification, Review, Push, and Pre-Tyche Gate

**Files:**
- Verify all files from Tasks 1–4.
- Update after the remote run: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html` and bounded evidence under its result directory.

**Interfaces:**
- Consumes: clean signed feature branch and immutable image SHA `dd32f77a0a6fb09710e31f87402f0433413b9c71120fe893297e2f46e32ce8be`.
- Produces: one monitored Pre-Tyche functional job and either a passing gate record or a bounded incident record.

- [ ] **Step 1: Run integrated local verification**

```bash
uv run pytest -q \
  tests/test_nemo2606_multinode_factorial_harness.py \
  tests/test_cutedsl_replicate_collector.py \
  tests/unit/utils/test_host_memory.py \
  tests/unit/models/policy/test_megatron_worker.py -k 'host_memory or offload or nemo2606 or cutedsl'
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh
uv run ruff check nemo_rl experiments/cutedsl_qwen3_30ba3b_oci_1n4g tests
uv run pyright nemo_rl/utils/host_memory.py nemo_rl/models/policy/workers/megatron_policy_worker.py
```

Expected: every command exits zero.

- [ ] **Step 2: Perform spec and code review**

Review against `docs/superpowers/specs/2026-07-12-nemo-2606-ep8-staged-validation-design.md`. Reject the implementation if timing mode changed thresholds/paired ordering, if functional mode can select OFF/full-CG/A2A/profile, if `segment_size=2` is absent from either scheduler or runtime config, or if collector accepts the functional manifest.

- [ ] **Step 3: Push only the feature branch**

```bash
git status --short
git log --format='%h %s%n%b' -4
git push fork sna/nemo-2606-cutedsl-a2a-factorial-20260712
```

Expected: clean status, every new commit contains `Signed-off-by`, and the feature ref updates without touching `main`.

- [ ] **Step 4: Synchronize the remote checkout and perform scheduler validation**

On Pre-Tyche, fetch the feature ref, hard-check that the checkout SHA equals the pushed SHA without changing another user's checkout, initialize recursive submodules, and run:

```bash
NEMO2606_FUNCTIONAL_GATE=1 CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh --test-only
```

Expected: exactly one test-only job description with `--nodes=2 --segment=2`; no job is queued.

- [ ] **Step 5: Submit and monitor for at least five minutes**

```bash
NEMO2606_FUNCTIONAL_GATE=1 CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_nemo2606_2n4g_factorial.sh
```

Poll queue state, bounded Slurm output, Ray worker health, and the event log for at least five minutes. Stop the performance pipeline immediately on image/source mismatch, Ray topology mismatch, worker death, CUDA OOM, host OOM, or missing optimizer-offload telemetry.

- [ ] **Step 6: Evaluate the functional pass contract**

Accept only when all three updates and all four components are present, every EP rank emits mature `after_completion`, Slurm accounting has no OOM, cgroup peak is below 95% when limited, and CuTeDSL activation is retained. Record source SHA, image SHA, topology, job ID, bounded log links, symptom/root-cause/fix/verification if failed, and explicitly label all functional metrics as non-performance evidence.

- [ ] **Step 7: Commit and push the deterministic report update**

```bash
git add experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results
git commit -s -m "docs: record EP8 functional gate result"
git push fork sna/nemo-2606-cutedsl-a2a-factorial-20260712
```

Expected: the report renders from bounded artifacts and makes no speedup claim.
