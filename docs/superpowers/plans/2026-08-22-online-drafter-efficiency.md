# Online Drafter Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove source-proven synchronization, copy, and collective overhead from Qwen3-8B online DFlash/DSpark training and refit while preserving one successful draft optimizer update and one applied draft refit on every online step.

**Architecture:** Begin only from a PR11 head with terminal GREEN exact-full and packed TP2 x CP2 DFlash-to-DSpark end-to-end receipts. Restack each optimization as a separate signed commit with a focused RED-to-GREEN gate, compose only accepted commits, then compare fixed, baseline-online, and optimized-online arms in rotated same-node triads. A source/call-count guard disqualifies any candidate whose new helpers execute on the fixed path, so every accepted triad has one unambiguous fixed control.

**Tech Stack:** Python >=3.13.14,<3.14, PyTorch, Megatron-Core, Ray, Hydra/OmegaConf, pytest, Torch profiler, Ruff through pre-commit, Pyrefly through pre-commit, SLURM/Pyxis on OCI-HSG GB200, Weights & Biases.

**Spec:** `docs/superpowers/specs/2026-08-22-online-drafter-efficiency-and-cadence-design.md`

## Global Constraints

- Do not create an implementation branch or begin Task 1 until the same PR11 commit has terminal GREEN receipts for the exact full gate and the packed TP2 x CP2 DFlash-to-DSpark end-to-end gate.
- Abort on any base-head drift; every implementation worktree, receipt, result, and report records the full 40-character product SHA and immutable container SHA256.
- Preserve one successful draft optimizer update and one applied post-update draft refit on every online step.
- Preserve fixed training, policy gradients, draft gradients, global normalization, optimizer/checkpoint state, exported tensor values, and generated-model tensor values.
- Keep `projected_streaming_vocab_parallel_soft_ce()` public and signature-compatible; the metadata-prevalidated bypass remains file-private and is reachable only from the already validated DFlash wrapper.
- The optional update probe performs no host transfer at probe start and exactly one packed device-to-host transfer per participating device when the whole optimizer-step probe is finalized.
- DFlash TP export completes fail-together manifest preflight before the first payload collective; a bucketing commit must never be pushed without that preflight.
- Cross-rank export metadata uses normalized device type and deterministic bucket ordinal, never a rank-local CUDA device index.
- Keep fixed-training source and resolved configuration unchanged; if any optimized helper executes on the fixed path, fail composition and do not run the performance matrix.
- Reproduce every optimization's RED on the terminal-GREEN final PR11 source in an isolated branch/worktree, preserve each accepted optimization as its own commit, and never cherry-pick a prior candidate commit without reproducing that RED.
- Do not adapt hidden capture commit `7169ab837` or change `torch.cuda.empty_cache()` unless the final-head profile and a counted RED test prove a remaining allocation/copy target and the existing peak-memory gate passes.
- Do not attribute runtime from operation counts; report counted operations and matched GPU timings separately.
- Project 1 success requires the exact primary reduction `(baseline_online_policy - optimized_online_policy) / (baseline_online_policy - fixed_policy) >= 0.20`, with a positive denominator, plus optimized-online E2E overhead at or below 5%.
- Generation non-inferiority requires lower paired 95% confidence bounds above -2% for canonical TPS, -1 percentage point for acceptance rate, and -0.1 token for accepted length.
- Every Python and shell product file outside `tests/` receives the 2026 NVIDIA copyright header; every new typed product module is listed in `pyrefly.toml`.
- Every commit uses `git commit -S -s`, passes `git verify-commit HEAD`, and is pushed before GPU submission; after push, GitHub must report `verified=true`.
- Before each GPU job, pull the exact pushed head, prove recursive worktree cleanliness and exact SHA, check FairShare, run `sbatch --test-only`, submit one job, and monitor it for at least five minutes.
- Runtime profiles, W&B exports, SLURM logs, and large result artifacts stay outside Git; Git contains schemas, launchers, analyzers, compact receipts, and Markdown conclusions.

## File Structure

- `nemo_rl/algorithms/loss/draft.py`: projected-loss validation and calculation; owns the private prevalidated core while the public generic API remains unchanged.
- `nemo_rl/algorithms/loss/wrapper.py`: synchronous versus deferred draft-loss metric materialization boundary.
- `nemo_rl/models/megatron/draft/perf_counters.py`: opt-in counters and NVTX regions whose disabled path performs no tensor operation or CUDA synchronization.
- `nemo_rl/models/megatron/draft/diagnostics.py`: device-resident whole-step update-probe statistics and reconstruction of the unchanged public `DraftUpdateResult`.
- `nemo_rl/models/megatron/draft/step_state.py`: split-step accumulation, one cached float32 reciprocal normalization scalar, and draft-gradient correction.
- `nemo_rl/models/megatron/draft/utils.py`: ordered DFlash export entries, normalized bucket metadata, manifest preflight, bucket payload gathering, and reconstruction behind the unchanged public export signature.
- `nemo_rl/models/megatron/draft/hidden_capture.py`, `nemo_rl/models/megatron/draft/training.py`, and `nemo_rl/models/policy/workers/megatron_policy_worker.py`: named profiling regions around the existing critical-path boundaries; no algorithm ownership moves into these files.
- `pyrefly.toml`: explicit inclusion for every new typed product and research module.
- `tests/unit/algorithms/test_dflash_metadata_performance.py`: DFlash wrapper and public-generic metadata validation call counts.
- `tests/unit/algorithms/test_dflash_projected_loss.py` and `tests/unit/distributed/test_projected_draft_soft_ce.py`: numerical parity and asymmetric-rank fail-together behavior.
- `tests/unit/algorithms/test_draft_loss_wrapper.py`: deferred metric type and scalar-materialization counts.
- `tests/unit/models/megatron/test_draft_step_state.py`: cached normalization, invalidation, CP count, and empty-owner behavior.
- `tests/unit/models/megatron/test_draft_diagnostics.py` and `tests/unit/models/megatron/test_draft_update_probe.py`: whole-step transfer count, multi-device grouping, public-result parity, and fail-loud invariants.
- `tests/unit/models/megatron/test_draft_perf_counters.py`: disabled-zero-cost and enabled counter behavior.
- `tests/unit/models/megatron/test_dflash_export_contract.py` and `tests/unit/models/megatron/test_draft_refit.py`: real TP2 payload-count, preflight ordering, mismatch, empty-owner, order, shape, dtype, and value parity.
- `research/qwen3_8b_online_drafter_efficiency/base_contract.py`: validates and records immutable final-head prerequisite receipts.
- `research/qwen3_8b_online_drafter_efficiency/base_contract.json`: compact committed base evidence populated only after both prerequisite gates are terminal GREEN.
- `research/qwen3_8b_online_drafter_efficiency/profile_contract.py`: profile-receipt schema and fail-closed analyzer.
- `research/qwen3_8b_online_drafter_efficiency/compose_guard.py`: approved-file diff, fixed-config byte parity, and fixed-path zero-call qualification.
- `research/qwen3_8b_online_drafter_efficiency/launch_matrix.py`: rotated same-node allocation plan and fail-closed SLURM command generation.
- `research/qwen3_8b_online_drafter_efficiency/analyze_matrix.py`: closed-window paired statistics and frozen pass/fail formulas.
- `research/qwen3_8b_online_drafter_efficiency/run_profile.sh` and `research/qwen3_8b_online_drafter_efficiency/run_matrix.sh`: reproducible OCI-HSG entrypoints.
- `research/qwen3_8b_online_drafter_efficiency/tests/`: base, profile, compose, launcher, and analyzer contract tests.
- `research/qwen3_8b_online_drafter_efficiency/README.md`: exact commands, supported topology, artifact layout, and claim boundaries.
- `research/qwen3_8b_online_drafter_efficiency/HANDOFF.md`: exact-head correctness, performance, signing, residual-risk, and separate Claude Code review inputs.

---

### Task 1: Freeze the terminal-GREEN PR11 base

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/base_contract.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/base_contract.json`
- Create: `research/qwen3_8b_online_drafter_efficiency/tests/test_base_contract.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/README.md`

**Interfaces:**

- Consumes: `NRL_EFFICIENCY_FULL_GATE_RECEIPT`, `NRL_EFFICIENCY_PACKED_E2E_RECEIPT`, and `NRL_EFFICIENCY_CONTAINER_SHA256`; each receipt is JSON with `job_id: int`, `head: str`, `result: "PASS"`, and `result_path: str`.
- Produces: `GateReceipt`, `BaseContract`, `load_base_contract(path: Path) -> BaseContract`, `assert_terminal_green_base(contract: BaseContract, *, current_head: str) -> None`, and the exact `base_contract.json` consumed by Tasks 2, 4, 11, 12, and 13.

- [ ] **Step 1: Write the failing base-contract test.** Add this exact contract case to `research/qwen3_8b_online_drafter_efficiency/tests/test_base_contract.py`:

```python
import json
from pathlib import Path

import pytest

from research.qwen3_8b_online_drafter_efficiency.base_contract import (
    assert_terminal_green_base,
    load_base_contract,
)


def test_base_requires_two_green_receipts_on_the_current_head(tmp_path: Path) -> None:
    head = "a" * 40
    path = tmp_path / "base_contract.json"
    path.write_text(
        json.dumps(
            {
                "product_head": head,
                "container_sha256": "b" * 64,
                "full_gate": {
                    "job_id": 1,
                    "head": head,
                    "result": "PASS",
                    "result_path": "/durable/full/result.json",
                },
                "packed_e2e": {
                    "job_id": 2,
                    "head": head,
                    "result": "PASS",
                    "result_path": "/durable/packed/result.json",
                },
            }
        )
    )
    contract = load_base_contract(path)
    assert_terminal_green_base(contract, current_head=head)
    with pytest.raises(RuntimeError, match="base head drift"):
        assert_terminal_green_base(contract, current_head="c" * 40)
```

- [ ] **Step 2: Run RED and verify the missing module failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_base_contract.py`; expect collection to fail with `ModuleNotFoundError: No module named 'research.qwen3_8b_online_drafter_efficiency.base_contract'`.

- [ ] **Step 3: Add the minimal typed loader and validator.** Implement these concrete types and checks in `base_contract.py`, with `Literal["PASS"]` for `GateReceipt.result`, `re.fullmatch(r"[0-9a-f]{40}", head)`, `re.fullmatch(r"[0-9a-f]{64}", container_sha256)`, nonempty durable result paths, equal receipt/product heads, and the exact drift error:

```python
@dataclass(frozen=True, slots=True)
class GateReceipt:
    job_id: int
    head: str
    result: Literal["PASS"]
    result_path: str


@dataclass(frozen=True, slots=True)
class BaseContract:
    product_head: str
    container_sha256: str
    full_gate: GateReceipt
    packed_e2e: GateReceipt


def assert_terminal_green_base(
    contract: BaseContract,
    *,
    current_head: str,
) -> None:
    if current_head != contract.product_head:
        raise RuntimeError(
            f"base head drift: current={current_head} recorded={contract.product_head}"
        )
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_base_contract.py`; expect `1 passed`.

- [ ] **Step 5: Record the real base and validate it.** Run `uv run research/qwen3_8b_online_drafter_efficiency/base_contract.py record --full-gate-receipt "${NRL_EFFICIENCY_FULL_GATE_RECEIPT:?required}" --packed-e2e-receipt "${NRL_EFFICIENCY_PACKED_E2E_RECEIPT:?required}" --container-sha256 "${NRL_EFFICIENCY_CONTAINER_SHA256:?required}" --output research/qwen3_8b_online_drafter_efficiency/base_contract.json`, then run `uv run research/qwen3_8b_online_drafter_efficiency/base_contract.py validate --contract research/qwen3_8b_online_drafter_efficiency/base_contract.json --current-head "$(git rev-parse HEAD)"`; expect `base_contract=PASS`.

- [ ] **Step 6: Run static checks.** Run `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/base_contract.py research/qwen3_8b_online_drafter_efficiency/tests/test_base_contract.py research/qwen3_8b_online_drafter_efficiency/README.md`; expect all hooks to pass.

- [ ] **Step 7: Commit the exact base receipt.** Run `git add research/qwen3_8b_online_drafter_efficiency && git commit -S -s -m "test(draft): pin online efficiency base evidence" && git verify-commit HEAD`.

### Task 2: Add disabled-zero-cost performance counters

**Files:**

- Create: `nemo_rl/models/megatron/draft/perf_counters.py`
- Create: `tests/unit/models/megatron/test_draft_perf_counters.py`
- Modify: `pyrefly.toml:222-236`

**Interfaces:**

- Consumes: `NRL_DRAFT_PERF_PROFILE` and explicit source-side `count_draft_perf(name: str, *, calls: int = 1, num_bytes: int = 0) -> None` calls.
- Produces: `DraftPerfSnapshot.to_json() -> str`, `DraftPerfSink.from_env(global_rank: int) -> DraftPerfSink | None`, `begin_draft_perf_step(step: int, *, microbatches: int) -> None`, `finish_draft_perf_step(step: int) -> DraftPerfSnapshot`, `abort_draft_perf_step() -> None`, `draft_perf_region(name: str) -> AbstractContextManager[None]`, and `count_draft_perf(name: str, *, calls: int = 1, num_bytes: int = 0) -> None`. When enabled, `begin_draft_perf_step()` enters `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=False, profile_memory=True)`; `draft_perf_region()` nests `torch.profiler.record_function(name)` with NVTX. Successful finish exits the profiler, derives region seconds from matching `prof.key_averages()` events, records CUDA peak allocated/reserved bytes, writes `$NRL_DRAFT_PERF_OUTPUT_DIR/rank-$global_rank/step-$step.trace.json`, and appends one fsync'd rank-qualified JSONL row. Abort exits without a completed row and removes any partial trace.

- [ ] **Step 1: Write the failing zero-cost and sink tests.** Add a `TorchDispatchMode` test that calls disabled begin/region/counter/finish and asserts zero calls for `_local_scalar_dense`, `clone`, `cat`, `all_gather`, `broadcast`, and `all_reduce`. With `NRL_DRAFT_PERF_PROFILE=1` and a temporary output directory, monkeypatch `torch.profiler.profile` and CUDA peak-memory getters, run global rank 3 / step 7 / two microbatches, and assert exactly `rank-3/step-7.trace.json`, one `rank-3/counters.jsonl` row equal to `json.loads(snapshot.to_json())`, region timing/peak fields, and `calls == {"metadata_collective": 2}` / `bytes == {"metadata_collective": 64}`. Run an aborted step and assert no second completed row or partial trace.

- [ ] **Step 2: Run RED and verify the missing module failure.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_perf_counters.py`; expect collection to fail because `perf_counters.py` does not exist.

- [ ] **Step 3: Implement the minimal counter, profiler, and sink state.** Use one `ContextVar[_DraftPerfStep | None]`; return `nullcontext()` when disabled; create the profiler and output directory only in `begin_draft_perf_step()` when both environment variables are present. `finish_draft_perf_step()` closes the profiler before exporting the trace and appends `snapshot.to_json() + "\n"` through `DraftPerfSink`. Keep all counter arithmetic in Python integers:

```python
@dataclass(frozen=True, slots=True)
class DraftPerfSnapshot:
    global_rank: int
    step: int
    microbatches: int
    region_seconds: dict[str, float]
    calls: dict[str, int]
    bytes: dict[str, int]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def count_draft_perf(
    name: str,
    *,
    calls: int = 1,
    num_bytes: int = 0,
) -> None:
    state = _COUNTERS.get()
    if state is None:
        return
    old_calls, old_bytes = state.get(name, (0, 0))
    state[name] = (old_calls + calls, old_bytes + num_bytes)
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_perf_counters.py`; expect both tests to pass.

- [ ] **Step 5: Register and statically check the module.** Add `"nemo_rl/models/megatron/draft/perf_counters.py",` to `pyrefly.toml` adjacent to the other draft modules, then run `uv run --group dev pre-commit run --files nemo_rl/models/megatron/draft/perf_counters.py tests/unit/models/megatron/test_draft_perf_counters.py pyrefly.toml`; expect all hooks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/models/megatron/draft/perf_counters.py tests/unit/models/megatron/test_draft_perf_counters.py pyrefly.toml && git commit -S -s -m "perf(draft): add zero-cost hot-path counters" && git verify-commit HEAD`.

### Task 3: Instrument the existing final-head critical-path boundaries

**Files:**

- Modify: `nemo_rl/algorithms/loss/draft.py`
- Modify: `nemo_rl/algorithms/loss/wrapper.py`
- Modify: `nemo_rl/models/megatron/draft/hidden_capture.py`
- Modify: `nemo_rl/models/megatron/draft/diagnostics.py`
- Modify: `nemo_rl/models/megatron/draft/training.py`
- Modify: `nemo_rl/models/megatron/draft/step_state.py:89-161`
- Modify: `nemo_rl/models/megatron/draft/utils.py:1836-1897`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py:1740-1935`
- Modify: `tests/unit/models/megatron/test_draft_perf_counters.py`

**Interfaces:**

- Consumes: Task 2 `draft_perf_region()` and `count_draft_perf()`.
- Produces: the stable region names `draft.hidden_capture`, `draft.provider_forward`, `draft.loss_backward`, `draft.finish_normalization`, `draft.optimizer_finalize`, `draft.export_reconstruct`, and `draft.refit_transfer`; operation names `scalar_materialization`, `tensor_materialization`, `metadata_collective`, and `refit_payload_collective`; and one rank-local JSON `DraftPerfSnapshot` plus profiler trace per completed policy step. `begin_train_step()` calls `begin_draft_perf_step()` before the first split microbatch, `train()` does the same before the monolithic path, both successful finish paths call `finish_draft_perf_step()`, and abort calls `abort_draft_perf_step()`.

- [ ] **Step 1: Write the failing integration-contract test.** Extend `test_draft_perf_counters.py` to inspect all eight producer files. Assert projected metadata counters occur in `draft.py`, deferred metric counters in `wrapper.py`, probe transfers in `diagnostics.py`, and export collectives in `utils.py`. Exercise both `train()` and `begin_train_step()` -> `finish_train_step()` with spies and assert each begins and finishes one matching step; `abort_train_step()` must call abort and emit none. Assert the sync and split optimizer-finalize regions separately instead of accepting one source occurrence.

- [ ] **Step 2: Run RED and verify missing producer/lifecycle assertions.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_perf_counters.py -k 'integration or step_snapshot'`; expect failures naming the absent producer counters and both missing reset/snapshot lifecycles.

- [ ] **Step 3: Add the minimal named regions, producer counters, and per-step lifecycle.** Wrap only existing boundaries and place counters immediately beside the operation they count. Call `begin_draft_perf_step(global_step)` at monolithic and split openings, `finish_draft_perf_step(global_step)` only after successful optimizer finish, and `abort_draft_perf_step()` on every exception/abort path. The Task 2 sink owns serialization and trace export. Preserve return values and exception behavior. Count bytes with `tensor.numel() * tensor.element_size()` before the measured collective:

```python
with draft_perf_region("draft.export_reconstruct"):
    count_draft_perf(
        "refit_payload_collective",
        num_bytes=tensor.numel() * tensor.element_size(),
    )
    tensor = _gather_tp_weight_if_needed(
        tensor,
        logical_shape,
        split_axis=split_axis,
    )
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_perf_counters.py`; expect producer placement, sync/split reset, completed-step serialization, abort suppression, and disabled counter tests to pass.

- [ ] **Step 5: Run focused regressions and static checks.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_draft_hidden_capture.py unit/models/megatron/test_draft_step_state.py unit/models/megatron/test_draft_refit.py --hf-gated --mcore-only`, return to the repository root, and run `uv run --group dev pre-commit run --files nemo_rl/algorithms/loss/draft.py nemo_rl/algorithms/loss/wrapper.py nemo_rl/models/megatron/draft/diagnostics.py nemo_rl/models/megatron/draft/hidden_capture.py nemo_rl/models/megatron/draft/training.py nemo_rl/models/megatron/draft/step_state.py nemo_rl/models/megatron/draft/utils.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_draft_perf_counters.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/algorithms/loss/draft.py nemo_rl/algorithms/loss/wrapper.py nemo_rl/models/megatron/draft/diagnostics.py nemo_rl/models/megatron/draft/hidden_capture.py nemo_rl/models/megatron/draft/training.py nemo_rl/models/megatron/draft/step_state.py nemo_rl/models/megatron/draft/utils.py nemo_rl/models/policy/workers/megatron_policy_worker.py tests/unit/models/megatron/test_draft_perf_counters.py && git commit -S -s -m "perf(draft): instrument online critical path" && git verify-commit HEAD`.

### Task 4: Build the fail-closed profile receipt and analyzer

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/profile_contract.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/run_profile.sh`
- Create: `research/qwen3_8b_online_drafter_efficiency/tests/test_profile_contract.py`
- Modify: `research/qwen3_8b_online_drafter_efficiency/README.md`

**Interfaces:**

- Consumes: Task 1 semantic `BaseContract`; the distinct Tasks 2-4 `instrumentation_head`; Task 3 rank-qualified per-step snapshots/traces; canonical W&B rows; immutable fixed and online resolved YAML files; job ID; run ID; container SHA256; and packing flag.
- Produces: `ProfileReceipt(semantic_base_head: str, instrumentation_head: str, world_size: int, steps: list[ProfileStep])`, `analyze_profile(receipt: ProfileReceipt) -> dict[str, object]`, and compact `profile_summary.json`/`profile_summary.md` with a complete global-rank x step grid, per-step/per-microbatch region timing, operation counts, transferred bytes, and peak allocated/reserved memory. Both baseline and optimized comparisons retain identical instrumentation commits; `semantic_base_head` remains the terminal PR11 prerequisite.

- [ ] **Step 1: Write the failing analyzer and launcher-contract tests.** Create a two-rank, two-step fixture with two microbatches per step and assert `analyze_profile()` returns the expected region/microbatch timing, exact operation totals/bytes, and peak maxima without double-counting replica-local work. Add failures for semantic-base or instrumentation-head mismatch, a missing/duplicate rank-step cell, non-finite timing, an online arm without `policy.draft.enabled=true`, a fixed arm without `policy.draft.enabled=false`, held-field mismatch, and any rank-step missing either trace or JSONL data.

- [ ] **Step 2: Run RED and verify the missing analyzer failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_profile_contract.py`; expect collection to fail because `profile_contract.py` does not exist.

- [ ] **Step 3: Implement the minimal typed analyzer and exact launcher contract.** Use the exact row type below and arithmetic means over a complete rank x step grid. `run_profile.sh` requires `NRL_EFFICIENCY_INSTRUMENTED_SOURCE_ROOT`, semantic base receipt, fixed/online configs, and expected world size; it verifies the source root equals the committed Tasks 2-4 `instrumentation_head`, whose first parent chain contains the semantic PR11 base. It resolves configs before allocation and permits only `policy.draft.enabled=false` versus `true`. Each arm invokes the explicit instrumented source entrypoint, sets `NRL_DRAFT_PERF_OUTPUT_DIR="$arm_root/profile"`, and requires `rank-$rank/counters.jsonl` plus `rank-$rank/step-$step.trace.json` for every rank-step cell in rows 5..29. Any arm/rank/artifact failure aborts the allocation.

```python
@dataclass(frozen=True, slots=True)
class ProfileStep:
    global_rank: int
    step: int
    microbatches: int
    step_seconds: float
    region_seconds: dict[str, float]
    operation_calls: dict[str, int]
    operation_bytes: dict[str, int]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_profile_contract.py`; expect all complete and fail-closed cases to pass.

- [ ] **Step 5: Run static checks.** Run `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/profile_contract.py research/qwen3_8b_online_drafter_efficiency/run_profile.sh research/qwen3_8b_online_drafter_efficiency/tests/test_profile_contract.py research/qwen3_8b_online_drafter_efficiency/README.md`; expect all hooks to pass.

- [ ] **Step 6: Commit.** Run `git add research/qwen3_8b_online_drafter_efficiency/profile_contract.py research/qwen3_8b_online_drafter_efficiency/run_profile.sh research/qwen3_8b_online_drafter_efficiency/tests/test_profile_contract.py research/qwen3_8b_online_drafter_efficiency/README.md && git commit -S -s -m "perf(draft): add final-head profile contract" && git verify-commit HEAD`.

- [ ] **Step 7: Run the instrumented baseline profile before selecting optimizations.** Push Tasks 2-4 and record that exact SHA as `instrumentation_head`; run `/submit research/qwen3_8b_online_drafter_efficiency/run_profile.sh --cluster=oci-hsg` from that head, require every `sbatch --test-only` probe to pass, and monitor five times at 60-second cadence. Analyze the baseline root and require complete fixed/online packing-disabled and packing-enabled rows 5..29, a complete rank x step artifact grid, exact instrumentation SHA/container/config parity, and semantic-base ancestry. Tasks 5-10 may begin only after this receipt is terminal GREEN. Optimized profiling must apply only optimization commits on top of the same instrumentation head.

### Task 5: Validate DFlash projected-loss metadata exactly once

**Files:**

- Modify: `nemo_rl/algorithms/loss/draft.py:647-934`
- Create: `tests/unit/algorithms/test_dflash_metadata_performance.py`
- Modify: `tests/unit/algorithms/test_dflash_projected_loss.py`
- Modify: `tests/unit/distributed/test_projected_draft_soft_ce.py`

**Interfaces:**

- Consumes: existing `_tp_assert_projected_metadata_agreement()`, existing `DraftLossStats`, and Task 2 `count_draft_perf()`.
- Produces: private `_projected_streaming_vocab_parallel_soft_ce(*, student_hidden: torch.Tensor, output_weight: torch.Tensor, selected_teacher_logits: torch.Tensor, mask: torch.Tensor, token_chunk_size: int, tp_group: torch.distributed.ProcessGroup | None, bin_ids: torch.Tensor | None = None, weights: torch.Tensor | None = None, metadata_prevalidated: bool) -> DraftLossStats`; public `projected_streaming_vocab_parallel_soft_ce()` keeps its current signature and always passes `metadata_prevalidated=False`.

The counted RED target is the redundant inner pass's one header `all_gather`, three broadcasts, three mismatch `all_reduce` calls, three mismatch scalar materializations, and three contiguous metadata clones per DFlash draft microbatch. The test reports these as source operations and does not convert them into a runtime claim.

- [ ] **Step 1: Write the failing validation-count tests.** Port the deterministic fixture from commit `15fc942fa62d18e6a0a013639ab2dbf9cbeaf882`, assert the DFlash wrapper calls `_tp_assert_projected_metadata_agreement` once, and add a direct generic call asserting one validation; retain a real TP2 asymmetric `sample_rows`, `label_positions`, and `loss_mask` test where both ranks raise before loss collectives.

- [ ] **Step 2: Run RED and verify the duplicate count.** Run `uv run pytest -q tests/unit/algorithms/test_dflash_metadata_performance.py tests/unit/algorithms/test_dflash_projected_loss.py`; expect `test_dflash_adapter_validates_tp_metadata_once` to fail with `2 == 1` while the direct generic case reports one call.

- [ ] **Step 3: Implement the minimal private bypass.** Rename the current implementation to the exact private signature above, guard its current agreement call with `if not metadata_prevalidated`, restore the public wrapper with its unchanged explicit parameters, and call the private core from `dflash_projected_vocab_parallel_soft_ce()` only after that wrapper's existing full agreement completes; increment `metadata_collective` only beside executed agreement collectives.

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/algorithms/test_dflash_metadata_performance.py tests/unit/algorithms/test_dflash_projected_loss.py`; expect the DFlash and direct-generic call counts to equal one.

- [ ] **Step 5: Run TP2 fail-together and static checks.** Run `cd tests && CUDA_VISIBLE_DEVICES=0,1 uv run --extra mcore pytest -q unit/distributed/test_projected_draft_soft_ce.py --hf-gated --mcore-only`, return to the repository root, and run `uv run --group dev pre-commit run --files nemo_rl/algorithms/loss/draft.py tests/unit/algorithms/test_dflash_metadata_performance.py tests/unit/algorithms/test_dflash_projected_loss.py tests/unit/distributed/test_projected_draft_soft_ce.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/algorithms/loss/draft.py tests/unit/algorithms/test_dflash_metadata_performance.py tests/unit/algorithms/test_dflash_projected_loss.py tests/unit/distributed/test_projected_draft_soft_ce.py && git commit -S -s -m "perf(dflash): validate projected metadata once" && git verify-commit HEAD`.

### Task 6: Defer split draft-loss metric scalar materialization

**Files:**

- Modify: `nemo_rl/algorithms/loss/wrapper.py:313-346`
- Modify: `tests/unit/algorithms/test_draft_loss_wrapper.py:204-251`

**Interfaces:**

- Consumes: `DraftLossWrapper.defer_normalization` and the existing `metrics: dict[str, Any]` returned by `DraftLossWrapper.__call__()`.
- Produces: `metrics["draft_loss"]: torch.Tensor` as a detached zero-dimensional tensor only when `defer_normalization=True`; non-deferred calls continue to produce `float`.

- [ ] **Step 1: Write the failing scalar-materialization test.** Wrap the existing split-step test call in a CPU Torch profiler, assert `isinstance(metrics["draft_loss"], torch.Tensor)`, assert it is detached and zero-dimensional, and assert the `_local_scalar_dense` count inside the wrapper call is zero; add a synchronous case asserting `float`.

- [ ] **Step 2: Run RED and verify one forced scalar.** Run `uv run pytest -q tests/unit/algorithms/test_draft_loss_wrapper.py -k 'defers_raw_stats or synchronous_metric'`; expect the deferred type assertion to fail and the profiler to report one `_local_scalar_dense` call.

- [ ] **Step 3: Implement the minimal mode-specific assignment.** Replace line 345 with this exact expression and leave combined-loss calculation and payload transport unchanged:

```python
metrics["draft_loss"] = (
    draft_loss.detach()
    if self.defer_normalization
    else float(draft_loss.detach().item())
)
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/algorithms/test_draft_loss_wrapper.py -k 'defers_raw_stats or synchronous_metric'`; expect zero deferred scalar materializations and a synchronous Python float.

- [ ] **Step 5: Run the complete wrapper suite and static checks.** Run `uv run pytest -q tests/unit/algorithms/test_draft_loss_wrapper.py` and `uv run --group dev pre-commit run --files nemo_rl/algorithms/loss/wrapper.py tests/unit/algorithms/test_draft_loss_wrapper.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/algorithms/loss/wrapper.py tests/unit/algorithms/test_draft_loss_wrapper.py && git commit -S -s -m "perf(draft): defer split loss metric sync" && git verify-commit HEAD`.

### Task 7: Cache one finalized normalization scalar per draft step

**Files:**

- Modify: `nemo_rl/models/megatron/draft/step_state.py:34-161`
- Modify: `tests/unit/models/megatron/test_draft_step_state.py`
- Modify: `tests/unit/algorithms/test_dflash_cp_zero_owner.py`

**Interfaces:**

- Consumes: `DraftStepState.set_global_counts()`, `normalize_metric()`, and `correct_main_grads()`.
- Produces: private field `_normalization_scale_value: float | None` and cached `DraftStepState._normalization_scale() -> float`; `set_global_counts()` invalidates the cache whenever finalized global counts are replaced.

For `M` split draft microbatches, Tasks 6-7 target the source-proven `3M + 2` forced scalar materializations without changing the numerator, denominator, or gradient-correction formulas.

- [ ] **Step 1: Write the failing cache and invalidation tests.** First profile three `normalize_metric()` calls against one finalized count set and require one scale-related `_local_scalar_dense`. Separately profile one `correct_main_grads()` call after the cache is warm and require one independent `_local_scalar_dense` for `policy_normalization_count` and zero additional scale materializations. Then call `set_global_counts()` with new CP2/CP4 values, require exactly one new scale materialization, and verify the new reciprocal; include empty-bin and zero-owner finite-zero assertions.

- [ ] **Step 2: Run RED and verify repeated materializations.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_step_state.py -k 'normalization_scalar or invalidates or empty_draft_bins'`; expect more than one `_local_scalar_dense` in the reuse case.

- [ ] **Step 3: Implement the minimal float32 reciprocal cache.** Add the field, assign `None` in `set_global_counts()`, and implement this exact finite-zero calculation without changing `correct_main_grads()` arithmetic:

```python
def _normalization_scale(self) -> float:
    if self._normalization_scale_value is not None:
        return self._normalization_scale_value
    if self._global_counts is None or self._weights is None:
        raise RuntimeError("global draft counts have not been finalized")
    denominator = (
        self._global_counts.to(dtype=torch.float32)
        * self._weights.to(device=self._global_counts.device, dtype=torch.float32)
    ).sum()
    scale = torch.where(
        denominator > 0,
        denominator.reciprocal(),
        torch.zeros_like(denominator),
    )
    self._normalization_scale_value = float(scale.item())
    return self._normalization_scale_value
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_step_state.py -k 'normalization_scalar or invalidates or empty_draft_bins'`; expect one scale materialization per finalized count set, one separate policy-count materialization in gradient correction, no scale rematerialization after warmup, and finite zero for empty/zero ownership.

- [ ] **Step 5: Run parity, zero-owner, and static checks.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_step_state.py tests/unit/algorithms/test_dflash_cp_zero_owner.py` and `uv run --group dev pre-commit run --files nemo_rl/models/megatron/draft/step_state.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/algorithms/test_dflash_cp_zero_owner.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/models/megatron/draft/step_state.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/algorithms/test_dflash_cp_zero_owner.py && git commit -S -s -m "perf(draft): cache finalized normalization scalar" && git verify-commit HEAD`.

### Task 8: Batch the optional update probe into one whole-step transfer

**Files:**

- Modify: `nemo_rl/models/megatron/draft/diagnostics.py:21-132`
- Modify: `tests/unit/models/megatron/test_draft_update_probe.py`
- Create: `tests/unit/models/megatron/test_draft_diagnostics.py`

**Interfaces:**

- Consumes: unchanged `start_draft_update_probe(module: nn.Module) -> DraftUpdateProbe` and `finalize_draft_update_probe(module: nn.Module, probe: DraftUpdateProbe) -> DraftUpdateResult` call sites in the sync and split workers.
- Produces: `DraftUpdateProbe.before_by_device: Mapping[torch.device, torch.Tensor]`, private `_module_statistics_by_device(module: nn.Module, *, include_gradients: bool) -> dict[torch.device, torch.Tensor]`, private `_copy_statistics_to_host(statistics: torch.Tensor) -> list[float]`, and the unchanged public fields of `DraftUpdateResult`.

- [ ] **Step 1: Write the failing whole-step transfer test.** Spy on `_copy_statistics_to_host()` to assert start invokes it zero times and finalize invokes it once for a CPU-only multi-parameter model; add a CUDA-gated CPU+CUDA model asserting one finalize invocation per participating device and use Torch profiler to confirm one device-to-host copy for the CUDA statistics tensor, plus value parity with the current checksum and gradient formulas.

- [ ] **Step 2: Run RED and verify start-time transfers.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_diagnostics.py tests/unit/models/megatron/test_draft_update_probe.py`; expect start to exceed zero transfers because current `_parameter_checksum()` and `_gradient_l2()` call `.item()` per statistic.

- [ ] **Step 3: Implement device-resident start and packed finalize.** Accumulate `[parameter_sum, parameter_l2_squared, gradient_l2_squared]` tensors by each value's actual device, retain the detached fixed-size tensors in `before_by_device`, calculate after statistics without gradients, pack before and after for each sorted device, call `_copy_statistics_to_host()` once per device only in finalize, and reconstruct `DraftUpdateResult` by summing host scalars and square-rooting the total gradient square.

```python
@dataclass(frozen=True, slots=True)
class DraftUpdateProbe:
    before_by_device: Mapping[torch.device, torch.Tensor]


def _device_sort_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else device.index


def _copy_statistics_to_host(statistics: torch.Tensor) -> list[float]:
    return statistics.cpu().tolist()
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q tests/unit/models/megatron/test_draft_diagnostics.py tests/unit/models/megatron/test_draft_update_probe.py`; expect zero start transfers, one finalize transfer per device, and unchanged public result values.

- [ ] **Step 5: Run worker probe and static checks.** Run `cd tests && uv run --extra mcore pytest -q unit/models/policy/test_dflash_worker_validation.py unit/models/megatron/test_draft_update_probe.py --hf-gated --mcore-only`, return to the repository root, and run `uv run --group dev pre-commit run --files nemo_rl/models/megatron/draft/diagnostics.py tests/unit/models/megatron/test_draft_update_probe.py tests/unit/models/megatron/test_draft_diagnostics.py`; verify the exact errors `draft update probe requires a nonzero gradient` and `draft update probe requires a parameter change` remain covered.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/models/megatron/draft/diagnostics.py tests/unit/models/megatron/test_draft_update_probe.py tests/unit/models/megatron/test_draft_diagnostics.py && git commit -S -s -m "perf(draft): batch whole-step update probe transfer" && git verify-commit HEAD`.

### Task 9: Bucket TP export payloads behind a fail-together preflight

**Files:**

- Modify: `nemo_rl/models/megatron/draft/utils.py:1836-1897`
- Modify: `tests/unit/models/megatron/test_dflash_export_contract.py`
- Modify: `tests/unit/models/megatron/test_draft_refit.py`

**Interfaces:**

- Consumes: ordered `state_dict()` items, `_dflash_weight_layout()`, `_all_gather_tp_shards()`, and the TP process group.
- Produces: private `_DFlashExportEntry`, `_DFlashExportBucket`, `_DFlashLocalBuildResult`, `_dflash_export_manifest(entries: Sequence[_DFlashExportEntry], buckets: Sequence[_DFlashExportBucket]) -> list[object]`, `_preflight_dflash_export_status_and_manifest(local_result: _DFlashLocalBuildResult) -> tuple[list[_DFlashExportEntry], list[_DFlashExportBucket]]`, and `_gather_dflash_export_bucket(bucket: _DFlashExportBucket, *, tp_world_size: int) -> dict[str, Tensor]`; public `export_dflash_weights_to_hf(model: torch.nn.Module) -> list[tuple[str, Tensor]]` remains unchanged.

- [ ] **Step 1: Write the failing real TP2 tests.** Fork two Gloo ranks, count baseline payload `all_gather` calls, compare ordered names/shapes/dtypes/values with the logical reference, and cover replicated and zero-sized tensors. On rank 1 separately inject failures from `unwrap_model()`, `state_dict()`, local validation, `_dflash_weight_layout()`, metadata bucket planning, and manifest/digest construction; assert every rank reaches exactly one status consensus, raises the same synchronized error, and performs zero payload tensor packing and zero payload gathers. Successful peers may construct their local metadata bucket plan before learning that another rank failed.

- [ ] **Step 2: Run RED and verify per-parameter gathering plus asymmetric entry.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py -k 'bucket or asymmetric or zero_sized' --hf-gated --mcore-only`; expect payload gather count to equal sharded parameter count and the asymmetric case not to fail together before payload.

- [ ] **Step 3: Implement one complete local-build transaction followed by fail-together preflight.** Put `unwrap_model()`, `state_dict()`, validation, layout lookup, entry construction, bucket planning, manifest construction, and digest construction inside one `try/except Exception`. Store the complete entries/buckets/digest or a bounded error code/message in `_DFlashLocalBuildResult`; no rank raises yet. The first TP consensus exchanges status plus digest and fails every rank if any rank reports an error. Only after all ranks report success and matching manifests may payload gathers begin. Reconstruct output in original `state_dict()` order.

```python
@dataclass(frozen=True, slots=True)
class _DFlashExportBucket:
    device_type: str
    bucket_ordinal: int
    dtype: torch.dtype
    device: torch.device
    entries: list[_DFlashExportEntry]


@dataclass(frozen=True, slots=True)
class _DFlashLocalBuildResult:
    entries: list[_DFlashExportEntry] | None
    buckets: list[_DFlashExportBucket] | None
    digest: bytes
    error_code: int
    error_message: str


def _bucket_manifest_row(bucket: _DFlashExportBucket) -> list[object]:
    return [
        bucket.device_type,
        bucket.bucket_ordinal,
        str(bucket.dtype),
        [(entry.name, entry.flat_offset) for entry in bucket.entries],
    ]
```

- [ ] **Step 4: Run GREEN.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py -k 'bucket or asymmetric or zero_sized or local_error' --hf-gated --mcore-only`; expect one payload gather per device/dtype bucket, exact output parity, and zero payload tensor packing/gathers after any rank-local error or manifest mismatch. Metadata entry/bucket-plan construction before consensus is expected.

- [ ] **Step 5: Run refit and static checks.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py unit/models/megatron/test_draft_refit.py --hf-gated --mcore-only`, return to the repository root, and run `uv run --group dev pre-commit run --files nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py tests/unit/models/megatron/test_draft_refit.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py tests/unit/models/megatron/test_draft_refit.py && git commit -S -s -m "perf(dflash): bucket TP refit payloads with preflight" && git verify-commit HEAD`.

### Task 10: Replace gathered manifests with one fixed-size consensus

**Files:**

- Modify: `nemo_rl/models/megatron/draft/utils.py`
- Modify: `tests/unit/models/megatron/test_dflash_export_contract.py`
- Modify: `tests/unit/models/megatron/test_draft_refit.py`

**Interfaces:**

- Consumes: Task 9 `_DFlashLocalBuildResult`, normalized ordered manifest, and `_preflight_dflash_export_status_and_manifest()` call position.
- Produces: one fixed-width `torch.int64` consensus vector containing local success/error code, entry count, bucket count, and 32 SHA256 bytes plus their `_MANIFEST_COMPLEMENT - value` complements. One `dist.all_reduce(consensus, op=dist.ReduceOp.MIN, group=tp_group)` proves every rank completed the required local metadata entry/bucket plan and that every manifest word agrees before any payload tensor packing or payload collective.

- [ ] **Step 1: Write the failing fixed-size consensus tests.** Spy on Task 9's object/variable-size collective, assert its serialized payload grows between one and twenty entries, assert payload tensor packing and payload collective functions are not called on name/shape/dtype/order mismatch or exporter error, and assert zero-owner ranks participate once. Local metadata builders may run before consensus.

- [ ] **Step 2: Run RED and verify variable-size metadata.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py -k 'fixed_size_manifest or manifest_mismatch or exporter_error' --hf-gated --mcore-only`; expect the size-invariance and one-consensus assertions to fail.

- [ ] **Step 3: Implement the minimal fixed-size status plus min/max encoding.** For a successful local build, hash `repr(_dflash_export_manifest(entries, buckets)).encode("utf-8")`; for a failed build, use zero counts/digest and the bounded nonzero error code. Prepend success/error code, entry count, and bucket count; reject values above `2**62 - 1`; concatenate values and complements; issue one MIN all-reduce; and recover maxima. Local metadata entry/bucket planning is required before this consensus. If any error code is nonzero, every rank raises the synchronized local-build error before payload tensor packing or payload collectives. Otherwise, any min/max mismatch raises `[draft] DFlash export manifest differs across TP ranks.` before payload tensor packing or payload collectives.

```python
_MANIFEST_COMPLEMENT = 2**62 - 1


def _manifest_consensus_tensor(
    *,
    error_code: int,
    entry_count: int,
    bucket_count: int,
    digest: bytes,
    device: torch.device,
) -> torch.Tensor:
    values = torch.tensor(
        [error_code, entry_count, bucket_count, *digest],
        dtype=torch.int64,
        device=device,
    )
    return torch.cat((values, _MANIFEST_COMPLEMENT - values))
```

- [ ] **Step 4: Run GREEN.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py -k 'fixed_size_manifest or manifest_mismatch or exporter_error' --hf-gated --mcore-only`; expect one fixed-size consensus, fail-together mismatch, zero payload-packing/payload-collective calls on failure, and zero-owner participation.

- [ ] **Step 5: Run all export/refit and static checks.** Run `cd tests && uv run --extra mcore pytest -q unit/models/megatron/test_dflash_export_contract.py unit/models/megatron/test_draft_refit.py --hf-gated --mcore-only`, return to the repository root, and run `uv run --group dev pre-commit run --files nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py tests/unit/models/megatron/test_draft_refit.py`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add nemo_rl/models/megatron/draft/utils.py tests/unit/models/megatron/test_dflash_export_contract.py tests/unit/models/megatron/test_draft_refit.py && git commit -S -s -m "perf(draft): use fixed-size refit manifest consensus" && git verify-commit HEAD`.

### Task 11: Compose accepted commits and guard the fixed path

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/compose_guard.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/tests/test_compose_guard.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/compose_proof.json`

**Interfaces:**

- Consumes: Task 1 base SHA, accepted instrumentation commit SHAs from Tasks 2-3, accepted optimization commit SHAs from Tasks 5-10, fixed baseline/optimized resolved YAML bytes, `git diff --name-only`, and Task 2 fixed-run counter snapshot.
- Produces: `ComposeProof`, `validate_composition(proof: ComposeProof) -> None`, and exact `compose_proof.json` consumed by Task 12. Validation fails if any accepted optimization helper executes on the fixed path.

- [ ] **Step 1: Write the failing compose-guard test.** Build fixtures for approved product files and byte-identical fixed configs; assert all-zero fixed-path helper calls pass, one helper call raises `RuntimeError("optimized helper reached fixed path")`, an unapproved file raises, and differing fixed configs raise.

- [ ] **Step 2: Run RED and verify the missing guard failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_compose_guard.py`; expect collection to fail because `compose_guard.py` does not exist.

- [ ] **Step 3: Implement the minimal guard and proof schema.** Use this exact decision rule after validating base SHA, accepted commits, approved files, and fixed config bytes:

```python
@dataclass(frozen=True, slots=True)
class ComposeProof:
    base_head: str
    optimized_head: str
    accepted_commits: list[str]
    changed_files: list[str]
    fixed_config_sha256: str
    optimized_fixed_config_sha256: str
    fixed_path_calls: dict[str, int]
    fixed_path_qualified: bool


_HELPERS_BY_OPTIMIZATION: dict[str, frozenset[str]] = {
    "metadata_once": frozenset({"metadata_prevalidated_core"}),
    "deferred_metric": frozenset({"deferred_draft_loss_metric"}),
    "cached_normalization": frozenset({"cached_normalization_scale"}),
    "batched_probe": frozenset({"whole_step_probe_transfer"}),
    "bucketed_export": frozenset({"bucketed_dflash_export"}),
    "fixed_manifest": frozenset({"fixed_manifest_consensus"}),
}


def assert_fixed_path_untouched(
    fixed_path_calls: Mapping[str, int],
    *,
    accepted_optimization_names: Collection[str],
) -> None:
    required = set().union(
        *(_HELPERS_BY_OPTIMIZATION[name] for name in accepted_optimization_names)
    )
    if not required:
        raise RuntimeError("no accepted optimization helper keys")
    if set(fixed_path_calls) != required:
        raise RuntimeError(
            f"fixed-path helper coverage mismatch: expected={sorted(required)} "
            f"actual={sorted(fixed_path_calls)}"
        )
    reached = {name: count for name, count in fixed_path_calls.items() if count != 0}
    if reached:
        raise RuntimeError(f"optimized helper reached fixed path: {reached}")
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_compose_guard.py`; expect approved composition, empty/missing/extra helper-map rejection, and nonzero fixed-path rejection to pass.

- [ ] **Step 5: Generate and statically validate the exact-head proof.** Run `uv run research/qwen3_8b_online_drafter_efficiency/compose_guard.py --base-contract research/qwen3_8b_online_drafter_efficiency/base_contract.json --fixed-config "${NRL_EFFICIENCY_FIXED_CONFIG:?required}" --optimized-fixed-config "${NRL_EFFICIENCY_OPTIMIZED_FIXED_CONFIG:?required}" --fixed-counter-receipt "${NRL_EFFICIENCY_FIXED_COUNTER_RECEIPT:?required}" --output research/qwen3_8b_online_drafter_efficiency/compose_proof.json`, then run `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/compose_guard.py research/qwen3_8b_online_drafter_efficiency/tests/test_compose_guard.py`; expect all checks to pass.

- [ ] **Step 6: Commit the guard and any necessary conflict resolution.** Run `git add research/qwen3_8b_online_drafter_efficiency/compose_guard.py research/qwen3_8b_online_drafter_efficiency/tests/test_compose_guard.py research/qwen3_8b_online_drafter_efficiency/compose_proof.json && git commit -S -s -m "test(draft): guard online efficiency composition" && git verify-commit HEAD`; add only already approved product files if conflict resolution changed them.

### Task 12: Build the rotated same-node matrix launcher

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/launch_matrix.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/run_matrix.sh`
- Create: `research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py`
- Modify: `research/qwen3_8b_online_drafter_efficiency/README.md`

**Interfaces:**

- Consumes: Task 1 `BaseContract`, Task 11 `ComposeProof`, exact recipe/config overrides, immutable container, output root, and W&B entity/project.
- Produces: `MatrixArm`, `MatrixAllocation`, `build_matrix(*, base_head: str, optimized_head: str, base_source_root: Path, optimized_source_root: Path) -> list[MatrixAllocation]`, unique run IDs/result roots, and exact `sbatch` commands for three replicates of each topology.

The primary shape is GBS64/MBS2. Within each replicate, all arms run sequentially on one retained node and hold target model, draft model, both revisions, K, data order, prompt order, seed, GBS, MBS, immutable image, generation settings, and CUDA Graph settings byte-identical. The launcher aborts if any held field differs.

- [ ] **Step 1: Write the failing launcher-contract tests.** Assert GBS64/MBS2 topologies are CP1 packing=false, CP1 packing=true, and CP2 packing=true; each has three independent retained-node replicates with sequential fixed/baseline-online/optimized-online arms and byte-identical held fields; arm order rotates; every arm runs through logical Step 30; W&B IDs/result roots are unique; base arms use an immutable source directory whose `git_meta.json` equals `base_head`; the optimized arm uses a separate immutable directory equal to `optimized_head`; and any source/config/artifact mismatch aborts the retained allocation.

- [ ] **Step 2: Run RED and verify the missing launcher failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py -k launcher`; expect collection to fail because `launch_matrix.py` does not exist.

- [ ] **Step 3: Implement the minimal immutable matrix builder.** Use these exact arm labels and rotation. Before `sbatch`, copy each clean pushed SHA into a separate read-only snapshot directory, record every tracked-file hash and `git_meta.json`, and mount both directories read-only. Build one external environment per SHA under `/raid/scratch/$SLURM_JOB_USER/eff-env/$sha` with `UV_PROJECT_ENVIRONMENT` pointing there and `uv sync --locked`; set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPYCACHEPREFIX`, `UV_CACHE_DIR`, and all model/Ray caches outside the snapshots. Inside the retained allocation, never invoke a script from the launcher's checkout: fixed and baseline-online run `(cd "$BASE_SOURCE_ROOT" && PYTHONPATH="$BASE_SOURCE_ROOT" UV_PROJECT_ENVIRONMENT="$BASE_ENV" uv run --no-sync "$BASE_SOURCE_ROOT/examples/run_grpo.py" ...)`; optimized-online uses the corresponding optimized source/environment. Validate tracked-file hashes immediately before and after every arm; never pull, checkout, or mutate either tree. Require each arm to emit `resolved.yaml`, `run_manifest.json`, terminal `result.txt`, and 25 canonical W&B rows before the next arm starts:

```python
_TRIAD = ("fixed", "baseline_online", "optimized_online")


def rotate_arms(arms: Sequence[str], replicate: int) -> list[str]:
    offset = replicate % len(arms)
    return list(arms[offset:]) + list(arms[:offset])


def source_root_for_arm(
    arm: str,
    *,
    base_source_root: Path,
    optimized_source_root: Path,
) -> Path:
    return optimized_source_root if arm == "optimized_online" else base_source_root
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py -k launcher`; expect topology, rotation, immutable-source routing, uniqueness, Step-30, and fail-closed artifact cases to pass.

- [ ] **Step 5: Dry-run and statically check the launcher.** Run `uv run research/qwen3_8b_online_drafter_efficiency/launch_matrix.py --base-contract research/qwen3_8b_online_drafter_efficiency/base_contract.json --compose-proof research/qwen3_8b_online_drafter_efficiency/compose_proof.json --output-root "${NRL_EFFICIENCY_OUTPUT_ROOT:?required}" --wandb-entity "${WANDB_ENTITY:?required}" --wandb-project "${WANDB_PROJECT:?required}" --dry-run` and `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/launch_matrix.py research/qwen3_8b_online_drafter_efficiency/run_matrix.sh research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py research/qwen3_8b_online_drafter_efficiency/README.md`; expect valid commands and all hooks to pass.

- [ ] **Step 6: Commit.** Run `git add research/qwen3_8b_online_drafter_efficiency/launch_matrix.py research/qwen3_8b_online_drafter_efficiency/run_matrix.sh research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py research/qwen3_8b_online_drafter_efficiency/README.md && git commit -S -s -m "perf(draft): add matched efficiency launcher" && git verify-commit HEAD`.

### Task 13: Implement the frozen paired analysis and 20% gate

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/analyze_matrix.py`
- Modify: `research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py`
- Modify: `research/qwen3_8b_online_drafter_efficiency/README.md`

**Interfaces:**

- Consumes: Task 12 run receipts and canonical W&B rows for `_step=0..29`; metrics are E2E, policy, refit, logprob, generation seconds, canonical generation TPS/GPU, acceptance rate, accepted length, draft loss, reward, `gen_kl_error`, update count, refit count, and optional peak memory.
- Produces: `paired_summary(values: Sequence[float]) -> PairedSummary`, `policy_delta_reduction(*, fixed_policy: float, baseline_online_policy: float, optimized_online_policy: float) -> float`, `optimized_online_overhead(*, fixed_e2e: float, optimized_online_e2e: float) -> float`, `relative_online_change(*, baseline_online: float, optimized_online: float) -> float`, frozen generation non-inferiority decisions, `summary.json`, `summary.csv`, and `summary.md`.

- [ ] **Step 1: Write the failing formula and closed-window tests.** Assert only `_step=5..29` is accepted with exactly 25 rows, missing/duplicate/non-finite rows fail, update/refit counts equal 25 for both online arms, `policy_delta_reduction(fixed_policy=3.4, baseline_online_policy=4.4, optimized_online_policy=4.0) == pytest.approx(0.4)`, a nonpositive baseline delta raises, and three paired replicates use sample standard deviation with t critical `4.302652729911275`; test lower-bound gates at -2% TPS, -1 percentage point acceptance, and -0.1 token accepted length on both sides of each strict threshold.

- [ ] **Step 2: Run RED and verify the missing analyzer failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py -k 'analysis or policy_delta or closed_window'`; expect import failure because `analyze_matrix.py` does not exist.

- [ ] **Step 3: Implement the exact formulas and fail-closed gates.** Use per-replicate paired differences as sampling units, never individual steps; implement the primary and E2E formulas exactly as follows:

```python
def policy_delta_reduction(
    *,
    fixed_policy: float,
    baseline_online_policy: float,
    optimized_online_policy: float,
) -> float:
    denominator = baseline_online_policy - fixed_policy
    if denominator <= 0.0:
        raise ValueError("baseline online policy delta must be positive")
    return (baseline_online_policy - optimized_online_policy) / denominator


def optimized_online_overhead(
    *,
    fixed_e2e: float,
    optimized_online_e2e: float,
) -> float:
    if fixed_e2e <= 0.0:
        raise ValueError("fixed E2E time must be positive")
    return (optimized_online_e2e - fixed_e2e) / fixed_e2e


def relative_online_change(
    *,
    baseline_online: float,
    optimized_online: float,
) -> float:
    if baseline_online <= 0.0:
        raise ValueError("baseline online metric must be positive")
    return (optimized_online - baseline_online) / baseline_online
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py -k 'analysis or policy_delta or closed_window'`; expect all formula, window, count, and confidence-interval cases to pass.

- [ ] **Step 5: Run the full harness tests and static checks.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests` and `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/analyze_matrix.py research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py research/qwen3_8b_online_drafter_efficiency/README.md`; expect all checks to pass.

- [ ] **Step 6: Commit.** Run `git add research/qwen3_8b_online_drafter_efficiency/analyze_matrix.py research/qwen3_8b_online_drafter_efficiency/tests/test_matrix_contract.py research/qwen3_8b_online_drafter_efficiency/README.md && git commit -S -s -m "perf(draft): add frozen efficiency analysis" && git verify-commit HEAD`.

### Task 14: Prove exact-head correctness and prepare the separate-review handoff

**Files:**

- Create: `research/qwen3_8b_online_drafter_efficiency/evidence.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/tests/test_evidence.py`
- Create: `research/qwen3_8b_online_drafter_efficiency/run_full_gate.sh`
- Create: `research/qwen3_8b_online_drafter_efficiency/run_packed_e2e.sh`
- Create: `research/qwen3_8b_online_drafter_efficiency/HANDOFF.md`
- Modify: `research/qwen3_8b_online_drafter_efficiency/README.md`
- Modify: `pyrefly.toml`

**Interfaces:**

- Consumes: exact optimized SHA, focused CPU/static results, exact full MCore receipt, packed TP2 x CP2 target-SP two-step DFlash-to-DSpark receipt, deterministic checkpoint/export parity receipt, matrix summary, signing audit, and GitHub verification audit.
- Produces: `EvidenceBundle`, `validate_terminal_evidence(bundle: EvidenceBundle) -> None`, and `HANDOFF.md` with the exact inputs for a separate Claude Code review; no product API and no review action in this plan.

- [ ] **Step 1: Write the failing evidence test.** Build a complete PASS fixture and parametrically remove each required receipt, change one receipt head, set one result to FAIL, remove one DFlash/DSpark marker, fail checkpoint tensor parity, and fail a performance/non-inferiority gate; assert every incomplete bundle raises before readiness, and assert both gate scripts name the exact existing test entrypoints they execute.

- [ ] **Step 2: Run RED and verify the missing validator failure.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_evidence.py`; expect collection to fail because `evidence.py` does not exist.

- [ ] **Step 3: Implement the minimal exact-head validator and gate wrappers.** Require all receipt heads equal `optimized_head`, both online arms contain one update and one applied refit per analyzed step, packed E2E contains finite loss/checkpoint/speculator-reload markers for DFlash and DSpark, checkpoint/export keys/shapes/dtypes match exactly, values satisfy configured tolerances, and every frozen Task 13 gate has an explicit boolean result. Register every new typed module under `research/qwen3_8b_online_drafter_efficiency/` in `pyrefly.toml`. Make `run_full_gate.sh` execute the explicit Project 1 MCore list with `--hf-gated --mcore-only`. Make `run_packed_e2e.sh` run four phases in one allocation: DFlash timeout-save after Step 1 and automatic resume through Step 2; then the same two phases for DSpark. Both invocations keep `grpo.max_num_steps=2`, the same checkpoint directory/config, and `checkpointing.save_optimizer=true`. Between invocations, `evidence.py load-checkpoint-receipt` constructs the exact provider/optimizer, loads Step 1, and writes immediate post-load policy/draft/optimizer hashes; those must match the save-time receipt before run 2 starts. Run 2 must log automatic checkpoint discovery and start at Step 2, then emit the next update/refit marker.

```bash
cd tests
uv run --extra mcore pytest -q \
  unit/algorithms/test_dflash_cp_zero_owner.py \
  unit/algorithms/test_dflash_projected_loss.py \
  unit/distributed/test_projected_draft_soft_ce.py \
  unit/models/megatron/test_dflash_block_plan.py \
  unit/models/megatron/test_dflash_cp_provider_contract.py \
  unit/models/megatron/test_dflash_export_contract.py \
  unit/models/megatron/test_dflash_training_provider.py \
  unit/models/megatron/test_draft_hidden_capture.py \
  unit/models/megatron/test_draft_refit.py \
  unit/models/megatron/test_draft_step_state.py \
  unit/models/megatron/test_dspark_training_provider.py \
  --hf-gated --mcore-only
```

```bash
for draft_type in dflash dspark; do
  checkpoint_root="$RESULT_ROOT/$draft_type/checkpoints"
  uv run examples/run_grpo.py --config "$CONFIG_ROOT/$draft_type.yaml" \
    policy.sequence_packing.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.context_parallel_size=2 \
    policy.megatron_cfg.sequence_parallel=true \
    checkpointing.enabled=true checkpointing.save_period=1 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_must_save_by=00:00:00:01 \
    checkpointing.checkpoint_dir="$checkpoint_root" grpo.max_num_steps=2
  uv run research/qwen3_8b_online_drafter_efficiency/evidence.py \
    save-checkpoint-receipt --checkpoint "$checkpoint_root/step_1" \
    --output "$RESULT_ROOT/$draft_type/save-receipt.json"
  uv run research/qwen3_8b_online_drafter_efficiency/evidence.py \
    load-checkpoint-receipt --checkpoint "$checkpoint_root/step_1" \
    --config "$CONFIG_ROOT/$draft_type.yaml" \
    --expected-save-receipt "$RESULT_ROOT/$draft_type/save-receipt.json" \
    --output "$RESULT_ROOT/$draft_type/load-receipt.json"
  uv run examples/run_grpo.py --config "$CONFIG_ROOT/$draft_type.yaml" \
    policy.sequence_packing.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.context_parallel_size=2 \
    policy.megatron_cfg.sequence_parallel=true \
    checkpointing.enabled=true checkpointing.save_period=1 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_dir="$checkpoint_root" grpo.max_num_steps=2
  uv run research/qwen3_8b_online_drafter_efficiency/evidence.py validate-resume \
    --draft-type "$draft_type" --result-root "$RESULT_ROOT/$draft_type"
done
```

- [ ] **Step 4: Run GREEN.** Run `uv run pytest -q research/qwen3_8b_online_drafter_efficiency/tests/test_evidence.py`; expect the complete fixture to pass and every incomplete fixture to fail closed.

- [ ] **Step 5: Run the complete focused local gate.** Run `uv run pytest -q tests/unit/algorithms/test_dflash_metadata_performance.py tests/unit/algorithms/test_dflash_projected_loss.py tests/unit/algorithms/test_draft_loss_wrapper.py tests/unit/models/megatron/test_draft_perf_counters.py tests/unit/models/megatron/test_draft_step_state.py tests/unit/models/megatron/test_draft_update_probe.py tests/unit/models/megatron/test_draft_diagnostics.py research/qwen3_8b_online_drafter_efficiency/tests`, `uv run pyrefly check`, and `uv run --group dev pre-commit run --all-files`; expect every test and hook to pass and every new research module to be type-checked.

- [ ] **Step 6: Commit the evidence validator.** Run `git add research/qwen3_8b_online_drafter_efficiency/evidence.py research/qwen3_8b_online_drafter_efficiency/tests/test_evidence.py research/qwen3_8b_online_drafter_efficiency/run_full_gate.sh research/qwen3_8b_online_drafter_efficiency/run_packed_e2e.sh research/qwen3_8b_online_drafter_efficiency/README.md pyrefly.toml && git commit -S -s -m "test(draft): validate online efficiency evidence" && git verify-commit HEAD`.

- [ ] **Step 7: Push and verify the exact implementation head.** Run `git push origin HEAD`, `git verify-commit HEAD`, and `gh api repos/NVIDIA-NeMo/RL/commits/"$(git rev-parse HEAD)" --jq '.verification.verified'`; expect `true`.

- [ ] **Step 8: Submit the exact full MCore gate.** Run `/submit research/qwen3_8b_online_drafter_efficiency/run_full_gate.sh --cluster=oci-hsg`; require CP1/CP2/CP4 loss and gradient parity, packed layout, target SP, zero-owner forward/backward, optimizer/checkpoint/export/refit, DFlash, and DSpark coverage on the exact head, record the job ID, then monitor state/logs at 60-second cadence for at least five minutes.

- [ ] **Step 9: Submit the packed DFlash-to-DSpark gate.** After the full gate is terminal GREEN, run `/submit research/qwen3_8b_online_drafter_efficiency/run_packed_e2e.sh --cluster=oci-hsg`; require TP2 x CP2, target SP, two online steps, finite loss, update, applied post-update refit, checkpoint, and speculator reload for both DFlash and DSpark.

- [ ] **Step 10: Validate deterministic state parity.** Run `uv run research/qwen3_8b_online_drafter_efficiency/evidence.py compare-state --baseline "${NRL_EFFICIENCY_BASELINE_STATE_RECEIPT:?required}" --optimized "${NRL_EFFICIENCY_OPTIMIZED_STATE_RECEIPT:?required}" --output "${NRL_EFFICIENCY_OUTPUT_ROOT:?required}/state_parity.json"`; expect `state_parity=PASS`.

- [ ] **Step 11: Run the optimized-head packing-disabled and packing-enabled profiles.** Run `/submit research/qwen3_8b_online_drafter_efficiency/run_profile.sh --cluster=oci-hsg` with `NRL_EFFICIENCY_PROFILE_VARIANT=optimized`; the script must complete both retained-node packing flags before terminal GREEN. Then run `uv run research/qwen3_8b_online_drafter_efficiency/profile_contract.py analyze --input-root "${NRL_EFFICIENCY_PROFILE_ROOT:?required}/optimized" --output "${NRL_EFFICIENCY_OUTPUT_ROOT:?required}/optimized_profile_summary.json"` and compare against Task 4's immutable baseline receipt without replacing it.

- [ ] **Step 12: Decide the excluded targets from counted evidence.** If the profile has no counted hidden-capture allocation/copy regression, write `hidden_capture_change=false`; if it has one, stop and amend the design before changing source. Apply the same fail-closed decision to `torch.cuda.empty_cache()` with the peak-memory/OOM evidence.

- [ ] **Step 13: Submit the rotated matrix.** Run `/submit research/qwen3_8b_online_drafter_efficiency/run_matrix.sh --cluster=oci-hsg`; require every `sbatch --test-only` probe to pass first and monitor each allocation at 60-second cadence for at least five minutes.

- [ ] **Step 14: Generate the frozen reports.** Run `uv run research/qwen3_8b_online_drafter_efficiency/analyze_matrix.py --input-root "${NRL_EFFICIENCY_MATRIX_ROOT:?required}" --output-root "${NRL_EFFICIENCY_OUTPUT_ROOT:?required}"`; require `summary.json`, `summary.csv`, and `summary.md`, including failed gates without changing margins or scope.

- [ ] **Step 15: Freeze the separate-review boundary.** Record the exact pushed SHA and stop product edits in this plan; any later review fix starts a separate RED-to-GREEN change and invalidates affected exact-head evidence.

- [ ] **Step 16: Write the evidence handoff.** Populate `HANDOFF.md` with exact SHAs, changed-file list, commit list, focused/full/E2E commands and results, job/W&B IDs, supported topology, operation counts, timing statistics, parity results, signing verification, unresolved risks, and explicit prompts for the separate Claude Code review; do not run or post that review in this session.

- [ ] **Step 17: Statically validate the handoff.** Run `uv run research/qwen3_8b_online_drafter_efficiency/evidence.py validate --handoff research/qwen3_8b_online_drafter_efficiency/HANDOFF.md --optimized-head "$(git rev-parse HEAD)"` and `uv run --group dev pre-commit run --files research/qwen3_8b_online_drafter_efficiency/HANDOFF.md`; expect `terminal_evidence=PASS` and all hooks to pass.

- [ ] **Step 18: Commit the final compact handoff.** Run `git add research/qwen3_8b_online_drafter_efficiency/HANDOFF.md research/qwen3_8b_online_drafter_efficiency/README.md && git commit -S -s -m "docs(draft): record online efficiency handoff" && git verify-commit HEAD`; do not add runtime profiles, SLURM logs, W&B exports, checkpoints, or large tensor artifacts.
