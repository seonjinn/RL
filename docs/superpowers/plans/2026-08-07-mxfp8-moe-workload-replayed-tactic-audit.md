# MXFP8 MoE Workload-Replayed Tactic Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Qwen3-30B-A3B audit that replays observed MXFP8 MoE expert-routing distributions, shmoos legal FlashInfer TRTLLM FC1/FC2 tactic pairs, emits a qualified autotune cache, and validates performance and correctness against the stock cache.

**Architecture:** Add an opt-in, trace-only hook to the vLLM 0.25.1 TRTLLM MXFP8 MoE boundary, then keep profile selection, tactic enumeration, qualification, cache construction, SLURM orchestration, and reporting in a NeMo-RL experiment package. The first audit uses the exact FlashInfer 0.6.13 private interfaces already exercised by its upstream tests to enumerate and force tactic pairs; no FlashInfer runtime source is changed until the audit demonstrates an upstream-worthy gain.

**Tech Stack:** Python 3.13, PyTorch 2.11, vLLM 0.25.1, FlashInfer 0.6.13, NeMo-RL, pytest, Pyright, Ruff, SLURM, GB200, CUDA Graphs, Nsight Systems, JSON/JSONL, Matplotlib/Seaborn

## Global Constraints

- Target Qwen3-30B-A3B first on Ptyche GB200 using the current four-node NeMo-RL MXFP8 performance recipe.
- Pin the custom vLLM base to commit `a76062edee3a3ac23d47a93c7ce466f06a19111f` and record the exact NeMo-RL, FlashInfer, CUDA, driver, container, model-revision, topology, and source fingerprints.
- Keep `moe_backend=flashinfer_trtllm`, dense linear backend, model revision, quantization scope, topology, generation settings, container, and node count identical between baseline and candidate arms.
- Keep CUDA Graphs enabled for shmoo replay, vLLM validation, and NeMo-RL performance measurements; use eager mode only for the dedicated routing-trace collection run.
- Collect only execution metadata. Never write prompts, token IDs, hidden values, model outputs, credentials, or Hugging Face/W&B tokens to artifacts.
- Select representative signatures covering at least 95% of observed MoE GPU time and preserve every observed signature in the raw trace artifact.
- Profile every legal FC1/FC2 tactic pair with three warmups, at least ten timed repetitions, CUDA Graph replay, and cold-L2 inputs.
- Promote a tactic only when weighted-median improvement is at least 2%, coefficient of variation is at most 3%, and no high-weight profile regresses by more than 1%.
- Missing cache entries and metadata mismatches must fall back to stock FlashInfer behavior; cache misses are not errors.
- The request path must not profile, parse JSON per call, inspect a dynamic expert histogram, allocate trace tensors, or synchronize with the host.
- Run a two-step NeMo-RL smoke before the eight-step comparison and report steady-state steps 3 through 8 without SLURM dependencies between arms.
- Require passing micro-correctness, CUDA Graph replay, deterministic vLLM generation, matched 1,319-example GSM8K, and NeMo-RL finite-metric gates before recommending promotion.
- Use local-first development, signed commits with specific files, `git pull` before submission, SLURM `--test-only` preflight, and five minutes of post-submission monitoring.

---

## Repository Layout and Ownership

The implementation spans two isolated worktrees and one existing evaluation utility:

- **vLLM worktree:** `/Users/sna/MXFP8_generation/.worktrees/vllm-v0251-moe-tactic-audit`
  - Branch: `sna/mxfp8-moe-tactic-audit-v0251`
  - Owns only the disabled-by-default MoE routing trace hook and its tests.
- **NeMo-RL worktree:** `/Users/sna/MXFP8_generation/.worktrees/nemorl-qwen30b-linear-backend-perf`
  - Branch: `sna/qwen30b-mxfp8-linear-backend-perf`
  - Owns profile selection, replay generation, shmoo, cache qualification, launchers, result collection, and reports.
- **Existing GSM8K evaluator:** `/Users/sna/MXFP8_generation/vllm-benchmark/experiments/eval/gsm8k_vllm_eval.py`
  - Reuse it unchanged with its immutable `gsm8k_test_openai_1319.jsonl` dataset.

Do not use `/Users/sna/MXFP8_generation/.worktrees/vllm-v0251-refit-safe-linear-backends` for edits because it contains unrelated untracked result artifacts.

### Task 1: Create the Isolated vLLM Worktree and Pin Provenance

**Files:**
- Create worktree: `/Users/sna/MXFP8_generation/.worktrees/vllm-v0251-moe-tactic-audit`
- Create: `experiments/mxfp8_moe_tactic_audit/README.md` in the NeMo-RL worktree
- Test: `tests/experiments/test_mxfp8_moe_tactic_audit_provenance.py` in the NeMo-RL worktree

**Interfaces:**
- Consumes: vLLM commit `a76062edee3a3ac23d47a93c7ce466f06a19111f` and the approved design document.
- Produces: clean vLLM branch `sna/mxfp8-moe-tactic-audit-v0251` and a machine-readable provenance contract used by every later launcher.

- [ ] **Step 1: Use the worktree skill and create the isolated vLLM checkout**

Run from `/Users/sna/MXFP8_generation/vllm`:

```bash
git worktree add -b sna/mxfp8-moe-tactic-audit-v0251 \
  /Users/sna/MXFP8_generation/.worktrees/vllm-v0251-moe-tactic-audit \
  a76062edee3a3ac23d47a93c7ce466f06a19111f
```

Expected: the new worktree is clean and `git rev-parse HEAD` prints the pinned commit.

- [ ] **Step 2: Write the failing provenance test**

Create `tests/experiments/test_mxfp8_moe_tactic_audit_provenance.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "experiments/mxfp8_moe_tactic_audit/README.md"


def test_readme_pins_runtime_and_privacy_contract() -> None:
    text = README.read_text()
    assert "a76062edee3a3ac23d47a93c7ce466f06a19111f" in text
    assert "FlashInfer 0.6.13" in text
    assert "Ptyche GB200" in text
    assert "prompts, token IDs, hidden values, or model outputs" in text
    assert "CUDA Graphs" in text
    assert "1,319-example GSM8K" in text
```

- [ ] **Step 3: Run the test and verify the missing README failure**

Run:

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_provenance.py
```

Expected: FAIL with `FileNotFoundError` for `experiments/mxfp8_moe_tactic_audit/README.md`.

- [ ] **Step 4: Create the experiment README with the exact contract**

The README must state the pinned vLLM commit, FlashInfer 0.6.13, Ptyche GB200 target, trace privacy boundary, 95% profile coverage rule, candidate thresholds, CUDA Graph requirement, fallback semantics, and the matched 1,319-example GSM8K gate. Include these entry points:

```text
submit_trace_ptyche.sh -> select_profiles.py -> shmoo_moe_tactics.py
-> qualify_cache.py -> submit_validation_ptyche.sh -> build_report.py
```

- [ ] **Step 5: Run the provenance test**

Run:

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_provenance.py
```

Expected: PASS.

- [ ] **Step 6: Commit the NeMo-RL scaffold**

```bash
git add experiments/mxfp8_moe_tactic_audit/README.md \
  tests/experiments/test_mxfp8_moe_tactic_audit_provenance.py
git commit -s -m "test: define MXFP8 MoE tactic audit provenance"
```

### Task 2: Implement the vLLM Routing Trace Data Model

**Files:**
- Create: `vllm/model_executor/layers/fused_moe/experts/trtllm_moe_trace.py`
- Create: `tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py`

**Interfaces:**
- Consumes: environment variable `VLLM_MXFP8_MOE_TRACE_DIR`.
- Produces:
  - `trace_enabled() -> bool`
  - `allocate_routing_replay(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor | None`
  - `record_routing_signature(topk_ids: torch.Tensor, metadata: MoeTraceMetadata, sampled_gpu_time_us: float) -> None`
  - one append-only JSONL file per rank and process named `moe-routing-rank{rank}-pid{pid}.jsonl`.

- [ ] **Step 1: Write data-model and disabled-path tests**

Use the following public model in the test:

```python
from dataclasses import replace
import json

import torch

from vllm.model_executor.layers.fused_moe.experts.trtllm_moe_trace import (
    MoeTraceMetadata,
    allocate_routing_replay,
    record_routing_signature,
    trace_enabled,
)


BASE = MoeTraceMetadata(
    schema_version=1,
    model_revision="qwen3-30ba3b-test",
    layer_family="routed_experts",
    global_num_experts=128,
    local_num_experts=128,
    top_k=8,
    hidden_size=2048,
    intermediate_size=768,
    tp_size=1,
    ep_size=1,
    dp_size=16,
    cuda_graph_state="trace-eager",
    weight_layout="MajorK",
    quantization="MXFP8",
    runtime_fingerprint="runtime-sha256",
)


def test_trace_is_disabled_without_directory(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MXFP8_MOE_TRACE_DIR", raising=False)
    assert not trace_enabled()
    assert allocate_routing_replay(4, 2, torch.device("cpu")) is None


def test_record_writes_histogram_without_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int16)
    record_routing_signature(
        topk_ids,
        replace(BASE, global_num_experts=4, top_k=2),
        sampled_gpu_time_us=17.5,
    )
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text().strip())
    assert row["expert_counts"] == [1, 2, 1, 0]
    assert row["num_tokens"] == 2
    assert row["sampled_gpu_time_us"] == 17.5
    assert "topk_ids" not in row
    assert "hidden_states" not in row
```

- [ ] **Step 2: Run the trace tests and verify import failure**

Run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py
```

Expected: FAIL because `trtllm_moe_trace.py` does not exist.

- [ ] **Step 3: Implement the metadata type and per-process JSONL writer**

Implement this exact frozen data model:

```python
@dataclass(frozen=True)
class MoeTraceMetadata:
    schema_version: int
    model_revision: str
    layer_family: str
    global_num_experts: int
    local_num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    tp_size: int
    ep_size: int
    dp_size: int
    cuda_graph_state: str
    weight_layout: str
    quantization: str
    runtime_fingerprint: str
```

Implement `record_routing_signature` by validating a contiguous rank-2 integer tensor, computing `torch.bincount(topk_ids.flatten().to(torch.int64), minlength=global_num_experts)`, synchronously copying only the counts in the dedicated trace run, and appending one ASCII JSON object. Require a finite, positive `sampled_gpu_time_us`. Reject IDs outside `[0, global_num_experts)` with `ValueError`. Never cache an open file handle across fork.

- [ ] **Step 4: Run unit tests and static checks**

Run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py
ruff check vllm/model_executor/layers/fused_moe/experts/trtllm_moe_trace.py \
  tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py
```

Expected: PASS.

- [ ] **Step 5: Commit the vLLM trace model**

```bash
git add vllm/model_executor/layers/fused_moe/experts/trtllm_moe_trace.py \
  tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py
git commit -s -m "feat: add opt-in MXFP8 MoE routing trace model"
```

### Task 3: Wire the Trace into Modular and Monolithic TRTLLM MXFP8 MoE

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`
- Modify: `tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py`
- Modify: `tests/kernels/moe/test_ocp_mx_moe.py`

**Interfaces:**
- Consumes: `allocate_routing_replay` and `record_routing_signature` from Task 2.
- Produces: modular tracing from existing `topk_ids`; monolithic tracing from FlashInfer `routing_replay_out`; trace-only CUDA-event GPU timing; zero changed arguments, events, and allocations when tracing is disabled.

- [ ] **Step 1: Write tests for disabled, modular, and monolithic dispatch**

Use monkeypatched FlashInfer calls to assert:

```python
def test_monolithic_trace_disabled_does_not_pass_replay_buffer(
    monolithic_call,
) -> None:
    result = monolithic_call(trace_dir=None)
    assert result.flashinfer_kwargs.get("routing_replay_out") is None
    assert result.created_cuda_events == 0


def test_monolithic_trace_enabled_passes_int16_replay_buffer(
    monolithic_call,
    tmp_path,
) -> None:
    result = monolithic_call(trace_dir=tmp_path)
    replay = result.flashinfer_kwargs["routing_replay_out"]
    assert replay.dtype == torch.int16
    assert replay.shape == (result.num_tokens, result.top_k)
    assert result.created_cuda_events == 2


def test_modular_trace_uses_existing_topk_without_replay_allocation(
    modular_call,
    tmp_path,
) -> None:
    result = modular_call(trace_dir=tmp_path)
    assert result.recorded_topk.data_ptr() == result.input_topk.data_ptr()
    assert "routing_replay_out" not in result.flashinfer_kwargs
```

Implement `monolithic_call` and `modular_call` as local pytest fixtures that construct the smallest valid `FusedMoEConfig`/`FusedMoEQuantConfig` with `block_shape=[1, 32]`, replace the FlashInfer function with a fake that fills `routing_replay_out` with a fixed valid histogram, replace `torch.cuda.Event` with a counting fake, and return a frozen `CallObservation` containing the captured kwargs, input tensors, token/top-k sizes, and event count. This keeps the tests CPU-only while asserting the exact dispatch contract.

- [ ] **Step 2: Run targeted tests and verify failures**

Run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py \
  -k "monolithic or modular"
```

Expected: FAIL because `trtllm_fp8_moe.py` does not call the trace helper.

- [ ] **Step 3: Add a single metadata builder in the experts base class**

Add `_trace_metadata(self, global_num_experts: int) -> MoeTraceMetadata`. Read immutable provenance once from environment variables:

```text
VLLM_MXFP8_MOE_MODEL_REVISION
VLLM_MXFP8_MOE_RUNTIME_FINGERPRINT
VLLM_MXFP8_MOE_DP_SIZE
```

Derive TP/EP from `self.moe_config.moe_parallel_config`, set `layer_family="routed_experts"`, `cuda_graph_state="trace-eager"`, `weight_layout="MajorK"`, and `quantization="MXFP8"`.

- [ ] **Step 4: Wire modular and monolithic calls**

For modular execution, create CUDA start/end events only when tracing is enabled, record the start immediately before the FlashInfer call, record the end immediately after it, synchronize the end event in the dedicated eager trace run, and call `record_routing_signature(topk_ids, metadata, start.elapsed_time(end) * 1000.0)`. For monolithic execution, use the same trace-only events, allocate `routing_replay_out` only when tracing is enabled, pass it in `kwargs`, invoke FlashInfer, then record the returned histogram and elapsed GPU time. Restrict tracing to MXFP8 block shape `[1, 32]`; BF16 and non-MXFP8 FP8 remain unchanged.

- [ ] **Step 5: Add the SM100 output-invariance test**

Extend the existing MXFP8 case in `tests/kernels/moe/test_ocp_mx_moe.py` to run once without tracing and once with a temporary trace directory using identical tensors. Assert:

```python
torch.testing.assert_close(traced_output, baseline_output, rtol=0, atol=0)
assert traced_output.isfinite().all()
assert sum(trace_row["expert_counts"]) == num_tokens * topk
```

- [ ] **Step 6: Run CPU and GB200 tests**

Local CPU run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py
```

Ptyche single-GPU container run:

```bash
pytest -q tests/kernels/moe/test_ocp_mx_moe.py -k "mxfp8 and trace"
```

Expected: both PASS; the GPU test is skipped only on non-SM100 local hosts.

- [ ] **Step 7: Commit the vLLM integration**

```bash
git add vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py \
  tests/model_executor/layers/fused_moe/test_trtllm_moe_trace.py \
  tests/kernels/moe/test_ocp_mx_moe.py
git commit -s -m "feat: trace TRTLLM MXFP8 MoE routing signatures"
```

### Task 4: Define Trace, Profile, and Shmoo Schemas in NeMo-RL

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/schema.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_audit_schema.py`

**Interfaces:**
- Consumes: vLLM JSONL rows from Task 3.
- Produces:
  - `RoutingSignature.from_json(row: Mapping[str, object]) -> RoutingSignature`
  - `ReplayProfile.from_signature(signature: RoutingSignature, weight: float) -> ReplayProfile`
  - `TacticPair(gemm1: int, gemm2: int)`
  - `TacticMeasurement`
  - canonical `signature_key()` and JSON serialization methods.

- [ ] **Step 1: Write schema validation tests**

Test exact invariants:

```python
def test_routing_signature_rejects_histogram_sum_mismatch() -> None:
    row = {
        "schema_version": 1,
        "model_revision": "qwen3-30ba3b-test",
        "layer_family": "routed_experts",
        "num_tokens": 2,
        "global_num_experts": 4,
        "local_num_experts": 4,
        "top_k": 2,
        "hidden_size": 2048,
        "intermediate_size": 768,
        "expert_counts": [1, 2, 1, 0],
        "sampled_gpu_time_us": 17.5,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 16,
        "cuda_graph_state": "trace-eager",
        "weight_layout": "MajorK",
        "quantization": "MXFP8",
        "runtime_fingerprint": "runtime-sha256",
    }
    row["expert_counts"] = [0] * row["global_num_experts"]
    with pytest.raises(ValueError, match="num_tokens \* top_k"):
        RoutingSignature.from_json(row)


def test_tactic_pair_round_trip() -> None:
    pair = TacticPair(gemm1=64, gemm2=11)
    assert TacticPair.from_json(pair.to_json()) == pair
```

- [ ] **Step 2: Run the tests and verify import failure**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
```

Expected: FAIL because `schema.py` does not exist.

- [ ] **Step 3: Implement frozen, typed dataclasses**

Define:

```python
@dataclass(frozen=True)
class RoutingSignature:
    schema_version: int
    model_revision: str
    layer_family: str
    num_tokens: int
    global_num_experts: int
    local_num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    expert_counts: tuple[int, ...]
    sampled_gpu_time_us: float
    tp_size: int
    ep_size: int
    dp_size: int
    cuda_graph_state: str
    weight_layout: str
    quantization: str
    runtime_fingerprint: str


@dataclass(frozen=True)
class TacticPair:
    gemm1: int
    gemm2: int


@dataclass(frozen=True)
class TacticMeasurement:
    signature_key: str
    tactic: TacticPair
    median_us: float
    p95_us: float
    cv: float
    warmups: int
    repetitions: int
    finite: bool
    deterministic: bool
    max_abs_error: float
    cosine_similarity: float
    failure: str | None


@dataclass(frozen=True)
class ReplayProfile:
    signature: RoutingSignature
    signature_key: str
    aggregate_gpu_time_us: float
    call_count: int
    normalized_weight: float
    skew_class: Literal["balanced", "median-skew", "high-skew"]
```

Canonical keys use sorted ASCII JSON and SHA256. `sampled_gpu_time_us` is an observation weight and is excluded from the structural signature key. Validation requires nonnegative counts, exact expert-count length, `sum(expert_counts) == num_tokens * top_k`, positive dimensions, finite positive GPU time, and `quantization == "MXFP8"`.

- [ ] **Step 4: Run tests and static checks**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
ruff check experiments/mxfp8_moe_tactic_audit/schema.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
pyright experiments/mxfp8_moe_tactic_audit/schema.py
```

Expected: PASS.

- [ ] **Step 5: Commit the schemas**

```bash
git add experiments/mxfp8_moe_tactic_audit/schema.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
git commit -s -m "feat: define MXFP8 MoE audit schemas"
```

### Task 5: Aggregate Traces and Select 95%-Coverage Replay Profiles

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/select_profiles.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_profile_selection.py`

**Interfaces:**
- Consumes: multiple rank-local JSONL trace files containing trace-only CUDA-event GPU timings.
- Produces:
  - `aggregate_signatures(paths: Sequence[Path]) -> list[ObservedSignature]`
  - `select_profiles(observed: Sequence[ObservedSignature], coverage: float = 0.95) -> ProfileSelection`
  - `selected_profiles.json` containing selected balanced, median-skew, and high-skew representatives plus all raw signature weights.

- [ ] **Step 1: Write deterministic selection tests**

Create fixtures with aggregate GPU-time weights `50, 30, 15, 5` and assert the first three profiles are selected for exactly 95% coverage. Add three equal-token profiles with different normalized entropy and assert one balanced, one median-skew, and one high-skew profile remain when all three are observed in a high-weight bucket.

```python
selection = select_profiles(observed, coverage=0.95)
assert selection.covered_weight == pytest.approx(0.95)
assert [item.signature_key for item in selection.selected] == ["a", "b", "c"]
assert {item.skew_class for item in selection.selected} >= {
    "balanced", "median-skew", "high-skew"
}
```

- [ ] **Step 2: Run tests and verify import failure**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_profile_selection.py
```

Expected: FAIL because `select_profiles.py` does not exist.

- [ ] **Step 3: Implement aggregation and skew classification**

Define the aggregation types before implementing selection:

```python
@dataclass(frozen=True)
class ObservedSignature:
    signature: RoutingSignature
    signature_key: str
    call_count: int
    aggregate_gpu_time_us: float


@dataclass(frozen=True)
class ProfileSelection:
    selected: tuple[ReplayProfile, ...]
    all_observed: tuple[ObservedSignature, ...]
    covered_weight: float
    total_gpu_time_us: float
```

Merge identical `signature_key()` rows by summing call count and `sampled_gpu_time_us`. Define profile weight strictly as aggregate sampled GPU microseconds; reject rows without valid timing rather than silently substituting call count. Define normalized entropy:

```python
probabilities = counts / counts.sum()
entropy = -(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum()
normalized_entropy = entropy / np.log(len(counts))
```

Classify `balanced` at `>=0.90`, `high-skew` at `<0.65`, and `median-skew` otherwise. Sort by descending weight and then signature key for reproducibility.

- [ ] **Step 4: Add the CLI**

The command must be:

```bash
python experiments/mxfp8_moe_tactic_audit/select_profiles.py \
  --trace-dir TRACE_DIR \
  --coverage 0.95 \
  --output selected_profiles.json
```

Exit nonzero when the trace is empty, fingerprints differ, or achieved coverage is below 0.95.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_profile_selection.py
ruff check experiments/mxfp8_moe_tactic_audit/select_profiles.py \
  tests/experiments/test_mxfp8_moe_tactic_profile_selection.py
pyright experiments/mxfp8_moe_tactic_audit/select_profiles.py
git add experiments/mxfp8_moe_tactic_audit/select_profiles.py \
  tests/experiments/test_mxfp8_moe_tactic_profile_selection.py
git commit -s -m "feat: select representative MoE routing profiles"
```

### Task 6: Build the Workload-Replayed FlashInfer Tactic Shmoo

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py`
- Create: `experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_shmoo.py`

**Interfaces:**
- Consumes: `selected_profiles.json`, kernel-ready Qwen3-30B MXFP8 weights/scales, and FlashInfer 0.6.13.
- Produces:
  - `enumerate_valid_tactics(case: MoeKernelCase) -> tuple[TacticPair, ...]`
  - `force_tactic(cache_key: str, tactic: TacticPair) -> ContextManager[None]`
  - `reconstruct_topk(profile: ReplayProfile, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]`
  - `profile_tactic(case: MoeKernelCase, tactic: TacticPair, warmups: int = 3, repetitions: int = 10) -> TacticMeasurement`
  - one JSONL measurement per profile and tactic.

- [ ] **Step 1: Write version and tactic-normalization tests**

```python
def test_adapter_rejects_unpinned_flashinfer(monkeypatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.6.14")
    with pytest.raises(RuntimeError, match="requires FlashInfer 0.6.13"):
        assert_supported_flashinfer()


@pytest.mark.parametrize(
    "raw,expected",
    [([17, 23], TacticPair(17, 23)), ((17, 23), TacticPair(17, 23))],
)
def test_normalize_tactic_pair(raw, expected) -> None:
    assert normalize_tactic_pair(raw) == expected
```

- [ ] **Step 2: Write histogram replay tests**

For `expert_counts=(2, 1, 1, 0)`, `num_tokens=2`, and `top_k=2`, require packed top-k IDs that reproduce the exact histogram, contain no duplicate expert within a token, and use BF16 routing weights that sum to one per token.

- [ ] **Step 3: Run tests and verify import failures**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py
```

Expected: FAIL because the adapter and shmoo modules do not exist.

- [ ] **Step 4: Implement the pinned FlashInfer adapter**

Define the in-memory kernel case explicitly; it is never serialized:

```python
@dataclass(frozen=True)
class MoeKernelCase:
    profile: ReplayProfile
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    output: torch.Tensor
    activation_type: int
    routing_method_type: int
    local_expert_offset: int
```

Mirror the already-upstream-tested mechanisms from FlashInfer's `tests/moe/test_trtllm_gen_moe_autotune_tactics.py`:

```python
moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
raw_tactics = moe_op.trtllm_get_valid_moe_configs(
    dtype_act,
    dtype_weights,
    Fp8QuantizationType.MxFp8,
    top_k,
    hidden_size,
    intermediate_size,
    local_num_experts,
    activation_type,
    True,
    WeightLayout.MajorK.value,
    False,
    num_tokens,
    False,
)
```

Force one tactic by clearing only the audit process's `AutoTuner` state and inserting the exact file key for `flashinfer::trtllm_fp8_block_scale_moe` and `MoERunner`. The adapter must snapshot and restore `_file_configs` and `profiling_cache` in `finally` so a failed tactic cannot leak into the next measurement.

- [ ] **Step 5: Implement deterministic routing reconstruction**

Use a greedy round-robin assignment seeded by `signature_key`. Reject impossible histograms where a token would require the same expert twice. Pack IDs and BF16 weights using:

```python
packed_topk = (topk_ids.to(torch.int32) << 16) | topk_weights.view(torch.int16).to(torch.int32)
```

- [ ] **Step 6: Implement CUDA Graph and cold-L2 timing**

For every tactic:

1. Produce identical kernel inputs and routing tensors.
2. Run the stock `[-1, -1]` output reference and a zero-LoRA-delta `do_finalize=False` reference that exposes the FC1 activated intermediate without changing the mathematical result.
3. Run three warmups.
4. Capture the exact tactic call into a CUDA Graph.
5. Before each of ten repetitions, touch a buffer larger than L2 and replay the graph between CUDA events.
6. Compare both the FC1 activated intermediate and final FC2 reduced output, then record median, p95, CV, NaN/Inf, deterministic replay, max absolute error, and cosine similarity.
7. Catch tactic exceptions and write `failure`, then continue.

The script must not benchmark one stage in isolation unless FlashInfer exposes a stage-safe API; tactic IDs remain a pair and NSys supplies FC1/FC2 component times. If FlashInfer 0.6.13 cannot return the FC1 intermediate with a zero `gemm1_lora_delta` and `do_finalize=False`, the script exits before cache promotion and records `flashinfer_intermediate_api_unavailable`; no candidate cache may be qualified from final-output checks alone.

- [ ] **Step 7: Add the single-profile GB200 smoke**

Run on Ptyche:

```bash
python experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \
  --profiles selected_profiles.json \
  --profile-limit 1 \
  --tactic-limit 2 \
  --warmups 3 \
  --repetitions 10 \
  --output smoke_measurements.jsonl
```

Expected: two completed or explicitly failed tactic rows, no process crash, and every successful row has `repetitions=10` and finite timing statistics.

- [ ] **Step 8: Run tests and commit**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py
ruff check experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py
pyright experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py
git add experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py
git commit -s -m "feat: shmoo workload-replayed MXFP8 MoE tactics"
```

### Task 7: Qualify Candidates and Emit a Versioned FlashInfer Cache

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/qualify_cache.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py`

**Interfaces:**
- Consumes: stock autotune JSON, selected profile weights, and shmoo JSONL.
- Produces:
  - `qualify_bucket(bucket: BucketAudit) -> QualificationDecision`
  - `build_candidate_cache(stock_cache: Path, decisions: Sequence[QualificationDecision], output: Path) -> CacheManifest`
  - `candidate/autotune_configs.json`
  - `candidate/cache_manifest.json`.

- [ ] **Step 1: Write threshold and fallback tests**

Cover all decision boundaries:

```python
def bucket(
    weighted_gain: float,
    cv: float,
    worst_regression: float,
) -> BucketAudit:
    return BucketAudit(
        cache_key="flashinfer-moe-bucket",
        stock=TacticPair(1, 2),
        candidate=TacticPair(3, 4),
        weighted_gain=weighted_gain,
        max_cv=cv,
        worst_high_weight_regression=worst_regression,
        all_correct=True,
    )


def test_promotes_only_robust_two_percent_gain() -> None:
    decision = qualify_bucket(bucket(weighted_gain=0.024, cv=0.02, worst_regression=0.009))
    assert decision.promoted


@pytest.mark.parametrize(
    "gain,cv,worst_regression,reason",
    [
        (0.019, 0.02, 0.0, "weighted gain below 2%"),
        (0.03, 0.031, 0.0, "coefficient of variation above 3%"),
        (0.03, 0.02, 0.011, "high-weight regression above 1%"),
    ],
)
def test_rejects_unqualified_candidate(gain, cv, worst_regression, reason) -> None:
    decision = qualify_bucket(
        bucket(
            weighted_gain=gain,
            cv=cv,
            worst_regression=worst_regression,
        )
    )
    assert not decision.promoted
    assert decision.reason == reason
```

- [ ] **Step 2: Run tests and verify import failure**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py
```

Expected: FAIL because `qualify_cache.py` does not exist.

- [ ] **Step 3: Implement call-weighted qualification**

Define the qualification contracts:

```python
@dataclass(frozen=True)
class BucketAudit:
    cache_key: str
    stock: TacticPair
    candidate: TacticPair
    weighted_gain: float
    max_cv: float
    worst_high_weight_regression: float
    all_correct: bool


@dataclass(frozen=True)
class QualificationDecision:
    cache_key: str
    selected: TacticPair
    promoted: bool
    reason: str


@dataclass(frozen=True)
class CacheManifest:
    stock_sha256: str
    candidate_sha256: str
    source_fingerprints: Mapping[str, str]
    promoted_entries: int
    retained_entries: int
```

Use the stock tactic row as denominator for each replay profile. Compute weighted median over selected profile weights, max regression over profiles contributing at least 5% of bucket weight, and maximum candidate CV. Reject any failed, nonfinite, nondeterministic, or micro-correctness-failing candidate before performance ranking.

- [ ] **Step 4: Implement cache replacement without a second runtime lookup**

Load the stock cache with `AutoTuner.load_configs`. Replace only exact `flashinfer::trtllm_fp8_block_scale_moe` entries that correspond to promoted token buckets, preserve every other entry byte-for-byte at the JSON-object level, and call `AutoTuner.save_configs` to write the candidate file. Immediately reload the candidate in a fresh process and assert every promoted key resolves to its tactic pair.

The manifest must include SHA256 values for stock cache, candidate cache, trace set, selected profiles, shmoo results, model revision, container, vLLM commit, FlashInfer version, CUDA version, GPU name, TP/EP/DP, and CUDA Graph mode.

- [ ] **Step 5: Add mismatch and miss tests**

Assert a runtime fingerprint mismatch rejects candidate loading and selects the stock cache path. Assert an unmodified key still returns the stock tactic and an absent key reaches the FlashInfer heuristic without raising.

- [ ] **Step 6: Run tests and commit**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py
ruff check experiments/mxfp8_moe_tactic_audit/qualify_cache.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py
pyright experiments/mxfp8_moe_tactic_audit/qualify_cache.py
git add experiments/mxfp8_moe_tactic_audit/qualify_cache.py \
  tests/experiments/test_mxfp8_moe_tactic_cache_qualification.py
git commit -s -m "feat: qualify and serialize MXFP8 MoE tactic cache"
```

### Task 8: Add Micro, vLLM, and GSM8K Correctness Gates

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/validate_correctness.py`
- Create: `experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_correctness.py`
- Reuse unchanged: `/Users/sna/MXFP8_generation/vllm-benchmark/experiments/eval/gsm8k_vllm_eval.py`
- Reuse unchanged: `/Users/sna/MXFP8_generation/vllm-benchmark/experiments/eval/data/gsm8k_test_openai_1319.jsonl`

**Interfaces:**
- Consumes: stock/candidate micro outputs, deterministic generation JSONL, and two GSM8K result directories.
- Produces:
  - `validate_micro(measurements: Sequence[TacticMeasurement]) -> CorrectnessSummary`
  - `compare_generations(stock: Path, candidate: Path) -> GenerationComparison`
  - `compare_gsm8k(stock: Path, candidate: Path) -> PairedGsm8kComparison`
  - nonzero exit on any promotion-blocking gate.

- [ ] **Step 1: Write paired correctness tests**

Test NaN rejection, routing-count mismatch rejection, deterministic token mismatch reporting, paired GSM8K disagreement counts, McNemar's exact test, and a paired bootstrap confidence interval with a fixed seed. The candidate passes the statistical-regression gate only when two-sided McNemar `p >= 0.05` and the 95% interval for `candidate_accuracy - stock_accuracy` includes zero.

- [ ] **Step 2: Run tests and verify import failure**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_correctness.py
```

Expected: FAIL because correctness modules do not exist.

- [ ] **Step 3: Implement micro and generation gates**

Define the result contracts:

```python
@dataclass(frozen=True)
class CorrectnessSummary:
    passed: bool
    checked_tactics: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class GenerationComparison:
    passed: bool
    compared_examples: int
    mismatched_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedGsm8kComparison:
    stock_accuracy: float
    candidate_accuracy: float
    candidate_only_wins: int
    stock_only_wins: int
    mcnemar_p_value: float
    delta_ci95: tuple[float, float]
    passed: bool
```

Require finite outputs, deterministic graph replays, unchanged routing counts, the upstream MXFP8 MoE numerical bounds, cosine similarity at least `0.999`, and no max-error outlier relative to the stock tactic. Compare FC1 activated intermediates and final FC2 reduced outputs against stock; for a representative balanced and high-skew profile, also compare the final output to the existing BF16/Python MoE reference used by the FlashInfer tests. Deterministic vLLM generation uses the same prompts, greedy decoding, seed, tokenizer, max tokens, and candidate cache as the performance run.

- [ ] **Step 4: Implement matched GSM8K comparison**

Require both `results.json` files to report the immutable dataset SHA256, `total=1319`, identical example IDs, model revision, tokenizer revision, generation arguments, and runtime fingerprint. Report exact match, candidate-only wins, stock-only wins, both-wrong, paired delta, and the fixed-seed paired bootstrap interval.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_correctness.py
ruff check experiments/mxfp8_moe_tactic_audit/validate_correctness.py \
  experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py
pyright experiments/mxfp8_moe_tactic_audit/validate_correctness.py \
  experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py
git add experiments/mxfp8_moe_tactic_audit/validate_correctness.py \
  experiments/mxfp8_moe_tactic_audit/compare_gsm8k.py \
  tests/experiments/test_mxfp8_moe_tactic_correctness.py
git commit -s -m "test: gate MXFP8 MoE tactics on matched correctness"
```

### Task 9: Add Ptyche Trace, Shmoo, and Validation Launchers

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/submit_trace_ptyche.sh`
- Create: `experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh`
- Create: `experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh`
- Create: `experiments/mxfp8_moe_tactic_audit/provenance.sh`
- Create: `tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py`

**Interfaces:**
- Consumes: custom vLLM worktree/remote checkout, current Qwen3 performance recipe, selected profiles, stock/candidate cache, and correctness tools.
- Produces: `ACTION=test-only`, `ACTION=dry-run`, and `ACTION=submit` entry points with immutable run manifests and isolated output roots.

- [ ] **Step 1: Write dry-run launcher tests**

Require the trace arm to set `enforce_eager=true`, `MAX_STEPS=2`, and `VLLM_MXFP8_MOE_TRACE_DIR`; require shmoo to request one GB200 for five hours; require validation baseline/candidate arms to set `enforce_eager=false`, use separate cache roots, and never emit `--dependency`.

```python
assert "policy.generation.vllm_cfg.enforce_eager=true" in trace_output
assert "--time=05:00:00" in shmoo_output
assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=" in candidate_output
assert "--dependency" not in validation_output
```

- [ ] **Step 2: Run tests and verify missing-script failures**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py
```

Expected: FAIL because the launchers do not exist.

- [ ] **Step 3: Implement common provenance checks**

`provenance.sh` must reject dirty tracked source, assert exact NeMo-RL and vLLM commits, hash the container, recipe, model snapshot, cache, and experiment scripts, run `git pull --ff-only` before submission, and write `run_manifest.json`. It must read credentials only from the environment and omit their values from output.

- [ ] **Step 4: Implement the trace launcher**

Use the shipped Qwen3-30B four-node recipe for two steps, enable only the trace environment variables, force eager execution, and write per-rank trace files under the run root. Disable W&B unless explicitly enabled. The trace result is metadata-only and is never used as a performance number.

- [ ] **Step 5: Implement the shmoo launcher**

Submit one GB200, one task, five hours, the same container and vLLM/FlashInfer environment, selected profile artifact, three warmups, ten repetitions, CUDA Graph replay, and NSys capture for selected winners plus stock. Write one result row even for a crashing tactic.

- [ ] **Step 6: Implement the validation launcher**

Support `ARM=stock|candidate` and `MAX_STEPS=2|8`. Both arms use CUDA Graphs and the same recipe. Stock points to the captured stock cache root; candidate points to the qualified cache root. Add deterministic generation and GSM8K subcommands after the two-step smoke succeeds. Do not couple the arms with dependencies.

- [ ] **Step 7: Run dry-run tests and shell syntax checks**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py
bash -n experiments/mxfp8_moe_tactic_audit/submit_trace_ptyche.sh
bash -n experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh
bash -n experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
```

Expected: PASS.

- [ ] **Step 8: Commit the launchers**

```bash
git add experiments/mxfp8_moe_tactic_audit/submit_trace_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh \
  experiments/mxfp8_moe_tactic_audit/provenance.sh \
  tests/experiments/test_mxfp8_moe_tactic_audit_launchers.py
git commit -s -m "feat: orchestrate Ptyche MXFP8 MoE tactic audit"
```

### Task 10: Build the Result Collector, Plots, and Reports

**Files:**
- Create: `experiments/mxfp8_moe_tactic_audit/collect_results.py`
- Create: `experiments/mxfp8_moe_tactic_audit/plot_results.py`
- Create: `experiments/mxfp8_moe_tactic_audit/build_report.py`
- Create: `tests/experiments/test_mxfp8_moe_tactic_audit_report.py`
- Create output: `experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.html`
- Create output: `experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.md`

**Interfaces:**
- Consumes: trace summary, shmoo JSONL, cache manifest, correctness summaries, NSys CSV, and stock/candidate NeMo-RL logs.
- Produces: normalized plots, raw tables, cache hit/fallback rates, correctness verdict, provenance appendix, and explicit keep/reject conclusion.

- [ ] **Step 1: Write parser and report tests with fixture artifacts**

Create fixtures for six measured steps per arm and require:

```python
def write_run_fixture(
    root: Path,
    *,
    steps: Iterable[int],
    tokens_per_second_per_gpu: float,
    generation_seconds: float,
    total_step_seconds: float,
) -> Path:
    run = root / "run"
    run.mkdir()
    blocks = []
    for _step in steps:
        blocks.append(
            "\n".join(
                [
                    "Training Results:",
                    f"  • Total step time: {total_step_seconds:.2f}s",
                    f"  • generation: {generation_seconds:.2f}s (26.2%)",
                    f"    - E2E (Tokens/sec/gpu): {tokens_per_second_per_gpu:.2f}",
                    f"    - Generation Worker Group (Tokens/sec/gpu): {tokens_per_second_per_gpu:.2f}",
                    "  • Reward: 0.5",
                    "  • KL: 0.01",
                    "  • Loss: 0.2",
                ]
            )
        )
    (run / "ray-driver.log").write_text("\n".join(blocks))
    return run


fixture_root = write_run_fixture(
    tmp_path,
    steps=range(1, 9),
    tokens_per_second_per_gpu=9500.0,
    generation_seconds=55.0,
    total_step_seconds=210.0,
)
summary = summarize_run(fixture_root, first_step=3, last_step=8)
assert summary.measured_steps == 6
assert summary.generated_tokens_per_second_per_gpu > 0
assert summary.all_metrics_finite
```

Assert the rendered report contains `FC1/GEMM1`, `FC2/GEMM2`, `95%`, `cache hit`, `fallback`, `GSM8K`, `steps 3-8`, all source hashes, and either `KEEP` or `REJECT`.

- [ ] **Step 2: Run tests and verify import failure**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_report.py
```

Expected: FAIL because report modules do not exist.

- [ ] **Step 3: Implement result collection**

Parse NeMo-RL `Training Results` blocks using the established Qwen30B parser pattern. Require exactly steps 3-8 for a complete measurement, finite reward/loss/KL metrics, successful refit/rollout/logprob/train phases, realized token counts, and identical manifests except cache identity.

- [ ] **Step 4: Implement publication plots**

Generate 600-DPI PNG and PDF plots:

1. call-weighted FC1/FC2 microbenchmark speedup distribution;
2. selected-tactic change share and cache hit/fallback share;
3. stock-normalized generation tokens/s/GPU and total step time;
4. per-step stock/candidate variation for steps 3-8.

Use a baseline line at 1.0 behind bars, direct value labels, batch/run metadata in captions, and no confidence band unless repeated measurements exist.

- [ ] **Step 5: Implement HTML and Markdown reports**

The conclusion logic is:

```python
keep = (
    all_correctness_gates_pass
    and end_to_end_speedup > run_to_run_variation
    and no_primary_metric_regression
)
```

If `keep` is false, state that stock FlashInfer autotuning is sufficient for this workload and preserve the audit as evidence. Distinguish microbenchmark opportunity from end-to-end gain.

- [ ] **Step 6: Run tests and commit**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_report.py
ruff check experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py
pyright experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py
git add experiments/mxfp8_moe_tactic_audit/collect_results.py \
  experiments/mxfp8_moe_tactic_audit/plot_results.py \
  experiments/mxfp8_moe_tactic_audit/build_report.py \
  tests/experiments/test_mxfp8_moe_tactic_audit_report.py
git commit -s -m "feat: report MXFP8 MoE tactic audit results"
```

### Task 11: Execute the Ptyche Experiment Pipeline

**Files:**
- Modify with collected artifacts: `experiments/mxfp8_moe_tactic_audit/results/`
- Generate: `experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.html`
- Generate: `experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.md`

**Interfaces:**
- Consumes: all implementation tasks, clean pushed branches, container, model snapshot, and Ptyche allocation.
- Produces: the six acceptance artifacts from the approved design and a keep/reject decision.

- [ ] **Step 1: Run the complete local verification suite**

```bash
pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_*.py
ruff check experiments/mxfp8_moe_tactic_audit tests/experiments/test_mxfp8_moe_tactic_audit_*.py
pyright experiments/mxfp8_moe_tactic_audit
```

Expected: PASS.

- [ ] **Step 2: Push the exact NeMo-RL and vLLM commits**

Verify only intended files are committed, then push both feature branches. Record both commit SHAs in the experiment manifest.

- [ ] **Step 3: Preflight and submit the trace run**

```bash
ACTION=test-only MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_trace_ptyche.sh
ACTION=submit MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_trace_ptyche.sh
```

Monitor for five minutes. Abort and fix initialization, import, NCCL, routing-trace, or disk errors before proceeding.

- [ ] **Step 4: Select profiles and verify coverage**

```bash
python experiments/mxfp8_moe_tactic_audit/select_profiles.py \
  --trace-dir RESULTS/trace \
  --coverage 0.95 \
  --output RESULTS/selected_profiles.json
```

Expected: achieved coverage is at least 0.95 and all runtime fingerprints match.

- [ ] **Step 5: Preflight and submit the shmoo**

```bash
ACTION=test-only ./experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh
ACTION=submit ./experiments/mxfp8_moe_tactic_audit/submit_shmoo_ptyche.sh
```

Monitor five minutes. After completion, run `qualify_cache.py` and require a reloadable candidate cache even when zero tactics qualify.

- [ ] **Step 6: Run two-step stock and candidate smokes independently**

```bash
ACTION=test-only ARM=stock MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
ACTION=test-only ARM=candidate MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
ACTION=submit ARM=stock MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
ACTION=submit ARM=candidate MAX_STEPS=2 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
```

Expected: both complete refit, rollout, logprob, and training with CUDA Graphs enabled and finite metrics.

- [ ] **Step 7: Run deterministic vLLM and matched GSM8K gates**

Use identical fixed prompts and greedy settings for stock/candidate. Then run the existing evaluator twice with `--limit 1319`, identical concurrency and generation settings, and separate output directories. Run `compare_gsm8k.py`; any provenance mismatch, two-sided McNemar `p < 0.05`, or paired 95% confidence interval that excludes zero in the negative direction blocks promotion.

- [ ] **Step 8: Submit eight-step stock and candidate measurements independently**

```bash
ACTION=submit ARM=stock MAX_STEPS=8 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
ACTION=submit ARM=candidate MAX_STEPS=8 \
  ./experiments/mxfp8_moe_tactic_audit/submit_validation_ptyche.sh
```

Monitor each for five minutes. Do not add dependencies.

- [ ] **Step 9: Collect results and build reports**

```bash
python experiments/mxfp8_moe_tactic_audit/collect_results.py \
  --results-root RESULTS \
  --output RESULTS/summary.json
python experiments/mxfp8_moe_tactic_audit/build_report.py \
  --summary RESULTS/summary.json \
  --html experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.html \
  --markdown experiments/mxfp8_moe_tactic_audit/report/mxfp8_moe_tactic_audit_latest.md
```

Expected: complete acceptance-criteria table, raw artifact links, normalized plots, and explicit `KEEP` or `REJECT`.

- [ ] **Step 10: Run final verification and commit only reproducible artifacts**

Run all experiment tests again, verify report links, and commit scripts, compact summaries, manifests, and reports. Do not commit credentials, large NSys profiles, model artifacts, container files, or raw per-example GSM8K completions.

```bash
git add experiments/mxfp8_moe_tactic_audit tests/experiments/test_mxfp8_moe_tactic_audit_*.py
git commit -s -m "docs: report workload-replayed MXFP8 MoE tactic audit"
```

## Upstream Follow-up Gate

Do not open a FlashInfer runtime PR solely from a microbenchmark win. If the candidate passes every correctness gate and improves end-to-end NeMo-RL performance beyond measured run-to-run variation, prepare these separate upstream changes:

1. **FlashInfer:** replace the experiment's private adapter with a public offline-only API that accepts representative packed top-k routing tensors, enumerates legal FC1/FC2 tactic pairs, and emits normal autotune-cache entries.
2. **vLLM:** retain only cache-path configuration, distributed cache loading, fingerprint invalidation, and fallback behavior. The trace hook should be proposed separately as a diagnostics feature, not required in the serving path.
3. **NeMo-RL:** expose an opt-in cache artifact/config path and preserve the workload collection and validation workflow outside the rollout request path.

If no candidate clears the end-to-end gate, keep the report and conclude that FlashInfer 0.6.13 stock MoE autotuning is sufficient for the tested Qwen3-30B-A3B workload.
