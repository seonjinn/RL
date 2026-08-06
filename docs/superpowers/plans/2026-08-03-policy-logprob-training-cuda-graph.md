# Policy Training and Logprob Transformer Engine CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current packed-THD Transformer Engine partial CUDA Graph
training path and add independent, correctness-preserving forward-only partial
graphs for policy and reference logprob in NeMo-RL.

**Architecture:** Megatron-Core owns an explicit `TRAINING` versus
`FORWARD_ONLY_EVAL` execution contract, context-aware TE graph banks, and
forward-only PP/VPP capture schedules. NeMo-RL owns fixed logprob packing,
policy/reference role lifecycles, storage-address validation, collective
fallback, stage-specific telemetry, and the persistent 5/20/100-step campaign.
Only one bank is installed at a time, but training, policy-logprob, and
reference-logprob banks are cached independently. Dynamic dropless-MoE expert
work stays eager outside the selected partial graph leaves.

**Tech Stack:** NeMo-RL Python 3.13.13, Bridge/MCore Python 3.12+, Pydantic v2,
PyTorch distributed, Transformer Engine, Megatron-Core, Megatron-Bridge,
NeMo-RL, Ray, SLURM, pytest, uv, W&B, TensorBoard, and static HTML reporting.

## Approved Contract and Baseline

- The approved design is
  `docs/superpowers/specs/2026-08-02-policy-logprob-training-cuda-graph-design.md`.
- Work only in
  `/Users/sna/CudaGraph_PR/RL-thd-cg-hybrid-nemotron-20260731` and its existing
  nested Bridge and MCore worktrees.
- The preserved pre-merge commits are NeMo-RL `f21423769405e2b986edf956ebb1999bd1ec18c8`,
  Bridge `69c29747e85328d7a5ba39f8cbea844d60314b11`, and MCore
  `5d320e339003f5c2820b1ca0a163e1ca44dfb31e`.
- Task 1 re-anchored the implementation baseline to NeMo-RL
  `6ced958f1156e1f70872630a34b135b6360cb449`, Bridge
  `7447569e3602f8d5c2142085bb44dfb7a0c6d046`, and MCore
  `22919a3d7d29e543722acd40f92adf466c8a2a6f`.
- The official-main tips fetched for that re-anchor were NeMo-RL
  `89f4d1f85`, Bridge `573e088c9`, and MCore `42460a7af`.
- Fetch and merge the then-latest official main in dependency order before
  production edits; never reset or recreate the worktrees.
- PR 5672 is reference history, not a branch to cherry-pick over the current
  implementation. Preserve the already ported packed-THD and partial-scope
  invariants and add the missing eval/no-grad execution domain.
- Use Transformer Engine partial graphs, not MCore local generation graphs or
  a process-global `FullCudaGraphWrapper`, for teacher-forced logprob.
- Warmup is exactly three successful calls per graph key. Training fallback is
  always zero. Logprob whole-call eager fallback is allowed only before model
  entry and only when `unseen_key_policy=eager`.
- Start with BF16 policy and reference logprob. FP8/NVFP4 reference replay stays
  fail-closed until extra-state address stability is proven.
- R3 plus router/preprocess logprob stays fail-closed until routed experts are
  explicit fixed-signature graph inputs or bank-owned persistent buffers.
- Checkpoints are disabled in campaign jobs. Use W&B project `sna-cg-study`.
- Commit and push MCore, then Bridge, then NeMo-RL. Every commit uses
  `Signed-off-by`; MCore commits also use the repository-required GPG signing.
- Do not build Transformer Engine natively per job. The immutable nightly
  container must pass the existing runtime attestation before GPU submission.
- The distributed MCore commands in Tasks 2-6 and 12 are test payloads, not
  login-node commands. Submit them through the persistent
  `scripts/run_mcore_scope.sub` in the attested container. Eight global ranks
  is the minimum capability smoke; GB200 uses 2 nodes x 4 GPUs and CW may use
  1 node x 8 GPUs. Required EP16/EP32 rows use 4/8 GB200 nodes x 4 GPUs rather
  than pretending an eight-rank smoke covers those topologies.

## Planned Public and Internal Interfaces

### Megatron-Core

```python
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto


class TECudaGraphExecutionKind(Enum):
    TRAINING = auto()
    FORWARD_ONLY_EVAL = auto()


_ACTIVE_TE_CUDA_GRAPH_EXECUTION_KIND: ContextVar[
    TECudaGraphExecutionKind
] = ContextVar(
    "active_te_cuda_graph_execution_kind",
    default=TECudaGraphExecutionKind.TRAINING,
)


def active_te_cuda_graph_execution_kind() -> TECudaGraphExecutionKind:
    return _ACTIVE_TE_CUDA_GRAPH_EXECUTION_KIND.get()


@contextmanager
def te_cuda_graph_execution_context(
    execution_kind: TECudaGraphExecutionKind,
) -> Iterator[None]:
    if not isinstance(execution_kind, TECudaGraphExecutionKind):
        raise TypeError("execution_kind must be a TECudaGraphExecutionKind")
    token = _ACTIVE_TE_CUDA_GRAPH_EXECUTION_KIND.set(execution_kind)
    try:
        yield
    finally:
        _ACTIVE_TE_CUDA_GRAPH_EXECUTION_KIND.reset(token)
```

The new `execution_kind` and
`cuda_graph_modules: Sequence[str] | None = None` keywords on
`TECudaGraphHelper` carry execution domain and per-helper scope into layer
discovery, schedule construction, capture, and `TECudaGraphBankFingerprint`.
`None` preserves legacy use of the base config; an explicit empty sequence
means whole layer. Bank installation records kind and normalized scope on
every owned graphable layer. Replay requires the installed bank, active
context, module train/eval mode, and grad mode to agree.

### NeMo-RL configuration

```python
class LogprobCudaGraphConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    implementation: Literal["transformer_engine"] = "transformer_engine"
    modules: list[str] = Field(default_factory=list)
    warmup_steps: PositiveInt = 3
    mb_tokens: PositiveInt | None = None
    max_packed_sequences: PositiveInt | None = None
    cache_size: PositiveInt = 2
    roles: list[Literal["policy", "reference"]] = Field(
        default_factory=lambda: ["policy", "reference"]
    )
    unseen_key_policy: Literal["eager", "error"] = "eager"
```

`PolicyConfig` gains only
`logprob_cuda_graph: NotRequired[LogprobCudaGraphConfig]`. One normalization
boundary produces a validated model; consumers use attributes without hidden
defaults.

### NeMo-RL graph identity

```python
class CudaGraphExecutionPath(StrEnum):
    TRAINING = "training"
    POLICY_LOGPROB = "policy_logprob"
    REFERENCE_LOGPROB = "reference_logprob"


@dataclass(frozen=True)
class TECudaGraphScheduleKey:
    execution_path: CudaGraphExecutionPath
    model_storage_generation: int
    grad_storage_generation: int | None
    num_microbatches: int
    mb_tokens: int
    max_packed_sequences: int
    padded_seq_length: int
    topology_fingerprint: str
    schedule_fingerprint: str
    module_scopes: tuple[str, ...]
    dispatcher_fingerprint: str
    router_replay: bool
    precision_fingerprint: str
    packed_metadata_fingerprint: str
```

Dynamic tensor values are copied into static buffers and are not key fields.
Every rank compares a stable serialized key digest before capture, replay, or
fallback.

---

### Task 1: Merge Latest Main and Re-anchor the Three Repositories

**Files:**
- Inspect: `AGENTS.md`
- Inspect: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/AGENTS.md`
- Inspect: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/AGENTS.md`
- Modify only if anchors moved: this plan and its approved design

**Purpose:** Remove branch-age ambiguity before implementation while
preserving every current custom commit and nested gitlink.

- [ ] **Step 1: Prove every worktree is clean and on the intended branch**

```bash
git status --short --branch
git submodule status --recursive
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge status --short --branch
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM status --short --branch
```

Expected: no unrelated changes; branches are
`experiment/thd-cg-hybrid-nemotron-20260731`,
`sna/thd-cg-hybrid-nemotron-20260731`, and
`sj/thd-cg-hybrid-nemotron-20260731`.

- [ ] **Step 2: Fetch without changing the worktrees**

```bash
git fetch origin main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge fetch upstream main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM fetch upstream main
```

- [ ] **Step 3: Merge MCore main, verify it, and push its baseline**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  merge --no-ff --signoff -S upstream/main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  diff --check
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  push origin sj/thd-cg-hybrid-nemotron-20260731
```

Expected: existing partial-THD work remains reachable. Resolve conflicts by
preserving both latest-main behavior and the approved design; do not accept an
entire side mechanically.

- [ ] **Step 4: Pin MCore, merge Bridge main, and push its baseline**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge add 3rdparty/Megatron-LM
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  commit -s -m "build: pin refreshed MCore baseline"
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  merge --no-ff --signoff upstream/main
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge diff --check
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  push origin sna/thd-cg-hybrid-nemotron-20260731
```

If Bridge main changed its MCore gitlink, resolve the merge to the pushed MCore
commit from Step 3, then complete the signed merge commit.

- [ ] **Step 5: Pin Bridge, merge NeMo-RL main, and preserve the custom stack**

```bash
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "build: pin refreshed Megatron Bridge baseline"
git merge --no-ff --signoff origin/main
```

If NeMo-RL main changed its Bridge gitlink, resolve it to the pushed Bridge
commit from Step 4. Do not push NeMo-RL until the re-anchored tests and design
anchors in Steps 6-7 pass.

- [ ] **Step 6: Re-run anchor discovery and record exact SHAs in the campaign metadata**

```bash
rg -n "class TECudaGraphHelper|forward_only=False|_should_call_te_cudagraph" \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer
rg -n "def get_logprobs|def use_reference_model|class TECudaGraphLifecycle" nemo_rl
git rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD
```

- [ ] **Step 7: Verify merge hygiene before production edits and push NeMo-RL**

```bash
git diff --check
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge diff --check
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM diff --check
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

### Task 2: Prove the Pinned TE Eval/No-grad Capability First

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_te_runtime.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_container_runtime.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/bridge_test_matrix.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_bridge_scope.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Modify: all three `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/*.env.example` templates
- Modify: `tests/unit/experiments/test_validate_te_runtime.py`
- Modify: `tests/unit/experiments/test_runtime_attestation.py`
- Modify: `tests/unit/experiments/test_container_harness_hardening.py`
- Create: `tests/unit/experiments/test_mcore_standalone_driver.py`
- Modify: `tests/unit/experiments/test_matrix_submitters.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Purpose:** Establish that the immutable TE build can graph all-eval
callables without backward graphs and document its eval buffer-reuse rule.
The committed runner is implemented before any GPU test. It may stage a
pushed MCore candidate SHA that is newer than the current Bridge gitlink, but
must record both the integration anchor and candidate SHA and must never use
that result as an integrated NeMo-RL attestation. Task 6 pins the validated
MCore SHA into Bridge/NeMo-RL before integrated tests.

- [ ] **Step 1: Build and commit the persistent distributed runner first**

Add the single allowlisted `scripts/run_mcore_training.py` driver, typed
`mcore_test_matrix.json`, and `submit_mcore_matrix.sh`. The manifest rows are
`te_eval_capability_8`, `execution_kind_bank_8`,
`forward_only_schedule_8`, `packed_eval_8`,
`packed_tp2_cp2_pp2_8`, `hybrid_ep16`, `hybrid_ep32`,
`router_replay_8`, and `router_replay_1f1b_8`, each with literal pytest nodes,
filters, and world size. `run_mcore_scope.sub` launches one torchrun agent per node,
supports typed 1x8 CW and 2x4, 4x4, or 8x4 GB200 allocations according to the
selected row, rejects world-size/allocation mismatches, and writes an atomic
attested result under
`RUN_LOG_ROOT/attestations/mcore/<candidate-sha>/<row-id>.json`.
Add the required absolute `RUN_LOG_ROOT` field to every profile template in
this task, before any capability job is submitted.
`run_scope.sh` accepts an explicit pushed
`MCORE_CANDIDATE_SHA`, creates an isolated source snapshot at that commit, and
never mutates the recursively pinned main snapshot.
The login submitter resolves the pushed branch once with `git ls-remote`,
requires one lowercase 40-hex SHA, writes it into the immutable submission
intent, and passes the literal value to every job; jobs never recompute a
moving branch tip.

Add the parallel typed `bridge_test_matrix.json`,
`submit_bridge_matrix.sh`, and `scripts/run_bridge_scope.sub` for the
`bridge_forward_only_eval_8` row. It accepts one pushed
`BRIDGE_CANDIDATE_SHA`, verifies that candidate's nested MCore SHA, and runs
the exact Bridge eval/GPT/preparer tests on eight global ranks. It shares the
same container, topology, provenance, submission-intent, and raw-command
rejection contract as the MCore runner.

Extend the runtime-preflight schema now with TE version/commit, all-eval
callable support, MCore effective eval buffer reuse, observed raw TE reuse,
integration and candidate SHAs, container SHA256, topology, and test-row ID.
Extend `verify_runtime_attestation.py` with a fail-closed matrix mode taking
`--profile-file`, `--candidate-kind`, `--candidate-sha`,
`--test-result-dir`, and one space-delimited `--required-rows` value. Matrix
mode first verifies the profile's immutable runtime attestation with the
existing source/container/TE checks, then requires exactly one passed,
content-bound `<row-id>.json` per requested row under the candidate directory;
unknown or extra rows fail.
Add fail-closed unit tests for raw commands, unknown rows, path escape,
candidate commits absent from the remote, and 8/16/32-rank layouts. Commit and
push this root infrastructure before submitting any GPU allocation.

```bash
uv run pytest -q tests/unit/experiments/test_validate_te_runtime.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_container_harness_hardening.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
bash -n experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_bridge_scope.sub
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_te_runtime.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_container_runtime.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/bridge_test_matrix.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_bridge_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/lyris.env.example \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/oci-hsg.env.example \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/ptyche.env.example \
  tests/unit/experiments/test_validate_te_runtime.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_container_harness_hardening.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "test: add attested distributed MCore runner"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 2: Add a red GPU capability test**

Add `test_te_make_graphed_callables_supports_eval_no_grad()` using one CUDA TE
module in `.eval()` mode, fixed BF16 input, and two distinct input values. The
test module increments a Python-side `forward_invocations` counter on every
real `forward()` entry. Record it after capture; a matched wrapper call must
change the CUDA output without incrementing the Python counter, proving TE
replay rather than eager fallback. A deliberate train/eval mismatch must
increment the counter, proving fallback. Also assert eager/replay parity, no
backward callable use, and no parameter gradients.

```python
with torch.no_grad():
    (graphed,) = make_graphed_callables(
        (module,),
        ((sample,),),
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=False,
    )
    expected = module(actual)
    before_replay = module.forward_invocations
    replayed = graphed(actual)
    assert module.forward_invocations == before_replay
    torch.testing.assert_close(replayed, expected)
assert all(parameter.grad is None for parameter in module.parameters())
```

The `-1` is permitted only if the pinned TE API requires a paired order
descriptor; the test must prove that it does not create or execute backward
work. If TE accepts a forward-only descriptor, use that instead.
Do not use MCore telemetry in this direct TE probe: both replay and fallback
would be zero before an MCore owner exists. Task 3 separately tests MCore
counter ownership around the same callable.

- [ ] **Step 3: Characterize eval output-buffer reuse and enforce the MCore policy**

Add `test_te_eval_graph_input_output_buffer_reuse_capability()` as a direct
probe of the pinned TE API. Record whether TE accepts or rejects
`_reuse_graph_input_output_buffers=True`; do not make an external TE rejection
a design requirement. This probe records the raw TE result only. Task 4 adds
the MCore assertion that every `FORWARD_ONLY_EVAL` helper passes false. A
`te_capability` artifact with MCore status `not_implemented` cannot authorize a
production leaf; the later integrated attestation must record
`mcore_eval_reuse_graph_io=false` even if raw TE accepts true.

The `te_eval_capability_8` manifest row selects exactly these fully qualified
nodes from `tests/unit_tests/transformer/test_cuda_graphs.py`:
`test_te_make_graphed_callables_supports_eval_no_grad` and
`test_te_eval_graph_input_output_buffer_reuse_capability`. The result schema
records each node separately; one passing node cannot mask the other's failure.

- [ ] **Step 4: Commit and push the standalone MCore probe**

Run this block from
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`:

```bash
uv run isort tests/unit_tests/transformer/test_cuda_graphs.py
git add tests/unit_tests/transformer/test_cuda_graphs.py
git commit -S -s -m "test: characterize forward-only TE graph capability"
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 5: Submit the exact attested capability row**

Run from the NeMo-RL root after pulling the pushed root runner on the selected
cluster:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS=te_eval_capability_8 \
  "$EXP/submit_mcore_matrix.sh"
```

Monitor for five minutes, then require the attested result to show eight joined
ranks, no idle allocation, direct TE replay parity, changed outputs, no grads,
and the observed reuse behavior. If it fails, stop implementation and keep
logprob graphs unsupported; never emulate eval while the module remains in
training mode. Preserve the capability artifact by SHA, but do not pin this
test-only MCore commit into Bridge yet.

### Task 3: Add Explicit MCore Execution Kinds and Context-aware Banks

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/te_cuda_graph_bank.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/module.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:** Add `TECudaGraphExecutionKind`,
`te_cuda_graph_execution_context()`, `active_te_cuda_graph_execution_kind()`,
`execution_kind`, and normalized per-helper `cuda_graph_modules` on
`TECudaGraphBankFingerprint`. Tasks 3-5 command blocks
run from
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`.

- [ ] **Step 1: Add red bank-isolation and wrong-mode tests**

Add exact test nodes
`test_bank_fingerprint_distinguishes_training_and_forward_only_eval`,
`test_bank_activation_installs_and_restores_execution_kind`,
`test_eval_bank_replay_requires_eval_no_grad_context`,
`test_training_bank_replay_rejects_forward_only_context`,
`test_eval_bank_rejects_backward_dw_launch`, and
`test_failed_context_install_rolls_back_every_layer`. Also add
`test_runtime_microbatch_provider_tracks_active_schedule_key` and
`test_pp1_eval_normalizes_graph_count_without_losing_call_geometry`,
`test_bank_scopes_may_use_ordered_subsets_of_one_manager_topology`, and
`test_capture_session_does_not_require_an_installed_bank`,
`test_bank_switch_installs_layer_active_scopes_and_effective_overlap`, and
`test_layer_switches_training_attn_overlap_to_eval_router_no_overlap`. Each test
asserts the named mode, ownership, schedule-count, or rollback invariant
directly.

The `execution_kind_bank_8` manifest row stores the fully qualified node for
every exact test named above from
`tests/unit_tests/transformer/test_te_cuda_graph_bank.py`; it does not use a
broad `-k` expression in the submitted GPU row.

Run the deterministic single-process unit subset and confirm failure because no
execution-kind contract exists. The later eight-rank proof must use the typed
`execution_kind_bank_8` row through `submit_mcore_matrix.sh`; never replace it
with an ad hoc distributed command:

```bash
uv run pytest -q tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  -k 'execution_kind or forward_only or wrong_mode'
```

- [ ] **Step 2: Implement the enum and nest-safe context manager**

Use a `ContextVar` token so nested contexts restore their exact predecessor.
Reject non-enum values. Preserve legacy callers by treating an absent explicit
context as `TRAINING` only. `FORWARD_ONLY_EVAL` requires eval mode and `not
torch.is_grad_enabled()` at replay. `TRAINING` requires module training mode
but must allow a no-grad reentrant-checkpoint initial forward.

- [ ] **Step 3: Extend transactional bank ownership**

Include execution kind in the fingerprint, saved `_LayerInstallation`, bank
registration, replay guard, execution-counter ownership, install, uninstall,
and reset paths. Installing an eval bank must not destroy a cached training
bank. A failed multi-layer install restores all attributes and the prior bank.
Keep exactly one manager for the union topology of every graphable layer.
Move scopes from manager-global state into each bank fingerprint and allow a
helper's ordered layer set to be a validated subset of the manager topology.
Activation clears graph surfaces on union layers absent from that bank.
Execution counters count only the active bank subset. Retain the current
no-argument mutable runtime-count provider: the caller sets the collectively
agreed normalized count for the exact training, policy, or reference key
before capture/activation. PP1 without overlap returns one even when the
logprob call contains multiple logical microbatches.

Do not rely on the process-wide `TransformerConfig.cuda_graph_modules` or
`overlap_moe_expert_parallel_comm` after a bank/capture session is selected.
Install nest-safe per-layer active scopes and one active
`effective_overlap_moe` value from the capture session or bank. Refactor every
TransformerLayer graph branch that selects the capture boundary, interprets
the output tuple, or takes a partial early return to use those active values.
Uninstall/rollback restores the prior layer surfaces transactionally. This is
required for a training `attn`/overlap-on bank to coexist with an eval
`moe_router`/overlap-off bank under one immutable base config.

- [ ] **Step 4: Replace the implicit module gate**

Split capture and replay predicates. During capture, a temporary transactional
capture session records manager owner, execution kind, scopes, and selected
layer IDs before any bank exists. During replay,
`GraphableMegatronModule._should_call_te_cudagraph()` requires:

```text
config implementation is transformer_engine
AND an exact bank is installed
AND active execution context equals installed bank kind
AND TRAINING implies module.training
OR FORWARD_ONLY_EVAL implies not module.training and grad disabled
```

The capture path requires the exact capture-session owner plus
`is_graph_capturing()`, not an installed bank. NeMo/Bridge count the three
full-call eager warmups separately. The post-capture eligible-call counter
follows the replay predicate. Backward-DW wrappers accept only `TRAINING`.

- [ ] **Step 5: Format, commit, push, and submit the focused MCore row**

```bash
uv run isort megatron/core/transformer/te_cuda_graph_bank.py \
  megatron/core/transformer/module.py \
  megatron/core/transformer/transformer_layer.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git diff --check
git add megatron/core/transformer/te_cuda_graph_bank.py \
  megatron/core/transformer/module.py \
  megatron/core/transformer/transformer_layer.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -S -s -m "feat: separate TE graph execution domains"
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

From the NeMo-RL root, submit the pushed candidate without changing gitlinks:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS=execution_kind_bank_8 \
  "$EXP/submit_mcore_matrix.sh"
```

Require the attested row to pass before Task 4. If it fails, add a new signed
fix commit, push, and resubmit the new SHA; never rewrite a SHA already named
by an artifact.

### Task 4: Build a Forward-only PP/VPP TE Capture Schedule

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py` only if a reusable forward-order helper is absent after the main merge
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/a2a_overlap/test_cuda_graphed_schedule_chunk_1f1b.py`

**Interfaces:** `TECudaGraphHelper` gains the keyword
`execution_kind: TECudaGraphExecutionKind = TECudaGraphExecutionKind.TRAINING`
and `cuda_graph_modules: Sequence[str] | None = None`, and remains backward
compatible. An explicit helper scope, including the empty whole-layer scope,
drives layer discovery and sample-boundary construction without mutating
`TransformerConfig`; only `None` reads the legacy config field.
`_get_cuda_graph_input_data()` derives `forward_only` from that explicit kind.

- [ ] **Step 1: Add red schedule tests**

Add exact test nodes
`test_forward_only_helper_uses_eval_pp_vpp_schedule`,
`test_helpers_discover_layers_from_independent_explicit_scopes`,
`test_forward_only_helper_captures_no_backward_graphs`,
`test_forward_only_helper_disables_graph_io_buffer_reuse`,
`test_forward_only_helper_rejects_training_modules`, and
`test_forward_only_helper_does_not_expand_overlap_moe_schedule`. Add
`test_forward_only_helper_uses_effective_overlap_false_everywhere` to cover
PP1 normalization, graph mapping, requested graph count, and manager runtime
normalization. Add
`test_set_current_microbatch_uses_active_bank_subset_not_base_scope` and a
PP2/VPP2 training-attn/eval-router alternating-scope replay-index test named
`test_active_bank_subset_controls_pp2_vpp2_replay_index`.

Use PP1, PP2, and PP2/VPP2 parameterization. Assert the logical execution
order has one forward visit per microbatch/chunk and no backward work. If TE
requires negative order markers as descriptors, separately assert that no
backward callable or backward-DW graph is produced.

The `forward_only_schedule_8` manifest row contains these literal nodes from
`tests/unit_tests/transformer/test_cuda_graphs.py`:
`test_forward_only_helper_uses_eval_pp_vpp_schedule`,
`test_forward_only_helper_captures_no_backward_graphs`,
`test_forward_only_helper_disables_graph_io_buffer_reuse`,
`test_forward_only_helper_uses_effective_overlap_false_everywhere`,
`test_set_current_microbatch_uses_active_bank_subset_not_base_scope`, and
`test_active_bank_subset_controls_pp2_vpp2_replay_index`. It may include the
other named Task 4 nodes but may not replace these with a broad `-k` filter.

- [ ] **Step 2: Thread execution kind through helper construction and capture**

Training retains `forward_only=False`, overlap-MoE expansion, and training
buffer reuse. Eval uses `forward_only=True`, no overlap-MoE schedule expansion,
no backward-DW graph, and `_reuse_graph_input_output_buffers=False`. Derive one
`effective_overlap = config.overlap_moe_expert_parallel_comm and execution_kind
is TRAINING` value and use it in PP1 normalization, schedule expansion,
graph-to-layer mapping, legacy requested count, and bank runtime validation.
Pass that same value into the Task 3 capture-session/bank layer surface; no
eval branch may read the immutable base overlap flag directly.

Refactor `set_current_microbatch()` and every runtime graph-index update to
iterate the active capture-session/bank layer descriptors, or the manager's
scope-independent union filtered by the active bank, never rediscover leaves
from `model_with_decoder.config`. Update current-microbatch and R3 schedule
slots for eval-only Mamba/router leaves before PP/VPP replay and clear inactive
leaves on bank switch.

- [ ] **Step 3: Freeze module and grad mode during capture**

Capture eval banks only after all graphable leaves are `.eval()` and inside
`torch.no_grad()` plus `te_cuda_graph_execution_context(FORWARD_ONLY_EVAL)`.
Capture training banks only in `.train()` with grad enabled. Validate every
leaf before entering TE.

- [ ] **Step 4: Update MCore validation text**

Document that TE partial graphs support training and teacher-forced
forward-only eval through separate owners. Keep `inference_cuda_graph_scope`
reserved for local dynamic inference.

- [ ] **Step 5: Format, commit, push, and submit the schedule row**

```bash
uv run isort megatron/core/transformer/cuda_graphs.py \
  megatron/core/transformer/transformer_config.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/a2a_overlap/test_cuda_graphed_schedule_chunk_1f1b.py
git diff --check
git add megatron/core/transformer/cuda_graphs.py \
  megatron/core/pipeline_parallel/schedules.py \
  megatron/core/transformer/transformer_config.py \
  tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/a2a_overlap/test_cuda_graphed_schedule_chunk_1f1b.py
git commit -S -s -m "feat: capture forward-only TE graph schedules"
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

Do not add `schedules.py` to the commit if no source change was required.
From the NeMo-RL root, submit `MCORE_TEST_ROWS=forward_only_schedule_8` with
the pushed `MCORE_CANDIDATE_SHA`, selected `CLUSTER`, and `PROFILE_FILE` through
`submit_mcore_matrix.sh`. Require an attested pass before Task 5; failures get
a new signed commit and result SHA.

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS=forward_only_schedule_8 \
  "$EXP/submit_mcore_matrix.sh"
```

### Task 5: Validate Packed THD, Mamba, and Partial-MoE Eval Scopes

**Capacity-policy supersession (2026-08-04):** The fixed-capacity requirement
for production Flex/HybridEP `moe_router+moe_preprocess` in this task is
superseded by
`docs/superpowers/plans/2026-08-04-dropless-partial-moe-capacity-policy.md`.
That focused plan requires a candidate-SHA-bound distributed changed-route
gate before NeMo-RL permits dropless partial HybridEP. Whole-MoE and
whole-layer capture still require validated fixed capacity and zero-drop
telemetry. Follow the focused plan first, then resume the forward-only logprob
tasks here.

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/packed_seq_params.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/ssm/mamba_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/cuda_graph_replay.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/ssm/test_mamba_layer.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_token_dispatcher.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`

**Purpose:** Reuse the current static packed descriptors for forward-only
capture without widening the dropless-MoE graph boundary.

- [ ] **Step 1: Add red execution-kind parameterization**

Parameterize existing packed attention, Mamba, router, preprocess, dispatcher
snapshot, and partial-flow contracts over `TRAINING` and
`FORWARD_ONLY_EVAL`. Add alternating logical occupancy under the same physical
signature. Contract tests alone do not satisfy this task's GPU gate.

- [ ] **Step 2: Preserve fixed physical THD metadata**

Eval uses the same capacity-sized cumulative lengths, structural mask,
`seq_idx`, CP/SP transform, dummy sequence, padding semantics, and exact tensor
signature checks as training. Never key a bank by route values or logical
occupancy.

- [ ] **Step 3: Keep dynamic MoE work eager**

For dropless MoE, graph only `moe_router` and supported
`moe_preprocess`. Dispatch communication, expert grouped GEMM, combine, and
postprocess remain eager. Keep DeepEP and NCCL-EP preprocess rejected;
HybridEP is enabled only under the already verified fixed-capacity contract.

- [ ] **Step 4: Add 20-replay parity assertions**

Implement the main changed-value matrix as the literal parameterized node
`test_packed_eval_partial_scopes_replay_changed_values`. For `attn`, `mlp`,
`attn,mlp`, `attn,mamba`, `moe_router`,
`moe_router,moe_preprocess`, and
`attn,mamba,moe_router,moe_preprocess`, compare eager versus graph valid-token
outputs, router IDs/counts, router probabilities, and padding behavior across
20 changed inputs. Run an actual HybridStack through
`GraphableMegatronModule.__call__`, the bank guard, and real TE callables;
assert `graph_calls > 0` on every selected scope so eager fallback cannot pass
the parity test. R3 is off in this task. Parameterize supported shared-expert
off/on rows and run explicit TP2, CP2, PP2, PP2/VPP2, and the model-required
EP topology. Every row records its dispatcher, TP/PP/CP/EP, shared-expert
state, execution kind, and physical packed signature; an unsupported row must
fail with its exact capability reason rather than disappear from the matrix.
Name the topology matrix
`test_packed_eval_partial_scopes_match_eager_across_parallel_topologies`, the
shared-expert matrix `test_packed_eval_shared_expert_scope_contract`, and the
fixed-capacity HybridEP matrix
`test_hybrid_ep_fixed_capacity_eval_scope` with literal parameter IDs `ep16`
and `ep32`. Add an alternating-bank row named
`test_training_eval_bank_switch_preserves_scope_and_overlap` that replays
training `attn` with base
overlap-MoE enabled, eval `moe_router` with effective overlap disabled, and
training `attn` again. Assert correct output arity/early-return boundary,
nonzero calls for only the active scope, and eager parity after every switch.

Map manifest rows to literal nodes: `packed_eval_8` selects
`test_packed_eval_partial_scopes_replay_changed_values` and
`test_packed_eval_shared_expert_scope_contract`;
`packed_tp2_cp2_pp2_8` selects
`test_packed_eval_partial_scopes_match_eager_across_parallel_topologies` and
`test_training_eval_bank_switch_preserves_scope_and_overlap`;
`hybrid_ep16` selects
`test_hybrid_ep_fixed_capacity_eval_scope[ep16]`; and `hybrid_ep32` selects
`test_hybrid_ep_fixed_capacity_eval_scope[ep32]`. Store the full pytest node
path in the typed manifest; no row may use a broad substring filter in place
of these nodes.

- [ ] **Step 5: Format, commit, push, and submit the topology rows**

```bash
uv run isort megatron/core/packed_seq_params.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/moe/cuda_graph_replay.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git diff --check
git add megatron/core/packed_seq_params.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/ssm/mamba_layer.py \
  megatron/core/transformer/moe/cuda_graph_replay.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py \
  tests/unit_tests/ssm/test_mamba_layer.py \
  tests/unit_tests/transformer/moe/test_token_dispatcher.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -S -s -m "test: prove packed eval partial graph scopes"
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

Only commit production files that required a real change after
parameterization exposed a missing invariant.

From the NeMo-RL root, re-run the complete seven-row pre-pin gate against this
final Task 5 candidate SHA. Earlier Task 2-4 artifacts remain useful failure
localization but cannot attest a newer commit. The submitter fans rows out
independently and does not run EP16/EP32 inside an eight-rank allocation.

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS='te_eval_capability_8 execution_kind_bank_8 forward_only_schedule_8 packed_eval_8 packed_tp2_cp2_pp2_8 hybrid_ep16 hybrid_ep32' \
  "$EXP/submit_mcore_matrix.sh"
```

All seven artifacts must therefore exist under the same final candidate-SHA
directory and pass before Bridge pins this MCore candidate in Task 6.

### Task 6: Add Bridge Standalone Forward-only Ownership and Pin MCore

**Files:**
- Create: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/cuda_graph_runtime.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/config.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/pretrain.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/state.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/forward_step_func_types.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/gpt_step.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/utils/train_utils.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/train.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/eval.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/utils/cuda_graph.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_pretrain.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_state.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_eval.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_gpt_step.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/utils/test_train_utils.py`
- Create: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_forward_only_cuda_graph_distributed.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_config.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`

**Interfaces:** Add a `TECudaGraphRuntimeOwner` on `GlobalState` that shares
one MCore bank manager and holds independent training/eval lifecycles for the
entire `pretrain()` call. Add this independent Bridge configuration contract
in `training/config.py` and expose it as
`ConfigContainer.eval_cuda_graph`; never read the training model's
`cuda_graph_modules` as an eval default:

```python
@dataclass(kw_only=True)
class EvalTECudaGraphConfig:
    enabled: bool = False
    modules: tuple[str, ...] = ()
    warmup_steps: int = 3
```

An explicit empty `modules` tuple requests a capability-gated whole-layer eval
graph. Validation requires `warmup_steps == 3`, TE implementation, a typed
sample preparer, fixed packed geometry, and model-present scopes whenever
`enabled`; disabled config creates no eval owner. Training and eval module
lists may differ and neither mutates the other. Add a `PreparedEvalMicrobatch`
value containing the
rank-local, CP-partitioned model inputs, `PackedSeqParams`, model-chunk/VPP
identity, and a restored iterator that still yields the peeked batch exactly
once. It activates eval after `.eval()` and restores the previous bank after
`.train()` in a `finally` block.
Steps 3 and 5 run from
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`. Steps 4, 6, and 7 run
from the NeMo-RL root as stated in those steps.

- [ ] **Step 1: Add red Bridge lifecycle tests**

Add exact test nodes
`test_evaluate_activates_forward_only_bank_after_model_eval`,
`test_evaluate_restores_training_bank_in_finally`,
`test_evaluate_failure_restores_training_mode_and_bank`,
`test_packed_eval_peek_preserves_rank_local_cp_partitioned_microbatch`,
`test_packed_eval_middle_pipeline_stage_uses_local_sample_signature`,
`test_packed_eval_vpp_chunks_preserve_first_microbatch_once`,
`test_eval_bank_fingerprint_rejects_changed_cp_or_process_groups`,
`test_final_eval_can_use_owner_after_training_cleanup`,
`test_skip_train_eval_can_capture_forward_only_bank`, and
`test_eval_graph_rejects_forward_step_without_sample_preparer`,
`test_gpt_sample_preparer_does_not_call_model_or_accumulate_flops`, and
`test_local_inference_scope_is_not_used_for_teacher_forced_eval`. Extend the
Nano, Super, and Ultra recipe tests to prove their attention, Mamba,
router/preprocess, dispatcher, and fixed-capacity combinations either satisfy
the forward-only contract or fail with the exact unsupported capability. Eval
TE graphs are opt-in and default off in Bridge configuration; warmup is
exactly three successful full calls.

Add exact config nodes
`test_eval_cuda_graph_config_defaults_disabled_and_independent`,
`test_eval_cuda_graph_config_rejects_training_scope_inheritance`, and
`test_eval_cuda_graph_config_requires_three_warmups_and_sample_preparer`.

Add one dedicated eight-rank GPU node in
`test_forward_only_cuda_graph_distributed.py` named
`test_bridge_forward_only_eval_pp2_vpp2_real_te`. It constructs real TE-backed
model chunks, runs Bridge `evaluate()` through PP2/VPP2 with packed changed
inputs, performs three warmups and 20 replays, and asserts eager parity,
`graph_calls > 0`, no backward graph/gradients, and exact mode/bank/iterator
restoration. The `bridge_forward_only_eval_8` manifest row selects the literal
fully qualified node
`tests/unit_tests/training/test_forward_only_cuda_graph_distributed.py::test_bridge_forward_only_eval_pp2_vpp2_real_te`;
running mocked unit tests eight times cannot satisfy the gate.

- [ ] **Step 2: Implement the focused runtime owner**

Do not add another execution counter manager to the same graphable leaves.
`pretrain()` constructs the owner after model/process-group setup, stores it on
`GlobalState`, and destroys it only after final validation/test or on the
outermost failure path. Training-loop cleanup must not destroy it because
`skip_train` and final evaluation still need it. The owner creates one union
manager, captures kind-specific banks, switches only at drained barriers, and
never infers execution kind from optimizer presence.

The forward-only preparation API consumes the first batch through the same
Bridge `forward_step_func` preparation, CP partition, PP-rank filtering, and
VPP chunk routing as the real schedule. It returns a
`PreparedEvalMicrobatch`; it never captures from the raw dataset batch or
`None`. Its restored iterator yields the prepared first batch once and then
continues the original iterator. Capture and replay fingerprints include the
effective eval PP/CP/VPP process-group topology and reject a different eval-CP
rebinding before model entry.

Do not infer a preparation hook with `hasattr`. Extend
`forward_step_func_types.py` with a typed `CudaGraphSamplePreparer` protocol
and make `prepare_forward_step_func()` return a typed wrapper containing the
callable plus an optional preparer. Refactor canonical `gpt_step.py` batch/CP/
packed preparation into a pure preparer that performs no model call, FLOP
accounting, loss accumulation, or iterator advance beyond the replayable
peek. Arbitrary user forward steps without that protocol fail setup only when
forward-only TE graphs are enabled; their eager evaluation behavior remains
unchanged.

Wrap training schedules in `te_cuda_graph_execution_context(TRAINING)` and
the complete eval schedule in
`te_cuda_graph_execution_context(FORWARD_ONLY_EVAL)`. `evaluate()` performs
three successful eager warmups, refreshes any manual forward-step hooks before
capture, and releases those hooks on success or failure. Wrap the whole body,
including time-limit early return, in `try/finally` so model mode, rerun mode,
hooks, and prior bank are restored in that order.

- [ ] **Step 3: Run Bridge unit tests**

```bash
uv run python -m pytest -q tests/unit_tests/training/test_eval.py \
  tests/unit_tests/training/test_gpt_step.py \
  tests/unit_tests/training/utils/test_train_utils.py \
  tests/unit_tests/training/test_pretrain.py \
  tests/unit_tests/training/test_state.py \
  tests/unit_tests/training/test_config.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py
```

- [ ] **Step 4: Verify the attested MCore gates before pinning**

Re-run `verify_runtime_attestation.py` against the exact Task 3
`execution_kind_bank_8`, Task 4 `forward_only_schedule_8`, and Task 5
`packed_eval_8`/topology artifacts. Require the final pushed MCore candidate
SHA, TE/container identity, nonzero graph calls, no backward eval graphs, and
eager parity to match. Do not run a raw local `torchrun` command or substitute
an eight-rank artifact for EP16/EP32.

Run from the NeMo-RL root with the same selected profile used for submission:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
. "$PROFILE_FILE"
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
uv run --locked python "$EXP/verify_runtime_attestation.py" \
  --profile-file "$PROFILE_FILE" \
  --candidate-kind mcore \
  --candidate-sha "$MCORE_CANDIDATE_SHA" \
  --test-result-dir "$RUN_LOG_ROOT/attestations/mcore/$MCORE_CANDIDATE_SHA" \
  --required-rows 'te_eval_capability_8 execution_kind_bank_8 forward_only_schedule_8 packed_eval_8 packed_tp2_cp2_pp2_8 hybrid_ep16 hybrid_ep32'
```

Expected: exit zero and one canonical JSON summary binding all seven row
digests. The verifier rejects a missing row, wrong candidate SHA, TE/container
mismatch, nonzero backward-eval count, zero graph calls, parity failure, or a
world size other than the typed manifest value.

- [ ] **Step 5: Push MCore, pin it in Bridge, then commit/push Bridge**

```bash
git -C 3rdparty/Megatron-LM push origin sj/thd-cg-hybrid-nemotron-20260731
uv run pre-commit run --all-files
git add 3rdparty/Megatron-LM src/megatron/bridge/training/cuda_graph_runtime.py \
  src/megatron/bridge/training/config.py \
  src/megatron/bridge/training/pretrain.py \
  src/megatron/bridge/training/state.py \
  src/megatron/bridge/training/forward_step_func_types.py \
  src/megatron/bridge/training/gpt_step.py \
  src/megatron/bridge/training/utils/train_utils.py \
  src/megatron/bridge/training/train.py src/megatron/bridge/training/eval.py \
  src/megatron/bridge/utils/cuda_graph.py \
  tests/unit_tests/training/test_pretrain.py tests/unit_tests/training/test_state.py \
  tests/unit_tests/training/test_eval.py tests/unit_tests/training/test_gpt_step.py \
  tests/unit_tests/training/utils/test_train_utils.py \
  tests/unit_tests/training/test_forward_only_cuda_graph_distributed.py \
  tests/unit_tests/training/test_config.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_nano.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_super.py \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_ultra.py
git commit -s -m "feat: own forward-only TE graph banks"
git push origin sna/thd-cg-hybrid-nemotron-20260731
```

Run the first command from the Bridge root; the remaining commands also run
from the Bridge root.

- [ ] **Step 6: Submit the pushed Bridge eval integration row**

From the NeMo-RL root on the selected cluster, resolve the pushed Bridge
candidate once and submit its typed distributed row:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
BRIDGE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  ls-remote origin refs/heads/sna/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  BRIDGE_CANDIDATE_SHA="$BRIDGE_CANDIDATE_SHA" \
  BRIDGE_TEST_ROWS=bridge_forward_only_eval_8 \
  "$EXP/submit_bridge_matrix.sh"
```

Require eight joined ranks, the exact nested MCore SHA already cleared by
Task 5, nonzero forward-only graph calls, eager parity, no backward eval graph,
and exercised `evaluate()`/preparer/restore paths. A raw MCore-only test cannot
satisfy this Bridge gate.

- [ ] **Step 7: Immediately pin the validated Bridge commit in NeMo-RL**

Run from the NeMo-RL root before making any NeMo-RL production edit:

```bash
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "build: pin forward-only Bridge graph runtime"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

Verify recursively that the NeMo gitlink resolves to the pushed Bridge commit
and that Bridge resolves to the pushed MCore commit. This is the first pin
cycle; Task 12 performs a second cycle for the R3 buffer fix.

### Task 7: Add and Normalize the NeMo-RL Logprob Graph Configuration

**Files:**
- Create: `nemo_rl/models/policy/cuda_graph_config.py`
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/teacher_worker_group.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `pyrefly.toml`
- Modify: `examples/configs/grpo_math_1B.yaml`
- Modify: `tests/unit/reference_configs/grpo_math_1B.yaml`
- Create: `tests/unit/models/policy/test_logprob_cuda_graph_config.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`
- Modify: `tests/unit/test_config_v2.py`

**Interfaces:** Implement the approved Pydantic model plus
`normalize_logprob_cuda_graph_config(policy_config)`. Call it exactly once in
`Policy.__init__` before workers are constructed for schema/default validation.
`_EffectiveTECudaGraphConfig` stores the validated object and exact enabled
role set. Separately call
`validate_logprob_cuda_graph_runtime_capability()` in each Megatron worker only
after `validate_and_set_config()` resolves model modules, dispatcher,
precision, and topology; parsing must not guess runtime capability.

- [ ] **Step 1: Add red model/default/unknown-key tests**

Test disabled defaults, explicit policy-only/reference-only roles, duplicate
roles, an empty enabled role list, duplicate scopes, unknown fields, invalid
router/preprocess combinations, and `moe`/`moe_router` exclusion. Preserve the
approved empty-module-list meaning: it requests a whole-layer graph and is
accepted only when the complete layer has fixed geometry. Add
`test_empty_modules_selects_capability_gated_whole_layer` and
`test_empty_enabled_roles_is_rejected` so the two empty-list meanings cannot
be conflated.

- [ ] **Step 2: Add red backend and geometry validation tests**

Enabled logprob graphs must reject non-Megatron policy, non-TE
implementation, warmup other than 3, dynamic batching, disabled packing,
non-fused attention, missing capacities, misalignment, unsupported dispatcher,
unsupported precision, and R3/router combinations.

Encode and test this initial capability table rather than scattering
conditionals:

| Precision / dispatcher / R3 | Initially enabled logprob scopes |
|---|---|
| BF16 / AllToAll / off | model-present `attn`, `mlp`, `mamba`, `moe_router`, and fixed `moe_preprocess` combinations |
| BF16 / HybridEP / off | model-present non-MoE leaves and `moe_router`; `moe_preprocess` only when the selector proves fixed capacity |
| BF16 / DeepEP or NCCL-EP / off | model-present non-MoE leaves; `moe_preprocess` rejected |
| BF16 / any / on, before Task 12 | model-present `attn`, `mlp`, and `mamba`; router/preprocess rejected |
| FP8 or NVFP4 / any / either | logprob graph rejected; eager logprob remains available |

Whole-layer (`modules=[]`) and whole-MoE rows require a separate fixed-geometry
capability bit. Model-missing scopes and every combination outside this table
fail during worker capability validation with one stable reason code.

`TeacherWorkerGroup` can construct workers without `Policy.__init__`. It must
explicitly normalize the provided block and pass it to Megatron workers for
the same post-override capability validation; non-Megatron teacher workers
force the path to disabled/N/A. No teacher path may inherit an implicit
enabled default. Add direct-construction tests for enabled Megatron reference,
disabled non-Megatron reference, and scope rejection after runtime overrides.

- [ ] **Step 3: Implement centralized parsing and validation**

The absent legacy field becomes `LogprobCudaGraphConfig()` only at the Policy
boundary. Do not use `.get()` with fallback defaults for any config-model field after
normalization. Keep training `cuda_graph_modules` and logprob `modules`
independent.

- [ ] **Step 4: Add discoverable disabled exemplar values**

Add the full block with approved defaults to both `grpo_math_1B.yaml` files.
Create enabled campaign recipes later in Task 14; do not silently enable base
performance recipes.

- [ ] **Step 5: Run config tests and commit**

```bash
uv run pytest -q tests/unit/models/policy/test_logprob_cuda_graph_config.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/test_config_v2.py
git diff --check
git add nemo_rl/models/policy/cuda_graph_config.py nemo_rl/models/policy/__init__.py \
  nemo_rl/models/policy/lm_policy.py \
  nemo_rl/models/policy/teacher_worker_group.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/megatron/setup.py \
  examples/configs/grpo_math_1B.yaml tests/unit/reference_configs/grpo_math_1B.yaml \
  tests/unit/models/policy/test_logprob_cuda_graph_config.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/test_config_v2.py \
  pyrefly.toml
git commit -s -m "feat: configure logprob TE partial graphs"
```

### Task 8: Reuse Fixed Packed-THD Geometry for Logprob Static Buffers

**Files:**
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `tests/unit/models/megatron/test_megatron_data.py`
- Modify: `tests/unit/models/policy/test_cuda_graph_policy_packing.py`

**Interfaces:** Replace the training-only boolean at the internal iterator
boundary with a typed execution path. `CudaGraphStaticCopyResult` is a frozen
dataclass with `tensor_count: int`, `copied_bytes: int`, and
`elapsed_seconds: float`. Split one processed call into
`CudaGraphPhysicalInputs` and `CallLogicalPackedMetadata`.
`CudaGraphStaticBuffers` owns only fixed-capacity model-facing tensors:
CP-sharded input IDs, labels, loss/attention masks, position IDs, the
capacity-sized `PackedSeqParams` arrays of exactly
`max_packed_sequences + 1`, fixed structural masks, MTP mask, and fixed route
source tensors. Its exact method is
`copy_from(source: CudaGraphPhysicalInputs) -> CudaGraphStaticCopyResult`.

The original `data_dict`, real `cu_seqlens*` arrays whose length is
`real_sequence_count + 1`, logical slice indices, and right-padding metadata
remain call-owned in `CallLogicalPackedMetadata`; they are never copied into a
graph owner or passed to a captured leaf. A `GraphReplayMicrobatch` pairs the
selected bank's static physical tensors with that call's logical metadata for
eager output postprocessing. A new internal `TECudaGraphOwnedBank` dataclass
owns one MCore bank, its static buffers, schedule key, and storage fingerprint
for the entire LRU lifetime. The input iterator owns no graph storage.

- [ ] **Step 1: Add red fixed-logprob geometry tests**

Cover policy and reference paths, disabled eager behavior, exact
`mb_tokens`, `max_packed_sequences + 1` cumulative-length entries, the one
dummy sequence, alternating occupancy, CP2 zigzag, SP slicing, and last-output
unpadding. Assert real `cu_seqlens*` shapes vary with occupancy while every
capacity model tensor and pointer remains fixed. Replace the current tests that assert logprob never receives graph
training capacity with enabled-versus-disabled parameterization.

- [ ] **Step 2: Add red static-copy tests**

Assert in-place copy preserves every destination pointer while updating token,
position, label, loss-mask, structural-mask, packed metadata, and optional R3
tensors. Reject shape, stride, dtype, device, layout, and static metadata
mismatches before any partial copy.

Implement `copy_from` as two passes: validate every physical field tensor and
non-Tensor capacity attribute without mutation, then perform deterministic
in-place copies. Add a negative test proving the variable-shape original
`data_dict` and real cumulative-length tensors cannot enter `copy_from`.
Assert eviction/reset releases the owner once and a failed validation leaves
every destination byte unchanged.

- [ ] **Step 3: Share the training packer**

Generalize the existing validation wrapper and call
`_pack_sequences_for_megatron_with_geometry`, `pad_sequence_for_thd`, and the
current CP/SP helpers for both execution kinds. Logprob graph capacity comes
only from `logprob_cuda_graph.mb_tokens`; never infer it from
`logprob_batch_size` or eager `sequence_packing.logprob_mb_tokens`.

- [ ] **Step 4: Propagate exact role identity through LM and TQ sharding**

Use `policy_logprob` for `prev_lp` and `reference_logprob` for `ref_lp`.
Return the logical slice metadata needed to reconstruct right-padded logprobs.

- [ ] **Step 5: Run packing tests and commit**

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/policy/test_cuda_graph_policy_packing.py
git diff --check
git add nemo_rl/models/megatron/data.py nemo_rl/models/policy/lm_policy.py \
  nemo_rl/models/policy/tq_policy.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/policy/test_cuda_graph_policy_packing.py
git commit -s -m "feat: add fixed logprob graph geometry"
```

### Task 9: Generalize NeMo-RL Lifecycle Keys, Warmup, and Collective Fallback

**Files:**
- Create: `nemo_rl/models/policy/cuda_graph_metrics.py`
- Modify: `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_cuda_graph_lifecycle.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`
- Modify: `pyrefly.toml`

**Interfaces:** Define `CudaGraphExecutionPath`, `CudaGraphPathMetrics`, and
`CudaGraphMetricsProvider` once in
`nemo_rl/models/policy/cuda_graph_metrics.py`. Add the expanded frozen schedule
key, a stable SHA256 key serializer, per-key successful-call warmup, and an
internal path-aware `CudaGraphMetricsAccumulator` that produces the public
snapshot/ack payload. Share one MCore bank manager across every key.

Use two cache partitions beneath that manager: the existing two-entry training
LRU and one forward-only LRU with `logprob_cuda_graph.cache_size` total entries
shared by policy and reference. Thus the canonical training, policy-logprob,
and reference-logprob banks do not evict one another merely because all three
paths are enabled.

- [ ] **Step 1: Add red domain and LRU tests**

Add exact test nodes
`test_policy_and_reference_warmup_are_independent`,
`test_training_and_logprob_keys_never_share_a_bank`,
`test_same_fixed_signature_with_changed_values_hits`,
`test_forward_only_two_entry_lru_holds_policy_and_reference`,
`test_training_lru_is_independent_from_forward_only_lru`,
`test_training_rejects_any_fallback`,
`test_logprob_unseen_key_selects_collective_whole_call_fallback`,
`test_logprob_unseen_key_error_policy_collectively_raises_before_forward`,
`test_error_policy_bootstraps_registered_canonical_key`,
`test_eager_fallback_does_not_advance_unadmitted_key`,
`test_repeated_fallback_key_is_admitted_only_at_drained_boundary`, and
`test_cross_rank_key_digest_mismatch_fails_before_forward`.

- [ ] **Step 2: Register canonical keys before warmup**

At setup, register bounded key templates from configured physical geometry,
role, scopes, dispatcher/precision, and PP/VPP topology; do not invent
`num_microbatches` before DP sharding and iterator construction. At the first
drained pre-forward boundary, derive the runtime count and storage generations,
collectively match one template, and atomically materialize its canonical
training, policy, or reference key. This first template-matching key is an
automatic canonical admission, not an unseen-key fallback, under both policies.
It progresses through exactly three `warming_eager` calls and then `capture`.
Add `test_first_runtime_microbatch_count_materializes_canonical_key` and prove
different ranks cannot materialize different counts.

A geometry not in the registered template set is truly unseen.
`unseen_key_policy=error` selects `raise_unsupported_key` before forward.
`unseen_key_policy=eager` selects stateless `whole_call_eager`: it does not
advance warmup, allocate buffers, or mutate an LRU. Repeated fallback telemetry
may propose the exact key for admission, but only an explicit collective
`admit_logprob_schedule_key()` call at a drained stage boundary can validate
capability/memory, evict an idle forward-only LRU entry, and start that key's
three-call warmup. Training never admits an unseen key dynamically.

- [ ] **Step 3: Implement per-key successful-call warmup**

Only a finite, completed call advances its own counter. Training advancement
still follows successful optimizer updates. Policy and reference logprob calls
advance only their exact keys and never each other.

- [ ] **Step 4: Implement collective pre-forward selection**

All ranks derive and compare the digest, validate capacity/storage/capability,
then select exactly one of `warming_eager`, `cache_hit`, `capture`, or
`whole_call_eager`; `unseen_key_policy=error` instead selects a collective
`raise_unsupported_key` transition. Uninstall the previous bank without
resetting it before an eager path. The error transition raises the same
serialized key digest and reason on every rank. No rank enters model forward
before agreement.

- [ ] **Step 5: Preserve transactional failure behavior**

Capture failure abandons the candidate on every rank, resets only the failed
candidate, and leaves prior valid banks reusable. An in-flight graph or model
collective forbids eviction and bank switching.

- [ ] **Step 6: Run lifecycle tests and commit**

```bash
uv run pytest -q tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  -k 'lifecycle or schedule_key or fallback or collective or bank'
git diff --check
git add nemo_rl/models/policy/cuda_graph_metrics.py \
  nemo_rl/models/megatron/cuda_graph_lifecycle.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py pyrefly.toml
git commit -s -m "feat: isolate CUDA graph execution lifecycles"
```

### Task 10: Fingerprint Storage and Make BF16 Reference Swaps Address-stable

**Files:**
- Create: `nemo_rl/models/megatron/cuda_graph_storage.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Create: `tests/unit/models/megatron/test_cuda_graph_storage.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`
- Modify: `pyrefly.toml`

**Interfaces:** Add frozen `TensorStorageFingerprint`,
`ModelStorageFingerprint`, and `CudaGraphStorageGeneration`. Include tensor
identity, untyped-storage pointer, data pointer, storage offset, shape, stride,
dtype, layout, device, parameter/buffer role, training gradient buffers,
distributed parameter-buffer generation, and tensor leaves of TE extra state.
The schedule key stores `model_storage_generation` for every path and
`grad_storage_generation` only for training.

- [ ] **Step 1: Add red pointer-stability and invalidation tests**

Test that in-place value copy preserves fingerprints, replacement/device move
changes them, gradient relocation invalidates training only, persistent-buffer
relocation invalidates all owning paths, and nested FP8 extra-state tensor
replacement is detected.

- [ ] **Step 2: Add red BF16 reference transition tests**

Capture training and policy-logprob banks, copy reference values in place,
capture/hit the reference bank, restore policy values in place, and assert the
training bank hits without recapture. Verify identical pointer sets before and
after. Add a deliberate `set_extra_state()` relocation test that fails closed.
Add `test_reference_body_exception_restores_policy_weights_sampling_hooks_and_bank`
and inject the exception after the reference copy but before normal context
exit. Also add
`test_reference_entry_copy_failure_rolls_back_every_mutated_tensor` and inject
failure after the Nth in-place parameter copy, before the context yields.

- [ ] **Step 3: Implement dependency-aware generations**

Validate the relevant fingerprint immediately before every replay. Parameter
or persistent-buffer relocation increments the shared model generation and
invalidates all owning banks. Gradient-buffer relocation increments only the
training generation. Keep combined-mode optimizer offload disabled until a
test proves it cannot relocate captured storage.

Before capture, compute the complete static-buffer byte plan without allocating
the bank owner and run a collective `CudaGraphMemoryPreflight`. Reset peak
stats before an eager warmup and record current allocated/reserved plus peak
allocated/reserved afterward. Estimate incremental capture-pool demand by
dry-run/probe where available, otherwise from peak live allocation above
current live allocation. Compare capture-pool demand plus the not-yet-allocated
static byte plan and a ten-percent total-device safety reserve against
`torch.cuda.mem_get_info()` free bytes on every rank. Only after every rank
passes may it allocate static buffers; verify the observed allocation delta
against the plan. Do not add any static allocation already reflected in free
bytes a second time. Fail before TE capture if any rank is short and persist
current allocated/reserved, planned/observed static bytes, incremental graph
demand, safety reserve, and free bytes in telemetry.

- [ ] **Step 4: Refactor `use_reference_model()`**

For BF16, copy reference and policy values into existing CUDA tensors and
verify fingerprints at each transition. Do not call the current unconditional
bank reset. For FP8/NVFP4 reference graph requests, raise at setup until the
extra-state test proves stable addresses; eager reference remains available.

Before mutating any value, validate the complete policy/reference source and
destination tensor signatures, sampling parameters, hook state, and restore
fingerprint. Enter the reference state only after that two-pass preflight.
Start one outer transaction before the first in-place copy, record each
successfully mutated destination, and keep it active through context entry,
the yielded body, and restoration. If the Nth entry copy fails, restore every
earlier destination from its validated policy source in reverse order before
propagating the error; the context must never yield a partially converted
model. In the outer `finally`, restore all policy tensor values in place,
sampling parameters, forward hooks, model mode, prior bank, and fingerprints
even when entry, logprob, or capture raises. If restoration itself fails,
surface a fatal combined error and invalidate every affected bank rather than
leaving a reference model active.

Refactor `prepare_for_lp_inference()` so combined training+logprob graph mode
keeps model parameters and gradient buffers resident. Reject
`offload_optimizer_for_logprob=true` during setup for the initial combined
mode. Add exact test nodes
`test_prepare_for_lp_inference_preserves_graph_owned_parameters_and_grad_buffers`,
`test_combined_mode_rejects_optimizer_offload_for_logprob`, and
`test_collective_graph_memory_preflight_fails_before_capture`.

- [ ] **Step 5: Run storage/reference tests and commit**

```bash
uv run pytest -q tests/unit/models/megatron/test_cuda_graph_storage.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_megatron_worker.py \
  -k 'storage or reference_policy_functionality or extra_state or relocation or prepare_for_lp_inference or optimizer_offload or memory_preflight'
git diff --check
git add nemo_rl/models/megatron/cuda_graph_storage.py \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_cuda_graph_storage.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_megatron_worker.py pyrefly.toml
git commit -s -m "feat: preserve graph storage across reference swaps"
```

### Task 11: Integrate Policy and Reference Logprob Capture/Replay

**Files:**
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `nemo_rl/models/policy/cuda_graph_metrics.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/workers/base_policy_worker.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/policy/teacher_worker_group.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `nemo_rl/data_plane/worker_mixin.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Modify: `nemo_rl/algorithms/utils.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`
- Modify: `tests/unit/models/policy/test_cuda_graph_policy_packing.py`
- Modify: `tests/unit/algorithms/test_cuda_graph_metrics.py`
- Modify: `tests/unit/algorithms/test_grpo.py`

**Interfaces:** Add an `execution_path: CudaGraphExecutionPath` keyword to the
role-aware internal `_get_logprobs`; the public policy and reference methods
select exact paths. `megatron_forward_backward()` enters the matching MCore
execution context. Keep `LogprobOutputSpec` and `ReferenceLogprobOutputSpec`
tensor-only. Use the exactly one serializable dataclass and protocol created
in `nemo_rl/models/policy/cuda_graph_metrics.py`:

```python
@dataclass(frozen=True)
class CudaGraphPathMetrics:
    execution_path: CudaGraphExecutionPath
    rank: int
    model_parallel_replica: str
    is_unique_work_representative: bool
    numeric: dict[str, int | float]
    active_key_digest: str | None
    fallback_reasons: tuple[str, ...]


class CudaGraphMetricsProvider(Protocol):
    def snapshot_cuda_graph_metrics(
        self, execution_path: CudaGraphExecutionPath, stage_token: str
    ) -> CudaGraphPathMetrics | None: ...

    def ack_cuda_graph_metrics(
        self, execution_path: CudaGraphExecutionPath, stage_token: str
    ) -> None: ...
```

Add separate snapshot and acknowledgment worker RPCs. The algorithm creates
one immutable `stage_token` for each completed policy-training, policy-logprob,
or reference-logprob stage. LM and TQ controllers snapshot every rank and
return one typed aggregate to the algorithm through a policy-level metrics
API; they do not own logging and do not acknowledge. After the algorithm's
logger succeeds, it calls the explicit policy-level
`commit_cuda_graph_metrics(execution_path, stage_token)` API, which fans an
idempotent acknowledgment out to every worker. Repeating a snapshot with the
same uncommitted token returns byte-identical cached payload; a different
token is rejected while the prior token is unacknowledged. A collective ack
advances only that path's accumulator. Each worker retains an acknowledged
token tombstone, so retrying a partially delivered commit is a no-op on
already acknowledged ranks and advances the remaining ranks exactly once.
The algorithm may not start the next token until every rank confirms the
commit; an unrecoverable worker loss fails the run. Telemetry never travels
through `BatchedDataDict` or TQ tensor columns.

- [ ] **Step 1: Replace eager-only AST tests with enabled/disabled behavior tests**

Disabled config must still uninstall graphs and pass the eager geometry.
Enabled policy/reference paths must pass fixed geometry, use
`forward_only=True`, run eval/no-grad, select their own banks, and emit their
own metrics. No test may merely assert that a source string changed.

- [ ] **Step 2: Implement role-aware preflight and static copy**

Materialize/validate the fixed microbatch schedule before model entry, select
the collective action, copy dynamic values into the selected bank's static
buffers, then enter the MCore execution context. Slice dummy outputs and
restore the original right-padded batch layout after replay.

- [ ] **Step 3: Preserve eager dynamic tails and logprob postprocessing**

Partial graph leaves replay inside `megatron_forward_backward`; fused linear
logprob/postprocessing, dynamic MoE dispatch/expert/combine, and output
broadcast remain eager unless independently proven fixed. Router replay stays
off for router/preprocess scopes until Task 12.

- [ ] **Step 4: Preserve metrics across LM and TQ aggregation**

Keep logprob and reference outputs tensor-only. After tensor aggregation,
`worker_mixin` and TQ invoke only the snapshot RPC and return the typed aggregate
through the policy metrics API; the algorithm performs the later commit RPC.
Remove the existing training result's embedded `cuda_graph_metrics` field so
training, policy logprob, and reference logprob have one authoritative
transport. Non-Megatron workers
return `None`/not-applicable through `CudaGraphMetricsProvider`, and teacher
forwarding preserves that protocol. Every participating rank returns its
identity and timing so aggregation can take the rank maximum; replicated
schedule/cache counters must match within a model-parallel replica, and token
or layer-call counts are taken only from
`is_unique_work_representative=True`. Add LM, TQ, teacher, non-Megatron,
disagreement, and representative-selection tests. Prove that snapshot retries
are byte-identical, partial-rank or logger failure leaves the stage
unacknowledged and retryable, a partially delivered acknowledgment converges
under same-token retry, collective acknowledgment advances exactly once, no
next token begins before all ranks confirm, and `BatchedDataDict.from_batches()`
and `.to()` never see nested telemetry.

- [ ] **Step 5: Run integration unit tests and commit**

```bash
uv run pytest -q tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_cuda_graph_policy_packing.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/algorithms/test_cuda_graph_metrics.py \
  tests/unit/algorithms/test_grpo.py
git diff --check
git add nemo_rl/models/megatron/train.py \
  nemo_rl/models/policy/cuda_graph_metrics.py \
  nemo_rl/models/policy/interfaces.py \
  nemo_rl/models/policy/workers/base_policy_worker.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/policy/teacher_worker_group.py \
  nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py \
  nemo_rl/data_plane/worker_mixin.py \
  nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py \
  nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/utils.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/models/policy/test_cuda_graph_policy_packing.py \
  tests/unit/algorithms/test_cuda_graph_metrics.py \
  tests/unit/algorithms/test_grpo.py
git commit -s -m "feat: graph policy and reference logprobs"
```

### Task 12: Make R3 Routes Explicit Graph Inputs

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/router_replay.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/te_cuda_graph_bank.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_router_replay.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_te_cuda_graph_bank.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Modify: `nemo_rl/models/megatron/router_replay.py`
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_router_replay.py`
- Modify: `tests/unit/algorithms/test_grpo_router_replay_async.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`

**Purpose:** Replace MCore's temporary `target_topk_idx` Python-attribute
ownership with bank-owned fixed-address tensors, then copy each NeMo-RL R3
route set into that exact owner before capture or replay. This prevents warmup
routes from being captured and replayed for later policy-logprob batches.

- [ ] **Step 1: Add red MCore route-owner tests**

In the nested MCore worktree, add
`test_replay_forward_routes_are_owned_by_the_active_graph_bank`,
`test_route_copy_preserves_bank_pointer_and_changes_values`,
`test_route_copy_rejects_signature_mismatch_before_partial_copy`,
`test_bank_eviction_releases_route_inputs_once`, and
`test_replay_forward_rejects_temporary_target_topk_idx`, and
`test_training_1f1b_preserves_routes_for_multiple_outstanding_microbatches`,
`test_eager_r3_context_survives_warmup_and_whole_call_fallback`, and
`test_attn_only_graph_keeps_router_replay_eager`.
Across 20 real TE
replays, change route values while preserving shape and pointer; assert exact
top-k IDs and expert counts and nonzero graph calls.

The `router_replay_8` manifest row selects the first five literal nodes plus
the eager-context and attention-only nodes. `router_replay_1f1b_8` selects
`test_training_1f1b_preserves_routes_for_multiple_outstanding_microbatches`
and its exact PP/VPP parameter IDs. Store fully qualified nodes from
`tests/unit_tests/transformer/moe/test_router_replay.py` and
`tests/unit_tests/transformer/test_te_cuda_graph_bank.py`; neither row uses a
broad substring filter.

- [ ] **Step 2: Implement the MCore bank-owned input contract**

Add a frozen `RouterReplayInputSignature` and schedule-indexed
`RouterReplayStaticInputs` beneath each `TECudaGraphBank`. Allocate one fixed
target-route buffer per `(execution_kind, layer_id, VPP chunk, schedule visit)`
before capture, not one mutable tensor for the whole bank. Each slot exposes a
two-phase `validate_and_copy_target_topk_idx(source)` operation. Capture and
`REPLAY_FORWARD` read the matching persistent slot through the active bank.
Training preserves the current FIFO semantics for multiple outstanding 1F1B
microbatches: recompute/backward consumes the exact forward slot and cannot
observe a later microbatch's overwrite. Eval has forward-only slots and no
backward queue. Remove the unsafe lifecycle in which
`set_target_indices()` stores a caller-owned temporary Tensor on a module
attribute. Include slot signatures, schedule shape, and R3 mode in the bank
fingerprint, never route values.

Preserve a separate explicit eager `RouterReplayContext` for warmups,
whole-call fallback, and graph scopes that leave the router eager. It owns the
same schedule-indexed/FIFO route sequence for one call but is not stored on a
module and is never captured. `set_target_indices()` becomes a compatibility
adapter into the active bank slot or eager context and raises if neither owner
exists. Thus an `attn`/`mamba`-only bank can replay those leaves while router
replay consumes the current eager routes.

- [ ] **Step 3: Format, sign, and push the MCore fix**

Run from the nested MCore root:

```bash
uv run isort megatron/core/transformer/moe/router_replay.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/moe/test_router_replay.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git diff --check
git add megatron/core/transformer/moe/router_replay.py \
  megatron/core/transformer/te_cuda_graph_bank.py \
  tests/unit_tests/transformer/moe/test_router_replay.py \
  tests/unit_tests/transformer/test_te_cuda_graph_bank.py \
  tests/unit_tests/transformer/test_cuda_graphs.py
git commit -S -s -m "fix: own router replay inputs in CUDA graph banks"
git push origin sj/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 4: Submit both attested router-replay rows**

From the NeMo-RL root on the selected cluster, resolve the pushed MCore branch
once and submit the fixed eight-rank and outstanding-1F1B rows:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
MCORE_CANDIDATE_SHA=$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-20260731 | awk '{print $1}')
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_CANDIDATE_SHA="$MCORE_CANDIDATE_SHA" \
  MCORE_TEST_ROWS='router_replay_8 router_replay_1f1b_8' \
  "$EXP/submit_mcore_matrix.sh"
```

Require changing-route eager parity, nonzero graph calls on selected router
scopes, the explicit eager-context rows, and exact training FIFO parity before
pinning this commit.

- [ ] **Step 5: Perform the second dependency pin cycle before NeMo edits**

From the Bridge root, pin the pushed MCore commit, run Bridge checks, commit,
and push. Then from the NeMo-RL root, pin the pushed Bridge commit, commit, and
push before changing the NeMo adapter:

```bash
# Bridge root
uv run pre-commit run --all-files
git add 3rdparty/Megatron-LM
git commit -s -m "build: pin bank-owned router replay inputs"
git push origin sna/thd-cg-hybrid-nemotron-20260731

# NeMo-RL root
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "build: pin explicit router replay graph inputs"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

- [ ] **Step 6: Add red NeMo stale-route and address tests**

Across 20 replays, change route values while preserving route-buffer shape and
pointer. Assert exact router top-k/expert-count parity with eager. Deliberately
skip a copy and require stale-route detection before model forward. Reject
route overflow, wrong dtype/device/stride, missing verifier, and reference-R3.
The test must fail if NeMo writes any module-level temporary attribute instead
of the selected bank's `RouterReplayStaticInputs`.

- [ ] **Step 7: Connect NeMo-RL to the MCore owner**

Copy CP-sharded routed expert IDs into the selected MCore bank input after the
collective key decision and before model entry. The NeMo static-buffer owner
retains the source route tensor for its LRU lifetime, but MCore owns the exact
captured destination. Include tensor signature and R3 mode in the graph key,
never route values. Require the copy-completion generation in the collective
pre-forward contract so a skipped or stale copy fails before any pipeline
rank starts forward.

For `warming_eager`, `whole_call_eager`, or a scope without graph-covered
router/preprocess, enter MCore's explicit eager `RouterReplayContext` instead
of requiring a bank slot. Validate/copy the same CP-sharded route schedule and
collectively close the context after the call.

- [ ] **Step 8: Preserve fail-closed scope rules**

R3-on `attn`/`mamba` may keep routing eager. Enable R3 plus
`moe_router`/`moe_preprocess` only after exact parity passes. Reference logprob
always performs its own routing.

- [ ] **Step 9: Update launcher capability only after tests pass**

Change model selector readiness for R3 router scopes from rejected to
supported only for the proven dispatcher/topology rows. Unsupported DeepEP,
NCCL-EP, or variable-capacity combinations remain rejected before SBATCH.

- [ ] **Step 10: Run NeMo R3 tests and commit**

```bash
uv run pytest -q tests/unit/models/megatron/test_router_replay.py \
  tests/unit/algorithms/test_grpo_router_replay_async.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  -k 'router_replay or stale_route or graph'
git diff --check
git add nemo_rl/models/megatron/router_replay.py nemo_rl/models/megatron/data.py \
  nemo_rl/models/megatron/train.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/megatron/test_router_replay.py \
  tests/unit/algorithms/test_grpo_router_replay_async.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py
git commit -s -m "feat: copy router replay inputs into graph banks"
```

### Task 13: Separate Stage Timers, Throughput, Coverage, and Correctness Metrics

**Files:**
- Modify: `nemo_rl/models/megatron/cuda_graph_lifecycle.py`
- Modify: `nemo_rl/models/policy/cuda_graph_metrics.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/tq_policy.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/algorithms/single_controller.py`
- Modify: `nemo_rl/algorithms/utils.py`
- Modify: `tests/unit/algorithms/test_cuda_graph_metrics.py`
- Modify: `tests/unit/algorithms/test_grpo.py`
- Modify: `tests/unit/models/policy/test_megatron_cuda_graph_worker.py`

**Metric schema:** Use exact namespaces
`cuda_graph/training/*`, `cuda_graph/policy_logprob/*`, and
`cuda_graph/reference_logprob/*`. Disabled paths are `not_applicable`, never
fabricated zeros.

Each enabled namespace has these per-optimizer-step numeric leaves:

| Leaf | Type and unit |
|---|---|
| `eligible_calls` | non-negative integer layer-call delta |
| `graph_calls` | non-negative integer layer-call delta |
| `eager_warmup_calls` | non-negative integer full-call delta |
| `capture_count` | non-negative integer key-capture delta |
| `capture_failure_count` | non-negative integer delta |
| `replay_count` | non-negative integer full-call delta |
| `cache_hit_count` | non-negative integer full-call delta |
| `cache_miss_count` | non-negative integer full-call delta |
| `eviction_count` | non-negative integer bank delta |
| `fallback_count` | non-negative integer full-call delta |
| `logical_tokens` | non-negative integer valid-token count |
| `padded_tokens` | non-negative integer tokens after ordinary packing/padding but before graph-capacity fill |
| `capacity_tokens` | non-negative integer allocated-token count |
| `real_packed_sequences` | non-negative integer real sequence count |
| `capacity_packed_sequences` | non-negative integer configured capacity |
| `graph_coverage` | finite float `graph_calls / eligible_calls` in `[0, 1]` |
| `capacity_utilization` | finite float `logical_tokens / capacity_tokens` in `[0, 1]` |
| `padding_utilization` | finite float `logical_tokens / padded_tokens` in `[0, 1]`, preserving the existing metric meaning |
| `physical_occupancy` | finite float `padded_tokens / capacity_tokens` in `[0, 1]` |
| `static_copy_time_seconds` | non-negative critical-path seconds |
| `capture_time_seconds` | non-negative critical-path seconds |
| `replay_time_seconds` | non-negative critical-path seconds |
| `storage_invalidation_count` | non-negative integer delta |

`active_key_digest`, per-key capture/hit counts, and a sorted unique
`fallback_reasons` enum list live in authoritative JSON metadata rather than
numeric median aggregation. Replicated schedule/cache counters must agree
across ranks and publish once; token and layer-call counts sum over unique
work, and timings use the participating-rank maximum.

- [ ] **Step 1: Add red namespace and aggregation tests**

Test separate path payloads, collision rejection, missing/disabled paths,
cache/call invariants, non-finite values, fallback policy, and aggregation
across workers. Training fallback remains invalid; measured logprob fallback
is valid telemetry but fails performance eligibility. A new schedule key may
capture before the measurement window; the gate rejects only a second capture
of an already attested key during its measured steps.

- [ ] **Step 2: Split algorithm timers**

Emit policy-only, reference-only, and derived combined logprob time and
tokens/sec/GPU while retaining `policy_and_reference_logprobs` for backward
compatibility. Keep E2E, Generation, and PolicyTraining metrics unchanged.

- [ ] **Step 3: Emit complete per-path graph telemetry**

Include eligible/warmup/capture/replay calls, cache hits/misses/evictions,
fallback count/reason, active digest, logical/padded/capacity tokens and
sequence counts, graph coverage, padding utilization, static-copy/capture/
replay time, physical occupancy, and storage invalidations.

- [ ] **Step 4: Preserve correctness evidence**

Carry policy/reference logprob max/mean absolute and relative error, ULP
envelope under these ten literal names:

- `correctness/policy_logprob_graph_vs_eager_max_abs`
- `correctness/policy_logprob_graph_vs_eager_mean_abs`
- `correctness/policy_logprob_graph_vs_eager_max_rel`
- `correctness/policy_logprob_graph_vs_eager_mean_rel`
- `correctness/policy_logprob_graph_vs_eager_max_ulp`
- `correctness/reference_logprob_graph_vs_eager_max_abs`
- `correctness/reference_logprob_graph_vs_eager_mean_abs`
- `correctness/reference_logprob_graph_vs_eager_max_rel`
- `correctness/reference_logprob_graph_vs_eager_mean_rel`
- `correctness/reference_logprob_graph_vs_eager_max_ulp`

Keep these distinct from rollout-versus-policy `token_mult_prob_error`,
sequence multiplicative error,
`num_masked_seqs_by_logprob_error`, `gen_kl_error`, policy/reference KL,
reward, loss, grad norm, parameter delta, route parity, and NaN/Inf status.

- [ ] **Step 5: Add a distributed same-state parity RPC**

Define serializable `CudaGraphParityRequest` and `CudaGraphParityShardResult`
beside the metrics protocol and add
`run_cuda_graph_fixed_batch_parity(request)` to the Megatron worker. LM/TQ
controllers invoke it on every rank in one quiesced distributed job. The
controller freezes and content-hashes one deterministic suite of 20 logical
packed microbatches sharing one physical signature, then each rank uses the
normal CP/SP/PP/VPP data path to derive its local input.

Preserve controller-specific data transport. LM sends the suite through the
existing `run_all_workers_sharded_data` contract: distinct DP shards and exact
replication within each TP/PP/CP model-parallel replica. TQ sends only its
normal `KVBatchMeta`; each worker obtains the data through `_fetch()` before
the same Megatron packing path. Never broadcast one full `BatchedDataDict` to
every rank. Add LM/TQ tests that compare shard IDs and content digests across
DP and model-parallel groups.

Before freezing comparison state, bootstrap the exact canonical keys through
the production lifecycle: run exactly three ordinary successful optimizer
updates for the training warmup, then the normal capture/update call; run three
successful policy and reference eager warmups and their normal capture calls.
Require installed cache-hit banks, take a prewarm metrics snapshot, and
collectively acknowledge that snapshot before freezing model/optimizer/RNG/
storage identity. The eager comparison temporarily
uninstalls without resetting those banks, and the graph comparison reactivates
them. The parity body itself performs no optimizer step, so it does not try to
advance training warmup without a successful update.

On each rank: fingerprint model, optimizer, and every optimizer-owned gradient
buffer, require a quiesced boundary with existing zero-valued gradient storage,
save CPU/CUDA/TE/tensor-parallel RNG state, run eager policy/reference forwards
and one training forward/backward without an optimizer step, and stream the
sharded eager gradients to bounded ephemeral host memory. Zero those existing
gradient buffers in place through one validated helper; never assign
`parameter.grad = None`, replace `main_grad`, or call a zeroing path that may
reallocate storage. Assert the gradient fingerprint and generation are
unchanged, restore all RNG state, and run the requested graph paths on the
identical local inputs. Compare outputs, losses, masks, routes, and every local
gradient shard before deleting the host copy. In `finally`, zero the same
buffers in place and reassert their original fingerprints, generations, and
zero values. Run a collective host-memory preflight and fail before eager
execution if the snapshot does not fit. Never write a checkpoint and never
clone the 235B model or optimizer state.

Do not execute two optimizer updates. Instead record parameter-delta
equivalence as derived only when every gradient shard matches, the optimizer
state/config fingerprint is unchanged, and no optimizer step occurred between
the two passes. The ordinary 20/100-step runs remain the actual parameter-
delta evidence. Restore mode, bank, gradients, RNG, hooks, and reference/policy
values in `finally`. Aggregate all rank results with exact rank/topology and
state-identity digests; a different distributed job/state is not fixed-input
parity.

Add exact tests
`test_fixed_batch_parity_restores_rank_state_on_failure`,
`test_fixed_batch_parity_never_steps_optimizer`,
`test_fixed_batch_parity_compares_every_gradient_shard`,
`test_fixed_batch_parity_rejects_cross_rank_state_mismatch`, and
`test_fixed_batch_parity_cleans_ephemeral_host_gradients`. Also add
`test_fixed_batch_parity_zeros_gradients_without_changing_storage` and
`test_fixed_batch_parity_restores_grad_fingerprint_after_failure`.

- [ ] **Step 6: Run algorithm and distributed-parity tests and commit**

```bash
uv run pytest -q tests/unit/algorithms/test_cuda_graph_metrics.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  -k 'cuda_graph or logprob or throughput'
git diff --check
git add nemo_rl/models/megatron/cuda_graph_lifecycle.py \
  nemo_rl/models/policy/cuda_graph_metrics.py \
  nemo_rl/models/policy/interfaces.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/tq_policy.py \
  nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py \
  nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/utils.py \
  tests/unit/algorithms/test_cuda_graph_metrics.py tests/unit/algorithms/test_grpo.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py
git commit -s -m "feat: report CUDA graph metrics by execution path"
```

### Task 14: Extend Persistent Scripts, Promotion Gates, and HTML Reporting

**Files:**
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/00_none.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/01_training.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/02_policy_logprob.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/03_reference_logprob.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/04_training_policy_logprob.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/05_training_reference_logprob.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/06_logprobs.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/07_all.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/compose_execution_path.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_fixed_batch_parity.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_fixed_batch_parity.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_parity.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_recipes/qwen3_30ba3b.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_recipes/qwen3_235b.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/mcore_test_matrix.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/bridge_test_matrix.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_fixed_batch_parity.sh`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/finalize_run_artifacts.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/build_completed_run_manifest.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_completed_runs.py`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/publish_campaign_gate.py`
- Modify: all five `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/*.env` selectors
- Modify: all three `experiments/cuda_graph/nemotron_thd_te_graph_20260731/profiles/*.env.example` templates
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scope_matrix.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_bridge_scope.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_scope.sub`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_smoke_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_performance_matrix.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_accuracy_soak.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/validate_campaign_gate.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_tensorboard.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/export_wandb.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/collect_results.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/render_report.py`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`
- Create: `examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-4n4g-logprob-cg.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-8n4g-logprob-cg.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-logprob-cg.yaml`
- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-logprob-cg.yaml`
- Create: `tests/unit/experiments/test_fixed_batch_parity.py`
- Create: `tests/unit/experiments/test_campaign_promotion.py`
- Create: `tests/unit/experiments/test_completed_run_export.py`
- Modify: `tests/unit/experiments/test_mcore_standalone_driver.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py`
- Modify: `tests/unit/experiments/test_matrix_submitters.py`
- Modify: `tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py`
- Modify: `tests/unit/experiments/test_export_tensorboard.py`
- Modify: `tests/unit/experiments/test_export_wandb.py`
- Modify: `tests/unit/experiments/test_analyze_cuda_graph_calls.py`

- [ ] **Step 1: Add red two-leaf composition and selector tests**

Require every run to select exactly one committed execution-path leaf plus one
committed scope, variant, or condition leaf. `compose_execution_path.sh`
accepts relative `EXECUTION_PATH_SCRIPT` and `WORKLOAD_LEAF` values, resolves
both with `realpath`, rejects symlinks, raw override strings, missing files,
duplicate leaves, and path escape, sources the execution leaf first, and then
invokes the workload leaf. Execution leaves export the sole canonical
`CUDA_GRAPH_EXECUTION_PATHS` value.

`00_none.sh` disables training, policy-logprob, and reference-logprob graphs;
it is the only valid partner for `scopes/00_baseline_no_cg.sh`. Add tests that
all other baseline combinations fail before SBATCH and that every candidate
has at least one enabled path. Add `SUPPORTED_GRAPH_PATHS=training`,
`CANDIDATE_GRAPH_PATHS`, and per-path capability reasons to all five model
selectors. Policy/reference paths remain candidate-only until a content-bound
smoke gate promotes the exact selector row. Ultra remains
`blocked_external`, not supported or failed.

For Qwen rendering, assert the exact cross-product: A/C emit `00_none` only,
B/E emit candidate paths only, A/C with `07_all` are never submitted, and B/E
with `00_none` are never submitted. The offline command may pass a superset of
leaves, but the submitter must deterministically filter it to these rows and
print every exclusion reason; silently accepting an invalid pair is a test
failure.

- [ ] **Step 2: Render independent path overrides and exact recipes**

Render training and logprob Hydra overrides independently and include the
normalized path tuple in the run name and metadata. Derive recipes without
modifying bases. The initial physical logprob geometry is exact:

| Model recipe | Nodes x GPUs | `mb_tokens` | `max_packed_sequences` |
|---|---:|---:|---:|
| `grpo-nanov3-30ba3b-4n4g-logprob-cg.yaml` | 4 x 4 | 8192 | 16 |
| `grpo-nemotron3-super-120BA12B-8n4g-logprob-cg.yaml` | 8 x 4 | 8192 | 16 |
| `grpo-qwen3-30ba3b-4n4g-logprob-cg.yaml` | 4 x 4 | 8192 | 16 |
| `grpo-qwen3-235b-16n4g-logprob-cg.yaml` | 16 x 4 | 8192 | 16 |

All use BF16 logprob, sequence packing, dynamic batching off, checkpointing
off, exactly three warmups, and W&B project `sna-cg-study`. Do not add an
Ultra derived recipe until its model/data/judge/profile inputs pass the
existing external gate.

Replace both Qwen selectors' `__REQUIRED_*_MCORE_RECIPE__` placeholders with
the two committed experiment adapters. Each adapter imports the reviewed
Bridge Qwen3-MoE GB200 provider, sets the exact 30B-A3B or 235B-A22B model
dimensions/checkpoint identity, mirrors the NeMo-RL 4x4 or 16x4 TP/PP/CP/EP
topology, uses the same packed token capacity, disables checkpoint saves, and
accepts only the typed overrides emitted by `run_mcore_training.py`. Unit
tests compare the standalone and NeMo-RL model/topology fingerprints before
allowing policy-training performance comparison.

- [ ] **Step 3: Build the frozen-batch correctness producer**

`run_fixed_batch_parity.py` orchestrates the production distributed parity RPC
from Task 13. It freezes one content-addressed 20-microbatch suite, policy and
reference storage generations, R3 routes, per-rank RNG states, packed masks,
and logical-output indices. The same live Ray/Megatron worker set runs eager,
restores RNG and gradients without an optimizer step, then runs the requested
graph paths. Compare valid-token policy/reference logprobs, layer outputs,
loss, every sharded gradient, derived parameter-delta equivalence, routes,
reward/KL/error metrics, masks, and NaN/Inf. Require exactly three warmups plus
20 changed-value replays of the same physical signature and nonzero
requested-path graph calls on the expected ranks.

Write an atomic, content-bound parity JSON containing input/state/RNG/route
digests, exact source/runtime/model/topology identity, literal correctness
metric names from Task 13, graph telemetry, job ID, and raw-output digests.
`validate_fixed_batch_parity.py` rejects stochastic independent runs,
different job/state identities, missing ranks or graph calls, an optimizer
mutation, or a metric outside the BF16 tolerance contract.
`scripts/run_nemorl_parity.sub` is the persistent SLURM
entrypoint. This artifact is mandatory for 5-to-20 promotion.

Extend `scripts/run_mcore_scope.sub` as the persistent standalone-MCore test
entrypoint. Add `scripts/run_mcore_training.py` as the only
`ALLOWED_MCORE_DRIVERS` entry in `scope_matrix.py`; it accepts the existing
keyword-only recipe/scope/iteration/checkpoint flags and an allowlisted
`mcore_test_matrix.json` row ID, never a raw Python or pytest command. The
manifest contains the exact capability, packed-THD, TP2/CP2/PP2/VPP2, EP16,
and EP32 pytest node selections and required world sizes.
`submit_mcore_matrix.sh` validates model/row/topology/profile, supports
`TEST_ONLY` and `SBATCH_TEST_ONLY`, and is the sole login-side submitter for
these rows. It accepts exactly one of `MCORE_TEST_ROWS` for the distributed
pytest matrix or `MCORE_MODELS` for gated 5/20-step standalone policy-training
baseline/candidate pairs using each selector's committed recipe. Standalone
results share the NeMo-RL provenance/geometry schema but report only the
PolicyTraining stage; they are never mislabeled as E2E GRPO.

The batch wrapper launches one `torch.distributed.run` agent per node through
`srun`, with `--nnodes=$SLURM_NNODES`,
`--nproc-per-node=$GPUS_PER_NODE`, the node rank from `SLURM_PROCID`, and one
shared rendezvous endpoint. It records the exact pytest selection, topology,
container/runtime/source digests, exit status, and result artifact. The
minimum 8-rank suite runs as 2x4 on GB200; TP2/CP2/PP2 and required EP rows set
their actual 8/16/32-rank world size explicitly. Tests reject a requested EP
degree larger than world size and reject an allocation with idle unlaunched
GPUs.

`submit_fixed_batch_parity.sh` is the corresponding login-side parity
submitter. It composes the exact model, execution-path, workload leaf,
profile, source/runtime attestations, account, partition, nodes,
`--gpus-per-node`, segment, and output identity; supports `TEST_ONLY` and
`SBATCH_TEST_ONLY`; and invokes only `scripts/run_nemorl_parity.sub`. No parity
job is assembled from an ad hoc `COMMAND` environment variable. Exact-row
mode requires both `EXECUTION_PATH_SCRIPT` and `WORKLOAD_LEAF`. Matrix mode
requires `PARITY_MATRIX=smoke` plus the same candidate path input used by the
smoke submitter and reuses its pure typed row enumerator. It emits every
capability-valid candidate `model/scope/graph_paths/R3/dispatcher/topology`
row, excludes the eager baseline, and accepts `QWEN_ARMS='B E'` only for Qwen
candidate arms. Add a set-equality test proving the parity matrix is exactly
the smoke candidate matrix after removal of baseline rows; a missing, extra,
or differently normalized row fails before SBATCH.

- [ ] **Step 4: Finalize and export only completed runs**

Invoke `finalize_run_artifacts.py` from an exit trap in
`scripts/run_nemorl_scope.sub`. It atomically writes `run-artifacts.json` with
exit status, SLURM job ID, exact metadata/runtime-attestation digests,
TensorBoard event directories, and the exact W&B run ID discovered from the
run-local W&B directory. A failed run is recorded but cannot be promoted.

Remove the current post-`sbatch` metadata race. Before calling `sbatch`,
`run_scope.sh` atomically writes a non-symlink `submission-intent.json` with
every known run field and passes its path and SHA256 to the job. The batch job
validates that intent before starting any worker and combines it with its own
`SLURM_JOB_ID`; the login process may append an atomic submission receipt but
is not the authority. The finalizer fails closed if the intent is absent or
changed, even when a job exits before the login-side `sbatch --parsable`
returns.

`build_completed_run_manifest.py` scans committed experiment log roots and
accepts only successful finalization records whose metadata and attestation
digests match. `export_completed_runs.py` deterministically selects local
TensorBoard first or the exact recorded W&B run as a documented fallback,
calls the existing exporters, and writes atomic `results/raw/*.jsonl` files.
It also copies the small normalized parity, submission-intent, run-artifacts,
and runtime-attestation JSON into content-addressed `results/evidence/`
objects. These normalized files are durable report inputs, not raw SLURM logs,
and are committed with the report.
Each committed profile example already contains the Task 2 `RUN_LOG_ROOT` and
must retain it; the manifest builder requires `--profile-file`, resolves that
absolute root, and never guesses from the checkout. Run exporters with the repository's pinned reporting
dependencies through `uv run --locked`; reserve `uv run --no-project` for the
stdlib-only collector/renderer. This creates a local reporting environment but
does not mutate any cluster training environment or lockfile.

- [ ] **Step 5: Publish content-bound 5-to-20 and 20-to-100 gates**

Five-step promotion requires finite correctness, requested-path nonzero
eligible/graph calls after capture, expected coverage, exact provenance, and no
forbidden fallback plus a passing fixed-batch parity SHA. Twenty-step
promotion additionally requires a matched baseline/candidate pair,
steady-state cache hits, zero post-warmup fallback, and zero recapture of an
already attested key during the measurement window. Disabled paths are
explicit N/A.

`publish_campaign_gate.py` is the only promotion producer. It validates raw
records, parity/runtime/profile/source digests, job IDs, and the exact selected
path/scope row, then atomically writes a canonical JSON gate and its SHA256.
Every gate contains relative evidence paths plus SHA256s for its immutable
phase manifest, normalized JSONL rows, parity manifest/JSON, submission
intents, run-artifacts, and runtime attestations. Preserve separate
`completed-runs-smoke.json`, `completed-runs-performance.json`,
`completed-runs-accuracy.json`, `completed-runs-all.json`, and
`fixed-batch-parity-manifest.json`; no later phase overwrites bytes bound by an
earlier gate.

`validate_campaign_gate.py` does not trust the producer flag or gate schema.
It reopens every content-addressed evidence file, recomputes its digest and
semantic eligibility, rebuilds canonical gate bytes with the same pure
function used by the producer, and byte-compares them with the supplied gate.
A hand-written gate without the complete reproducible evidence closure is
rejected; byte-identical regenerated content backed by identical evidence is
equivalent by definition. Extend `submit_qwen_router_validation.sh`
with `PHASE=accuracy`, exactly 100 steps, and mandatory
`PERFORMANCE_PROMOTION_FILE`/`PERFORMANCE_PROMOTION_SHA256`; generic
performance/accuracy submitters enforce the same gates. With no positional
scope arguments, performance and accuracy submitters enumerate only rows
listed in the verified promotion gate; they never fall back to their historical
default scope arrays.

For Qwen, submit A/C eager baselines once per repeat for R3 off/on,
respectively. B is the R3-off router candidate and E is the R3-on attention
candidate, each crossed only with selected graph-path leaves. Never submit
router plus R3. Gate identity includes A/B/C/E arm and graph path. In
performance and accuracy phases, omitting positional arms makes the submitter
read the exact promoted arms from the verified input gate; it must not fall
back to a hard-coded A/B default.

- [ ] **Step 6: Define complete result identity and paired comparison keys**

Add policy/reference timers and throughput aliases plus all three path metric
families. Every raw record includes `num_nodes`, `gpus_per_node`, `tp`, `pp`,
`cp`, `ep`, `topology_digest`, `sequence_packing`, `train_mb_tokens`,
`logprob_mb_tokens`, `thd_max_packed_sequences`, `warmup_steps`,
`measurement_first_step`, `measurement_last_step`, `profile_sha256`,
`runtime_attestation_sha256`, `runtime_preflight_job_id`, model/checkpoint
identity, three repository SHAs, container SHA256, graph paths, scope,
Transformer Engine commit and version from runtime attestation, dispatcher,
R3, repeat, job ID, and raw artifact SHA256. Include TE commit/version in the
parity artifact, gate, `baseline_cell_id`, `comparison_cell_id`, and HTML; a
container digest alone is insufficient. Enabled paths require
every metric; disabled paths use a typed `not_applicable` status and never
zero substitution.

Create `baseline_cell_id` from immutable model/checkpoint, R3, topology,
packing/geometry, provenance, measurement window, and repeat fields. Create
`comparison_cell_id` by adding candidate scope and graph paths. Require
exactly one `00_none + 00_baseline_no_cg` record per `baseline_cell_id`; every
candidate points to it. Reject a missing or duplicate baseline before speedup
calculation.

- [ ] **Step 7: Extend matched HTML reporting and gate state**

Add `smoke_gate_status`, `smoke_gate_sha256`, `performance_gate_status`,
`performance_gate_sha256`, promoted job IDs, and a stable failure reason.
`collect_results.py` deterministically writes `results.json`, `results.csv`,
and the new-schema `paired_20step_summary.json` from the all-phase manifest;
the renderer consumes those products and never leaves the historical Nano-only
summary in place.
Render separate tables for PolicyTraining, policy Logprob, reference Logprob,
combined Logprob, Generation, E2E, per-path coverage/cache/packing/storage,
fixed-batch correctness, soak accuracy, and promotion state. Surface failures
and `blocked_external`; never let them enter a speedup average.

- [ ] **Step 8: Run launcher, parity, promotion, and report tests**

```bash
uv run pytest -q tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py \
  tests/unit/experiments/test_fixed_batch_parity.py \
  tests/unit/experiments/test_campaign_promotion.py \
  tests/unit/experiments/test_completed_run_export.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_export_wandb.py \
  tests/unit/experiments/test_analyze_cuda_graph_calls.py
bash -n experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/compose_execution_path.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_smoke_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_performance_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_accuracy_soak.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_qwen_router_validation.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_mcore_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_bridge_matrix.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_fixed_batch_parity.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_bridge_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_scope.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_parity.sub \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/*.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/variants/*.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/execution_paths/*.sh \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/*.sh
uv run python -m py_compile \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_mcore_training.py
```

- [ ] **Step 9: Run exact offline matrix rendering and commit**

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
CLUSTER=oci-hsg PROFILE_FILE="$EXP/profiles/oci-hsg.env.example" \
  TEST_ONLY=1 MODEL=nano \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/07_all.sh' \
  "$EXP/submit_smoke_matrix.sh"
CLUSTER=oci-hsg PROFILE_FILE="$EXP/profiles/oci-hsg.env.example" \
  TEST_ONLY=1 MODEL=qwen3_30ba3b PHASE=smoke \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/07_all.sh' \
  "$EXP/submit_qwen_router_validation.sh" A B C E
CLUSTER=oci-hsg PROFILE_FILE="$EXP/profiles/oci-hsg.env.example" \
  TEST_ONLY=1 MODEL=qwen3_235b PHASE=smoke \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/07_all.sh' \
  "$EXP/submit_qwen_router_validation.sh" A B
git diff --check
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731 \
  examples/configs/recipes/llm/performance \
  tests/unit/experiments
git commit -s -m "feat: benchmark CUDA graphs by execution path"
```

### Task 15: Pin Dependencies, Verify Locally, Submit Gates, and Publish Results

**Files:**
- Inspect: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` gitlink
- Inspect: `uv.lock`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-smoke.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-performance.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-accuracy.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-all.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/fixed-batch-parity-manifest.json`
- Create: normalized `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/raw/*.jsonl`
- Create: content-addressed `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/evidence/*.json`
- Create: content-addressed `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/parity/*.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/gates/smoke-promotion.json`
- Create: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/gates/performance-promotion.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.csv`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/paired_20step_summary.json`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html`
- Modify: `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md`

- [ ] **Step 1: Run the complete focused NeMo-RL suite**

```bash
uv run pytest -q tests/unit/models/policy/test_logprob_cuda_graph_config.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_cuda_graph_lifecycle.py \
  tests/unit/models/megatron/test_cuda_graph_storage.py \
  tests/unit/models/megatron/test_router_replay.py \
  tests/unit/models/policy/test_cuda_graph_policy_packing.py \
  tests/unit/models/policy/test_megatron_cuda_graph_worker.py \
  tests/unit/algorithms/test_cuda_graph_metrics.py \
  tests/unit/algorithms/test_grpo_router_replay_async.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_reporting.py \
  tests/unit/experiments/test_fixed_batch_parity.py \
  tests/unit/experiments/test_campaign_promotion.py \
  tests/unit/experiments/test_completed_run_export.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_export_tensorboard.py \
  tests/unit/experiments/test_export_wandb.py \
  tests/unit/experiments/test_analyze_cuda_graph_calls.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/algorithms/test_grpo.py
uv run pre-commit run --all-files
git diff --check
```

- [ ] **Step 2: Verify recursive pushed pins and the committed lockfile**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge status --short
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM status --short
git submodule status --recursive
git ls-remote seonjinn experiment/thd-cg-hybrid-nemotron-20260731
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  ls-remote origin sna/thd-cg-hybrid-nemotron-20260731
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  ls-remote origin sj/thd-cg-hybrid-nemotron-20260731
uv lock --check
git diff --exit-code -- uv.lock \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

Expected: the gitlink points to the pushed Bridge commit, which points to the
pushed MCore commit. Task 12 already performed the final dependency pin, so a
diff here is an error, not a reason to create an empty or late pin commit. If a
dependency declaration changed in an earlier task, that task must have run
`uv lock` and committed `uv.lock` with the change. No job is submitted from an
unpushed source state.

- [ ] **Step 3: Pull the pushed source and run scheduler test-only**

Use `/fairshare` to select OCI-HSG, Ptyche, or Lyris, then `git pull` the pushed
branch in the clean cluster snapshot. Populate that cluster's untracked
`profiles/<cluster>.env`, stage/attest the immutable nightly image, and run the
committed submitters with `SBATCH_TEST_ONLY=1`. Confirm account, `batch`
partition, nodes, `--gpus-per-node`, segment, source snapshot, container
SHA256, runtime-attestation SHA256, and log path. Never use backfill.

Run these exact scheduler checks after exporting the selected `CLUSTER` and
absolute `PROFILE_FILE`:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
PATHS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh'
for MODEL in nano super; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    EXECUTION_PATH_SCRIPTS="$PATHS" SBATCH_TEST_ONLY=1 \
    "$EXP/submit_smoke_matrix.sh"
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PARITY_MATRIX=smoke EXECUTION_PATH_SCRIPTS="$PATHS" \
    SBATCH_TEST_ONLY=1 "$EXP/submit_fixed_batch_parity.sh"
done
for MODEL in qwen3_30ba3b qwen3_235b; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PHASE=smoke EXECUTION_PATH_SCRIPTS="$PATHS" SBATCH_TEST_ONLY=1 \
    "$EXP/submit_qwen_router_validation.sh" A B C E
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PARITY_MATRIX=smoke QWEN_ARMS='B E' EXECUTION_PATH_SCRIPTS="$PATHS" \
    SBATCH_TEST_ONLY=1 "$EXP/submit_fixed_batch_parity.sh"
done
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_TEST_ROWS='te_eval_capability_8 packed_tp2_cp2_pp2_8 hybrid_ep16 hybrid_ep32' \
  SBATCH_TEST_ONLY=1 "$EXP/submit_mcore_matrix.sh"
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_MODELS='nano super qwen3_30ba3b qwen3_235b' STEPS=5 \
  SBATCH_TEST_ONLY=1 "$EXP/submit_mcore_matrix.sh"
```

The Qwen3-235B selector prints capability exclusions for C/E if its R3
preflight is absent; it must not submit them accidentally.

- [ ] **Step 4: Submit all eligible five-step smokes and parity jobs in parallel**

Submit baseline plus supported module scopes for Nano, Super, Qwen3-30B-A3B,
and Qwen3-235B across selected candidate path leaves. Submit exactly one eager
baseline per model/R3/topology/repeat, not one baseline per candidate path.
Submit the corresponding fixed-batch parity jobs at the same time. Submit
the standalone MCore 8-rank capability suite plus TP2/CP2/PP2 and each
model-required EP parity row through `scripts/run_mcore_scope.sub` at the same
time; these produce separate attested result records and do not serialize the
NeMo-RL jobs. Submit
Ultra only after its
existing launcher-validation and external-artifact gates pass; until then the
report records `blocked_external` rather than a fabricated zero or failed CG
row. The selector must exclude unsupported model/scope/dispatcher/R3 rows
before SBATCH. Monitor every real job for five minutes and cancel any job that
allocates idle GPUs because workers failed to join.

For Qwen, `submit_qwen_router_validation.sh PHASE=smoke` accepts A/B/C/E for
30B-A3B and capability-valid arms for 235B. A/C use only `00_none`; B/E cross
candidate paths. Jobs are submitted without dependencies so independent rows
can schedule concurrently.

With `CLUSTER`, `PROFILE_FILE`, and the attestation fields exported by Step 3,
the launch fan-out is persistent and reproducible:

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
for MODEL in nano super; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
    "$EXP/submit_smoke_matrix.sh"
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PARITY_MATRIX=smoke \
    EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
    "$EXP/submit_fixed_batch_parity.sh"
done
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL=qwen3_30ba3b \
  PHASE=smoke \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
  "$EXP/submit_qwen_router_validation.sh" A B C E
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL=qwen3_235b \
  PHASE=smoke \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
  "$EXP/submit_qwen_router_validation.sh" A B
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL=qwen3_30ba3b \
  PARITY_MATRIX=smoke QWEN_ARMS='B E' \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
  "$EXP/submit_fixed_batch_parity.sh"
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL=qwen3_235b \
  PARITY_MATRIX=smoke QWEN_ARMS='B' \
  EXECUTION_PATH_SCRIPTS='execution_paths/00_none.sh execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
  "$EXP/submit_fixed_batch_parity.sh"
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_TEST_ROWS='te_eval_capability_8 packed_tp2_cp2_pp2_8 hybrid_ep16 hybrid_ep32' \
  "$EXP/submit_mcore_matrix.sh"
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_MODELS='nano super qwen3_30ba3b qwen3_235b' STEPS=5 \
  "$EXP/submit_mcore_matrix.sh"
```

- [ ] **Step 5: Export smoke artifacts and publish the 5-to-20 gate**

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
uv run --locked python "$EXP/build_completed_run_manifest.py" \
  --phase smoke --profile-file "$PROFILE_FILE" \
  --output "$EXP/results/manifests/completed-runs-smoke.json"
uv run --locked python "$EXP/validate_fixed_batch_parity.py" \
  --profile-file "$PROFILE_FILE" --output-dir "$EXP/results/parity" \
  --manifest "$EXP/results/manifests/fixed-batch-parity-manifest.json"
uv run --locked python "$EXP/export_completed_runs.py" \
  --manifest "$EXP/results/manifests/completed-runs-smoke.json" \
  --output-dir "$EXP/results/raw" --evidence-dir "$EXP/results/evidence"
uv run --no-project python "$EXP/publish_campaign_gate.py" smoke \
  --manifest "$EXP/results/manifests/completed-runs-smoke.json" \
  --raw-dir "$EXP/results/raw" \
  --parity-manifest "$EXP/results/manifests/fixed-batch-parity-manifest.json" \
  --output "$EXP/results/gates/smoke-promotion.json"
```

The producer prints the gate SHA256. A failed/missing parity artifact or a run
without `run-artifacts.json` cannot appear in the gate. The parity validator
copies accepted small parity JSON into the content-addressed results directory
before writing its immutable manifest.

- [ ] **Step 6: Promote passing rows to paired 20-step performance**

For each accepted row, run matched eager and graph repeats. Require exact
source/container/model/topology/geometry identity, nonzero requested-path graph
coverage, cache hits, zero post-warmup fallback, zero recapture in the measured
window, and all correctness gates before computing a speedup. Pass the exact
`SMOKE_PROMOTION_FILE` and `SMOKE_PROMOTION_SHA256` to every generic and Qwen
performance submitter. Submit eligible rows together, then monitor all for at
least five minutes.

The verified smoke gate supplies candidate rows and their `baseline_cell_id`.
`submit_performance_matrix.sh` must automatically render exactly one 20-step
`00_none + scopes/00_baseline_no_cg` job for every unique required baseline
cell, regardless of the candidate-only `EXECUTION_PATH_SCRIPTS` filter below.
For Qwen, B requires its R3-off A baseline and E requires its R3-on C baseline.
The Qwen submitter applies the same one-baseline-per-cell rule. Tests compare
rendered and submitted row sets, reject missing/duplicate baselines, and prove
that a candidate filter cannot suppress its matched eager baseline.

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
SMOKE_PROMOTION_FILE=$(realpath "$EXP/results/gates/smoke-promotion.json")
SMOKE_PROMOTION_SHA256=$(sha256sum "$SMOKE_PROMOTION_FILE" | awk '{print $1}')
export SMOKE_PROMOTION_FILE SMOKE_PROMOTION_SHA256
for MODEL in nano super; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    EXECUTION_PATH_SCRIPTS='execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
    "$EXP/submit_performance_matrix.sh"
done
for MODEL in qwen3_30ba3b qwen3_235b; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PHASE=performance REPEATS=3 \
    EXECUTION_PATH_SCRIPTS='execution_paths/02_policy_logprob.sh execution_paths/03_reference_logprob.sh execution_paths/07_all.sh' \
    "$EXP/submit_qwen_router_validation.sh"
done
CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" \
  MCORE_MODELS='nano super qwen3_30ba3b qwen3_235b' STEPS=20 \
  "$EXP/submit_mcore_matrix.sh"
```

- [ ] **Step 7: Publish the 20-to-100 gate and run accuracy soaks**

Rebuild/export the completed manifest for `performance`, publish
`results/gates/performance-promotion.json`, and pass its exact file and SHA to
generic `submit_accuracy_soak.sh` or Qwen
`submit_qwen_router_validation.sh PHASE=accuracy`. Run BF16 first for exactly
100 steps. Track reward, logprob errors, KL metrics, loss, grad norm, parameter
deltas, routes, storage generation, capture count, GPU utilization, and
NaN/Inf. Each accuracy submitter reads the exact promoted scope/path rows for
its model from the verified gate; it does not choose an unbound default. A
performance win with correctness failure is rejected.

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
uv run --locked python "$EXP/build_completed_run_manifest.py" \
  --phase performance --profile-file "$PROFILE_FILE" \
  --output "$EXP/results/manifests/completed-runs-performance.json"
uv run --locked python "$EXP/export_completed_runs.py" \
  --manifest "$EXP/results/manifests/completed-runs-performance.json" \
  --output-dir "$EXP/results/raw" --evidence-dir "$EXP/results/evidence"
uv run --no-project python "$EXP/publish_campaign_gate.py" performance \
  --manifest "$EXP/results/manifests/completed-runs-performance.json" \
  --raw-dir "$EXP/results/raw" \
  --smoke-gate "$EXP/results/gates/smoke-promotion.json" \
  --output "$EXP/results/gates/performance-promotion.json"
PERFORMANCE_PROMOTION_FILE=$(realpath "$EXP/results/gates/performance-promotion.json")
PERFORMANCE_PROMOTION_SHA256=$(sha256sum "$PERFORMANCE_PROMOTION_FILE" | awk '{print $1}')
export PERFORMANCE_PROMOTION_FILE PERFORMANCE_PROMOTION_SHA256
for MODEL in nano super; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    "$EXP/submit_accuracy_soak.sh"
done
for MODEL in qwen3_30ba3b qwen3_235b; do
  CLUSTER="$CLUSTER" PROFILE_FILE="$PROFILE_FILE" MODEL="$MODEL" \
    PHASE=accuracy REPEATS=1 \
    "$EXP/submit_qwen_router_validation.sh"
done
```

- [ ] **Step 8: Collect and render the durable report**

```bash
EXP=experiments/cuda_graph/nemotron_thd_te_graph_20260731
uv run --locked python "$EXP/build_completed_run_manifest.py" \
  --phase accuracy --profile-file "$PROFILE_FILE" \
  --output "$EXP/results/manifests/completed-runs-accuracy.json"
uv run --locked python "$EXP/export_completed_runs.py" \
  --manifest "$EXP/results/manifests/completed-runs-accuracy.json" \
  --output-dir "$EXP/results/raw" --evidence-dir "$EXP/results/evidence"
uv run --locked python "$EXP/build_completed_run_manifest.py" \
  --phase all --profile-file "$PROFILE_FILE" \
  --output "$EXP/results/manifests/completed-runs-all.json"
uv run --locked python "$EXP/export_completed_runs.py" \
  --manifest "$EXP/results/manifests/completed-runs-all.json" \
  --output-dir "$EXP/results/raw" --evidence-dir "$EXP/results/evidence"
uv run --no-project python "$EXP/collect_results.py" \
  --manifest "$EXP/results/manifests/completed-runs-all.json" \
  --raw-dir "$EXP/results/raw"
uv run --no-project python "$EXP/render_report.py"
git diff --check
```

The HTML must show, per model/scope/path, E2E step time and tokens/sec/GPU,
Generation, PolicyTraining, policy Logprob, reference Logprob, combined
Logprob, graph-call coverage, cache/fallback/recapture, padding utilization,
physical occupancy, correctness, job ID, exact provenance, and promotion-gate
digests. The immutable phase manifests, normalized JSONL, parity JSON, and
content-addressed evidence objects are committed so every rendered row and
gate remains auditable without W&B or ephemeral cluster logs.

- [ ] **Step 9: Final review and durable results commit**

Use `superpowers:requesting-code-review`, fix every correctness or provenance
finding, re-run affected tests, then commit only the generated result records,
README update, gate manifests, completed-run manifest, CSV/JSON tables, and
HTML artifacts. Results are ignored by default, so use a reviewed force-add
for exactly these paths and never force-add raw logs:

```bash
git add -f \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-smoke.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-performance.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-accuracy.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/completed-runs-all.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/manifests/fixed-batch-parity-manifest.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/gates/smoke-promotion.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/gates/performance-promotion.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/raw/*.jsonl \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/evidence/*.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/parity/*.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/results.csv \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/paired_20step_summary.json \
  experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/report.html
git add experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md
git commit -s -m "docs: report training and logprob CUDA graph results"
git push seonjinn experiment/thd-cg-hybrid-nemotron-20260731
```

## Completion Gate

Do not call the implementation complete until all conditions hold:

- MCore capability tests prove real eval/no-grad TE partial replay and no
  backward graph execution.
- Training, policy-logprob, and reference-logprob banks never cross execution
  kind, role, mode, storage generation, or schedule identity.
- Fixed packed THD inputs replay across changing logical occupancy without
  stale tokens, masks, cumulative lengths, routes, or outputs.
- BF16 reference value swaps preserve storage addresses and the training bank
  survives a full training-logprob-reference-training transition.
- Dense, Mamba, router, preprocess, and supported combined scopes pass exact
  eager-versus-graph correctness gates.
- Training R3 preserves per-schedule-slot routes with multiple outstanding
  1F1B microbatches; forward-only R3 never reuses a stale training slot.
- A distributed same-job parity artifact proves eager-versus-graph policy and
  reference outputs, routes, and gradient shards from identical state before
  any performance promotion.
- Attested standalone MCore results cover the minimum 8-rank suite and every
  model-required TP/PP/CP/EP topology; an undersized smoke is not substituted.
- Unsupported precision, dispatcher, R3, scope, or unseen geometry is rejected
  or selects collective whole-call fallback before model entry.
- Five-step smokes, paired 20-step performance, and the selected 100-step soak
  pass for every promoted row with exact provenance and nonzero requested-path
  graph coverage.
- Ultra's external model, data, judge, and profile inputs are supplied and its
  eligible gates pass. If it remains `blocked_external`, the implementation
  may be code-complete but this campaign completion gate remains unsatisfied.
- The HTML report separates E2E, Generation, PolicyTraining, policy Logprob,
  reference Logprob, and combined Logprob step time and tokens/sec/GPU.
- MCore, Bridge, and NeMo-RL commits are signed, pushed, and pinned in dependency
  order; no required changes or result artifacts remain uncommitted.
