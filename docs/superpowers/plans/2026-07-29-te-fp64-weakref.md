# Transformer Engine FP64 Weak-Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lossless FP64 support to Transformer Engine's CUDA Graph weak-reference path and validate six matched 20-step packed Nemotron Nano runs.

**Architecture:** Patch the exact Transformer Engine v2.15 Python dtype registry and test pointer, storage, forward, and backward parity before integrating it. Keep the immutable NeMo-RL nightly native libraries, overlay only the reviewed `utils.py` on every Slurm container, and fail before Ray startup when version or provenance differs. Run all experiment rows with the same overlay and aggregate steady-state steps 6–20 in the HTML report.

**Tech Stack:** Python 3.13, PyTorch CUDA, Transformer Engine 2.15, Megatron Core, NeMo-RL, Ray, SLURM/Pyxis, Bash, pytest, W&B JSON export, static HTML.

## Global Constraints

- Base Transformer Engine source is exact tag `v2.15`, commit `42b840051647eef89761a16dfdff87e82bb253ab`.
- Add only `torch.float64: "<f8"` to the production dtype map; do not cast router probabilities or change CUDA Graph buffer reuse.
- Use a fresh Transformer Engine worktree; do not modify the dirty `/Users/sna/CudaGraph_PR/TransformerEngine` checkout.
- Do not build Transformer Engine native CUDA extensions.
- Keep NeMo-RL nightly image `/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260729_2472184.sqsh` unchanged.
- The nightly image SHA256 remains `cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91`.
- The installed Transformer Engine version must equal `2.15.0+42b84005`.
- The overlay target is `/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/transformer_engine/pytorch/utils.py`.
- Run three successful eager optimizer updates before CUDA Graph capture.
- All performance runs use 20 steps, sequence packing, cache capacity 2, at most 16 packed sequences, checkpointing disabled, and W&B project `sna-cg-study`.
- Primary performance measurements use steps 6–20 only.
- Follow RED–GREEN TDD and preserve every failing/passing job ID.
- Before every submission: commit and push, pull the exact commit on Ptyche, check FairShare, run `sbatch --test-only`, and monitor for at least five minutes.
- Never stage or revert the pre-existing NeMo-RL `uv.lock` change.

---

### Task 1: Create the Isolated Transformer Engine Branch and Fork

**Files:**
- Create worktree: `/Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729`
- No tracked source changes in this task.

**Interfaces:**
- Consumes: NVIDIA Transformer Engine tag `v2.15`.
- Produces: branch `sj/fp64-cuda-graph-weakref-20260729` with a writable `seonjinn` remote.

- [ ] **Step 1: Fetch the exact tag without touching the dirty worktree**

Run from `/Users/sna/CudaGraph_PR/TransformerEngine`:

```bash
git fetch origin \
  refs/tags/v2.15:refs/tags/v2.15 \
  refs/heads/release_v2.15:refs/remotes/origin/release_v2.15
git rev-parse v2.15^{commit}
```

Expected output:

```text
42b840051647eef89761a16dfdff87e82bb253ab
```

- [ ] **Step 2: Create the isolated worktree**

```bash
git worktree add \
  /Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729 \
  -b sj/fp64-cuda-graph-weakref-20260729 \
  v2.15
```

- [ ] **Step 3: Create the personal fork and configure remotes**

```bash
gh repo fork NVIDIA/TransformerEngine --clone=false
git -C /Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729 \
  remote add seonjinn https://github.com/seonjinn/TransformerEngine.git
git -C /Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729 \
  remote -v
```

Expected: `origin` points to NVIDIA and `seonjinn` points to the new personal
fork.

- [ ] **Step 4: Verify isolation**

```bash
git -C /Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729 \
  status --short
git -C /Users/sna/CudaGraph_PR/TransformerEngine \
  status --short
```

Expected: the new worktree is clean; the original checkout still shows only its
pre-existing `transformer_engine/pytorch/graph.py` change.

### Task 2: Add FP64 Weak-Reference Tests and Verify RED

**Files:**
- Modify: `/Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729/tests/pytorch/test_cuda_graphs.py`
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_te_fp64_weakref.sub`

**Interfaces:**
- Consumes: unmodified TE v2.15 dtype map.
- Produces: focused tests that fail specifically with `Unsupported dtype: torch.float64` and a reusable one-GPU Ptyche launcher.

- [ ] **Step 1: Add imports and the pure dtype-contract test**

Add:

```python
from transformer_engine.pytorch.utils import _WeakRefTensor, make_weak_ref


def test_fp64_weak_ref_cuda_array_interface() -> None:
    weak = _WeakRefTensor(0x1000, torch.float64, (2, 3))

    interface = weak.__cuda_array_interface__

    assert interface["typestr"] == "<f8"
    assert interface["shape"] == (2, 3)
    assert interface["data"] == (0x1000, False)
```

The production change that makes this pass is the single FP64 dtype-map entry.

- [ ] **Step 2: Add the CUDA pointer/storage test**

Add:

```python
def test_make_weak_ref_preserves_fp64_cuda_storage() -> None:
    source = torch.arange(8, device="cuda", dtype=torch.float64)

    weak = make_weak_ref(source)

    assert weak.dtype is torch.float64
    assert weak.shape == source.shape
    assert weak.data_ptr() == source.data_ptr()
    weak.add_(1.0)
    torch.testing.assert_close(weak, source, rtol=0.0, atol=0.0)
```

- [ ] **Step 3: Add the buffer-reuse forward/backward test**

Add:

```python
class _FP64GraphOutput(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.float64).square()


def test_make_graphed_callables_reuses_fp64_outputs() -> None:
    module = _FP64GraphOutput().cuda().train()
    eager_inputs = [
        torch.linspace(-1.0, 1.0, 8, device="cuda", requires_grad=True)
        for _ in range(2)
    ]
    graph_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in eager_inputs
    ]
    eager_outputs = [module(tensor) for tensor in eager_inputs]
    for output in eager_outputs:
        output.sum().backward()

    graphed = make_graphed_callables(
        (module,),
        tuple((tensor,) for tensor in graph_inputs),
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
    )
    graph_outputs = [callable_(tensor) for callable_, tensor in zip(graphed, graph_inputs)]
    for output in reversed(graph_outputs):
        output.sum().backward()

    for eager_output, graph_output in zip(eager_outputs, graph_outputs):
        assert graph_output.dtype is torch.float64
        torch.testing.assert_close(graph_output, eager_output, rtol=0.0, atol=0.0)
    for eager_input, graph_input in zip(eager_inputs, graph_inputs):
        torch.testing.assert_close(
            graph_input.grad,
            eager_input.grad,
            rtol=0.0,
            atol=0.0,
        )
```

- [ ] **Step 4: Create the persistent one-GPU SLURM launcher**

The launcher must:

- use the immutable NeMo-RL nightly image;
- mount `/lustre:/lustre`;
- accept `TE_SOURCE_ROOT` and `TE_OVERLAY_ENABLED=0|1`;
- add the `utils.py` bind mount only when the overlay is enabled;
- run only the three focused pytest tests with `-vv`;
- write stdout/stderr under
  `exp_logs/mamba_moe_te_graph_20260729/te-fp64-weakref/`.

Use:

```bash
python -m pytest -vv \
  "${TE_SOURCE_ROOT}/tests/pytorch/test_cuda_graphs.py" \
  -k "fp64_weak_ref or reuses_fp64_outputs"
```

- [ ] **Step 5: Commit and push the RED tests before submission**

Commit the TE test file:

```bash
git add tests/pytorch/test_cuda_graphs.py
git commit -s -m "test: cover FP64 CUDA graph weak references"
git push -u seonjinn sj/fp64-cuda-graph-weakref-20260729
```

Commit the launcher separately in NeMo-RL without `uv.lock`:

```bash
git add \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_te_fp64_weakref.sub
git commit -s -m "test: add TE FP64 weak-ref GPU gate"
git push seonjinn experiment/pr5672-mamba-moe-graph-cache-20260729
```

- [ ] **Step 6: Sync, preflight, submit, and verify RED**

Pull both pushed branches on Ptyche. Check FairShare, then run:

```bash
sbatch --test-only \
  --account=coreai_dlalgo_llm \
  --partition=batch \
  --nodes=1 \
  --gres=gpu:1 \
  --export=ALL,TE_OVERLAY_ENABLED=0,TE_SOURCE_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/TransformerEngine-fp64-weakref-20260729 \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_te_fp64_weakref.sub
```

Submit the same command without `--test-only`, record the job ID, and monitor
for at least five minutes.

Expected: the tests fail at the FP64 dtype conversion with
`TypeError: Unsupported dtype: torch.float64`, proving RED.

### Task 3: Implement the Minimal Transformer Engine Fix and Verify GREEN

**Files:**
- Modify: `/Users/sna/CudaGraph_PR/TransformerEngine-fp64-weakref-20260729/transformer_engine/pytorch/utils.py`

**Interfaces:**
- Consumes: the RED tests from Task 2.
- Produces: FP64 CUDA-array-interface support with identical pointer, storage, output, and gradient behavior.

- [ ] **Step 1: Add the single production mapping**

Insert immediately after `torch.float32`:

```python
    torch.float64: "<f8",
```

Do not change `make_weak_ref()`, graph scheduling, or any numeric operation.

- [ ] **Step 2: Run format and diff checks**

```bash
git diff --check
python -m compileall -q transformer_engine/pytorch/utils.py
git diff -- transformer_engine/pytorch/utils.py
```

Expected: the production diff contains one added mapping line.

- [ ] **Step 3: Commit and push GREEN implementation**

```bash
git add transformer_engine/pytorch/utils.py
git commit -s -m "fix: support FP64 CUDA graph weak references"
git push seonjinn sj/fp64-cuda-graph-weakref-20260729
```

- [ ] **Step 4: Compute and record immutable overlay provenance**

```bash
git rev-parse HEAD
sha256sum transformer_engine/pytorch/utils.py
```

Record the literal commit and file SHA256 in the NeMo-RL Ptyche profile,
experiment manifest, and report inputs during Task 4.

- [ ] **Step 5: Verify GREEN on one GB200**

Pull the new TE commit on Ptyche. Run the Task 2 launcher with
`TE_OVERLAY_ENABLED=1`, first through `sbatch --test-only`, then as a real job.

Expected:

- all three focused tests pass;
- reported TE version is `2.15.0+42b84005`;
- FP64 source and weak-reference pointers match;
- output and input-gradient comparisons are exact;
- no native TE compilation occurs.

Monitor at least five minutes and preserve the job ID and log path.

### Task 4: Integrate and Preflight the TE Source Overlay in NeMo-RL

**Files:**
- Create: `experiments/cuda_graph/mamba_moe_te_graph_20260729/validate_te_fp64_overlay.py`
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/profiles/ptyche.env`
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/run_scope.sh`
- Modify: `tests/unit/experiments/test_mamba_moe_te_graph_launchers.py`
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`

**Interfaces:**
- Consumes: pushed TE commit and `utils.py` SHA256 from Task 3.
- Produces: identical baseline/CG software stacks with a version- and SHA-gated overlay on every node.

- [ ] **Step 1: Write failing launcher contract tests**

Add assertions that Ptyche `TEST_ONLY=1` output contains:

- the exact TE source commit from Task 3;
- the exact TE overlay file SHA256 from Task 3;
- source path under the immutable Ptyche TE checkout;
- target
  `/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/transformer_engine/pytorch/utils.py`;
- a `:ro` bind mount;
- the committed overlay validator in `SETUP_COMMAND`;
- the same overlay for baseline and TE scope launchers.

Also add a negative test that replaces the expected SHA and asserts the
validator exits nonzero before training.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen python -m pytest -q \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py \
  -k "fp64_overlay"
```

Expected: FAIL because overlay provenance and validation are not wired.

- [ ] **Step 3: Implement the typed validator**

Implement:

```python
def validate_overlay(
    *,
    expected_version: str,
    expected_sha256: str,
) -> dict[str, str]:
    ...
```

It must:

- import `torch`, `transformer_engine`, and
  `transformer_engine.pytorch.utils`;
- hash `te_utils.__file__`;
- compare the exact version and file SHA256;
- require `_torch_dtype_to_np_typestr_dict[torch.float64] == "<f8"`;
- allocate one CUDA FP64 Tensor and require `make_weak_ref()` to preserve
  dtype, shape, and pointer;
- print a JSON provenance record;
- raise before Ray startup on any mismatch.

- [ ] **Step 4: Wire the immutable overlay**

In `ptyche.env`, add literal values from Task 3 for:

```text
TE_FP64_WEAKREF_COMMIT
TE_FP64_WEAKREF_SHA256
TE_FP64_WEAKREF_SOURCE
TE_FP64_WEAKREF_TARGET
TE_EXPECTED_VERSION
```

Append the source-to-target `:ro` mount to `MOUNTS`. Append one invocation of
`validate_te_fp64_overlay.py` to the existing head-node `SETUP_COMMAND`.
Because every node uses the same immutable container and shared bind mount, do
not duplicate the expensive package reinstall on workers.

Expose the provenance values in `run_scope.sh` output and refuse submission
when any field is empty.

- [ ] **Step 5: Verify GREEN**

```bash
uv run --frozen python -m pytest -q \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
uv run --frozen ruff check \
  experiments/cuda_graph/mamba_moe_te_graph_20260729/validate_te_fp64_overlay.py \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py
git diff --check
```

- [ ] **Step 6: Commit and push**

Stage only the five task-owned paths, commit with:

```text
feat: validate FP64 TE weak-ref overlay
```

Push the NeMo-RL branch. Confirm `git status --short` still lists only the
pre-existing `uv.lock`.

### Task 5: Add Overlay Provenance and Steady-State Aggregation to the Report

**Files:**
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/render_report.py`
- Modify: `experiments/cuda_graph/mamba_moe_te_graph_20260729/collect_results.py`
- Modify: `tests/unit/experiments/test_mamba_moe_te_graph_launchers.py`

**Interfaces:**
- Consumes: normalized per-step rows and Task 3 provenance.
- Produces: HTML tables for steps 6–20 with baseline ratios, medians, p95, correctness deltas, and TE overlay identity.

- [ ] **Step 1: Write failing aggregation and provenance tests**

Use synthetic baseline and CG rows for steps 4–20. Assert:

- steps 4 and 5 do not enter the steady-state aggregate;
- steps 6–20 produce the expected median and nearest-rank p95;
- each CG throughput is divided by the baseline throughput;
- any eviction or fallback marks the row invalid;
- HTML contains TE version, TE commit, overlay SHA256, container SHA256, and
  measurement window `6–20`.

- [ ] **Step 2: Verify RED**

```bash
uv run --frozen python -m pytest -q \
  tests/unit/experiments/test_mamba_moe_te_graph_launchers.py \
  -k "steady_state or overlay_provenance"
```

- [ ] **Step 3: Implement aggregation**

Add typed functions:

```python
def steady_state_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    first_step: int = 6,
    last_step: int = 20,
) -> list[Mapping[str, str]]:
    ...


def aggregate_performance(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    ...
```

Group by `scope` and `job_id`. Compute median and nearest-rank p95 for all four
time and throughput categories. Add baseline throughput ratios and an explicit
valid/invalid reason.

- [ ] **Step 4: Add provenance arguments and HTML fields**

Add required render arguments:

```text
--te-version
--te-source-commit
--te-overlay-sha256
```

Render them beside the immutable container SHA. Add steady-state and
correctness-delta tables without removing the raw failure ledger.

- [ ] **Step 5: Verify GREEN and commit**

Run the full experiment unit test, Ruff, and `git diff --check`. Commit only
the three task files with:

```text
report: aggregate FP64 TE graph validation
```

Push the NeMo-RL branch.

### Task 6: Run the GPU Integration Gate Before NeMo-RL

**Files:**
- Modify:
  `experiments/cuda_graph/mamba_moe_te_graph_20260729/scripts/validate_nemorl_integration.sub`
- Update: `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`

**Interfaces:**
- Consumes: green TE overlay and current Megatron Core graph-bank branch.
- Produces: one passing GPU gate for packed attention/Mamba/MoE router capture, replay, and backward.

- [ ] **Step 1: Add the overlay to the existing integration launcher**

Use the same `:ro` mount and call the same validator before pytest. Keep the
existing NeMo-RL test list and add this exact Megatron Core command:

```bash
MCORE_ROOT=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
python -m pytest -q \
  "${MCORE_ROOT}/tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py" \
  "${MCORE_ROOT}/tests/unit_tests/transformer/test_te_cuda_graph_bank.py" \
  "${MCORE_ROOT}/tests/unit_tests/transformer/test_cuda_graphs.py::test_moe_router_fp64_output_is_preserved_at_te_graph_boundary" \
  "${MCORE_ROOT}/tests/unit_tests/transformer/test_cuda_graphs.py::test_packed_mamba_te_cuda_graph_parity" \
  "${MCORE_ROOT}/tests/unit_tests/transformer/test_cuda_graphs.py::test_te_graph_bank_schedule_switch_5_3_5"
```

- [ ] **Step 2: Run local launcher tests and commit**

Run the experiment unit suite, commit only the launcher/README changes, and
push.

- [ ] **Step 3: Preflight and submit one GPU integration job**

On Ptyche:

1. pull the exact NeMo-RL, Megatron-Bridge, Megatron-LM, and TE commits;
2. check FairShare;
3. run `sbatch --test-only`;
4. submit the existing integration launcher;
5. monitor for at least five minutes.

Gate requirements:

- every focused test passes;
- eager/graph router top-k IDs and masks are exact;
- FP64 probabilities, outputs, and gradients satisfy the existing tolerances;
- capture and replay both occur;
- no native TE build appears in logs.

Do not submit the 20-step matrix until this gate passes.

### Task 7: Submit the Six 20-Step Nano Runs in Parallel

**Files:**
- Update after submission:
  `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_submissions.json`

**Interfaces:**
- Consumes: passing Task 6 gate and committed performance launchers.
- Produces: six matched job IDs and complete per-run logs/W&B data.

- [ ] **Step 1: Run exact local and remote preflight**

Use:

```bash
CLUSTER=ptyche \
MODEL=nano-hybrid \
TEST_ONLY=1 \
RUN_TAG=te-fp64-weakref-20step-20260729 \
bash experiments/cuda_graph/mamba_moe_te_graph_20260729/submit_performance.sh \
  scopes/00_baseline_no_cg.sh \
  scopes/21_attn_mamba.sh \
  scopes/02_moe.sh \
  scopes/03_moe_router.sh \
  scopes/04_moe_router_preprocess.sh \
  scopes/24_attn_mamba_moe_router_preprocess.sh
```

Verify six commands, each with `STEPS=20`, checkpointing false, warmup 3,
packing enabled, cache capacity 2, the same overlay commit/SHA, and no singleton
dependency.

- [ ] **Step 2: Check scheduling**

Run `sbatch --test-only` for all six generated commands and inspect current
FairShare for `coreai_dlalgo_llm`. Do not serialize them with dependencies.

- [ ] **Step 3: Submit all six rows**

Run the same `submit_performance.sh` command with `TEST_ONLY=0`. Capture the
launcher-to-job mapping directly from `sbatch --parsable` output and write it
to the submission ledger.

- [ ] **Step 4: Monitor for at least five minutes**

For every job, confirm:

- Ray head and workers start;
- the TE overlay preflight passes;
- model/checkpoint loading begins;
- no registry, mount, port, authentication, CUDA, or NCCL failure appears.

Continue monitoring through step four long enough to confirm actual graph
capture for CG rows. Cancel only a proven invalid duplicate or a run that
cannot satisfy the correctness gate.

### Task 8: Collect Results, Refresh HTML, and Record the Decision

**Files:**
- Update: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_submissions.json`
- Generate: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_results.csv`
- Generate: `experiments/cuda_graph/results/mamba_moe_te_graph_20260729_report.html`
- Update: `experiments/cuda_graph/mamba_moe_te_graph_20260729/README.md`

**Interfaces:**
- Consumes: completed six-run ledger, W&B export, Slurm graph telemetry, and exact source provenance.
- Produces: reviewable performance/correctness comparison and accept/reject decision.

- [ ] **Step 1: Mark each run from scheduler and logs**

Record final Slurm state, exit code, completed optimizer steps, first capture
step, graph keys, capture/replay counts, cache hits, evictions, and fallbacks.

- [ ] **Step 2: Normalize per-step metrics**

Collect E2E, Generation, Policy Training, and Logprob time/throughput plus loss,
reward, generation KL error, token probability error, and gradient norm.
Preserve steps 1–20; do not pre-average away capture behavior.

- [ ] **Step 3: Render the report**

Run the collector and renderer with exact NeMo-RL, Bridge, MCore, container,
TE source, model, and tokenizer provenance. Verify the primary table uses only
steps 6–20.

- [ ] **Step 4: Apply decision gates**

Reject a row for:

- fewer than 20 optimizer updates;
- NaN/Inf, illegal memory access, NCCL failure, skipped update;
- any eager fallback or eviction;
- failed fixed-input output/gradient parity.

Mark a row correctness-acceptable when Policy and E2E throughput are at least
`0.98x` baseline. Claim improvement only when Policy throughput is at least
`1.05x` and E2E regression is at most `2%`.

- [ ] **Step 5: Commit and push the final evidence**

Stage only the ledger, CSV, HTML, and README. Commit with:

```text
report: record FP64 TE CUDA graph results
```

Push the NeMo-RL branch and report the job IDs, steady-state table, correctness
gates, failures, and local HTML path to the user.
