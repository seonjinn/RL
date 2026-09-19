# Refit-aware vLLM Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant vLLM parameter backups at safe Sync-colocated
training boundaries while proving that the following full IPC refit rebuilds
every runtime parameter and failing the run if it does not.

**Architecture:** The generation caller sends a backend-neutral next-phase
intent. vLLM selects level-2 sleep only when the intent is
`TRAIN_THEN_FULL_REFIT` and the previous real IPC refit established a complete
runtime-coverage attestation. IPC synchronization invalidates that attestation
at entry, re-establishes it only after source-manifest validation, destination
load tracking, and finalization all succeed, and wakes KV cache only after the
transaction is complete.

**Tech Stack:** Python 3.12, Ray, PyTorch, vLLM 0.25.1, pytest, NeMo-RL
IPC/ZMQ weight synchronization, GB200 integration tests.

**Spec:** `docs/superpowers/specs/2026-09-18-refit-aware-vllm-sleep-design.md`

## Global Constraints

- Base production commits on NeMo-RL `main` at or after `b7a4d95d`.
- Do not add a recipe flag. Unknown or unproved cases preserve weights.
- The first refit always follows level 1; only a later training boundary may
  use an attestation established by a completed real refit.
- Preserve level 1 for sparse/delta refit, multimodal models, any realized MTP
  or external drafter, dynamic sampling, validation, setup, and checkpoint
  boundaries.
- Keep non-colocated Async-1off and NCCL Reshard behavior unchanged.
- Commit and push before every GB200 submission. Use the nightly image,
  `/home` for source, `/raid/scratch` for caches, and `/lustre` only for durable
  artifacts.
- Report 20-step performance over steps 2 through 19.

---

## Task 1: Preserve vLLM loader evidence and attest real refit coverage

**Files:**

- Modify: `nemo_rl/models/generation/vllm/quantization/fp8.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Test: `tests/unit/models/generation/test_vllm_refit_loader.py`
- Test: `tests/unit/models/generation/test_vllm_backend.py`
- Test: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
- Test: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**

- Produces: `fp8.load_weights(...) -> set[str] | None`.
- Produces: `VllmInternalWorkerExtension.refit_reconstructs_all_runtime_weights() -> bool`.
- Produces: `VllmGeneration.refit_reconstructs_all_runtime_weights() -> bool`.
- Contract: coverage becomes true only after one successful real IPC update;
  metadata completeness alone is never sufficient.

- [ ] Add failing loader-return tests.

  Assert that `quantization.fp8.load_weights()` returns the exact `set[str]`
  or `None` returned by `model.load_weights()`. Assert that
  `VllmInternalWorkerExtension._load_hf_weights()` and `_load_weights()` preserve
  the same result instead of dropping it.

- [ ] Add failing worker-attestation tests.

  Use the existing fake IPC socket and `_make_collective_update_extension`
  helpers. Cover these cases:

  - metadata initialization leaves coverage `False`;
  - a complete source manifest plus all runtime destination parameters loaded
    plus successful finalization sets coverage `True`;
  - a missing destination parameter, a loader result of `None`, an empty load
    result, incomplete source manifest, or failed finalizer leaves it `False`;
  - any realized drafter, including co-trained MTP, leaves it `False`;
  - multimodal and sparse/delta refit leave it `False`;
  - calling `prepare_refit_info()` again clears prior coverage;
  - a second successful A-to-B update may re-establish coverage after reset.

  Add a ModelOpt/MXFP8 case proving that parameters reconstructed by a
  successful quantization finalizer count as rebuilt only after finalization.

- [ ] Run the red tests.

```bash
uv run --group test pytest \
  tests/unit/models/generation/test_vllm_refit_loader.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  -k "loader_result or runtime_coverage or refit_attestation" -vv
```

Expected: failures show that FP8 and `_load_weights()` currently discard the
loader return value and no post-refit attestation exists.

- [ ] Implement destination load tracking.

  Change `fp8.load_weights()` to return `set[str] | None`. Propagate the result
  through `_load_hf_weights()` and `_load_weights()`. During
  `update_weights_via_ipc_zmq()`, union the names reported for every accepted
  batch. Do not derive success from source names alone.

  At the `COMPLETE` message, require this order:

  1. `manifest.require_complete()`;
  2. native or normal finalization;
  3. canonicalize loader-reported names through the same realized loader/refit
     ownership mapping used by the update, then compare them with
     `_get_named_parameters()`;
  4. store the worker attestation;
  5. send the completion ACK.

  Parameters owned by a module finalized through the native layerwise or
  quantization post-load lifecycle may be added to the reconstructed set only
  after that lifecycle succeeds. Derive ownership from realized modules and
  their quantization methods, not model-name or parameter-substring rules.

  Reset the worker attestation in `prepare_refit_info()`, at update entry, and
  on every exception. Worker construction naturally starts `False`.

- [ ] Expose strict aggregation through both engine wrappers.

  Add a worker RPC that returns the stored attestation. Sync and async NeMo-RL
  wrappers call vLLM `collective_rpc`; `VllmGeneration` returns `True` only for
  a nonempty result list where every expected model owner returned literal
  `True`. `[]`, `None`, missing workers, or partial results return `False`.

- [ ] Run focused tests and commit.

```bash
uv run --group test pytest \
  tests/unit/models/generation/test_vllm_refit_loader.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  tests/unit/models/generation/test_vllm_generation.py \
  -k "loader_result or runtime_coverage or refit_attestation" -vv
git add nemo_rl/models/generation/vllm/quantization/fp8.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  tests/unit/models/generation/test_vllm_refit_loader.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  tests/unit/models/generation/test_vllm_generation.py
git commit -s -m "feat(refit): attest complete vllm runtime coverage"
```

## Task 2: Add backend-neutral next-phase intent and safe level selection

**Files:**

- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Test: `tests/unit/models/generation/test_vllm_generation.py`
- Test: `tests/unit/single_controller/test_setup.py`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`

**Interfaces:**

- Consumes: `VllmGeneration.refit_reconstructs_all_runtime_weights()` from Task 1.
- Produces: `GenerationNextPhase.PRESERVE` and
  `GenerationNextPhase.TRAIN_THEN_FULL_REFIT`.
- Produces: `GenerationInterface.finish_generation_for_next_phase(next_phase: GenerationNextPhase) -> bool`.
- Produces: `WeightSynchronizer.can_discard_generation_weights -> bool`.
- Produces: `WeightSynchronizer.generation_weights_discarded -> bool` and
  `mark_generation_weights_discarded() -> None`.

- [ ] Add the semantic interface tests first.

  Define `GenerationNextPhase` with `PRESERVE` and
  `TRAIN_THEN_FULL_REFIT`. Add
  `GenerationInterface.finish_generation_for_next_phase(next_phase)` whose
  default delegates to `finish_generation()` without forwarding a new keyword.
  Test that the Megatron-compatible default does not change its concrete
  `finish_generation(*, release_gpu=True)` signature.

- [ ] Add the vLLM decision matrix tests.

  Cover all combinations of next-phase intent and synchronizer capability.
  Only `TRAIN_THEN_FULL_REFIT` plus capability `True` dispatches
  `discard_weights=True`; every other combination dispatches `False`. A direct
  call to existing `finish_generation()` remains preserving.

  Add sync and async worker tests mapping `discard_weights=False` to vLLM
  level 1 and `True` to level 2. Require worker sleep methods to return literal
  `True` after success.

- [ ] Add conservative synchronizer state.

  `WeightSynchronizer` defaults:

```python
@property
def can_discard_generation_weights(self) -> bool:
    return False

@property
def generation_weights_discarded(self) -> bool:
    return False

def mark_generation_weights_discarded(self) -> None:
    raise RuntimeError("This synchronizer cannot reconstruct discarded weights")
```

  The IPC implementation stores both booleans. Capability starts `False` and
  is not changed by `init_communicator()`.

- [ ] Implement vLLM phase selection transactionally.

  `VllmGeneration.finish_generation_for_next_phase()` selects level 2 only for
  destructive intent plus IPC capability. Before dispatching that sleep, call
  `mark_generation_weights_discarded()` so a partial Ray failure cannot lose
  the fact that some workers may already have discarded weights. A failed
  destructive dispatch raises. A preserving dispatch keeps historical
  `False` return behavior.

  Keep `finish_generation()` as a preserving compatibility wrapper. Do not pass
  vLLM-specific arguments through the generic Megatron backend.

- [ ] Run and commit.

```bash
uv run --group test pytest \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/single_controller/test_setup.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  -k "next_phase or discard_generation or sleep_level" -vv
git add nemo_rl/models/generation/interfaces.py \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  nemo_rl/models/generation/vllm/vllm_worker.py \
  nemo_rl/models/generation/vllm/vllm_worker_async.py \
  nemo_rl/weight_sync/interfaces.py \
  nemo_rl/weight_sync/ipc_weight_synchronizer.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/single_controller/test_setup.py \
  tests/unit/weight_sync/test_weight_synchronizer.py
git commit -s -m "perf(vllm): select sleep from verified refit intent"
```

## Task 3: Mark only guaranteed Sync GRPO training boundaries

**Files:**

- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `nemo_rl/algorithms/grpo_sync.py`
- Modify: `nemo_rl/experience/sync_rollout_actor.py`
- Test: `tests/unit/algorithms/test_grpo.py`
- Create: `tests/unit/experience/test_sync_rollout_actor.py`

**Interfaces:**

- Consumes: `GenerationNextPhase` and
  `finish_generation_for_next_phase()` from Task 2.
- Produces: Sync GRPO and TQ call sites that request destructive intent only
  at a guaranteed non-dynamic training boundary.

- [ ] Add failing caller tests.

  Standard non-dynamic colocated training must call
  `finish_generation_for_next_phase(TRAIN_THEN_FULL_REFIT)`. Setup, validation,
  checkpoint, non-colocated, and dynamic-sampling paths must call `PRESERVE` or
  existing `finish_generation()`. Assert that all Async-1off controller calls
  remain unchanged.

  For TQ, expose the unwrapped actor with
  `SyncRolloutActor.__ray_metadata__.modified_class` and test a focused helper
  that forwards the enum. Do not depend on the existing fixture returning the
  actor; it currently returns only an `ExitStack`.

- [ ] Run the red tests.

```bash
uv run --group test pytest \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/experience/test_sync_rollout_actor.py \
  -k "next_phase or training_rollout_finish" -vv
```

- [ ] Update exact Sync call sites.

  In the standard GRPO training boundary, request destructive intent only when
  inference is colocated and dynamic sampling is disabled. Dynamic sampling
  finishes generation before it knows whether another rollout is required, so
  it preserves. Carry the same condition through `grpo_sync.py` and
  `SyncRolloutActor.rollout_to_tq()`. Leave setup, validation outer finishes,
  and Async-1off calls preserving.

- [ ] Run and commit.

```bash
uv run --group test pytest \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/experience/test_sync_rollout_actor.py \
  -k "next_phase or training_rollout_finish" -vv
git add nemo_rl/algorithms/grpo.py nemo_rl/algorithms/grpo_sync.py \
  nemo_rl/experience/sync_rollout_actor.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/experience/test_sync_rollout_actor.py
git commit -s -m "feat(grpo): mark guaranteed full-refit boundaries"
```

## Task 4: Make IPC refit transactional and fail fast

**Files:**

- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_generation.py`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Test: `tests/unit/models/generation/test_vllm_generation.py`

**Interfaces:**

- Consumes: runtime coverage, capability, and discarded state from Tasks 1-2.
- Produces: an IPC transaction that clears discarded/stale state only after
  transfer, required attestation, and KV wake all succeed.

- [ ] Replace permissive failure expectations with transactional tests.

  Cover weight wake `False`, empty/`None` update results, partial update
  results, transfer exception, coverage-attestation failure, policy cleanup
  failure, and KV wake `False`. Every failed case leaves `is_stale=True` and
  does not resume generation. KV wake must not run after transfer failure.

  Distinguish preserving and destructive refits:

  - after a preserving refit, missing coverage disables future level 2 but does
    not invalidate an otherwise successful weight update;
  - after discarded weights, missing post-refit coverage is fatal;
  - only transfer + attestation (when required) + KV wake clears discarded
    state and staleness.

- [ ] Run the red tests.

```bash
uv run --group test pytest \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/models/generation/test_vllm_generation.py \
  -k "transaction or stale or discarded or empty_result" -vv
```

- [ ] Implement the ordered transaction.

  At sync entry set `_stale=True` and invalidate
  `_can_discard_generation_weights`. Snapshot whether weights had been
  discarded. Require `prepare_for_generation(tags=["weights"]) is True`.
  Require a nonempty update result where every expected worker returns literal
  `True`. Always restore policy state in bounded cleanup while preserving the
  primary exception.

  Query the Task 1 attestation only after the update has finalized. If weights
  were discarded and attestation is not complete, raise before KV wake. If
  weights were preserved, retain the successful refit but keep future discard
  disabled. Require `prepare_for_generation(tags=["kv_cache"]) is True`, then
  atomically set staleness and discarded state to `False` and retain the new
  capability value.

  Tighten `VllmGeneration.prepare_for_generation()` aggregation so an empty or
  all-`None` RPC result is not treated as success.

- [ ] Run and commit.

```bash
uv run --group test pytest \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/models/generation/test_vllm_generation.py \
  -k "IPCWeightSynchronizer or prepare_for_generation or finish_generation" -vv
git add nemo_rl/weight_sync/ipc_weight_synchronizer.py \
  nemo_rl/models/generation/vllm/vllm_generation.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/models/generation/test_vllm_generation.py
git commit -s -m "fix(refit): fail fast after destructive vllm sleep"
```

## Task 5: Complete local verification and PR self-review

**Files:** Review every production and test file modified in Tasks 1-4.

**Interfaces:**

- Consumes: the complete local implementation.
- Produces: a signed, pushed branch with passing affected suites and a completed
  NeMo-RL self-review.

- [ ] Run format and lint on every touched file.

```bash
TOUCHED=(
  nemo_rl/algorithms/grpo.py
  nemo_rl/algorithms/grpo_sync.py
  nemo_rl/experience/sync_rollout_actor.py
  nemo_rl/models/generation/interfaces.py
  nemo_rl/models/generation/vllm/quantization/fp8.py
  nemo_rl/models/generation/vllm/vllm_backend.py
  nemo_rl/models/generation/vllm/vllm_generation.py
  nemo_rl/models/generation/vllm/vllm_worker.py
  nemo_rl/models/generation/vllm/vllm_worker_async.py
  nemo_rl/weight_sync/interfaces.py
  nemo_rl/weight_sync/ipc_weight_synchronizer.py
  tests/unit/algorithms/test_grpo.py
  tests/unit/experience/test_sync_rollout_actor.py
  tests/unit/models/generation/test_vllm_backend.py
  tests/unit/models/generation/test_vllm_generation.py
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
  tests/unit/models/generation/test_vllm_refit_loader.py
  tests/unit/single_controller/test_setup.py
  tests/unit/weight_sync/test_weight_synchronizer.py
)
uv run ruff format --check "${TOUCHED[@]}"
uv run ruff check "${TOUCHED[@]}"
```

- [ ] Run complete affected suites.

```bash
uv run --group test pytest \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/models/generation/test_vllm_refit_loader.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/single_controller/test_setup.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/experience/test_sync_rollout_actor.py -vv
```

- [ ] Run `nemo-rl-pr-review` against `origin/main`, fix every blocking or
  maintainability finding, rerun the affected tests, and push the signed
  commits to `fork/sna/refit-aware-vllm-sleep-20260918`.

## Task 6: Prove numerical correctness on GB200

**Files:**

- Modify: `tests/unit/models/generation/test_vllm_generation.py`
- Create: `tests/test_suites/llm/vllm-destructive-refit-qwen3-30ba3b-4n4g.sh`
- Modify: `tests/test_suites/nightly_gb200.txt`

**Interfaces:**

- Consumes: the pushed implementation branch from Task 5.
- Produces: Blackwell A-to-B-to-C logits/logprob parity evidence and an injected
  failure result proving prompt termination.

- [ ] Extend the existing real MoE Megatron/vLLM integration test rather than
  creating a synthetic grouped-MoE model.

  Use the existing Qwen3-30B-A3B Sync performance topology: four GB200 nodes,
  16 GPUs, colocated policy/vLLM, EP16, FlashInfer TRTLLM, and the exact MXFP8
  ignore patterns from
  `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`.

  For fixed input tokens:

  1. perform the initial preserving refit of policy state A, verify that it
     establishes coverage, and record generation result A;
  2. apply a deterministic policy update, destructively sleep, refit B, and
     record B;
  3. apply a different deterministic update, destructively sleep, refit C, and
     record C;
  4. recreate a fresh vLLM generation object for B and C and compare each fresh
     result with the corresponding destructive-refit result;
  5. compare generation and Megatron multi-logprob/KL parity using a tolerance
     pinned in the test after recording the observed Blackwell range.

  Assert A differs from B and B differs from C so stale weights cannot pass.
  Also run a BF16 FlashInfer TRTLLM packed-layout control using the existing
  Qwen3.5 BF16 integration recipe, but keep it preserving if that recipe is not
  colocated.

- [ ] Add the nightly wrapper.

  Record exact Git SHA and nightly image digest, use node-local caches, invoke
  only the new pytest node IDs, and copy the final pytest/Ray logs to the durable
  experiment directory. Register the script in `nightly_gb200.txt`.

- [ ] Commit, push, submit with `sbatch --test-only`, and monitor one filtered
  job query per minute. The gate passes only when A-to-B-to-C parity, level-2
  selection after the first attested refit, and immediate injected-failure
  termination all pass.

## Task 7: Measure Qwen3-235B and Nemotron3 Super Sync

Create experiment artifacts outside the production PR. Do not edit the
upstream performance YAMLs.

**Files:** Create submission metadata, commands, logs, W&B CSV, and report under
the MXFP8 experiment workspace; do not add them to the NeMo-RL PR.

**Interfaces:**

- Consumes: the Task 5 branch and the separate optimizer-buffer reuse branch.
- Produces: matched 20-step performance and correctness tables for the two
  unchanged Sync recipes.

- [ ] Freeze these recipes and their exact GBS, sequence length, parallelism,
  logprob count, CUDA Graph, and FlashInfer TRTLLM settings:

```text
examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml
examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml
examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-mxfp8-rollout.yaml
```

- [ ] For each model submit current level 1, refit-aware level 2, and
  refit-aware level 2 plus the separate optimizer-buffer reuse branch. Before
  submission record NeMo-RL SHA, extra SHAs, container path/digest, complete
  Hydra command, account, GPU count, and W&B name. Commit and push first.

- [ ] Run `sbatch --test-only`, choose the highest-fairshare approved account,
  and monitor all submitted arms with one filtered scheduler query no more than
  once per minute. Require 20 logged steps, steps 2 through 19 present, no NaN,
  no refit failure, and matching effective configs. Record OOM or incomplete
  arms as failures instead of changing one arm's topology.

- [ ] Produce a matched W&B table with E2E, generation, policy training,
  logprob, refit, transfer/update, E2E and generation tokens/s/GPU, peak host
  RSS, refit median/p95/max, `gen_kl_error`, reward, entropy, W&B URL, and SHA.
  Keep old diagnostics separate from the new A/B. For Nemotron3 Super, state
  explicitly whether removing the parameter backup changes the prior host OOM
  or long-tail stall into a completed run.
