# Megatron Energon worker-venv dependency fix

## Context

Ptyche job `2458270` passed local Hugging Face snapshot resolution, offline
vLLM initialization, and Ray cluster startup. It failed while importing
`MegatronPolicyWorker` because the isolated MCore venv did not contain
`megatron.energon`.

The NeMo-RL root project intentionally omits the editable workspace dependency
`megatron-core[dev,mlm]` from its static Bridge metadata to avoid uv
workspace-name shadowing. NeMo-RL's `mcore` extra manually supplies the MCore
runtime dependencies, but it omitted `megatron-energon`. Bridge imports
Energon eagerly through its training configuration even for this text-only
policy-worker path.

## Decision

Add `megatron-energon[av-decode]~=7.0` directly to NeMo-RL's `mcore` extra,
then regenerate `uv.lock`.

Adding `megatron-core[dev,mlm]` to the root static Bridge metadata was tested
and rejected: it violates the repository's explicit
`OMITTED_WORKSPACE_DEPS` contract and produces conflicting Git URLs for
`fast-hadamard-transform`. Changing Bridge's eager imports is broader than
this experiment and remains out of scope.

## Validation

1. Add a regression test requiring the `mcore` extra to include the direct
   Energon runtime requirement.
2. Confirm that the test fails on the current tree and passes after the fix.
3. Regenerate the lock and verify that the selected `mcore` environment
   contains `megatron-energon`.
4. Run an isolated worker import smoke for `megatron.energon` and
   `MegatronPolicyWorker` on Ptyche.
5. Submit the same four-node, 20-step, checkpoint-disabled baseline using the
   verified local model snapshot and offline Hub mode.
6. Monitor the job for at least five minutes and record its state in the
   experiment HTML report.

## Scope

The change is limited to NeMo-RL dependency metadata, its lockfile, regression
tests, and experiment evidence. Megatron-Bridge, Megatron-LM, PR5672 CUDA
Graph behavior, model configuration, and training semantics remain unchanged.
