# Megatron Energon worker-venv dependency fix

## Context

Ptyche job `2458270` passed local Hugging Face snapshot resolution, offline
vLLM initialization, and Ray cluster startup. It failed while importing
`MegatronPolicyWorker` because the isolated MCore venv did not contain
`megatron.energon`.

The NeMo-RL root project supplies static dependency metadata for the editable
`megatron-bridge` workspace. That metadata omits the
`megatron-core[dev,mlm]` dependency declared by both the Bridge source
`pyproject.toml` and the NeMo-RL Bridge workspace proxy. The omitted `dev`
extra is what provides `megatron-energon`.

## Decision

Align NeMo-RL's static `megatron-bridge` metadata with the Bridge workspace
proxy by restoring `megatron-core[dev,mlm]`, then regenerate `uv.lock`.

This fixes the dependency graph at its source. Adding only
`megatron-energon` would hide the metadata drift and could expose another
missing Bridge dependency later. Changing Bridge to lazily import Energon is
broader than this experiment and is out of scope.

## Validation

1. Add a regression test that compares the static Bridge dependency metadata
   with `CACHED_DEPENDENCIES` from the workspace proxy.
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
