# SpecDec-RL Remote Patches

This file records the remote-only NeMo-RL changes currently required to make
Qwen3-235B SWE rollout data usable for Eagle3 draft-model training on
`oci-hsg-cs-001-vscode-02`.

Remote checkout:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL
```

## Purpose

The target corpus is not a plain math prompt dataset. For SWE/RL usage we need
Qwen3-235B rollout conversations that preserve message roles and can be
normalized into Eagle3 assistant-token training data. The patches below are the
current bridge between the existing RL rollout stack and the ModelOpt Eagle3
pipeline.

## Current Diff Summary

`nemo_rl/algorithms/grpo.py`

- Preserve `role` alongside logged `content` for both sync and async GRPO
  logging paths.
- This lets the rollout normalizer reconstruct conversation turns instead of
  guessing assistant/user/tool roles from text.

`nemo_rl/utils/logger.py`

- Make `mlflow`, `swanlab`, and `wandb` optional imports.
- Raise a clear `ModuleNotFoundError` only if the corresponding logger backend
  is enabled.
- This is required because rollout capture disables those loggers, but the
  system `/opt/venv` container does not always include every optional backend.

`nemo_rl/distributed/ray_actor_environment_registry.py`

- Add per-backend actor executable toggles:
  - `NEMO_RL_VLLM_EXECUTABLE_SYSTEM`
  - `NEMO_RL_SGLANG_EXECUTABLE_SYSTEM`
  - `NEMO_RL_MCORE_EXECUTABLE_SYSTEM`
  - `NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM`
- Keep the older broad `NEMO_RL_PY_EXECUTABLES_SYSTEM` compatibility behavior
  for NemoGym defaults.
- This is needed because the available container has a usable system torch/MCore
  path, while uv-built actor environments can fail on source builds such as
  `deep_ep`, and the vLLM actor path needs to be controlled separately.

`nemo_rl/models/generation/vllm/vllm_worker.py`

- Make the runtime vLLM Ray executor patch search multiple file layouts:
  `v1/executor/ray_executor.py`, `v1/executor/ray_distributed_executor.py`, and
  `executor/ray_distributed_executor.py`.
- This keeps the newer NeMo-RL patch path while supporting the source-built
  vLLM `0.10.2` layout used on oci-hsg.

`nemo_rl/models/generation/vllm/vllm_worker_async.py`

- Instantiate `OpenAIServingModels`, `OpenAIServingChat`, and
  `OpenAIServingTokenization` with `model_config` only when the loaded vLLM
  class constructor requires it.
- This is an RL-context compatibility fix: SWE-Gym uses the OpenAI-compatible
  chat/tokenize server path, so bare vLLM generation import probes are not
  enough evidence.
- Jobs `2857503` and `2857581` reached this serving setup and failed with
  `OpenAIServingChat.__init__() missing 1 required positional argument:
  'model_config'`. Retry `2858232` includes this patch.

`nemo_rl/models/policy/workers/megatron_policy_worker.py`

- Add a compatibility fallback when
  `megatron.bridge.training.utils.pg_utils.get_pg_collection` is missing from
  the installed Megatron-Bridge package.
- The fallback first looks for `_pg_collection` already attached to the model
  or wrapped model, then falls back to `parallel_state.get_model_parallel_group()`
  for the model-parallel reductions used in this worker.
- Job `2858232` proved that vLLM model load and OpenAI serving startup work with
  the previous patches, then failed when policy workers imported the missing
  `pg_utils` module. Retry `2858693` proved that direct `pg_utils` import can be
  bypassed but exposed the same container's missing `ProcessGroupCollection`.

`nemo_rl/models/megatron/setup.py`

- Make `ProcessGroupCollection` optional and pass `pg_collection` into
  Megatron-Bridge calls only when the installed API exposes both
  `ProcessGroupCollection` and a `pg_collection` parameter.
- This keeps the newer code path intact on recent Megatron-Bridge while letting
  the current oci-hsg NeMo container use its older `/opt/megatron-lm` API.
- Add a local fallback for `calculate_padded_vocab_size` when
  `megatron.bridge.utils.vocab_utils` is not present.

`nemo_rl/models/policy/workers/patches.py`

- Move the Torch 2.9-only DTensor alias-patch imports inside
  `apply_torch_aten_alias_tensor_patch()`.
- The current oci-hsg container uses NVIDIA Torch 2.8, where
  `propagate_single_input_strategy` is not exported from the same internal path.
  Importing the worker should not fail on that unused 2.9-only patch path.

## Current Launcher Strategy

`run_grpo_qwen3_235b_swe.sh` currently uses:

```text
RUN_UV_SYNC=false
DRIVER_LAUNCHER=/opt/venv/bin/python
NEMO_RL_PY_EXECUTABLES_SYSTEM=0
NEMO_RL_VLLM_EXECUTABLE_SYSTEM=1
NEMO_RL_MCORE_EXECUTABLE_SYSTEM=1
NEMO_RL_NEMO_GYM_EXECUTABLE_SYSTEM=1
INSTALL_VLLM_IN_SYSTEM=true
```

The setup command skips full `uv sync` to avoid torch/native-library skew, then
sources `experiments/eagle3_qwen3_235b/bootstrap_system_vllm_site.sh` before
the NeMo-RL driver starts. That bootstrap prepends a shared Lustre Python target
to `PYTHONPATH`:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312
```

This used to install `vllm==0.10.2` plus selected Python runtime dependencies
with `pip --target`. That is no longer considered sufficient evidence: the
bootstrap check now imports `vllm._C` and `CompilationConfig`, not only pure
Python symbols. The target container probes as `torch 2.8.0a0+nv25.05` / CUDA
12.9 on aarch64, and wheel target sites for vLLM `0.10.2`, `0.11.2`, and
`0.13.0` all fail native import with the same `c10::cuda::SetDevice` undefined
symbol. The current path is to build vLLM from source inside the target NeMo
container, install it into a shared Lustre Python target, then point all Ray
workers at that source-built target. `bootstrap_system_vllm_site.sh` now
preserves a `.vllm_bootstrap_spec` marker containing `source-build`; if native
import fails from such a target, it exits instead of overwriting the source
build with a pip wheel. Secret token values are not expanded into the logged
`COMMAND` string; they remain shell references until the job shell executes.

## Rollout Attempts

- `2854113`: fixed container/system launcher reached logger import and failed
  on missing `mlflow`.
- `2854569`: optional logger patch worked; failed because the configured full
  SWE dataset path was missing.
- `2854614`: used SWE-Gym 4-row example data; failed because vLLM actor was sent
  to system env without `vllm` installed.
- `2854647`: sent vLLM back through uv actor env; failed while building
  `deep_ep` because the isolated build env could not import torch.
- `2854690`: system vLLM/MCore/NemoGym actor envs; reached worker
  initialization but failed because the vLLM install was not visible on worker
  nodes.
- `2854736`: shared Lustre vLLM `--target` install was wired into `PYTHONPATH`
  before Ray actors are launched, but failed because the forced wheel URL was
  x86_64 while the OCI HSG nodes are `aarch64`.
- Next retry: keep the shared `PYTHONPATH` target but install `vllm==0.11.2`
  from PyPI so pip selects the NeMo-RL-pinned aarch64 wheel.
- `2854766`: PyPI selected the aarch64 vLLM wheel correctly, but worker
  initialization then failed on missing vLLM runtime dependency `msgspec`.
- Next retry: install selected vLLM runtime Python dependencies into the shared
  target as well, still avoiding torch/ray replacement.
- `2854801`: active retry with selected vLLM runtime Python dependencies
  installed into the shared target; it exposed a vLLM native extension / torch
  ABI mismatch with `vllm==0.13.0`.
- Next retry: use the repo-pinned `vllm==0.11.2` shared target.
- `2854875`: submitted with shared `vllm==0.11.2` target and SWE-Gym 4-row
  smoke data; it failed after startup with the same vLLM/torch ABI mismatch.
- `2855078`: 1-node container probe confirmed `/opt/venv/bin/python`,
  aarch64, CUDA 12.9, `torch 2.8.0a0+5228986c39.nv25.05`, and no default vLLM.
- `2855164`: 1-node shared-site probe confirmed `vllm==0.10.2` imports
  `SamplingParams` and `vllm.logger.init_logger` successfully with
  `transformers==4.55.2`, `tokenizers==0.21.4`, `pydantic==2.13.4`, and
  `pydantic-core==2.46.4`.
- `2855243`: submitted with the vLLM 0.10.2 shared site. It passed the native
  vLLM import/bootstrap step, then failed because NeMo-RL's worker patch assumed
  the newer `vllm/v1/executor/ray_executor.py` file. vLLM 0.10.2 uses
  `vllm/executor/ray_distributed_executor.py` for the `_init_workers_ray` call.
- Current SpecDec-RL patch: `nemo_rl/models/generation/vllm/vllm_worker.py`
  now searches multiple vLLM Ray executor layouts and applies the runtime-env
  patch to the first file containing the expected markers. This keeps the newer
  vLLM layout path while supporting the older 0.10.2 Ray executor layout; the
  0.10.2 wheel itself is not native-ABI-compatible with the target container.
- `2855291`: submitted after the worker patch as the `vllm0102-raypatch`
  SWE-Gym 1-step smoke. It reached Ray/vLLM worker startup, then failed at
  `from vllm.config import CompilationConfig` because `vllm._C.abi3.so` could
  not resolve `_ZN3c104cuda9SetDeviceEab` against the container Torch build.
- `2855450`: native ABI probe tested the shared vLLM `0.10.2`, `0.11.2`, and
  `0.13.0` target sites in the target container. All three failed `import
  vllm._C` with the same undefined symbol.
- `2855535`: active source-build job for vLLM `0.10.2+cu129` from the sdist,
  targeting `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_10_2_cu129_torch28nv_source_py312`.
  If this build passes native import, the next retry should set
  `SHARED_VLLM_SITE` to that source-built target and rerun the 1-step Qwen3-235B
  SWE rollout smoke.
- `2856410`: source-built vLLM target moved past native ABI, then failed on
  missing `cpuinfo`.
- `2856536`: after installing `py-cpuinfo`, failed during Qwen3-MoE inspection
  because `compressed_tensors` needed `frozendict`.
- `2856596`: after dependency fixes, reached model load and vLLM engine setup,
  then failed in vLLM compile/Inductor path because the target NVIDIA Torch
  build lacks `torch._inductor.standalone_compile`.
- `2857291` / `2857334`: compile-off retries failed because new
  `compilation_config` keys were passed without Hydra append syntax.
- `2857503` / `2857581`: compile-off syntax fixed; both reached vLLM OpenAI
  serving setup and failed on `OpenAIServingChat` `model_config` API drift.
- `2858232`: compact retry with the `model_config` compatibility patch,
  `enforce_eager=True`, `compilation_config.level=0`, and
  `compilation_config.use_inductor=False`. It reached Qwen3-MoE vLLM model
  load, KV-cache setup, and OpenAI server startup, then failed while importing
  `megatron_policy_worker.py` because the installed Megatron-Bridge package no
  longer exposes `megatron.bridge.training.utils.pg_utils`.
- `2858693`: compact retry after adding the first `get_pg_collection`
  compatibility fallback to `megatron_policy_worker.py`. It bypassed the direct
  `pg_utils` import, then failed because the same container also lacks
  `ProcessGroupCollection`.
- `2858759`, `2858840`, `2858886`: one-node container import probes used to
  flush the remaining import-time drift without spending another 16-node
  rollout attempt. They exposed missing `mcore_fsdp_adapter`, missing
  `vocab_utils`, and a Torch 2.9-only DTensor patch import mismatch.
- `2858922`: one-node container import probe PASS. It confirms
  `megatron_policy_worker.py` and `nemo_rl.models.megatron.setup` import in the
  target NeMo container. It also confirms this container's `get_model` and
  `_update_model_config_funcs` signatures do not accept `pg_collection`, and
  `ProcessGroupCollection` is absent. The reusable replay entry point is now
  `experiments/eagle3_qwen3_235b/submit_megatron_compat_probe.sh`.
- `2858959`: current compact rollout retry with the lower-level
  `parallel_state.get_model_parallel_group()` fallback in
  `megatron_policy_worker.py`, optional-`pg_collection` compatibility in
  `nemo_rl/models/megatron/setup.py`, vocab-size fallback, optional custom-FSDP
  import, and lazy Torch alias-patch imports.

The SWE-Gym 4-row data is a smoke-test input only. It must not be promoted to
the canonical Eagle3 training corpus. Production needs the real SWE rollout
dataset or a replacement SWE/code rollout source.
