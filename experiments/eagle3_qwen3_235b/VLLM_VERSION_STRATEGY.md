# Qwen3-235B vLLM Version Strategy

Last updated: 2026-05-22 16:21 PDT

## Current Answer

`vLLM 0.10.2` is not being used because it is new. It is being used as the
first source-build recovery path for the already patched NeMo-RL/vLLM worker
integration, and that source-built path now passes native ABI in the target
container.

The target runtime is currently:

```text
container: /lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh
python: /opt/venv/bin/python
torch: 2.8.0a0+5228986c39.nv25.05
cuda: 12.9
arch: aarch64
```

This is not a nightly vLLM Docker image. It is a fixed NeMo 25.07.01 squashfs
image visible on oci-hsg. The `2026-05-22 13:24 PDT` oci-hsg check of the
visible llmservice container directory found only:

```text
nemo_25.04.02.sqsh
nemo_25.07.01.sqsh
nemo-rl.sqsh
```

No visible `nemo_25.09`, `nemo_25.11`, or standalone vLLM container was found in
that directory. If a newer or nightly image is staged elsewhere, it still needs
the same container preflight plus vLLM native/runtime probes before replacing
the current rollout runtime.

## Why The Current Job Uses 0.10.2

The earlier shared-wheel ABI probe tested these target sites:

```text
vllm 0.10.2 wheel target: FAIL import vllm._C
vllm 0.11.2 wheel target: FAIL import vllm._C
vllm 0.13.0 wheel target: FAIL import vllm._C
```

All three failed with the same unresolved torch/CUDA symbol:

```text
undefined symbol: _ZN3c104cuda9SetDeviceEab
```

That makes the immediate problem native ABI compatibility with the target
container torch build, not simply the public vLLM version number. Source-building
inside the exact NeMo container is the current gate.

Job `2855535` built `vllm-0.10.2.tar.gz` in-container. It produced a
`vllm-0.10.2+cu129` aarch64 wheel, then failed only because `pybase64` was
missing from the target site. Finalize job `2856310` installed that dependency
into the tmp site and wrote source-build PASS. ABI probe job `2856339` then
passed. The first rollout smoke `2856410` then failed on missing `cpuinfo` when
importing `AsyncLLM`; patch job `2856499` installed `py-cpuinfo` and verified
`AsyncLLM` import. The next retry `2856536` failed on missing `frozendict`
inside `compressed_tensors` during Qwen3-MoE model inspection; patch job
`2856588` installed `frozendict` and verified `Qwen3MoeForCausalLM` import. The
lightweight runtime probe `2856645` also passed `AsyncEngineArgs.create_engine_config()`
for Qwen3-235B Thinking. Strict probe `2856680` then passed the same runtime
path but failed `pip check` on package metadata and missing optional/runtime
dependencies. Patch `2856741` installed the low-risk missing packages, then its
probe found `pycountry`; follow-up `2856752` installed `pycountry` and verified
imports. Post-patch probe `2856767` passed the runtime path again and failed
only at strict `pip check` on deferred stack mismatches. The current canonical
path is:

```text
0.10.2 source build PASS -> source-site ABI probe PASS -> AsyncLLM import PASS -> Qwen3Moe import PASS -> engine-config probe PASS -> strict pip-check probe exposed cleanup deps -> patch jobs 2856741/2856752 completed useful deps -> post-patch runtime path PASS, strict pip check still FAILS on deferred stack mismatches -> rollout smoke 2856596 reached model load, then failed in vLLM compile/Inductor path -> compile-off retries 2857291/2857334 failed on Hydra append syntax -> retries 2857503 and 2857581 reached OpenAI serving setup, then failed on `OpenAIServingChat`/`OpenAIServingTokenization` `model_config` API drift -> `vllm_worker_async.py` now passes `model_config` based on constructor signatures -> retries 2858232/2858693 exposed Megatron-Bridge API drift -> one-node import probe 2858922 PASS -> active retry 2858959 is pending Slurm resources
```

The patch jobs deliberately installed only the missing low-risk packages
`opencv-python-headless`, `astor`, `interegular`, `pydantic-extra-types`, and
`levenshtein`, plus `pycountry` required by `pydantic-extra-types`. They do not
replace the container's NVIDIA Torch build, Ray, TorchVision, setuptools, or
Triton baseline.

Job `2856596` proved that the source-built vLLM site can launch the distributed
Qwen3-MoE path far enough to load the model, but it failed when vLLM's V1 engine
profile run entered `torch.compile`/Inductor and tried to import
`torch._inductor.standalone_compile`. The current fix is to keep vLLM 0.10.2
source-built but run rollout smoke with:

```text
policy.generation.vllm_cfg.enforce_eager=True
+policy.generation.vllm_kwargs.compilation_config.level=0
+policy.generation.vllm_kwargs.compilation_config.use_inductor=False
```

The next failure was a real RL-context integration issue: NeMo-RL exposes a vLLM
OpenAI-compatible chat/tokenize server for SWE-Gym, so it touches
`OpenAIServingChat` and `OpenAIServingTokenization`, not only bare vLLM
generation. Jobs `2857503` and `2857581` failed because the source-built vLLM
0.10.2 API requires `model_config` for these constructors. The current
SpecDec-RL patch uses `inspect.signature()` to pass `model_config` only when the
loaded vLLM class requires it. Follow-up retries then exposed Megatron-Bridge
API drift in the current NeMo container: missing `pg_utils`,
`ProcessGroupCollection`, `mcore_fsdp_adapter`, and `vocab_utils`, plus a Torch
2.9-only DTensor patch import. The current SpecDec-RL patch set handles those
differences, and one-node container probe `2858922` proves the patched
Megatron imports against the target container. Retry `2858959` is the active
compact smoke for that full compatibility set.

## Higher-Version Candidate

The next higher-version source-build candidate is `vLLM 0.13.0`, because:

- an aarch64 wheel and sdist exist for this release;
- the shared wheel site was already part of the ABI probe matrix;
- API drift is smaller than jumping directly to the latest vLLM branch;
- it is a more reasonable NeMo-RL compatibility target than the newest PyPI
  release while the local worker patch is still being stabilized.

Important caveat: PyPI metadata for `vLLM 0.13.0` declares `torch==2.9.0`,
while the target NeMo 25.07.01 container has
`torch 2.8.0a0+5228986c39.nv25.05`. The wrapper deliberately keeps using the
container torch and does not pip-install a second torch into the target site.
That means `0.13.0` is a compatibility candidate, not a guaranteed better
runtime.

Prepared wrapper, intentionally using separate reports and job file so it does
not overwrite the canonical `2855535` watcher state. The first automatic
fallback submit attempt failed before scheduling because it used `06:00:00`,
which exceeds the current oci-hsg `batch` partition limit. The wrapper and
Slurm build script now default to `04:00:00`.

```bash
SUBMIT=false \
SBATCH_ACCOUNT=coreai_dlalgo_nemorl \
SBATCH_PARTITION=batch \
bash experiments/eagle3_qwen3_235b/submit_vllm_native_source_build_0_13_0.sh
```

If submitted, run its ABI probe against only:

```text
/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/python_site/vllm_0_13_0_cu129_torch28nv_source_py312
```

Prepared companion wrappers:

```text
experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe_0_13_0.sh
experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke_0_13_0.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_0_13_0_then_rollout.sh
experiments/eagle3_qwen3_235b/watch_vllm_source_build_fallback_0_13_0.sh
```

`watch_vllm_source_build_fallback_0_13_0.sh` does not interrupt a running
canonical build. It can submit the `0.13.0` candidate only after a non-timeout
terminal failure; TIMEOUT/CANCELLED remain owned by the longer `0.10.2` retry
watchdog.

The 0.13.0 source-build fallback ran as job `2857812`, with watcher
`/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/reports/watch_vllm_source_build_2857812_0_13_0_then_abi.log`.
The wheel build itself completed, but the native import probe failed before a
PASS report was written because the target site loaded `transformers` requiring
`tokenizers>=0.22.0,<=0.23.0` while the runtime still resolved
`tokenizers==0.21.4`.
That watcher has `SUBMIT_ROLLOUT=false`, so it stops after source-build plus
native ABI validation and will not add another 32-node rollout job by itself.

Do not switch rollout to 0.13.0 until that source-built site passes native
imports and the NeMo-RL worker patch is checked against the 0.13 executor file
layout. Since the source-built `0.10.2` site now passes native ABI, `0.13.0` is
a fallback/runtime-quality candidate rather than the current unblocker.

## Latest vLLM And Speculators Are A Separate Track

As of the 2026-05-22 check, PyPI lists `vLLM 0.21.0` as the latest release,
released on 2026-05-15. That is much newer than the current NeMo 25.07.01
container, which is still on NVIDIA's `torch 2.8.0a0+5228986c39.nv25.05`.
The vLLM `speculators` online EAGLE3 tutorial also assumes a separate serving
environment with `vllm>=0.18` and a training environment with
`speculators>=0.5.0`, so it should be treated as a new backend probe rather
than a patch-level replacement for the current rollout runtime.
Jumping directly to the newest release is risky for this path because:

- public wheels may target a different CUDA/PyTorch baseline than the NeMo
  25.07.01 training container;
- vLLM executor internals have already changed across versions, which is why the
  SpecDec-RL patch searches multiple Ray executor layouts;
- the current goal is not standalone vLLM serving, but NeMo-RL rollout capture
  plus ModelOpt Eagle3 hidden-state/training/export.
- Speculators requires its own data-preparation and hidden-extraction path; it
  can use the same normalized rollout conversations as source data, but it is
  not wired into the current ModelOpt completion-audit chain yet.

Use the newest vLLM only after either:

- a matching newer NeMo/vLLM container is available on oci-hsg and passes the
  container preflight, or
- a source build of that exact vLLM release passes native ABI and the NeMo-RL
  patch is updated for its executor layout.

Source references for version tracking:

- PyPI release history: <https://pypi.org/project/vllm/>
- vLLM release index: <https://vllm.ai/releases>
- vLLM Speculators docs: <https://docs.vllm.ai/projects/speculators/en/latest/>
- Backend comparison: `SPECULATIVE_TRAINING_BACKENDS.md`

## Decision

Keep the source-built `0.10.2` path as the active RL smoke path while compact
retry `2858959` waits/runs. Treat `0.13.0` as a failed fallback until its
target-site dependency conflict is fixed and native ABI/import probes pass. Use
it for rollout only if `0.10.2` fails due runtime/API incompatibility after the
current Megatron compatibility patch set, acceptance/speed from the first
rollout smoke suggests a newer vLLM baseline is needed, or the operator
explicitly chooses to spend another rollout slot.
