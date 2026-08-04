# Task 5 Report: Legacy W4A4 Receiver Refit

## Result

Implemented legacy IPC/ZMQ BF16-source refits into vLLM W4A4 ModelOpt
destinations. `prepare_refit_info()` keeps the Task 3 source classifier and
exact receiver-owned BF16 target set authoritative. Only that source/mode
combination requires `VLLM_MODELOPT_CALIBRATION_PATH` and
`VLLM_MODELOPT_CALIBRATION_QUANT_CFG`, then calls
`load_nvfp4_calibration()` once with the exact target set.

Provenance uses the vLLM 0.25.1 `ModelConfig` API directly: `model` supplies
the model id, `revision` must be an explicit non-empty revision, and the
resolved `hf_config._commit_hash` is preferred when available. A configured
explicit revision is the fallback. Missing provenance and artifact
model/revision/config or projection mismatches fail during setup.

BF16 groups now serialize with `mode="w4a4"` and the loaded calibration.
Canonical per-projection input scales are cached only after the base vLLM
loader succeeds. Later refits serialize new weights with the in-memory
calibration, replace their generated scale values with the fixed cache, and
replay those values internally into vLLM layerwise reload metadata. This
replay is required because vLLM 0.25.1 restores and counts every registered
layer tensor during each `initialize_layerwise_reload()` cycle. It does not
reopen the artifact or add scale tensors to the BF16 transport manifest.

The focused two-refit test deliberately makes the serializer propose `0.25`
then `0.75`. Both reload cycles receive the cached `0.25`; after each
finalizer, the calibrated value and original weight, input-scale parameter,
and kernel identities are preserved. The artifact loader is called once at
setup, and each incoming transport batch contains only the BF16 weight.

The prepacked QARL W4A4 path remains separate. It does not require or call the
artifact loader and continues forwarding actor-supplied input scales on every
refit. Split gate/up staging, pre-ACK device synchronization, native ModelOpt
completion checks, and finalizer behavior are unchanged. BF16 expert output
uses only exact gate/up/down projection families; no fused `w13` or `w2`
checkpoint names are generated.

## RED

Setup, artifact-boundary, provenance, and QARL regression batch:

```text
PYTHONPATH=. .venv/bin/python -m pytest \
  --confcutdir=tests/unit/models/generation --maxfail=0 \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q \
  -k 'bf16_w4a4_prepare or prepacked_w4a4_keeps_actor_scales_without_artifact'

9 failed, 1 passed, 96 deselected, 16 warnings in 2.85s
```

All nine BF16 W4A4 cases reached the old setup rejection:

```text
ValueError: BF16 receiver refit currently supports W4A16 NVFP4 only;
W4A4 calibration is not available
```

The one passing case was the intentional prepacked QARL no-artifact
regression.

Dense two-refit lifecycle and complete expert-family batch:

```text
PYTHONPATH=. .venv/bin/python -m pytest \
  --confcutdir=tests/unit/models/generation --maxfail=0 \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q \
  -k 'bf16_w4a4_two_refits or bf16_w4a4_experts_emit'

2 failed, 106 deselected in 25.36s
```

Both failed at the same old W4A4 setup rejection before lifecycle behavior
could run.

## GREEN

Focused two-refit artifact/replay/identity lifecycle requested during Task 5:

```text
PYTHONPATH=. uv run --no-sync pytest \
  --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_bf16_w4a4_two_refits_open_artifact_once_and_replay_fixed_scale \
  -q --disable-warnings

1 passed in 2.57s
```

Full ModelOpt generation backend module:

```text
PYTHONPATH=. uv run --no-sync pytest \
  --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  -q --disable-warnings

110 passed, 1 skipped, 16 warnings in 3.92s
```

The skip is the pre-existing optional
`modelopt.torch.quantization.calib` import. The warnings are the existing
macOS pytest temporary-directory cleanup warnings.

Task 2 serializer suite:

```text
PYTHONPATH=. uv run --no-sync pytest \
  --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_nvfp4_refit.py -q --disable-warnings

24 passed, 16 warnings in 2.62s
```

Task 4 artifact suite:

```text
PYTHONPATH=. uv run --no-sync pytest \
  --confcutdir=tests/unit/modelopt \
  tests/unit/modelopt/test_calibration_artifact.py -q --disable-warnings

23 passed, 16 warnings in 0.84s
```

Static checks:

```text
.venv/bin/ruff check \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
All checks passed!

uvx --from pyrefly==0.24.2 pyrefly check \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py
errors shown: 0, errors ignored: 23, modules: 1

PYTHONPATH=. .venv/bin/python -m py_compile \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
passed

git diff --check -- \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  .superpowers/sdd/2026-08-04-bf16-nvfp4-rollout/task-5-report.md
passed
```

No `tests/unit/unit_results.json` or `tests/unit/unit_results/` artifacts were
generated.

## Changed Files

- `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
- `.superpowers/sdd/2026-08-04-bf16-nvfp4-rollout/task-5-report.md`

No Task 4 artifact, exporter, worker, or helper file was modified. The
pre-existing untracked `session/` directory was preserved.
