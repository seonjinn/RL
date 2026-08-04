# Task 4 Report: W4A4 Calibration Artifact

## Result

Implemented provenance-checked W4A4 calibration artifacts in safetensors.
Artifact tensors are stored under exact Hugging Face projection `.weight`
names without model-prefix rewriting. ModelOpt
`.input_quantizer._amax` names are canonicalized only when writing, and
ambiguous aliases are rejected. Loading validates all required JSON metadata,
the exact model/revision/quant-config identity, scalar finite positive values,
and an optional exact expected projection-name set with missing and unexpected
names reported together.

Added a standalone exporter that loads a temporary Hugging Face model, reuses
the existing ModelOpt `get_tokenizer()` and `quantize_model()` calibration path
(which delegates dataset loop construction to `get_forward_loop_func()`),
collects enabled input quantizer values, saves the artifact, and reopens it for
full identity and projection-set validation. It rejects synthetic `random`
data and non-W4A4 quantization configs.

The review follow-up inspected ModelOpt at the exact repository pin
`c3b913b9cc1d82d5a0af9fa77b4db87829e6f158`. Its generic fused Hugging Face
expert wrapper exposes 3-D `gate_up_proj` and `down_proj` parameters,
`num_experts`, `gate_up_proj_input_quantizer`, and
`down_proj_input_quantizer`. The exporter now detects that structural API,
validates the expert count and projection dimensions, and expands each shared
quantizer value into exact logical HF expert projection names. It does not use
class-name matching. Missing, disabled, non-scalar, nonfinite, or nonpositive
required fused quantizers fail immediately. Ordinary quantized linears retain
their existing collection path.

`--model-revision` now pins both Hugging Face loads. The ModelOpt tokenizer
helper has an optional `revision` parameter; its existing default path is
unchanged, while an explicit revision is forwarded to
`AutoTokenizer.from_pretrained`. The exporter passes the same revision to that
helper and `AutoModelForCausalLM.from_pretrained`.

`real_quant_calibration_path` is propagated as an absolute
`VLLM_MODELOPT_CALIBRATION_PATH` value to inner vLLM workers only for W4A4.
W4A4 also propagates the original calibration quant-config identity through
the dedicated `VLLM_MODELOPT_CALIBRATION_QUANT_CFG` variable. File configs are
canonicalized to an absolute worker-visible path by the shared
`normalize_quant_cfg_identity()` helper; named configs retain their exact
identity. The exporter uses the same helper for saved metadata and its reopen
validation, preventing relative config paths from failing Task 5 provenance
checks.
W4A16, fake quantization, and stale worker environments ignore and clear it.
No source-aware requirement was added here; Task 5 retains ownership of the
BF16-versus-prepacked-QARL decision after `prepare_refit_info()`.

## RED

The initial artifact test run failed during collection as intended:

```text
ModuleNotFoundError: No module named 'nemo_rl.modelopt.calibration_artifact'
```

After adding the exporter behavior test, collection independently confirmed:

```text
ModuleNotFoundError: No module named 'examples.modelopt.export_nvfp4_calibration'
```

The isolated config test failed on the missing schema key:

```text
AssertionError: assert 'real_quant_calibration_path' in VllmConfig.__optional_keys__
1 failed, 22 deselected
```

Review follow-up RED captured the fused-wrapper and revision regressions:

```text
test_collect_input_amax_expands_fused_experts_to_exact_hf_names
AssertionError: assert [] == [
  'layers.0.mlp.experts.0.gate_proj.weight', ...
]

test_collect_input_amax_rejects_missing_fused_expert_quantizer
Failed: DID NOT RAISE RuntimeError

test_exporter_pins_model_and_tokenizer_to_same_revision
AssertionError: {'tokenizer_revision': None} !=
  {'tokenizer_revision': '0123456789abcdef'}
```

The late Task 5 integration preflight added one more RED assertion:

```text
test_w4a4_calibration_path_is_absolute_and_forwarded_to_workers
KeyError: 'VLLM_MODELOPT_CALIBRATION_QUANT_CFG'

test_exporter_pins_model_and_tokenizer_to_same_revision
assert '"nvfp4.yaml"' == '"/absolute/path/nvfp4.yaml"'
```

## GREEN

Artifact and exporter behavior:

```text
PYTHONPATH=. .venv/bin/python -m pytest --confcutdir=tests/unit/modelopt \
  tests/unit/modelopt/test_calibration_artifact.py -q
23 passed, 16 warnings
```

ModelOpt calibration helper coverage:

```text
PYTHONPATH=. .venv/bin/python -m pytest --confcutdir=tests/unit/models/policy \
  tests/unit/models/policy/test_modelopt_worker_utils.py -q
14 passed
```

Serializer and owned config/path propagation:

```text
PYTHONPATH=. .venv/bin/python -m pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_nvfp4_refit.py -q
24 passed, 16 warnings
```

Existing quant-worker regression coverage from the concurrently owned test
module:

```text
PYTHONPATH=. .venv/bin/python -m pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q \
  -k 'configure_quant_engine_kwargs'
5 passed, 91 deselected, 16 warnings
```

The warnings are local pytest temporary-directory cleanup warnings on macOS.
One first artifact run used the repository-wide `tests/unit` conftest and was
blocked before assertions because the minimal local Ray install lacks
`aiohttp_cors`; the dependency-light `--confcutdir` rerun above passed.

Static checks:

```text
.venv/bin/ruff check <Task 4 and helper Python files>
All checks passed!

.venv/bin/ruff format --check <Task 4 and helper Python files>
8 files already formatted

uvx --from pyrefly==0.24.2 pyrefly check \
  nemo_rl/modelopt/calibration_artifact.py \
  examples/modelopt/export_nvfp4_calibration.py
errors shown: 0, errors ignored: 0, modules: 2

.venv/bin/python -m py_compile <Task 4 and helper Python files>
passed

git diff --check
passed
```

## Configuration

`tests/unit/test_config_v2.py` confirms nested vLLM generation configuration
is still a legacy `TypedDict`, not an existing `BaseModel`. Following the
repository config convention, the field is `NotRequired[str | None]` on
`VllmConfig`; the Task 1 W4A4 rollout recipe already provides its documented
`null` value. No exemplar or reference YAML update is required. Both new
modules were added to the explicit `pyrefly.toml` project include list and
type-checked successfully.

## Cleanup And Scope

Generated bytecode and `tests/unit/unit_results*` files were removed. The
pre-existing untracked `session/` directory was left unchanged. Concurrent
changes in `vllm_quant_backend.py` and
`test_vllm_modelopt_real_quant_config.py` were used only for regression tests
and will not be staged in the Task 4 commit. The follow-up additionally owns
`nemo_rl/modelopt/models/policy/workers/utils.py` and
`tests/unit/models/policy/test_modelopt_worker_utils.py`; no common tokenizer,
YAML, Task 3, or session files were modified.
