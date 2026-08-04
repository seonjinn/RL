# Task 3 Report: BF16 W4A16 Receiver Refit

## Result

Implemented legacy IPC/ZMQ BF16 receiver refits into W4A16 NVFP4 ModelOpt
layouts. Source classification now resolves actual vLLM receiver ownership
through `LinearBase`/`ParallelLMHead`, `RoutedExperts`, and
`ModelOptNvFp4Config`, reusing the existing HF-to-vLLM resolver when available
and using an explicit mapping fallback for lightweight stubs.

BF16 candidates are serialized only for resolved quantized destinations.
Embeddings, unquantized linear layers, layernorms, biases, and configured
ignored weights pass through; ignored scale tensors retain existing filtering.
Gate/up groups can span transport batches and are cloned into owned staging
until complete. Incomplete groups fail before layerwise finalization with
missing names. BF16 serialized expert names are forwarded directly to the base
loader, while prepacked QARL batching remains unchanged.

## RED

- Focused receiver tests initially exposed a mixed-manifest fixture using a
  non-target packed prefix; the classifier correctly did not reject it. The
  fixture was changed to use packed scale components in the same resolved
  `q_proj` scope.
- The split-batch test initially used an expert name without the required
  model prefix, so the serializer treated it as a singleton. The fixture was
  changed to the actual `model.layers.0.mlp.experts.0` layout.
- The initial macOS focused run was blocked by the local environment's
  `transformers 5.14.1` assertion. The external `.venv` was adjusted to
  `transformers 5.11.0` and missing test-only packages; no repository files
  were changed for this setup.

## GREEN

Focused receiver tests:

```text
PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q \
  -k 'real_quant_prepare or real_quant_bf16 or real_quant_load_weights or real_quant_pre_ack or real_quant_rejects_incomplete or real_quant_accepts_processed or real_quant_scopes_native or real_quant_caches_scoped or real_quant_discovers or target_resolver'
24 passed, 67 deselected, 1 warning
```

Task 2 serializer regression tests:

```text
PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_nvfp4_refit.py -q
20 passed
```

Static checks:

```text
.venv/bin/ruff check nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
All checks passed!

.venv/bin/python -m py_compile nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
passed

git diff --check
passed
```

The repository's `uv run --frozen` Ruff command cannot start on this macOS
arm64 host because the lockfile only contains Linux environments. The
equivalent Ruff check passed from the local virtual environment.

## Commit

Feature commit: `f1f677cc7` (`feat(modelopt): load BF16 refits into W4A16 rollout`)

Changed files in the feature commit:

- `nemo_rl/modelopt/models/generation/vllm_quant_backend.py`
- `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`

This report is the only additional Task 3 artifact. No implementation blocker
remains. The initial-run environment limitations above are superseded by the
fresh follow-up results below.

## Independent Review Follow-Up

Resolved the independent review against vLLM 0.25.1 semantics:

- Receiver target discovery now calls `WeightsMapper.apply_list()` on complete
  weight-name variants, including Qwen3-MoE stacked q/k/v, dense MLP gate/up,
  and shared-expert gate/up mappings. The compatibility fallback uses only
  `packed_modules_mapping`; tests no longer invent a Qwen gate/up entry there.
- Receiver-owned non-BF16 `.weight` entries participate in packed-family
  classification, so an incomplete uint8 family fails with missing scales.
- Empty and all-ignored manifests fail with the no-receiver-target error.
- Receiver-owned QARL families are validated once, and that filtered pass
  produces the w13 shard metadata used during loading.
- The second-refit identity test now uses a fake layerwise lifecycle that
  replaces both the parameter and kernel during each load, restores the
  original runtime identities at finalization, and verifies loaded values over
  two refits.

### Follow-Up RED

The new mapper regression first failed resolving the complete stacked q-proj
name:

```text
PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q \
  -k 'target_resolver_handles_fused_linear_mapper_variants or empty_or_all_ignored or incomplete_packed_weight_family or derives_w13_metadata_once or complete_group_finalizes_once'
1 failed, 89 deselected, 1 warning
```

The remaining manifest regressions then produced the expected failures:

```text
PYTHONPATH=. .venv/bin/pytest --confcutdir=tests/unit/models/generation \
  --maxfail=0 tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  -q -k 'empty_or_all_ignored or incomplete_packed_weight_family or derives_w13_metadata_once or complete_group_finalizes_once'
4 failed, 1 passed, 90 deselected, 1 warning
```

The meaningful replacement/restoration identity test was the one passing case;
it strengthened coverage without requiring a production change.

### Follow-Up GREEN

Exact requested receiver suite:

```text
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py -q
94 passed, 1 skipped, 16 warnings
```

The skip is the pre-existing optional `modelopt.torch.quantization.calib`
import. The warnings are macOS temporary-directory cleanup warnings.

Exact requested serializer suite, including concurrent Task 2 additions:

```text
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_nvfp4_refit.py -q
23 passed, 16 warnings
```

Static checks:

```text
.venv/bin/ruff check nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
All checks passed!

PYTHONPATH=. .venv/bin/python -m py_compile \
  nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py
passed

git diff --check -- nemo_rl/modelopt/models/generation/vllm_quant_backend.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py \
  .superpowers/sdd/2026-08-04-bf16-nvfp4-rollout/task-3-report.md
passed
```

No generated `unit_results.json` or `unit_results/` artifacts remained after
the runs. Unrelated concurrent worktree edits were not staged or modified.

Follow-up implementation commit: `3b09ad94a`
(`fix(modelopt): honor vLLM weight mapping for BF16 refits`), DCO-signed.

No Task 3 implementation blocker remains.
