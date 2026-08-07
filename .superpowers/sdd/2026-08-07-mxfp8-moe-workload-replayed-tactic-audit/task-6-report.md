# Task 6 Report: Workload-Replayed FlashInfer Tactic Shmoo

## Outcome

Implemented and signed the Task 6 adapter, shmoo CLI, and CPU/mock-safe unit tests.

- Implementation commit: `cbdb93b51ae220e29b1d449a65b6e93837b382ed`
- Commit subject: `feat: shmoo workload-replayed MXFP8 MoE tactics`
- Review-fix commit: `c0b54396a6f68060180cf4d4881075c9353a15ed`
- Review-fix subject: `fix: harden FlashInfer tactic shmoo contracts`
- Re-review commit: `1c027fe0f929d52771e492531f4ac05012adffda`
- Re-review subject: `fix: preserve bounded Task 6 smoke compatibility`
- FlashInfer is rejected unless the installed `flashinfer-python` distribution is exactly `0.6.13`.
- Legal tactics are enumerated as paired FC1/GEMM1 and FC2/GEMM2 IDs through the TRTLLM Gen MoE API.
- Tactic forcing snapshots and restores `AutoTuner._file_configs` and `AutoTuner.profiling_cache`, including exception paths. It temporarily resets and restores the exact `_logged_file_hits` marker needed to prove a fresh manual file-config hit.
- Routing reconstruction is deterministic from `signature_key`, preserves the exact expert histogram, rejects duplicate-expert requirements, and emits BF16 weights that sum exactly to one per token.
- Each measured tactic remains a pair. The timed CUDA Graph contains the complete routed MoE operation, with exactly three warmups, ten or more repetitions, and a greater-than-L2 touch before every replay.
- Candidate FC1 activated intermediates and final FC2 reduced outputs are both checked against stock `[-1, -1]` references.
- A missing intermediate return contract emits `flashinfer_intermediate_api_unavailable`; ordinary tactic exceptions emit a serializable failure row and do not stop later tactics.
- Production CLI execution requires `--weights` pointing to a validated `flashinfer_mxfp8_moe_prepacked_v1` artifact. Synthetic weight creation is available through explicit `--synthetic-smoke` and through only the sanctioned source-less `--profile-limit=1 --tactic-limit=2` smoke shape; both paths apply the pinned upstream gated-row reorder and matrix/scale-factor shuffle pipeline and mark every row with `"synthetic": true`.

## Review Corrections

All six review findings were implemented in the four Task 6 code/test files.

1. Prepacked artifacts require exact FlashInfer/model/quantization/layout/preparation metadata plus contiguous FP8/uint8 tensor shapes. Missing shuffle markers, mismatched `MajorK`, wrong dimensions/dtypes/devices, and invalid expert offsets fail before kernel construction.
2. Explicit synthetic smoke quantizes each expert without scale swizzling, reorders both FC1 weights and scales for SwiGLU, then calls `shuffle_matrix_a` and `shuffle_matrix_sf_a` with `epilogue_tile_m=128` for FC1 and FC2 exactly as the pinned test does.
3. Forced contexts discard the expected `(flashinfer::trtllm_fp8_block_scale_moe, MoERunner)` log marker, insert one exact file key and tactic pair, require the fresh logged hit, verify the inserted pair was not changed, and restore cache and marker state in `finally`.
4. `MoePairResult` and all raw four-tensor interpretation now live in `flashinfer_adapter.py`; the shmoo only consumes typed `final_output` and `activated_intermediate` fields.
5. Type errors, unsupported intermediate calls, malformed returns, and all stock zero-LoRA preflight invocation failures normalize to `flashinfer_intermediate_api_unavailable`. Both stock references must be finite before candidate forcing.
6. Warmups must equal three; repetitions remain at least ten.

## Re-review Corrections

1. `--weights` and `--synthetic-smoke` remain mutually exclusive. A source-less invocation is accepted only when both `--profile-limit=1` and `--tactic-limit=2` are present, preserving the exact brief smoke command without opening broader runs to synthetic weights. Every explicit or implicit synthetic measurement row carries a top-level `"synthetic": true` provenance marker. Any other source-less shape exits before FlashInfer initialization or profile loading.
2. Candidate finiteness now covers the first FC1 intermediate, the repeated FC1 intermediate used for determinism, the first final FC2 output, and every captured final replay. A NaN in the repeated intermediate therefore fails the tactic row even when the first intermediate and all final outputs are finite.

## TDD Evidence

Initial RED:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

ERROR collecting both tests:
ModuleNotFoundError: experiments.mxfp8_moe_tactic_audit.flashinfer_adapter
```

Subsequent focused RED checks established missing behavior before implementation:

```text
ImportError: cannot import name 'cache_key_for_case'
```

```text
test_force_stock_tactic_inserts_literal_fallback_pair
Expected the exact MoERunner [-1, -1] file-config entry; observed an empty entry set.
```

During final verification, the graph-orchestration mock exposed an uninitialized `torch.empty` test output. The isolated test failed with `finite=False`; changing only the fixture to deterministic BF16 zeros made the same isolated test pass. No production behavior changed for that test repair.

Review-fix RED evidence:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py

ImportError: cannot import name 'PREPACKED_ARTIFACT_FORMAT'
```

After the adapter cycle, the shmoo cycle failed on the old warmup contract:

```text
Expected regex: warmups must equal 3
Actual message: warmups must be at least 3
```

Final GREEN:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

34 passed, 16 warnings in 10.21s
```

The warnings are pre-existing pytest temporary-directory cleanup warnings under macOS `/private/var/folders`; no Task 6 assertion warned or failed.

Re-review RED evidence for the exact brief invocation:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py \
  -k "exact_brief_smoke_args or repeated_intermediate_nan"

FAILED test_exact_brief_smoke_args_use_marked_bounded_synthetic_source
SystemExit: 2
shmoo_moe_tactics.py: error: one of the arguments --weights --synthetic-smoke is required
```

Re-review RED evidence for repeated-intermediate finiteness:

```text
.venv/bin/python -m pytest -q \
  'tests/experiments/test_mxfp8_moe_tactic_shmoo.py::test_profile_tactic_uses_paired_graph_replay_and_cold_l2_each_time[True]'

FAILED: assert result.finite is not repeated_intermediate_nan
1 failed in 9.78s
```

Focused GREEN:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py::test_exact_brief_smoke_args_use_marked_bounded_synthetic_source \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py::test_cli_rejects_broader_source_less_invocation \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py::test_profile_tactic_uses_paired_graph_replay_and_cold_l2_each_time

4 passed, 16 warnings in 10.48s
```

Re-review final GREEN:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

36 passed, 16 warnings in 6.67s
```

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_schema.py \
  tests/experiments/test_mxfp8_moe_tactic_profile_selection.py

41 passed, 16 warnings in 0.18s
```

Additional schema/profile regression coverage:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_schema.py \
  tests/experiments/test_mxfp8_moe_tactic_profile_selection.py

41 passed, 16 warnings in 0.40s
```

An exhaustive small routing-property check covered 3,529 feasible token/expert combinations and confirmed exact histograms, no duplicate expert per token, BF16 weights, and exact per-token BF16 sums:

```text
checked=3529
```

## Static Verification

```text
.venv/bin/ruff check \
  experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

All checks passed!
```

```text
.venv/bin/ruff format --check \
  experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

4 files already formatted
```

```text
.venv/bin/pyright \
  experiments/mxfp8_moe_tactic_audit/flashinfer_adapter.py \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py

0 errors, 0 warnings, 0 informations
```

The same Pyright command with `--pythonpath .venv/bin/python` also returned zero errors and warnings. The re-review Pyright command used that interpreter explicitly and returned `0 errors, 0 warnings, 0 informations`. `git diff --check` was clean before the re-review implementation commit. The CPU-local CLI help path exits successfully and exposes required `--profiles`, optional mutually exclusive `--weights`/`--synthetic-smoke`, profile/tactic limits, warmups, repetitions, device, and output; runtime validation enforces the narrow source-less compatibility rule.

```text
.venv/bin/python \
  experiments/mxfp8_moe_tactic_audit/shmoo_moe_tactics.py --help

usage: shmoo_moe_tactics.py [-h] --profiles PROFILES [--weights WEIGHTS |
                            --synthetic-smoke] [--profile-limit PROFILE_LIMIT]
                            [--tactic-limit TACTIC_LIMIT] [--warmups WARMUPS]
                            [--repetitions REPETITIONS] [--device DEVICE]
                            --output OUTPUT
```

## FlashInfer Sources Inspected

Inspected checkout:

`/Users/sna/MXFP8_generation/.worktrees/flashinfer-v0613-zero-weight-hybrid`

The checkout describes as `v0.6.13-6-g2879d391`; `git diff --stat v0.6.13` was empty for every relevant file below:

- `tests/moe/test_trtllm_gen_moe_autotune_tactics.py`
- `tests/moe/test_trtllm_gen_routed_fused_moe.py`
- `tests/moe/test_trtllm_gen_fused_moe.py`
- `tests/autotuner/test_trtllm_fused_moe_autotuner_integration.py`
- `flashinfer/fused_moe/core.py`
- `flashinfer/autotuner.py`
- `flashinfer/fused_moe/utils.py`
- `flashinfer/jit/fused_moe.py`
- `benchmarks/bench_trtllm_gen_fused_moe_autotuner.py`

## Private API Assumptions

All FlashInfer private-interface use is confined to `flashinfer_adapter.py`.

- Tactic enumeration uses `gen_trtllm_gen_fused_moe_sm100_module().build_and_load().trtllm_get_valid_moe_configs(...)` with `DtypeTrtllmGen.MxE4m3` for activations and weights, `Fp8QuantizationType.MxFp8`, shuffled weights, `WeightLayout.MajorK.value`, and the exact profile shape fields.
- The tuner singleton is `flashinfer.autotuner.AutoTuner.get()`. The audit process is assumed to serialize tactic contexts; the adapter snapshots and restores `_file_configs` and `profiling_cache` in `finally` and separately restores the prior presence of the one op/runner log marker it temporarily resets.
- The exact file key is the string form of `(custom_op, runner_name, profile_shapes, extras)` for `flashinfer::trtllm_fp8_block_scale_moe`, `MoERunner`, and empty extras. `profiling_cache` is cleared because FlashInfer checks it before `_file_configs`.
- The pinned file-config hit signal is `AutoTuner._logged_file_hits` containing `(custom_op, runner_name)`. Because that signal does not include tactic IDs, the adapter combines a fresh exact op/runner hit with verification that the sole inserted exact file key still maps to the requested two-ID tactic. A missing hit or changed pair raises `TacticDispatchError` and becomes a failed measurement row.
- Stock behavior is pinned with the literal paired fallback tactic `[-1, -1]` so bundled file configs cannot replace the reference.
- `MoEInputs._FIELDS` ordering is `output`, `routing_logits`, `topk_ids`, `expert_weights`, `hidden_states`, `hidden_states_scale`, `gemm1_lora_delta`, and `per_token_scale`; cache profile shapes follow this order and the upstream power-of-two bucket convention.
- `trtllm_fp8_block_scale_routed_moe` is the complete routed MXFP8 wrapper. For `do_finalize=False` with a non-`None` zero BF16 LoRA delta, `_unpack_trtllm_moe_output` returns four tensors and the fourth is the activated `gemm1_output` with `num_tokens * top_k * intermediate_size` BF16 elements. Any contract mismatch fails closed.
- Production weights are already prepared by the artifact producer and must declare `MajorK`, shuffled weights/scales, gated-row reorder, SwiGLU, MXFP8, FlashInfer 0.6.13, model revision, expert topology, and dimensions. The loader requires FP8 weight tensors and uint8 scale tensors in exact kernel shapes.
- Synthetic smoke mirrors pinned test lines 553-622: per-expert `mxfp8_quantize(..., False)`, `reorder_rows_for_gated_act_gemm` on FC1 weights and scales, and `shuffle_matrix_a`/`shuffle_matrix_sf_a` with tile 128 for both GEMMs. Hidden activations use `mxfp8_quantize(..., False)`.

## Deferred GPU Validation

This macOS worktree has no CUDA device and no locally installed FlashInfer binary, so no native TRTLLM Gen kernel, CUDA Graph, cold-L2 timing, or GB200 numerical result was claimed. The CLI and per-tactic row behavior are covered by CPU mocks; exact-version checks fail closed locally.

Task 11 must run the approved single-profile, two-tactic smoke on Ptyche/GB200 with FlashInfer 0.6.13, verify the live private return contract and cache keys, confirm graph replay and cold-L2 execution, and inspect successful or explicitly failed JSONL rows carrying `"synthetic": true` before any cache promotion.

Production Task 11 invocation must add `--weights <prepacked-qwen-mxfp8.pt>`. A broader layout-only smoke must add `--synthetic-smoke`. The exact brief command may omit both sources only with `--profile-limit=1 --tactic-limit=2`; `--weights` and `--synthetic-smoke` remain mutually exclusive.
