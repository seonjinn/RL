# Task 6 Report: Workload-Replayed FlashInfer Tactic Shmoo

## Outcome

Implemented and signed the Task 6 adapter, shmoo CLI, and CPU/mock-safe unit tests.

- Implementation commit: `cbdb93b51ae220e29b1d449a65b6e93837b382ed`
- Commit subject: `feat: shmoo workload-replayed MXFP8 MoE tactics`
- FlashInfer is rejected unless the installed `flashinfer-python` distribution is exactly `0.6.13`.
- Legal tactics are enumerated as paired FC1/GEMM1 and FC2/GEMM2 IDs through the TRTLLM Gen MoE API.
- Tactic forcing snapshots and restores only `AutoTuner._file_configs` and `AutoTuner.profiling_cache`, including exception paths.
- Routing reconstruction is deterministic from `signature_key`, preserves the exact expert histogram, rejects duplicate-expert requirements, and emits BF16 weights that sum exactly to one per token.
- Each measured tactic remains a pair. The timed CUDA Graph contains the complete routed MoE operation, with three or more warmups, ten or more repetitions, and a greater-than-L2 touch before every replay.
- Candidate FC1 activated intermediates and final FC2 reduced outputs are both checked against stock `[-1, -1]` references.
- A missing intermediate return contract emits `flashinfer_intermediate_api_unavailable`; ordinary tactic exceptions emit a serializable failure row and do not stop later tactics.

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

Final GREEN:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_flashinfer_adapter.py \
  tests/experiments/test_mxfp8_moe_tactic_shmoo.py

21 passed, 16 warnings in 9.14s
```

The warnings are pre-existing pytest temporary-directory cleanup warnings under macOS `/private/var/folders`; no Task 6 assertion warned or failed.

Additional schema/profile regression coverage:

```text
.venv/bin/python -m pytest -q \
  tests/experiments/test_mxfp8_moe_tactic_audit_schema.py \
  tests/experiments/test_mxfp8_moe_tactic_profile_selection.py

41 passed, 16 warnings in 0.47s
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

The same Pyright command with `--pythonpath .venv/bin/python` also returned zero errors and warnings. `git diff --cached --check` was clean before the implementation commit. The CPU-local CLI help path exits successfully and exposes `--profiles`, profile/tactic limits, warmups, repetitions, device, and output.

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
- The tuner singleton is `flashinfer.autotuner.AutoTuner.get()`. The audit process is assumed to serialize tactic contexts; the adapter snapshots dictionary contents and restores only `_file_configs` and `profiling_cache` in `finally`.
- The exact file key is the string form of `(custom_op, runner_name, profile_shapes, extras)` for `flashinfer::trtllm_fp8_block_scale_moe`, `MoERunner`, and empty extras. `profiling_cache` is cleared because FlashInfer checks it before `_file_configs`.
- Stock behavior is pinned with the literal paired fallback tactic `[-1, -1]` so bundled file configs cannot replace the reference.
- `MoEInputs._FIELDS` ordering is `output`, `routing_logits`, `topk_ids`, `expert_weights`, `hidden_states`, `hidden_states_scale`, `gemm1_lora_delta`, and `per_token_scale`; cache profile shapes follow this order and the upstream power-of-two bucket convention.
- `trtllm_fp8_block_scale_routed_moe` is the complete routed MXFP8 wrapper. For `do_finalize=False` with a non-`None` zero BF16 LoRA delta, `_unpack_trtllm_moe_output` returns four tensors and the fourth is the activated `gemm1_output` with `num_tokens * top_k * intermediate_size` BF16 elements. Any contract mismatch fails closed.
- Kernel inputs use FlashInfer `mxfp8_quantize`: hidden activations with `is_sf_swizzled_layout=False`, expert weights with `True`, and scales viewed as `uint8` in the upstream-tested layouts.

## Deferred GPU Validation

This macOS worktree has no CUDA device and no locally installed FlashInfer binary, so no native TRTLLM Gen kernel, CUDA Graph, cold-L2 timing, or GB200 numerical result was claimed. The CLI and per-tactic row behavior are covered by CPU mocks; exact-version checks fail closed locally.

Task 11 must run the approved single-profile, two-tactic smoke on Ptyche/GB200 with FlashInfer 0.6.13, verify the live private return contract and cache keys, confirm graph replay and cold-L2 execution, and inspect successful or explicitly failed JSONL rows before any cache promotion.
