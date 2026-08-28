Status: DONE_WITH_CONCERNS

Files changed:
- `nemo_rl/weight_sync/nccl_reshard_utils.py`
- `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- `.superpowers/sdd/2026-08-28-deepseek-custom-pp-nccl-reshard/task-1-report.md`

Commit SHA: PENDING

Implementation summary:
- Added the local `PipelineLayerLayout` protocol for runtime custom PP layouts.
- Implemented `build_layer_to_pp_stage_from_custom_layout(...)` as a keyword-only helper that validates PP size, VPP size, integer layer IDs, out-of-range IDs, duplicates, and exact coverage before emitting canonical HF keys.
- Added DeepSeek PP8 mapping coverage and malformed-layout rejection tests in `tests/unit/weight_sync/test_nccl_reshard_utils.py`.

Tests and exact outcomes:
- `uv run --no-sync pytest tests/unit/weight_sync/test_nccl_reshard_utils.py -k custom_layout -q`
  - Failed before collection with `ModuleNotFoundError: No module named 'ray'` from `tests/unit/conftest.py:24`.
- `uv run --no-sync pytest tests/unit/weight_sync/test_nccl_reshard_utils.py -q`
  - Failed before collection with `ModuleNotFoundError: No module named 'ray'` from `tests/unit/conftest.py:24`.
- `uv run --no-sync python -m py_compile nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`
  - Passed.
- `ruff check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`
  - Passed.
- `ruff format --check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`
  - Initially reported both files would be reformatted.
- `ruff format nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`
  - Reformatted both files.
- `ruff format --check nemo_rl/weight_sync/nccl_reshard_utils.py tests/unit/weight_sync/test_nccl_reshard_utils.py`
  - Passed after formatting.
- `git diff --check`
  - Passed.
- `uv run --no-sync python - <<'PY' ... PY`
  - Passed a dependency-free direct-file harness that stubbed `torch` imports, loaded `nemo_rl/weight_sync/nccl_reshard_utils.py`, verified the DeepSeek PP8 mapping, and verified the malformed-layout `ValueError` substrings.

Self-review findings:
- No actionable findings in the final diff.

Remaining concerns:
- The canonical pytest checks are environment-blocked on this macOS host because the repo test environment is missing `ray`, and the shared unit `conftest.py` imports it before collection.
- I could not run the Linux-container path requested in the brief from this host, so the new pytest cases remain unexecuted in their intended environment.
