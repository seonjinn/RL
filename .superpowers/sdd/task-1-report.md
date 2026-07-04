# Task 1 Implementation Report

## Status

Implemented the declarative NeMo-RL Math continuation launcher and its focused
tests. No jobs were submitted and no remote connection was attempted.

## Files

- `experiments/eagle3_online/submit_lyris_nemorl_math_continuation_20260704.sh`
- `tests/test_submit_lyris_nemorl_math_continuation.py`
- `.superpowers/sdd/task-1-report.md`

## Implementation

- Renders the 12 supported model/mode/method commands locally with `DRY_RUN=true`.
- Accepts comma- or whitespace-separated `MODELS`, `MODES`, and `METHODS` filters.
- Rejects unknown selections and selections with no supported matrix rows locally.
- Pins the remote repository HEAD to
  `1271b1530181a7378e40de40b4b46ad223e6596c`.
- Validates only the configs, target snapshots, draft snapshots, and Suffix site
  required by selected rows.
- Computes the selected container's SHA-256 remotely before submission and records
  it in every CSV manifest row.
- Propagates shared `HF_HOME` and `HF_DATASETS_CACHE` through `ray.sub` and the
  rendered training command.
- Configures Suffix K32 with the pinned Arctic site, tree depth 24, cache size
  10000, spec factor 1.0, and minimum token probability 0.1.
- Includes an explicit Qwen3-235B Triton-attention baseline and applies
  `fuse_allreduce_rms=false` to every Qwen3-235B method.
- Uses model-specific networking: Qwen3-30B-A3B and Qwen3-32B render SHARP;
  Qwen3-235B never does.
- Builds commands from the declared matrix and does not inspect, recover, or
  evaluate historical commands or logs.

## TDD And Verification

1. `python3 -m pytest -q tests/test_submit_lyris_nemorl_math_continuation.py`
   before implementation: `7 failed`; every failure was caused by the absent
   launcher.
2. `bash -n experiments/eagle3_online/submit_lyris_nemorl_math_continuation_20260704.sh`
   and the focused pytest module after implementation: `7 passed in 0.86s`.
3. `shellcheck experiments/eagle3_online/submit_lyris_nemorl_math_continuation_20260704.sh`:
   not run because `shellcheck` is not installed in this environment.
4. `ruff check tests/test_submit_lyris_nemorl_math_continuation.py`: passed with
   no findings.
5. Existing focused baseline
   `python3 -m pytest -q tests/test_submit_lyris_qwen235b_eagle3_k_sweep.py`:
   `4 passed in 0.15s`.
6. Plain `python3 -m pytest -q` did not collect because the pre-existing
   `tests/test_build_latest_specdec_html_pages.py` import path cannot resolve
   `vllm024_dflare_report`. A retry with `PYTHONPATH=scripts` was interrupted by
   the user before a result and was stopped.

## Concerns

- Per instruction, remote preflight was not run. The default container checksum
  value is therefore not known locally; the launcher computes and records it
  before any future submission.
- `shellcheck` was unavailable; Bash syntax validation passed.
- The unrelated full-suite import-path collection failure remains unchanged.
