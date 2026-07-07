# Task 5: Event-Scoped SFT Validation

## Scope

Implemented only Task 5 on branch `sna/sft-validation-event-batch-20260706`,
starting from `69c0a7110eb7d71fd57f8eca179f1a51e521c2c2`.

- Added `sft.validation_execution_mode` with accepted values `per_batch` and
  `event_batch`; the default remains `per_batch`.
- Kept the legacy validation call path and its `gbs=val_data.size` behavior.
- Added an event path that collects four already-prepared validation batches,
  validates them, combines them in order with `BatchedDataDict.from_batches`,
  and calls `Policy.train` once with `eval_mode=True`, `gbs=64`, and `mbs=1`.
- Aligned `sft_superv3_prepacked.yaml` to `val_period=20`, `val_batches=4`,
  validation GBS 64, and validation MBS 1.
- Did not change policy workers, evaluation fast paths, timers, or performance
  instrumentation.

## Correctness

The combined event has 256 rows and is submitted with GBS 64, so policy
workers derive exactly four global batches. Every source batch must contain 64
rows after the existing validation padding step. The combiner rejects wrong
batch counts, row counts, key sets, tensor dtype/device/rank/shape mismatches,
inconsistent `PackedTensor` dimensions, and malformed packed sequence
metadata.

Validation loss is reduced in original batch order using each prepared batch's
valid-token count. Megatron divides each per-global-batch loss by
`num_global_batches` inside one `train` call, so the event path reverses that
factor before applying the unchanged token-weighted reduction. DTensor losses
are not rescaled. Zero-token batches are excluded, and an all-zero event keeps
the legacy zero loss and warning behavior.

For sequence length 262,144, the combined primary packed tensors occupy at
least 1,879,051,264 bytes (about 1.75 GiB): three int64 token/position tensors,
one float32 token mask, input lengths, and sample mask. Packed sequence
metadata and Python lists add to this lower bound. The small-node smoke should
record the actual Ray payload and peak host/object-store memory.

## TDD Evidence

Tests were added before production changes for:

- one event-scoped policy call, original row order, and exact call arguments;
- four-global-batch worker sizing;
- token-weighted loss equivalence, including one zero-mask padded row;
- all-zero-valid-token behavior;
- wrong event count, inconsistent packed metadata, and wrong row count;
- unchanged four-call legacy behavior;
- config default and invalid enum value;
- `BatchedDataDict.from_batches` packed order and `-1` metadata padding.

RED was observed with the source contract failing because
`validation_execution_mode` did not exist. Repository pytest collection could
not reach the new tests because the host environment lacks Ray.

## Verification

Passed locally:

- `python3 -m py_compile nemo_rl/algorithms/sft.py tests/unit/algorithms/test_sft.py tests/unit/data/test_megatron_sft_packed.py`
- Task 5 AST/source contract check
- isolated Megatron/DTensor event-loss normalization check
- YAML parsing for all three changed config files
- `ruff check` on changed Python files
- `ruff format --check` on changed Python files
- `git diff --check`

Environment-blocked:

- `uv run pytest ...`: the lockfile supports Linux x86_64/aarch64 only, not
  this macOS host.
- `python3 -m pytest ...`: `tests/unit/conftest.py` fails to import missing
  `ray`; host Python also lacks `torch`.
- `pyright nemo_rl/algorithms/sft.py`: host dependency resolution is missing
  `torch`, `pydantic`, `torchdata`, and `transformers`; it reports 24 errors,
  including pre-existing diagnostics in this file, before a configured Linux
  environment can provide a meaningful result.

## Remaining Validation

Run the focused pytest and Pyright commands from the approved plan in the
Linux NeMo-RL environment. The smoke must confirm the four-element worker loss
vector, exact loss equality against `per_batch`, and acceptable host/Ray memory
for the approximately 1.75 GiB combined packed payload.
