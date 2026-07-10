# Task 4 Tail-Gate Metrics and W&B Derivations

## Status

Implemented Task 4 on `sna/nemorl-vllm024-tail-gating`.

## Changed Files

- `nemo_rl/models/generation/vllm/vllm_backend.py`
  - Merges cumulative model-runner `_nrl_tail_gate_metrics` into the existing
    worker metric RPC without changing CUDA-graph metric names or aggregation.
  - Rejects list-valued tail-gate counters before they can conflict with scalar
    worker metrics.
- `nemo_rl/models/generation/vllm/utils.py`
  - Deltas tail-gate cumulative counters before deriving enabled and
    advance-only ratios, activation batch and sequence-length means, predicted
    speedup mean, and any reported K-histogram ratios.
  - Uses `0.0` for every zero denominator.
- `tests/unit/models/generation/test_vllm_generation.py`
  - Covers worker RPC merging, unchanged CUDA-graph counters, step-metric
    delivery, and list/scalar rejection.
- `tests/unit/models/generation/test_vllm_utils.py`
  - Covers cumulative-snapshot derivations, K-histogram ratios, and zero
    denominator behavior.

## TDD Evidence

### RED

```bash
uv run --no-sync pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py -k 'tail_gate' -q
```

Result: failed as expected before implementation at
`test_tail_gate_worker_counters_are_reported_and_derived` with
`KeyError: 'vllm:spec_decode_tail_gate_decisions'`.

### GREEN

```bash
uv run --no-sync pytest --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py -k 'tail_gate' -q
```

Result: `13 passed, 156 deselected`.

```bash
uv run --no-sync ruff check \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/models/generation/vllm/utils.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py
```

Result: Ruff reported `All checks passed!`.

## Requested Test Command

```bash
uv run --no-sync pytest \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py -q
```

The command was attempted. The root unit-test autouse Ray fixture packages the
worktree, creates a separate `.venv`, then its raylet fails with
`ModuleNotFoundError: No module named 'ray'` before pytest emits a final test
summary. The focused GREEN command above avoids that broken remote runtime and
exercises all Task 4 cases.

## Self-Review

- CUDA-graph counters remain on their existing names and aggregation path.
- Tail-gate values are treated only as cumulative sums/counts and are deltaed
  before every ratio or mean.
- Existing SpecDec aggregation remains unchanged.
- Tail-gate K metrics are emitted only from raw `tail_gate_k_*_steps` counters;
  no K distribution is inferred from instantaneous state.
- No sampling or training behavior changed.

## Concerns

- The required full pytest command cannot provide a reliable result on this
  macOS worktree because its Ray runtime environment omits `ray`.
- A full Pyright invocation still reports pre-existing typing errors in the
  broad vLLM utility/backend/test files. The new `model_runner` declaration and
  counter-test annotations avoid adding a distinct typing issue.
- Concurrent edits to `patches.py`, `tail_gate_scheduler.py`, launcher files,
  and `tests/unit/unit_results` remain outside this task and are not staged.

## Review Follow-Up

### Changes

- `compute_spec_decode_metrics()` now derives activation means only from the
  cumulative `tail_gate_activation_batch_sum` and
  `tail_gate_activation_sequence_length_sum` counters.
- Producer-shaped snapshots include four decision-level observations and one
  activation, proving the activation metrics remain `16` and `8192` rather
  than using the decision-level totals.
- K histogram tests use the exact cumulative
  `tail_gate_k_<effective_k>_steps` producer names and verify both counts and
  ratios.
- `test_vllm_generation.py` no longer installs a process-global fake `vllm`.
  Extension tests use a function-scoped module with a valid `ModuleSpec` and
  remove the imported backend module during fixture cleanup.

### Review RED

```bash
uv run --no-sync pytest --noconftest -o addopts='' \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py \
  -k 'tail_gate_worker or derives_tail_gate or tail_gate_zero' -q
```

Result: `1 failed, 3 passed`. The producer-shaped snapshot returned activation
batch `80.0` instead of `16.0`, proving the consumer used the every-decision
sum.

### Review GREEN

The same focused command passed: `4 passed, 165 deselected`.

Both collection orders below passed with `7 passed, 200 deselected`:

```bash
uv run --no-sync pytest --noconftest -o addopts='' \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_utils.py \
  -k 'tail_gate_worker or get_cudagraph_dispatch_metrics or derives_tail_gate or tail_gate_zero' -q

uv run --no-sync pytest --noconftest -o addopts='' \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py \
  -k 'tail_gate_worker or get_cudagraph_dispatch_metrics or derives_tail_gate or tail_gate_zero' -q
```

Ruff, Python compilation, and scoped `git diff --check` passed. Pyrefly is not
installed in the local environment. The Pyright fallback still reports the
files' existing type-check debt (`336 errors`), but no new activation-consumer
or fixture-loader diagnostics remain.
