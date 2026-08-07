# Task 4 Report

## Delivered

- Added frozen, typed schemas for routing traces, tactic pairs, tactic
  measurements, and replay profiles.
- Canonical routing signature keys use SHA256 over sorted, ASCII JSON and omit
  `sampled_gpu_time_us`.
- Added strict known-field parsing, JSON round trips, and validation for the
  trace invariants in the task brief.

## TDD Evidence

- RED: `PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py`
  failed during collection with `ModuleNotFoundError: No module named
  'experiments.mxfp8_moe_tactic_audit.schema'` before `schema.py` existed.
- GREEN: the same targeted test command passed with `22 passed` after the
  implementation.

## Verification

```text
PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
.venv/bin/ruff check experiments/mxfp8_moe_tactic_audit/schema.py tests/experiments/test_mxfp8_moe_tactic_audit_schema.py
.venv/bin/pyright experiments/mxfp8_moe_tactic_audit/schema.py
```

All commands completed successfully: 22 tests passed, Ruff reported no
findings, and Pyright reported 0 errors, warnings, or information messages.

## Follow-up Review Fix

- Normalized direct `RoutingSignature` construction to store
  `expert_counts` as a tuple before validation, so frozen instances cannot
  retain caller-owned mutable lists.
- Added a regression test covering list normalization, source-list mutation,
  tuple immutability, and exact JSON round trips.
- Added a named Task 3 JSONL fixture and integration-style assertion that the
  complete Task 3 metadata, shape, timing, and histogram field set is accepted.
- RED: the new regression test failed because direct construction stored the
  caller-provided list.
- GREEN: `PYTHONPATH="$PWD" .venv/bin/pytest -q tests/experiments/test_mxfp8_moe_tactic_audit_schema.py`
  passed with 24 tests after the fix.

## Environment Note

The requested `uv run pytest ...` invocation could not resolve the repository
workspace because `nemo-gym` is declared as a workspace source but is not a
workspace member. Verification therefore used the worktree `.venv` directly;
`PYTHONPATH="$PWD"` exposes the uninstalled `experiments` namespace.

## Producer-Derived Task 3 Fixture

`tests/fixtures/mxfp8_moe_tactic_audit/task3-routing-signature.jsonl` was
generated from the Task 3 vLLM worktree at commit
`ba437b2c81aa0253e81571c53299cb3f55458d2a`, not authored as a local schema
fixture. The broader vLLM package was not imported; the producer module was
loaded directly so its `record_routing_signature` function and
`MoeTraceMetadata` dataclass generated the JSONL contract.

Exact generation command:

```bash
trace_dir=$(mktemp -d)
export VLLM_TRACE_MODULE=/Users/sna/MXFP8_generation/.worktrees/vllm-v0251-moe-tactic-audit/vllm/model_executor/layers/fused_moe/experts/trtllm_moe_trace.py
VLLM_MXFP8_MOE_TRACE_DIR="$trace_dir" /Users/sna/MXFP8_generation/.worktrees/vllm-v0251-moe-tactic-audit/.venv/bin/python - <<'PY'
import importlib.util
import os
from pathlib import Path

import torch

module_path = Path(os.environ["VLLM_TRACE_MODULE"])
spec = importlib.util.spec_from_file_location("task3_trace", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load Task 3 trace module")
trace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_module)
metadata = trace_module.MoeTraceMetadata(
    schema_version=1,
    model_revision="qwen3-30ba3b-test",
    layer_family="routed_experts",
    global_num_experts=4,
    local_num_experts=4,
    top_k=2,
    hidden_size=2048,
    intermediate_size=768,
    tp_size=1,
    ep_size=1,
    dp_size=16,
    cuda_graph_state="trace-eager",
    weight_layout="MajorK",
    quantization="MXFP8",
    runtime_fingerprint="runtime-sha256",
)
trace_module.record_routing_signature(
    torch.tensor([[0, 1], [1, 2]], dtype=torch.int16, device="cpu"),
    metadata,
    sampled_gpu_time_us=17.5,
)
PY
```

The schema integration test reads that one-line fixture and verifies that
`RoutingSignature.from_json` accepts it and `to_json` returns the exact row.
