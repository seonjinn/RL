# Task 1 Report: Qwen3-235B MXFP8 Generation-Only Trace Gate

## Status

Implemented the isolated Qwen3-235B generation-only trace gate. No job was
submitted.

## Artifacts

- `experiments/mxfp8_adaptive_rollout_v0251/configs/eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml`
  - Uses `Qwen/Qwen3-235B-A22B` with eight GPUs on two Ptyche nodes.
  - Configures TP4, PP1, and EP4 per engine without an explicit replica count,
    preserving evaluator-owned external replica derivation.
  - Keeps CUDA Graphs enabled with `enforce_eager: false`.
  - Keeps `q_proj`, `k_proj`, `v_proj`, `o_proj`, and `.mlp.gate` out of
    MXFP8 quantization. The gate exclusion retains router logits in BF16.
  - Reuses the qualified 32K generation-only shape-trace workload and trace
    environment contract.
- `experiments/mxfp8_adaptive_rollout_v0251/submit_qwen235_32k_trace_ptyche.sh`
  - Uses a Qwen235-specific result root and SLURM job name.
  - Preserves clean NeMo-RL and vLLM worktree checks, the pinned vLLM revision,
    `git pull --ff-only`, and runtime-cache derivation.
  - Requests two nodes, four GPUs per node, segment 2, and five hours with an
    empty dependency.
- `tests/unit/experiments/test_mxfp8_qwen235_trace_submission.py`
  - Statically verifies the model, topology, CUDA Graph mode, quantization
    exclusions, trace environment, provenance checks, and scheduler contract.

## Validation

- YAML parsing: `python3 -c 'from pathlib import Path; import yaml; yaml.safe_load(Path("experiments/mxfp8_adaptive_rollout_v0251/configs/eval_qwen3_235ba22b_32k_cuda_graph_trace.yaml").read_text(encoding="utf-8"))'` passed.
- Shell syntax: `bash -n experiments/mxfp8_adaptive_rollout_v0251/submit_qwen235_32k_trace_ptyche.sh` passed.
- Focused static tests: `pytest --confcutdir=tests/unit/experiments tests/unit/experiments/test_mxfp8_qwen235_trace_submission.py -q` passed, 2 tests.
- Whitespace validation: `git diff --check` passed.

## Environment Note

`uv run pytest` cannot resolve this worktree because its `nemo-gym` source is
declared as a workspace dependency but is not a workspace member. The ambient
pytest collection also lacks Ray through the repository-wide `tests/unit`
conftest. The focused tests are dependency-free static checks and were run with
`--confcutdir=tests/unit/experiments` to avoid that unrelated fixture.
