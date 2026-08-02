# Files

## Inspected

- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md` - experiment gates, current Nano result, supported models, and reporting workflow.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/run_scope.sh` - model selector, dry-run, provenance, and SLURM submission contract.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_30ba3b.env` - existing 4n4g Qwen3-30B-A3B selector.
- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml` - current 30B performance topology.
- `examples/configs/recipes/llm/grpo-qwen3-30ba3b-8n8g-megatron-cp2-r3.yaml` - official Router Replay recipe.
- `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml` - 235B-A22B performance topology.

## Changed

- `docs/superpowers/specs/2026-08-02-qwen-moe-router-cuda-graph-validation-design.md` - approved staged experiment design, safety boundary, correctness gates, and stop rules; committed as `d94ccc8d8`.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/models/qwen3_235b.env` - Qwen3-235B-A22B 16n4g NeMo-RL selector.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/conditions/qwen_{A,B,C,E}_*.sh` - persistent safe A/B/C/E launch conditions.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/{run_scope.sh,scope_matrix.py,collect_results.py,render_report.py}` - R3/router safety validation and R3-aware result identity/reporting.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/README.md` - Qwen campaign commands, fail-closed boundary, result checks, and OCI launch readiness.

## Generated

- `session/20260802_122126/session_state.md` - durable campaign state.
- `session/20260802_122126/timeline.md` - append-only decision log.
- `session/20260802_122126/files.md` - artifact inventory.
- `session/20260802_122126/handoff.md` - resume instructions.
