# Files

## 2026-08-05 Campaign Refresh

### Inspected

- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/submit_performance_matrix.sh` - existing repeated matrix orchestration.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/submit_oci_nano_direct.sh` - exact OCI Nano performance command and scheduler contract used by recent successful jobs.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scopes/*.sh` - persistent leaves covering every valid four-axis scope subset.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/results/persistent_bank_scope_sweep_*.csv` - prior performance, cache, coverage, and correctness evidence.

### Changed

- `session/20260802_122126/session_state.md` - refreshed the active objective, verified status, and launch plan.
- `session/20260802_122126/timeline.md` - recorded the Nano four-axis campaign decision.
- `session/20260802_122126/files.md` - recorded newly inspected campaign artifacts.
- `session/20260802_122126/handoff.md` - replaced stale Qwen-only resume instructions with the Nano matrix handoff.

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
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/{profile_snapshot.py,validate_campaign_gate.py,verify_runtime_attestation.py}` - no-exec profile parsing and content-bound gate/runtime validation.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/{run_scope.sh,submit_qwen_router_validation.sh,ray.sub,virtual_cluster.py}` - fail-closed launch validation, exact submission metadata, and Router Replay execution handoff.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/{run_r3_validated_command.py,run_r3_validated_command.sh}` - content-bound R3 driver/checker execution and atomic terminal records.
- `experiments/cuda_graph/nemotron_thd_te_graph_20260731/{export_tensorboard.py,export_wandb.py}` - shared strict identity/provenance validation and complete per-step metric export.
- `nemo_rl/models/megatron/cuda_graph_lifecycle.py` - exact CUDA Graph step metric schema and counter-order validation.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py` - source lookup miss accounting for warming and capture outcomes.
- `nemo_rl/algorithms/utils.py` - exact distributed aggregation/logging schema for cache misses.
- `tests/unit/experiments/{test_matrix_submitters.py,test_nemotron_thd_te_graph_launchers.py,test_export_tensorboard.py,test_export_wandb.py,test_collect_results.py,test_render_report.py,test_nemotron_thd_te_graph_reporting.py}` - campaign, exporter, collector, and reporting regressions.
- `tests/unit/models/megatron/test_cuda_graph_lifecycle.py` and `tests/unit/models/policy/{test_megatron_cuda_graph_worker.py,test_cuda_graph_policy_packing.py}` - lifecycle and worker miss semantics.
- `tests/unit/algorithms/{test_cuda_graph_metrics.py,test_grpo.py}` and `tests/unit/single_controller/test_sc_utils_helpers.py` - aggregation and logger schema coverage.
- `.superpowers/sdd/2026-08-02-qwen-campaign-review-remediation/task-{1,2,3,4}-report.md` - review-remediation evidence ledgers.

## Generated

- `session/20260802_122126/session_state.md` - durable campaign state.
- `session/20260802_122126/timeline.md` - append-only decision log.
- `session/20260802_122126/files.md` - artifact inventory.
- `session/20260802_122126/handoff.md` - resume instructions.
