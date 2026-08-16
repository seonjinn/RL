# Current MXFP8 MoE Rollout A/B

## Objective

Remeasure BF16 training with BF16 rollout against BF16 training with routed
expert-only MXFP8 rollout for Qwen3-30B-A3B and Nemotron3 Nano. Apply all
current refit and rollout performance optimizations to both arms.

## Pinned Source

- Branch: `sna/exp-current-bf16-mxfp8-superset-20260816`
- Commit: `c3696b257d47b151e201ed2e88acc10bdc6c1ecf`
- Both arms use this same source revision.
- The branch combines the current MXFP8 receiver path with the BF16 TRTLLM
  reload and two-gather expert conversion optimizations.

## Matched Conditions

- Hardware: GB200
- Topology: 8 nodes, 4 GPUs per node, 4 policy nodes and 4 rollout nodes
- Training precision: BF16
- Refit transport: NCCL Reshard
- MoE backend: FlashInfer TRTLLM
- CUDA Graphs: enabled (`enforce_eager=false`)
- Steps: 20
- Seed: 42
- MXFP8 scope: routed expert FC1/FC2 only
- BF16 scope: no rollout quantization

## Files Added For This Experiment

- `examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-bf16-rollout-nccl.yaml`
- `examples/configs/recipes/llm/performance/grpo-nanov3-30ba3b-8n4g-mxfp8-rollout-nccl.yaml`
- `experiments/current_mxfp8_moe_rollout_ab/README.md`
- `experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh`

## Verification So Far

- `bash -n experiments/current_mxfp8_moe_rollout_ab/submit_oci_hsg.sh`: pass
- Both Nano YAML files parse with PyYAML: pass
- `python3 -m py_compile nemo_rl/models/generation/vllm/vllm_backend.py`: pass
- `git diff --check`: pass
- Local pytest: unavailable because the locked environment supports Linux,
  while the local host is macOS. Run the focused unit suite in the pinned
  cluster container before performance submission.

## Remaining Steps

1. Create the pinned remote checkout and initialize exact submodule revisions.
2. Run focused Linux unit tests in the pinned container.
3. Run SLURM `--test-only` for all four experiment arms.
4. Submit and monitor the four 20-step jobs.
5. Compare steady steps for E2E, generation, policy, logprob, refit, throughput,
   reward, and `gen_kl_error`.
