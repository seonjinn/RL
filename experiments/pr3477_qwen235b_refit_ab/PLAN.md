# Qwen3-235B PR 3477 Refit A/B Plan

**Goal:** Measure the refit-time impact of NCCL-Reshard for BF16 training and
MoE-only MXFP8 rollout on Qwen3-235B-A22B.

**Architecture:** Run a matched 20-step GRPO pair from the same PR 3477 source
commit, container, model, data, seed, and 128-GPU topology. The control uses the
legacy non-colocated collective synchronizer (`refit_transport=null`); the
treatment uses `refit_transport=nccl_reshard`.

**Tech stack:** NeMo-RL, Megatron-Core BF16 training, vLLM MXFP8 generation,
Ray, SLURM, GCP-NRT B200, W&B.

## Fixed Setup

- Cluster: GCP-NRT B200, 8 GPUs per node
- Allocation: 16 nodes, 128 GPUs total
- Split: 8 trainer nodes and 8 generation nodes
- Scheduling: full-node exclusive allocation
- vLLM worker start method: `spawn` to avoid forking imported CUTLASS/MLIR state
- Recipe: `grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml`
- Parallelism: trainer TP2/PP4/CP2/EP16; generation TP4/PP1/EP1, DP16
- GRPO: 16 prompts, 32 generations per prompt, GBS 512, seed 42
- Evaluation window: steps 3-20 after warmup
- Checkpointing: disabled
- Validation: recipe-default periodic validation at steps 10 and 20
- W&B project: `sna-pr3477-qwen235b-refit-ab`

## Execution

1. Run `bash -n` locally, then run the script's `dry-run` action on the
   GCP-NRT login node where the shared runtime artifacts are mounted.
2. Commit and push the experiment branch to `seonjinn/RL`.
3. On GCP-NRT, pull the exact branch and verify container, model cache, and
   submodule availability.
4. Run both arms with `ACTION=test-only` to validate SLURM scheduling.
5. Submit both arms with one shared `RUN_SUFFIX`.
6. Monitor SLURM and logs for at least five minutes after allocation.
7. After completion, compare `transfer/update`, total refit, generation,
   logprob, policy training, E2E step time, and tokens/s/GPU over steps 3-20.

## Commands

```bash
RUN_SUFFIX=qwen235b-$(date +%Y%m%d-%H%M%S)

MODE=legacy ACTION=test-only RUN_SUFFIX=${RUN_SUFFIX} \
  bash experiments/pr3477_qwen235b_refit_ab/submit_gcp_nrt.sh
MODE=nccl ACTION=test-only RUN_SUFFIX=${RUN_SUFFIX} \
  bash experiments/pr3477_qwen235b_refit_ab/submit_gcp_nrt.sh

MODE=legacy ACTION=submit RUN_SUFFIX=${RUN_SUFFIX} \
  bash experiments/pr3477_qwen235b_refit_ab/submit_gcp_nrt.sh
MODE=nccl ACTION=submit RUN_SUFFIX=${RUN_SUFFIX} \
  bash experiments/pr3477_qwen235b_refit_ab/submit_gcp_nrt.sh
```
