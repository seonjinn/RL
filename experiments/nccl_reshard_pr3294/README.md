# BF16 to MXFP8 NCCL-Reshard A/B

This experiment measures trainer-side MXFP8 prequantization over NCCL-Reshard
on the same source commit, container, model, topology, batch, and seed.

## Comparison

| Mode | Trainer storage | Rollout storage | Transport |
|---|---|---|---|
| `bf16` | BF16 | BF16 | NCCL-Reshard |
| `mxfp8-rollout` | BF16 | MXFP8 | Legacy collective |
| `mxfp8-nccl-prequant` | BF16 | MXFP8 | NCCL-Reshard value + E8M0 scale pair |

Each mode runs as an independent SLURM job so Ray and actor lifecycles cannot
leak between comparison arms. The first functional target is GCP-NRT B200:
4 nodes x 8 GPUs, split into 2 training and 2 generation nodes.
Qwen3-30B-A3B uses trainer EP16 and vLLM TP1. Q/K/V/O projections stay BF16;
MXFP8 applies to eligible MoE weights. Importance-sampling correction is
disabled in all modes and `force_on_policy_ratio=true`.

The isolated transport comparison is `mxfp8-rollout` versus
`mxfp8-nccl-prequant`; both use the same MXFP8 recipe and generation backend.
`bf16` is an end-to-end BF16 reference and may use a different MoE backend.

Use `MAX_STEPS=5` for functional smoke tests and `MAX_STEPS=20` for the reported
A/B. Report `transfer_and_update_weights`, total refit, generation, E2E step
time, and logged tokens/s/GPU over steps 3-20.

Before reporting performance, run a two-step correctness gate. Step 2 verifies
that weights changed by the first optimizer update are refit correctly. This
gate deliberately computes previous-policy logprobs and enables the runtime
batched-shuffle verifier:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh \
ACTION=submit \
MODES=mxfp8-nccl-prequant \
MAX_STEPS=2 \
NUM_PROMPTS_PER_STEP=4 \
NUM_GENERATIONS_PER_PROMPT=4 \
TRAIN_GLOBAL_BATCH_SIZE=16 \
MAX_TOTAL_SEQUENCE_LENGTH=512 \
FORCE_ON_POLICY_RATIO=false \
USE_IMPORTANCE_SAMPLING_CORRECTION=true \
REFERENCE_POLICY_KL_PENALTY=0 \
SKIP_REFERENCE_POLICY_LOGPROBS=true \
MXFP8_SHUFFLE_VERIFY=1 \
./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```

Require two completed training steps, real NCCL-Reshard selection, non-zero
NCCL MXFP8 payload, no NaN/Inf, `train/gen_kl_error < 0.05`, and
`train/token_mult_prob_error < 2.0`. A Python transport fallback is sufficient
for functional debugging but not for a reportable performance result.

With 2 trainer nodes x 8 GPUs and TP1/PP1/CP1, trainer data parallelism is 16;
the correctness batch must therefore be a multiple of 16.

If source-managed actor environments are not already populated, build them in
a CPU-only job before allocating GPUs. This avoids idle-GPU reaper cancellation
during first-time TransformerEngine and vLLM dependency builds:

```bash
sbatch \
  --account=coreai_chef_posttrain \
  --partition=cpu \
  --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr3294-nccl-mxfp8-prequant/gcp-b200/slurm/%x-%j.out \
  --export=ALL,REPO=${PWD},EXPECTED_REPO_SHA=$(git rev-parse HEAD),CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh,CACHE_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/mopd_nano_fast/.cache/nccl-reshard-pr3294/v2-vllm025-py31313,SHARED_UV_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/mopd_nano_fast/.cache/nccl-reshard-pr3294/v2-vllm025-shared/uv,RAY_BOOTSTRAP_ARCHIVE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/mopd_nano_fast/.cache/nccl-reshard-pr3294/bootstrap/ray-2.56.1-py31313.tar.gz,BUILD_TARGETS='mcore vllm' \
  experiments/nccl_reshard_pr3294/build_actor_venvs.sbatch
```

For the BF16 versus MXFP8 NCCL prequantization A/B on GCP-NRT:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh \
ACTION=test-only \
MAX_STEPS=5 \
./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```
