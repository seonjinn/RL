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
enabled in both MXFP8 modes and `force_on_policy_ratio=true`.

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

The generic transform-contract implementation was revalidated at commit
`fbe22cc3dcb10b9edf26cb4234341a9485cd22d9` on GCP-NRT. Job `488645`
completed both steps with a `27.84 GiB` MXFP8 reshard payload, `0.00436`
generation KL error, `1.033` token-mult probability error, and no NaN/Inf.
The run is available in [W&B](https://wandb.ai/nvidia/sna-pr3294-nccl-mxfp8-prequant/runs/dpsenun7).

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

To isolate the receiver-side PR 3294 optimizations after switching to the
transform-aware NCCL-Reshard transport, hold prequantization constant and run
the full shuffle/cache factorial ablation:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh \
MODES=mxfp8-nccl-prequant \
ARMS="baseline batched-shuffle loader-cache optimized" \
MAX_STEPS=20 \
ACTION=submit \
./experiments/nccl_reshard_pr3294/submit_prequant_ab.sh
```

| Arm | Batched MoE shuffle | Loader-route cache |
|---|---:|---:|
| `baseline` | Off | Off |
| `batched-shuffle` | On | Off |
| `loader-cache` | Off | On |
| `optimized` | On | On |

Trainer-side MXFP8 prequantization remains enabled in all four arms because it
is the required BF16-to-MXFP8 storage conversion for this transport. This
factorial design measures both individual effects and their interaction.
The wrapper defaults to the source-managed vLLM 0.25 environment used by the
validated runs; the container actor venv does not contain the required
`routed_experts` module for this source revision.

## Cumulative PR 3294 Ablation

The cumulative experiment adds one optimization at a time:

| Step | Transport | Trainer prequantization | Batched MoE shuffle | Loader-route cache |
|---|---|---:|---:|---:|
| Baseline | Legacy collective | Off | Off | Off |
| + Prequantization | Legacy collective | On | Off | Off |
| + Batched shuffle | Legacy collective | On | On | Off |
| + Loader cache | Legacy collective | On | On | On |
| + NCCL-Reshard | NCCL-Reshard | On | On | On |

The first four rows are the cumulative PR 3294 ablation. They retain the same
legacy collective transport so each delta isolates one optimization. The last
row measures the additional transport change. A no-prequantization
BF16-to-MXFP8 NCCL-Reshard row is intentionally absent: prequantization is the
cross-precision wire transform required by the current NCCL-Reshard contract,
so disabling it is rejected before launch.

Run the complete matrix with identical model, batch, topology, and training
settings:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-refresh/nemo_rl_nightly_20260730_483099.sqsh \
ACTION=test-only \
MAX_STEPS=20 \
./experiments/nccl_reshard_pr3294/submit_cumulative_ablation.sh
```

After the scheduling check succeeds, change `ACTION=submit`. For performance
reporting, compare steps 3-20 and include both incremental and cumulative
deltas from the baseline.
