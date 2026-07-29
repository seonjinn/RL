# OCI-HSG HybridEP GRPO launchers

`submit_grpo.sh` is a thin, reproducible wrapper around the repository's
`ray.sub`. Model shape and recipe selection live in `models/*.env`.

Run from a clean, pushed NeMo-RL checkout:

```bash
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
```

The default Qwen3-30B-A3B profile uses:

- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`
- 4 OCI-HSG GB200 nodes, 4 GPUs per node, and `--segment=4`
- HybridEP flex dispatch with 32 SMs
- the recipe's sequence packing and fused loss
- 20 GRPO steps
- current nightly image staged under the user NeMo-RL Lustre area

To run with a custom DeepEP wheel:

```bash
DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685 \
DEEPEP_WHEEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/deepep-wheels/f725d29699f5bda9ba789456bb9579af69844685-doca-sm100/deep_ep-1.2.1+f725d29-cp313-cp313-linux_aarch64.whl \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
```

To measure the fake-token padding introduced by the HybridEP sequence-packing
alignment, enable bounded rank-0 logging. The reduction reports group-wide raw,
padded, and added token counts; use this diagnostic run for padding analysis,
not timing comparison, because the logging path adds synchronization:

```bash
NEMO_RL_HYBRIDEP_LOG_PACKING=1 \
NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS=4096 \
NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS=0 \
NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE=1 \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
```

Set `DISPATCHER_MODE=recipe` to preserve the recipe's default dispatcher
without adding any HybridEP override. This is the accuracy and performance
control arm; the default remains `DISPATCHER_MODE=hybridep`.

To run the 32-node Nemotron3 Super 120B async 1-off recipe with the same
HybridEP overrides, the model profile also exports `NCCL_NVLS_ENABLE=0` to
match the official performance harness and avoid NVLS-related OOM:

```bash
WANDB_ENABLED=False \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-async-1off.env
```

The synchronous Super A/B uses two explicit recipes so the YAML file is the
only dispatcher-control surface. The baseline preserves the existing
recipe-native `alltoall` dispatcher:

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync.env
```

The paired HybridEP profile selects the inherited sync recipe that changes
only the dispatcher type, backend, and SM count:

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync-hybridep.env
```

## x86 H100 and B200

Megatron-Bridge supports HybridEP on Ampere, Hopper, and Blackwell. The x86
profiles provide matched all-to-all and HybridEP recipe pairs for Qwen3-30B,
Qwen3-235B, synchronous Nemotron3 Super, and DeepSeek-V3. The canonical
HybridEP recipes follow the upstream x86 performance defaults: 32 dispatcher
SMs, an eight-rank NVLink domain, non-MNNVL topology, and combine chunk size
128. Every profile sets `DISPATCHER_MODE=recipe`, so recipe selection is the
only dispatcher-control surface.

Build and validate an immutable `f725d296` wheel for the target GPU before
submitting. Set `TORCH_CUDA_ARCH_LIST=9.0` for H100 or `10.0` for B200 and
build with `HYBRID_EP_MULTINODE=1`. Export the resulting wheel and cluster
paths:

```bash
export DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
export DEEPEP_WHEEL=/absolute/shared/path/deep_ep-f725-x86_64.whl
export CONTAINER=/absolute/shared/path/nemo_rl_nightly.sqsh
export HF_HOME=/absolute/shared/path/hf_home
```

On CW-DFW, load the H100 hardware profile first. It supplies
`GPU_ARCH=9.0`, the eight-rank NVLink-domain topology, non-MNNVL mode, the
combine chunk size, and the default account/partition:

```bash
source scripts/experiments/x86/hybridep/clusters/cw-dfw-h100.env
```

These values mirror Megatron-Bridge's `h100` HybridEP performance path. The
SM90 wheel must still pass imports of `deep_ep`, `deep_ep_cpp`, and
`hybrid_ep_cpp` inside the exact NeMo-RL container before a training job is
considered valid.

If the nightly image and repository lock contain different Ray versions,
prepare one shared driver environment with
`scripts/experiments/x86/hybridep/submit_driver_venv.sh`, then use it for both
the Ray daemons and driver. Keep the environment and UV cache on `/lustre`:

```bash
export DRIVER_VENV=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/$USER/hybridep-x86/driver-venv
export RAY_VENV="${DRIVER_VENV}"
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/$USER/hybridep-x86/uv-cache

scripts/experiments/x86/hybridep/submit_driver_venv.sh
```

The x86 model profiles set `NRL_FORCE_REBUILD_VENVS=false`: this prebuild and
reuse workflow replaces per-actor `venvs.py` cache isolation for matched runs.
Do not submit either arm while actors can rebuild from source, and do not
override that setting to `true`; both arms must reuse the prepared driver/Ray
environment and shared cache. When multiple nodes populate one mounted UV
cache during the one preparation job, source builds are serialized by a
distribution lock. The launcher exports
`UV_LOCK_TIMEOUT=1800` by default so a second node can reuse the first node's
completed build instead of failing at UV's 300-second default. Override it
only with a positive integer number of seconds.

Use the following matched profiles after the shared runtime is prepared:

| Model | All-to-all profile | HybridEP profile |
| --- | --- | --- |
| Qwen3-30B-A3B | `qwen3-30ba3b-4n8g-x86.env` | `qwen3-30ba3b-4n8g-x86-hybridep.env` |
| Qwen3-235B | `qwen3-235b-16n8g-x86.env` | `qwen3-235b-16n8g-x86-hybridep.env` |
| Nemotron3 Super sync | `nemotron3-super-120ba12b-32n8g-sync-x86.env` | `nemotron3-super-120ba12b-32n8g-sync-x86-hybridep.env` |
| DeepSeek-V3 | `deepseek-v3-32n8g-x86.env` | `deepseek-v3-32n8g-x86-hybridep.env` |

The DeepSeek profiles are reusable, but there is no CW DeepSeek-V3 BF16
checkpoint configured currently. Do not submit either DeepSeek arm until
`NRL_DEEPSEEK_V3_BF16_CKPT` is set to a verified CW checkpoint path.

Run the two-node, three-step compatibility gate first:

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
NUM_ACTOR_NODES=2 \
SEGMENT_SIZE=2 \
MAX_STEPS=3 \
RUN_NAME=qwen3-30ba3b-2n8g-x86-alltoall-smoke \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86.env

DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
NUM_ACTOR_NODES=2 \
SEGMENT_SIZE=2 \
MAX_STEPS=3 \
RUN_NAME=qwen3-30ba3b-2n8g-x86-hybridep-smoke \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-hybridep.env
```

After both smoke jobs complete, run the four-node, 20-step performance pair:

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
RUN_NAME=qwen3-30ba3b-4n8g-x86-alltoall-20step \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86.env

DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
RUN_NAME=qwen3-30ba3b-4n8g-x86-hybridep-20step \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/qwen3-30ba3b-4n8g-x86-hybridep.env
```

The launcher selects the highest current user-level FairShare account. Set
`ACCOUNT` only to override that choice. Set `WANDB_ENABLED=True` to enable W&B;
the launcher requires `WANDB_API_KEY` in the environment and never writes its
value to metadata.

Each submission first runs `sbatch --test-only`. It then records the model
profile, all source commits, DeepEP selection, image SHA256, job ID, and log
paths under `exp_logs/hybridep/<model>/<run-name>/submission.env`.
