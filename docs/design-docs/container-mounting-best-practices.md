# Container Mounting Best Practices for NeMo-RL

The container image contains pre-compiled, highly-optimized virtual environments and submodules (e.g., custom `vllm`, `Megatron-LM`, `Megatron-Bridge`, `Gym`).

Over-mounting the entire parent directory shadows these pre-compiled submodules with uncompiled host folders, forcing slow host-side compilation (`tools/build-custom-vllm.sh`). The best practice is to use **Selective Overlay Mounts**.

---

## 1. Quick Decision Matrix

| Submodule | If NOT Editing (Default) | If Actively Modifying Code |
| :--- | :--- | :--- |
| **`nemo_rl`** | **Mount** (`${NEMORL}/nemo_rl:/opt/nemo-rl/nemo_rl`) | **Mount** (`${NEMORL}/nemo_rl:/opt/nemo-rl/nemo_rl`) |
| **`vllm`** | **Do Not Mount** (Uses container prebuilt) | **Mount** (e.g. `${NEMORL}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm`) + Run host build script first |
| **`Gym`** / **Megatron** | **Do Not Mount** (Uses container prebuilt) | **Mount** their specific path (e.g. `${NEMORL}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym`) |

---

## 2. Best-Practice Skeleton Launch Script

```bash
#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

# 1. Define workspace root on host
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# 2. SELECTIVE MOUNTS: Mount only what you are editing
# Leaving 3rdparty folders unmounted preserves pristine pre-compiled submodules in the container.
export MOUNTS="/lustre:/lustre,\
${NEMORL}/nemo_rl:/opt/nemo-rl/nemo_rl,\
${NEMORL}/examples/configs:/opt/nemo-rl/examples/configs"

# 3. CONTAINER EXECUTION: cd into container root, source env, and execute
# We run from /opt/nemo-rl (the container root) and source the pre-baked nemo-rl.env.
export COMMAND="\
cd /opt/nemo-rl && \
source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && \
uv run --no-sync <TRAINING_SCRIPT_OR_ENTRYPOINT> \
  <TRAINING_ARGUMENTS_AND_HYDRA_OVERRIDES>"

# 4. Submit Slurm job
sbatch \
    --nodes=<NUM_NODES> \
    --gres=gpu:<GPUS_PER_NODE> \
    --container-image="<CONTAINER_IMAGE_PATH>" \
    --container-mounts="${MOUNTS}" \
    ray.sub
```
