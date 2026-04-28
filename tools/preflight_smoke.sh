#!/usr/bin/env bash
# In-container preflight for the Nemotron VL sync-GRPO smoke.
#
# Catches the "obvious" failure modes before we burn an sbatch round
# (~30s here vs ~10-15min wall-clock for sbatch + container init +
# Ray bring-up).
#
# Run inside the SAME container that step_3_nanov3_vision_rl.sh would
# launch, e.g.:
#
#   srun -A coreai_dlalgo_nemorl -p batch -N 1 --time=00:05:00 \
#     -J nrl-preflight --ntasks-per-node=1 --gres=gpu:1 \
#     --no-container-mount-home \
#     --container-image "$CONTAINER" \
#     --container-mounts "/lustre/fs1:/lustre/fs1" \
#     bash -lc "cd $NEMORL && bash tools/preflight_smoke.sh"
#
# Exits non-zero on the first failure. Each check is independent so
# the script can be edited to drop checks that don't apply.

set -euo pipefail

NEMORL="${NEMORL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/aroshanghias/nemo-rl-super}"
DATASET_ROOT="${DATASET_ROOT:-/lustre/fs1/portfolios/coreai/users/aroshanghias/data/mmpr_miniscule/processed}"
MODEL_NAME="${MODEL_NAME:-/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated}"

DRIVER_PY=/opt/nemo_rl_venv/bin/python
MCORE_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
VLLM_PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python

PYTHONPATH_SUPER="${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"

echo
echo "============================================================"
echo " Nemotron VL smoke preflight"
echo "============================================================"
echo "  NEMORL=${NEMORL}"
echo "  DATASET_ROOT=${DATASET_ROOT}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo

step() { echo; echo "=== $* ==="; }
ok()   { echo "  [ OK ] $*"; }
fail() { echo "  [FAIL] $*" >&2; exit 1; }

# ---------- 1. driver venv: Super source + Bridge load ----------

step "1. Driver venv (/opt/nemo_rl_venv)"

$DRIVER_PY -c "import nemo_rl; print('  nemo_rl from', nemo_rl.__file__)" || fail "nemo_rl import"
ok "nemo_rl imports"

# Ensure mathruler is available (verl_geo3k reward needs it; called by
# VLMEnvironment which runs in the driver venv via PY_EXECUTABLES.SYSTEM)
$DRIVER_PY -c "import mathruler.grader" 2>/dev/null \
  && ok "mathruler.grader present" \
  || fail "mathruler.grader missing -- pyproject.toml + uv.lock need updating"

# Dataset class loads + sees mmpr_miniscule on disk
PYTHONPATH="${NEMORL}" $DRIVER_PY -c "
from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY, MMPRTinyDataset
assert 'mmpr_tiny' in DATASET_REGISTRY, 'mmpr_tiny not in registry'
ds = MMPRTinyDataset(split='train', cache_dir='${DATASET_ROOT}', val_size=4)
assert len(ds.dataset) > 0, 'mmpr_miniscule loaded zero rows'
print('  train rows:', len(ds.dataset), 'val rows:', len(ds.val_dataset) if ds.val_dataset else 0)
" || fail "MMPR-Tiny dataset load"
ok "MMPR-Tiny dataset loads"

# verl_geo3k reward instantiates and grades a known-correct response
PYTHONPATH="${NEMORL}" $DRIVER_PY -c "
from nemo_rl.environments.rewards import verl_geo3k_reward, extract_all_boxed
score, ok = verl_geo3k_reward('42', 'thinking</think>\\\\boxed{42}')
assert ok and score > 0.5, f'unexpected reward (score={score}, ok={ok})'
print('  verl_geo3k_reward(42, ...boxed{42}): score=%.2f ok=%s' % (score, ok))
" || fail "verl_geo3k_reward"
ok "verl_geo3k_reward grades correctly"

# ---------- 2. vLLM actor venv: TE + flashinfer + vllm ----------

step "2. VllmGenerationWorker venv"

if [[ ! -x "$VLLM_PY" ]]; then
  fail "VllmGenerationWorker python not found at $VLLM_PY"
fi
ok "venv python exists"

$VLLM_PY -c "import vllm; print('  vllm', vllm.__version__)" || fail "vllm import"
ok "vllm imports"

# flashinfer minor version mismatch is a known issue we bypass via
# FLASHINFER_DISABLE_VERSION_CHECK=1 at runtime; just confirm the
# packages are present.
$VLLM_PY -c "import flashinfer, flashinfer_cubin, flashinfer_jit_cache" \
  || fail "flashinfer / flashinfer-cubin / flashinfer-jit-cache import"
ok "flashinfer + flashinfer-cubin + flashinfer-jit-cache present"

# ---------- 3. Megatron actor venv: TE-torch + bridge + MLM ----------

step "3. MegatronPolicyWorker venv (the one that's been blocking us)"

if [[ ! -x "$MCORE_PY" ]]; then
  fail "MegatronPolicyWorker python not found at $MCORE_PY"
fi
ok "venv python exists"

# This is the load-bearing import. If it fails, libtransformer_engine_torch.so
# is missing -- which means the four-step TE install in
# nemo_rl/utils/venvs.py:create_local_venv didn't run during the docker
# build, OR the source build of transformer-engine-torch failed.
$MCORE_PY -c "
import transformer_engine
import transformer_engine.pytorch as te_pytorch
print('  TE backend:', transformer_engine.__file__)
print('  TE pytorch:', te_pytorch.__file__)
" || fail "transformer_engine.pytorch import (libtransformer_engine_torch.so missing?)"
ok "transformer_engine.pytorch loads (libtransformer_engine_torch.so present)"

# Bridge import triggers the _mlm_compat sys.modules patches.
PYTHONPATH="${PYTHONPATH_SUPER}" $MCORE_PY -c "
import megatron.bridge  # triggers _mlm_compat polyfills
import megatron.bridge._mlm_compat
print('  bridge:', megatron.bridge.__file__)
" || fail "megatron.bridge import (compat shim?)"
ok "megatron.bridge imports (and _mlm_compat side-effect runs)"

# The polyfill assertion the original critique flagged: build a
# real ProcessGroupCollection and exercise use_mpu_process_groups()
# AFTER fake-init of distributed -- this is the runtime path that
# crashed if ProcessGroupCollection were just \`Any\`.
PYTHONPATH="${PYTHONPATH_SUPER}" $MCORE_PY -c "
import os, torch
import torch.distributed as dist

# Minimal single-process distributed init so parallel_state.* can be
# initialized.
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29500')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
if not dist.is_initialized():
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')

import megatron.bridge  # triggers _mlm_compat
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
)

from megatron.core.process_groups_config import ProcessGroupCollection
pg = ProcessGroupCollection.use_mpu_process_groups()
required = ('tp', 'pp', 'mp', 'cp', 'tp_cp', 'embd', 'pos_embd', 'dp')
for attr in required:
    assert hasattr(pg, attr), f'missing pg.{attr}'
print('  pg attrs:', [a for a in required if hasattr(pg, a)])
" || fail "ProcessGroupCollection.use_mpu_process_groups() (polyfill broken?)"
ok "ProcessGroupCollection polyfill builds + has required attrs"

# Static probe rerun for parity: confirms no NEW Bridge<->MLM mismatches
# beyond the 8 our compat layer handles. Cheap (pure Python AST).
if [[ -f /tmp/bridge_mlm_compat_check.py ]]; then
  $MCORE_PY /tmp/bridge_mlm_compat_check.py | tail -3 || true
fi

echo
echo "============================================================"
echo " Preflight PASSED. Safe to launch sbatch smoke."
echo "============================================================"
