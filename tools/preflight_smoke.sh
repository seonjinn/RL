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
# VLMEnvironment which runs in the driver venv via PY_EXECUTABLES.SYSTEM).
# Self-heal: install if missing. This is the same hot-fix the launcher does
# at COMMAND start; needed for containers built from a pyproject snapshot
# that predates the mathruler dep landing in uv.lock (e.g. the 10f917c
# build, where pyproject had mathruler but the lock didn't).
if ! $DRIVER_PY -c "import mathruler.grader" 2>/dev/null; then
  echo "  mathruler.grader missing -- installing (~30s)"
  # NOTE: mathruler-0.1.0 declares zero runtime deps in its wheel metadata
  # even though grader.py imports pylatexenc + sympy. We install all three
  # explicitly so a plain `pip install mathruler` doesn't leave us with a
  # broken import.
  $DRIVER_PY -m pip install --quiet --disable-pip-version-check --no-input \
    mathruler pylatexenc sympy || fail "mathruler install"
fi
$DRIVER_PY -c "import mathruler.grader" \
  && ok "mathruler.grader present" \
  || fail "mathruler.grader still missing after install attempt"

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

# flashinfer minor version mismatch (jit-cache 0.6.5+cu129 vs flashinfer 0.6.9)
# is a known issue we bypass via FLASHINFER_DISABLE_VERSION_CHECK=1 at runtime
# in the launcher. Mirror that here so the preflight matches launcher conditions.
FLASHINFER_DISABLE_VERSION_CHECK=1 $VLLM_PY -c "import flashinfer, flashinfer_cubin, flashinfer_jit_cache" \
  || fail "flashinfer / flashinfer-cubin / flashinfer-jit-cache import"
ok "flashinfer + flashinfer-cubin + flashinfer-jit-cache present (FLASHINFER_DISABLE_VERSION_CHECK=1)"

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

# Bridge import. With the smohsenitahe Bridge pin (f9eadbfa) coordinated
# with the smohsenitahe MLM pin (7611502a), we no longer need the
# _mlm_compat sys.modules polyfill -- ProcessGroupCollection and friends
# resolve natively from megatron.core.
PYTHONPATH="${PYTHONPATH_SUPER}" $MCORE_PY -c "
import megatron.bridge
print('  bridge:', megatron.bridge.__file__)
" || fail "megatron.bridge import"
ok "megatron.bridge imports cleanly (no compat shim required)"

# Import the exact worker modules used by the real Stage 6 / Gate 9 path.
# This catches hard import failures in the baked actor venv without overfitting
# to synthetic ProcessGroupCollection calls that may request process groups the
# real topology never uses.
PYTHONPATH="${PYTHONPATH_SUPER}" $MCORE_PY -c "
import nemo_rl.models.policy.workers.megatron_policy_worker
import nemo_rl.models.generation.vllm.vllm_worker
print('  worker modules import')
" || fail "worker module imports"
ok "worker modules import"

echo
echo "============================================================"
echo " Preflight PASSED. Safe to launch sbatch smoke."
echo "============================================================"
