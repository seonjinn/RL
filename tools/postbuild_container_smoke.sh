#!/usr/bin/env bash
# Verify the reward deps and MegatronPolicyWorker actor venv are baked correctly.
# Run inside the container, e.g.:
#
#   srun -A coreai_dlalgo_nemorl -p batch -N 1 --time=00:30:00 \
#        --gpus-per-node=1 --no-container-mount-home \
#        --container-image "$CONTAINER" \
#        bash -lc "$(cat tools/probe_mcore_venv.sh)"
#
# docker run --rm -it nemo-rl:${CONTAINER_NAME} \
#   bash -lc "$(cat /localhome/local-aroshanghias/nemo_rl/postbuild_container_smoke.sh)"
#
# Exits non-zero on any missing piece. Useful as a post-build sanity check
# against any new container.

set -euo pipefail

NEMO_RL_PY="${NEMO_RL_PY:-/opt/nemo_rl_venv/bin/python3}"
RAY_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
MCORE_VENV="$RAY_VENV_DIR/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
MCORE_PY="$MCORE_VENV/bin/python"
VLLM_VENV="$RAY_VENV_DIR/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
VLLM_ASYNC_VENV="$RAY_VENV_DIR/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
DTENSOR_VENV="$RAY_VENV_DIR/nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
DTENSOR_V2_VENV="$RAY_VENV_DIR/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
ASYNC_TRAJ_VENV="$RAY_VENV_DIR/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
REPLAY_BUFFER_VENV="$RAY_VENV_DIR/nemo_rl.algorithms.async_utils.ReplayBuffer"
NEMO_GYM_VENV="$RAY_VENV_DIR/nemo_rl.environments.nemo_gym.NemoGym"

step() { echo; echo "=== $* ==="; }
ok()   { echo "  [ OK ] $*"; }
fail() { echo "  [FAIL] $*" >&2; exit 1; }

step "0. baked venvs and wrapper scripts exist"
[[ -x "$NEMO_RL_PY" ]] || fail "no python at $NEMO_RL_PY (base NeMo RL venv missing)"
ok "python at $NEMO_RL_PY"

venvs=(
  "$VLLM_VENV"
  "$VLLM_ASYNC_VENV"
  "$ASYNC_TRAJ_VENV"
  "$REPLAY_BUFFER_VENV"
  "$DTENSOR_VENV"
  "$DTENSOR_V2_VENV"
  "$MCORE_VENV"
  "$NEMO_GYM_VENV"
)
for venv in "${venvs[@]}"; do
  [[ -x "$venv/bin/python" ]] || fail "no python at $venv/bin/python (prefetched Ray venv missing)"
  echo "  ok: $venv/bin/python"
done

wrappers=(
  python-VllmGenerationWorker
  python-VllmAsyncGenerationWorker
  python-AsyncTrajectoryCollector
  python-ReplayBuffer
  python-DTensorPolicyWorker
  python-DTensorPolicyWorkerV2
  python-MegatronPolicyWorker
  python-NemoGym
)
for wrapper in "${wrappers[@]}"; do
  command -v "$wrapper" >/dev/null || fail "wrapper script not found on PATH: $wrapper"
  echo "  ok: $wrapper -> $(command -v "$wrapper")"
done

SP=$("$MCORE_PY" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
) || fail "resolve mcore site-packages"

step "1. reward dependency import smoke (/opt/nemo_rl_venv)"
"$NEMO_RL_PY" - <<'PY' || fail "reward dependency import smoke"
import sys
import mathruler.grader
import pylatexenc
import sympy

print(sys.executable)
print("reward deps ok")
PY
ok "mathruler.grader + undeclared runtime deps import"

step "2. base NeMo RL import smoke"
"$NEMO_RL_PY" - <<'PY' || fail "base NeMo RL import smoke"
import ray
import torch
import transformers
import nemo_rl

print(f"  python      : {__import__('sys').executable}")
print(f"  torch       : {torch.__version__}")
print(f"  transformers: {transformers.__version__}")
print(f"  ray         : {ray.__version__}")
print(f"  nemo_rl     : {getattr(nemo_rl, '__file__', '<namespace package>')}")
PY
ok "base NeMo RL imports"

step "3. flashinfer package and cubin cache"
"$NEMO_RL_PY" - <<'PY' || fail "flashinfer import"
import flashinfer
import flashinfer_cubin
import importlib.metadata
import os

def version(dist_name, module):
    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        return module_version
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "<unknown>"

cubin_dir = os.environ.get(
    "FLASHINFER_CUBIN_DIR",
    "/opt/nemo_rl_venv/lib/python3.12/site-packages/flashinfer_cubin/cubins",
)
print(f"  flashinfer      : {version('flashinfer-python', flashinfer)}")
print(f"  flashinfer_cubin: {version('flashinfer-cubin', flashinfer_cubin)}")
print(f"  cubin dir       : {cubin_dir}")
if not os.path.isdir(cubin_dir):
    raise SystemExit(f"missing cubin directory: {cubin_dir}")
checksums = []
for root, _dirs, files in os.walk(cubin_dir):
    if "checksums.txt" in files:
        checksums.append(os.path.join(root, "checksums.txt"))
print(f"  checksums files : {len(checksums)}")
if len(checksums) < 4:
    raise SystemExit("too few FlashInfer cubin checksum files; cubin prefetch likely failed")
PY
ok "flashinfer import and cubin cache"

step "4. site-packages count (~600 expected for healthy mcore venv, 56 = broken)"
COUNT=$(ls -1 "$SP" 2>/dev/null | wc -l)
echo "  total entries: $COUNT"
[[ "$COUNT" -ge 400 ]] || fail "site-packages has only $COUNT entries; mcore install likely failed"
ok "site-packages count looks healthy ($COUNT >= 400)"

step "5. mcore stack: required Python packages"
required=(
  transformer_engine
  megatron
  mamba_ssm
  causal_conv1d
  nv_grouped_gemm
  flash_attn
)
missing=()
for pkg in "${required[@]}"; do
  if [[ -d "$SP/$pkg" ]] || [[ -f "$SP/${pkg/_/-}.dist-info/METADATA" ]] || ls -d "$SP"/${pkg}*-*.dist-info >/dev/null 2>&1; then
    echo "  ok: $pkg"
  else
    echo "  MISSING: $pkg"
    missing+=("$pkg")
  fi
done
[[ ${#missing[@]} -eq 0 ]] || fail "${#missing[@]} required packages missing: ${missing[*]}"
ok "all required Python packages present"

step "6. transformer-engine torch extension"
TE_TORCH_SO=$(
  find "$SP/transformer_engine" \
    \( -name "libtransformer_engine_torch.so" -o -name "transformer_engine_torch*.so" \) \
    2>/dev/null | head -1
)
[[ -n "$TE_TORCH_SO" ]] || fail "Transformer Engine torch extension not found under $SP/transformer_engine"
echo "  found: $TE_TORCH_SO"
ok "TE torch extension present"

step "7. import smoke (transformer_engine.pytorch + megatron.core + megatron.bridge)"
"$MCORE_PY" - <<'PY' || fail "import smoke"
import transformer_engine
import transformer_engine.pytorch as te_pytorch
import megatron.core
import megatron.bridge
print(f"  TE         : {transformer_engine.__file__}")
print(f"  TE-pytorch : {te_pytorch.__file__}")
print(f"  mcore      : {megatron.core.__file__}")
print(f"  bridge     : {megatron.bridge.__file__}")
PY
ok "transformer_engine.pytorch + megatron.core + megatron.bridge import"

step "8. vLLM worker venv import smoke"
"$VLLM_VENV/bin/python" - <<'PY' || fail "vLLM worker import smoke"
import vllm
import flashinfer
import nemo_rl.models.generation.vllm.vllm_worker
import nemo_rl.models.generation.vllm.vllm_worker_async

print(f"  vllm      : {vllm.__version__}")
print(f"  flashinfer: {flashinfer.__version__}")
PY
ok "vLLM worker imports"

step "9. FSDP/Automodel/NemoGym venv import smoke"
"$DTENSOR_VENV/bin/python" - <<'PY' || fail "DTensorPolicyWorker import smoke"
import nemo_rl.models.policy.workers.dtensor_policy_worker
print("  DTensorPolicyWorker import ok")
PY
"$DTENSOR_V2_VENV/bin/python" - <<'PY' || fail "DTensorPolicyWorkerV2 import smoke"
import nemo_rl.models.policy.workers.dtensor_policy_worker_v2
print("  DTensorPolicyWorkerV2 import ok")
PY
"$NEMO_GYM_VENV/bin/python" - <<'PY' || fail "NemoGym import smoke"
import nemo_rl.environments.nemo_gym
print("  NemoGym import ok")
PY
ok "FSDP/Automodel/NemoGym imports"

echo
echo "============================================================"
echo " Container Python environments look healthy."
echo "============================================================"