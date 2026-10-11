#!/bin/bash
set -euo pipefail
umask 077

arm=$1
source_commit=$2
job_id=${SLURM_JOB_ID:?}
rank=${SLURM_PROCID:?}
node_count=${SLURM_NNODES:?}
workspace=$PWD
runtime_name="supervl35-runtime-${source_commit:0:12}"
runtime="/raid/scratch/sna/$runtime_name"
state="$workspace/runs/$job_id/$arm/state"
local_root="/raid/scratch/sna/supervl35-$job_id-$arm"
durable_root="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010"
mkdir -p "$state" "$local_root" "$durable_root/runs/$job_id/$arm"
exec >"$local_root/node.log" 2>&1

cleanup() {
    status=$?
    trap - EXIT
    if (( status != 0 )); then printf '%s\n' "$status" >"$state/failed-$rank"; fi
    if (( rank == 0 )); then printf '%s\n' "$status" >"$state/exit-code"; fi
    /opt/nemo_rl_venv/bin/ray stop >/dev/null 2>&1 || true
    tar -czf "$durable_root/runs/$job_id/$arm/node-$rank.tar.gz" \
        --exclude='*.lock' -C "$local_root" node.log results ray 2>/dev/null || true
    mkdir -p "$workspace/runs/$job_id/$arm/logs"
    tail -n 200 "$local_root/node.log" >"$workspace/runs/$job_id/$arm/logs/node-$rank.log"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

test "$node_count" -eq 32
sleep "$rank"
mkdir -p "$local_root/tmp" "$local_root/results" "$local_root/ray"
expected_sha=$(awk '{print $1}' "$durable_root/runtime/$runtime_name.tar.gz.sha256")
if ! test -f "$runtime/.verified-sha256" || [[ $(cat "$runtime/.verified-sha256") != "$expected_sha" ]]; then
    cp "$durable_root/runtime/$runtime_name.tar.gz" "$local_root/runtime.tar.gz"
    printf '%s  %s\n' "$expected_sha" "$local_root/runtime.tar.gz" | sha256sum -c -
    tar -xzf "$local_root/runtime.tar.gz" -C /raid/scratch/sna
    printf '%s\n' "$expected_sha" >"$runtime/.verified-sha256"
fi
test "$(cat "$runtime/SOURCE_COMMIT")" = "$source_commit"
export NRL_EXPERIMENT_SOURCE="$runtime/source"
export TMPDIR="$local_root/tmp" PYTHONPYCACHEPREFIX="$local_root/pycache"
export RAY_TMPDIR="$local_root"
export HF_HOME="$local_root/hf" UV_CACHE_DIR="$local_root/uv"
export XDG_CACHE_HOME="$local_root/cache" TRITON_CACHE_DIR="$local_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" CUDA_CACHE_PATH="$local_root/cuda"
export PYTHONPATH="$NRL_EXPERIMENT_SOURCE:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$NRL_EXPERIMENT_SOURCE/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$NRL_EXPERIMENT_SOURCE/3rdparty/Gym-workspace/Gym:$NRL_EXPERIMENT_SOURCE/examples/nemo_gym/supervl3p5"
export NEMO_GYM_EXTRA_ROOTS="$NRL_EXPERIMENT_SOURCE/3rdparty/Gym-workspace/Gym:$NRL_EXPERIMENT_SOURCE/examples/nemo_gym/supervl3p5"
export MM_TRAINER_MODEL_PATH=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf
cp /lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/datasets/mm-trainer-unified/training.jsonl "$local_root/training.jsonl"
export MM_TRAINER_DATA_PATH="$local_root/training.jsonl"
export MM_TRAINER_RESULTS_DIR="$local_root/results"
export MM_TRAINER_MEDIA_ROOT=/lustre MM_TRAINER_GYM_VENV_DIR="$runtime/gym_venvs"
export MM_TRAINER_WANDB_ID="supervl35-$job_id-$arm" MM_TRAINER_WANDB_NAME="$arm-$job_id"
export NRL_MEGATRON_CHECKPOINT_DIR="$durable_root/checkpoints/$source_commit"
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_FORCE_REBUILD_VENVS=false
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
export VLLM_TRITON_FORCE_FIRST_CONFIG=1 NRL_VIDEO_BACKEND=torchcodec VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export PATH="/opt/nemo_rl_venv/bin:$PATH"

if (( rank == 0 )); then
    /opt/nemo_rl_venv/bin/python -c 'import secrets,sys;open(sys.argv[1],"w").write(secrets.token_hex(32))' "$state/auth"
fi
for (( attempt=0; attempt<120; attempt++ )); do
    test -s "$state/auth" && break
    sleep 5
done
export NRL_TOKEN_CAPTURE_AUTH
NRL_TOKEN_CAPTURE_AUTH=$(cat "$state/auth")

export NRL_GPU_CPU_AFFINITY_FILE="$local_root/gpu_cpu_affinity"
nvidia-smi topo -m | awk '/^GPU[0-9]/ {gpu=$1;sub(/GPU/,"",gpu);numa=$(NF-1);sub(/[,-].*/,"",numa);if(numa~/^[0-9]+$/)print gpu,numa}' | \
    while read -r gpu numa; do printf '%s:%s\n' "$gpu" "$(cat "/sys/devices/system/node/node$numa/cpulist")"; done >"$NRL_GPU_CPU_AFFINITY_FILE"
cluster_uuid=$(nvidia-smi -q | awk -F: '/ClusterUUID/{gsub(/ /,"",$2);print $2;exit}')
resources=$(python -c 'import json,sys;r={"slurm_managed_ray_cluster":1,"worker_units":4,"topo_rank":int(sys.argv[1])+1};r.update({"nvlink_domain_"+sys.argv[2]:1} if sys.argv[2] else {});print(json.dumps(r))' "$rank" "$cluster_uuid")
node_ip=$(hostname -I | awk '{print $1}')
for variable in $(compgen -e); do
    case "$variable" in PMI_*|PMIX_*|MPI_*|OMPI_*|SLURM_*) unset "$variable";; esac
done
cd "$NRL_EXPERIMENT_SOURCE"
common=(--disable-usage-stats --num-gpus=4 --num-cpus=144 --resources="$resources" --node-ip-address="$node_ip" --min-worker-port=2000 --max-worker-port=2999)
if (( rank == 0 )); then
    ray start --head "${common[@]}" --port=1200 --ray-client-server-port=1201 \
        --temp-dir="$local_root/ray" --include-dashboard=False \
        --node-manager-port=1302 --object-manager-port=1304 --runtime-env-agent-port=1306 \
        --dashboard-agent-grpc-port=1308 --metrics-export-port=1310 --dashboard-agent-listen-port=1312
    printf '%s\n' "$node_ip" >"$state/head-ip.tmp"
    mv "$state/head-ip.tmp" "$state/head-ip"
    export RAY_ADDRESS="$node_ip:1200"
    python - <<'PY'
import time
import ray
ray.init(address='auto')
deadline=time.monotonic()+900
while time.monotonic()<deadline:
    nodes=[n for n in ray.nodes() if n['Alive']]
    if len(nodes)==32 and sum(n['Resources'].get('GPU',0) for n in nodes)==128:
        break
    time.sleep(5)
else:
    raise RuntimeError('Expected 32 live Ray nodes and 128 GPUs')
ray.shutdown()
PY
    python examples/run_grpo_single_controller.py \
        --config "experiments/supervl35_option_b_20261010/configs/$arm.yaml"
else
    for (( attempt=0; attempt<180; attempt++ )); do
        test -s "$state/head-ip" && break
        test ! -e "$state/exit-code" || exit 1
        sleep 5
    done
    head_ip=$(cat "$state/head-ip")
    export RAY_ADDRESS="$head_ip:1200"
    ray start "${common[@]}" --address="$RAY_ADDRESS" \
        --node-manager-port=1301 --object-manager-port=1303 --runtime-env-agent-port=1305 \
        --dashboard-agent-grpc-port=1307 --metrics-export-port=1309 --dashboard-agent-listen-port=1311
    while ! test -s "$state/exit-code"; do sleep 5; done
    exit "$(cat "$state/exit-code")"
fi
